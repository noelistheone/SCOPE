"""Which modality seeds the set-completion head?

The head initializes its item embeddings from a fixed random projection of CONTENT features (text by
default) and then trains them under the masked-set objective. The item-level analysis finds a task
flip: IMAGE is the stronger *item-level* co-purchase modality (higher learned item-item AUC) yet content
is redundant at *ranking* once strong CF is present. A design-relevant prediction follows: seeding the
head from the stronger item-level modality (image) should NOT yield a stronger *ranking* head -- because
the ranking bottleneck is complementarity with collaborative signal, not raw content quality.

We test it by re-seeding the full deployed head (embeddings trained, only the seed changes) from
{text, image, text+image}, identical protocol, and reporting standalone head and base-fused Recall@20/
NDCG@20 (val-tuned gamma). Writes results/scope/w17_seed_modality_<ds>.json.
Usage: python w17_seed_modality.py [datasets...]   (default baby sports clothing)
"""
from __future__ import annotations
import sys, os, json, math, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import (Rmat, build_lists, closed_form_base, SCOPE, sigreg, zr, evalS_trusted, DEV, ROOT, OUT)
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset


def make_seed(dset, modality, d, gen):
    """normalize(feat) @ (fixed random projection) / sqrt(d). 'both' = mean of the two content seeds."""
    def one(feat):
        X = F.normalize(torch.from_numpy(np.asarray(feat)).float().to(DEV), dim=1)
        Wp = F.normalize(torch.randn(X.shape[1], d, generator=gen, device=DEV), dim=0)
        return (X @ Wp) / math.sqrt(d)
    if modality == "text":  return one(dset.t_feat[:])
    if modality == "image": return one(dset.v_feat[:])
    return 0.5 * one(dset.t_feat[:]) + 0.5 * one(dset.v_feat[:])       # both


def train_seeded(dset, seed_E, ctx, epochs=400, lr=3e-3, wd=1e-6, le=1.0, patience=20, seed=2024):
    """ctx = shared per-dataset objects (R, items/vmask/deg, base, gev) computed ONCE in run()."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    R, items, vmask, deg, degf = ctx["R"], ctx["items"], ctx["vmask"], ctx["deg"], ctx["degf"]
    base, half, gev = ctx["base"], ctx["half"], ctx["gev"]
    m = SCOPE(dset.n_items, seed_E.shape[1]).to(DEV); m.E.data.copy_(seed_E)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    tu = torch.where(deg >= 2)[0]; bs = 8192; best = {"r20": -1, "state": None}; bad = 0
    for ep in range(epochs):
        m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt = m.forward_train(items[b], vmask[b], deg[b])
            logits = m.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, dim=1)
            tgt_lp = (logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)
            loss = -tgt_lp.mean() + le * sigreg(m.E)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            m.eval(); S = m.score_all(R, degf); vr = gev.eval(S)["Recall@20"]
            if vr > best["r20"]:
                best = {"r20": vr, "ep": ep, "state": {k: v.detach().clone() for k, v in m.state_dict().items()}}; bad = 0
            else: bad += 1
            if bad >= patience: break
    m.load_state_dict(best["state"]); m.eval()
    S = m.score_all(R, degf); Sz = zr(S).to(torch.float16 if half else torch.float32)
    bg = (0.0, gev.eval(Sz)["Recall@20"])
    for g in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]:
        v = gev.eval(Sz + g * base)["Recall@20"]
        if v > bg[1]: bg = (g, v)
    pure = evalS_trusted(Sz, dset, "test"); fused = evalS_trusted(Sz + bg[0] * base, dset, "test")
    del m, opt, Sz, S; torch.cuda.empty_cache()
    return {"best_ep": best["ep"], "gamma": bg[0],
            "standalone": {"R20": round(pure["Recall@20"], 4), "N20": round(pure["NDCG@20"], 4)},
            "fused": {"R20": round(fused["Recall@20"], 4), "N20": round(fused["NDCG@20"], 4)}}


def run(ds):
    dset = RecDataset(Config("scope", ds))
    half = dset.n_items > 20000 or dset.n_users > 50000
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    # compute all shared per-dataset objects ONCE, while memory is free (base needs a 3.6GB dense-Gram transient)
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    base = closed_form_base(R, dset, gev, half=half)
    ctx = {"R": R, "items": items, "vmask": vmask, "deg": deg, "degf": degf,
           "base": base, "half": half, "gev": gev}
    torch.cuda.empty_cache()
    gen = torch.Generator(device=DEV).manual_seed(777)          # ONE fixed projection draw, shared across modalities
    res = {"dataset": ds, "d": 256, "modalities": {}}
    for mod in ["text", "image", "both"]:
        if mod != "text" and dset.v_feat is None:
            continue
        seed_E = make_seed(dset, mod, 256, gen)
        r = train_seeded(dset, seed_E, ctx)
        res["modalities"][mod] = r
        print(f"[{ds}] seed={mod:5s} standalone R@20={r['standalone']['R20']:.4f} N@20={r['standalone']['N20']:.4f} "
              f"| fused R@20={r['fused']['R20']:.4f} N@20={r['fused']['N20']:.4f} (gamma={r['gamma']}, ep{r['best_ep']})", flush=True)
        del seed_E; torch.cuda.empty_cache()
    json.dump(res, open(OUT / f"w17_seed_modality_{ds}.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W17_SEED_DONE", flush=True)

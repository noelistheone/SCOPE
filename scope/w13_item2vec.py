"""item2vec/CBOW control (earlier comparison, superseded by neighbors_matched.py): is the set-completion head just a relabel of item2vec/CBOW?

We train a vanilla CBOW item2vec head under the SAME masked-set protocol as SCOPE but stripped of the
three SCOPE-specific pieces: (i) random init instead of content-seeding, (ii) plain mean-pool with NO
residual MLP encoder, (iii) no isotropy regularizer. Comparing this to the full SCOPE head isolates what
those pieces add on top of a CBOW-style order-free predictor. Reports standalone and fused (with the
closed-form base) test Recall@20/NDCG@20 for: CBOW (random), CBOW+content-seed, and (from the shipped
scope_<ds>.json) the full SCOPE head. Writes results/scope/w13_item2vec_<ds>.json. GPU.
Usage: python w13_item2vec.py [datasets...]
"""
from __future__ import annotations
import sys, os, json, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, evalS_trusted, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

GAMMA = {"baby": 0.3, "sports": 0.3, "clothing": 0.6}


class CBOW(nn.Module):
    """item2vec/CBOW head: mean-pool of observed item embeddings, temperature-scaled cosine. No MLP."""
    def __init__(self, n_items, d=256, init=None):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))
        if init is not None:
            self.E.data.copy_(init)
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))

    def latent(self, ctx_sum, n):
        return ctx_sum / n.clamp(min=1).unsqueeze(1)          # plain mean-pool (no residual MLP)

    def logits_from(self, z):
        return (F.normalize(z, 1) @ F.normalize(self.E, 1).t()) / self.logtau.exp().clamp(min=1e-3)

    @torch.no_grad()
    def score_all(self, R, deg):
        return self.logits_from(self.latent(torch.sparse.mm(R, self.E), deg))


def train_cbow(dset, R, items, vmask, deg, gev, init, epochs=300, lr=3e-3, bs=8192, patience=18, seed=2024):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = CBOW(dset.n_items, 256, init).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]; best = {"r": -1}; bad = 0
    for ep in range(epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            it = items[b]; vm = vmask[b]; dg = deg[b]
            keys = torch.where(vm > 0, torch.rand_like(vm), torch.full_like(vm, 1e9))
            ranks = keys.argsort(1).argsort(1).float()
            nctx = (torch.rand(dg.shape, device=DEV) * (dg - 1).clamp(min=1)).floor() + 1
            nctx = torch.minimum(nctx, (dg - 1).clamp(min=1))
            ctx = ((ranks < nctx.unsqueeze(1)) & (vm > 0)).float()
            tgt = ((ranks >= nctx.unsqueeze(1)) & (vm > 0)).float()
            z = model.latent((model.E[it] * ctx.unsqueeze(2)).sum(1), ctx.sum(1))
            logits = model.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1)
            loss = -((logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)).mean()  # NO sigreg
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            model.eval(); vr = gev.eval(model.score_all(R, deg.float()))["Recall@20"]
            if vr > best["r"]:
                best = {"r": vr, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}; bad = 0
            else: bad += 1
            if bad >= patience: break
    model.load_state_dict(best["state"]); model.eval()
    return model


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    gev = GPUEval(dset, "valid", DEV)
    base = closed_form_base(R, dset, gev, half=(dset.n_items > 20000))
    g = GAMMA[ds]
    Xt = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), 1)
    Wp = F.normalize(torch.randn(Xt.shape[1], 256, device=DEV), 0); content_init = (Xt @ Wp) / math.sqrt(256)

    res = {"dataset": ds, "gamma": g, "variants": {}}
    for name, init in [("cbow_random", None), ("cbow_contentseed", content_init)]:
        m = train_cbow(dset, R, items, vmask, deg, gev, init)
        Sz = zr(m.score_all(R, degf)).to(base.dtype)
        alone = evalS_trusted(Sz, dset, "test")
        fused = evalS_trusted(Sz + g * base, dset, "test")
        res["variants"][name] = {"alone_R20": round(alone["Recall@20"], 4), "alone_N20": round(alone["NDCG@20"], 4),
                                 "fused_R20": round(fused["Recall@20"], 4), "fused_N20": round(fused["NDCG@20"], 4)}
        print(f"[{ds}] {name:18s} alone R@20={alone['Recall@20']:.4f}  fused R@20={fused['Recall@20']:.4f}", flush=True)
        del m, Sz; torch.cuda.empty_cache()
    # full SCOPE head numbers from the shipped run (for the comparison row)
    sj = json.load(open(ROOT / "results" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.json"))
    res["variants"]["scope_full"] = {"alone_R20": round(sj["scope_pure"]["Recall@20"], 4), "fused_R20": round(sj["fused"]["Recall@20"], 4)}
    print(f"[{ds}] scope_full         alone R@20={sj['scope_pure']['Recall@20']:.4f}  fused R@20={sj['fused']['Recall@20']:.4f}", flush=True)
    json.dump(res, open(ROOT / "results" / "scope" / f"w13_item2vec_{ds}.json", "w"), indent=2)
    del base, R; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W13_ITEM2VEC_DONE", flush=True)

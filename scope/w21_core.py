"""CORE: CF-conditional functional-gradient boosting of the set-completion head.

A set head trained by an unconditional masked-set softmax largely re-derives the co-purchase directions
that strong CF already has. CORE changes what the head learns: the frozen strong-CF logit s_CF is placed
as an offset inside the training softmax denominator, so
    p(u,i) = softmax_i( beta * s_CF[u,i] + h(u,i) ),   L = -mean_u mean_{i in T} log p(u,i)
The gradient on a target i in T is proportional to (1 - p(u,i)): ~0 where CF already ranks i high, full where
CF misses i, so h is fit as the additive boosting stage on CF's residual. SCOPE-v1 is exactly beta=0.

At inference the head h alone (no offset) is z-scored and fused with the wall. Decisive test: does the
fused {base,GUME,h_boosted} beat the {base,GUME} wall by >= the 80%-power MDE (.0026/.0022/.0017), paired
user-level bootstrap? Also isolate the mechanism: fusion(h_boosted) - fusion(h_{beta=0}). Mechanistic probe:
within-user Pearson(h, s_CF) should DROP as beta rises (capacity reallocated off CF). Writes
results/scope/w21_core_<ds>.json. Usage: python w21_core.py [datasets...]
"""
from __future__ import annotations
import sys, os, json, math, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, SCOPE, sigreg, zr, evalS_trusted, DEV, ROOT
from gpu_eval import GPUEval
from ensemble_control import gate_select
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

MDE = {"baby": 0.00255, "sports": 0.00216, "clothing": 0.00174}   # 80%-power MDE from w16_power


def within_user_pearson(A, B, sample=4000):
    """mean over a sample of users of Pearson corr between rows A[u], B[u]."""
    U = A.shape[0]; idx = torch.randperm(U, device=A.device)[:min(sample, U)]
    a = A[idx].float(); b = B[idx].float()
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    num = (a * b).sum(1); den = (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8)
    return float((num / den).mean())


def train_core(dset, R, items, vmask, deg, degf, zgume, beta, wall_S, gev, half,
               epochs=300, lr=3e-3, wd=1e-6, le=1.0, patience=15, seed=2024):
    """Train the content-seeded head with CF offset beta*zgume inside the masked-set softmax."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    init = None
    if dset.t_feat is not None:
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
        Wp = F.normalize(torch.randn(X.shape[1], 256, device=DEV), dim=0); init = (X @ Wp) / math.sqrt(256)
    m = SCOPE(dset.n_items, 256, init).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    tu = torch.where(deg >= 2)[0]; bs = 8192; best = {"r20": -1, "state": None}; bad = 0
    for ep in range(epochs):
        m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt = m.forward_train(items[b], vmask[b], deg[b])
            logits = m.logits_from(z)                          # [B,N] content head logit h
            if beta != 0.0:
                logits = logits + beta * zgume[b].float()      # CF offset inside the softmax (frozen, no grad)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, dim=1)
            tgt_lp = (logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)
            loss = -tgt_lp.mean() + le * sigreg(m.E)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                h = zr(m.score_all(R, degf)).to(wall_S.dtype)
            # early-stop on val R@20 of the FUSED score (quick gamma sweep of h onto the fixed wall)
            vbest = -1
            for gg in (0.3, 0.6, 1.0, 1.5, 2.0):
                v = gev.recall_per_user(wall_S + gg * h).mean().item()
                if v > vbest: vbest = v
            if vbest > best["r20"]:
                best = {"r20": vbest, "ep": ep, "state": {k: v.detach().clone() for k, v in m.state_dict().items()}}; bad = 0
            else: bad += 1
            del h
            if bad >= patience: break
    m.load_state_dict(best["state"]); m.eval()
    return m, best["ep"]


def _real_run(ds, betas=(0.0, 0.5, 1.0, 2.0, 4.0)):
    dset = RecDataset(Config("scope", ds))
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    zbase = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    zgume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    w0, S0 = gate_select({"item": zbase, "gume": zgume}, ["item", "gume"], gev)
    ru_S0 = gevT.recall_per_user(S0).cpu().numpy()

    res = {"dataset": ds, "mde": MDE[ds], "wall_R20": round(float(ru_S0.mean()), 4), "wall_gate": list(w0), "betas": {}}
    ru_beta0 = None
    for beta in betas:
        m, ep = train_core(dset, R, items, vmask, deg, degf, zgume, beta, S0, gev, half)
        with torch.no_grad():
            h = zr(m.score_all(R, degf)).to(dt)
        pear = within_user_pearson(h, zgume)
        w1, S1 = gate_select({"item": zbase, "gume": zgume, "set": h}, ["item", "gume", "set"], gev)
        ru_S1 = gevT.recall_per_user(S1).cpu().numpy()
        bs_wall = paired_bootstrap(ru_S1, ru_S0)                 # PRIMARY: fused-boosted vs wall
        rec = {"best_ep": ep, "gate": [round(x, 3) for x in w1], "pearson_h_gume": round(pear, 3),
               "fused_R20": round(float(ru_S1.mean()), 4), "vs_wall": bs_wall,
               "set_gate_weight": round(float(w1[2]), 3)}
        if beta == 0.0: ru_beta0 = ru_S1
        else: rec["vs_beta0"] = paired_bootstrap(ru_S1, ru_beta0) if ru_beta0 is not None else None
        res["betas"][f"b{beta}"] = rec
        sig = bs_wall["p_two_sided"] < 0.05 and bs_wall["mean_delta"] > 0
        beats = bs_wall["mean_delta"] >= MDE[ds] and sig
        print(f"[{ds}] beta={beta} fused_R20={rec['fused_R20']} vs_wall d={bs_wall['mean_delta']:+.4f} "
              f"p={bs_wall['p_two_sided']:.2g} pearson(h,gume)={pear:.3f} setgate={rec['set_gate_weight']}"
              f"{'  **>=MDE & sig' if beats else ('  *sig<MDE' if sig else '')}", flush=True)
        del m, h, S1; torch.cuda.empty_cache()
    json.dump(res, open(ROOT / "results" / "scope" / f"w21_core_{ds}.json", "w"), indent=2)
    del zbase, zgume, S0; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            _real_run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W21_CORE_DONE", flush=True)

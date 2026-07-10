"""Kernel complementarity: WHY do the closed-form kernel and the set head fuse so well?

The paper reports an aggregate per-user Spearman of 0.45-0.53 between the base and set rankings. Here we
report the full DISTRIBUTION and show the mechanism: fusion helps exactly the users whose two views
DISAGREE. We compute, per test user, the Spearman rank correlation between the base (EASE+text) and set
scores, and the per-user fusion gain Recall@20(fused) - max(Recall@20(base), Recall@20(set)); we report the
correlation distribution and that the fusion gain is concentrated on low-correlation users.
Writes results/scope/kernel_complementarity_<ds>.json. GPU. Usage: python kernel_complementarity.py <ds...>
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def gate(base, other, gev):
    best = (0.0, -1.0)
    for g in GR:
        r = gev.eval(base + g * other)["Recall@20"]
        if r > best[1]: best = (g, r)
    return base + best[0] * other


def spearman_rows(A, B):
    """per-row Spearman between A[u,:] and B[u,:] (dense rows)."""
    ra = A.argsort(1).argsort(1).float(); rb = B.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True)
    num = (ra * rb).sum(1); den = (ra.norm(dim=1) * rb.norm(dim=1)).clamp(min=1e-9)
    return (num / den)


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    base = zr(closed_form_base(R, dset, gevV, half=half)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad(): sset = zr(m.score_all(R, degf)).to(dt)
    setanchored = zr(sset.float()).to(dt)
    fused = gate(setanchored, base.float(), gevV)                  # SCOPE-v1 (set-anchored + gamma*base)

    # per-user metrics on a sample of test users (Spearman over all items is O(U*I log I))
    U = gevT.users
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randperm(U.numel(), generator=g, device=DEV)[: min(3000, U.numel())]
    su = U[idx]
    rho = []
    for s in range(0, su.numel(), 512):
        bu = su[s:s + 512]
        rho.append(spearman_rows(base[bu].float(), sset[bu].float()).cpu())
    rho = torch.cat(rho).numpy()
    rb = gevT.recall_per_user(base.float(), 20)[idx].cpu().numpy()
    rs = gevT.recall_per_user(sset.float(), 20)[idx].cpu().numpy()
    rf = gevT.recall_per_user(fused.float(), 20)[idx].cpu().numpy()
    gain = rf - np.maximum(rb, rs)                                  # fusion gain over the better single view
    # split users into low- vs high-correlation halves
    med = float(np.median(rho)); lo = rho <= med; hi = rho > med
    out = {
        "dataset": ds, "n_sample": int(len(rho)),
        "spearman_mean": float(rho.mean()), "spearman_median": med,
        "spearman_q": {"p10": float(np.quantile(rho, .1)), "p25": float(np.quantile(rho, .25)),
                        "p50": med, "p75": float(np.quantile(rho, .75)), "p90": float(np.quantile(rho, .9))},
        "frac_corr_below_0.5": float((rho < 0.5).mean()),
        "frac_fused_ge_better_single": float((rf >= np.maximum(rb, rs) - 1e-9).mean()),
        "mean_fusion_gain_low_corr": float(gain[lo].mean()), "mean_fusion_gain_high_corr": float(gain[hi].mean()),
        "corr_spearman_vs_gain": float(np.corrcoef(rho, gain)[0, 1]),
    }
    json.dump(out, open(ROOT / "results" / "scope" / f"kernel_complementarity_{ds}.json", "w"), indent=2)
    print(f"[{ds}] Spearman mean={out['spearman_mean']:.3f} med={med:.3f} frac<0.5={out['frac_corr_below_0.5']:.2f} "
          f"| fused>=better {out['frac_fused_ge_better_single']:.3f} | gain low-corr={out['mean_fusion_gain_low_corr']:+.4f} "
          f"high-corr={out['mean_fusion_gain_high_corr']:+.4f} | corr(rho,gain)={out['corr_spearman_vs_gain']:+.2f}", flush=True)
    del R, base, sset, fused; torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        run(ds)
    print("KERNEL_COMPLEMENTARITY_DONE", flush=True)

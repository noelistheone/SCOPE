#!/usr/bin/env python
"""Sensitivity of the score fusion to the weight gamma (Figure S2).

SCOPE-v1 fuses the set-completion scores with the training-free closed-form base as
  S = z(S_set) + gamma * z(S_base),
where z(.) is a per-user z-score and gamma is selected on validation. This script sweeps
gamma over a fine grid (0.0 ... 3.0, step 0.1 below 1.0) and records validation Recall@20
of the fused scorer on baby / sports / clothing, together with the two single-component
levels (set-only, i.e. gamma=0, and base-only). The base is rebuilt with the same
validation-tuned hyper-parameters as in training; the set-completion scores come from the
shipped checkpoint ckpts/scope/scope_<ds>_d256_le1.0_lz1.0_lr0.003.pt. The test Recall@20
at the selected gamma (validation argmax) is also reported, via the trusted full-sort
evaluator. Tune on validation, test once.

Writes results/scope/gamma_sweep_<ds>.json and the three-panel figure
results/scope/fig_gamma.pdf (one panel per dataset; the interior optimum is starred).

Usage: python gamma_sweep.py [dataset ...]      (default: baby sports clothing)
"""
from __future__ import annotations
import sys, os, json, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scope import Rmat, build_lists, closed_form_base, zr, SCOPE, evalS_trusted, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

GAMMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
OUT = ROOT / "results" / "scope"; OUT.mkdir(parents=True, exist_ok=True)
CK = ROOT / "ckpts" / "scope"

# figure ink (validated for a light/print surface)
C_FUSED, C_SET, C_BASE = "#2a78d6", "#008300", "#eb6834"
C_GRID, C_AXIS, C_INK, C_MUT = "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"


def run(ds):
    torch.manual_seed(2024); np.random.seed(2024); random.seed(2024)   # same seeding as training
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); items, vmask, deg = build_lists(dset)
    half = dset.n_items > 20000 or dset.n_users > 50000                # fp16 score matrices for large datasets
    gev = GPUEval(dset, "valid", DEV)
    base = closed_form_base(R, dset, gev, half=half)                   # z-scored base, val-tuned (lam, a) as in training
    model = SCOPE(dset.n_items, 256).to(DEV)
    model.load_state_dict(torch.load(CK / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV))
    model.eval()
    with torch.no_grad():
        Sz = zr(model.score_all(R, deg.float())).to(base.dtype)        # z(S_set)
    curve = []
    for g in GAMMAS:
        v = gev.eval(Sz if g == 0.0 else Sz + g * base)["Recall@20"]
        curve.append(float(v)); print(f"[{ds}] gamma={g:.1f} val_R20={v:.4f}", flush=True)
    base_only = float(gev.eval(base)["Recall@20"])
    i_star = int(np.argmax(curve)); g_star = GAMMAS[i_star]
    test_star = evalS_trusted(Sz if g_star == 0.0 else Sz + g_star * base, dset, "test")
    rep = {"dataset": ds, "gammas": GAMMAS, "val_recall20": curve,
           "set_only_val": curve[0], "base_only_val": base_only,
           "gamma_star": g_star, "val_at_star": curve[i_star],
           "test_at_star": {k: float(v) for k, v in test_star.items()}}
    (OUT / f"gamma_sweep_{ds}.json").write_text(json.dumps(rep, indent=2))
    print(f"[{ds}] gamma*={g_star} val_R20={curve[i_star]:.4f} | set-only={curve[0]:.4f} "
          f"base-only={base_only:.4f} | test_R20@gamma*={test_star['Recall@20']:.4f}", flush=True)
    del R, base, Sz, model; torch.cuda.empty_cache()
    return rep


def make_figure(datasets):
    reps = []
    for ds in datasets:
        p = OUT / f"gamma_sweep_{ds}.json"
        if p.exists(): reps.append(json.loads(p.read_text()))
    if not reps: return
    fig, axes = plt.subplots(1, len(reps), figsize=(3.5 * len(reps), 3.0))
    if len(reps) == 1: axes = [axes]
    for j, (ax, rep) in enumerate(zip(axes, reps)):
        g, v = rep["gammas"], rep["val_recall20"]
        ax.axhline(rep["set_only_val"], color=C_SET, ls="--", lw=1.6, label="set-only ($\\gamma=0$)")
        ax.axhline(rep["base_only_val"], color=C_BASE, ls=":", lw=1.8, label="base-only")
        ax.plot(g, v, color=C_FUSED, lw=2.0, marker="o", ms=3.5,
                label="fused $z(S_{set})+\\gamma\\, z(S_{base})$")
        ax.plot([rep["gamma_star"]], [rep["val_at_star"]], marker="*", ms=15, color=C_FUSED,
                mec="white", mew=0.8, ls="none", zorder=5,
                label="$\\gamma^{*}$ (val argmax)" if j == 0 else None)
        ax.set_title(f"{rep['dataset']}  ($\\gamma^{{*}}={rep['gamma_star']:g}$)", fontsize=11, color=C_INK)
        ax.set_xlabel("fusion weight $\\gamma$", fontsize=10, color=C_INK)
        if j == 0: ax.set_ylabel("validation Recall@20", fontsize=10, color=C_INK)
        ax.grid(True, color=C_GRID, lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(C_AXIS)
        ax.tick_params(colors=C_MUT, labelsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gamma.pdf", bbox_inches="tight")
    print(f"figure -> {OUT / 'fig_gamma.pdf'}", flush=True)


if __name__ == "__main__":
    dss = sys.argv[1:] or ["baby", "sports", "clothing"]
    for ds in dss:
        run(ds)
    make_figure(dss)
    print("GAMMA_SWEEP_DONE", flush=True)

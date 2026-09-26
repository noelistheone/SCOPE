#!/usr/bin/env python
"""Fusion normalisation: the SCOPE-v1 complementarity result (fused base+set beats the better single view)
was verified only under per-user z-score standardization. Here we re-test it under TWO other
standardizations — per-user percentile RANK fusion and per-user MIN-MAX fusion — to show the
complementarity is not an artifact of the z-score scheme.

Single-model base+set fusion only. For each scheme: normalize both
views per user, val-tune the base weight gamma, test once with the trusted evaluator. base-only /
set-only test scores are scheme-invariant (a single view's ranking is invariant to any per-user
monotone normalization), so only the FUSED row changes across schemes.
Results -> results/scope/fusion_norm_ablation.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, closed_form_base, SCOPE, evalS_trusted, BAR, DEV, OUT


def _chunked(S, fn, out_dtype=torch.float16, rows=4096):
    """Apply a per-row normalizer fn in row-chunks (keeps argsort temporaries small); fp16 output."""
    out = torch.empty_like(S, dtype=out_dtype)
    for s in range(0, S.shape[0], rows):
        out[s:s + rows] = fn(S[s:s + rows].float()).to(out_dtype)
    return out

def znorm(S):
    return _chunked(S, lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-9))

def ranknorm(S):
    # per-user percentile rank in [0,1] (higher score -> higher value)
    N = S.shape[1]
    return _chunked(S, lambda x: x.argsort(1).argsort(1).float() / (N - 1))

def minmaxnorm(S):
    def f(x):
        lo = x.min(1, keepdim=True).values; hi = x.max(1, keepdim=True).values
        return (x - lo) / (hi - lo + 1e-9)
    return _chunked(S, f)

SCHEMES = {"zscore": znorm, "rank": ranknorm, "minmax": minmaxnorm}
GAMMA = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def run(dataset, seed=2024):
    dset = RecDataset(Config("scope", dataset))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    bar_r, bar_n = BAR[dataset]
    half = dset.n_items > 20000
    gev = GPUEval(dset, "valid", DEV)
    S_base = closed_form_base(R, dset, gev, half=half).float()
    stag = f"scope_{dataset}_d256_le1.0_lz1.0_lr0.003" + ('' if seed == 2024 else f'_s{seed}')
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"{stag}.pt", map_location=DEV)); m.eval()
    with torch.no_grad(): S_set = m.score_all(R, degf).float()

    base_t = evalS_trusted(S_base, dset, "test")
    set_t = evalS_trusted(S_set, dset, "test")
    better = max(base_t["Recall@20"], set_t["Recall@20"])
    out = {"dataset": dataset, "seed": seed, "bar": {"R@20": bar_r, "N@20": bar_n},
           "base": {"R@20": base_t["Recall@20"], "N@20": base_t["NDCG@20"]},
           "set": {"R@20": set_t["Recall@20"], "N@20": set_t["NDCG@20"]},
           "schemes": {}}
    print(f"\n=== {dataset} (seed {seed}) base R@20={base_t['Recall@20']:.4f} set R@20={set_t['Recall@20']:.4f} ===", flush=True)
    for name, fn in SCHEMES.items():
        nb = fn(S_base); ns = fn(S_set)
        # fused = norm(set) + gamma*norm(base); gamma tuned on val Recall@20
        best = None
        for g in GAMMA:
            vr = gev.eval(ns + g * nb)["Recall@20"]
            if best is None or vr > best[0]: best = (vr, g)
        g = best[1]
        ft = evalS_trusted(ns + g * nb, dset, "test")
        comp = ft["Recall@20"] - better
        out["schemes"][name] = {"gamma": g, "R@20": ft["Recall@20"], "N@20": ft["NDCG@20"],
                                "complementarity_vs_better_single": comp, "beats_better_single": bool(comp > 0)}
        print(f"  {name:8s} gamma={g:<4} fused R@20={ft['Recall@20']:.4f} N@20={ft['NDCG@20']:.4f} "
              f"| complementarity(fused-betterSingle)={comp:+.4f} {'OK' if comp > 0 else 'NO'}", flush=True)
    p = OUT / (f"fusion_norm_ablation_{dataset}" + ("" if seed == 2024 else f"_s{seed}") + ".json")
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> {p}", flush=True)
    del R, S_base, S_set; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024])
    a = ap.parse_args()
    allout = {}
    for ds in a.datasets:
        for sd in a.seeds:
            try: allout[f"{ds}_s{sd}"] = run(ds, seed=sd)
            except Exception as e:
                import traceback; print(f"[{ds} s{sd}] ERR {type(e).__name__}: {e}"); traceback.print_exc()

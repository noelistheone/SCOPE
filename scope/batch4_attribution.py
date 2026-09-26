#!/usr/bin/env python
"""Batch 4 (GPU) — within-model attribution:
  B1  add-one-in ladder (SCOPE-v1): EASE-1hop -> +text-kNN -> +2-hop rollout -> (=full base) -> +set head (fused)
      => marginal R@20/N@20 of each component (lam fixed=800 for the ladder; c2,a forced toggles).
  B2  PLACEBO-view control: marginal of adding the SET view to {col,item} vs adding a RANDOM placebo view,
      under identical val gate re-selection. Proves V_set's marginal exceeds a noise view's (not gate-overfitting).
Writes results/scope/significance/batch4_{ds}.json.
"""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_views, random_tower, test_metrics, OUT, GRID, DEV
from scope_full import closed_form_base

import os
if os.environ.get("SCOPE_HEAD") != "full":
    print("note: SCOPE_HEAD is not 'full'; the views come from scope.py. The reported numbers of this analysis used the "
          "pre-pruning model: train it with scope_full.py and run with SCOPE_HEAD=full.", flush=True)


def best_combo(gevV, views, allow):
    """val-select nonneg gate over the listed views (each in GRID if allowed else {0}); returns combo + val R@20."""
    grids = [GRID if al else [0.0] for al in allow]
    best, br = None, -1.0
    def rec(i, acc):
        nonlocal best, br
        if i == len(views):
            if any(acc):
                S = sum(w * v for w, v in zip(acc, views) if w)
                r = gevV.recall_per_user(S, 20).mean().item()
                if r > br: br = r; best = tuple(acc)
            return
        for w in grids[i]: rec(i + 1, acc + [w])
    rec(0, [])
    return best, br


def run(ds, which='all'):
    V = load_views(ds); bar_r, bar_n = V['bar']
    fp = OUT / f"batch4_{ds}.json"
    rep = json.loads(fp.read_text()) if fp.exists() else {"dataset": ds, "bar": {"R20": bar_r, "N20": bar_n}}
    if which == 'b1':
        del V['S_col']; torch.cuda.empty_cache()           # free col (B1 doesn't use it) for big datasets

    if which in ('all', 'b1'):
        _b1(V, rep)
    if which in ('all', 'b2'):
        _b2(V, ds, rep)
    fp.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n========== {ds.upper()} (which={which}) ==========")
    if 'B1_ladder' in rep:
        print("[B1] add-one-in ladder:")
        for r in rep['B1_ladder']:
            print(f"     {r['rung']:28s} R@20={r['R20']:.4f} (margR={r['marg_R']}) N@20={r['N20']:.4f} (margN={r['marg_N']})")
    if 'B2_placebo' in rep:
        b2 = rep['B2_placebo']
        print(f"[B2] noSET(col+item) R@20={b2['noSET_col_item']['R20']:.4f}  "
              f"+SET marginal={b2['add_SET']['marginal_R']:+.4f}  +PLACEBO marginal={b2['add_PLACEBO']['marginal_R']:+.4f} "
              f"(placebo weight selected={b2['add_PLACEBO']['weight']})")
    del V; torch.cuda.empty_cache()
    return rep


def _b1(V, rep):
    bar_r, bar_n = V['bar']
    # ---------- B1 add-one-in ladder ----------
    R, dset = V['R'], V['dset']
    rungs = [("EASE_1hop", dict(c2=0.0, a=0.0)), ("+text-kNN", dict(c2=0.0, a=0.5)),
             ("+2hop (=full base)", dict(c2=0.2, a=0.5))]   # properly NESTED add-one-in
    ladder = []
    prevR = prevN = None
    for nm, kw in rungs:
        S = closed_form_base(R, dset, None, lam=800, half=V['half'], **kw)
        t = test_metrics(V, S); del S; torch.cuda.empty_cache()
        mR = None if prevR is None else round(t['Recall@20'] - prevR, 4)
        mN = None if prevN is None else round(t['NDCG@20'] - prevN, 4)
        ladder.append(dict(rung=nm, R20=t['Recall@20'], N20=t['NDCG@20'], marg_R=mR, marg_N=mN))
        prevR, prevN = t['Recall@20'], t['NDCG@20']
    # + set head fused (val-tuned gamma) on top of full base (use the resident tuned base for the final rung)
    bg = (0.0, V['gevV'].recall_per_user(V['S_item'], 20).mean().item())
    for g in GRID[1:]:
        r = V['gevV'].recall_per_user(V['S_item'] + g * V['S_set'], 20).mean().item()
        if r > bg[1]: bg = (g, r)
    tf = test_metrics(V, V['S_item'] + bg[0] * V['S_set'])
    ladder.append(dict(rung="+set head (SCOPE-v1 fused)", R20=tf['Recall@20'], N20=tf['NDCG@20'],
                       marg_R=round(tf['Recall@20'] - ladder[-1]['R20'], 4),
                       marg_N=round(tf['NDCG@20'] - ladder[-1]['N20'], 4), gamma=bg[0]))
    rep['B1_ladder'] = ladder


def _b2(V, ds, rep):
    # ---------- B2 placebo-view control (frugal: hold val-selected col+item gate, 1-D sweep added column) ----------
    abl = json.loads(((Path(__file__).resolve().parents[1] / "results" / "scope") / f"scope_u_ablate_ease_{ds}.json").read_text())
    na, nb, _ = [float(x) for x in abl['rows']['noSET = CF-ens (col+item)']['combo']]
    S_ci = na * V['S_col'] + nb * V['S_item']                     # fixed col+item base
    del V['S_col'], V['S_item']; torch.cuda.empty_cache()         # only S_ci + added column needed now
    noset_t = test_metrics(V, S_ci)

    def add_one(col_view):
        bg = (0.0, V['gevV'].recall_per_user(S_ci, 20).mean().item())
        for w in GRID[1:]:
            r = V['gevV'].recall_per_user(S_ci + w * col_view, 20).mean().item()
            if r > bg[1]: bg = (w, r)
        t = test_metrics(V, S_ci + bg[0] * col_view) if bg[0] > 0 else noset_t
        return bg[0], t

    w_set, set_t = add_one(V['S_set'])
    V_rand = random_tower(V['n_items'], V['R'], V['degf'], 256, seed=0, dt=V['dt'])
    w_pla, pla_t = add_one(V_rand); del V_rand, S_ci; torch.cuda.empty_cache()
    rep['B2_placebo'] = dict(
        protocol="hold val-selected col+item gate; val-select the added column weight in GRID",
        noSET_col_item=dict(gate=(na, nb), R20=noset_t['Recall@20'], N20=noset_t['NDCG@20']),
        add_SET=dict(weight=w_set, R20=set_t['Recall@20'], N20=set_t['NDCG@20'],
                     marginal_R=round(set_t['Recall@20'] - noset_t['Recall@20'], 4),
                     marginal_N=round(set_t['NDCG@20'] - noset_t['NDCG@20'], 4)),
        add_PLACEBO=dict(weight=w_pla, R20=pla_t['Recall@20'], N20=pla_t['NDCG@20'],
                         marginal_R=round(pla_t['Recall@20'] - noset_t['Recall@20'], 4)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--only", default="all", choices=["all", "b1", "b2"])
    a = ap.parse_args()
    for ds in a.datasets: run(ds, which=a.only)
    print("\nBATCH4 DONE ->", OUT)

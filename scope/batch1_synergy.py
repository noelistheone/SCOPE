#!/usr/bin/env python
"""Batch 1 — attribution controls, per dataset, one view-load each:
  C2  null-tower control : EASE+set vs EASE+random-tower vs EASE+cooc-kNN  -> is the lift SPECIFIC to the learned set view?
  C4  gamma-sweep        : fused(g)=S_set+g*S_item -> interior optimum + fused>best-single (super-additivity)
  C3  decorrelation      : per-user rho(item,set), win/tie/harm, %fused>max, error-corr / variance reduction
  C7  paired bootstrap   : fused_v1 vs {EASE, FREEDOM, set} (Holm-corrected) + dump per-user arrays for multi-seed
Writes results/scope/significance/batch1_{ds}.json + per-user arrays batch1_{ds}_peruser.npz.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (load_views, random_tower, cooc_knn_view, best_gamma,
                     per_user, test_metrics, paired_bootstrap, OUT, GRID)
from scipy.stats import spearmanr, pearsonr

import os
if os.environ.get("SCOPE_HEAD") != "full":
    print("note: SCOPE_HEAD is not 'full'; the views come from scope.py. The reported numbers of this analysis used the "
          "pre-pruning model: train it with scope_full.py and run with SCOPE_HEAD=full.", flush=True)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(run, 1.0)
    return adj.tolist()


def run(ds):
    V = load_views(ds); bar_r, bar_n = V['bar']
    rep = {"dataset": ds, "bar": {"R20": bar_r, "N20": bar_n}}

    # ---------- C4 gamma sweep: fused(g) = S_set + g*S_item ----------
    gammas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    curve = []
    for g in gammas:
        S = V['S_set'] + g * V['S_item']
        vr = V['gevV'].recall_per_user(S, 20).mean().item()
        tm = V['gevT'].eval(S)
        curve.append(dict(gamma=g, val_R20=vr, test_R20=tm['Recall@20'], test_N20=tm['NDCG@20']))
        del S; torch.cuda.empty_cache()
    g_star = max(curve, key=lambda r: r['val_R20'])['gamma']           # val-selected
    base_alone = test_metrics(V, V['S_item']); head_alone = test_metrics(V, V['S_set'])
    fused_t = test_metrics(V, V['S_set'] + g_star * V['S_item'])
    interior = 0.0 < g_star < gammas[-1]
    best_single = max(base_alone['Recall@20'], head_alone['Recall@20'])
    rep['C4_gamma_sweep'] = dict(curve=curve, g_star=g_star, interior_optimum=bool(interior),
                                 base_alone=base_alone, head_alone=head_alone, fused=fused_t,
                                 fused_minus_best_single_R20=fused_t['Recall@20'] - best_single)

    # ---------- small per-user vectors first (free big fused matrices immediately) ----------
    r_item = per_user(V, V['S_item']); r_set = per_user(V, V['S_set']); r_col = per_user(V, V['S_col'])
    r_fused = per_user(V, V['S_set'] + g_star * V['S_item'])
    ease_alone_R = base_alone['Recall@20']

    # ---------- C2 null-tower control (build/eval/free one tower at a time) ----------
    null = {}
    # set view (already resident)
    g_set, _ = best_gamma(V['gevV'], V['S_item'], V['S_set'])
    tset = test_metrics(V, V['S_item'] + g_set * V['S_set'])
    null['set'] = dict(g_star=g_set, fused_R20=tset['Recall@20'], fused_N20=tset['NDCG@20'],
                       lift_R20=tset['Recall@20'] - ease_alone_R)
    r_easeset = per_user(V, V['S_item'] + g_set * V['S_set'])
    # random tower
    V_rand = random_tower(V['n_items'], V['R'], V['degf'], 256, 0, V['dt'])
    g_rnd, _ = best_gamma(V['gevV'], V['S_item'], V_rand)
    trnd = test_metrics(V, V['S_item'] + g_rnd * V_rand)
    null['random'] = dict(g_star=g_rnd, fused_R20=trnd['Recall@20'], fused_N20=trnd['NDCG@20'],
                          lift_R20=trnd['Recall@20'] - ease_alone_R,
                          alone_R20=test_metrics(V, V_rand)['Recall@20'])
    r_easernd = per_user(V, V['S_item'] + g_rnd * V_rand)
    del V_rand; torch.cuda.empty_cache()
    # cooc-knn tower
    V_cooc = cooc_knn_view(V['R'], V['n_items'], 20, V['dt'])
    g_coo, _ = best_gamma(V['gevV'], V['S_item'], V_cooc)
    tcoo = test_metrics(V, V['S_item'] + g_coo * V_cooc)
    null['cooc_knn'] = dict(g_star=g_coo, fused_R20=tcoo['Recall@20'], fused_N20=tcoo['NDCG@20'],
                            lift_R20=tcoo['Recall@20'] - ease_alone_R,
                            alone_R20=test_metrics(V, V_cooc)['Recall@20'])
    del V_cooc; torch.cuda.empty_cache()
    rep['C2_null_tower'] = dict(
        ease_alone_R20=ease_alone_R, towers=null,
        dLift_set_vs_random=null['set']['lift_R20'] - null['random']['lift_R20'],
        dLift_set_vs_cooc=null['set']['lift_R20'] - null['cooc_knn']['lift_R20'])

    np.savez(OUT / f"batch1_{ds}_peruser.npz", r_item=r_item, r_set=r_set, r_col=r_col,
             r_fused=r_fused, r_easeset=r_easeset, r_easernd=r_easernd)

    # ---------- C3 decorrelation ----------
    def winharm(a, b):  # a vs b
        return dict(a_wins=float((a > b).mean()), tie=float((a == b).mean()), b_wins=float((a < b).mean()))
    e_item, e_set = 1 - r_item, 1 - r_set
    rep['C3_decorr'] = dict(
        spearman_item_set=float(spearmanr(r_item, r_set).statistic),
        pearson_item_set=float(pearsonr(r_item, r_set).statistic),
        spearman_col_set=float(spearmanr(r_col, r_set).statistic),
        err_corr_item_set=float(pearsonr(e_item, e_set).statistic),
        set_vs_item=winharm(r_set, r_item), set_vs_col=winharm(r_set, r_col),
        pct_fused_ge_max_item_set=float((r_fused >= np.maximum(r_item, r_set) - 1e-9).mean()),
        pct_fused_gt_max_item_set=float((r_fused > np.maximum(r_item, r_set) + 1e-9).mean()),
        var_item=float(r_item.var()), var_set=float(r_set.var()), var_fused=float(r_fused.var()),
        var_reduction_pred_eq=float((1 + pearsonr(e_item, e_set).statistic) / 2))

    # ---------- C7 paired bootstrap (Holm over the 3 comparisons) ----------
    comps = {"fused_vs_EASE": (r_fused, r_item), "fused_vs_FREEDOM": (r_fused, r_col),
             "fused_vs_set": (r_fused, r_set), "EASEset_vs_EASErandom": (r_easeset, r_easernd)}
    bs = {k: paired_bootstrap(a, b) for k, (a, b) in comps.items()}
    padj = holm([bs[k]['p_two_sided'] for k in comps])
    for k, pa in zip(comps, padj): bs[k]['p_holm'] = pa
    rep['C7_bootstrap'] = bs

    (OUT / f"batch1_{ds}.json").write_text(json.dumps(rep, indent=2, default=str))
    # console
    print(f"\n========== {ds.upper()} (bar R@20={bar_r:.4f}) ==========")
    print(f"[C4] g*={g_star} interior={interior}  base={base_alone['Recall@20']:.4f} head={head_alone['Recall@20']:.4f} "
          f"fused={fused_t['Recall@20']:.4f}  fused-best_single={fused_t['Recall@20']-best_single:+.4f}")
    print(f"[C2] EASE_alone={ease_alone_R:.4f} | EASE+set={null['set']['fused_R20']:.4f}(lift {null['set']['lift_R20']:+.4f}) "
          f"EASE+rand={null['random']['fused_R20']:.4f}(lift {null['random']['lift_R20']:+.4f}) "
          f"EASE+cooc={null['cooc_knn']['fused_R20']:.4f}(lift {null['cooc_knn']['lift_R20']:+.4f})")
    print(f"     dLift set-vs-random={rep['C2_null_tower']['dLift_set_vs_random']:+.4f}  set-vs-cooc={rep['C2_null_tower']['dLift_set_vs_cooc']:+.4f}")
    print(f"[C3] spearman(item,set)={rep['C3_decorr']['spearman_item_set']:.3f}  set_vs_item win/tie/loss="
          f"{rep['C3_decorr']['set_vs_item']['a_wins']:.2f}/{rep['C3_decorr']['set_vs_item']['tie']:.2f}/{rep['C3_decorr']['set_vs_item']['b_wins']:.2f}"
          f"  fused>=max={rep['C3_decorr']['pct_fused_ge_max_item_set']:.2f}")
    for k, b in bs.items():
        print(f"[C7] {k:24s} dR@20={b['mean_delta']:+.4f} CI[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] p_holm={b['p_holm']:.4g}")
    del V; torch.cuda.empty_cache()
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    a = ap.parse_args()
    for ds in a.datasets:
        run(ds)
    print("\nBATCH1 DONE ->", OUT)

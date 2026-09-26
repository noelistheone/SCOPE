#!/usr/bin/env python
"""Robust 2-hop pruning check on the DEPLOYED model (no retraining, no trainer offset):
load the canonical SCOPE head ckpt, fuse with the val-tuned base (c2 free) vs a c2=0 base (1-hop+text only).
If fused(c2=0) >= fused(tuned), the 2-hop rollout (c2*B^2) is removable from the shipped model. Trusted test.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_views, best_gamma, test_metrics, OUT
from train_prune import prune_base
from scope_full import closed_form_base

import os
if os.environ.get("SCOPE_HEAD") != "full":
    print("note: SCOPE_HEAD is not 'full'; the views come from scope.py. The reported numbers of this analysis used the "
          "pre-pruning model: train it with scope_full.py and run with SCOPE_HEAD=full.", flush=True)


def run(ds):
    V = load_views(ds)                                  # loads canonical scope ckpt -> S_set (z-scored)
    Sset = V['S_set']
    base_tuned = V['S_item']                            # val-tuned EASE-2hop+text
    base_1hop = prune_base(V['R'], V['dset'], V['gevV'], V['half'], c2=0.0)
    out = {}
    for nm, base in [("tuned_base_c2_free", base_tuned), ("base_c2_0_1hop+text", base_1hop)]:
        g, _ = best_gamma(V['gevV'], Sset, base)
        t = test_metrics(V, Sset + g * base)
        out[nm] = dict(gamma=g, R20=t['Recall@20'], N20=t['NDCG@20'])
    out["2hop_marginal_R"] = round(out["tuned_base_c2_free"]["R20"] - out["base_c2_0_1hop+text"]["R20"], 4)
    out["2hop_marginal_N"] = round(out["tuned_base_c2_free"]["N20"] - out["base_c2_0_1hop+text"]["N20"], 4)
    out["bar"] = V['bar']
    (OUT / f"prune2hop_{ds}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[{ds}] deployed head ⊕ tuned-base = {out['tuned_base_c2_free']['R20']:.4f}/{out['tuned_base_c2_free']['N20']:.4f}  "
          f"| ⊕ c2=0 base = {out['base_c2_0_1hop+text']['R20']:.4f}/{out['base_c2_0_1hop+text']['N20']:.4f}  "
          f"| 2-hop marginal = {out['2hop_marginal_R']:+.4f}R/{out['2hop_marginal_N']:+.4f}N  (bar {V['bar'][0]}/{V['bar'][1]})", flush=True)
    del V, base_1hop; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    a = ap.parse_args()
    for ds in a.datasets: run(ds)
    print("\nPRUNE2HOP DONE ->", OUT)

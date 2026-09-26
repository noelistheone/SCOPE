#!/usr/bin/env python
"""Leak-safety as an actual test (not an assertion).

Train/val/test ALIGNMENT signature on SCOPE's components. For a clean model the score on a user's
TRAIN-fit items is highest, and VALID == TEST (both unseen, no test-specific information). A leaked
feature set shows TEST >= VALID / TEST >~ TRAIN.

We measure, per view (EASE base S_item ; text-affinity-only base), the mean z-scored score on each
user's train / valid / test positive items (sampled users). Report train/valid/test means and the
(test - valid) gap (the leak signature). Near-zero gap + train highest = clean.
Writes results/scope/significance/leak_test_{ds}.json.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_views, OUT, DEV
from scope_full import closed_form_base

import os
if os.environ.get("SCOPE_HEAD") != "full":
    print("note: SCOPE_HEAD is not 'full'; the views come from scope.py. The reported numbers of this analysis used the "
          "pre-pruning model: train it with scope_full.py and run with SCOPE_HEAD=full.", flush=True)


def split_means(S, users, pos):
    """mean of S[u, item] over each user's positive items in `pos` [U,P] (-1 pad); aligned with `users`."""
    vals = []
    for s in range(0, users.numel(), 4096):
        bu = users[s:s + 4096]; bp = pos[s:s + 4096]
        sc = S[bu].float(); mask = bp >= 0
        gathered = torch.gather(sc, 1, bp.clamp(min=0))
        gathered = torch.where(mask, gathered, torch.full_like(gathered, float('nan')))
        vals.append(gathered.reshape(-1))
    v = torch.cat(vals); v = v[~torch.isnan(v)]
    return float(v.mean().item()), int(v.numel())


def run(ds):
    V = load_views(ds)
    gevV, gevT = V['gevV'], V['gevT']
    # train positives per user: GPUEval(test).hist holds train interactions (used for masking)
    # build a [U,Pmax] train-pos tensor aligned with gevT.users
    usersT = gevT.users
    # map train history (hist_u, hist_i) into per-user lists for usersT
    train_pos = {}
    if gevT.hist_u is not None:
        hu = gevT.hist_u.cpu().numpy(); hi = gevT.hist_i.cpu().numpy()
        for u, i in zip(hu, hi): train_pos.setdefault(int(u), []).append(int(i))
    Pmax = max((len(v) for v in train_pos.values()), default=1)
    tp = torch.full((usersT.numel(), Pmax), -1, dtype=torch.long, device=DEV)
    for r, u in enumerate(usersT.cpu().numpy()):
        items = train_pos.get(int(u), [])
        if items: tp[r, :len(items)] = torch.tensor(items[:Pmax], device=DEV)

    # align valid positives to the TEST user order (gevV may differ); use gevV on its own users
    out = {}
    for nm, S in [("EASE_base", V['S_item']),
                  ("text_affinity_only", closed_form_base(V['R'], V['dset'], None, lam=800, c2=0.0, a=1.0, half=V['half']))]:
        tr_m, tr_n = split_means(S, usersT, tp)
        te_m, te_n = split_means(S, gevT.users, gevT.pos)
        va_m, va_n = split_means(S, gevV.users, gevV.pos)
        out[nm] = dict(train_mean=round(tr_m, 4), valid_mean=round(va_m, 4), test_mean=round(te_m, 4),
                       test_minus_valid=round(te_m - va_m, 4), train_minus_test=round(tr_m - te_m, 4),
                       n=dict(train=tr_n, valid=va_n, test=te_n),
                       clean_signature=bool(tr_m > te_m and abs(te_m - va_m) < 0.15 * max(abs(tr_m), 1e-6)))
    rep = dict(dataset=ds, note="z-scored scores; clean = train highest & test~=valid (no test-specific info).",
               views=out)
    (OUT / f"leak_test_{ds}.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n========== {ds.upper()} leak-alignment (z-scored score on positives) ==========")
    for nm, o in out.items():
        print(f"  {nm:20s} train={o['train_mean']:+.4f}  valid={o['valid_mean']:+.4f}  test={o['test_mean']:+.4f}  "
              f"(test-valid={o['test_minus_valid']:+.4f}, train-test={o['train_minus_test']:+.4f})  clean={o['clean_signature']}")
    del V; torch.cuda.empty_cache()
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    a = ap.parse_args()
    for ds in a.datasets: run(ds)
    print("\nLEAK_TEST DONE ->", OUT)

#!/usr/bin/env python
"""Ablation: what is SCOPE-U WITHOUT the EASE (item-item) view?

SCOPE-U fuses 3 z-scored views with a val-selected gate:
  col  = a collaborative graph-CF view
  item = closed-form EASE + text        <-- the view we remove
  set  = SCOPE set-completion head

We re-run the EXACT same val grid-search + single test eval (trusted evaluator) but force
the EASE weight to 0 ("no EASE" = col+set), and for context also report each view ALONE,
every pairwise 2-view fusion, and the full 3-view (which reproduces SCOPE-U).
Weights tuned on validation Recall@20, test computed once.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, closed_form_base, SCOPE, evalS_trusted, zr, BAR, DEV

GRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def run(dataset):
    dset = RecDataset(Config("scope", dataset))
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    bar_r, bar_n = BAR[dataset]; half = dset.n_items > 20000; dt = torch.float16 if half else torch.float32
    gevV = GPUEval(dset, "valid", DEV)
    S_item = closed_form_base(R, dset, gevV, half=half).to(dt)
    model = SCOPE(dset.n_items, 256).to(DEV)
    model.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{dataset}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); model.eval()
    S_set = zr(model.score_all(R, degf)).to(dt)
    S_col = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"freedom_{dataset}_scores.npy")).to(dt).to(DEV))
    V = {"col": S_col, "item": S_item, "set": S_set}

    def best_on_val(allow):  # allow = (use_col, use_item, use_set)
        gc = GRID if allow[0] else [0.0]; gi = GRID if allow[1] else [0.0]; gs = GRID if allow[2] else [0.0]
        combos = [(a, b, c) for a in gc for b in gi for c in gs if (a or b or c)]
        best = None; bestr = -1.0
        for (a, b, c) in combos:
            S = a * S_col + b * S_item + c * S_set
            r = gevV.recall_per_user(S, 20).mean().item()
            if r > bestr: bestr = r; best = (a, b, c)
        return best

    def test_of(combo):
        a, b, c = combo
        return evalS_trusted(a * S_col + b * S_item + c * S_set, dset, "test")

    def pct(x, bar): return (x / bar - 1) * 100

    out = {"dataset": dataset, "bar": {"R20": bar_r, "N20": bar_n}, "rows": {}}
    # each view alone
    for nm, S in V.items():
        t = evalS_trusted(S, dset, "test"); out["rows"][f"{nm}_alone"] = {"combo": None, "test": t}
    # pairwise + full
    plan = {
        "noEASE (col+set)":        (True, False, True),
        "noFREEDOM (item+set)":    (False, True, True),
        "noSET = CF-ens (col+item)": (True, True, False),
        "FULL SCOPE-U (col+item+set)": (True, True, True),
    }
    for nm, allow in plan.items():
        combo = best_on_val(allow); t = test_of(combo)
        out["rows"][nm] = {"combo": combo, "test": t}

    print(f"\n========== {dataset.upper()}  (bar R@20={bar_r:.4f} / N@20={bar_n:.4f}; bar+10% = {1.1*bar_r:.4f}/{1.1*bar_n:.4f}) ==========")
    order = ["set_alone", "item_alone", "col_alone", "noEASE (col+set)", "noFREEDOM (item+set)",
             "noSET = CF-ens (col+item)", "FULL SCOPE-U (col+item+set)"]
    for nm in order:
        r = out["rows"][nm]; t = r["test"]; cb = r["combo"]
        print(f"  {nm:30s} combo={str(cb):20s} R@20={t['Recall@20']:.4f}({pct(t['Recall@20'],bar_r):+5.1f}%)  N@20={t['NDCG@20']:.4f}({pct(t['NDCG@20'],bar_n):+5.1f}%)")
    (ROOT / "results" / "scope" / f"scope_u_ablate_ease_{dataset}.json").write_text(json.dumps(out, indent=2, default=str))
    del S_col, S_item, S_set, R; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"]); a = ap.parse_args()
    for ds in a.datasets:
        try: run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()

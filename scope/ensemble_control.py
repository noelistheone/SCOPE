#!/usr/bin/env python
"""View-ablation / ensemble controls for SCOPE-U.

Evaluates EVERY view combination under SCOPE-U's identical protocol — per-combination gate
weights grid-searched on validation Recall@20, then a single trusted test eval — so the
comparison is apples-to-apples. No view is re-tuned beyond the shared gate grid. This
isolates (a) whether SCOPE-U's gain is a generic ensemble effect, (b) the marginal value of
the set-completion view inside the ensemble, and (c) the choice of collaborative view.

Views (all z-scored, exactly as SCOPE-U):
  item = closed-form EASE + text base        (val-tuned lam/a)
  set  = SCOPE set-completion head           (loaded checkpoint)
  col  = a graph-CF collaborative view       (cached scores)
  gume = a stronger dense collaborative view (cached scores)

Comparisons produced, per dataset:
  - set's marginal value inside the ensemble : [col,item,set] vs [col,item], and
                                               [gume,item,set] vs [gume,item]
  - generic strong-pair ensembles            : [gume,item], [col,gume]
  - collaborative-view choice                : [col,item,set] vs [gume,item,set]
  - 4-view ceiling                           : [col,item,set,gume]
Tune on validation, test once.
"""
from __future__ import annotations
import sys, json, itertools
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, closed_form_base, SCOPE, evalS_trusted, zr, BAR, DEV, OUT, CK, ROOT as SROOT

GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]   # shared gate grid

# combos: name -> list of view keys (subset/order of ['col','item','set','gume'])
COMBOS = {
    "GUME":                       ["gume"],
    "FREEDOM":                    ["col"],
    "base(EASE+text)":            ["item"],
    "set":                        ["set"],
    "base+set(SCOPE-v1/G)":       ["item", "set"],
    "base+FREEDOM(no set)":       ["col", "item"],
    "base+GUME(no set)":          ["gume", "item"],          # == GUME + EASE+text ensemble
    "FREEDOM+GUME":               ["col", "gume"],
    "FREEDOM+GUME+base":          ["col", "gume", "item"],   # 3 non-novel views
    "SCOPE-U(base+set+FREEDOM)":  ["col", "item", "set"],
    "GUMEswap(base+set+GUME)":    ["gume", "item", "set"],
    "4view(base+set+FRE+GUME)":   ["col", "item", "set", "gume"],
}


GR4 = [0.0, 0.3, 0.6, 1.0, 2.0]   # coarser grid for >=4-view combos (memory/time)


def gate_select(views, keys, gev):
    """Grid-search nonneg gate on val Recall@20 over the given view keys; return (weights, S)."""
    Vs = [views[k] for k in keys]
    grid = GR if len(keys) < 4 else GR4
    best = None
    for ws in itertools.product(grid, repeat=len(keys)):
        if all(w == 0 for w in ws):
            continue
        S = sum(w * V for w, V in zip(ws, Vs))
        vr = gev.eval(S)["Recall@20"]
        if best is None or vr > best[0]:
            best = (vr, ws)
        del S
    ws = best[1]
    S = sum(w * V for w, V in zip(ws, Vs))
    return ws, S


def run(dataset, seed=2024):
    dset = RecDataset(Config("scope", dataset))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    bar_r, bar_n = BAR.get(dataset, (1.0, 1.0))   # bar only drives the %-display; raw R@20/N@20 are bar-independent
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16   # fp16 views (z-scored; ranking-safe) to fit 4 matrices
    gev = GPUEval(dset, "valid", DEV)

    # build the four views (z-scored)
    V_item = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    stag = f"scope_{dataset}_d256_le1.0_lz1.0_lr0.003" + ('' if seed == 2024 else f'_s{seed}')
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(SROOT / "ckpts" / "scope" / f"{stag}.pt", map_location=DEV)); m.eval()
    with torch.no_grad(): V_set = zr(m.score_all(R, degf)).to(dt)
    V_col = zr(torch.from_numpy(np.load(SROOT / "results" / "baseline_scores" / f"freedom_{dataset}_scores.npy")).to(dt).to(DEV))
    V_gume = zr(torch.from_numpy(np.load(SROOT / "results" / "baseline_scores" / f"gume_{dataset}_scores.npy")).to(dt).to(DEV))
    views = {"col": V_col, "item": V_item, "set": V_set, "gume": V_gume}

    rows = {}
    print(f"\n=== {dataset} (seed {seed}) bar R@20={bar_r} N@20={bar_n} ===", flush=True)
    for name, keys in COMBOS.items():
        try:
            ws, S = gate_select(views, keys, gev)
            t = evalS_trusted(S, dset, "test"); del S; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache(); print(f"  {name:28s} OOM (skipped)", flush=True); continue
        rows[name] = {"keys": keys, "gate": list(ws),
                      "R@20": t["Recall@20"], "N@20": t["NDCG@20"],
                      "R@10": t["Recall@10"], "N@10": t["NDCG@10"]}
        dr = (t["Recall@20"] / bar_r - 1) * 100; dn = (t["NDCG@20"] / bar_n - 1) * 100
        gstr = str([round(x, 1) for x in ws])
        print(f"  {name:28s} gate={gstr:<26} "
              f"R@20={t['Recall@20']:.4f}({dr:+.1f}%) N@20={t['NDCG@20']:.4f}({dn:+.1f}%)", flush=True)
        # incremental save so a later OOM never loses completed rows
        (OUT / (f"ensemble_control_{dataset}" + (f"_s{seed}" if seed != 2024 else "") + ".json")).write_text(
            json.dumps({"dataset": dataset, "seed": seed, "bar": {"R@20": bar_r, "N@20": bar_n},
                        "grid": GR, "rows": rows}, indent=2, default=str))

    out = OUT / (f"ensemble_control_{dataset}" + (f"_s{seed}" if seed != 2024 else "") + ".json")
    out.write_text(json.dumps({"dataset": dataset, "seed": seed, "bar": {"R@20": bar_r, "N@20": bar_n},
                               "grid": GR, "rows": rows}, indent=2, default=str))
    print(f"  -> {out}", flush=True)
    del R, V_item, V_set, V_col, V_gume; torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024])
    a = ap.parse_args()
    for ds in a.datasets:
        for sd in a.seeds:
            try:
                run(ds, seed=sd)
            except Exception as e:
                import traceback; print(f"[{ds} s{sd}] ERR {type(e).__name__}: {e}"); traceback.print_exc()

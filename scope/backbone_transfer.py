#!/usr/bin/env python
"""Backbone portability of the set-completion view (Table S8).

For each heavy multimodal backbone B in {FREEDOM, LGMRec, MGCN, GUME} and each dataset,
this script compares base+B (set-free) against base+B+set, where each ensemble's gate
weights are grid-searched on validation Recall@20 (ensemble_control.gate_select) and the
test comparison is a paired user-level bootstrap on the per-user test Recall@20 marginal
(base+B+set) - (base+B). Each of the 12 table cells (4 backbones x 3 datasets) reports the
test DeltaRecall@20 of adding the set view, its 95% CI, and a two-sided bootstrap p-value.
A consistently positive marginal across backbones shows the set-completion view is a
portable complement to strong collaborative backbones, not tied to one particular view.

Consumes cached per-backbone score matrices results/baseline_scores/{B}_{ds}_scores.npy;
produce them first via: python dump_baseline_scores.py --model <B> --dataset <ds>.
Writes results/scope/backbone_transfer_<ds>.json (rows merge across partial runs, so any
single cell can be reproduced in isolation). GPU required.
Usage: python backbone_transfer.py [--datasets baby sports clothing] [--backbones freedom lgmrec mgcn gume]
"""
from __future__ import annotations
import sys, os, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, ROOT, CK, OUT
from gpu_eval import GPUEval
from ensemble_control import gate_select
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

BACKBONES = ["freedom", "lgmrec", "mgcn", "gume"]
PRED = ROOT / "results" / "baseline_scores"


def run(ds, backbones):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    dt = torch.float16   # fp16 z-scored views (ranking-safe) so several [U,I] matrices fit at once
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

    # shared views: closed-form base (val-tuned) + set-completion head (loaded checkpoint)
    V_item = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(CK / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        V_set = zr(m.score_all(R, degf)).to(dt)
    ru_base = gevT.recall_per_user(V_item).cpu().numpy()

    outfile = OUT / f"backbone_transfer_{ds}.json"
    rows = {}
    if outfile.is_file():   # merge with earlier partial runs so cells can be produced one at a time
        try: rows = json.loads(outfile.read_text()).get("backbones", {})
        except Exception: rows = {}

    for B in backbones:
        f = PRED / f"{B}_{ds}_scores.npy"
        if not f.is_file():
            print(f"[{ds}] {B}: MISSING {f.name} — run: python dump_baseline_scores.py "
                  f"--model {B} --dataset {ds}", flush=True); continue
        V_b = zr(torch.from_numpy(np.load(f)).to(dt).to(DEV))
        views = {"item": V_item, "set": V_set, B: V_b}
        # base+B (set-free) and base+B+set, each gate-selected on validation
        w_bb, S_bb = gate_select(views, ["item", B], gev)
        w_bbs, S_bbs = gate_select(views, ["item", B, "set"], gev)
        ru_bb = gevT.recall_per_user(S_bb).cpu().numpy()
        ru_bbs = gevT.recall_per_user(S_bbs).cpu().numpy()
        # significance of the set marginal ON TOP OF this backbone, and of base+B over base
        set_marg = paired_bootstrap(ru_bbs, ru_bb)
        bb_over_base = paired_bootstrap(ru_bb, ru_base)
        rows[B] = {
            "gate_baseB": list(w_bb), "gate_baseBset": list(w_bbs),
            "R20_baseB": round(float(ru_bb.mean()), 4), "R20_baseBset": round(float(ru_bbs.mean()), 4),
            "set_marginal": set_marg, "baseB_over_base": bb_over_base,
        }
        sm = set_marg
        print(f"[{ds}] {B:8s} base+B={ru_bb.mean():.4f} -> +set={ru_bbs.mean():.4f}  "
              f"set-marg d={sm['mean_delta']:+.4f} ci={[round(x, 4) for x in sm['ci95']]} p={sm['p_two_sided']:.3g}"
              f"{'  *SIG' if sm['p_two_sided'] < 0.05 else ''}", flush=True)
        # incremental save so a later failure never loses completed cells
        outfile.write_text(json.dumps({"dataset": ds, "R20_base": round(float(ru_base.mean()), 4),
                                       "backbones": rows}, indent=2))
        del V_b, S_bb, S_bbs; torch.cuda.empty_cache()

    out = {"dataset": ds, "R20_base": round(float(ru_base.mean()), 4), "backbones": rows}
    outfile.write_text(json.dumps(out, indent=2))
    print(f"  -> {outfile}", flush=True)
    # summary line: how many backbones show a positive / significant set marginal
    pos = sum(1 for r in rows.values() if r["set_marginal"]["mean_delta"] > 0)
    sig = sum(1 for r in rows.values() if r["set_marginal"]["p_two_sided"] < 0.05 and r["set_marginal"]["mean_delta"] > 0)
    print(f"[{ds}] SUMMARY: set marginal positive on {pos}/{len(rows)} backbones, significant on {sig}/{len(rows)}", flush=True)
    del R, V_item, V_set; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    a = ap.parse_args()
    for ds in a.datasets:
        try:
            run(ds, a.backbones)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("BACKBONE_TRANSFER_DONE", flush=True)

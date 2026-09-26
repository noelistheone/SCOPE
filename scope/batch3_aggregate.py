#!/usr/bin/env python
"""Batch 3 (CPU) — aggregate the trained ablations:
  C7-seed  multi-seed mean+/-std of SCOPE-v1 fused R@20/N@20 (margin vs bar)
  B4       SIGReg-in-FULL: fused le0lz0 vs le1lz1 (is SIGReg vestigial in the fused model?)
  B5       text-init-in-FULL: fused ti0 vs ti1 (is the FUSED model text-dependent?)
Reads results/scope/exp_design/fused_*.json (train_fused outputs) + canonical scope_*.json. Writes batch3_aggregate.json.
"""
import json
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1] / "results" / "scope"; OUT = W / "exp_design"
import os
STEM = "scope_full" if os.environ.get("SCOPE_HEAD") == "full" else "scope"   # deployed-run files of the pre-pruning model (scope_full.py) or of scope.py
DS = ["baby", "sports", "clothing"]


def rn(d): return (d.get("Recall@20", 0.0), d.get("NDCG@20", 0.0))


def fused_json(tag):
    p = OUT / f"fused_{tag}.json"
    return json.loads(p.read_text())["fused"] if p.exists() else None


rep = {"C7_multiseed": {}, "B4_sigreg_in_full": {}, "B5_textinit_in_full": {}}

# ---------- C7 multi-seed ----------
for ds in DS:
    base_j = json.loads((W / f"{STEM}_{ds}_d256_le1.0_lz1.0_lr0.003.json").read_text())
    bar_r, bar_n = base_j["bar"]["R20"], base_j["bar"]["N20"]
    Rs, Ns = [], []
    s2024 = rn(base_j["fused"]); Rs.append(s2024[0]); Ns.append(s2024[1])
    if ds == "clothing":
        for s in (2025, 2026):
            j = json.loads((W / f"{STEM}_clothing_d256_le1.0_lz1.0_lr0.003_s{s}.json").read_text())
            Rs.append(j["fused"]["Recall@20"]); Ns.append(j["fused"]["NDCG@20"])
    else:
        for s in (2025, 2026):
            f = fused_json(f"{ds}_ti1_le1.0_lz1.0_s{s}")
            if f: Rs.append(f["Recall@20"]); Ns.append(f["NDCG@20"])
    rep["C7_multiseed"][ds] = dict(
        n_seeds=len(Rs), R20_mean=float(np.mean(Rs)), R20_std=float(np.std(Rs)), R20_all=[round(x, 4) for x in Rs],
        N20_mean=float(np.mean(Ns)), N20_std=float(np.std(Ns)), N20_all=[round(x, 4) for x in Ns],
        bar=(bar_r, bar_n), R20_min_minus_bar=float(min(Rs) - bar_r), N20_min_minus_bar=float(min(Ns) - bar_n),
        all_seeds_above_bar=bool(min(Rs) > bar_r and min(Ns) > bar_n))

# ---------- B4 SIGReg-in-full + B5 text-init-in-full ----------
for ds in DS:
    base_j = json.loads((W / f"{STEM}_{ds}_d256_le1.0_lz1.0_lr0.003.json").read_text())
    full = rn(base_j["fused"]); bar_r, bar_n = base_j["bar"]["R20"], base_j["bar"]["N20"]
    f_le0 = fused_json(f"{ds}_ti1_le0.0_lz0.0_s2024")
    f_ti0 = fused_json(f"{ds}_ti0_le1.0_lz1.0_s2024")
    if f_le0:
        rep["B4_sigreg_in_full"][ds] = dict(
            fused_le1lz1=full, fused_le0lz0=rn(f_le0),
            sigreg_marginal_R=round(full[0] - f_le0["Recall@20"], 4), sigreg_marginal_N=round(full[1] - f_le0["NDCG@20"], 4),
            both_above_bar=dict(le1lz1=(full[0] > bar_r and full[1] > bar_n),
                               le0lz0=(f_le0["Recall@20"] > bar_r and f_le0["NDCG@20"] > bar_n)))
    if f_ti0:
        rep["B5_textinit_in_full"][ds] = dict(
            fused_ti1=full, fused_ti0=rn(f_ti0),
            textinit_marginal_R=round(full[0] - f_ti0["Recall@20"], 4),
            textinit_marginal_R_pct=round((full[0] / f_ti0["Recall@20"] - 1) * 100, 1),
            textinit_marginal_N=round(full[1] - f_ti0["NDCG@20"], 4),
            ti0_above_bar=(f_ti0["Recall@20"] > bar_r and f_ti0["NDCG@20"] > bar_n))

(OUT / "batch3_aggregate.json").write_text(json.dumps(rep, indent=2, default=str))
print("=== C7 multi-seed SCOPE-v1 fused (mean+/-std) ===")
for ds in DS:
    m = rep["C7_multiseed"][ds]
    print(f"  {ds:9} n={m['n_seeds']} R@20={m['R20_mean']:.4f}+/-{m['R20_std']:.4f} {m['R20_all']}  "
          f"N@20={m['N20_mean']:.4f}+/-{m['N20_std']:.4f}  bar={m['bar']}  all-seeds-beat-bar={m['all_seeds_beat_bar']}")
print("\n=== B4 SIGReg-in-FULL (fused le1lz1 vs le0lz0) ===")
for ds in DS:
    b = rep["B4_sigreg_in_full"].get(ds)
    if b: print(f"  {ds:9} le1lz1={b['fused_le1lz1']} le0lz0={b['fused_le0lz0']} "
                f"SIGReg-marginal R={b['sigreg_marginal_R']:+.4f}/N={b['sigreg_marginal_N']:+.4f} beat-bar={b['both_beat_bar']}")
print("\n=== B5 text-init-in-FULL (fused ti1 vs ti0) ===")
for ds in DS:
    b = rep["B5_textinit_in_full"].get(ds)
    if b: print(f"  {ds:9} ti1={b['fused_ti1']} ti0={b['fused_ti0']} "
                f"text-marginal R={b['textinit_marginal_R']:+.4f}({b['textinit_marginal_R_pct']:+.1f}%)  ti0-beats-bar={b['ti0_still_beats_bar']}")
print("\nBATCH3 DONE ->", OUT / "batch3_aggregate.json")

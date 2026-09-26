"""Masking-ratio robustness: is the set-completion result stable across the training mask ratio?

Default SCOPE training draws the context size uniformly in {1..|S_u|-1} (a random target fraction). Here we
retrain SCOPE-v1 with a FIXED target fraction p_mask in {0.1,0.3,0.5,0.7,0.9} and compare the fused (base+set)
test Recall@20/NDCG@20 to the default. Stability across ratios shows the masked-set objective is general, not
a brittle tuning choice. p_mask runs are namespaced (_pm tag) so the default checkpoints are never overwritten.
Writes results/scope/masking_ratio.json. GPU. Usage: python masking_ratio.py
"""
from __future__ import annotations
import json
from pathlib import Path
import scope as S

ROOT = Path(__file__).resolve().parents[1]
import sys
RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
DATASETS = sys.argv[1:] or ["baby", "sports"]

_mrp = ROOT / "results" / "scope" / "masking_ratio.json"
out = json.load(open(_mrp)) if _mrp.exists() else {}      # merge, do not clobber prior datasets
for ds in DATASETS:
    row = {"ratios": {}}
    # default (uniform random target fraction): read the default fused result
    dflt = json.load(open(ROOT / "results" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.json"))["fused"]
    row["uniform_default"] = {"R20": dflt["Recall@20"], "N20": dflt["NDCG@20"]}
    for r in RATIOS:
        rep = S.train(ds, p_mask=r)                    # epochs/patience defaults; namespaced _pm tag
        row["ratios"][str(r)] = {"R20": rep["fused"]["Recall@20"], "N20": rep["fused"]["NDCG@20"]}
        print(f"  [{ds}] p_mask={r}: fused R20={rep['fused']['Recall@20']:.4f} N20={rep['fused']['NDCG@20']:.4f}", flush=True)
    r20s = [row["uniform_default"]["R20"]] + [v["R20"] for v in row["ratios"].values()]
    row["R20_min"] = min(r20s); row["R20_max"] = max(r20s); row["R20_spread"] = max(r20s) - min(r20s)
    out[ds] = row
    print(f"[{ds}] R20 across uniform+5 ratios: min={row['R20_min']:.4f} max={row['R20_max']:.4f} spread={row['R20_spread']:.4f}", flush=True)
    json.dump(out, open(ROOT / "results" / "scope" / "masking_ratio.json", "w"), indent=2)
print("MASKING_RATIO_DONE", flush=True)

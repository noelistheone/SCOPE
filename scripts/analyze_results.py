#!/usr/bin/env python
"""Aggregate baseline results into a summary report.

Reads all `logs/<run>/result.json` files, the diagnostic CSVs, and produces:
  - `results/summary.md` — main report
  - `results/summary.json` — machine-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gather_runs() -> list[dict]:
    runs = []
    for sub in (PROJECT_ROOT / "logs").iterdir():
        if not sub.is_dir():
            continue
        res = sub / "result.json"
        if not res.is_file():
            continue
        try:
            runs.append(json.loads(res.read_text()))
        except Exception:
            continue
    return runs


def best_per_key(runs: list[dict]) -> dict:
    """Keep latest run per (model, dataset, seed)."""
    latest = {}
    for r in runs:
        key = (r["model"], r["dataset"], int(r["seed"]))
        if (key not in latest
                or r.get("run_name", "") > latest[key].get("run_name", "")):
            latest[key] = r
    return latest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md",
                        default=str(PROJECT_ROOT / "results" / "summary.md"))
    parser.add_argument("--output-json",
                        default=str(PROJECT_ROOT / "results" / "summary.json"))
    args = parser.parse_args()

    runs_all = gather_runs()
    runs = list(best_per_key(runs_all).values())
    if not runs:
        print("No runs found.")
        return 1

    by_ds_model = defaultdict(list)
    for r in runs:
        by_ds_model[(r["dataset"], r["model"])].append(r)

    datasets = sorted({r["dataset"] for r in runs})
    models = sorted({r["model"] for r in runs})

    primary_metrics = ["Recall@10", "Recall@20", "Recall@50",
                       "NDCG@10", "NDCG@20", "NDCG@50"]

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# SCOPE Baseline Summary\n\n")
        for ds in datasets:
            f.write(f"## Amazon-{ds}\n\n")
            f.write("| Model | best_epoch | train (min) | "
                    + " | ".join(primary_metrics) + " |\n")
            f.write("|---" * (3 + len(primary_metrics)) + "|\n")
            for m in models:
                rs = by_ds_model.get((ds, m), [])
                if not rs:
                    f.write(f"| {m} | — | — |" + " — |" * len(primary_metrics) + "\n")
                    continue
                r = rs[0]  # one row per (ds, m) for single-seed runs
                test = r.get("test_result", {})
                f.write(f"| {m} | {r.get('best_epoch','—')} | "
                        f"{r.get('train_time_min',0):.1f} | ")
                for k in primary_metrics:
                    v = test.get(k)
                    f.write(f"{v:.4f} | " if v is not None else "— | ")
                f.write("\n")
            f.write("\n")

        # Best per dataset.
        f.write("## Best per dataset (by Recall@20)\n\n")
        for ds in datasets:
            ds_runs = [r for r in runs if r["dataset"] == ds]
            ds_runs = [r for r in ds_runs if r.get("test_result", {}).get("Recall@20") is not None]
            if not ds_runs:
                continue
            ds_runs.sort(key=lambda r: r["test_result"]["Recall@20"], reverse=True)
            top = ds_runs[:5]
            f.write(f"### {ds}\n\n")
            for r in top:
                f.write(f"- **{r['model']}** R@20={r['test_result']['Recall@20']:.4f}"
                        f" (epoch {r['best_epoch']}, {r['train_time_min']:.1f} min)\n")
            f.write("\n")

    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2)

    print(f"Wrote {out_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Compile a comprehensive baseline comparison table from all result.json files.

Aggregates per (model, dataset, seed) → metrics and writes a markdown table
plus a CSV.

Usage:
    python scripts/compile_baseline_table.py --output results/baseline_master.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gather_runs():
    log_root = PROJECT_ROOT / "logs"
    runs = []
    for sub in log_root.iterdir():
        if not sub.is_dir():
            continue
        res = sub / "result.json"
        if not res.is_file():
            continue
        try:
            data = json.loads(res.read_text())
            runs.append(data)
        except Exception:
            continue
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+",
                        default=["Recall@10", "Recall@20", "Recall@50",
                                 "NDCG@10", "NDCG@20", "NDCG@50"])
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "baseline_master.md"))
    args = parser.parse_args()

    runs = gather_runs()
    if not runs:
        print("No result.json found.")
        return 1

    # For each (model, dataset, seed), keep the most recent run.
    latest: dict[tuple[str, str, int], dict] = {}
    for r in runs:
        key = (r.get("model", "?"), r.get("dataset", "?"), int(r.get("seed", -1)))
        prev = latest.get(key)
        if prev is None or r.get("run_name", "") > prev.get("run_name", ""):
            latest[key] = r

    runs = sorted(latest.values(),
                  key=lambda d: (d.get("dataset"), d.get("model"), d.get("seed")))

    datasets = sorted({r["dataset"] for r in runs})
    models = sorted({r["model"] for r in runs})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Baseline Comparison Master Table\n\n")
        for ds in datasets:
            f.write(f"## Amazon-{ds.capitalize()}\n\n")
            cols = ["Model", "best_epoch", "train_min"] + list(args.metrics)
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            for m in models:
                row = next((r for r in runs if r["model"] == m and r["dataset"] == ds), None)
                if row is None:
                    cells = [m, "—", "—"] + ["—"] * len(args.metrics)
                else:
                    cells = [m,
                             str(row.get("best_epoch", "—")),
                             f"{row.get('train_time_min', 0):.1f}"]
                    test = row.get("test_result", {})
                    for k in args.metrics:
                        v = test.get(k)
                        cells.append(f"{v:.4f}" if v is not None else "—")
                f.write("| " + " | ".join(cells) + " |\n")
            f.write("\n")

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

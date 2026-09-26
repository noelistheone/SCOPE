#!/usr/bin/env python
"""Orchestrate training of multiple (model, dataset) pairs.

Discovers `logs/<run>/result.json` files to detect already-completed runs and
skips them by default. Each new run invokes `python -m src.main`.

Usage:
    python scripts/run_all_baselines.py --models lightgcn vbpr mmgcn --datasets baby
    python scripts/run_all_baselines.py --models all --datasets baby
    python scripts/run_all_baselines.py --models all --datasets baby sports clothing --skip-done
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MODEL_REGISTRY  # noqa: E402

ALL_MODELS = sorted(MODEL_REGISTRY.keys())
ALL_DATASETS = ("baby", "sports", "clothing", "elec", "microlens")
RESULTS_DIR = PROJECT_ROOT / "results"


def existing_runs(model: str, dataset: str, seed: int) -> list[Path]:
    """Return any logs/<run>/result.json that match (model, dataset, seed)."""
    found = []
    log_root = PROJECT_ROOT / "logs"
    if not log_root.is_dir():
        return found
    for sub in log_root.iterdir():
        if not sub.is_dir():
            continue
        res = sub / "result.json"
        if not res.is_file():
            continue
        try:
            data = json.loads(res.read_text())
            if (data.get("model") == model
                    and data.get("dataset") == dataset
                    and int(data.get("seed", -1)) == seed):
                found.append(res)
        except Exception:
            continue
    return found


def run_one(model: str, dataset: str, gpu: int, seed: int) -> tuple[int, Path | None]:
    """Invoke ``python -m src.main``. Returns (returncode, result_json_path)."""
    cmd = [
        sys.executable, "-m", "src.main",
        "--model", model,
        "--dataset", dataset,
        "--gpu", str(gpu),
        "--seed", str(seed),
    ]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, "4")

    print(f"\n=== [{time.strftime('%H:%M:%S')}] Training {model} on {dataset} (seed={seed}, gpu={gpu}) ===",
          flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    dt = (time.time() - t0) / 60.0
    print(f"=== [{time.strftime('%H:%M:%S')}] Finished in {dt:.1f} min, rc={proc.returncode} ===",
          flush=True)

    if proc.returncode != 0:
        return proc.returncode, None
    # Find the result.json (newest matching run)
    matches = existing_runs(model, dataset, seed)
    if not matches:
        return 1, None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    return 0, latest


def compile_csv(dataset: str, seed: int) -> None:
    """Build/refresh results/baseline_<dataset>_seed<seed>.csv from all matching JSONs."""
    runs = []
    log_root = PROJECT_ROOT / "logs"
    if not log_root.is_dir():
        return
    for sub in log_root.iterdir():
        res = sub / "result.json"
        if not res.is_file():
            continue
        try:
            data = json.loads(res.read_text())
        except Exception:
            continue
        if data.get("dataset") != dataset or int(data.get("seed", -1)) != seed:
            continue
        runs.append(data)

    if not runs:
        return

    runs.sort(key=lambda d: d["model"])
    keys = set()
    for r in runs:
        keys.update(r.get("test_result", {}).keys())
    metric_keys = sorted(keys)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"baseline_{dataset}_seed{seed}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "seed", "best_epoch",
                    "train_time_min", "best_valid_score"] + metric_keys)
        for r in runs:
            row = [r["model"], r["dataset"], r["seed"], r["best_epoch"],
                   round(r["train_time_min"], 1), round(r["best_valid_score"], 4)]
            for k in metric_keys:
                v = r["test_result"].get(k)
                row.append(round(v, 4) if v is not None else "")
            w.writerow(row)
    print(f"Wrote {out} with {len(runs)} rows", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names, or 'all'.")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset names, or 'all'.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip (model, dataset) pairs that already have a result.json")
    args = parser.parse_args()

    models = ALL_MODELS if args.models == ["all"] else list(args.models)
    datasets = list(ALL_DATASETS) if args.datasets == ["all"] else list(args.datasets)
    for m in models:
        if m not in MODEL_REGISTRY:
            sys.exit(f"Unknown model: {m}")
    for d in datasets:
        if d not in ALL_DATASETS:
            sys.exit(f"Unknown dataset: {d}")

    total = len(models) * len(datasets)
    done, skipped, failed = 0, 0, 0
    for ds in datasets:
        for m in models:
            if args.skip_done and existing_runs(m, ds, args.seed):
                print(f"SKIP {m} on {ds} (seed {args.seed}) — result.json exists",
                      flush=True)
                skipped += 1
                continue
            rc, _ = run_one(m, ds, args.gpu, args.seed)
            if rc == 0:
                done += 1
            else:
                failed += 1
            compile_csv(ds, args.seed)

    print(f"\nDone: {done} succeeded, {skipped} skipped, {failed} failed (of {total})",
          flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

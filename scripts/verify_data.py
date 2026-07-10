#!/usr/bin/env python
"""Verify dataset integrity: file presence, shape, dtype, basic stats.

Usage:
    python scripts/verify_data.py                # verifies all three
    python scripts/verify_data.py --dataset baby
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
KNOWN_DATASETS = ("baby", "sports", "elec", "clothing")


def _sha256_short(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def verify_one(name: str) -> bool:
    root = DATA_ROOT / name
    if not root.is_dir():
        print(f"  MISSING DIR: {root}")
        return False

    inter = root / f"{name}.inter"
    img = root / "image_feat.npy"
    txt = root / "text_feat.npy"

    ok = True

    # Inter file.
    if not inter.is_file():
        print(f"  MISSING: {inter}")
        ok = False
    else:
        with inter.open("r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print(f"  {inter.name}: {n_lines} lines, sha256:{_sha256_short(inter)}")

    # Image feat.
    if not img.is_file():
        print(f"  MISSING: {img}")
        ok = False
    else:
        arr = np.load(img, mmap_mode="r")
        print(f"  {img.name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"sha256:{_sha256_short(img)}")
        if arr.ndim != 2:
            print(f"    ERROR: image_feat must be 2D")
            ok = False
        if arr.dtype not in (np.float32, np.float16, np.float64):
            print(f"    ERROR: image_feat must be floating dtype")
            ok = False

    # Text feat.
    if not txt.is_file():
        print(f"  MISSING: {txt}")
        ok = False
    else:
        arr = np.load(txt, mmap_mode="r")
        print(f"  {txt.name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"sha256:{_sha256_short(txt)}")
        if arr.ndim != 2:
            print(f"    ERROR: text_feat must be 2D")
            ok = False

    # Consistency: image_feat.shape[0] == text_feat.shape[0]
    if img.is_file() and txt.is_file():
        a = np.load(img, mmap_mode="r")
        b = np.load(txt, mmap_mode="r")
        if a.shape[0] != b.shape[0]:
            print(f"    ERROR: image_feat n_items ({a.shape[0]}) != "
                  f"text_feat n_items ({b.shape[0]})")
            ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=None,
                        choices=list(KNOWN_DATASETS),
                        help="Which dataset(s) to verify (default: all present).")
    args = parser.parse_args()

    if args.dataset:
        names = args.dataset
    else:
        names = [p.name for p in sorted(DATA_ROOT.iterdir())
                 if p.is_dir() and p.name in KNOWN_DATASETS]
        if not names:
            print(f"No dataset directories found under {DATA_ROOT}.")
            print("Run `python scripts/download_data.py` first.")
            return 1

    all_ok = True
    for n in names:
        print(f"\n[{n}]")
        if not verify_one(n):
            all_ok = False

    print()
    if all_ok:
        print("All checks passed.")
        return 0
    print("Some checks failed (see above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

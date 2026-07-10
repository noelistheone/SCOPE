#!/usr/bin/env python
"""Download MMRec-preprocessed Amazon datasets via gdown.

The MMRec authors host Baby/Sports/Elec at:
    https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG

Usage:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset baby
    python scripts/download_data.py --dataset baby sports

Notes:
- Requires `gdown` (pip install gdown).
- "clothing" is NOT in the public MMRec Drive folder. We surface a clear error
  pointing the user at https://nijianmo.github.io/amazon/index.html if they want
  it; we don't attempt to fetch raw Amazon Reviews automatically.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

MMREC_FOLDER_URL = (
    "https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG"
)

KNOWN_DATASETS = ("baby", "sports", "elec", "clothing")
PUBLIC_DATASETS = ("baby", "sports", "elec")  # actually in the MMRec folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def _have_gdown() -> bool:
    return shutil.which("gdown") is not None


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def download_folder(target_dir: Path) -> int:
    """Pull the full MMRec Drive folder into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gdown", "--folder", MMREC_FOLDER_URL, "-O", str(target_dir)]
    return _run(cmd)


def organize_downloaded(raw_dir: Path,
                        datasets: Iterable[str]) -> None:
    """Move/rename files from ``raw_dir`` (where gdown landed) into ``data/<ds>/``.

    The MMRec Drive layout is: each dataset has its own subdirectory inside the
    pulled folder. We just rename the inner dir to lower-case and move its
    contents into ``data/<ds>/``.
    """
    if not raw_dir.exists():
        print(f"ERROR: raw download dir does not exist: {raw_dir}", file=sys.stderr)
        return
    # gdown places the folder as raw_dir/<folder-name>/<ds-subdir>/files.
    # Find the inner per-dataset directories regardless of how it nested.
    candidates = list(raw_dir.rglob("*"))
    inner_dirs = {p.name.lower(): p for p in candidates if p.is_dir()}

    for ds in datasets:
        ds_lower = ds.lower()
        if ds_lower not in inner_dirs:
            print(f"NOTE: '{ds_lower}' not found in downloaded folder; skipping.")
            continue
        src = inner_dirs[ds_lower]
        dest = DATA_ROOT / ds_lower
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if not f.is_file():
                continue
            target = dest / f.name
            if target.exists():
                continue
            shutil.move(str(f), str(target))
        print(f"  -> {ds_lower}: {len(list(dest.iterdir()))} files in {dest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["all"],
                        choices=list(KNOWN_DATASETS) + ["all"],
                        help="Dataset(s) to fetch.")
    parser.add_argument("--raw-dir", type=Path, default=DATA_ROOT / "_mmrec_raw",
                        help="Where to land the gdown folder before organizing.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Only run the organize step on an existing raw dir.")
    args = parser.parse_args()

    if "all" in args.dataset:
        chosen = list(PUBLIC_DATASETS)
        if "clothing" in args.dataset:
            chosen.append("clothing")
    else:
        chosen = list(args.dataset)

    # Warn on clothing (not in the public folder).
    if "clothing" in chosen:
        print("WARNING: Amazon-Clothing is not in the public MMRec Drive folder.\n"
              "         You may need to construct it from raw Amazon Reviews:\n"
              "         https://nijianmo.github.io/amazon/index.html\n"
              "         This script will only fetch Baby/Sports/Elec from the Drive folder.\n")

    if not args.skip_download:
        if not _have_gdown():
            print("ERROR: `gdown` not found. Install it with:", file=sys.stderr)
            print("    pip install gdown", file=sys.stderr)
            return 1
        rc = download_folder(args.raw_dir)
        if rc != 0:
            print(f"gdown failed with exit code {rc}. "
                  "Falling back to manual download — open:", file=sys.stderr)
            print(f"    {MMREC_FOLDER_URL}", file=sys.stderr)
            return rc

    organize_downloaded(args.raw_dir, chosen)
    print("\nDone. Verify with: python scripts/verify_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

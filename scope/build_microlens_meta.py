#!/usr/bin/env python
"""Item metadata for MicroLens in the column layout of the Amazon meta-<ds>.csv files.

The MMRec release of MicroLens ships no item metadata, so the tag and TF-IDF closed forms of G3 had no side
information there. The official MicroLens-100k release (recsys.westlake.edu.cn/MicroLens-100k-Dataset) has, per
video, an English title (MicroLens-100k_title_en.csv, "videoID, title") and one category label
(tags_to_summary.csv, "videoID,category"). This script maps them onto the MMRec item order through
data/microlens/i_id_mapping.csv (column `asin` = official video ID) and writes data/microlens/meta-microlens.csv
with the Amazon columns: `categories` = [['<category>']], `title` = the English title, and `description`, `brand`
and the remaining Amazon columns left empty (MicroLens has no such fields).

Only item-side attributes are used. The official comments and like/view counts are NOT used: they are produced by
user behaviour and could carry held-out interactions.

Before writing, the ID mapping is verified against the official interaction file: every (user, item) pair of
microlens.inter, mapped back through u_id_mapping.csv and i_id_mapping.csv, must occur in MicroLens-100k_pairs.csv.
Writes meta-microlens.csv (never overwritten) and meta-microlens.provenance.json (sha256 of every input and output).
Download the four official files into data/microlens/official/ first (see data/README.md).
Usage: python scope/build_microlens_meta.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = Path(os.environ["SCOPE_ROOT"]).resolve() if os.environ.get("SCOPE_ROOT") else HERE.parents[1]
D = ROOT / "data" / "microlens"
OFF = D / "official"
AMAZON_COLS = ["itemID", "asin", "categories", "description", "title", "price", "imUrl", "brand", "related",
               "salesRank"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def read_id_value(p: Path) -> dict[int, str]:
    """'<id>,<value>' per line; the value may itself contain commas (titles), so split on the first comma only."""
    out = {}
    with open(p, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            k, sep, v = line.partition(",")
            if not sep or not k.strip().isdigit():
                raise ValueError(f"{p.name}:{ln}: unexpected line {line[:80]!r}")
            k = int(k)
            if k in out:
                raise ValueError(f"{p.name}:{ln}: duplicate id {k}")
            out[k] = v.strip()
    return out


def main():
    out_csv, out_prov = D / "meta-microlens.csv", D / "meta-microlens.provenance.json"
    if out_csv.exists():
        sys.exit(f"{out_csv} exists; refusing to overwrite")
    src = {"title": OFF / "MicroLens-100k_title_en.csv", "category": OFF / "tags_to_summary.csv",
           "pairs": OFF / "MicroLens-100k_pairs.csv", "source_note": OFF / "SOURCE.txt",
           "i_id_mapping": D / "i_id_mapping.csv", "u_id_mapping": D / "u_id_mapping.csv",
           "inter": D / "microlens.inter"}
    for k, p in src.items():
        if not p.is_file():
            sys.exit(f"missing input {k}: {p}")

    im = pd.read_csv(src["i_id_mapping"], sep="\t").sort_values("itemID").reset_index(drop=True)
    um = pd.read_csv(src["u_id_mapping"], sep="\t")
    n = len(im)
    assert np.array_equal(im["itemID"].to_numpy(), np.arange(n)), "i_id_mapping itemIDs are not 0..n-1"
    inter = pd.read_csv(src["inter"], sep="\t", usecols=["userID", "itemID"])
    assert inter["itemID"].max() < n

    # 1) the ID mapping reproduces the official interactions exactly
    pairs = pd.read_csv(src["pairs"])
    ucol, icol = pairs.columns[0], pairs.columns[1]
    official = set(zip(pairs[ucol].astype(np.int64).to_numpy(), pairs[icol].astype(np.int64).to_numpy()))
    ou = inter["userID"].map(dict(zip(um["userID"], um["user_id"]))).astype(np.int64).to_numpy()
    oi = inter["itemID"].map(dict(zip(im["itemID"], im["asin"]))).astype(np.int64).to_numpy()
    found = np.fromiter(((a, b) in official for a, b in zip(ou, oi)), bool, len(ou))
    if not found.all():
        sys.exit(f"ID mapping check FAILED: {int((~found).sum())} of {len(found)} benchmark interactions are not in "
                 f"the official pairs file")

    # 2) titles and categories for every item, in itemID order
    titles, cats = read_id_value(src["title"]), read_id_value(src["category"])
    vids = im["asin"].astype(np.int64).to_numpy()
    miss_t = [int(v) for v in vids if int(v) not in titles]
    miss_c = [int(v) for v in vids if int(v) not in cats]
    if miss_t or miss_c:
        sys.exit(f"missing titles for {len(miss_t)} items, categories for {len(miss_c)} items")
    rows = []
    for item_id, vid in enumerate(vids):
        c = cats[int(vid)]
        assert c and "," not in c and "'" not in c and "[" not in c, f"unexpected category label {c!r}"
        rows.append({"itemID": item_id, "asin": str(int(vid)), "categories": f"[['{c}']]", "description": "",
                     "title": titles[int(vid)], "price": "", "imUrl": "", "brand": "", "related": "",
                     "salesRank": ""})
    meta = pd.DataFrame(rows, columns=AMAZON_COLS)
    meta.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    back = pd.read_csv(out_csv, dtype={"asin": str})
    assert len(back) == n and (back["itemID"].to_numpy() == np.arange(n)).all()
    assert (back["asin"].to_numpy() == im["asin"].astype(str).to_numpy()).all()
    cat_counts = back["categories"].value_counts()
    prov = dict(script="build_microlens_meta.py", n_items=n,
                inputs={k: dict(file=str(p.relative_to(ROOT)), sha256=sha256(p)) for k, p in src.items()},
                output=dict(file=str(out_csv.relative_to(ROOT)), sha256=sha256(out_csv)),
                id_mapping_check=dict(benchmark_interactions=int(len(found)), found_in_official_pairs=int(found.sum())),
                n_categories=int(cat_counts.size), items_per_category={k: int(v) for k, v in cat_counts.items()},
                empty_titles=int(back["title"].isna().sum()),
                fields_used=["title (MicroLens-100k_title_en.csv)", "one category label (tags_to_summary.csv)"],
                fields_not_used=["comments and like/view counts (user behaviour; could carry held-out interactions)"],
                columns_left_empty=["description", "brand", "price", "imUrl", "related", "salesRank"])
    out_prov.write_text(json.dumps(prov, indent=1))
    print(f"wrote {out_csv} ({n} items, {cat_counts.size} categories, {prov['empty_titles']} empty titles); "
          f"ID mapping: {int(found.sum())}/{len(found)} benchmark interactions found in the official pairs")


if __name__ == "__main__":
    main()

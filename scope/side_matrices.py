#!/usr/bin/env python
"""X4 -- side-information matrices for the content closed-form baselines of G3.

CPU only, deterministic. For each Amazon dataset (Baby/Sports/Clothing) it reads data/<ds>/meta-<ds>.csv,
asserts that the metadata rows are in the framework's itemID order (0..n_items-1, where n_items comes from the
same RecDataset the SCOPE harness uses, and asin-for-asin identical to data/<ds>/i_id_mapping.csv when that file
exists), and writes three ITEM-MAJOR sparse matrices ([n_items, V], CSR, float32) with scipy.sparse.save_npz:

  <ds>_tags_jeunen.npz          binary CEASE tag matrix, faithful to the reference code of Jeunen, Van Balen and
                                Goethals (RecSys'20, github olivierjeunen/ease-side-info-recsys-2020, MIT,
                                src/PreprocessAmazonSportsOutdoors.py), read 2026-09-25:
                                  categories: str.replace('[', ']', "'" -> ''), split(','), strip; the dataset's
                                    root category label is dropped; keep labels in >= 2 items (no upper bound);
                                  description/title/brand: str(s).lower(), string.punctuation -> ' ', split(' ');
                                    keep tokens in >= 3 and <= n_items//4 items (n_items = distinct items rated);
                                  per-field vocabularies, blocks stacked in the order (cat, desc, title, brand).
  <ds>_tags_uniform_min3_maxIdiv4.npz
                                binary tag matrix of the uniform variant (all four fields lower-cased,
                                punctuation -> space, whitespace split, per-field item frequency in [3, n_items//4]).
  <ds>_tfidf_min3_max0.25_l2.npz
                                TF-IDF of the concatenated fields (title, brand, categories, description),
                                sklearn TfidfVectorizer(min_df=3, max_df=0.25, norm='l2').

plus <ds>_side_meta.json (file names, sha256, shapes, nnz, vocabulary sizes, alignment report, n_items, sha256 of the
.inter and the metadata file) and <ds>_vocab_<variant>.json. G3 (closedform_family.py) refuses side files whose
n_items / .inter sha256 / file sha256 disagree with this metadata. In G3 the paper's T (|V| x |I|) is the transpose
of these item-major matrices, i.e. the tag Gram T^T T is computed as X X^T.

Deviations from the reference code (all forced or bug-avoiding, recorded in the JSON):
  * a missing (NaN) categories cell becomes '' (the reference code would crash on a float); empty category labels
    are dropped. NaN description/title/brand cells become the token 'nan' in the Jeunen variant exactly as the
    reference code's str(s) does (it is then subject to the same support filter); the uniform variant maps NaN to ''.
  * the root label is dataset specific (Jeunen only had Sports: 'Sports & Outdoors'). Baby: 'Baby'. Clothing's root
    'Clothing, Shoes & Jewelry' is itself split by the reference split(',') into 'Clothing' and 'Shoes & Jewelry';
    both are dropped.
MicroLens: the MMRec release has no item metadata; data/microlens/meta-microlens.csv is built from the official
MicroLens-100k release by build_microlens_meta.py (title + one category label per video; see that script).

Usage (from the repository root; SCOPE_ROOT overrides the root derived from this file's location):
  python scope/side_matrices.py --datasets baby sports clothing
  python scope/side_matrices.py --datasets baby --smoke      # writes into <out>/smoke/
Existing outputs are never overwritten: a dataset whose <ds>_side_meta.json exists is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve()
ROOT = Path(os.environ["SCOPE_ROOT"]).resolve() if os.environ.get("SCOPE_ROOT") else HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AMAZON = ("baby", "sports", "clothing")
# MicroLens: meta-microlens.csv is built from the official MicroLens-100k release by build_microlens_meta.py (English
# title + one category label per video, the other Amazon columns empty); it has no root category label.
SIDE_DATASETS = AMAZON + ("microlens",)
ROOT_CATS = {"baby": {"Baby"}, "sports": {"Sports & Outdoors"}, "clothing": {"Clothing", "Shoes & Jewelry"},
             "microlens": set()}
FIELD_ORDER = ("categories", "description", "title", "brand")      # reference vstack order: cat, desc, title, brand
PUNCT = str.maketrans(string.punctuation, " " * len(string.punctuation))
MINSUP = 3                    # description/title/brand (and every field of the uniform variant)
CAT_MINSUP_JEUNEN = 2         # reference code: categories kept if in >= 2 items, no maximum
TFIDF_KW = dict(min_df=3, max_df=0.25, norm="l2")
FILES = {"tags_jeunen": "{ds}_tags_jeunen.npz",
         "tags_uniform": "{ds}_tags_uniform_min3_maxIdiv4.npz",
         "tfidf": "{ds}_tfidf_min3_max0.25_l2.npz"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def rel(p) -> str:
    """Path relative to the repository root when possible (portable records), else absolute."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(p).resolve())


# ------------------------------------------------------------------------------------------------ tokenisers
def jeunen_word_sets(values):
    """Reference: str(s).lower().translate(punct->space); .str.split(' '); per-(item, token) de-duplication.
    '' tokens (double spaces) and 'nan' (missing cell) are kept here and go through the same support filter."""
    return [set(str(v).lower().translate(PUNCT).split(" ")) for v in values]


def jeunen_cat_sets(values, root):
    """Reference: s.replace('[','').replace(']','').replace("'",'').strip(); split(','); strip; drop root label."""
    out = []
    for v in values:
        s = "" if is_nan(v) else str(v)
        s = s.replace("[", "").replace("]", "").replace("'", "").strip()
        toks = {t.strip() for t in s.split(",")}
        out.append({t for t in toks if t not in ("", " ") and t not in root})
    return out


def uniform_sets(values):
    """Uniform tokenisation: lower-case, punctuation -> space, whitespace split (no empty tokens); NaN -> ''."""
    return [set(("" if is_nan(v) else str(v)).lower().translate(PUNCT).split()) for v in values]


def field_block(sets, n, lo, hi):
    """Binary [n, V] CSR for one field; vocabulary = tokens whose ITEM frequency lies in [lo, hi] (hi=None: none).
    Built through COO (never dok: scipy 1.15 dict.update(dok) is a silent no-op) and nnz-asserted."""
    cnt = Counter(t for s in sets for t in s)
    vocab = sorted(t for t, c in cnt.items() if c >= lo and (hi is None or c <= hi))
    index = {t: j for j, t in enumerate(vocab)}
    rows, cols = [], []
    for i, s in enumerate(sets):
        for t in s:
            j = index.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
    X = sp.coo_matrix((np.ones(len(rows), np.float32), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
                      shape=(n, len(vocab))).tocsr()
    X.sum_duplicates()
    assert X.nnz == len(rows), "duplicate (item, token) pairs after per-item de-duplication"
    assert X.nnz == 0 or bool(np.all(X.data == 1.0)), "tag matrix is not binary"
    stats = dict(vocab=len(vocab), nnz=int(X.nnz), n_tokens_before_filter=len(cnt),
                 support_lo=lo, support_hi=hi)
    return X, vocab, stats


def stack_fields(blocks, n):
    mats = [b[0] for b in blocks if b[0].shape[1] > 0]
    X = sp.hstack(mats, format="csr") if mats else sp.csr_matrix((n, 0), dtype=np.float32)
    X = X.astype(np.float32).tocsr()
    assert X.shape[0] == n and X.nnz == sum(b[0].nnz for b in blocks), "field stacking lost nonzeros"
    return X


def row_stats(X):
    per = np.diff(X.indptr)
    return dict(shape=[int(X.shape[0]), int(X.shape[1])], nnz=int(X.nnz), items_without_entries=int((per == 0).sum()),
                entries_per_item_mean=float(per.mean()), entries_per_item_median=float(np.median(per)),
                entries_per_item_max=int(per.max()) if per.size else 0)


# ------------------------------------------------------------------------------------------------ alignment
def load_framework_dataset(ds):
    import logging
    logging.disable(logging.INFO)
    from src.utils import Config
    from src.data.dataset import RecDataset
    return RecDataset(Config("scope", ds))


def check_alignment(ds, meta, dset):
    n = int(dset.n_items)
    rep = {"n_items_framework": n}
    ids = meta["itemID"].to_numpy()
    assert len(meta) == n, f"[{ds}] metadata rows {len(meta)} != framework n_items {n}"
    assert np.array_equal(ids, np.arange(n)), f"[{ds}] metadata itemID column is not 0..n_items-1 in file order"
    assert meta["asin"].is_unique, f"[{ds}] duplicate asin in metadata"
    f = dset.item_field
    used = np.unique(np.concatenate([dset.train_df[f].to_numpy(), dset.valid_df[f].to_numpy(),
                                     dset.test_df[f].to_numpy()]))
    assert used.min() >= 0 and used.max() < n
    rep["n_items_in_inter"] = int(used.size)
    mp = Path(dset.data_path) / "i_id_mapping.csv"
    if mp.is_file():
        m = pd.read_csv(mp, sep="\t", dtype={"asin": str}).sort_values("itemID").reset_index(drop=True)
        assert np.array_equal(m["itemID"].to_numpy(), np.arange(n)), f"[{ds}] i_id_mapping itemIDs != 0..n_items-1"
        assert bool((m["asin"].to_numpy() == meta["asin"].astype(str).to_numpy()).all()), \
            f"[{ds}] asin order in meta csv differs from i_id_mapping.csv"
        rep["i_id_mapping"] = f"{rel(mp)}: asin identical row by row"
        rep["i_id_mapping_sha256"] = sha256_file(mp)
    else:
        rep["i_id_mapping"] = "absent (MMRec ships none for Baby); itemID order of the meta csv asserted only"
    if dset.t_feat is not None:
        assert dset.t_feat.shape[0] == n, f"[{ds}] text_feat rows {dset.t_feat.shape[0]} != n_items"
        rep["text_feat_shape"] = list(dset.t_feat.shape)
    empty = 0
    for _, r in meta[list(FIELD_ORDER)].iterrows():
        if all(is_nan(r[c]) or str(r[c]).strip() == "" for c in FIELD_ORDER):
            empty += 1
    rep["items_with_all_fields_empty"] = empty
    return rep


# ------------------------------------------------------------------------------------------------ builders
def build_jeunen(ds, meta, n, n_rated):
    maxsup = n_rated // 4
    blocks, vocab, stats = [], {}, {}
    cat = field_block(jeunen_cat_sets(meta["categories"].tolist(), ROOT_CATS[ds]), n, CAT_MINSUP_JEUNEN, None)
    blocks.append(cat); vocab["categories"] = cat[1]; stats["categories"] = cat[2]
    for fld in ("description", "title", "brand"):
        b = field_block(jeunen_word_sets(meta[fld].tolist()), n, MINSUP, maxsup)
        blocks.append(b); vocab[fld] = b[1]; stats[fld] = b[2]
    X = stack_fields(blocks, n)
    return X, vocab, dict(fields=stats, maxsup=maxsup, root_labels_dropped=sorted(ROOT_CATS[ds]))


def build_uniform(meta, n):
    maxsup = n // 4
    blocks, vocab, stats = [], {}, {}
    for fld in FIELD_ORDER:
        b = field_block(uniform_sets(meta[fld].tolist()), n, MINSUP, maxsup)
        blocks.append(b); vocab[fld] = b[1]; stats[fld] = b[2]
    X = stack_fields(blocks, n)
    return X, vocab, dict(fields=stats, maxsup=maxsup)


def build_tfidf(meta, n):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = []
    for t, b, c, d in zip(meta["title"], meta["brand"], meta["categories"], meta["description"]):
        parts = ["" if is_nan(v) else str(v) for v in (t, b, c, d)]
        docs.append(" ".join(parts).lower().translate(PUNCT))
    vec = TfidfVectorizer(lowercase=True, dtype=np.float32, **TFIDF_KW)
    X = vec.fit_transform(docs).tocsr().astype(np.float32)
    assert X.shape[0] == n and X.nnz > 0
    nz = np.diff(X.indptr) > 0
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    assert np.allclose(norms[nz], 1.0, atol=1e-4), "TF-IDF rows are not l2-normalised"
    vocab = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    params = {k: v for k, v in vec.get_params().items()
              if k in ("min_df", "max_df", "norm", "token_pattern", "sublinear_tf", "smooth_idf", "use_idf")}
    return X, {"all": vocab}, dict(vocab=len(vocab), sklearn_params=params)


def process(ds, out_dir, smoke):
    t0 = time.time()
    meta_json = out_dir / f"{ds}_side_meta.json"
    if meta_json.exists():
        print(f"[{ds}] {meta_json} exists -- skipped (outputs are never overwritten)", flush=True)
        return None
    dset = load_framework_dataset(ds)
    n = int(dset.n_items)
    meta_csv = Path(dset.data_path) / f"meta-{ds}.csv"
    meta = pd.read_csv(meta_csv, dtype={"asin": str})
    for c in ("itemID", "asin") + FIELD_ORDER:
        assert c in meta.columns, f"[{ds}] column {c} missing in {meta_csv}"
    align = check_alignment(ds, meta, dset)
    n_rated = align["n_items_in_inter"]
    inter = Path(dset.data_path) / str(dset.config["inter_file_name"])
    record = dict(script="side_matrices.py", dataset=ds, smoke=bool(smoke), n_items=n,
                  inter_file=rel(inter), inter_sha256=sha256_file(inter),
                  meta_file=rel(meta_csv), meta_sha256=sha256_file(meta_csv),
                  orientation="item-major [n_items, V] CSR float32 (the paper's T is the transpose)",
                  alignment=align, files={}, deviations_from_reference=[
                      "NaN categories -> '' and empty category labels dropped (reference would crash on NaN)",
                      "dataset-specific root category labels dropped: " + ", ".join(sorted(ROOT_CATS[ds])),
                      "uniform variant: NaN -> '' ; Jeunen variant keeps str(NaN)='nan' exactly like the reference"],
                  reference_code="github.com/olivierjeunen/ease-side-info-recsys-2020 "
                                 "src/PreprocessAmazonSportsOutdoors.py (MIT)")
    builders = [("tags_jeunen", lambda: build_jeunen(ds, meta, n, n_rated)),
                ("tags_uniform", lambda: build_uniform(meta, n)),
                ("tfidf", lambda: build_tfidf(meta, n))]
    for kind, fn in builders:
        tb = time.time()
        X, vocab, info = fn()
        assert X.shape[0] == n and X.nnz > 0, f"[{ds}] {kind}: empty matrix"
        path = out_dir / FILES[kind].format(ds=ds)
        assert not path.exists(), f"{path} exists; refusing to overwrite"
        sp.save_npz(path, X, compressed=True)
        back = sp.load_npz(path)
        assert back.shape == X.shape and back.nnz == X.nnz, f"[{ds}] {kind}: save/load round trip changed the matrix"
        vpath = out_dir / f"{ds}_vocab_{kind}.json"
        vpath.write_text(json.dumps(vocab))
        record["files"][kind] = dict(file=path.name, sha256=sha256_file(path), vocab_file=vpath.name,
                                     build_s=round(time.time() - tb, 2), info=info, **row_stats(X))
        rs = record["files"][kind]
        print(f"[{ds}] {kind:13s} shape={rs['shape']} nnz={rs['nnz']} "
              f"items_without_entries={rs['items_without_entries']} mean/item={rs['entries_per_item_mean']:.1f} "
              f"({rs['build_s']}s) -> {path.name}", flush=True)
    record["wall_s"] = round(time.time() - t0, 2)
    record["cuda_max_memory_allocated"] = "n/a (CPU-only script)"
    tmp = meta_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, meta_json)
    print(f"[{ds}] done in {record['wall_s']}s -> {meta_json}", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description="X4: CEASE tag and TF-IDF side matrices for G3")
    ap.add_argument("--datasets", nargs="+", default=list(SIDE_DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024],
                    help="recorded only: the construction is deterministic (no randomness is used)")
    ap.add_argument("--smoke", action="store_true", help="write into <out>/smoke/ instead of <out>/")
    ap.add_argument("--out", default=None, help="default: <root>/results/scope/rev/closedform_side")
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else ROOT / "results" / "scope" / "rev" / "closedform_side"
    if a.smoke:
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[X4] root={ROOT} out={out_dir} seeds={a.seeds} (unused: deterministic)", flush=True)
    for ds in a.datasets:
        if ds not in SIDE_DATASETS:
            print(f"[{ds}] no item metadata for this dataset (Elec): G3 uses dense frozen features only",
                  flush=True)
            continue
        process(ds, out_dir, a.smoke)


if __name__ == "__main__":
    main()

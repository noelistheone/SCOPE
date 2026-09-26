# Datasets

Datasets are **not** committed to this repository; they are downloaded and
regenerated locally. This directory is otherwise git-ignored.

## Download (Baby / Sports / Elec)

```bash
pip install gdown
python scripts/download_data.py --dataset all
python scripts/verify_data.py
```

These use the standard MMRec-preprocessed splits and frozen multimodal features.

## Expected layout

Each dataset lives under `data/<name>/`:

```
<name>.inter        TSV with columns: userID, itemID, x_label
                    x_label: 0 = train, 1 = valid, 2 = test  (MMRec convention;
                    user/item IDs are already 0-indexed and contiguous)
image_feat.npy      float32 [n_items, D_v]   frozen visual features
text_feat.npy       float32 [n_items, D_t]   frozen text (Sentence-BERT) features
```

## Other datasets

- **Amazon-Clothing** is not in the public MMRec Drive folder; it can be built from
  the raw Amazon Reviews data (`scripts/download_data.py` prints a pointer).
- **MicroLens** (short-video platform; 1024-d image and text vectors per item)
  is available from its official public release, https://github.com/westlake-repl/MicroLens
  (the MMRec-preprocessed version, `microlens.inter` + `image_feat.npy` +
  `text_feat.npy`, is linked from the MMRec data page). Arrange either dataset
  into the layout above before use.
- The tag and TF-IDF variants of the closed-form content models
  (`scope/side_matrices.py`, consumed by `scope/closedform_family.py`) and the
  item-level content probe (`scope/cfblind_probe.py`) also need the item-metadata
  file `data/<name>/meta-<name>.csv`: columns `itemID`, `asin`, `title`, `brand`,
  `categories`, `description`, one row per item in itemID order (0 .. n_items-1).
  When `data/<name>/i_id_mapping.csv` is present (tab-separated, columns `itemID`,
  `asin`) the asin order is checked against it. For the Amazon datasets the file
  comes with the MMRec release.
- MicroLens: the MMRec release has no item metadata. Download
  `MicroLens-100k_title_en.csv`, `tags_to_summary.csv` and `MicroLens-100k_pairs.csv`
  from the official MicroLens-100k release (https://recsys.westlake.edu.cn/, folder
  `MicroLens-100k-Dataset`) into `data/microlens/official/` and run
  `python scope/build_microlens_meta.py`. It checks that every benchmark interaction
  maps back onto the official interaction file through `u_id_mapping.csv` /
  `i_id_mapping.csv`, then writes `meta-microlens.csv` with each video's English
  title and category label (the other columns empty). User comments and like/view
  counts are not used.
- Any dataset matching this layout can be added by creating a new
  `configs/dataset/<name>.yaml` (copy an existing one and adjust `data_path`,
  the feature filenames, and the field names).

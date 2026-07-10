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
- **MicroLens** is available from its official public release. Arrange either into
  the layout above before use.
- Any dataset matching this layout can be added by creating a new
  `configs/dataset/<name>.yaml` (copy an existing one and adjust `data_path`,
  the feature filenames, and the field names).

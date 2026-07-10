# SCOPE: Recommendation as Masked Set Completion

This repository contains the reference implementation and experiment code for
**SCOPE**, a family of multimodal recommenders built on a *masked set completion*
view of recommendation.

> We recast recommendation as **masked set completion**: treat a user's history as a
> *set* of items and predict a held-out part of it from the rest (like completing a
> partly observed shopping basket), with no per-user parameters. On top of this view
> we build **SCOPE**, which fuses a lightweight set-completion head with a
> training-free base (item co-occurrence + text similarity). Because it has no
> per-user parameters, SCOPE is **inductive** — it scores unseen users and items
> without retraining.

The repository is fully self-contained: it includes (1) the SCOPE method
(`SCOPE-v1`, `SCOPE-G`, `SCOPE-U`), (2) a reproducible baseline framework covering
the baselines reported in the paper, and (3) the analysis scripts behind
every experimental claim. It does **not** ship datasets or checkpoints; both are
regenerated from the code and a one-line download.

---

## Repository layout

```
configs/            Layered YAML configs (overall <- dataset <- model <- CLI)
  dataset/          Per-dataset paths & fields (baby/sports/clothing/elec/microlens)
  model/            Per-model hyperparameters (baselines + scope)
src/                Baseline training framework
  common/           Abstract recommender, trainer, losses
  data/             Dataset, dataloaders, graph utilities
  models/           Baseline implementations (see "Baselines" below)
  evaluation/       Full-sort Recall / NDCG / Precision evaluators
  utils/            Config loader, seeding, logging, resource guards
  main.py           Baseline training entry point
scope/              The SCOPE method and all analysis experiments
  scope.py          SCOPE-v1: closed-form base + set-completion head
  scope_g.py        SCOPE-G: graph-propagation set-completion head
  ensemble_control.py, scope_u_ablate_ease.py
                    SCOPE-U: gated fusion with a strong collaborative view
  gpu_eval.py       GPU-resident full-sort evaluator
  harness.py        Paired user-level bootstrap significance test
  <analysis>.py     Cold-start, leakage audit, stratification, ... (see below)
scripts/            Data download & environment/pipeline verification
data/               Datasets (downloaded, git-ignored)
```

## Installation

```bash
# Python 3.10; a CUDA-capable GPU is recommended for training.
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt
python scripts/verify_env.py                           # imports + CUDA check
```

## Data

The Amazon datasets use the standard MMRec-preprocessed splits and frozen
multimodal features (a visual and a Sentence-BERT text vector per item):

```bash
pip install gdown
python scripts/download_data.py --dataset all          # Baby / Sports / Elec
python scripts/verify_data.py
```

Each dataset lives under `data/<name>/` with:

```
<name>.inter        TSV: userID, itemID, x_label (0=train, 1=valid, 2=test)
image_feat.npy      float32 [n_items, D_v]   frozen visual features
text_feat.npy       float32 [n_items, D_t]   frozen text features
```

Amazon-Clothing and MicroLens are not in the public MMRec folder. Amazon-Clothing
can be constructed from the raw Amazon Reviews data (`scripts/download_data.py`
prints a pointer); MicroLens is available from its official public release. Once
either is arranged into the layout above, add it via a new
`configs/dataset/<name>.yaml`. Any dataset matching this layout works.

## Reproducing the main results (Table 1)

**Baselines** train through the unified framework:

```bash
python -m src.main --model freedom --dataset baby --gpu 0
python -m src.main --model gume    --dataset baby --gpu 0
# ... any key in src/models/__init__.py: lightgcn, vbpr, mmgcn, lattice, grcn,
#     bm3, mgcn, mentor, lgmrec, diffmm, smore, dragon, damrs, cohesion,
#     llmrec, rlmrec, mllmrec
```

**SCOPE.** The training-free base and set-completion head:

```bash
python scope/scope.py    --dataset baby      # SCOPE-v1 (closed-form base + set head)
python scope/scope_g.py  --dataset baby      # SCOPE-G  (graph-propagation head)
python scope/ease_baseline.py                # EASE / EASE+text base (Baby/Sports/Clothing)
python scope/admmslim_baseline.py            # ADMM-SLIM base (Baby/Sports/Clothing)
```

**SCOPE-U** fuses SCOPE with a strong collaborative view. It consumes cached
baseline score matrices, so first dump them, then run the gated fusion:

```bash
python scope/dump_baseline_scores.py --model freedom --dataset baby
python scope/dump_baseline_scores.py --model gume    --dataset baby
python scope/ensemble_control.py --datasets baby   # all view combinations under one protocol
```

Every SCOPE script tunes on validation and reports a single trusted test
evaluation; results are written as JSON under `results/`.

## Reproducing the analyses

| Claim in the paper                                    | Script                              |
|-------------------------------------------------------|-------------------------------------|
| Significance vs. strong CF + attribution ladder       | `scope/significance_ladder.py`      |
| Item cold-start (genuinely unseen items)              | `scope/coldstart_item.py`           |
| Inductive scoring of unseen users/items               | `scope/inductive_coldstart.py`, `scope/inductive_baselines.py` |
| Leakage screen (train/test-alignment)                 | `scope/leakage_protocol.py`         |
| User-activity stratification                          | `scope/stratify_setsize.py`         |
| Embedding-dimension robustness                        | `scope/scope.py --d {32,64,128,256,512}` |
| View-ablation / ensemble controls                     | `scope/ensemble_control.py`, `scope/scope_u_ablate_ease.py` |
| Task-formulation flip (co-purchase AUC vs. ranking)   | `scope/copurchase_auc.py`, `scope/content_ranking_null.py`, `scope/cfblind_probe.py` |
| Masking-ratio ablation                                | `scope/masking_ratio.py`            |
| Set-completion task diagnostics                       | `scope/setcompletion_task.py`       |
| Kernel complementarity                                | `scope/kernel_complementarity.py`   |
| Efficiency (wall-clock / memory)                      | `scope/efficiency.py`               |
| Dataset statistics                                    | `scope/dataset_stats.py`            |

Statistical significance uses a paired, user-level bootstrap
(`scope/harness.py`).

## Design principles

- **Reproducibility.** Fixed seeds; `cudnn.deterministic=True`. Tune on the
  validation split, evaluate once on test with the trusted full-sort evaluator.
- **GPU-first.** SCOPE keeps scoring on the GPU and avoids per-batch host
  transfers; large datasets fall back to chunked / fp16 score matrices.
- **Layered configs.** `overall.yaml` <- `dataset/<ds>.yaml` <- `model/<m>.yaml`
  <- CLI overrides.

## Verifying the framework

```bash
python scripts/verify_pipeline.py    # CPU smoke test of every registered model
```

## License

This code is released under the MIT License (see `LICENSE`). Baseline models are
faithful re-implementations / ports of prior work and retain the licenses and
attribution of their original authors, cited in the header of each file in
`src/models/`.

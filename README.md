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

Note: `torch_scatter` and `torch_geometric` wheels must match your torch/CUDA
build — if `pip` cannot resolve them, install from the matching wheel index,
e.g. `pip install torch_scatter torch_geometric -f https://data.pyg.org/whl/torch-2.4.0+cu121.html`.

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

Each supplement table/figure maps to its generating script (run order: train
`scope/scope.py` and `scope/scope_g.py` first — both save their checkpoints under
`ckpts/scope/` — and dump any needed baseline score matrices with
`scope/dump_baseline_scores.py`):

| Paper item                                            | Script                              |
|-------------------------------------------------------|-------------------------------------|
| Table 1 / S14 (main results, multi-seed)              | `src/main.py` (baselines), `scope/scope.py`, `scope/scope_g.py`, `scope/ensemble_control.py`; Elec: `scope/scope_elec.py` |
| Table S1 (dataset statistics)                         | `scope/dataset_stats.py`            |
| Table S2 + Fig. S1 (efficiency, Pareto)               | `scope/efficiency.py`, `scope/pareto_figure.py` |
| Table S3 (user-activity stratification)               | `scope/stratify_setsize.py`         |
| Table S4 (leakage screen, train/test-alignment)       | `scope/leakage_protocol.py`         |
| Table S5 (component attribution, null towers)         | `scope/kernel_complementarity.py`, `scope/content_ranking_null.py` |
| Table S6 (build-up ladder)                            | `scope/significance_ladder.py`      |
| Table S7 (composition: SCOPE-v2/U)                    | `scope/ensemble_control.py`, `scope/scope_u_ablate_ease.py` |
| Table S8 (backbone portability, 4 backbones)          | `scope/backbone_transfer.py`        |
| Table S9 (masked/AE neighbours: BERT4Rec, SASRec, Mult-VAE, item2vec) | `scope/masked_neighbors.py` |
| Table S10 (inductive new users + EASE-inductive)      | `scope/inductive_baselines.py`, `scope/coldstart_fewshot.py` |
| Table S11 (item cold-start, unseen items)             | `scope/coldstart_item.py`, `scope/inductive_coldstart.py` |
| Table S12 (paired bootstrap significance)             | `scope/significance_ladder.py`      |
| Table S13 (embedding-dimension sweep)                 | `scope/scope.py --d {32,64,128,256,512}` |
| Fig. S2 (fusion-weight sweep)                         | `scope/gamma_sweep.py`              |
| Fig. S3 + S4 (coverage, per-user breadth)             | `scope/coverage_breadth.py`         |
| SCOPE-G tail-popularity analysis                      | `scope/scope_g_tail.py`             |
| Masking-ratio ablation                                | `scope/masking_ratio.py`            |
| Set-completion task diagnostics                       | `scope/setcompletion_task.py`       |

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

# SCOPE: Set-Completion Prediction over Closed-Form Item Kernels

Reference implementation and experiment code for the paper *SCOPE: Set-Completion
Prediction over Closed-Form Item Kernels for Multimodal Recommendation*.

SCOPE casts recommendation as **masked set completion**: part of a user's item set
is hidden and a model learns to recover it from the rest by scoring the whole
catalog at once. The set-completion **head** has no per-user parameters (a user is
a function of the observed items) and reads the set as a whole rather than as a sum
of item pairs. The head is fused with a closed-form item--item **base** from the
EASE line (a one-hop EASE solve plus a text-kNN affinity term, prior art), and the
paper measures what the head adds over that base, over the strongest published
closed-form content kernel, and inside compositions with external collaborative
recommenders (FREEDOM, LGMRec, MGCN, GUME); inside SCOPE-U the set view's further
increment is equivalent to zero at the scale of its gain over the base.

The repository contains (1) the SCOPE method (`SCOPE-v1`, `SCOPE-G`, `SCOPE-v2`,
`SCOPE-U`), (2) a reproducible baseline framework covering the eighteen learned
baselines of the paper (EASE and ADMM-SLIM are closed-form scripts under `scope/`),
(3) the closed-form content family (CEASE, Add-EASE, FEASE, L3AE) on one shared
validation grid, and (4) the analysis scripts behind the tables of the paper and the
sections of the appendix (the few appendix files that are logs or hand-aggregated
summaries are marked as such in its experiment index). It ships no datasets or
checkpoints: Baby, Sports and Electronics download with one command, Clothing and
MicroLens are arranged by hand (`data/README.md`), and everything else is
regenerated from the code.

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
                    SCOPE-v2 / SCOPE-U: gated composition with FREEDOM / GUME
  closedform_family.py, side_matrices.py
                    Closed-form content kernels on one grid; head on the strongest kernel
  neighbors_matched.py, neighbors_precompute_base.py, neighbors_tables.py
                    Matched comparison of the head with its learned relatives
  gpu_eval.py       GPU-resident full-sort evaluator
  harness.py        Paired user-level bootstrap significance test
  <analysis>.py     Attribution, provenance, coverage, robustness, ... (see below)
scripts/            Data download & environment/pipeline verification
data/               Datasets (downloaded, git-ignored; see data/README.md)
```

## Installation

```bash
# Python 3.10; a CUDA-capable GPU is recommended for training.
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt
python scripts/verify_env.py                           # imports + CUDA check
```

Note: `torch_scatter` and `torch_geometric` wheels must match your torch/CUDA
build; if `pip` cannot resolve them, install from the matching wheel index,
e.g. `pip install torch_scatter torch_geometric -f https://data.pyg.org/whl/torch-2.4.0+cu121.html`.

## Data

```bash
python scripts/download_data.py --dataset all     # MMRec-preprocessed Amazon splits + frozen features
python scripts/verify_data.py
```

`data/README.md` describes the expected layout, the sources of Amazon-Clothing and
MicroLens, and the item-metadata file needed by the tag / TF-IDF closed forms.

## Reproducing the main table (paper Table 3)

**Learned baselines** train through the unified framework, one run per (model,
dataset) in the released configuration (validation-selected exceptions are listed
under Table 3 of the paper):

```bash
python -m src.main --model freedom --dataset baby --gpu 0
python -m src.main --model gume    --dataset baby --gpu 0
# ... any key in src/models/__init__.py: lightgcn, vbpr, mmgcn, lattice, grcn,
#     bm3, mgcn, mentor, lgmrec, diffmm, smore, dragon, damrs, cohesion,
#     llmrec, rlmrec
# GUME's released configuration differs per dataset; configs/model/gume.yaml holds the Baby
# values, the others are passed as overrides (the same values must be passed when dumping scores):
python -m src.main --model gume --dataset sports   --gpu 0 --override n_layers=1 bm_temp=0.2 um_loss=0.01 um_temp=0.1 vt_loss=0.01
python -m src.main --model gume --dataset clothing --gpu 0 --override n_layers=1 bm_temp=0.2 um_loss=0.1 um_temp=0.2 vt_loss=0.001
python -m src.main --model gume --dataset microlens --gpu 0                    # initial run of the nine-setting grid
python -m src.main --model gume --dataset microlens --gpu 0 --override learning_rate=0.0005 n_layers=1   # validation-selected row
python scripts/run_all_baselines.py --help     # batch runner; writes baseline_<ds>_seed2024.csv
```

DiffMM, LLMRec and RLMRec are simplified re-implementations (no diffusion stage,
no LLM-generated augmentations), as disclosed in the notes of Table 3.
`src/models/damrs.py` builds DA-MRS's user--item graph by coordinate-list
assignment: the released code fills a SciPy `dok_matrix` through a private method
that SciPy 1.13 removed, and the usual replacement, `dict.update` on the matrix,
silently leaves the graph empty on current SciPy. On Baby, Sports
and Clothing DA-MRS is selected on validation over four settings,
`--override knn_k=10 kl_weight=<0.1|1.0> neighbor_weight=<0.001|0.01>`; MicroLens
uses the released setting (kl_weight 1.0, neighbor_weight 0.001).

**Linear models and the closed-form content family:**

```bash
python scope/ease_baseline.py                 # EASE / EASE+text base (Baby/Sports/Clothing)
python scope/admmslim_baseline.py             # ADMM-SLIM
python scope/microlens_linear.py              # EASE / ADMM-SLIM on MicroLens (98K users, chunked)
python scope/build_microlens_meta.py          # MicroLens item metadata from the official release (data/README.md)
python scope/side_matrices.py --datasets baby sports clothing microlens     # tag / TF-IDF side matrices
python scope/closedform_family.py --stage all --datasets baby sports clothing microlens
#   stage g3: CEASE-emb/-tags, Add-EASE-emb/-tags, FEASE (prior only) tfidf/emb, FEASE-full-tfidf, L3AE,
#   plus the base re-tuned on the same grid; validation selection, one trusted test evaluation.
#   stage g4 (needs the three SCOPE-v1 head checkpoints): the head fused onto the strongest kernel,
#   with the per-seed paired tests (the last rows of Table 3 and part of Table 4)
```

**SCOPE.** The closed-form base and the set-completion head:

```bash
python scope/scope.py    --dataset baby      # SCOPE-v1 (base + set head), seeds 2024/2025/2026 via --seed
python scope/scope_g.py  --dataset baby      # SCOPE-G  (frozen co-occurrence + text graph propagation)
```

**SCOPE-v2 / SCOPE-U** compose the base and the head with the score matrix of a
separately trained recommender (FREEDOM for v2, GUME for U). Dump the scores of
the trained baseline first, then run the gated composition, which also produces
every set-free control (base+GUME, base+FREEDOM+GUME, ...):

```bash
python scope/dump_baseline_scores.py --model freedom --dataset baby
python scope/dump_baseline_scores.py --model gume    --dataset baby
python scope/dump_baseline_scores.py --model gume    --dataset sports --override n_layers=1 bm_temp=0.2 um_loss=0.01 um_temp=0.1 vt_loss=0.01
python scope/ensemble_control.py --datasets baby --seeds 2024     # all view combinations under one protocol
# MicroLens (98K users): the memory-lean equivalent, on the validation-selected GUME and seeds 2024-2026
python scope/dump_baseline_scores.py --model gume --dataset microlens --override learning_rate=0.0005 n_layers=1 --ckpt <that run's checkpoint>
python scope/dump_baseline_scores.py --model freedom --dataset microlens
python scope/ensemble_control_lean.py --datasets microlens --seeds 2024 2025 2026 --with-ease --tag gume_lr0.0005_nl1 \
    --gume-scores results/baseline_scores/gume_microlens_scores.npy
```

Every SCOPE script selects on validation Recall@20 and reports one trusted test
evaluation; results are written as JSON under `results/scope/`.

## Reproducing the attribution (paper Tables 4-7 and Sections 4.3-4.5)

| Paper item | Script |
|---|---|
| Table 4, set view over the base; effect sizes, Holm | `scope/canonical_significance.py` (paired user-level bootstrap in `scope/harness.py`) |
| Table 4, head over the strongest kernel (all seeds, both metrics) | `scope/closedform_family.py --stage g4` |
| Section 4.3, untrained-view controls, placebo view, fusion-weight optimum | `scope/batch1_synergy.py`, `scope/batch4_attribution.py`, run with `SCOPE_HEAD=full` on the checkpoint of the pre-pruning model `scope/scope_full.py` (the "earlier head checkpoint" of the paper; `batch4` needs `scope/scope_u_ablate_ease.py` first) |
| Section 4.3, seeding / objective controls (co-occurrence or random seed, BPR) | `scope/scope_seed_obj_control.py` |
| Table 5, inside SCOPE-U (decomposition, per-user changes) | `scope/ensemble_control.py` (MicroLens: `scope/ensemble_control_lean.py`), `scope/rq3_ceiling_breadth.py` |
| Table 5, power and equivalence (MDE, TOST) | `scope/w16_power.py` |
| Table 6, the set view over four collaborative backbones | `scope/backbone_transfer.py` |
| Table 7, matched comparison with item2vec/CBOW, BERT4Rec- and SASRec-style set Transformers, Mult-VAE | `scope/neighbors_precompute_base.py --datasets baby sports`, then `scope/neighbors_matched.py --stage all --datasets baby sports`, then `scope/neighbors_tables.py` |
| Section 4.4, sign rule on the released item features | `scope/leakage_protocol.py` |
| Section 4.4, coverage and the union oracle | `scope/coverage_analysis.py --collab gume --datasets baby sports clothing`, `scope/rq3_ceiling_breadth.py` |
| Section 4.5, seeds, head width, masking, graph depth | `scope/scope.py --seed / --d`, `scope/masking_ratio.py baby sports clothing`, `scope/scope_g.py --K` (runs with K > 1, another seed or a single graph write suffixed files, so the deployed K=1 run is never overwritten) |
| Table 1 / Section 3, parameter counts, fit and scoring cost | `scope/efficiency_clean.py` |
| Table 2, dataset statistics | `scope/dataset_stats.py` (Amazon); the MicroLens counts are printed by the dataset loader of any framework run |

## Reproducing the appendix

The appendix (a separate document) is organised in sections A-J; each maps to
the scripts below. Result-file names in its experiment index are the file
names these scripts write.

| Appendix section | Script |
|---|---|
| A. Amazon-Electronics (63K items) | `scope/scope_elec.py` (trains the Electronics head; its own three-view composition is not reported), then `scope/scope_elec_v1.py` (SCOPE-v1), `scope/scope_elec_u.py` (four-view SCOPE-U) and `scope/scope_g_elec.py` (SCOPE-G); sparse co-occurrence view in `scope/cooc_knn.py`. As disclosed in the appendix, these scripts score the set view with L1-normalised logits. |
| B. New users with truncated context; items without interactions | `scope/coldstart_fewshot.py` (needs the SCOPE-G checkpoint of `scope/scope_g_tail.py`), `scope/inductive_baselines.py`, `scope/w12_coldstart_sig.py`, `scope/w12b_inductive_baseline.py`; `scope/coldstart_item.py` |
| C. Closed-form content kernels; head on the strongest kernel | `scope/build_microlens_meta.py` (MicroLens side information), `scope/side_matrices.py`, `scope/closedform_family.py` |
| D. Matched neighbour comparison in full | `scope/neighbors_matched.py`, `scope/neighbors_tables.py`; the earlier comparison of D13: `scope/w13_item2vec.py`, `scope/masked_neighbors.py` |
| E. Coverage, breadth, activity strata, popularity concentration | `scope/coverage_analysis.py --collab gume --datasets baby sports clothing`, `scope/rq3_ceiling_breadth.py`, `scope/stratify_setsize.py` |
| F1-F3. Head width, masking ratio, seed modality | `scope/scope.py --d`, `scope/masking_ratio.py baby sports clothing`, `scope/w17_seed_modality.py` |
| F4. Item-level content signal; frozen content views | `scope/copurchase_auc.py`, `scope/cfblind_probe.py`, `scope/welding.py` |
| F5. Further signals over base + GUME | `scope/w16_power.py`, `scope/w19_orthogonal_diag.py`, `scope/w20_userside_diag.py`, `scope/w21_core.py`, `scope/w22_seedcheck.py`, `scope/w23_spice_probe.py` |
| F6. Zero-shot transfer of the set operator | `scope/w15_transfer.py` |
| F7. Attribution ladder, null views, placebo, gamma sweep; pruning; seed / objective controls | `scope/batch1_synergy.py`, `scope/batch4_attribution.py` (its placebo control reads the output of `scope/scope_u_ablate_ease.py`), `scope/train_prune.py`, `scope/prune_2hop_realckpt.py`, `scope/train_fused.py`, `scope/batch3_aggregate.py`: these ran on the pre-pruning model, so train it first with `scope/scope_full.py` (it writes `scope_full_*` files) and run them with `SCOPE_HEAD=full`; `scope/scope_seed_obj_control.py`, `scope/welding.py` |
| F8. User-conditioned fusion | `scope/stack_route.py`, `scope/moe_fusion.py baby` |
| F9. Fusion normalisation | `scope/fusion_norm_ablation.py` |
| F10. SCOPE-G graph ablation | `scope/scope_g.py --graph {both,cooc,text}` |
| F11. Set-free fusion of more backbones | `scope/full_ensemble.py` |
| F12. Set completion on training sets | `scope/setcompletion_task.py` |
| F13. Pairwise reproduction of the head | `scope/w20_higher_order.py` |
| G. Published versus reproduced baselines; configurations | `src/main.py` runs (published values are transcribed from the cited papers); GUME on MicroLens: `python -m src.main --model gume --dataset microlens --override learning_rate=<lr> n_layers=<n>` over lr in {5e-4, 1e-3, 2e-3} and n in {1, 2, 3} |
| H. Feature provenance (sign rule on item-side views) | `scope/leakage_protocol.py`; earlier version `scope/leak_test.py` (`SCOPE_HEAD=full`) |
| I. Implementation details, footprint | `configs/`, `scope/efficiency_clean.py` |
| J. Experiment index | the result files named there are the JSON outputs of the scripts above |

MicroLens rows of the main table also use `scope/microlens_linear.py` (EASE and
ADMM-SLIM at 98K users), `scope/ensemble_control_lean.py` (SCOPE-U, seeds 2024-2026)
and LGMRec trained with the standard command (`--model lgmrec --dataset microlens`);
`scope/microlens_robust_seeds.py` is the earlier three-view seed check;
`scope/w23_scope_v2_supplement.py` adds the SCOPE-v2 MicroLens seeds and its
significance test against GUME.

## Notes on the shipped analysis scripts

- Run every script from the repository root (`python scope/<script>.py ...`);
  outputs go to `results/`, `ckpts/` and `logs/` under the root.
- The attribution controls of Section 4.3 / appendix F7 (`batch1_synergy.py`,
  `batch4_attribution.py`, `train_prune.py`, `prune_2hop_realckpt.py`,
  `train_fused.py`, `batch3_aggregate.py`, `leak_test.py`, and the
  `scope_u_ablate_ease.py` run that `batch4` reads) were produced with the
  pre-pruning model. Train it with `scope/scope_full.py` (it writes
  `scope_full_*` files) and run those scripts with the environment variable
  `SCOPE_HEAD=full`; without it they use the deployed `scope.py` model and print
  a note.
- Several earlier scripts score with `F.normalize(x, 1)`, i.e. L1-normalised
  logits instead of the cosine of the paper's Eq. 3, and are shipped as they were
  run because the appendix reports their numbers with that disclosure:
  `scope_elec_v1.py`, `scope_elec_u.py`, `coldstart_fewshot.py`,
  `w12_coldstart_sig.py`, `w12b_inductive_baseline.py`, `w13_item2vec.py`,
  `masked_neighbors.py`, `train_prune.py`. The matched neighbour comparison of
  Table 7 (`neighbors_matched.py`) and every SCOPE row of the paper use the
  cosine head.

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
re-implementations or ports of prior work (the simplifications are listed above)
and retain the licenses and attribution of their original authors, cited in the header of each file in
`src/models/`. The closed-form content kernels are implemented from the equations
of the cited papers; no third-party code is copied.

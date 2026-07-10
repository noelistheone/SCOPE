#!/usr/bin/env bash
# Convenience wrapper: train one (model, dataset) combo.
#
# Usage:
#   bash scripts/run_baseline.sh freedom baby
#   bash scripts/run_baseline.sh lightgcn baby 0       # gpu_id=0
set -euo pipefail

MODEL="${1:?usage: run_baseline.sh <model> <dataset> [gpu_id]}"
DATASET="${2:?usage: run_baseline.sh <model> <dataset> [gpu_id]}"
GPU="${3:-0}"

cd "$(dirname "$0")/.."

# Cap CPU threads early.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python -m src.main --model "$MODEL" --dataset "$DATASET" --gpu "$GPU"

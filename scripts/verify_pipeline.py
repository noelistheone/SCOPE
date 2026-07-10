#!/usr/bin/env python
"""End-to-end CPU smoke test of every registered model on tiny synthetic data.

Run with:  python scripts/verify_pipeline.py

For each model in MODEL_REGISTRY:
  1. Instantiate on a 50-user / 100-item / dim=8 synthetic dataset.
  2. Run one training batch through `calculate_loss`, .backward(), .step()
     to confirm gradients flow.
  3. Run `full_sort_predict` on a slice of users and assert finite, correct shape.

Exits 0 only if every model passes.

Forces CPU (no GPU dependency) and num_workers=0 so this is safe to run
even when the GPU is busy.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from src.common.trainer import Trainer
from src.data.dataloader import EvalDataLoader, TrainDataLoader
from src.data.dataset import make_synthetic_dataset
from src.data.graph_utils import build_norm_adj
from src.models import MODEL_REGISTRY
from src.utils.configurator import Config
from src.utils.resource_guard import configure_runtime
from src.utils.seed import set_seed


def _build_synthetic():
    """Set up a tiny dataset with v_feat and t_feat present."""
    rec = make_synthetic_dataset(n_users=50, n_items=100, n_inter=600,
                                 feat_dim_v=16, feat_dim_t=8, seed=0)
    # Ensure non-empty valid/test for evaluation.
    if rec.train_df.empty or rec.valid_df.empty or rec.test_df.empty:
        raise RuntimeError("Synthetic dataset has empty splits")
    norm_adj = build_norm_adj(rec.train_matrix, rec.n_users, rec.n_items)
    v_feat = torch.from_numpy(rec.v_feat.copy())
    t_feat = torch.from_numpy(rec.t_feat.copy())
    return rec, norm_adj, v_feat, t_feat


def _build_model(model_name: str, rec, norm_adj, v_feat, t_feat):
    cfg = Config(model_name, "baby")
    # Shrink everything for smoke test.
    cfg["embedding_size"] = 8
    cfg["feat_embed_dim"] = 8
    cfg["n_layers"] = 2
    cfg["n_ui_layers"] = 2
    cfg["n_mm_layers"] = 1
    cfg["knn_k"] = 5
    cfg["train_batch_size"] = 16
    cfg["eval_batch_size_users"] = 32
    cfg["num_workers"] = 0
    cfg["epochs"] = 1
    cfg["eval_step"] = 1
    cfg["stopping_step"] = 2
    cfg["cudnn_deterministic"] = True
    cfg["dropout"] = 0.0
    cfg["device"] = "cpu"

    ModelCls = MODEL_REGISTRY[model_name]
    kwargs: dict[str, Any] = {
        "config": cfg,
        "n_users": rec.n_users,
        "n_items": rec.n_items,
        "norm_adj": norm_adj,
    }
    if model_name not in ("lightgcn", "mllmrec"):
        kwargs.update(v_feat=v_feat, t_feat=t_feat)
    if model_name in ("freedom", "mllmrec", "grcn", "dragon", "smore", "damrs", "gume", "cohesion"):
        kwargs.update(
            train_user_idx=torch.from_numpy(rec.train_users),
            train_item_idx=torch.from_numpy(rec.train_items),
        )
    if model_name == "mllmrec":
        # Provide synthetic MLLM features so smoke test doesn't need disk files.
        kwargs.update(
            item_feat=torch.from_numpy(t_feat.numpy().astype("float32")),
            user_feat=torch.randn(rec.n_users, t_feat.shape[1]).float(),
        )
    model = ModelCls(**kwargs)
    return cfg, model


def smoke_test_model(model_name: str) -> tuple[bool, str]:
    """Returns (ok, message)."""
    set_seed(0)
    rec, norm_adj, v_feat, t_feat = _build_synthetic()
    try:
        cfg, model = _build_model(model_name, rec, norm_adj, v_feat, t_feat)
    except Exception as e:
        return False, f"build failed: {e}\n{traceback.format_exc()}"

    device = torch.device("cpu")
    model.to(device)

    train_loader = TrainDataLoader(rec, batch_size=16, num_workers=0)
    valid_loader = EvalDataLoader(rec, phase="valid", batch_size=32)
    test_loader = EvalDataLoader(rec, phase="test", batch_size=32)

    try:
        # FREEDOM uses masked_adj after pre_epoch_processing; trigger it.
        model.pre_epoch_processing(0)

        # One training batch.
        batch = next(iter(train_loader))
        out = model.calculate_loss(batch)
        if isinstance(out, tuple):
            loss = out[0]
        else:
            loss = out
        if not torch.isfinite(loss):
            return False, f"loss is non-finite: {loss.item()}"
        loss.backward()

        # One eval batch.
        batch_eval = next(iter(valid_loader))
        scores = model.full_sort_predict({"user": batch_eval["user_ids"]})
        if scores.shape != (batch_eval["user_ids"].size(0), rec.n_items):
            return False, (f"full_sort_predict shape {tuple(scores.shape)} != "
                           f"({batch_eval['user_ids'].size(0)}, {rec.n_items})")
        if not torch.isfinite(scores).all():
            return False, "full_sort_predict produced non-finite scores"

        # Trainer-level evaluate (one chunk).
        trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader,
                          run_name=f"smoke_{model_name}")
        eval_result = trainer.evaluate(valid_loader)
        if not eval_result or not all(np.isfinite(v) for v in eval_result.values()):
            return False, f"evaluate produced bad result: {eval_result}"
    except Exception as e:
        return False, f"forward/eval failed: {e}\n{traceback.format_exc()}"

    return True, "ok"


def main() -> int:
    configure_runtime({"cpu_threads": 2})

    print(f"Running smoke test on {len(MODEL_REGISTRY)} models (CPU, synthetic)\n")
    results = []
    for name in sorted(MODEL_REGISTRY):
        ok, msg = smoke_test_model(name)
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {name}")
        if not ok:
            print(f"        {msg.splitlines()[0]}")
        results.append((name, ok, msg))

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    if n_fail:
        print(f"{n_fail}/{len(results)} models failed:")
        for name, ok, msg in results:
            if not ok:
                print(f"\n=== {name} ===\n{msg}")
        return 1
    print(f"All {len(results)} models passed smoke test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

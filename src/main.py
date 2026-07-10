"""CLI entry point.

Usage:
    python -m src.main --model freedom --dataset baby
    python -m src.main --model lightgcn --dataset baby --gpu 0 \
        --override learning_rate=5e-4 train_batch_size=4096
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from src.common.trainer import Trainer
from src.data.dataloader import EvalDataLoader, TrainDataLoader
from src.data.dataset import RecDataset
from src.data.graph_utils import build_norm_adj
from src.models import get_model
from src.utils import (
    Config,
    build_logger,
    check_gpu_available,
    configure_runtime,
    ensure_dir,
    get_local_time,
    set_seed,
)

LOGGER_NAME = "recsys.main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recsys baseline runner")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id (default: from config)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Additional config overrides as key=value pairs (e.g. lr=5e-4)")
    return parser.parse_args()


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``--override key=value key2=value2`` into a typed dict."""
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--override item must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.lower() == "null":
            out[k] = None
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _resolve_device(config: Config, cpu: bool, gpu_override: int | None) -> torch.device:
    if cpu:
        return torch.device("cpu")
    gpu_id = gpu_override if gpu_override is not None else int(config.get("gpu_id", 0))
    if config.get("device") == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")


def main() -> int:
    args = parse_args()
    overrides = _parse_overrides(args.override)
    if args.seed is not None:
        overrides["seed"] = args.seed

    cfg = Config(args.model, args.dataset, cli_overrides=overrides)

    # CPU caps must happen before heavy imports / first parallel op.
    configure_runtime(cfg)

    # Logging.
    run_name = f"{cfg['model']}_{cfg['dataset']}_{get_local_time()}"
    log_dir = ensure_dir(Path(cfg.get("log_dir", "logs")) / run_name)
    logger = build_logger(LOGGER_NAME, log_dir=log_dir)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Resolved config keys: {sorted(cfg.keys())}")
    cfg.dump_to(log_dir / "config.yaml")

    # Seed.
    set_seed(int(cfg.get("seed", 2024)),
             deterministic=bool(cfg.get("cudnn_deterministic", True)))

    # Device.
    device = _resolve_device(cfg, cpu=args.cpu, gpu_override=args.gpu)
    cfg["resolved_device"] = str(device)
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        ok, info = check_gpu_available(min_free_mb=4000,
                                       device_id=device.index or 0)
        logger.info(info)
        if not ok:
            logger.error("GPU has insufficient free memory; aborting. "
                         "Free up memory or pass --cpu to run on CPU.")
            return 2

    # Data.
    dataset = RecDataset(cfg)
    logger.info(repr(dataset))
    norm_adj = build_norm_adj(dataset.train_matrix, dataset.n_users, dataset.n_items)

    train_loader = TrainDataLoader(
        dataset,
        batch_size=int(cfg["train_batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        max_neg_tries=int(cfg.get("neg_sampling_max_tries", 100)),
    )
    valid_loader = EvalDataLoader(
        dataset, phase="valid",
        batch_size=int(cfg.get("eval_batch_size_users", 1024)),
    )
    test_loader = EvalDataLoader(
        dataset, phase="test",
        batch_size=int(cfg.get("eval_batch_size_users", 1024)),
    )

    # Model.
    ModelCls = get_model(args.model)
    v_feat = (torch.from_numpy(dataset.v_feat[:].copy())
              if dataset.v_feat is not None else None)
    t_feat = (torch.from_numpy(dataset.t_feat[:].copy())
              if dataset.t_feat is not None else None)

    model_kwargs: dict[str, Any] = {
        "config": cfg,
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "norm_adj": norm_adj,
    }
    model_name_l = args.model.lower()
    # Pure-CF or MLLM-feature models don't need v_feat/t_feat injection.
    if model_name_l not in ("lightgcn", "mllmrec"):
        model_kwargs.update(v_feat=v_feat, t_feat=t_feat)
    # Models that need raw train edge indices.
    if model_name_l in ("freedom", "mllmrec", "grcn", "dragon", "smore", "damrs", "gume", "cohesion"):
        model_kwargs.update(
            train_user_idx=torch.from_numpy(dataset.train_users),
            train_item_idx=torch.from_numpy(dataset.train_items),
        )

    model = ModelCls(**model_kwargs).to(device)
    logger.info(str(model))

    trainer = Trainer(cfg, model, train_loader, valid_loader, test_loader,
                      logger=logger, run_name=run_name)
    import time
    train_start = time.time()
    result = trainer.fit()
    train_minutes = (time.time() - train_start) / 60.0
    logger.info(f"Done. Best valid {trainer.valid_metric}={result['best_valid']:.4f} "
                f"at epoch {result['best_epoch']}. Train time: {train_minutes:.1f} min")

    # Persist final result as JSON for the orchestrator.
    import json
    summary = {
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "seed": int(cfg.get("seed", 2024)),
        "best_valid_metric": trainer.valid_metric,
        "best_valid_score": float(result["best_valid"]),
        "best_epoch": int(result["best_epoch"]),
        "train_time_min": float(train_minutes),
        "test_result": {k: float(v) for k, v in result["test_result"].items()},
        "best_valid_result": {k: float(v) for k, v in result.get("best_valid_result", {}).items()},
        "run_name": run_name,
    }
    with (log_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Result summary written to {log_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

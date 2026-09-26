"""Generic Trainer for recommender models.

Handles train/eval loop, early stopping, checkpointing, and chunked full-sort
evaluation. Models pluginned in via the AbstractRecommender interface.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from src.common.abstract_recommender import AbstractRecommender
from src.evaluation.topk_evaluator import TopKEvaluator
from src.utils.early_stopping import EarlyStopper
from src.utils.misc import dict_to_str, ensure_dir, get_local_time

LOGGER = logging.getLogger("scope.trainer")


def _build_optimizer(name: str, params, lr: float,
                     weight_decay: float) -> Optimizer:
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; choose adam/adamw/sgd")


def _build_scheduler(spec: Optional[Any],
                     optimizer: Optimizer) -> Optional[_LRScheduler]:
    if spec is None:
        return None
    # Expected: [step_size, gamma]
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        step_size, gamma = spec
        return StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))
    raise ValueError(f"Unsupported learning_rate_scheduler spec: {spec!r}")


class Trainer:
    """Training & evaluation driver.

    Parameters
    ----------
    config : Mapping
        Resolved config.
    model : AbstractRecommender
        The model. Already moved to device by caller.
    train_loader : DataLoader
        Yields dict batches with 'user', 'pos_item', 'neg_item'.
    valid_loader, test_loader :
        Eval loaders yielding ('user_ids', 'history_mask') chunks.
    logger :
        Python logger (optional; falls back to module logger).
    """

    def __init__(self,
                 config: Mapping[str, Any],
                 model: AbstractRecommender,
                 train_loader,
                 valid_loader,
                 test_loader,
                 logger: Optional[logging.Logger] = None,
                 run_name: Optional[str] = None) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.logger = logger or LOGGER

        self.device = next(model.parameters()).device
        self.epochs = int(config.get("epochs", 1000))
        self.eval_step = int(config.get("eval_step", 1))
        self.clip_grad_norm = float(config.get("clip_grad_norm", 0.0))
        self.use_amp = bool(config.get("use_amp", False)) and self.device.type == "cuda"
        self.valid_metric = str(config.get("valid_metric", "Recall@20"))
        self.show_progress = bool(config.get("show_progress", True))

        self.optimizer = _build_optimizer(
            str(config.get("learner", "adam")),
            self.model.parameters(),
            float(config.get("learning_rate", 1e-3)),
            float(config.get("weight_decay", 0.0)),
        )
        self.scheduler = _build_scheduler(
            config.get("learning_rate_scheduler"), self.optimizer)

        self.scaler = (torch.cuda.amp.GradScaler() if self.use_amp else None)

        self.evaluator = TopKEvaluator(
            metrics=list(config.get("metrics", ["Recall", "NDCG"])),
            topk=list(config.get("topk", [10, 20])),
        )

        self.stopper = EarlyStopper(
            patience=int(config.get("stopping_step", 20)), mode="max")

        self.run_name = run_name or (
            f"{config['model']}_{config['dataset']}_{get_local_time()}")
        self.ckpt_dir = ensure_dir(config.get("ckpt_dir", "ckpts"))
        self.best_ckpt: Optional[Path] = None
        self.best_valid_score: float = float("-inf")

    # =========================================================
    # Public API
    # =========================================================

    def fit(self) -> dict[str, Any]:
        """Run the full training loop and final test evaluation. Returns:
            {'best_valid': float, 'best_epoch': int, 'test_result': dict}
        """
        best_epoch = -1
        best_result: dict[str, float] = {}

        for epoch in range(self.epochs):
            self.model.pre_epoch_processing(epoch)

            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            t_train = time.time() - t0

            if self.scheduler is not None:
                self.scheduler.step()

            self.model.post_epoch_processing(epoch)

            log_msg = f"epoch {epoch:>4d} | train_loss {train_loss:.4f} | time {t_train:.1f}s"
            self.logger.info(log_msg)

            if (epoch + 1) % self.eval_step == 0:
                valid_result = self._valid_epoch()
                score = valid_result.get(self.valid_metric, float("nan"))
                self.logger.info(
                    f"epoch {epoch:>4d} | valid  | {dict_to_str(valid_result)}")

                improved = self.stopper.step(score)
                if improved:
                    best_epoch = epoch
                    best_result = dict(valid_result)
                    self._save_checkpoint(epoch)
                if self.stopper.should_stop:
                    self.logger.info(
                        f"Early stop at epoch {epoch} (best epoch {best_epoch}, "
                        f"best {self.valid_metric}={self.stopper.best:.4f})")
                    break

        # Load best and run test.
        if self.best_ckpt is not None and self.best_ckpt.is_file():
            self._load_checkpoint(self.best_ckpt)
        test_result = self.evaluate(self.test_loader)
        self.logger.info(f"FINAL test | {dict_to_str(test_result)}")
        self.best_valid_score = self.stopper.best

        return {
            "best_valid": self.stopper.best,
            "best_epoch": best_epoch,
            "best_valid_result": best_result,
            "test_result": test_result,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """Run full-sort top-K evaluation. Chunked over user batches."""
        self.model.eval()
        self.evaluator.reset()
        topk_max = max(self.evaluator.topk)

        for batch in loader:
            user_ids = batch["user_ids"].to(self.device, non_blocking=True)
            history_indices = batch["history_indices"].to(
                self.device, non_blocking=True)
            history_values = batch["history_values"].to(
                self.device, non_blocking=True)
            positive_items = batch["positive_items"]
            positive_lengths = batch["positive_lengths"]

            scores = self.model.full_sort_predict({"user": user_ids})
            # Mask history (training items): set their score to -inf.
            # history_indices is [B, max_hist] with -1 padding.
            if history_indices.numel() > 0:
                mask = history_values.bool()
                row_idx = (torch.arange(scores.size(0), device=self.device)
                           .unsqueeze(1).expand_as(history_indices))
                # Replace padded -1 with 0 to keep indexing legal; mask will zero its effect.
                safe_cols = torch.where(history_indices >= 0,
                                        history_indices,
                                        torch.zeros_like(history_indices))
                scores[row_idx[mask], safe_cols[mask]] = float("-inf")

            _, topk_idx = torch.topk(scores, k=topk_max, dim=-1)

            self.evaluator.collect(
                topk_idx.cpu(),
                positive_items,    # list[np.ndarray] (variable-length)
                positive_lengths,  # tensor [B]
            )

        result = self.evaluator.compute()
        self.model.train()
        return result

    # =========================================================
    # Internals
    # =========================================================

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        n_batches = 0
        total_loss = 0.0
        loss_components_acc: dict[str, float] = {}

        for batch in self.train_loader:
            batch_on_device = {
                k: (v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss_out = self.model.calculate_loss(batch_on_device)
                loss, components = self._unpack_loss(loss_out)
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_out = self.model.calculate_loss(batch_on_device)
                loss, components = self._unpack_loss(loss_out)
                loss.backward()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            total_loss += loss.detach().item()
            for k, v in components.items():
                loss_components_acc[k] = loss_components_acc.get(k, 0.0) + v
            n_batches += 1

        if n_batches == 0:
            return float("nan")

        if loss_components_acc:
            comp_str = ", ".join(
                f"{k}={v / n_batches:.4f}" for k, v in loss_components_acc.items())
            self.logger.debug(f"  components: {comp_str}")

        return total_loss / n_batches

    @staticmethod
    def _unpack_loss(loss_out) -> Tuple[torch.Tensor, dict[str, float]]:
        """Models may return either a scalar Tensor or (Tensor, dict)."""
        if isinstance(loss_out, tuple):
            if len(loss_out) != 2:
                raise ValueError(
                    "Model returned a tuple of length != 2 from calculate_loss")
            loss, comps = loss_out
            components = {k: float(v) for k, v in comps.items()}
            return loss, components
        return loss_out, {}

    def _valid_epoch(self) -> dict[str, float]:
        return self.evaluate(self.valid_loader)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.ckpt_dir / f"{self.run_name}.pt"
        state = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "config_snapshot": dict(self.config),
        }
        # Atomic write: save to temp then rename.
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        # Cleanup previous best
        if self.best_ckpt is not None and self.best_ckpt != path and self.best_ckpt.is_file():
            try:
                self.best_ckpt.unlink()
            except OSError:
                pass
        self.best_ckpt = path

    def _load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.logger.info(f"Loaded best checkpoint from {path}")

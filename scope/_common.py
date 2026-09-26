"""Shared helpers for scoring a trained baseline from its checkpoint.

These functions reuse the project's existing infrastructure to load a model
from a checkpoint and compute its top-K predictions on the test set.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import EvalDataLoader  # noqa: E402
from src.data.dataset import RecDataset  # noqa: E402
from src.data.graph_utils import build_norm_adj  # noqa: E402
from src.models import get_model  # noqa: E402
from src.utils.configurator import Config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def find_checkpoint(model: str, dataset: str, seed: int = 2024) -> Path:
    """Find the most recent ckpts/<model>_<dataset>_<ts>.pt file."""
    ckpt_dir = PROJECT_ROOT / "ckpts"
    candidates = list(ckpt_dir.glob(f"{model}_{dataset}_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for {model}/{dataset} in {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model_for_eval(model: str, dataset: str,
                        device: str = "cuda",
                        seed: int = 2024) -> tuple[Any, RecDataset, Config]:
    """Build the model and load its checkpoint."""
    cfg = Config(model, dataset, cli_overrides={"seed": seed})
    set_seed(int(cfg.get("seed", seed)))
    rec = RecDataset(cfg)
    norm_adj = build_norm_adj(rec.train_matrix, rec.n_users, rec.n_items)

    v_feat = (torch.from_numpy(rec.v_feat[:].copy())
              if rec.v_feat is not None else None)
    t_feat = (torch.from_numpy(rec.t_feat[:].copy())
              if rec.t_feat is not None else None)

    ModelCls = get_model(model)
    kwargs: dict[str, Any] = {
        "config": cfg, "n_users": rec.n_users, "n_items": rec.n_items,
        "norm_adj": norm_adj,
    }
    if model.lower() != "lightgcn":
        kwargs.update(v_feat=v_feat, t_feat=t_feat)
    if model.lower() in ("freedom", "grcn", "dragon", "smore", "damrs", "gume", "cohesion"):
        kwargs.update(
            train_user_idx=torch.from_numpy(rec.train_users),
            train_item_idx=torch.from_numpy(rec.train_items),
        )

    m = ModelCls(**kwargs).to(device)
    ckpt = find_checkpoint(model, dataset, seed)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    return m, rec, cfg


@torch.no_grad()
def compute_topk_predictions(model, rec: RecDataset,
                             phase: str = "test",
                             topk: int = 50,
                             device: str = "cuda",
                             batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Run full-sort inference and return (user_ids, top_k_items).

    user_ids : [n_eval_users]
    top_k_items : [n_eval_users, topk]
    """
    loader = EvalDataLoader(rec, phase=phase, batch_size=batch_size)
    all_users = []
    all_topk = []
    for batch in loader:
        users = batch["user_ids"].to(device)
        hist_idx = batch["history_indices"].to(device)
        hist_val = batch["history_values"].to(device)

        scores = model.full_sort_predict({"user": users})
        if hist_idx.numel() > 0:
            mask = hist_val.bool()
            row_idx = (torch.arange(scores.size(0), device=device)
                       .unsqueeze(1).expand_as(hist_idx))
            safe_cols = torch.where(hist_idx >= 0,
                                    hist_idx,
                                    torch.zeros_like(hist_idx))
            scores[row_idx[mask], safe_cols[mask]] = float("-inf")
        _, top = torch.topk(scores, k=topk, dim=-1)
        all_users.append(users.cpu().numpy())
        all_topk.append(top.cpu().numpy())
    return np.concatenate(all_users), np.concatenate(all_topk, axis=0)


def hits_per_user(top_items: np.ndarray,
                  ground_truth: dict[int, np.ndarray],
                  user_ids: np.ndarray,
                  k: int) -> np.ndarray:
    """For each row, count hits-in-top-k against the ground-truth set."""
    n = top_items.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        u = int(user_ids[i])
        gt = ground_truth.get(u, np.array([], dtype=np.int64))
        if len(gt) == 0:
            continue
        out[i] = np.isin(top_items[i, :k], gt).sum()
    return out


def recall_per_user(hits: np.ndarray,
                    user_ids: np.ndarray,
                    ground_truth: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros_like(hits, dtype=np.float64)
    for i in range(len(hits)):
        gt = ground_truth.get(int(user_ids[i]), np.array([], dtype=np.int64))
        if len(gt) > 0:
            out[i] = hits[i] / len(gt)
    return out

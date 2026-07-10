"""Streaming top-K evaluator.

``collect`` accumulates per-batch top-K recommendations against ground-truth
positives. ``compute`` produces a dict like ``{'Recall@20': 0.0992, ...}``.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch

from src.evaluation.metrics import METRIC_FUNCS


class TopKEvaluator:
    """Accumulator + final reducer for top-K metrics.

    Parameters
    ----------
    metrics : list of metric names (case-insensitive): Recall, NDCG, Precision, MAP.
    topk : list of K values to report.
    """

    def __init__(self,
                 metrics: Iterable[str],
                 topk: Iterable[int]) -> None:
        self.metrics = [m for m in metrics]
        self.topk = sorted({int(k) for k in topk})
        if not self.topk:
            raise ValueError("topk must be non-empty")
        for m in self.metrics:
            if m.lower() not in METRIC_FUNCS:
                raise ValueError(
                    f"Unknown metric {m!r}; available: {list(METRIC_FUNCS)}")
        self._hits_chunks: List[np.ndarray] = []   # each [B, max_k]
        self._n_rel_chunks: List[np.ndarray] = []  # each [B]
        self._max_k = max(self.topk)

    def reset(self) -> None:
        self._hits_chunks.clear()
        self._n_rel_chunks.clear()

    def collect(self,
                topk_idx: torch.Tensor,
                positive_items: Sequence[np.ndarray],
                positive_lengths: torch.Tensor) -> None:
        """Record one batch.

        topk_idx : LongTensor [B, K]  (K >= max(self.topk))
        positive_items : sequence of length B; each entry is an np.ndarray of items.
        positive_lengths : LongTensor [B]
        """
        if topk_idx.dim() != 2 or topk_idx.size(1) < self._max_k:
            raise ValueError(
                f"topk_idx must be [B, K>={self._max_k}], got {tuple(topk_idx.shape)}")
        if len(positive_items) != topk_idx.size(0):
            raise ValueError("positive_items length != batch size")

        idx_np = topk_idx[:, :self._max_k].cpu().numpy()    # [B, max_k]
        hits = np.zeros_like(idx_np, dtype=np.bool_)
        for r, pos in enumerate(positive_items):
            if len(pos) == 0:
                continue
            pos_set = np.asarray(pos, dtype=np.int64)
            # np.isin is efficient even for small pos_set.
            hits[r] = np.isin(idx_np[r], pos_set, assume_unique=False)

        self._hits_chunks.append(hits)
        self._n_rel_chunks.append(positive_lengths.cpu().numpy().astype(np.int64))

    def compute(self) -> dict[str, float]:
        if not self._hits_chunks:
            return {f"{m}@{k}": 0.0 for m in self.metrics for k in self.topk}
        hits = np.concatenate(self._hits_chunks, axis=0)
        n_rel = np.concatenate(self._n_rel_chunks, axis=0)
        out: dict[str, float] = {}
        for m in self.metrics:
            fn = METRIC_FUNCS[m.lower()]
            for k in self.topk:
                out[f"{m}@{k}"] = fn(hits, n_rel, k)
        return out

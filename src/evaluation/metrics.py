"""Vectorized top-K metric kernels.

All kernels accept a numpy boolean ``hits`` matrix of shape [B, K] (True iff
the rank-k recommendation is in the user's ground-truth set) plus a numpy
integer vector ``n_relevant`` of shape [B] giving the size of each user's
ground-truth set.
"""

from __future__ import annotations

import numpy as np


def _check_inputs(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> None:
    if hits.ndim != 2:
        raise ValueError(f"hits must be 2D (got shape {hits.shape})")
    if hits.dtype != np.bool_:
        raise ValueError(f"hits must be bool (got {hits.dtype})")
    if n_relevant.ndim != 1 or n_relevant.shape[0] != hits.shape[0]:
        raise ValueError(
            f"n_relevant shape {n_relevant.shape} incompatible with hits {hits.shape}")
    if k < 1 or k > hits.shape[1]:
        raise ValueError(f"k={k} out of bounds for hits with K={hits.shape[1]}")


def recall_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> float:
    """Mean over users of (hits in top-k) / n_relevant.

    Users with no relevant items are skipped.
    """
    _check_inputs(hits, n_relevant, k)
    mask = n_relevant > 0
    if not np.any(mask):
        return 0.0
    top_hits = hits[:, :k].sum(axis=1).astype(np.float64)
    denom = n_relevant.astype(np.float64).clip(min=1)
    recall = top_hits / denom
    return float(recall[mask].mean())


def precision_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> float:
    """Mean over users of (hits in top-k) / k."""
    _check_inputs(hits, n_relevant, k)
    mask = n_relevant > 0
    if not np.any(mask):
        return 0.0
    top_hits = hits[:, :k].sum(axis=1).astype(np.float64)
    return float((top_hits[mask] / k).mean())


def ndcg_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> float:
    """Mean over users of DCG@k / IDCG@k. Binary relevance."""
    _check_inputs(hits, n_relevant, k)
    mask = n_relevant > 0
    if not np.any(mask):
        return 0.0

    # gain = 1 / log2(rank+2)  with rank = 0..k-1
    log_disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits[:, :k].astype(np.float64) * log_disc[None, :]).sum(axis=1)

    # IDCG: top-min(n_rel, k) ones
    min_rel_k = np.minimum(n_relevant, k).astype(np.int64)
    # Precompute cumulative discount.
    cum_disc = np.concatenate([[0.0], np.cumsum(log_disc)])
    idcg = cum_disc[min_rel_k]                # shape [B]

    ndcg = np.zeros_like(dcg)
    np.divide(dcg, idcg, out=ndcg, where=idcg > 0)
    return float(ndcg[mask].mean())


def map_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> float:
    """Mean average precision @ k."""
    _check_inputs(hits, n_relevant, k)
    mask = n_relevant > 0
    if not np.any(mask):
        return 0.0
    top_hits = hits[:, :k].astype(np.float64)
    cum_hits = np.cumsum(top_hits, axis=1)
    ranks = np.arange(1, k + 1).astype(np.float64)
    precision_at_each = cum_hits / ranks[None, :]
    ap = (precision_at_each * top_hits).sum(axis=1)
    denom = np.minimum(n_relevant, k).astype(np.float64).clip(min=1)
    ap = ap / denom
    return float(ap[mask].mean())


METRIC_FUNCS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ndcg": ndcg_at_k,
    "map": map_at_k,
}

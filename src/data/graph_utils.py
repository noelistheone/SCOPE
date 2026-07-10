"""Graph utilities: bipartite user-item adjacency, item-item kNN graph.

All functions are memory-conservative:
- ``build_norm_adj`` builds a sparse COO directly, never densifies.
- ``build_knn_graph`` chunks the item-item similarity computation so we
  never materialize a full [n_items, n_items] matrix.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a coalesced ``torch.sparse_coo_tensor``."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def build_norm_adj(train_matrix: sp.csr_matrix,
                   n_users: int,
                   n_items: int) -> torch.Tensor:
    """Build the symmetric-normalized bipartite user-item adjacency.

    A = [[0,   R],
         [R^T, 0]]
    Returns ``D^{-1/2} A D^{-1/2}`` as a coalesced sparse COO tensor of shape
    [n_users + n_items, n_users + n_items].
    """
    if train_matrix.shape != (n_users, n_items):
        raise ValueError(
            f"train_matrix shape {train_matrix.shape} != ({n_users}, {n_items})")

    n = n_users + n_items
    R = train_matrix.tocoo()

    # Build A in COO form: edges in both directions.
    row = np.concatenate([R.row, R.col + n_users])
    col = np.concatenate([R.col + n_users, R.row])
    data = np.ones(row.shape[0], dtype=np.float32)

    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    # Degree (row sums of A).
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D = sp.diags(deg_inv_sqrt)

    norm = D @ A @ D
    return sparse_mx_to_torch_sparse_tensor(norm)


def build_knn_graph(feat: torch.Tensor,
                    k: int,
                    sym: bool = True,
                    chunk_size: int = 1024,
                    self_loop: bool = False) -> torch.Tensor:
    """Build a symmetric, normalized kNN graph from item features.

    For each item we keep its top-``k`` cosine-similarity neighbours. The
    resulting sparse adjacency is symmetrized (``max(A, A^T)``) when
    ``sym=True`` and Laplacian-normalized: ``D^{-1/2} A D^{-1/2}``.

    Returns a coalesced sparse COO tensor of shape [n_items, n_items].

    Computed in chunks of ``chunk_size`` rows to bound peak memory at
    O(chunk_size * n_items) instead of O(n_items^2).
    """
    if feat.dim() != 2:
        raise ValueError(f"feat must be 2D (got {feat.dim()}D)")
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    n_items = feat.shape[0]
    if n_items == 0:
        raise ValueError("feat has zero rows")
    k_eff = min(k, n_items - (0 if self_loop else 1))
    if k_eff < 1:
        raise ValueError(f"k_eff resolved to {k_eff}; need more rows or smaller k")

    feat = F.normalize(feat.float(), dim=-1)

    rows_list = []
    cols_list = []
    vals_list = []

    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        block = feat[start:end]                                # [B, D]
        sims = block @ feat.t()                                # [B, n_items]
        if not self_loop:
            # Suppress self-similarity for diagonal entries in this block.
            diag_rows = torch.arange(start, end, device=feat.device)
            sims[torch.arange(end - start, device=feat.device), diag_rows] = -float("inf")
        topv, topi = torch.topk(sims, k=k_eff, dim=-1)         # [B, k_eff]
        block_rows = (torch.arange(start, end, device=feat.device)
                      .unsqueeze(1).expand_as(topi))
        rows_list.append(block_rows.reshape(-1))
        cols_list.append(topi.reshape(-1))
        vals_list.append(topv.reshape(-1))

    rows = torch.cat(rows_list).cpu().numpy()
    cols = torch.cat(cols_list).cpu().numpy()
    vals = torch.cat(vals_list).cpu().numpy()

    # Drop any -inf rows that may have slipped through edge cases.
    finite = np.isfinite(vals)
    rows, cols, vals = rows[finite], cols[finite], vals[finite]

    # Use binary edges (MMRec convention): turn off similarity weighting.
    A = sp.coo_matrix(
        (np.ones_like(vals, dtype=np.float32), (rows, cols)),
        shape=(n_items, n_items),
    ).tocsr()

    if sym:
        A = A.maximum(A.T)

    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D = sp.diags(deg_inv_sqrt)
    norm = D @ A @ D
    return sparse_mx_to_torch_sparse_tensor(norm)


def sparse_row_topk(sp: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the top-``k`` values per row of a 2-D sparse COO tensor.

    Stays in sparse form (no full densification). Useful for fusing several
    sparse kNN graphs without blowing up to dense [n, n] memory.

    Returns a coalesced sparse COO tensor.
    """
    sp = sp.coalesce()
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    idx = sp.indices()
    val = sp.values()
    if val.numel() == 0:
        return sp

    rows = idx[0]
    cols = idx[1]

    # Two-pass stable sort: (1) by -value, then (2) by row. End state is grouped
    # by row, with descending-value order within each group.
    perm1 = torch.argsort(-val, stable=True)
    rows = rows[perm1]
    cols = cols[perm1]
    val = val[perm1]

    perm2 = torch.argsort(rows, stable=True)
    rows = rows[perm2]
    cols = cols[perm2]
    val = val[perm2]

    # Within-row rank via cumcount over groups of equal `rows`.
    _, first_occur = torch.unique_consecutive(rows, return_inverse=True)
    _, counts = torch.unique_consecutive(rows, return_counts=True)
    starts = torch.cat([
        torch.zeros(1, dtype=counts.dtype, device=counts.device),
        torch.cumsum(counts[:-1], dim=0),
    ])
    pos = torch.arange(rows.numel(), device=rows.device)
    rank_within = pos - starts[first_occur]
    keep_mask = rank_within < k

    return torch.sparse_coo_tensor(
        torch.stack([rows[keep_mask], cols[keep_mask]]),
        val[keep_mask],
        sp.shape,
    ).coalesce()


def split_norm_adj_blocks(norm_adj: torch.Tensor,
                          n_users: int,
                          n_items: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a full (n_users+n_items)^2 norm_adj into (R_norm, R_norm^T).

    Returns the (n_users, n_items) top-right block and its transpose, both
    as coalesced sparse COO tensors. Useful for models that propagate
    user-side and item-side messages with explicit two-step matmuls.
    """
    if not norm_adj.is_sparse:
        raise ValueError("norm_adj must be sparse")
    n_total = n_users + n_items
    if tuple(norm_adj.shape) != (n_total, n_total):
        raise ValueError(
            f"norm_adj shape {tuple(norm_adj.shape)} != "
            f"({n_total}, {n_total})")
    idx = norm_adj.indices()
    val = norm_adj.values()
    rows, cols = idx[0], idx[1]
    mask_top_right = (rows < n_users) & (cols >= n_users)
    r = rows[mask_top_right]
    c = cols[mask_top_right] - n_users
    v = val[mask_top_right]

    R_norm = torch.sparse_coo_tensor(
        torch.stack([r, c]), v, (n_users, n_items)).coalesce()
    R_normT = torch.sparse_coo_tensor(
        torch.stack([c, r]), v, (n_items, n_users)).coalesce()
    return R_norm, R_normT

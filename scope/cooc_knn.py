#!/usr/bin/env python
"""Scalable sparse item-kNN co-occurrence VIEW (a 'sharp' EASE-proxy) for large datasets like Elec,
where dense EASE (63k x 63k) is infeasible. Builds G = R^T R in item-chunks on GPU, keeps top-k
neighbors per item, returns a GPU sparse [N,N] tensor. Score for a user-chunk: R[chunk] @ Gknn.
Provides the complementary 'sharp item-item' signal that EASE gives on the dense datasets.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def build_cooc_knn(R_sparse, k=100, chunk=2048, device="cuda:0"):
    """R_sparse: [U,N] coalesced sparse. Returns sparse [N,N] item-item top-k co-occurrence (diag 0)."""
    N = R_sparse.shape[1]
    Rt = R_sparse.t().coalesce()                      # [N,U] sparse
    rows, cols, vals = [], [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        # dense rows of R^T for items [s:e]  -> [|c|, U]
        idx = Rt.indices(); v = Rt.values()
        m = (idx[0] >= s) & (idx[0] < e)
        sub = torch.sparse_coo_tensor(torch.stack([idx[0][m] - s, idx[1][m]]), v[m], (e - s, R_sparse.shape[0])).to(device)
        Cc = torch.sparse.mm(sub, R_sparse)           # [|c|, N] co-occurrence rows (dense)
        Cc = Cc.to_dense() if Cc.is_sparse else Cc
        # zero self
        ar = torch.arange(s, e, device=device)
        Cc[torch.arange(e - s, device=device), ar] = 0.0
        kk = min(k, N - 1)
        tv, ti = torch.topk(Cc, kk, dim=1)
        rr = ar.unsqueeze(1).expand(-1, kk).reshape(-1)
        rows.append(rr); cols.append(ti.reshape(-1)); vals.append(tv.reshape(-1))
        del Cc, sub; torch.cuda.empty_cache()
    ridx = torch.cat(rows); cidx = torch.cat(cols); vval = torch.cat(vals)
    G = torch.sparse_coo_tensor(torch.stack([ridx, cidx]), vval, (N, N)).coalesce()
    return G

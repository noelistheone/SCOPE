"""MLLMRec: Dang et al., AAAI 2025.

Two-stage approach:
  - Stage 1 (offline, not done here): an MLLM (Gemma3 in the paper, Qwen2-VL in
    our reproduction) generates a semantic profile for each item and each user
    from their interaction history. The MLLM outputs are pre-encoded into
    384-dim Sentence-Transformer embeddings and stored on disk as
    ``mllm_item_feat.npy`` and ``mllm_user_feat.npy`` in the dataset directory.
  - Stage 2 (this module): a lightweight graph model that combines the
    MLLM-derived item-semantic graph with a behaviour-Jaccard graph, then
    propagates ``item_feat`` through the resulting item-item graph.

Reference: https://github.com/Yuzhuo-Dang/MLLMRec
Paper: arXiv 2508.15304
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import GeneralRecommender
from src.common.loss import BPRLoss


def _build_knn_neighborhood(sim: torch.Tensor, topk: int) -> torch.Tensor:
    """Keep only top-``topk`` similarity per row, zero the rest. Dense output."""
    knn_val, knn_ind = torch.topk(sim, topk, dim=-1)
    out = torch.zeros_like(sim)
    out.scatter_(-1, knn_ind, knn_val)
    return out


def _dense_sym_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric Laplacian normalization D^{-1/2} A D^{-1/2} of a dense adj."""
    rowsum = adj.sum(dim=-1)
    d_inv = torch.pow(rowsum + 1e-7, -0.5)
    d_inv[torch.isinf(d_inv)] = 0.0
    D = torch.diagflat(d_inv)
    return D @ adj @ D


class MLLMRec(GeneralRecommender):
    """MLLMRec — graph-refinement stage on pre-computed MLLM features.

    The MLLM features are loaded from disk: looks for
    ``<data_path>/mllm_item_feat.npy`` and ``<data_path>/mllm_user_feat.npy``.
    Both are expected to be float32 arrays of shape ``[n, D]``.
    """

    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: torch.Tensor,
                 train_user_idx: torch.Tensor,
                 train_item_idx: torch.Tensor,
                 item_feat: torch.Tensor | None = None,
                 user_feat: torch.Tensor | None = None) -> None:
        super().__init__(config, n_users, n_items)

        self.feat_embed_dim = int(config.get("feat_embed_dim", 64))
        self.n_ii_layers = int(config.get("n_ii_layers", 1))
        self.n_ui_layers = int(config.get("n_ui_layers", 2))
        self.knn_k = int(config.get("knn_k", 10))
        self.knn_jac = int(config.get("knn_jac", 10))
        self.pure = float(config.get("pure", 0.7))
        self.norm_adj = norm_adj

        # Load MLLM features from disk if not provided.
        if item_feat is None or user_feat is None:
            project_root = Path(__file__).resolve().parents[2]
            data_path = project_root / str(config["data_path"])
            item_path = data_path / str(config.get("mllm_item_file", "mllm_item_feat.npy"))
            user_path = data_path / str(config.get("mllm_user_file", "mllm_user_feat.npy"))
            if not item_path.is_file() or not user_path.is_file():
                raise FileNotFoundError(
                    f"MLLMRec needs mllm_item_feat.npy and mllm_user_feat.npy in "
                    f"{data_path}. Precompute these MLLM feature files first.")
            item_feat = torch.from_numpy(np.load(item_path).astype(np.float32))
            user_feat = torch.from_numpy(np.load(user_path).astype(np.float32))

        if item_feat.shape[0] != self.n_items:
            raise ValueError(
                f"mllm_item_feat n_items {item_feat.shape[0]} != {self.n_items}")
        if user_feat.shape[0] != self.n_users:
            raise ValueError(
                f"mllm_user_feat n_users {user_feat.shape[0]} != {self.n_users}")
        item_feat = item_feat.float()
        user_feat = user_feat.float()
        self.register_buffer("item_feat", item_feat, persistent=False)
        self.register_buffer("user_feat", user_feat, persistent=False)
        in_dim = item_feat.shape[1]

        # Build item-item graph (one-shot at init).
        # Semantic similarity: cosine over item_feat -> top-k -> threshold by `pure`
        with torch.no_grad():
            ifeat_n = F.normalize(item_feat, dim=-1)
            sem_sim = ifeat_n @ ifeat_n.t()
            sem_adj = _build_knn_neighborhood(sem_sim, self.knn_k)
            sem_adj = torch.where(sem_adj >= self.pure,
                                  torch.ones_like(sem_adj),
                                  torch.zeros_like(sem_adj))

            # Behaviour Jaccard between items.
            R = torch.zeros(self.n_items, self.n_users, dtype=torch.float32)
            R[train_item_idx.long(), train_user_idx.long()] = 1.0
            intersection = R @ R.t()
            row_sum = R.sum(dim=-1, keepdim=True)
            union = row_sum + row_sum.t() - intersection
            jac = intersection / (union + 1e-7)
            jac.fill_diagonal_(0.0)
            jac_adj = _build_knn_neighborhood(jac, self.knn_jac)

            ii_mat = _dense_sym_laplacian(sem_adj + jac_adj)
        # Store as sparse to save memory (item_count^2 is small on Baby = 50M floats = 200MB)
        self.register_buffer("ii_mat", ii_mat.to_sparse_coo().coalesce(),
                             persistent=False)

        # MLPs.
        self.item_mlp = nn.Sequential(
            nn.Linear(in_dim, 4 * self.feat_embed_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * self.feat_embed_dim, self.feat_embed_dim),
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(user_feat.shape[1], 4 * self.feat_embed_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * self.feat_embed_dim, self.feat_embed_dim),
        )

        self.bpr_loss = BPRLoss()

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Item-item GCN on raw item_feat then MLP.
        e_i = self.item_feat
        all_embs = [e_i]
        for _ in range(self.n_ii_layers):
            e_i = torch.sparse.mm(self.ii_mat, e_i)
            all_embs.append(e_i)
        e_i = torch.stack(all_embs, dim=1).sum(dim=1)
        h_i = self.item_mlp(e_i)
        h_u = self.user_mlp(self.user_feat)
        return h_u, h_i

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        h_u, h_i = self._propagate()
        u_e, p_e, n_e = h_u[users], h_i[pos], h_i[neg]
        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        loss = self.bpr_loss(pos_score, neg_score)
        return loss, {"mf": loss.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        h_u, h_i = self._propagate()
        return h_u[users] @ h_i.t()

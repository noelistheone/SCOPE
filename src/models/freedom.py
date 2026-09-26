"""FREEDOM: Zhou & Shen, MM 2023.

Two graphs:
  - A *frozen* item-item multimodal graph (precomputed kNN over raw v/t features).
  - The user-item bipartite graph, with degree-sensitive edge dropout each epoch.

Loss = BPR(scored on propagated embeddings) + reg_weight * BPR(scored on each
modality's raw projection). The modality features are loaded into trainable
``nn.Embedding`` tables initialized from disk (frozen=False).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import BPRLoss
from src.data.graph_utils import build_knn_graph


class FREEDOM(MultimodalRecommender):
    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: torch.Tensor,
                 train_user_idx: torch.Tensor,
                 train_item_idx: torch.Tensor,
                 v_feat: Optional[torch.Tensor] = None,
                 t_feat: Optional[torch.Tensor] = None) -> None:
        super().__init__(config, n_users, n_items,
                         norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        if not self.has_modalities():
            raise ValueError("FREEDOM requires at least one of v_feat or t_feat")

        self.n_layers = int(config.get("n_layers", 2))         # user-item layers
        self.n_mm_layers = int(config.get("n_mm_layers", 1))   # item-graph layers
        self.knn_k = int(config.get("knn_k", 10))
        self.mm_image_weight = float(config.get("mm_image_weight", 0.1))
        self.dropout = float(config.get("dropout", 0.8))

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat.clone(),
                                                                freeze=False)
            self.image_trs = nn.Linear(self.v_feat_dim, self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat.clone(),
                                                               freeze=False)
            self.text_trs = nn.Linear(self.t_feat_dim, self.feat_embed_dim)

        # Frozen multimodal item-item graph (built once from raw features).
        mm_adj = self._build_mm_adj()
        self.register_buffer("mm_adj", mm_adj, persistent=False)

        # Edge info for degree-sensitive dropout.
        if not isinstance(train_user_idx, torch.Tensor):
            train_user_idx = torch.as_tensor(np.asarray(train_user_idx), dtype=torch.long)
        if not isinstance(train_item_idx, torch.Tensor):
            train_item_idx = torch.as_tensor(np.asarray(train_item_idx), dtype=torch.long)
        edge_indices = torch.stack([train_user_idx.long(),
                                    train_item_idx.long()], dim=0)
        edge_values = self._row_col_norm(edge_indices)
        self.register_buffer("edge_indices", edge_indices, persistent=False)
        self.register_buffer("edge_values", edge_values, persistent=False)

        self.masked_adj: Optional[torch.Tensor] = None
        self.bpr_loss = BPRLoss()

    # ------------------------------------------------------------------

    def _build_mm_adj(self) -> torch.Tensor:
        if self.v_feat is not None and self.t_feat is not None:
            img = build_knn_graph(self.v_feat, self.knn_k)
            txt = build_knn_graph(self.t_feat, self.knn_k)
            # Combine in sparse form (avoid materializing two [n_items, n_items]
            # dense matrices, which is 2 GB each for Clothing).
            indices = torch.cat([img.indices(), txt.indices()], dim=1)
            values = torch.cat([
                self.mm_image_weight * img.values(),
                (1.0 - self.mm_image_weight) * txt.values(),
            ])
            # coalesce() sums duplicate (row, col) entries.
            return torch.sparse_coo_tensor(indices, values, img.shape).coalesce()
        if self.v_feat is not None:
            return build_knn_graph(self.v_feat, self.knn_k)
        return build_knn_graph(self.t_feat, self.knn_k)

    def _row_col_norm(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute D_row^{-1/2} * D_col^{-1/2} weight for each edge."""
        eps = 1e-7
        row_count = torch.zeros(self.n_users)
        col_count = torch.zeros(self.n_items)
        row_count.scatter_add_(0, indices[0].cpu(), torch.ones_like(indices[0], dtype=torch.float))
        col_count.scatter_add_(0, indices[1].cpu(), torch.ones_like(indices[1], dtype=torch.float))
        r_inv = torch.pow(row_count + eps, -0.5)
        c_inv = torch.pow(col_count + eps, -0.5)
        return r_inv[indices[0].cpu()] * c_inv[indices[1].cpu()]

    def pre_epoch_processing(self, epoch: int) -> None:
        if self.dropout <= 0.0:
            self.masked_adj = self.norm_adj
            return
        keep_n = int(self.edge_values.numel() * (1.0 - self.dropout))
        keep_n = max(1, keep_n)
        # Degree-sensitive sampling: edges with higher normalized weight more
        # likely to survive.
        keep_idx = torch.multinomial(self.edge_values, num_samples=keep_n,
                                     replacement=False)
        sel_edges = self.edge_indices[:, keep_idx]
        sel_vals = self.edge_values[keep_idx]

        # Build the symmetric [n_users + n_items, n_users + n_items] sparse adj.
        u_to_i_rows = sel_edges[0]
        u_to_i_cols = sel_edges[1] + self.n_users
        i_to_u_rows = u_to_i_cols
        i_to_u_cols = u_to_i_rows
        all_rows = torch.cat([u_to_i_rows, i_to_u_rows])
        all_cols = torch.cat([u_to_i_cols, i_to_u_cols])
        all_vals = torch.cat([sel_vals, sel_vals])
        n_total = self.n_users + self.n_items
        self.masked_adj = torch.sparse_coo_tensor(
            torch.stack([all_rows, all_cols]),
            all_vals,
            (n_total, n_total),
            device=self.norm_adj.device,
        ).coalesce()

    # ------------------------------------------------------------------

    def _propagate(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Item-graph propagation on item IDs.
        h = self.item_id_embedding.weight
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # User-item LightGCN-style propagation.
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adj, ego)
            all_embs.append(ego)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        user_e, item_e = torch.split(out, [self.n_users, self.n_items], dim=0)
        return user_e, item_e + h

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        adj = self.masked_adj if self.masked_adj is not None else self.norm_adj
        u_all, i_all = self._propagate(adj)
        u_e = u_all[users]
        p_e = i_all[pos]
        n_e = i_all[neg]
        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        mf_loss = self.bpr_loss(pos_score, neg_score)

        # Auxiliary modality losses.
        mf_v_loss = mf_t_loss = u_all.new_zeros(())
        if self.t_feat is not None:
            text = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(
                (u_e * text[pos]).sum(dim=-1),
                (u_e * text[neg]).sum(dim=-1),
            )
        if self.v_feat is not None:
            image = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(
                (u_e * image[pos]).sum(dim=-1),
                (u_e * image[neg]).sum(dim=-1),
            )

        total = mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)
        return total, {
            "mf": mf_loss.item(),
            "mf_v": float(mf_v_loss) if isinstance(mf_v_loss, torch.Tensor) else mf_v_loss,
            "mf_t": float(mf_t_loss) if isinstance(mf_t_loss, torch.Tensor) else mf_t_loss,
        }

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u_all, i_all = self._propagate(self.norm_adj)
        return u_all[users] @ i_all.t()

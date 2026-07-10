"""LightGCN: He et al., SIGIR 2020.

Pure-CF baseline. Stacks ``n_layers`` of normalized adjacency propagation,
averages the resulting layer-wise embeddings, and scores via dot product.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.common.abstract_recommender import GeneralGraphRecommender
from src.common.init import xavier_uniform_initialization
from src.common.loss import BPRLoss, EmbLoss


class LightGCN(GeneralGraphRecommender):
    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: torch.Tensor) -> None:
        super().__init__(config, n_users, n_items, norm_adj=norm_adj)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.apply(xavier_uniform_initialization)

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    # ----- core propagation -----

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        stacked = torch.stack(all_embs, dim=1)        # [N, L+1, D]
        out = stacked.mean(dim=1)
        user_e, item_e = torch.split(out, [self.n_users, self.n_items], dim=0)
        return user_e, item_e

    # ----- API -----

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate()

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        user_all, item_all = self._propagate()
        u_e = user_all[users]
        p_e = item_all[pos]
        n_e = item_all[neg]

        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        mf_loss = self.bpr_loss(pos_score, neg_score)

        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos)
        n0 = self.item_embedding(neg)
        reg_loss = self.reg_loss(u0, p0, n0)

        loss = mf_loss + self.reg_weight * reg_loss
        return loss, {"mf": mf_loss.item(), "reg": reg_loss.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        user_all, item_all = self._propagate()
        u_e = user_all[users]
        return u_e @ item_all.t()

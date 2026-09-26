"""VBPR: He & McAuley, AAAI 2016.

Item embedding = [ID embedding || projected multimodal feature]. No graph,
no propagation. The user embedding is also doubled to match the concatenated
item dimension.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from src.common.abstract_recommender import MultimodalRecommender
from src.common.init import xavier_normal_initialization
from src.common.loss import BPRLoss, EmbLoss


class VBPR(MultimodalRecommender):
    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: Optional[torch.Tensor] = None,
                 v_feat: Optional[torch.Tensor] = None,
                 t_feat: Optional[torch.Tensor] = None) -> None:
        super().__init__(config, n_users, n_items,
                         norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        if not self.has_modalities():
            raise ValueError("VBPR requires at least one of v_feat or t_feat")

        # User embedding is 2x size to match [id || feat] concatenated item emb.
        self.user_embedding = nn.Parameter(
            torch.empty(self.n_users, self.embedding_size * 2))
        self.item_id_embedding = nn.Parameter(
            torch.empty(self.n_items, self.embedding_size))

        feat_dim_total = self.v_feat_dim + self.t_feat_dim
        self.item_linear = nn.Linear(feat_dim_total, self.embedding_size)

        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_id_embedding)
        self.apply(xavier_normal_initialization)

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def _item_embeddings(self) -> torch.Tensor:
        if self.v_feat is not None and self.t_feat is not None:
            feat = torch.cat([self.t_feat, self.v_feat], dim=-1)
        elif self.v_feat is not None:
            feat = self.v_feat
        else:
            feat = self.t_feat
        feat_proj = self.item_linear(feat)
        return torch.cat([self.item_id_embedding, feat_proj], dim=-1)

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        item_all = self._item_embeddings()
        u_e = self.user_embedding[users]
        p_e = item_all[pos]
        n_e = item_all[neg]

        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        mf_loss = self.bpr_loss(pos_score, neg_score)
        reg = self.reg_loss(u_e, p_e, n_e)
        loss = mf_loss + self.reg_weight * reg
        return loss, {"mf": mf_loss.item(), "reg": reg.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        item_all = self._item_embeddings()
        u_e = self.user_embedding[users]
        return u_e @ item_all.t()

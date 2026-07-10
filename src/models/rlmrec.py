"""RLMRec (Ren et al., WWW 2024) — clean-room reimplementation.

Core idea: align LLM-derived user/item profiles with collaborative embeddings
via a **mutual-information** objective (InfoNCE). The paper uses ChatGPT to
generate user and item profile descriptions, then a pretrained text encoder
to produce embeddings.

We substitute the LLM-derived profile embeddings with what is already
available in MMRec's pre-processed datasets:
  - Item profile ≈ ``t_feat`` (Sentence-BERT over item text).
  - User profile ≈ mean of t_feat over their training items.

Loss = BPR + λ * (InfoNCE on item alignment + InfoNCE on user alignment).

Reference: https://github.com/HKUDS/RLMRec
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import BPRLoss, EmbLoss, InfoNCELoss
from src.data.graph_utils import split_norm_adj_blocks


class RLMRec(MultimodalRecommender):
    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: torch.Tensor,
                 v_feat: Optional[torch.Tensor] = None,
                 t_feat: Optional[torch.Tensor] = None) -> None:
        super().__init__(config, n_users, n_items,
                         norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        if self.t_feat is None:
            raise ValueError("RLMRec requires t_feat (LLM profile substitute)")

        emb = self.embedding_size
        self.n_layers = int(config.get("n_layers", 2))
        self.cl_weight = float(config.get("cl_weight", 0.1))
        self.cl_temp = float(config.get("cl_temp", 0.2))

        self.user_embedding = nn.Embedding(self.n_users, emb)
        self.item_embedding = nn.Embedding(self.n_items, emb)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.item_proj = nn.Linear(self.t_feat_dim, emb)
        self.user_proj = nn.Linear(self.t_feat_dim, emb)

        # Precompute user profile = mean of t_feat over their train items.
        R, _ = split_norm_adj_blocks(self.norm_adj, self.n_users, self.n_items)
        with torch.no_grad():
            user_prof = torch.sparse.mm(R, self.t_feat)
        self.register_buffer("user_profile", user_prof, persistent=False)

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.infonce = InfoNCELoss(temperature=self.cl_temp)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat([self.user_embedding.weight,
                         self.item_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        return torch.split(out, [self.n_users, self.n_items], dim=0)

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        u, i = self._propagate()
        u_e, p_e, n_e = u[users], i[pos], i[neg]
        pos_s = (u_e * p_e).sum(dim=-1)
        neg_s = (u_e * n_e).sum(dim=-1)
        mf = self.bpr_loss(pos_s, neg_s)

        # Profile alignment.
        item_prof_e = self.item_proj(self.t_feat[pos])
        user_prof_e = self.user_proj(self.user_profile[users])
        cl = (self.infonce(i[pos], item_prof_e)
              + self.infonce(u_e, user_prof_e)) * 0.5

        reg = self.reg_loss(self.user_embedding(users),
                            self.item_embedding(pos),
                            self.item_embedding(neg))
        total = mf + self.cl_weight * cl + self.reg_weight * reg
        return total, {"mf": mf.item(), "cl": cl.item(), "reg": reg.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u, i = self._propagate()
        return u[users] @ i.t()

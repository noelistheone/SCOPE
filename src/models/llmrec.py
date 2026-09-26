"""LLMRec (HKUDS, WSDM 2024) — simplified clean-room implementation.

The full LLMRec paper uses an LLM to (1) generate plausible user-item
augmented interactions, (2) summarise textual side info per item, and
(3) regularize the recommender with InfoNCE between LLM-derived embeddings
and ID-based embeddings. The expensive LLM stages are run offline.

This implementation captures the **alignment** core:
  - LightGCN backbone over the bipartite graph.
  - Each item gets an "LLM text" embedding (substituted with our ``t_feat`` —
    in the original it would come from an LLM-summarised description).
  - InfoNCE between the propagated item ID embedding and the projected text
    embedding at positive items, encouraging the CF tower to align with the
    LLM-derived semantic space.

Reference: https://github.com/HKUDS/LLMRec
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import BPRLoss, EmbLoss, InfoNCELoss


class LLMRec(MultimodalRecommender):
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
            raise ValueError("LLMRec requires t_feat (LLM-derived text feature)")

        emb = self.embedding_size
        self.n_layers = int(config.get("n_layers", 2))
        self.cl_weight = float(config.get("cl_weight", 0.1))
        self.cl_temp = float(config.get("cl_temp", 0.2))

        self.user_embedding = nn.Embedding(self.n_users, emb)
        self.item_embedding = nn.Embedding(self.n_items, emb)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.text_proj = nn.Linear(self.t_feat_dim, emb)
        # Optional visual projection.
        if self.v_feat is not None:
            self.image_proj = nn.Linear(self.v_feat_dim, emb)

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.infonce = InfoNCELoss(temperature=self.cl_temp)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego = torch.cat([self.user_embedding.weight,
                         self.item_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        u_e, i_e = torch.split(out, [self.n_users, self.n_items], dim=0)

        t_emb = self.text_proj(self.t_feat)
        return u_e, i_e, t_emb

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        u, i, t_emb = self._propagate()
        u_e, p_e, n_e = u[users], i[pos], i[neg]
        pos_s = (u_e * p_e).sum(dim=-1)
        neg_s = (u_e * n_e).sum(dim=-1)
        mf = self.bpr_loss(pos_s, neg_s)

        # CF-to-LLM alignment at positive items (item-side).
        cl = self.infonce(i[pos], t_emb[pos])

        reg = self.reg_loss(self.user_embedding(users),
                            self.item_embedding(pos),
                            self.item_embedding(neg))
        total = mf + self.cl_weight * cl + self.reg_weight * reg
        return total, {"mf": mf.item(), "cl": cl.item(), "reg": reg.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u, i, _ = self._propagate()
        return u[users] @ i.t()

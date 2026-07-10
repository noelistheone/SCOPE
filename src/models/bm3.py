"""BM3: Zhou et al., WWW 2023.

Bootstrap-style multimodal recommender. No explicit BPR loss — instead it
uses cosine alignment between an "online" (predictor-augmented) view and a
"target" (dropout-perturbed, detached) view across user, item, and modality
embeddings.

Note: this model does NOT use negative samples. The Trainer still hands them
in (they're cheap to sample), but ``calculate_loss`` simply ignores them.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import EmbLoss


class BM3(MultimodalRecommender):
    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: torch.Tensor,
                 v_feat: Optional[torch.Tensor] = None,
                 t_feat: Optional[torch.Tensor] = None) -> None:
        super().__init__(config, n_users, n_items,
                         norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        if not self.has_modalities():
            raise ValueError("BM3 requires at least one of v_feat or t_feat")

        self.n_layers = int(config.get("n_layers", 2))
        self.dropout = float(config.get("dropout", 0.5))
        self.cl_weight = float(config.get("cl_weight", 2.0))

        # BM3 forces feat_embed_dim == embedding_size (single projection space).
        emb = self.embedding_size

        self.user_embedding = nn.Embedding(self.n_users, emb)
        self.item_id_embedding = nn.Embedding(self.n_items, emb)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(emb, emb)
        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat.clone(),
                                                                freeze=False)
            self.image_trs = nn.Linear(self.v_feat_dim, emb)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat.clone(),
                                                               freeze=False)
            self.text_trs = nn.Linear(self.t_feat_dim, emb)
            nn.init.xavier_normal_(self.text_trs.weight)

        self.reg_loss = EmbLoss()

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.item_id_embedding.weight
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        u_e, i_e = torch.split(out, [self.n_users, self.n_items], dim=0)
        return u_e, i_e + h

    def calculate_loss(self, interaction):
        users = interaction["user"]
        items = interaction["pos_item"]

        u_online_ori, i_online_ori = self._propagate()
        t_feat_online = (self.text_trs(self.text_embedding.weight)
                         if self.t_feat is not None else None)
        v_feat_online = (self.image_trs(self.image_embedding.weight)
                         if self.v_feat is not None else None)

        with torch.no_grad():
            u_target = F.dropout(u_online_ori.clone().detach(), self.dropout)
            i_target = F.dropout(i_online_ori.clone().detach(), self.dropout)
            t_target = (F.dropout(t_feat_online.clone().detach(), self.dropout)
                        if t_feat_online is not None else None)
            v_target = (F.dropout(v_feat_online.clone().detach(), self.dropout)
                        if v_feat_online is not None else None)

        u_online = self.predictor(u_online_ori)
        i_online = self.predictor(i_online_ori)

        u_o = u_online[users]
        i_o = i_online[items]
        u_t = u_target[users]
        i_t = i_target[items]

        loss_ui = 1 - F.cosine_similarity(u_o, i_t, dim=-1).mean()
        loss_iu = 1 - F.cosine_similarity(i_o, u_t, dim=-1).mean()

        loss_t = u_online_ori.new_zeros(())
        loss_v = u_online_ori.new_zeros(())
        loss_tv = u_online_ori.new_zeros(())
        loss_vt = u_online_ori.new_zeros(())

        if t_feat_online is not None:
            t_o = self.predictor(t_feat_online)[items]
            t_tg = t_target[items]
            loss_t = 1 - F.cosine_similarity(t_o, i_t, dim=-1).mean()
            loss_tv = 1 - F.cosine_similarity(t_o, t_tg, dim=-1).mean()
        if v_feat_online is not None:
            v_o = self.predictor(v_feat_online)[items]
            v_tg = v_target[items]
            loss_v = 1 - F.cosine_similarity(v_o, i_t, dim=-1).mean()
            loss_vt = 1 - F.cosine_similarity(v_o, v_tg, dim=-1).mean()

        reg = self.reg_loss(u_online_ori, i_online_ori)
        total = (loss_ui + loss_iu) \
            + self.reg_weight * reg \
            + self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt)
        return total, {
            "ui": loss_ui.item(),
            "iu": loss_iu.item(),
            "reg": reg.item(),
            "cl": float(loss_t + loss_v + loss_tv + loss_vt),
        }

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u_e, i_e = self._propagate()
        u_e = self.predictor(u_e)
        i_e = self.predictor(i_e)
        return u_e[users] @ i_e.t()

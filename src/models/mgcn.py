"""MGCN: Yu et al., MM 2023.

Multi-View Graph Convolutional Network. Three signals are fused:
  - User-Item view: standard LightGCN on the bipartite graph.
  - Image view: item embeddings gated by image features, propagated through
    a frozen image kNN graph, then aggregated to users via R.
  - Text view: same with text.

The image and text representations are split into a "common" part (attention
fused) and a "specific" residual; an SSL InfoNCE loss aligns the fused side
view with the content (user-item) view at the user and pos-item indices.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import BPRLoss, EmbLoss, InfoNCELoss
from src.data.graph_utils import build_knn_graph, split_norm_adj_blocks


class MGCN(MultimodalRecommender):
    def __init__(self,
                 config: Mapping[str, Any],
                 n_users: int,
                 n_items: int,
                 norm_adj: torch.Tensor,
                 v_feat: Optional[torch.Tensor] = None,
                 t_feat: Optional[torch.Tensor] = None) -> None:
        super().__init__(config, n_users, n_items,
                         norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        if not (self.v_feat is not None and self.t_feat is not None):
            raise ValueError("MGCN requires BOTH v_feat and t_feat")

        self.n_layers = int(config.get("n_layers", 2))            # item-graph layers
        self.n_ui_layers = int(config.get("n_ui_layers", 2))      # user-item layers
        self.knn_k = int(config.get("knn_k", 10))
        self.cl_weight = float(config.get("cl_weight", 1e-3))
        emb = self.embedding_size

        self.user_embedding = nn.Embedding(self.n_users, emb)
        self.item_id_embedding = nn.Embedding(self.n_items, emb)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat.clone(),
                                                            freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat.clone(),
                                                           freeze=False)
        self.image_trs = nn.Linear(self.v_feat_dim, emb)
        self.text_trs = nn.Linear(self.t_feat_dim, emb)

        # Frozen modal item-item graphs.
        self.register_buffer("image_adj",
                             build_knn_graph(self.v_feat, self.knn_k),
                             persistent=False)
        self.register_buffer("text_adj",
                             build_knn_graph(self.t_feat, self.knn_k),
                             persistent=False)

        # Normalized R block (n_users x n_items) extracted from norm_adj.
        R, _ = split_norm_adj_blocks(self.norm_adj, self.n_users, self.n_items)
        self.register_buffer("R", R, persistent=False)

        self.query_common = nn.Sequential(
            nn.Linear(emb, emb),
            nn.Tanh(),
            nn.Linear(emb, 1, bias=False),
        )
        self.gate_v = nn.Sequential(nn.Linear(emb, emb), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(emb, emb), nn.Sigmoid())
        self.gate_image_prefer = nn.Sequential(nn.Linear(emb, emb), nn.Sigmoid())
        self.gate_text_prefer = nn.Sequential(nn.Linear(emb, emb), nn.Sigmoid())

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.infonce = InfoNCELoss(temperature=0.2)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        image_item_embeds = self.item_id_embedding.weight * self.gate_v(image_feats)
        text_item_embeds = self.item_id_embedding.weight * self.gate_t(text_feats)

        # User-Item view (LightGCN).
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        content_embeds = torch.stack(all_embs, dim=1).mean(dim=1)

        # Image view.
        for _ in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        # Text view.
        for _ in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Behavior-Aware Fuser.
        att_common = torch.cat([self.query_common(image_embeds),
                                self.query_common(text_embeds)], dim=-1)
        weight_common = F.softmax(att_common, dim=-1)
        common = weight_common[:, 0:1] * image_embeds + weight_common[:, 1:2] * text_embeds
        sep_image = image_embeds - common
        sep_text = text_embeds - common

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image = image_prefer * sep_image
        sep_text = text_prefer * sep_text
        side_embeds = (sep_image + sep_text + common) / 3

        all_embeds = content_embeds + side_embeds
        u_e, i_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        return u_e, i_e, side_embeds, content_embeds

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        u_all, i_all, side_embeds, content_embeds = self._propagate()
        u_e, p_e, n_e = u_all[users], i_all[pos], i_all[neg]
        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        mf_loss = self.bpr_loss(pos_score, neg_score)
        reg = self.reg_loss(
            self.user_embedding(users),
            self.item_id_embedding(pos),
            self.item_id_embedding(neg),
        )

        side_u, side_i = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        cont_u, cont_i = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.infonce(side_i[pos], cont_i[pos]) + \
                  self.infonce(side_u[users], cont_u[users])

        total = mf_loss + self.reg_weight * reg + self.cl_weight * cl_loss
        return total, {
            "mf": mf_loss.item(),
            "reg": reg.item(),
            "cl": cl_loss.item(),
        }

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u_all, i_all, _, _ = self._propagate()
        return u_all[users] @ i_all.t()

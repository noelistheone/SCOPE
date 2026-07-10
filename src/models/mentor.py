"""MENTOR (Xu et al., AAAI 2024): a clean-room reimplementation.

Multi-level self-supervised learning for multimodal recommendation. Three signals:
  - User-item LightGCN view producing (u, i_id) embeddings.
  - Image view: items propagated through a frozen image-item kNN graph.
  - Text view: items propagated through a frozen text-item kNN graph.

Two SSL losses on top of BPR:
  - **align**: cross-modal alignment between projected image/text item features
    (cosine-similarity gap, encouraging the same item's image and text encoders
    to agree).
  - **ssl**: ID-to-modality InfoNCE between the propagated item embedding and
    its modality view at positive items.

This is *not* a literal port (the official MENTOR code wasn't available in
MMRec); it follows the published description and is consistent with the
hyperparameter shape MENTOR reports.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import BPRLoss, EmbLoss, InfoNCELoss
from src.data.graph_utils import build_knn_graph


class MENTOR(MultimodalRecommender):
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
            raise ValueError("MENTOR requires BOTH v_feat and t_feat")

        self.n_layers = int(config.get("n_layers", 2))
        self.n_ui_layers = int(config.get("n_ui_layers", 2))
        self.knn_k = int(config.get("knn_k", 10))
        self.align_weight = float(config.get("align_weight", 0.1))
        self.ssl_weight = float(config.get("ssl_weight", 0.1))
        ssl_temp = float(config.get("ssl_temp", 0.2))
        emb = self.embedding_size

        self.user_embedding = nn.Embedding(self.n_users, emb)
        self.item_id_embedding = nn.Embedding(self.n_items, emb)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_trs = nn.Linear(self.v_feat_dim, emb)
        self.text_trs = nn.Linear(self.t_feat_dim, emb)

        self.register_buffer("image_adj",
                             build_knn_graph(self.v_feat, self.knn_k),
                             persistent=False)
        self.register_buffer("text_adj",
                             build_knn_graph(self.t_feat, self.knn_k),
                             persistent=False)

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.infonce = InfoNCELoss(temperature=ssl_temp)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (u_all, i_all, image_item, text_item).

        - u_all, i_all: user-item LightGCN outputs combined with modality items.
        - image_item, text_item: per-modality item representations after
          propagation through their respective kNN graphs.
        """
        image_item = self.image_trs(self.v_feat)
        text_item = self.text_trs(self.t_feat)

        # Modality-side propagation through the (frozen) kNN item graphs.
        for _ in range(self.n_layers):
            image_item = torch.sparse.mm(self.image_adj, image_item)
            text_item = torch.sparse.mm(self.text_adj, text_item)

        # User-item LightGCN.
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        u_e, i_id = torch.split(out, [self.n_users, self.n_items], dim=0)

        # Fuse: items pick up the multimodal signal as a normalized residual.
        # Normalization prevents the (typically larger) modality embeddings from
        # swamping the LightGCN ID signal — this was the root cause of an early
        # failure mode (R@20 collapsed to ~0.01 when the modality residual
        # dominated).
        mm = F.normalize(image_item, dim=-1) + F.normalize(text_item, dim=-1)
        i_e = i_id + 0.5 * mm
        return u_e, i_e, image_item, text_item

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        u_all, i_all, image_item, text_item = self._propagate()
        u_e, p_e, n_e = u_all[users], i_all[pos], i_all[neg]
        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        mf_loss = self.bpr_loss(pos_score, neg_score)
        reg = self.reg_loss(
            self.user_embedding(users),
            self.item_id_embedding(pos),
            self.item_id_embedding(neg),
        )

        # Cross-modal alignment at positive items.
        img_p = F.normalize(image_item[pos], dim=-1)
        txt_p = F.normalize(text_item[pos], dim=-1)
        align_loss = (1.0 - (img_p * txt_p).sum(dim=-1)).mean()

        # SSL InfoNCE: ID branch vs modality view at positive items.
        ssl_loss = 0.5 * (self.infonce(self.item_id_embedding(pos), image_item[pos])
                          + self.infonce(self.item_id_embedding(pos), text_item[pos]))

        total = (mf_loss
                 + self.reg_weight * reg
                 + self.align_weight * align_loss
                 + self.ssl_weight * ssl_loss)
        return total, {
            "mf": mf_loss.item(),
            "reg": reg.item(),
            "align": align_loss.item(),
            "ssl": ssl_loss.item(),
        }

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u_all, i_all, _, _ = self._propagate()
        return u_all[users] @ i_all.t()

"""LATTICE: Zhang et al., MM 2021.

Builds item-item kNN graphs from each modality, learns a soft fusion of them,
and adds the propagated item-graph signal on top of LightGCN's user-item
propagation. The frozen "original" item-graph is mixed (``lambda_coeff``)
with the periodically-rebuilt "learned" item-graph derived from the current
projected features.

Memory note: instead of recomputing the kNN graph each batch, we rebuild only
at the start of each epoch (``pre_epoch_processing``), matching MMRec's
``build_item_graph`` semantics.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.init import xavier_uniform_initialization
from src.common.loss import BPRLoss, EmbLoss
from src.data.graph_utils import build_knn_graph, sparse_row_topk


class LATTICE(MultimodalRecommender):
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
            raise ValueError("LATTICE requires at least one of v_feat or t_feat")

        self.knn_k = int(config.get("knn_k", 10))
        self.lambda_coeff = float(config.get("lambda_coeff", 0.9))
        self.n_layers = int(config.get("n_layers", 1))            # item-graph layers
        self.n_ui_layers = int(config.get("n_ui_layers", 2))      # user-item layers

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat_dim, self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat_dim, self.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Frozen original kNN graphs (built once from raw features).
        self.register_buffer(
            "_orig_image_adj",
            build_knn_graph(self.v_feat, self.knn_k) if self.v_feat is not None
            else torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_orig_text_adj",
            build_knn_graph(self.t_feat, self.knn_k) if self.t_feat is not None
            else torch.empty(0),
            persistent=False,
        )

        self._item_adj: Optional[torch.Tensor] = None
        self._build_item_graph = True

        self.apply(xavier_uniform_initialization)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def pre_epoch_processing(self, epoch: int) -> None:
        # Rebuild the learned item-item graph once per epoch.
        self._build_item_graph = True

    def _rebuild_item_adj(self) -> torch.Tensor:
        """Build the fused item-item graph entirely in sparse form.

        We never materialize a dense [n_items, n_items] tensor — that would be
        ~2 GB on Clothing. Instead, each modality's learned + original kNN
        graph contributes a weighted sparse matrix; we concatenate them and
        let ``coalesce`` add overlapping entries, then take per-row top-k.
        """
        weight = F.softmax(self.modal_weight, dim=0)
        target_device = self.modal_weight.device

        parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.v_feat is not None:
            img_proj = self.image_trs(self.v_feat).detach()
            learned_img = build_knn_graph(img_proj, self.knn_k)
            w_idx = 0
            parts.append((learned_img.to(target_device),
                          weight[w_idx] * (1 - self.lambda_coeff)))
            parts.append((self._orig_image_adj.to(target_device),
                          weight[w_idx] * self.lambda_coeff))
        if self.t_feat is not None:
            txt_proj = self.text_trs(self.t_feat).detach()
            learned_txt = build_knn_graph(txt_proj, self.knn_k)
            w_idx = 1 if self.v_feat is not None else 0
            parts.append((learned_txt.to(target_device),
                          weight[w_idx] * (1 - self.lambda_coeff)))
            parts.append((self._orig_text_adj.to(target_device),
                          weight[w_idx] * self.lambda_coeff))

        indices = torch.cat([sp.indices() for sp, _ in parts], dim=1)
        values = torch.cat([sp.values() * w for sp, w in parts])
        shape = parts[0][0].shape
        fused = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        return sparse_row_topk(fused, self.knn_k)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._build_item_graph or self._item_adj is None:
            self._item_adj = self._rebuild_item_adj()
            self._build_item_graph = False
        else:
            # Detach to keep the graph fixed within an epoch (matches MMRec).
            self._item_adj = self._item_adj.detach()

        # Item-graph propagation on item ID embeddings.
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self._item_adj, h)

        # User-item LightGCN propagation.
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        user_e, item_e = torch.split(out, [self.n_users, self.n_items], dim=0)
        item_e = item_e + F.normalize(h, p=2, dim=-1)
        return user_e, item_e

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        u_all, i_all = self._propagate()
        u_e, p_e, n_e = u_all[users], i_all[pos], i_all[neg]
        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        mf_loss = self.bpr_loss(pos_score, neg_score)
        reg = self.reg_loss(
            self.user_embedding(users),
            self.item_id_embedding(pos),
            self.item_id_embedding(neg),
        )
        loss = mf_loss + self.reg_weight * reg
        return loss, {"mf": mf_loss.item(), "reg": reg.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u_all, i_all = self._propagate()
        return u_all[users] @ i_all.t()

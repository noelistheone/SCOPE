"""MMGCN: Wei et al., MM 2019.

Per-modality GCN tower over the user-item bipartite graph. Each modality
combines a per-user preference vector with item features, propagates through
the graph, and the modality outputs are averaged. Final scoring is dot
product on the combined [n_users + n_items, D] tensor.

This is a clean reimplementation using sparse adjacency mm (functionally
equivalent to torch_geometric's MessagePassing with ``aggr='mean'`` when the
adjacency is row-normalized — but we use the symmetric LightGCN normalization
which is what MMRec actually uses across other models).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.init import xavier_normal_initialization
from src.common.loss import BPRLoss, EmbLoss


class _ModalityTower(nn.Module):
    """Per-modality GCN tower used by MMGCN."""

    def __init__(self,
                 n_users: int,
                 n_items: int,
                 feat_dim: int,
                 dim_latent: int,
                 dim_id: int,
                 n_layers: int,
                 has_id: bool = True) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_id = has_id

        # User preference per modality (learned from scratch).
        self.preference = nn.Parameter(torch.empty(n_users, dim_latent))
        nn.init.xavier_normal_(self.preference)

        # Item feature -> dim_latent (no projection if already that dim).
        self.feat_mlp = nn.Linear(feat_dim, dim_latent)

        # n_layers blocks. Each block: linear conv + identity-residual + MLP.
        self.conv_w = nn.ParameterList([
            nn.Parameter(torch.empty(dim_latent if l == 0 else dim_id, dim_id))
            for l in range(n_layers)
        ])
        for w in self.conv_w:
            nn.init.xavier_normal_(w)

        self.linear_layers = nn.ModuleList([
            nn.Linear(dim_latent if l == 0 else dim_id, dim_id)
            for l in range(n_layers)
        ])
        self.g_layers = nn.ModuleList([
            nn.Linear(dim_id, dim_id) for _ in range(n_layers)
        ])

    def forward(self,
                norm_adj: torch.Tensor,
                feat: torch.Tensor,
                id_embedding: torch.Tensor) -> torch.Tensor:
        item_feat = self.feat_mlp(feat)
        x = torch.cat([self.preference, item_feat], dim=0)
        x = F.normalize(x, dim=-1)

        for l in range(self.n_layers):
            h = torch.sparse.mm(norm_adj, x @ self.conv_w[l])
            h = F.leaky_relu(h)
            x_hat = F.leaky_relu(self.linear_layers[l](x))
            if self.has_id:
                x_hat = x_hat + id_embedding
            x = F.leaky_relu(self.g_layers[l](h) + x_hat)

        return x


class MMGCN(MultimodalRecommender):
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
            raise ValueError("MMGCN requires at least one of v_feat or t_feat")

        dim_id = self.embedding_size
        dim_latent = self.feat_embed_dim
        n_layers = self.n_layers

        self.id_embedding = nn.Parameter(
            torch.empty(self.n_users + self.n_items, dim_id))
        nn.init.xavier_normal_(self.id_embedding)

        self.towers = nn.ModuleDict()
        if self.v_feat is not None:
            self.towers["v"] = _ModalityTower(
                self.n_users, self.n_items,
                feat_dim=self.v_feat_dim,
                dim_latent=dim_latent, dim_id=dim_id, n_layers=n_layers)
        if self.t_feat is not None:
            self.towers["t"] = _ModalityTower(
                self.n_users, self.n_items,
                feat_dim=self.t_feat_dim,
                dim_latent=dim_latent, dim_id=dim_id, n_layers=n_layers)

        self.apply(xavier_normal_initialization)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def _propagate(self) -> torch.Tensor:
        outs = []
        if "v" in self.towers:
            outs.append(self.towers["v"](self.norm_adj, self.v_feat, self.id_embedding))
        if "t" in self.towers:
            outs.append(self.towers["t"](self.norm_adj, self.t_feat, self.id_embedding))
        rep = torch.stack(outs, dim=0).mean(dim=0)   # average modalities
        return rep

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"] + self.n_users
        neg = interaction["neg_item"] + self.n_users

        rep = self._propagate()
        u_e = rep[users]
        p_e = rep[pos]
        n_e = rep[neg]

        pos_score = (u_e * p_e).sum(dim=-1)
        neg_score = (u_e * n_e).sum(dim=-1)
        mf_loss = self.bpr_loss(pos_score, neg_score)

        reg = self.reg_loss(self.id_embedding[users],
                            self.id_embedding[pos],
                            self.id_embedding[neg])
        loss = mf_loss + self.reg_weight * reg
        return loss, {"mf": mf_loss.item(), "reg": reg.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        rep = self._propagate()
        u_e = rep[:self.n_users][users]
        item_e = rep[self.n_users:]
        return u_e @ item_e.t()

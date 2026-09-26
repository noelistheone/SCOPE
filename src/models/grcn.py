"""GRCN: Graph-Refined Convolutional Network (Wei et al., ACM MM 2020).

Faithful port of the MMRec reference implementation (enoche/MMRec) retargeted to this
framework's ``MultimodalRecommender`` interface. Uses torch_geometric message passing.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from src.common.abstract_recommender import MultimodalRecommender


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add"):
        super().__init__(aggr=aggr)
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super().__init__(aggr="add")
        self.self_loops = self_loops
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_i, x_j, size_i, edge_index_i):
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        self.alpha = softmax(self.alpha, edge_index_i, num_nodes=size_i)
        return x_j * self.alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class EGCN(nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super().__init__()
        self.num_user, self.num_item, self.dim_E = num_user, num_item, dim_E
        self.has_act, self.has_norm = has_act, has_norm
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))
        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)
        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        if self.has_norm:
            x = F.normalize(x)
        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)
        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)
        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)
        return x + x_hat_1 + x_hat_2


class CGCN(nn.Module):
    def __init__(self, features, num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm):
        super().__init__()
        self.num_user, self.num_item = num_user, num_item
        self.num_routing, self.has_act, self.has_norm, self.dim_C = num_routing, has_act, has_norm, dim_C
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(dim_C, dim_C)
        self.register_buffer("features", features, persistent=False)
        self.MLP = nn.Linear(features.size(1), dim_C)
        nn.init.xavier_normal_(self.MLP.weight)

    def forward(self, edge_index):
        features = F.leaky_relu(self.MLP(self.features))
        preference = self.preference
        if self.has_norm:
            preference = F.normalize(preference)
            features = F.normalize(features)
        for _ in range(self.num_routing):
            x = torch.cat((preference, features), dim=0)
            x_hat_1 = self.conv_embed_1(x, edge_index)
            preference = preference + x_hat_1[: self.num_user]
            if self.has_norm:
                preference = F.normalize(preference)
        x = torch.cat((preference, features), dim=0)
        edge_index2 = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        x_hat_1 = self.conv_embed_1(x, edge_index2)
        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)
        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)


class GRCN(MultimodalRecommender):
    def __init__(self, config, n_users, n_items, norm_adj=None,
                 train_user_idx=None, train_item_idx=None,
                 v_feat=None, t_feat=None):
        super().__init__(config, n_users, n_items, norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        if not self.has_modalities():
            raise ValueError("GRCN requires at least one modality")
        num_user, num_item = self.n_users, self.n_items
        dim_x = self.embedding_size
        dim_C = int(config.get("latent_embedding", 64))
        num_routing = int(config.get("n_layers", 2))
        self.aggr_mode, self.weight_mode, self.fusion_mode = "add", "confid", "concat"
        has_act, has_norm = False, True
        self.reg_weight = float(config.get("reg_weight", 1e-3))
        self.dropout = float(config.get("dropout", 0.0))
        self.register_buffer("bpr_w", torch.tensor([[1.0], [-1.0]]), persistent=False)

        u = torch.as_tensor(np.asarray(train_user_idx), dtype=torch.long)
        it = torch.as_tensor(np.asarray(train_item_idx), dtype=torch.long) + num_user
        self.register_buffer("edge_index", torch.stack([u, it], dim=0), persistent=False)

        self.pruning = True
        self.id_gcn = EGCN(num_user, num_item, dim_x, self.aggr_mode, has_act, has_norm)
        num_model = 0
        if self.v_feat is not None:
            self.v_gcn = CGCN(self.v_feat, num_user, num_item, dim_C, self.aggr_mode, num_routing, has_act, has_norm)
            num_model += 1
        if self.t_feat is not None:
            self.t_gcn = CGCN(self.t_feat, num_user, num_item, dim_C, self.aggr_mode, num_routing, has_act, has_norm)
            num_model += 1
        self.model_specific_conf = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, num_model))))
        self.register_buffer("result", nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))), persistent=False)

    def _drop(self, edge_index):
        if self.dropout <= 0 or not self.training:
            return edge_index
        mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropout
        return edge_index[:, mask]

    def forward(self):
        weight, content_rep = None, None
        edge_index = self._drop(self.edge_index)
        if self.v_feat is not None:
            v_rep, weight_v = self.v_gcn(edge_index)
            weight, content_rep = weight_v, v_rep
        if self.t_feat is not None:
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight, content_rep = weight_t, t_rep
            else:
                content_rep = torch.cat((content_rep, t_rep), dim=1)
                weight = torch.cat((weight, weight_t), dim=1)
        if self.weight_mode == "confid":
            confidence = torch.cat((self.model_specific_conf[edge_index[0]],
                                    self.model_specific_conf[edge_index[1]]), dim=0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)
        if self.pruning:
            weight = torch.relu(weight)
        id_rep = self.id_gcn(edge_index, weight)
        representation = torch.cat((id_rep, content_rep), dim=1)
        self.result = representation
        return representation

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"] + self.n_users
        neg = interaction["neg_item"] + self.n_users
        user_tensor = users.repeat_interleave(2)
        item_tensor = torch.stack((pos, neg)).t().contiguous().view(-1)
        out = self.forward()
        score = torch.sum(out[user_tensor] * out[item_tensor], dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.bpr_w))))
        reg = (self.id_gcn.id_embedding[user_tensor] ** 2 + self.id_gcn.id_embedding[item_tensor] ** 2).mean()
        rc = out.new_zeros(())
        if self.v_feat is not None:
            rc = rc + (self.v_gcn.preference[user_tensor] ** 2).mean()
        if self.t_feat is not None:
            rc = rc + (self.t_gcn.preference[user_tensor] ** 2).mean()
        return loss + self.reg_weight * (reg + rc)

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        self.forward()
        users = interaction["user"]
        u = self.result[: self.n_users][users]
        return u @ self.result[self.n_users:].t()

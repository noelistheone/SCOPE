"""DRAGON (ECAI 2023) — ported faithfully into the SCOPE framework.

Dynamic and multi-view GRAph cONtrastive (DRAGON): a dual-graph multimodal
recommender that learns a user-user co-occurrence graph alongside the item-item
modality (KNN) graph. Reference: Zhou et al., ECAI 2023; official code in the
MMRec collection (https://github.com/enoche/MMRec).

The algorithm (per-modality GCN over the bipartite graph with feature dropout,
multi-modal item-item KNN aggregation, the User_Graph_sample top-k user-neighbour
aggregation, the 'cat' weighted modality construction, the BPR + preference-reg
loss, and full_sort_predict) is copied VERBATIM from the official implementation.

Only the framework glue differs:
  * MultimodalRecommender base (self.v_feat / self.t_feat / self.n_users /
    self.n_items + dict-form interactions) — same protocol/split/eval as every
    other table baseline.
  * The bipartite edge_index and the per-modality feature-dropout edge sets are
    rebuilt with DRAGON's own pack_edge_index / drop logic from the train edges
    (identical to the official code), registered as non-persistent buffers.
  * mm_adj (item-item KNN Laplacian) is built in __init__ from the frozen
    features (verbatim get_knn_adj_mat / compute_normalized_laplacian) instead
    of being torch.load-ed from a cache file.
  * user_graph_dict (per-user top-k user neighbours + co-occurrence weights) is
    NOT np.load-ed from an external .npy and NOT built with the O(n^2) Python
    double loop. It is built VECTORIZED from the interaction matrix as
    C_uu = R @ R^T (sparse), diagonal zeroed, top-k per row — producing exactly
    the {user_idx: [neighbour_idx_list, weight_list]} structure the original
    dualgnn-gen-u-u-matrix.py produced and that topk_sample consumes.
  * runtime .to(self.device) in the GCN submodule is replaced by deriving the
    device from the module's own parameters (the Trainer moves the model once).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from src.common.abstract_recommender import MultimodalRecommender


class DRAGON(MultimodalRecommender):
    def __init__(self, config, n_users, n_items, norm_adj=None,
                 v_feat=None, t_feat=None, train_user_idx=None, train_item_idx=None):
        super().__init__(config, n_users, n_items, norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = int(config.get("train_batch_size", 2048))  # not used
        dim_x = int(config.get("embedding_size", 64))
        self.feat_embed_dim = int(config.get("feat_embed_dim", 64))
        self.n_layers = int(config.get("n_mm_layers", 1))
        self.knn_k = int(config.get("knn_k", 10))
        self.mm_image_weight = float(config.get("mm_image_weight", 0.1))
        has_id = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = int(config.get("user_topk", 40))
        # aggr_mode in the source yaml is a list ['add']; accept either str or list.
        aggr_mode = config.get("aggr_mode", "add")
        if isinstance(aggr_mode, (list, tuple)):
            aggr_mode = aggr_mode[0]
        self.aggr_mode = aggr_mode
        self.user_aggr_mode = "softmax"
        self.num_layer = 1
        self.cold_start = 0
        self.dataset = config.get("dataset", "")
        # self.construction = 'weighted_max'
        self.construction = "cat"
        self.reg_weight = float(config.get("reg_weight", 1e-3))
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        # mm_adj is registered as a buffer below (after KNN build); no None placeholder
        # here, to avoid clobbering the buffer slot.

        # ---- raw user-item interaction (scipy coo) from the train edges ----
        ui = sp.coo_matrix(
            (np.ones(train_user_idx.numel(), dtype=np.float32),
             (train_user_idx.numpy(), train_item_idx.numpy())),
            shape=(n_users, n_items)).astype(np.float32)
        self.interaction_matrix = ui

        # ---- per-user top-k user-neighbour graph: vectorized C_uu = R @ R^T ----
        # Equivalent to dualgnn-gen-u-u-matrix.py (co-occurrence counts, diagonal
        # zeroed, top-k per row) — without the O(n^2) Python double loop / .npy.
        self.user_graph_dict = self._build_user_graph_dict(ui, cap=200)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat.clone(), freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat.clone(), freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # ---- item-item multimodal KNN Laplacian (verbatim get_knn_adj_mat) ----
        mm_adj = None
        image_adj = None
        text_adj = None
        if self.v_feat is not None:
            _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            mm_adj = image_adj
        if self.t_feat is not None:
            _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj
            del image_adj
        self.register_buffer("mm_adj", mm_adj.coalesce(), persistent=False)

        # ---- pack interactions into edge_index (verbatim pack_edge_index) ----
        train_interactions = ui  # already coo float32
        edge_index = self.pack_edge_index(train_interactions)  # [E, 2] numpy
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        ei = torch.cat((ei, ei[[1, 0]]), dim=1)
        self.register_buffer("edge_index", ei, persistent=False)

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        # ---- per-modality feature-dropout edge sets (verbatim logic, vectorized count) ----
        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        dropv_set = set(self.dropv_node_idx.tolist())
        dropt_set = set(self.dropt_node_idx.tolist())
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in dropv_set else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in dropt_set else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        eidv = torch.tensor(edge_index_dropv).t().contiguous()
        eidt = torch.tensor(edge_index_dropt).t().contiguous()
        eidv = torch.cat((eidv, eidv[[1, 0]]), dim=1)
        eidt = torch.cat((eidt, eidt[[1, 0]]), dim=1)
        self.register_buffer("edge_index_dropv", eidv, persistent=False)
        self.register_buffer("edge_index_dropt", eidt, persistent=False)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.register_buffer("v_drop_ze",
                                 torch.zeros(len(self.dropv_node_idx), self.v_feat.size(1)),
                                 persistent=False)
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate,
                             dim_latent=64, features=self.v_feat)  # 256)
        if self.t_feat is not None:
            self.register_buffer("t_drop_ze",
                                 torch.zeros(len(self.dropt_node_idx), self.t_feat.size(1)),
                                 persistent=False)
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate,
                             dim_latent=64, features=self.t_feat)

        self.user_graph = User_Graph_sample(num_user, "add", self.dim_latent)

        # result_embed: source declares it as a Parameter then overwrites it with a
        # plain tensor inside forward(); keep it as a registered buffer placeholder.
        self.register_buffer(
            "result_embed",
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x),
                                                dtype=torch.float32)),
            persistent=False)

        # filled by pre_epoch_processing before the first training epoch
        self.epoch_user_graph = None
        self.user_weight_matrix = None

    # ------------------------------------------------------------------ #
    #  vectorized user-user co-occurrence graph (replaces the .npy / loop)
    # ------------------------------------------------------------------ #
    def _build_user_graph_dict(self, ui_coo, cap=200):
        """Build {user: [neighbour_idx_list, weight_list]} from C_uu = R @ R^T.

        Mirrors dualgnn-gen-u-u-matrix.py: weights are integer co-occurrence
        counts (number of shared items), the self-loop (diagonal) is removed,
        and each user keeps its top-`cap` (=200) neighbours sorted by weight.
        topk_sample(self.k) later slices the first k of these per user.
        """
        R = ui_coo.tocsr().astype(np.float32)
        C = (R @ R.T).tocsr()          # user-user co-occurrence (counts)
        C.setdiag(0)                   # drop self-loops (matches head<rear loop)
        C.eliminate_zeros()
        n = C.shape[0]
        user_graph_dict = {}
        indptr, indices, data = C.indptr, C.indices, C.data
        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            cols = indices[start:end]
            vals = data[start:end]
            if cols.size == 0:
                user_graph_dict[i] = [[], []]
                continue
            kk = int(min(cap, cols.size))
            # top-kk by weight, descending (torch.topk matches the source semantics)
            order = np.argsort(-vals, kind="stable")[:kk]
            user_graph_dict[i] = [cols[order].tolist(), vals[order].astype(np.int64).tolist()]
        return user_graph_dict

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0])
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def pre_epoch_processing(self, epoch=0):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.result_embed.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([E, 2])
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes = pos_item_nodes + self.n_users
        neg_item_nodes = neg_item_nodes + self.n_users
        representation = None

        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
            representation = self.v_rep
        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
            if representation is None:
                representation = self.t_rep
            else:
                if self.construction == "cat":
                    representation = torch.cat((self.v_rep, self.t_rep), dim=1)
                else:
                    representation += self.t_rep

        if self.construction == "weighted_sum":
            if self.v_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                                        self.weight_u)
            user_rep = torch.squeeze(user_rep)

        if self.construction == "weighted_max":
            # pdb.set_trace()
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1, 2) * user_rep
            user_rep = torch.max(user_rep, dim=2).values
        if self.construction == "cat":
            # pdb.set_trace()
            if self.v_rep is not None:
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
                user_rep = self.weight_u.transpose(1, 2) * user_rep
                user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        item_rep = representation[self.num_user:]

        # ################################# multi-modal information aggregation
        h = item_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)
        user_rep = user_rep + h_u1
        item_rep = item_rep + h
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]
        user = users
        pos_scores, neg_scores = self.forward((users, pos, neg))
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        if self.construction == "weighted_sum":
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == "cat":
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        elif self.construction == "cat_mlp":
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()
        return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction["user"]
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[user, :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == "softmax":
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight, dtype=torch.float32), dim=0)  # softmax
                if self.user_aggr_mode == "mean":
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == "softmax":
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight, dtype=torch.float32), dim=0)  # softmax
            if self.user_aggr_mode == "mean":
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = h + x + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr="add", **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == "add":
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return "{}({},{})".format(self.__class__.__name__, self.in_channels, self.out_channels)

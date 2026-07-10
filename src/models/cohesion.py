"""COHESION (SIGIR 2025) — ported faithfully into the SCOPE framework.

Reference: COHESION (SIGIR 2025), official author implementation. Reported
R@20 ~0.1052 / 0.1137 / 0.0983 (Baby / Sports / Clothing).

The algorithm — the per-modality `GCNLayer` (geometric-mean modality/ID fusion +
cosine-reweighted layer aggregation), the item-item kNN multimodal graph (`mm_adj`),
the user-user graph propagation (`User_Graph_sample` + `topk_sample`), the adaptive
three-modality weighting, and the BPR + preference-reg loss — is copied VERBATIM from
the official implementation.

Only the framework glue differs:
  * subclass MultimodalRecommender (self.v_feat / self.t_feat / self.n_users /
    self.n_items / self.embedding_size + dict-form interactions);
  * the raw user-item interaction is rebuilt from the train edges passed by the
    orchestrator (scipy coo), and COHESION's own get_norm_adj_mat / get_edge_info /
    get_knn_adj_mat are run on it VERBATIM;
  * the `user_graph_dict` (the source np.load's a precomputed file produced by
    preprocessing/dualgnn-gen-u-u-matrix.py = per-user top-k user neighbours by
    shared-item co-occurrence count) is built VECTORIZED in __init__ from
    C_uu = R @ R^T (sparse), diagonal zeroed, top-min(nnz, 200) per row — identical
    to the preprocessing script, no external .npy and no O(n^2) Python double loop;
  * graphs are built on CPU and register_buffer'd so the trainer's .to(device)
    moves them; `.cuda()`/`.to(self.device)` in __init__ are dropped;
  * torch.sparse.FloatTensor -> torch.sparse_coo_tensor; np.float/int modernized.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender


class COHESION(MultimodalRecommender):
    def __init__(self, config, n_users, n_items, norm_adj=None,
                 v_feat=None, t_feat=None, train_user_idx=None, train_item_idx=None):
        super().__init__(config, n_users, n_items, norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        self.setup_parameters(config, train_user_idx, train_item_idx)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def setup_parameters(self, config, train_user_idx, train_item_idx):
        # Initialize parameters and embeddings (VERBATIM from source.setup_parameters)
        self.n_nodes = self.n_users + self.n_items
        self.dim = int(config.get("embedding_size", 64))
        self.feat_embed_dim = int(config.get("feat_embed_dim", 64))
        self.n_layers = int(config.get("n_mm_layers", 1))
        self.knn_k = int(config.get("knn_k", 10))
        self.mm_image_weight = float(config.get("mm_image_weight", 0.1))
        self.dropout = float(config.get("dropout", 0.0))
        self.k = 40
        self.num_layer = int(config.get("num_layer", 2))
        self.reg_weight = float(config.get("reg_weight", 0.001))
        self.drop_rate = 0.1
        self.v_rep, self.t_rep, self.id_rep = None, None, None
        self.v_preference, self.t_preference, self.id_preference = None, None, None
        self.dim_latent = 64
        self.mm_adj = None

        # ---- user_graph_dict: built VECTORIZED from R @ R^T (replaces source np.load) ----
        # Raw user-item interaction matrix R (scipy coo) from the passed train edges.
        ui = sp.coo_matrix(
            (np.ones(train_user_idx.numel(), dtype=np.float32),
             (train_user_idx.numpy(), train_item_idx.numpy())),
            shape=(self.n_users, self.n_items)).astype(np.float32)
        self.train_interactions = ui
        self.user_graph_dict = self._build_user_graph_dict(ui)

        # ---- Load or generate mm_adj (source caches to disk; we build in-memory) ----
        # Build the embedding tables used elsewhere first, then the kNN modality graph.
        self.initialize_embeddings(config)
        self.mm_adj = self.generate_mm_adj()

        # ---- Construct interaction edge (VERBATIM, CPU; registered as buffers) ----
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.register_buffer("edge_indices_buf", self.edge_indices, persistent=False)
        self.register_buffer("edge_values_buf", self.edge_values, persistent=False)

        # ---- Create normalized adjacency matrices (VERBATIM, CPU) ----
        norm_adj_mat = self.get_norm_adj_mat()
        self.register_buffer("norm_adj_buf", norm_adj_mat, persistent=False)
        self.masked_adj = norm_adj_mat

        # ---- Create GCN layers for different modalities ----
        self.create_gcn_layers()

        # ---- Create user graph and result embeddings ----
        self.user_graph = User_Graph_sample(self.n_users, self.dim_latent)
        # Source does `self.result_embed = nn.Parameter(...).to(device)`; the .to() on a
        # Parameter yields a plain (non-leaf) tensor, so it is NOT a trainable parameter —
        # it is just a scratch attribute that forward() overwrites each call. We replicate
        # that as a plain tensor attribute (object.__setattr__ to bypass nn.Module's
        # Parameter slot, since forward reassigns it with a non-Parameter tensor).
        object.__setattr__(
            self, "result_embed",
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_nodes, self.dim), dtype=torch.float32)).detach()
        )

    # ------------------------------------------------------------------ #
    # vectorized user_graph_dict  (== preprocessing/dualgnn-gen-u-u-matrix.py)
    # ------------------------------------------------------------------ #
    def _build_user_graph_dict(self, ui_coo):
        """Build {user: [neighbor_idx_list, shared-item-count_list]} vectorized.

        The preprocessing script sets user_graph_matrix[i][j] = |items(i) ∩ items(j)|
        for i != j, then for each user keeps the top-min(nnz, 200) neighbours by that
        count. The shared-item-count matrix is exactly R @ R^T (off-diagonal). We
        compute it as a sparse matmul, zero the diagonal, and take a batched top-k per
        row. Weights are the raw counts (matching the source — topk_sample softmaxes
        them later).
        """
        R = ui_coo.tocsr().astype(np.float32)
        Cuu = (R @ R.transpose()).tocsr()           # [n_users, n_users] shared-item counts
        Cuu.setdiag(0.0)                              # source loop never sets the diagonal
        Cuu.eliminate_zeros()

        cap = 200                                     # source caps at 200 neighbours per user
        n_users = self.n_users
        user_graph_dict = {}
        # Row-wise top-k: iterate over rows but only over their non-zeros (sparse,
        # cheap), then sort/truncate — no n^2 work, no dense [n_users, n_users] tensor.
        indptr, indices, data = Cuu.indptr, Cuu.indices, Cuu.data
        for i in range(n_users):
            start, end = indptr[i], indptr[i + 1]
            if end == start:
                user_graph_dict[i] = [[], []]
                continue
            nbr = indices[start:end]
            wts = data[start:end]
            nnz = nbr.shape[0]
            kk = min(nnz, cap)
            # top-kk by weight, descending (torch.topk semantics in the source)
            if kk < nnz:
                top = np.argpartition(wts, nnz - kk)[nnz - kk:]
                order = top[np.argsort(-wts[top], kind="stable")]
            else:
                order = np.argsort(-wts, kind="stable")
            user_graph_dict[i] = [nbr[order].tolist(), wts[order].astype(np.float32).tolist()]
        return user_graph_dict

    # ------------------------------------------------------------------ #
    # mm_adj  (VERBATIM source generate_mm_adj / get_knn_adj_mat)
    # ------------------------------------------------------------------ #
    def generate_mm_adj(self):
        image_adj, text_adj = None, None
        if self.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        if self.t_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())

        if self.v_feat is not None and self.t_feat is not None:
            mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj, image_adj
        else:
            mm_adj = image_adj if image_adj is not None else text_adj

        # register so the trainer moves it to device; keep self.mm_adj as the attr forward uses
        self.register_buffer("mm_adj_buf", mm_adj.coalesce(), persistent=False)
        return self.mm_adj_buf

    def initialize_embeddings(self, config):
        # VERBATIM source.initialize_embeddings (minus the disk path it never uses here)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        self.weight_u = nn.Parameter(
            nn.init.xavier_normal_(
                torch.tensor(np.random.randn(self.n_users, 2, 1), dtype=torch.float32, requires_grad=True))
        )
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

    def create_gcn_layers(self):
        # VERBATIM source.create_gcn_layers (device -> None; layers built on CPU)
        if self.v_feat is not None:
            self.v_gcn = GCNLayer(self.n_users, self.n_items, num_layer=self.num_layer, dim_latent=64,
                                  device=None, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCNLayer(self.n_users, self.n_items, num_layer=self.num_layer, dim_latent=64,
                                  device=None, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1))
        self.id_gcn = GCNLayer(self.n_users, self.n_items, num_layer=self.num_layer, dim_latent=64,
                               device=None, features=self.id_feat)

    # ------------------------------------------------------------------ #
    # bipartite norm-adj / edges  (VERBATIM source)
    # ------------------------------------------------------------------ #
    def get_edge_info(self):
        rows = torch.from_numpy(self.train_interactions.row)
        cols = torch.from_numpy(self.train_interactions.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]).float(), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_norm_adj_mat(self):
        # NOTE: the source populates a dok_matrix via `A._update(data_dict)`. In
        # scipy>=1.14 dok_matrix no longer subclasses dict and `_update`/`.update`/
        # raw `dict.update(A, ...)` all FAIL SILENTLY (leave A all-zero -> the U-I
        # GCN propagates an empty adjacency -> the model collapses to its layer-0
        # ego embeddings, i.e. ~LightGCN). We populate the SAME symmetric bipartite
        # A via lil assignment, which yields the identical D^{-1/2} A D^{-1/2}.
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32).tolil()
        R = self.train_interactions.tolil()
        A[:self.n_users, self.n_users:] = R
        A[self.n_users:, :self.n_users] = R.T
        A = A.todok()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes))).coalesce()

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
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]).float(), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    # ------------------------------------------------------------------ #
    # per-epoch  (VERBATIM source.pre_epoch_processing; epoch arg added for our trainer)
    # ------------------------------------------------------------------ #
    def pre_epoch_processing(self, epoch=0):
        device = self.edge_values_buf.device
        epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        # Source indexes features[epoch_user_graph] relying on implicit nested-list ->
        # tensor coercion (dropped in newer torch); make it an explicit [n_users, k] LongTensor.
        self.epoch_user_graph = torch.as_tensor(epoch_user_graph, dtype=torch.long, device=device)
        self.user_weight_matrix = self.user_weight_matrix.to(device)
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj_buf
            return
        degree_len = int(self.edge_values_buf.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values_buf, degree_len)
        # random sample
        keep_indices = self.edge_indices_buf[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices = keep_indices.clone()
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj_buf.shape).coalesce().to(device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    # ------------------------------------------------------------------ #
    # forward / propagation  (VERBATIM source; dict interaction adapted)
    # ------------------------------------------------------------------ #
    def forward(self, interaction):
        user_nodes = interaction["user"]
        pos_item_nodes = interaction["pos_item"] + self.n_users
        neg_item_nodes = interaction["neg_item"] + self.n_users

        # get representation and id_rep_data
        representation, id_rep_data = self.build_representation()

        # get user and item representation
        user_rep, item_rep = self.process_user_item_representation(representation, id_rep_data)

        # get user and item tensor
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]

        # Adaptively optimize the weight of the three modalities
        adaptive_weight = self.adaptive_optimization(user_tensor, pos_item_tensor, neg_item_tensor)
        pos_scores = torch.sum(user_tensor * pos_item_tensor * adaptive_weight, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor * adaptive_weight, dim=1)
        return pos_scores, neg_scores

    def build_representation(self):
        id_rep, id_preference = self.id_gcn(self.id_feat, self.id_feat, self.masked_adj)
        id_rep_data = id_rep.data

        representation = id_rep_data

        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.v_feat, self.id_feat, self.masked_adj)
            representation = torch.cat((id_rep_data, self.v_rep), dim=1)

        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.t_feat, self.id_feat, self.masked_adj)
            representation = torch.cat((id_rep_data, self.t_rep) if representation is None
                                       else (id_rep_data, self.v_rep, self.t_rep), dim=1)

        self.v_rep = torch.unsqueeze(self.v_rep, 2)
        self.t_rep = torch.unsqueeze(self.t_rep, 2)
        id_rep_data = torch.unsqueeze(id_rep_data, 2)

        return representation, id_rep_data

    def process_user_item_representation(self, representation, id_rep_data):
        user_rep, item_rep = None, None

        if self.v_rep is not None and self.t_rep is not None:
            user_rep = torch.cat((id_rep_data[:self.n_users], self.v_rep[:self.n_users], self.t_rep[:self.n_users]),
                                 dim=2)
            user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)

        item_rep = representation[self.n_users:]

        h_i = item_rep
        for i in range(self.n_layers):
            h_i = torch.sparse.mm(self.mm_adj_buf, h_i)
        h_u = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)

        user_rep = user_rep + h_u
        item_rep = item_rep + h_i

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        return user_rep, item_rep

    def adaptive_optimization(self, user_e, pos_e, neg_e):
        pos_score_ = torch.mul(user_e, pos_e).view(-1, 3, self.dim_latent).sum(dim=-1)
        neg_score_ = torch.mul(user_e, neg_e).view(-1, 3, self.dim_latent).sum(dim=-1)
        modality_indicator = 1 - (pos_score_ - neg_score_).softmax(-1).detach()

        adaptive_weight = torch.tile(modality_indicator.view(-1, 3, 1), [1, 1, self.dim_latent])
        adaptive_weight = adaptive_weight.view(-1, 3 * self.dim_latent)

        return adaptive_weight

    def calculate_loss(self, interaction):
        user = interaction["user"]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        # Source reads the stale self.result_embed from the last train forward; to
        # produce correct eval scores we recompute the SAME representation (the
        # identical forward propagation) then score. Algorithm is unchanged.
        representation, id_rep_data = self.build_representation()
        user_rep, item_rep = self.process_user_item_representation(representation, id_rep_data)
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction["user"], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    # ------------------------------------------------------------------ #
    # topk_sample  (VERBATIM source)
    # ------------------------------------------------------------------ #
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

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


class User_Graph_sample(torch.nn.Module):
    """
        user-user graph
    """

    def __init__(self, num_user, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCNLayer(torch.nn.Module):
    def __init__(self, num_user, num_item, num_layer, dim_latent=None, device=None, features=None):
        super(GCNLayer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features, id_embd, adj):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        temp_features = torch.abs(
            ((torch.mul(id_embd, id_embd) + torch.mul(temp_features, temp_features)) / 2) + 1e-8).sqrt()
        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)
        ego_embeddings = x
        all_embeddings = ego_embeddings
        embeddings_layers = [all_embeddings]

        for layer_idx in range(self.num_layer):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)

        return ui_all_embeddings, self.preference


class BGCNLayer(torch.nn.Module):
    """
        basic layer-refined GCN
    """

    def __init__(self, num_user, num_item, num_layer, dim_latent=None, device=None, features=None):
        super(BGCNLayer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features, id_embd, adj):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)
        ego_embeddings = x
        all_embeddings = ego_embeddings
        embeddings_layers = [all_embeddings]

        for layer_idx in range(self.num_layer):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)

        return ui_all_embeddings, self.preference

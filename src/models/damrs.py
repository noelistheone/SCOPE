"""DA-MRS (KDD 2024) — ported faithfully into the SCOPE framework.

Denoising and Aligning Multi-modal Recommender System.
Reference: Xv et al., KDD 2024; official MMRec-style implementation
(https://github.com/enoche/MMRec).

A LightGCN user-item backbone is combined with three frozen item-item graphs
(visual-knn, text-knn, and a "session" co-occurrence graph built from the
interaction matrix) propagated over the item-id embeddings; a denoising BPR loss
re-weights positives/negatives per-sample using cross-modal preference variance,
plus a neighbor-discrimination contrastive loss and a symmetric KL alignment loss.

The algorithm (get_knn_adj_mat, compute_normalized_laplacian, get_session_adj,
label_prediction, generate_pesudo_labels, neighbor_discrimination, KL,
get_weight_modal, bpr_loss, forward, calculate_loss, full_sort_predict) is copied
VERBATIM from the official implementation. Only the framework glue differs:

  * MultimodalRecommender base (self.v_feat / self.t_feat / self.n_users /
    self.n_items), dict-form interactions, config.get for HP.
  * The bipartite norm-adj is rebuilt with DA-MRS's own get_norm_adj_mat from the
    train edges (verbatim), not reused from the framework.
  * The per-item top-k neighbor graph (the source np.load's item_graph_dict2.npy)
    is built VECTORIZED inside __init__ from item-item co-occurrence C_ii = R^T @ R
    (shared-user counts, zero diagonal) and stored in the exact
    {item: [neighbor_idx_list, weight_list]} structure get_session_adj consumes —
    no external .npy and no O(n^2) Python double-loop. This reproduces the official
    preprocessing in build_iib_graph.py EXACTLY: (a) only item pairs with co-occurrence
    >= 2 shared users are kept (inter_len >= 2 in gen_item_matrix), entries < 2 are
    zeroed BEFORE selection; (b) top-k is top_k=2 (README: `--topk=2`,
    item_graph_dict2.npy), NOT knn_k=10; if an item has <= top_k nonzero neighbors all
    of them are kept. Weights are the raw shared-user counts.
  * Graphs are built on CPU and registered as non-persistent buffers so the trainer
    moves them to device; .cuda()/.to(self.device) in __init__ are dropped.
"""
from __future__ import annotations

import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender


class DAMRS(MultimodalRecommender):
    def __init__(self, config, n_users, n_items, norm_adj=None,
                 v_feat=None, t_feat=None, train_user_idx=None, train_item_idx=None):
        super().__init__(config, n_users, n_items, norm_adj=norm_adj,
                         v_feat=v_feat, t_feat=t_feat)

        assert self.v_feat is not None and self.t_feat is not None, \
            "DAMRS requires both visual and text features"

        self.embedding_dim = int(config.get("embedding_size", 64))

        self.lambda_coeff = float(config.get("lambda_coeff", 0.9))
        self.cf_model = config.get("cf_model", "lightgcn")

        self.knn_k = int(config.get("knn_k", 10))
        self.n_layers = int(config.get("n_mm_layers", 1))

        self.n_ui_layers = int(config.get("n_ui_layers", 2))
        self.reg_weight = float(config.get("reg_weight", 0.0))
        self.kl_weight = float(config.get("kl_weight", 1.0))
        self.neighbor_weight = float(config.get("neighbor_weight", 0.001))
        # top-k item neighbors per row for the co-occurrence "session" graph.
        # Official build (build_iib_graph.py / README): `--topk=2` -> item_graph_dict2.npy
        # for baby/sports/clothing. The "2" is top_k, NOT knn_k(=10).
        self.item_graph_k = int(config.get("item_graph_k", 2))
        # minimum shared-user co-occurrence to keep an item-item edge (gen_item_matrix
        # only writes a count when len(intersection) >= 2).
        self.item_graph_min_cooc = int(config.get("item_graph_min_cooc", 2))
        self.build_item_graph = True

        self.n_nodes = self.n_users + self.n_items

        # ---- build the raw user-item interaction (scipy coo) from train edges ----
        ui = sp.coo_matrix(
            (np.ones(train_user_idx.numel(), dtype=np.float32),
             (train_user_idx.numpy(), train_item_idx.numpy())),
            shape=(self.n_users, self.n_items)).astype(np.float32)
        self.interaction_matrix = ui
        # get_norm_adj_mat consumes self.interaction_matrix (verbatim from source)
        norm_adj_mat = self.get_norm_adj_mat()
        self.register_buffer("norm_adj", norm_adj_mat, persistent=False)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        image_adj, text_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(),
                                                   self.text_embedding.weight.detach())
        self.register_buffer("image_adj", image_adj, persistent=False)
        self.register_buffer("text_adj", text_adj, persistent=False)

        # item_graph_dict: built vectorized from item-item co-occurrence (R^T @ R),
        # replacing the source's np.load(item_graph_dict_file) + O(n^2) preprocessing.
        self.item_graph_dict = self._build_item_graph_dict(ui, self.item_graph_k)

        __, session_adj = self.get_session_adj()
        self.register_buffer("session_adj", session_adj, persistent=False)

    # ------------------------------------------------------------------
    # vectorized item-item co-occurrence graph (replaces external .npy)
    # ------------------------------------------------------------------
    def _build_item_graph_dict(self, ui, k):
        """Build {item_idx: [neighbor_idx_list, weight_list]} EXACTLY as DA-MRS's
        official build_iib_graph.py (item_graph_dict2.npy), but vectorized in-memory.

        Official gen_item_matrix(): item_graph_matrix[a,b] = #shared users only when
        that count is >= 2 (inter_len >= 2), else 0. Then, per item i, item_num[i] =
        #nonzero neighbors; if item_num[i] <= top_k take topk(row, item_num[i]) (all
        of them), else topk(row, top_k). Weights are the raw shared-user counts.

        We reproduce that with C = R^T @ R (shared-user counts), zero diagonal, then
        threshold entries < item_graph_min_cooc (=2) to zero BEFORE the top-k. This
        matches the official graph's edge set and weights exactly (the topk tie-break
        on equal counts can differ in index order, but the value set is identical)."""
        R = ui.tocsr()
        C = (R.T @ R).tocsr()           # [n_items, n_items], C[a,b] = #shared users
        C.setdiag(0)
        C.eliminate_zeros()
        n_items = self.n_items
        thr = self.item_graph_min_cooc
        item_graph_dict = {}
        # batch over rows so we never materialize a dense [n_items, n_items]
        batch = 2048
        kk = min(k, n_items)
        for start in range(0, n_items, batch):
            end = min(start + batch, n_items)
            block = torch.from_numpy(C[start:end].toarray().astype(np.float32))  # [b, n_items]
            # official: only co-occurrences >= thr (=2) are written into the matrix;
            # smaller counts are treated as no edge.
            block[block < thr] = 0
            vals, inds = torch.topk(block, kk, dim=-1)
            for r in range(block.shape[0]):
                i = start + r
                v = vals[r]
                nz = v > 0
                if nz.any():
                    neigh = inds[r][nz].tolist()
                    wts = v[nz].tolist()
                else:
                    neigh = []
                    wts = []
                item_graph_dict[i] = [neigh, wts]
        return item_graph_dict

    # ------------------------------------------------------------------
    # ---- everything below is VERBATIM from the source (glue only) ----
    # ------------------------------------------------------------------
    def get_knn_adj_mat(self, v_embeddings, t_embeddings):
        v_context_norm = v_embeddings.div(torch.norm(v_embeddings, p=2, dim=-1, keepdim=True))
        v_sim = torch.mm(v_context_norm, v_context_norm.transpose(1, 0))

        t_context_norm = t_embeddings.div(torch.norm(t_embeddings, p=2, dim=-1, keepdim=True))
        t_sim = torch.mm(t_context_norm, t_context_norm.transpose(1, 0))

        mask_v = v_sim < v_sim.mean()
        mask_t = t_sim < t_sim.mean()

        t_sim[mask_v] = 0
        v_sim[mask_t] = 0
        t_sim[mask_t] = 0
        v_sim[mask_v] = 0

        index_x = []
        index_v = []
        index_t = []

        all_items = np.arange(self.n_items).tolist()

        def _random():
            rd_id = random.sample(all_items, 9)  # [0]
            return rd_id

        for i in range(self.n_items):
            item_num = len(torch.nonzero(t_sim[i]))
            if item_num <= self.knn_k:
                _, v_knn_ind = torch.topk(v_sim[i], item_num)
                _, t_knn_ind = torch.topk(t_sim[i], item_num)
            else:
                _, v_knn_ind = torch.topk(v_sim[i], self.knn_k)
                _, t_knn_ind = torch.topk(t_sim[i], self.knn_k)

            index_x.append(torch.ones_like(v_knn_ind) * i)
            index_v.append(v_knn_ind)
            index_t.append(t_knn_ind)

        index_x = torch.cat(index_x, dim=0)
        index_v = torch.cat(index_v, dim=0)
        index_t = torch.cat(index_t, dim=0)

        adj_size = (self.n_items, self.n_items)
        del v_sim, t_sim

        v_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_v)), 0)
        t_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_t)), 0)
        # norm
        return self.compute_normalized_laplacian(v_indices, adj_size), self.compute_normalized_laplacian(t_indices,
                                                                                                         adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size).coalesce()

    def get_session_adj(self):
        index_x = []
        index_y = []
        values = []
        for i in range(self.n_items):
            index_x.append(i)
            index_y.append(i)
            values.append(1)
            if i in self.item_graph_dict.keys():
                item_graph_sample = self.item_graph_dict[i][0]
                item_graph_weight = self.item_graph_dict[i][1]

                for j in range(len(item_graph_sample)):
                    index_x.append(i)
                    index_y.append(item_graph_sample[j])
                    values.append(item_graph_weight[j])
        index_x = torch.tensor(index_x, dtype=torch.long)
        index_y = torch.tensor(index_y, dtype=torch.long)
        indices = torch.stack((index_x, index_y), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, (self.n_items, self.n_items))

    def label_prediction(self, emb, aug_emb):
        n_emb = F.normalize(emb, dim=1)
        n_aug_emb = F.normalize(aug_emb, dim=1)
        prob = torch.mm(n_emb, n_aug_emb.transpose(0, 1))
        prob = F.softmax(prob, dim=1)
        del n_emb, n_aug_emb
        return prob

    def generate_pesudo_labels(self, prob1, prob2, prob3):
        positive = prob1 + prob2 + prob3 + prob3
        _, mm_pos_ind = torch.topk(positive, 10, dim=-1)
        prob = prob3.clone()
        prob.scatter_(1, mm_pos_ind, 0)
        _, single_pos_ind = torch.topk(prob, 10, dim=-1)
        return mm_pos_ind, single_pos_ind

    def neighbor_discrimination(self, mm_positive, s_positive, emb, aug_emb, temperature=0.2):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=2)

        n_aug_emb = F.normalize(aug_emb, dim=1)
        n_emb = F.normalize(emb, dim=1)

        mm_pos_emb = n_aug_emb[mm_positive]
        s_pos_emb = n_aug_emb[s_positive]

        emb2 = torch.reshape(n_emb, [-1, 1, self.embedding_dim])
        emb2 = torch.tile(emb2, [1, 10, 1])

        mm_pos_score = score(emb2, mm_pos_emb)
        s_pos_score = score(emb2, s_pos_emb)
        ttl_score = torch.matmul(n_emb, n_aug_emb.transpose(0, 1))

        mm_pos_score = torch.sum(torch.exp(mm_pos_score / temperature), dim=1)
        s_pos_score = torch.sum(torch.exp(s_pos_score / temperature), dim=1)
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)  # 1

        cl_loss = - torch.log(mm_pos_score / (ttl_score) + 10e-10) - torch.log(
            s_pos_score / (ttl_score - mm_pos_score) + 10e-10)
        return torch.mean(cl_loss)

    def KL(self, p1, p2):
        return p1 * torch.log(p1) - p1 * torch.log(p2) + \
               (1 - p1) * torch.log(1 - p1) - (1 - p1) * torch.log(1 - p2)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        dict.update(A, data_dict)
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

    def forward(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        del ego_embeddings, side_embeddings

        # text emb
        h_t = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_t = torch.sparse.mm(self.text_adj, h_t)

        # image emb
        h_v = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_v = torch.sparse.mm(self.image_adj, h_v)

        # session emb
        h_s = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_s = torch.sparse.mm(self.session_adj, h_s)

        return u_g_embeddings, i_g_embeddings, h_t, h_v, h_s

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos_items = interaction["pos_item"]
        neg_items = interaction["neg_item"]

        user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward()
        self.build_item_graph = False

        u_idx = torch.unique(users, return_inverse=True, sorted=False)
        i_idx = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        u_id = u_idx[0]
        i_id = i_idx[0]

        # text
        label_prediction_t = self.label_prediction(h_t[i_id], h_t)
        # visual
        label_prediction_v = self.label_prediction(h_v[i_id], h_v)
        # session
        label_prediction_s = self.label_prediction(h_s[i_id], h_s)

        mm_postive_s, s_postive_s = self.generate_pesudo_labels(label_prediction_t, label_prediction_v,
                                                                label_prediction_s)
        neighbor_dis_loss_1 = self.neighbor_discrimination(mm_postive_s, s_postive_s, h_s[i_id], h_s)

        mm_postive_v, s_postive_v = self.generate_pesudo_labels(label_prediction_t, label_prediction_s,
                                                                label_prediction_v)
        neighbor_dis_loss_2 = self.neighbor_discrimination(mm_postive_v, s_postive_v, h_v[i_id], h_v)

        mm_postive_t, s_postive_t = self.generate_pesudo_labels(label_prediction_v, label_prediction_s,
                                                                label_prediction_t)
        neighbor_dis_loss_3 = self.neighbor_discrimination(mm_postive_t, s_postive_t, h_t[i_id], h_t)

        neighbor_dis_loss = (neighbor_dis_loss_1 + neighbor_dis_loss_2 + neighbor_dis_loss_3) / 3.0

        n_u_g_embeddings = user_embeddings[u_id]
        it_embeddings = (h_t + h_s + h_v) / 3.0

        p_g = F.sigmoid(torch.matmul(n_u_g_embeddings, F.normalize(item_embeddings[i_id], dim=-1).transpose(0, 1)))
        p_t = F.sigmoid(torch.matmul(n_u_g_embeddings, F.normalize(it_embeddings[i_id], dim=-1).transpose(0, 1)))

        KL_loss = torch.mean(self.KL(p_g, p_t) + self.KL(p_t, p_g))

        p_weight, n_weight = self.get_weight_modal(users, pos_items, neg_items, user_embeddings, h_t, h_v, h_s)

        u_g_embeddings = user_embeddings[users]
        ia_embeddings = item_embeddings + (h_t + h_v + h_s) / 3.0
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, p_weight, n_weight)

        return batch_mf_loss + self.neighbor_weight * (neighbor_dis_loss) + KL_loss * self.kl_weight

    def full_sort_predict(self, interaction):
        user = interaction["user"]
        user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward()  #

        user_e = user_embeddings[user, :]
        i_embedding = (h_v + h_t + h_s) / 3.0
        all_item_e = item_embeddings + i_embedding
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def get_weight_modal(self, users, pos_items, neg_items, user_embeddings, h_t, h_v, h_s):
        u_g_embeddings = user_embeddings[users]

        p_t = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_t[pos_items], dim=-1)), dim=1)
        p_v = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_s[pos_items], dim=-1)), dim=1)
        p_s = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_v[pos_items], dim=-1)), dim=1)

        n_t = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_t[neg_items], dim=-1)), dim=1)
        n_v = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_s[neg_items], dim=-1)), dim=1)
        n_s = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_v[neg_items], dim=-1)), dim=1)

        p_tensor = F.sigmoid(torch.stack([p_t, p_v, p_s]))
        p_variance = torch.var(p_tensor, dim=0).data
        p_mean_value = torch.mean(p_tensor, dim=0).data
        p_max_value, _ = torch.max(p_tensor, dim=0)

        n_tensor = F.sigmoid(torch.stack([n_t, n_v, n_s]))
        n_mean_value = torch.mean(n_tensor).data

        p_mean_probability = torch.pow(p_mean_value, 1.0).data
        p_var_probability = torch.pow(torch.exp(-p_variance).data, 2.0)  # 0 ~ 1
        pos_weight = p_mean_probability * p_var_probability
        pos_weight = torch.clamp(pos_weight, 0, 1).data

        mask = torch.zeros_like(p_mean_value)
        mask[p_mean_value < n_mean_value] = 1

        neg_weight_max = torch.pow((p_max_value - n_mean_value.data), 1.0) * mask
        neg_weight = torch.clamp(neg_weight_max, 0, 1).data
        # print(neg_weight)

        return pos_weight, neg_weight

    def bpr_loss(self, users, pos_items, neg_items, p_weight, n_weight):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        p_maxi = torch.log(F.sigmoid(pos_scores - neg_scores)) * p_weight
        n_maxi = torch.log(F.sigmoid(neg_scores - pos_scores)) * n_weight
        mf_loss = -torch.mean(p_maxi + n_maxi)
        # mf_loss = -torch.sum(maxi)
        return mf_loss

    def pre_epoch_processing(self, epoch=0):
        pass

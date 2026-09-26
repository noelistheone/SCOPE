"""GUME (CIKM 2024) — ported faithfully into the SCOPE framework.

Graph-based and Uniform Modality Enhancement for multimodal recommendation.
Reference: Lin et al., CIKM 2024; official author implementation.

The algorithm (multi-modal encoding with image/text item-item kNN propagation,
user-item LightGCN over a UI graph AUGMENTED with an item-item block built from
the kNN-intersection of the image and text neighbourhoods, attribute separation
into coarse/fine-grained embeds, BPR + variance/mean alignment + behavior-modality
InfoNCE + user-modality InfoNCE with noise perturbation) is copied VERBATIM from the
official implementation. Only the framework glue differs: it uses this repo's
MultimodalRecommender base (self.v_feat/self.t_feat/self.n_users/self.norm_adj +
dict-form interactions), so the numbers are produced under the SAME protocol/split/
eval as every other table baseline.

Two source side-effects are replaced by faithful equivalents:
  * the item-item modality graphs and the kNN-intersection (find_inter) are built
    in-memory and VECTORIZED (no inter.json cache, no per-item Python grouping loop);
  * the bipartite norm-adj + R are rebuilt with GUME's own get_adj_mat from the train
    edges (identical to the official code), not reused from the framework.
None of these change the numerical algorithm.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender


def build_sim(context):
    """Cosine similarity matrix (verbatim GUME build_sim, +1e-12 for safety)."""
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True) + 1e-12)
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def build_knn_normalized_graph(adj, topk, norm_type='sym'):
    """Vectorized equivalent of GUME's sparse build_knn_normalized_graph
    (get_sparse_laplacian, sym) — no torch_scatter, no Python edge loop.

    Returns (sparse_coo_tensor, knn_ind) where knn_ind is the [N, topk] index
    tensor (row order) that find_inter consumes."""
    device = adj.device
    N = adj.shape[0]
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)             # [N, topk]
    rows = torch.arange(N, device=device).view(-1, 1).expand(-1, topk).reshape(-1)
    cols = knn_ind.reshape(-1)
    vals = knn_val.reshape(-1)
    if norm_type == 'sym':
        deg = torch.zeros(N, device=device).scatter_add_(0, rows, vals)
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        vals = d_inv_sqrt[rows] * vals * d_inv_sqrt[cols]
    g = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, N)).coalesce()
    return g, knn_ind


class GUME(MultimodalRecommender):
    def __init__(self, config, n_users, n_items, norm_adj=None,
                 v_feat=None, t_feat=None, train_user_idx=None, train_item_idx=None):
        super().__init__(config, n_users, n_items, norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        assert self.v_feat is not None and self.t_feat is not None, \
            "GUME requires both image (v_feat) and text (t_feat) features."

        self.sparse = True
        self.bm_loss = float(config.get("bm_loss", 0.01))
        self.um_loss = float(config.get("um_loss", 0.01))
        self.vt_loss = float(config.get("vt_loss", 0.1))
        self.reg_weight_1 = float(config.get("reg_weight_1", 1e-5))
        self.reg_weight_2 = float(config.get("reg_weight_2", 0.1))
        self.bm_temp = float(config.get("bm_temp", 0.4))
        self.um_temp = float(config.get("um_temp", 0.1))
        self.n_ui_layers = int(config.get("n_ui_layers", 3))
        self.embedding_dim = int(config.get("embedding_size", 64))
        self.knn_k = int(config.get("knn_k", 10))
        self.n_layers = int(config.get("n_layers", 2))
        self.batch_size = int(config.get("train_batch_size", 2048))
        self.tau = 0.5

        # raw user-item interaction (coo) from the passed train edges
        ui = sp.coo_matrix(
            (np.ones(train_user_idx.numel(), dtype=np.float32),
             (train_user_idx.numpy(), train_item_idx.numpy())),
            shape=(n_users, n_items)).astype(np.float32)
        self.interaction_matrix = ui

        # id / user embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.extended_image_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_image_user.weight)

        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        # learnable modality embeddings (init from frozen-but-trainable raw features)
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat.clone(), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat.clone(), freeze=False)

        # image / text item-item kNN normalized graphs (built on CPU; moved by trainer)
        image_adj, image_knn_ind = build_knn_normalized_graph(
            build_sim(self.v_feat), topk=self.knn_k, norm_type='sym')
        text_adj, text_knn_ind = build_knn_normalized_graph(
            build_sim(self.t_feat), topk=self.knn_k, norm_type='sym')
        self.register_buffer("image_original_adj", image_adj, persistent=False)
        self.register_buffer("text_original_adj", text_adj, persistent=False)

        # Enhancing User-Item Graph: item-item block from kNN-intersection (vectorized)
        ii_adj = self._find_inter_add_edge(image_knn_ind, text_knn_ind)   # scipy coo, int
        norm_adj, R = self._get_adj_mat(ii_adj.tolil())
        self.register_buffer("gume_norm_adj", self._sp2t(norm_adj), persistent=False)
        self.register_buffer("R", self._sp2t(R), persistent=False)

        # modality projection / behaviour heads (verbatim)
        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )

        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )

        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

    # ---- vectorized find_inter + add_edge ----
    def _find_inter_add_edge(self, image_knn_ind, text_knn_ind):
        """Per-item intersection of the image-kNN and text-kNN neighbour sets,
        self-loops removed; returns the item-item adjacency as a scipy coo (int).

        Vectorized replacement for GUME.find_inter + GUME.add_edge. The source
        groups the sparse adjacency indices into chunks of knn_k per item (row
        order) and takes set(img_nbrs) & set(txt_nbrs), dropping the item id
        itself. Here image_knn_ind/text_knn_ind are already [n_items, knn_k] in
        row order, so the same intersection is computed with a membership test.
        """
        n = self.n_items
        k = image_knn_ind.shape[1]
        img = image_knn_ind.cpu()                                   # [n, k]
        txt = text_knn_ind.cpu()                                    # [n, k]

        # membership: for each image neighbour, is it also a text neighbour of the same item?
        # eq[i, a, b] = (img[i,a] == txt[i,b]); keep img[i,a] if it appears in txt[i]
        eq = (img.unsqueeze(2) == txt.unsqueeze(1))                 # [n, k, k]
        in_both = eq.any(dim=2)                                     # [n, k] bool
        item_ids = torch.arange(n).view(-1, 1).expand(-1, k)       # [n, k]
        not_self = (img != item_ids)                                # drop self-loop
        mask = in_both & not_self                                   # [n, k]

        rows = item_ids[mask].numpy()
        cols = img[mask].numpy()
        vals = np.ones(rows.shape[0], dtype=int)
        item_adj = sp.coo_matrix((vals, (rows, cols)),
                                 shape=(self.n_items, self.n_items), dtype=int)
        return item_adj

    # ---- graph construction (verbatim GUME.get_adj_mat) ----
    def _get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj

        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()

        R = norm_adj_mat[:self.n_users, self.n_users:]

        return norm_adj_mat.tocsr(), R.tocsr()

    @staticmethod
    def _sp2t(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse coo tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def pre_epoch_processing(self, epoch=0):
        pass

    # ---- propagation (verbatim) ----
    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def forward(self, adj, train=False):
        #  Encoding Multiple Modalities
        image_item_embeds = torch.multiply(
            self.item_id_embedding.weight, self.image_space_trans(self.image_embedding.weight))
        text_item_embeds = torch.multiply(
            self.item_id_embedding.weight, self.text_space_trans(self.text_embedding.weight))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)

        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)

        extended_image_embeds = self.conv_ui(adj, self.extended_image_user.weight, explicit_image_item)

        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)

        extended_text_embeds = self.conv_ui(adj, self.extended_text_user.weight, explicit_text_item)

        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2

        # Attributes Separation for Better Integration
        image_weights, text_weights = torch.split(
            self.softmax(
                torch.cat([
                    self.separate_coarse(explicit_image_embeds),
                    self.separate_coarse(explicit_text_embeds)
                ], dim=-1)
            ),
            1,
            dim=-1
        )
        coarse_grained_embeds = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds

        fine_grained_image = torch.multiply(
            self.image_behavior(extended_id_embeds), (explicit_image_embeds - coarse_grained_embeds))
        fine_grained_text = torch.multiply(
            self.text_behavior(extended_id_embeds), (explicit_text_embeds - coarse_grained_embeds))
        integration_embeds = (fine_grained_image + fine_grained_text + coarse_grained_embeds) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train:
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), \
                (explicit_image_embeds, explicit_text_embeds)

        return all_embeds

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos_items = interaction["pos_item"]
        neg_items = interaction["neg_item"]

        embeds_1, embeds_2, embeds_3 = self.forward(self.gume_norm_adj, train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)

        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        explicit_image_embeds, explicit_text_embeds = embeds_3

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(explicit_image_embeds, explicit_text_embeds)

        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        bm_loss = self.bm_loss * (
            self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp)
            + self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp))

        al_loss = vt_loss + bm_loss

        extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)

        # Enhancing User Modality Representation
        c_loss = self.InfoNCE(extended_it_user[users], integration_users[users], self.um_temp)
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        reg_loss_2 = self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2

        return bpr_loss + al_loss + um_loss + reg_loss

    def cal_noise_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(x.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss

    def align_vt(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)

        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()

        return vt_loss

    def full_sort_predict(self, interaction):
        user = interaction["user"]

        all_embeds = self.forward(self.gume_norm_adj)
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

"""SMORE (WSDM 2025) — ported faithfully into the SCOPE framework.

Spectrum-based Modality Representation Fusion GCN for multimodal recommendation.
Reference: Ong & Khong, WSDM 2025; official code https://github.com/kennethorq/SMORE.

The algorithm (FFT spectrum convolution, multi-view item-item propagation, modality-
preference gating, BPR + InfoNCE) is copied VERBATIM from the official implementation.
Only the framework glue differs: it uses this repo's MultimodalRecommender base
(self.v_feat/self.t_feat/self.n_users/self.norm_adj + dict-form interactions), so the
numbers are produced under the SAME protocol/split/eval as every other table baseline.
The bipartite norm-adj and R are rebuilt with SMORE's own get_adj_mat from the train
edges (identical to the official code), not reused from the framework, for faithfulness.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.abstract_recommender import MultimodalRecommender


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True) + 1e-12)
    return torch.mm(context_norm, context_norm.transpose(1, 0))


def build_knn_normalized_graph(adj, topk, norm_type='sym'):
    """Vectorized equivalent of the official sparse build_knn_normalized_graph
    (get_sparse_laplacian, sym) — no torch_scatter, no Python edge loop."""
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
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, N)).coalesce()


class SMORE(MultimodalRecommender):
    def __init__(self, config, n_users, n_items, norm_adj=None,
                 v_feat=None, t_feat=None, train_user_idx=None, train_item_idx=None):
        super().__init__(config, n_users, n_items, norm_adj=norm_adj, v_feat=v_feat, t_feat=t_feat)
        self.sparse = True
        self.cl_loss = float(config.get("cl_loss", 0.01))
        self.n_ui_layers = int(config.get("n_ui_layers", 2))
        self.embedding_dim = int(config.get("embedding_size", 64))
        self.n_layers = int(config.get("n_layers", 1))
        self.reg_weight = float(config.get("reg_weight", 1e-4))
        self.image_knn_k = int(config.get("image_knn_k", 10))
        self.text_knn_k = int(config.get("text_knn_k", 10))
        self.dropout_rate = float(config.get("dropout_rate", 0.8))
        self.batch_size = int(config.get("train_batch_size", 2048))
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # rebuild SMORE's own normalized bipartite adj + R from the train edges (verbatim get_adj_mat)
        ui = sp.coo_matrix(
            (np.ones(train_user_idx.numel(), dtype=np.float32),
             (train_user_idx.numpy(), train_item_idx.numpy())),
            shape=(n_users, n_items)).astype(np.float32)
        self.interaction_matrix = ui
        norm_bip, R = self._get_adj_mat(ui)
        self.register_buffer("smore_norm_adj", self._sp2t(norm_bip), persistent=False)
        self.register_buffer("R", self._sp2t(R), persistent=False)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # item-item modality graphs from the (raw) frozen features — same features the table uses
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat.clone(), freeze=False)
            image_adj = build_knn_normalized_graph(build_sim(self.v_feat), self.image_knn_k, 'sym')
            self.register_buffer("image_original_adj", image_adj, persistent=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat.clone(), freeze=False)
            text_adj = build_knn_normalized_graph(build_sim(self.t_feat), self.text_knn_k, 'sym')
            self.register_buffer("text_original_adj", text_adj, persistent=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.register_buffer("fusion_adj", self._max_pool_fusion(image_adj, text_adj), persistent=False)
        self.softmax = nn.Softmax(dim=-1)
        mk = lambda: nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
                                   nn.Linear(self.embedding_dim, self.embedding_dim, bias=False))
        gk = lambda: nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.query_v, self.query_t = mk(), mk()
        self.gate_v, self.gate_t, self.gate_f = gk(), gk(), gk()
        self.gate_image_prefer, self.gate_text_prefer, self.gate_fusion_prefer = gk(), gk(), gk()
        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2))

    # ---- graph construction (verbatim SMORE.get_adj_mat) ----
    def _get_adj_mat(self, inter):
        n = self.n_users + self.n_items
        adj = sp.dok_matrix((n, n), dtype=np.float32).tolil()
        R = inter.tolil()
        adj[:self.n_users, self.n_users:] = R
        adj[self.n_users:, :self.n_users] = R.T
        adj = adj.todok()
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm = d_mat.dot(adj).dot(d_mat).tolil()
        return norm.tocsr(), norm[:self.n_users, self.n_users:].tocsr()

    @staticmethod
    def _sp2t(m):
        m = m.tocoo().astype(np.float32)
        idx = torch.from_numpy(np.vstack((m.row, m.col)).astype(np.int64))
        return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data), torch.Size(m.shape)).coalesce()

    def _max_pool_fusion(self, image_adj, text_adj):
        ia, ta = image_adj.coalesce(), text_adj.coalesce()
        ii, iv = ia.indices(), ia.values()
        ti, tv = ta.indices(), ta.values()
        comb = torch.cat((ii, ti), dim=1)
        comb, uidx = torch.unique(comb, dim=1, return_inverse=True)
        cvi = torch.full((comb.size(1),), float('-inf'))
        cvt = torch.full((comb.size(1),), float('-inf'))
        cvi[uidx[:ii.size(1)]] = iv
        cvt[uidx[ii.size(1):]] = tv
        cv, _ = torch.max(torch.stack((cvi, cvt)), dim=0)
        return torch.sparse_coo_tensor(comb, cv, ia.size()).coalesce()

    def pre_epoch_processing(self, epoch=0):
        pass

    def spectrum_convolution(self, image_embeds, text_embeds):
        image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')
        text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')
        iw = torch.view_as_complex(self.image_complex_weight)
        tw = torch.view_as_complex(self.text_complex_weight)
        fw = torch.view_as_complex(self.fusion_complex_weight)
        image_conv = torch.fft.irfft(image_fft * iw, n=image_embeds.shape[1], dim=1, norm='ortho')
        text_conv = torch.fft.irfft(text_fft * tw, n=text_embeds.shape[1], dim=1, norm='ortho')
        fusion_conv = torch.fft.irfft(text_fft * image_fft * fw, n=text_embeds.shape[1], dim=1, norm='ortho')
        return image_conv, text_conv, fusion_conv

    def forward(self, adj, train=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
        fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego = torch.cat([user_embeds, item_embeds], dim=0)
        all_emb = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(adj, ego)
            all_emb += [ego]
        content_embeds = torch.stack(all_emb, dim=1).mean(dim=1, keepdim=False)

        for _ in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        for _ in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)
        for _ in range(self.n_layers):
            fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        agg_image_embeds = self.softmax(fusion_att_v) * image_embeds
        agg_text_embeds = self.softmax(fusion_att_t) * text_embeds
        image_prefer = self.dropout(self.gate_image_prefer(content_embeds))
        text_prefer = self.dropout(self.gate_text_prefer(content_embeds))
        fusion_prefer = self.dropout(self.gate_fusion_prefer(content_embeds))
        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
        fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)
        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0)
        all_embeds = content_embeds + side_embeds
        u, i = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        if train:
            return u, i, side_embeds, content_embeds
        return u, i

    def bpr_loss(self, users, pos, neg):
        pos_s = torch.sum(torch.mul(users, pos), dim=1)
        neg_s = torch.sum(torch.mul(users, neg), dim=1)
        reg = 0.5 * (users ** 2).sum() + 0.5 * (pos ** 2).sum() + 0.5 * (neg ** 2).sum()
        reg = reg / self.batch_size
        mf = -torch.mean(F.logsigmoid(pos_s - neg_s))
        return mf, self.reg_weight * reg

    def InfoNCE(self, v1, v2, temp):
        v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
        pos = torch.exp((v1 * v2).sum(dim=-1) / temp)
        ttl = torch.exp(torch.matmul(v1, v2.transpose(0, 1)) / temp).sum(dim=1)
        return torch.mean(-torch.log(pos / ttl))

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]
        ua, ia, side, content = self.forward(self.smore_norm_adj, train=True)
        mf, emb = self.bpr_loss(ua[users], ia[pos], ia[neg])
        side_u, side_i = torch.split(side, [self.n_users, self.n_items], dim=0)
        cont_u, cont_i = torch.split(content, [self.n_users, self.n_items], dim=0)
        cl = self.InfoNCE(side_i[pos], cont_i[pos], 0.2) + self.InfoNCE(side_u[users], cont_u[users], 0.2)
        return mf + emb + self.cl_loss * cl

    def full_sort_predict(self, interaction):
        user = interaction["user"]
        ue, ie = self.forward(self.smore_norm_adj)
        return torch.matmul(ue[user], ie.transpose(0, 1))

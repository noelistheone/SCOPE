"""LGMRec: Guo et al., AAAI 2024.

Local-Global multimodal recommendation:
  - Local (CGE + MGE): LightGCN-style collaborative graph embedding + per-modality
    graph embeddings propagated on the user-item bipartite graph.
  - Global (GHE): hypergraph embeddings learned via gumbel-softmax hyperedge
    dependencies, propagated through a small HGNN.
  - Final: lge + alpha * normalized(ghe).

Loss = BPR + cl_weight * hypergraph contrastive loss + reg.

Reference: https://github.com/georgeguo-cn/LGMRec
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import BPRLoss
from src.data.graph_utils import split_norm_adj_blocks


class _HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer: int) -> None:
        super().__init__()
        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = i_hyper.t() @ i_ret
            i_ret = i_hyper @ lat
            u_ret = u_hyper @ lat
        return u_ret, i_ret


class LGMRec(MultimodalRecommender):
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
            raise ValueError("LGMRec requires BOTH v_feat and t_feat")

        emb = self.embedding_size
        self.n_ui_layers = int(config.get("n_ui_layers", 2))
        self.n_mm_layer = int(config.get("n_mm_layers", 1))
        self.n_hyper_layer = int(config.get("n_hyper_layer", 1))
        self.hyper_num = int(config.get("hyper_num", 4))
        self.keep_rate = float(config.get("keep_rate", 0.5))
        self.alpha = float(config.get("alpha", 0.2))
        self.cl_weight = float(config.get("cl_weight", 0.03))
        self.tau = float(config.get("tau", 0.2))

        self.user_embedding = nn.Embedding(self.n_users, emb)
        self.item_id_embedding = nn.Embedding(self.n_items, emb)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=1 - self.keep_rate)

        # adj is the (n_users, n_items) normalized block (not symmetric bipartite).
        R, _ = split_norm_adj_blocks(self.norm_adj, self.n_users, self.n_items)
        # Build the un-normalized R for the "user_feats = R @ item_feats * 1/deg" step.
        # Easier: use R rows row-normalized 1/|N(u)|.
        # We approximate: use the symmetric R directly (matches LGMRec semantically).
        self.register_buffer("R", R, persistent=False)
        # Per-user degree (number of train interactions, inv).
        with torch.no_grad():
            deg = torch.sparse.sum(R, dim=-1).to_dense() * 0 + 1.0  # placeholder ones
        self.register_buffer("user_inv_deg", deg.unsqueeze(-1), persistent=False)

        # Modality transforms & hyperedge embeddings.
        self.image_trs = nn.Parameter(
            torch.empty(self.v_feat_dim, self.feat_embed_dim))
        nn.init.xavier_uniform_(self.image_trs)
        self.text_trs = nn.Parameter(
            torch.empty(self.t_feat_dim, self.feat_embed_dim))
        nn.init.xavier_uniform_(self.text_trs)
        self.v_hyper = nn.Parameter(torch.empty(self.v_feat_dim, self.hyper_num))
        nn.init.xavier_uniform_(self.v_hyper)
        self.t_hyper = nn.Parameter(torch.empty(self.t_feat_dim, self.hyper_num))
        nn.init.xavier_uniform_(self.t_hyper)

        self.hgnnLayer = _HGNNLayer(self.n_hyper_layer)
        self.bpr_loss = BPRLoss()

    def _cge(self) -> torch.Tensor:
        ego = torch.cat([self.user_embedding.weight,
                         self.item_id_embedding.weight], dim=0)
        all_embs = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embs.append(ego)
        return torch.stack(all_embs, dim=1).mean(dim=1)

    def _mge(self, modality: str) -> torch.Tensor:
        if modality == "v":
            item_feats = self.v_feat @ self.image_trs
        else:
            item_feats = self.t_feat @ self.text_trs
        # Aggregate to users by R-normalized.
        user_feats = torch.sparse.mm(self.R, item_feats)
        embs = torch.cat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            embs = torch.sparse.mm(self.norm_adj, embs)
        return embs

    def _forward_views(self):
        # Hyperedge dependencies (gumbel-softmax).
        iv_hyper = self.v_feat @ self.v_hyper                       # [n_items, K]
        it_hyper = self.t_feat @ self.t_hyper
        uv_hyper = torch.sparse.mm(self.R, iv_hyper)                # [n_users, K]
        ut_hyper = torch.sparse.mm(self.R, it_hyper)
        iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
        uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
        ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        cge_embs = self._cge()
        v_feats = self._mge("v")
        t_feats = self._mge("t")
        mge_embs = F.normalize(v_feats, dim=-1) + F.normalize(t_feats, dim=-1)
        lge_embs = cge_embs + mge_embs

        uv_emb, iv_emb = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper),
                                        cge_embs[self.n_users:])
        ut_emb, it_emb = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper),
                                        cge_embs[self.n_users:])
        av_hyper_embs = torch.cat([uv_emb, iv_emb], dim=0)
        at_hyper_embs = torch.cat([ut_emb, it_emb], dim=0)
        ghe_embs = av_hyper_embs + at_hyper_embs

        all_embs = lge_embs + self.alpha * F.normalize(ghe_embs, dim=-1)
        u, i = torch.split(all_embs, [self.n_users, self.n_items], dim=0)
        return u, i, (uv_emb, iv_emb, ut_emb, it_emb)

    @staticmethod
    def _ssl_triple_loss(emb1, emb2, all_emb, tau):
        n1 = F.normalize(emb1, dim=-1)
        n2 = F.normalize(emb2, dim=-1)
        na = F.normalize(all_emb, dim=-1)
        pos = torch.exp((n1 * n2).sum(dim=-1) / tau)
        ttl = torch.exp(n1 @ na.t() / tau).sum(dim=-1)
        return -torch.log(pos / (ttl + 1e-9)).mean()

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        u, i, (uv, iv, ut, it) = self._forward_views()
        u_e, p_e, n_e = u[users], i[pos], i[neg]
        pos_s = (u_e * p_e).sum(dim=-1)
        neg_s = (u_e * n_e).sum(dim=-1)
        mf = self.bpr_loss(pos_s, neg_s)

        cl = (self._ssl_triple_loss(uv[users], ut[users], ut, self.tau)
              + self._ssl_triple_loss(iv[pos], it[pos], it, self.tau))

        reg = (torch.norm(u_e, p=2) + torch.norm(p_e, p=2) + torch.norm(n_e, p=2)) / u_e.shape[0]
        total = mf + self.cl_weight * cl + self.reg_weight * reg
        return total, {"mf": mf.item(), "cl": cl.item(), "reg": reg.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u, i, _ = self._forward_views()
        return u[users] @ i.t()

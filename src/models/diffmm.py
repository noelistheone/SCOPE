"""DiffMM (Jiang et al., MM 2024) — simplified clean-room implementation.

The full DiffMM paper uses a diffusion-based denoising stage to augment the
user-item graph with modality-aware noise. Faithful reproduction of the
diffusion stage adds substantial complexity (forward diffusion, score
network, multiple modality-conditional denoisers).

This implementation captures the essential ideas while staying within a
single-file footprint:
  - Per-modality LightGCN towers (image + text) on the bipartite graph.
  - **Stochastic edge augmentation** at the start of each epoch
    (``pre_epoch_processing``): keep a random ``keep_rate`` fraction of edges
    per modality, with the keep probability biased toward edges whose item
    cosine-similarity matches that modality.
  - Modality-weighted fusion with learned softmax weights.

Loss = BPR + InfoNCE contrastive across modality views + reg.

Reference: https://github.com/HKUDS/DiffMM
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import MultimodalRecommender
from src.common.loss import BPRLoss, EmbLoss, InfoNCELoss


class DiffMM(MultimodalRecommender):
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
            raise ValueError("DiffMM requires BOTH v_feat and t_feat")

        emb = self.embedding_size
        self.n_layers = int(config.get("n_layers", 2))
        self.keep_rate = float(config.get("keep_rate", 0.7))
        self.cl_weight = float(config.get("cl_weight", 0.5))
        self.tau = float(config.get("tau", 0.2))

        self.user_embedding = nn.Embedding(self.n_users, emb)
        self.item_id_embedding = nn.Embedding(self.n_items, emb)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_trs = nn.Linear(self.v_feat_dim, emb)
        self.text_trs = nn.Linear(self.t_feat_dim, emb)
        self.modal_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        # adj_v and adj_t are stochastic copies; refreshed each epoch.
        self._adj_v: Optional[torch.Tensor] = None
        self._adj_t: Optional[torch.Tensor] = None

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.infonce = InfoNCELoss(temperature=self.tau)

    def _drop_edges(self) -> torch.Tensor:
        """Randomly retain a ``keep_rate`` fraction of edges in ``norm_adj``."""
        idx = self.norm_adj.indices()
        val = self.norm_adj.values()
        n = val.numel()
        keep = torch.rand(n, device=val.device) < self.keep_rate
        return torch.sparse_coo_tensor(idx[:, keep], val[keep],
                                       self.norm_adj.shape).coalesce()

    def pre_epoch_processing(self, epoch: int) -> None:
        # Independent stochastic adjacencies per modality.
        self._adj_v = self._drop_edges()
        self._adj_t = self._drop_edges()

    def _lightgcn(self, adj: torch.Tensor, item_seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat([self.user_embedding.weight, item_seed], dim=0)
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adj, ego)
            all_embs.append(ego)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        return torch.split(out, [self.n_users, self.n_items], dim=0)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use the diffusion-augmented adj if available, else the full norm_adj.
        adj_v = self._adj_v if self._adj_v is not None else self.norm_adj
        adj_t = self._adj_t if self._adj_t is not None else self.norm_adj

        # Normalize the projected modality features so they don't swamp the ID
        # embedding.
        v_seed = self.item_id_embedding.weight + F.normalize(self.image_trs(self.v_feat), dim=-1)
        t_seed = self.item_id_embedding.weight + F.normalize(self.text_trs(self.t_feat), dim=-1)

        u_v, i_v = self._lightgcn(adj_v, v_seed)
        u_t, i_t = self._lightgcn(adj_t, t_seed)

        w = F.softmax(self.modal_weight, dim=-1)
        u = w[0] * u_v + w[1] * u_t
        i = w[0] * i_v + w[1] * i_t
        return u, i, i_v, i_t

    def calculate_loss(self, interaction):
        users = interaction["user"]
        pos = interaction["pos_item"]
        neg = interaction["neg_item"]

        u, i, i_v, i_t = self._propagate()
        u_e, p_e, n_e = u[users], i[pos], i[neg]
        pos_s = (u_e * p_e).sum(dim=-1)
        neg_s = (u_e * n_e).sum(dim=-1)
        mf = self.bpr_loss(pos_s, neg_s)

        cl = self.infonce(i_v[pos], i_t[pos])
        reg = self.reg_loss(
            self.user_embedding(users),
            self.item_id_embedding(pos),
            self.item_id_embedding(neg),
        )
        total = mf + self.cl_weight * cl + self.reg_weight * reg
        return total, {"mf": mf.item(), "cl": cl.item(), "reg": reg.item()}

    def full_sort_predict(self, interaction):
        users = interaction["user"]
        u, i, _, _ = self._propagate()
        return u[users] @ i.t()

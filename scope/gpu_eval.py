#!/usr/bin/env python
"""Fully GPU-resident full-sort evaluator (no per-batch .cpu()). Eliminates the CPU bottleneck of
TopKEvaluator for the many-eval training loops. Computes Recall/NDCG/Precision@{10,20} on the GPU
from a dense score matrix S [n_users, n_items], the train mask, and per-user test positives.

Build once per dataset: GPUEval(dset, phase). Then call ev.eval(S) -> dict of metrics.
Validated to match TopKEvaluator (popularity sanity) to <1e-4.
"""
from __future__ import annotations
import numpy as np, torch
from src.data.dataloader import EvalDataLoader


class GPUEval:
    def __init__(self, dset, phase, device, topks=(10, 20), max_pos=None):
        self.dev = device; self.topks = topks; self.maxk = max(topks)
        # gather per-user positives + train-history for masking, fully on GPU
        loader = EvalDataLoader(dset, phase=phase, batch_size=4096)
        users, pos_list, hist_pairs = [], [], []
        for b in loader:
            uids = b["user_ids"].tolist(); users.extend(uids)
            pi = b["positive_items"]
            for r, u in enumerate(uids):
                pos_list.append([int(x) for x in pi[r]])
            hi, hv = b["history_indices"], b["history_values"]
            if hi.numel() > 0:
                mk = hv.bool()
                for r, u in enumerate(uids):
                    h = hi[r][mk[r]]; h = h[h >= 0]
                    for it in h.tolist(): hist_pairs.append((u, it))
        self.users = torch.tensor(users, device=device, dtype=torch.long)
        self.nfit = torch.tensor([len(p) for p in pos_list], device=device, dtype=torch.float)
        # pad positives -> [U, P]
        P = max(len(p) for p in pos_list)
        pos = np.full((len(pos_list), P), -1, np.int64)
        for i, p in enumerate(pos_list): pos[i, :len(p)] = p
        self.pos = torch.from_numpy(pos).to(device)            # [U,P] (-1 pad)
        if hist_pairs:
            hp = torch.tensor(hist_pairs, device=device, dtype=torch.long)
            self.hist_u, self.hist_i = hp[:, 0], hp[:, 1]
        else:
            self.hist_u = self.hist_i = None
        # idcg denominator per #relevant
        disc = 1.0 / torch.log2(torch.arange(2, self.maxk + 2, device=device).float())
        self.cumdisc = torch.cat([torch.zeros(1, device=device), disc.cumsum(0)])  # [maxk+1]

    @torch.no_grad()
    def eval(self, S, batch=4096):
        return self._run(lambda bu, s: S[bu].clone().float(), batch)

    @torch.no_grad()
    def eval_streaming(self, score_fn, batch=2048):
        """score_fn(bu_user_ids)->[b,N] dense scores; never materializes the full matrix (for Elec)."""
        return self._run(lambda bu, s: score_fn(bu).float(), batch)

    @torch.no_grad()
    def _run(self, get_scores, batch):
        out = {f"{m}@{k}": 0.0 for k in self.topks for m in ("Recall", "NDCG", "Precision")}
        U = self.users.numel()
        for s in range(0, U, batch):
            bu = self.users[s:s + batch]
            sc = get_scores(bu, s)
            sc = self._mask(sc, bu)            # mask each user's train history (GPU, searchsorted)
            _, idx = torch.topk(sc, self.maxk, dim=1)                # [b, maxk]
            bp = self.pos[s:s + batch]                               # [b, P]
            hit = (idx.unsqueeze(2) == bp.unsqueeze(1)).any(2).float()  # [b, maxk] 1 if topk item is a positive
            nrel = self.nfit[s:s + batch].clamp(min=1)
            for k in self.topks:
                hk = hit[:, :k]
                nhit = hk.sum(1)
                out[f"Recall@{k}"] += (nhit / nrel).sum().item()
                out[f"Precision@{k}"] += (nhit / k).sum().item()
                dcg = (hk * (1.0 / torch.log2(torch.arange(2, k + 2, device=self.dev).float())).unsqueeze(0)).sum(1)
                ideal_n = torch.minimum(nrel, torch.full_like(nrel, k)).long()
                idcg = self.cumdisc[ideal_n]
                out[f"NDCG@{k}"] += (dcg / idcg.clamp(min=1e-9)).sum().item()
        return {kk: vv / U for kk, vv in out.items()}

    @torch.no_grad()
    def recall_per_user(self, S, k=20, batch=4096):
        """Return [U] tensor of Recall@k per user (aligned with self.users order)."""
        U = self.users.numel(); out = torch.zeros(U, device=self.dev)
        for s in range(0, U, batch):
            bu = self.users[s:s + batch]
            sc = self._mask(S[bu].clone().float(), bu)
            _, idx = torch.topk(sc, k, dim=1)
            bp = self.pos[s:s + batch]
            hit = (idx.unsqueeze(2) == bp.unsqueeze(1)).any(2).float().sum(1)  # [b] #hits
            out[s:s + bu.numel()] = hit / self.nfit[s:s + batch].clamp(min=1)
        return out

    def _mask(self, sc, bu):
        if self.hist_u is None: return sc
        # map global user id -> local row via a lookup built per batch
        order = torch.argsort(bu)
        bu_sorted = bu[order]
        pos_in_batch = torch.searchsorted(bu_sorted, self.hist_u)
        pos_in_batch = pos_in_batch.clamp(max=bu.numel() - 1)
        valid = (bu_sorted[pos_in_batch] == self.hist_u)
        rows_local = order[pos_in_batch[valid]]
        cols = self.hist_i[valid]
        sc[rows_local, cols] = float("-inf")
        return sc

#!/usr/bin/env python
"""Shared harness for the SCOPE experiment-design suite.

Loads the 3 z-scored views as scope_u_ablate_ease.py does:
  col  = z(FREEDOM cached scores)          results/baseline_scores/freedom_{ds}_scores.npy
  item = closed-form EASE+text base        closed_form_base(R, dset, val-tuned)
  set  = z(SCOPE set-completion head)      ckpts/scope/scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt
Plus per-user Recall@20 vectors (aligned user order) via GPUEval(test), for bootstrap/decorrelation.
Integrity: closed_form_base val-tunes (lam, a) on valid; views z-scored; test via trusted evaluator.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import (Rmat, build_lists, closed_form_base, SCOPE, evalS_trusted, zr, gram, BAR, DEV)

CK = ROOT / "ckpts" / "scope"
PRED = ROOT / "results" / "baseline_scores"
OUT = ROOT / "results" / "scope" / "significance"; OUT.mkdir(parents=True, exist_ok=True)
GRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def load_views(ds, set_ckpt=None):
    """Return all artifacts needed by the experiment scripts. set_ckpt overrides the SCOPE ckpt path."""
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    bar_r, bar_n = BAR[ds]; half = dset.n_items > 20000; dt = torch.float16 if half else torch.float32
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    S_item = closed_form_base(R, dset, gevV, half=half).to(dt)
    ckpt = set_ckpt or (CK / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt")
    model = SCOPE(dset.n_items, 256).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV)); model.eval()
    S_set = zr(model.score_all(R, degf)).to(dt)
    S_col = zr(torch.from_numpy(np.load(PRED / f"freedom_{ds}_scores.npy")).to(dt).to(DEV))
    return dict(ds=ds, dset=dset, R=R, deg=deg, degf=degf, items=items, vmask=vmask,
                gevV=gevV, gevT=gevT, S_col=S_col, S_item=S_item, S_set=S_set,
                bar=(bar_r, bar_n), dt=dt, half=half, model=model, n_items=dset.n_items)


def _zr_chunked(z, En, n_items, dt, bs=4096):
    """z-scored (z @ En^T) without ever holding the full fp32 [U,n_items]; output fp16/dt."""
    U = z.shape[0]; out = torch.empty(U, n_items, dtype=dt, device=DEV)
    for s in range(0, U, bs):
        blk = (z[s:s + bs] @ En.t()).float()
        m = blk.mean(1, keepdim=True); sd = blk.std(1, keepdim=True) + 1e-9
        out[s:s + blk.shape[0]] = ((blk - m) / sd).to(dt); del blk
    return out


def random_tower(n_items, R, degf, d=256, seed=0, dt=torch.float16):
    """NON-LEARNED rank-d set-completion 'null' tower: random item embeddings, mean-pool, cosine (chunked, dt out)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    E = torch.randn(n_items, d, generator=g, device=DEV) / (d ** 0.5)
    En = F.normalize(E, dim=1)
    z = F.normalize(torch.sparse.mm(R, E) / degf.clamp(min=1).unsqueeze(1), dim=1)
    out = _zr_chunked(z, En, n_items, dt)
    del E, En, z; torch.cuda.empty_cache(); return out


def cooc_knn_view(R, n_items, k=20, dt=torch.float16):
    """NON-LEARNED co-occurrence kNN item-item CF view: sym-normalized top-k Gram, R-propagated (frugal, dt out)."""
    G = gram(R)                                   # R^T R  [n_items, n_items] fp32
    kth = torch.topk(G, k + 1, 1).values[:, -1:]
    A = torch.where(G >= kth, G, torch.zeros_like(G)); A.fill_diagonal_(0.0); del G
    d = A.sum(1).clamp(min=1e-6); A = A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0)  # fp32 for spmm
    torch.cuda.empty_cache()
    S = zr(torch.sparse.mm(R, A)).to(dt)          # spmm in fp32, cast result to dt
    del A; torch.cuda.empty_cache()
    return S


def best_gamma(gevV, base, cand, grid=None):
    """Val-tune fused = base + g*cand (set-anchored). base, cand are score tensors. Returns (g*, val_r)."""
    grid = grid or GRID
    best = (0.0, -1.0)
    for g in grid:
        r = gevV.recall_per_user(base + g * cand, 20).mean().item()
        if r > best[1]: best = (g, r)
    return best


def per_user(V, S):
    """Per-user Recall@20 on TEST (aligned user order), as a numpy array."""
    return V['gevT'].recall_per_user(S, 20).cpu().numpy()


def topk_all(V, S, k=20, batch=4096):
    """Train-masked top-k item ids per TEST user, [U,k], aligned with gevT.users order."""
    g = V['gevT']; U = g.users.numel(); out = torch.zeros(U, k, dtype=torch.long, device=DEV)
    for s in range(0, U, batch):
        bu = g.users[s:s + batch]; sc = g._mask(S[bu].clone().float(), bu)
        _, idx = torch.topk(sc, k, 1); out[s:s + bu.numel()] = idx
    return out


def hit_mask_on_pos(V, topk):
    """[U,P] bool: which of each user's held-out positives are in their top-k. Also returns npos [U]."""
    g = V['gevT']; pos = g.pos                                   # [U,P], -1 pad
    hit = (topk.unsqueeze(2) == pos.unsqueeze(1)).any(1)         # [U,P]
    hit = hit & (pos >= 0)
    return hit, (pos >= 0).sum(1)


def item_degree(V):
    """Train degree per item (n users who interacted), [n_items] float on GPU."""
    return torch.sparse.sum(V['R'], 0).to_dense().float()


def gini(x):
    """Gini coefficient of a nonneg vector (recommendation-frequency concentration)."""
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    if s <= 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def test_metrics(V, S):
    return evalS_trusted(S, V['dset'], "test")


def paired_bootstrap(a, b, B=10000, seed=0):
    """Paired user-level bootstrap of mean(a-b). a,b: per-user metric arrays. Returns mean delta, 95% CI, two-sided p."""
    a = np.asarray(a, float); b = np.asarray(b, float); diff = a - b; n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = diff[idx].mean(1)
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(mean_delta=float(diff.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                p_two_sided=float(min(p, 1.0)), n_users=int(n))

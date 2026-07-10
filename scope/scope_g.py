#!/usr/bin/env python
"""SCOPE-G: a graph-propagation set-completion head.

SCOPE-G replaces SCOPE-v1's mean-pool MLP head with a graph-propagation head, so the single
model internalizes the multi-hop 'smooth' collaborative signal directly. Item embeddings are
propagated K hops on a FROZEN item-item graph (co-occurrence kNN + text kNN) and summed
LightGCN-style to give E_prop; the set-completion objective is trained on E_prop with the
isotropy regularizer. The result is fused only with the training-free closed-form EASE base
(an in-framework data statistic) — no external trained model.

GPU-first. Tune on validation, test once with the trusted evaluator.
Reference bar (best baseline R@20/N@20), baby: 0.0965 / 0.0422.
"""
from __future__ import annotations
import sys, json, argparse, math, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import (Rmat, build_lists, closed_form_base, sigreg, evalS_trusted, zr, spmm, mm_affinity, ease_B, gram, BAR, DEV, OUT, CK)


def sym_norm(A):
    A = A.clone(); A.fill_diagonal_(0); d = A.sum(1).clamp(min=1e-6).pow(-0.5)
    return d.unsqueeze(1) * A * d.unsqueeze(0)


def build_item_graph(dset, G, kg=20, mode="both"):
    """Frozen item-item propagation graph. mode: 'both' (cooc+text, default), 'cooc', or 'text'.
    Ablation (mode!='both') isolates whether the graph gain needs the collaborative co-occurrence half."""
    parts = []
    if mode in ("both", "cooc"):
        Gc = G.clone(); Gc.fill_diagonal_(0)
        kth = torch.topk(Gc, kg, 1).values[:, -1:]; Gc = torch.where(Gc >= kth, Gc, torch.zeros_like(Gc))
        parts.append(sym_norm(Gc))
    if mode in ("both", "text") and dset.t_feat is not None:
        parts.append(sym_norm(mm_affinity(dset.t_feat[:], k=kg)))
    A = sum(parts) / len(parts)
    return A


class SCOPEG(nn.Module):
    def __init__(self, n_items, d=256, K=1, init=None):
        super().__init__()
        self.K = K
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))
        if init is not None: self.E.data.copy_(init)
        self.enc = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))
        self.layer_w = nn.Parameter(torch.ones(K + 1))

    def propagated(self, A):
        # LightGCN-style: E_prop = sum_k w_k A^k E  (multi-hop smooth reps), A frozen
        w = torch.softmax(self.layer_w, 0)
        out = w[0] * self.E; cur = self.E
        for k in range(1, self.K + 1):
            cur = A @ cur; out = out + w[k] * cur
        return out

    def latent(self, ctx_sum, n):
        z = ctx_sum / n.clamp(min=1).unsqueeze(1)
        return z + self.enc(z)                         # single residual MLP

    def logits_from(self, z, Ep):
        return (F.normalize(z, dim=1) @ F.normalize(Ep, dim=1).t()) / self.logtau.exp().clamp(min=1e-3)

    def forward_train(self, items, vmask, deg, A):
        Ep = self.propagated(A)
        Eit = Ep[items]
        keys = torch.where(vmask > 0, torch.rand_like(vmask), torch.full_like(vmask, 1e9))
        ranks = keys.argsort(1).argsort(1).float()
        nctx = (torch.rand(deg.shape, device=DEV) * (deg - 1).clamp(min=1)).floor() + 1
        nctx = torch.minimum(nctx, (deg - 1).clamp(min=1))
        ctx = ((ranks < nctx.unsqueeze(1)) & (vmask > 0)).float()
        tgt = ((ranks >= nctx.unsqueeze(1)) & (vmask > 0)).float()
        z = self.latent((Eit * ctx.unsqueeze(2)).sum(1), ctx.sum(1))
        return z, items, ctx, tgt, Ep

    @torch.no_grad()
    def score_all(self, R, deg, A):
        Ep = self.propagated(A)
        z = self.latent(torch.sparse.mm(R, Ep), deg)
        if z.shape[0] > 50000:        # huge user count: fp16 logits to avoid a 6GB+ fp32 [U,I] transient
            zt = F.normalize(z, dim=1).half(); Et = F.normalize(Ep, dim=1).half()
            return (zt @ Et.t()) / self.logtau.exp().clamp(min=1e-3).half()
        S = self.logits_from(z, Ep)
        return S.half() if S.shape[1] > 20000 else S    # fp16 score matrix for large datasets


def train(dataset, d=256, K=1, lr=3e-3, le=1.0, lz=0.0, epochs=220, bs=8192, patience=18, seed=2024, graph="both"):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dset = RecDataset(Config("scope", dataset))
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    bar_r, bar_n = BAR[dataset]
    gev = GPUEval(dset, "valid", DEV)
    G = (R.to_dense().t() @ R.to_dense()) if False else None
    half = dset.n_items > 20000 or dset.n_users > 50000
    G = gram(R)                                   # chunked for huge n_users (avoids a dense [U,I] transient)
    A = build_item_graph(dset, G, mode=graph); del G; torch.cuda.empty_cache()   # G only needed to build A
    base = closed_form_base(R, dset, gev, half=half)
    init = None
    if dset.t_feat is not None:
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
        Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), dim=0); init = (X @ Wp) / math.sqrt(d)
    model = SCOPEG(dset.n_items, d, K, init).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]; best = {"r": -1}; bad = 0
    for ep in range(epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt, Ep = model.forward_train(items[b], vmask[b], deg[b], A)
            logits = model.logits_from(z, Ep)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1)
            loss = -((logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)).mean() + le * sigreg(model.E)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            model.eval(); vr = gev.eval(model.score_all(R, degf, A))["Recall@20"]
            if vr > best["r"]: best = {"r": vr, "ep": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}; bad = 0
            else: bad += 1
            print(f"[{dataset}] SCOPE-G K{K} ep{ep:3d} val_R20={vr:.4f} best={best['r']:.4f}", flush=True)
            if bad >= patience: print(f"[{dataset}] early stop ep{ep}", flush=True); break
    model.load_state_dict(best["state"]); model.eval()
    Sz = zr(model.score_all(R, degf, A))
    pure = evalS_trusted(Sz, dset, "test")
    bg = (0.0, gev.eval(Sz)["Recall@20"])
    for g in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]:
        v = gev.eval(Sz + g * base)["Recall@20"]
        if v > bg[1]: bg = (g, v)
    g = bg[0]; fused = evalS_trusted(Sz + g * base, dset, "test")
    tr, tn = 1.1 * bar_r, 1.1 * bar_n
    p10 = fused["Recall@20"] > tr and fused["NDCG@20"] > tn
    print(f"[{dataset}] SCOPE-G pure R@20={pure['Recall@20']:.4f} N@20={pure['NDCG@20']:.4f}", flush=True)
    print(f"[{dataset}] SCOPE-G ⊕ EASE-base (g={g}) R@20={fused['Recall@20']:.4f}({(fused['Recall@20']/bar_r-1)*100:+.1f}%) "
          f"N@20={fused['NDCG@20']:.4f}({(fused['NDCG@20']/bar_n-1)*100:+.1f}%) | beats bar+10%? {'YES' if p10 else 'NO'}", flush=True)
    _sfx = ('' if seed == 2024 else f'_s{seed}') + ('' if graph == 'both' else f'_{graph}')
    (OUT / f"scope_g_{dataset}{_sfx}.json").write_text(json.dumps(
        {"dataset": dataset, "K": K, "pure": pure, "fused_with_ease": fused, "gamma": g, "beats_bar_plus10": bool(p10),
         "layer_w": torch.softmax(model.layer_w, 0).tolist()}, indent=2, default=str))
    print(f"[{dataset}] layer weights (hop 0..K): {[round(x,3) for x in torch.softmax(model.layer_w,0).tolist()]}", flush=True)
    return pure, fused


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="baby"); ap.add_argument("--K", type=int, default=1)  # default number of hops
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--graph", default="both", choices=["both", "cooc", "text"]); a = ap.parse_args()
    train(a.dataset, K=a.K, lr=a.lr, epochs=a.epochs, seed=a.seed, graph=a.graph)

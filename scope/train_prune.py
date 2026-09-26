#!/usr/bin/env python
"""Pruning study — which parts of the SCOPE-v1 head/base contribute.

Head architecture ablation (head_mode):  pool (linear mean-pool, NO MLP) | enc (one residual MLP) | encpred (two, current).
Regularizer ablation:                      le (SIGReg E), lz (SIGReg z).
Base 2-hop ablation:                       evaluate FUSED with the val-tuned base (c2 free) AND a c2=0 base (1-hop+text only).
Trains the head, fuses with each base (gamma val-tuned), tests once (trusted). Writes results/scope/exp_design/prune_{tag}.json.
"""
import sys, json, math, random, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope_full import (Rmat, build_lists, closed_form_base, sigreg, evalS_trusted, zr,
                   gram, ease_B, mm_affinity, spmm, BAR, DEV)
OUT = ROOT / "results" / "scope" / "exp_design"; OUT.mkdir(parents=True, exist_ok=True)


class SCOPEP(nn.Module):
    """SCOPE head with prunable architecture."""
    def __init__(self, n_items, d=256, init=None, head_mode='encpred'):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))
        if init is not None: self.E.data.copy_(init)
        self.head_mode = head_mode
        if head_mode in ('enc', 'encpred'):
            self.enc = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        if head_mode == 'encpred':
            self.pred = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))

    def latent(self, ctx_sum, n):
        z = ctx_sum / n.clamp(min=1).unsqueeze(1)
        if self.head_mode == 'pool': return z
        h = z + self.enc(z)
        if self.head_mode == 'enc': return h
        return h + self.pred(h)

    def logits_from(self, z):
        return (F.normalize(z, 1) @ F.normalize(self.E, 1).t()) / self.logtau.exp().clamp(min=1e-3)

    def forward_train(self, items, vmask, deg):
        Eit = self.E[items]
        keys = torch.where(vmask > 0, torch.rand_like(vmask), torch.full_like(vmask, 1e9))
        ranks = keys.argsort(1).argsort(1).float()
        n_ctx = (torch.rand(deg.shape, device=DEV) * (deg - 1).clamp(min=1)).floor() + 1
        n_ctx = torch.minimum(n_ctx, (deg - 1).clamp(min=1))
        ctx = ((ranks < n_ctx.unsqueeze(1)) & (vmask > 0)).float()
        tgt = ((ranks >= n_ctx.unsqueeze(1)) & (vmask > 0)).float()
        z = self.latent((Eit * ctx.unsqueeze(2)).sum(1), ctx.sum(1))
        return z, items, ctx, tgt

    @torch.no_grad()
    def score_all(self, R, deg):
        return self.logits_from(self.latent(torch.sparse.mm(R, self.E), deg))


def prune_base(R, dset, gevV, half, c2=0.0):
    """EASE base with the 2-hop coeff FIXED to c2 (0 => 1-hop+text only); lam,a val-tuned."""
    dt = torch.float16 if half else torch.float32
    G = gram(R); txt = None
    if dset.t_feat is not None:
        Aff = mm_affinity(dset.t_feat[:]); txt = zr(spmm(R, Aff)).to(dt); del Aff; torch.cuda.empty_cache()
    best = None
    for lam in [400, 800, 1500]:
        B = ease_B(G, lam); RB = spmm(R, B).to(dt)
        RB2 = (RB.float() @ B).to(dt) if c2 else None; del B; torch.cuda.empty_cache()
        S0 = zr(RB + c2 * RB2) if c2 else zr(RB); del RB
        if RB2 is not None: del RB2
        for a in ([0.0, 0.3, 0.5, 0.7] if txt is not None else [0.0]):
            S = S0 + a * txt if (txt is not None and a) else S0
            v = gevV.recall_per_user(S, 20).mean().item()
            if best is None or v > best[0]: best = (v, lam, a)
        del S0; torch.cuda.empty_cache()
    _, lam, a = best
    B = ease_B(G, lam); RB = spmm(R, B).to(dt); RB2 = (RB.float() @ B).to(dt) if c2 else None; del B, G
    S = zr(RB + c2 * RB2) if c2 else zr(RB)
    if txt is not None and a: S = S + a * txt
    torch.cuda.empty_cache(); return S


def fuse_test(Sz, base, dset, gevV):
    bg = (0.0, gevV.recall_per_user(Sz, 20).mean().item())
    for g in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]:
        v = gevV.recall_per_user(Sz + g * base, 20).mean().item()
        if v > bg[1]: bg = (g, v)
    return bg[0], evalS_trusted(Sz + bg[0] * base, dset, "test")


def run(ds, head_mode='encpred', le=0.0, lz=0.0, text_init=True, seed=2024, d=256, lr=3e-3, epochs=400, patience=20, bs=8192):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dset = RecDataset(Config("scope", ds)); R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    bar_r, bar_n = BAR[ds]; half = dset.n_items > 20000
    gevV = GPUEval(dset, "valid", DEV)
    base_tuned = closed_form_base(R, dset, gevV, half=half)
    init = None
    if text_init and dset.t_feat is not None:
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), 1)
        Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), 0); init = (X @ Wp) / math.sqrt(d)
    model = SCOPEP(dset.n_items, d, init, head_mode).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]; best = {"r20": -1}; bad = 0
    for ep in range(epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt = model.forward_train(items[b], vmask[b], deg[b]); logits = model.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1); tgt_lp = (logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)
            loss = -tgt_lp.mean() + le * sigreg(model.E) + (lz * sigreg(z) if lz else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            model.eval(); S = model.score_all(R, degf); vr = gevV.recall_per_user(S, 20).mean().item(); del S
            if vr > best["r20"]:
                best = {"r20": vr, "ep": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}; bad = 0
            else: bad += 1
            if bad >= patience: break
    model.load_state_dict(best["state"]); model.eval()
    Sz = zr(model.score_all(R, degf)).to(base_tuned.dtype)
    pure = evalS_trusted(Sz, dset, "test")
    g_t, fused_tuned = fuse_test(Sz, base_tuned, dset, gevV)
    base_c2_0 = prune_base(R, dset, gevV, half, c2=0.0)
    g_0, fused_c2_0 = fuse_test(Sz, base_c2_0, dset, gevV)
    tag = f"{ds}_{head_mode}_le{le}_lz{lz}_ti{int(text_init)}_s{seed}"
    rep = dict(dataset=ds, head_mode=head_mode, hp=dict(le=le, lz=lz, text_init=text_init, seed=seed),
               bar=dict(R20=bar_r, N20=bar_n), pure=pure,
               fused_tuned_base=dict(gamma=g_t, **fused_tuned),
               fused_c2_0_base=dict(gamma=g_0, **fused_c2_0))
    (OUT / f"prune_{tag}.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[{tag}] head={pure['Recall@20']:.4f} | fused(tuned base)={fused_tuned['Recall@20']:.4f}/{fused_tuned['NDCG@20']:.4f} "
          f"| fused(c2=0 base)={fused_c2_0['Recall@20']:.4f}/{fused_c2_0['NDCG@20']:.4f}  bar={bar_r}/{bar_n}", flush=True)
    del R, base_tuned, base_c2_0, Sz, model; torch.cuda.empty_cache()
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--head_mode", default="encpred", choices=["pool", "enc", "encpred"])
    ap.add_argument("--le", type=float, default=0.0); ap.add_argument("--lz", type=float, default=0.0)
    ap.add_argument("--no_text_init", action="store_true"); ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    run(a.dataset, head_mode=a.head_mode, le=a.le, lz=a.lz, text_init=not a.no_text_init, seed=a.seed)

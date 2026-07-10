#!/usr/bin/env python
"""SCOPE-v1 — recommendation as masked set completion.

A user's history is treated as a SET of items; a lightweight set-completion head predicts
a held-out part of the set from the rest, with no per-user parameters:
  mean-pool the observed item embeddings -> one residual MLP (enc) -> predicted latent;
  score every item by cosine / temperature.
Objective = set-completion softmax (predict the held-out items) + le * SIGReg(E), a sliced
isotropy regularizer (sliced Epps-Pulley on the item embeddings) that prevents collapse.
The set-completion scores are fused with a training-free closed-form base (1-hop EASE
co-occurrence + text-kNN affinity):  S = z(S_set) + gamma * z(S_base), gamma val-selected.

GPU-first; a fast GPU evaluator is used for validation and the final test score is
re-checked with the trusted full-sort evaluator. Tune on validation, test once.
Reference bars (best baseline R@20/N@20): baby .0965/.0422  sports .1067/.0470  clothing .0914/.0406.
"""
from __future__ import annotations
import sys, json, argparse, math, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from src.data.dataloader import EvalDataLoader
from src.evaluation.topk_evaluator import TopKEvaluator
from gpu_eval import GPUEval
DEV = torch.device("cuda:0")
OUT = ROOT / "results" / "scope"; OUT.mkdir(parents=True, exist_ok=True)
CK = ROOT / "ckpts" / "scope"; CK.mkdir(parents=True, exist_ok=True)
BAR = {"baby": (0.0965, 0.0422), "sports": (0.1067, 0.0470), "clothing": (0.0914, 0.0406),
       "microlens": (0.1017, 0.0432)}  # best non-mllmrec per metric (microlens bar = LGMRec, test)


def Rmat(dset):
    """Sparse user-item matrix on GPU (memory-frugal; only left-multiplications needed)."""
    u = torch.from_numpy(dset.train_users.astype(np.int64)); it = torch.from_numpy(dset.train_items.astype(np.int64))
    return torch.sparse_coo_tensor(torch.stack([u, it]), torch.ones(u.numel()),
                                   (dset.n_users, dset.n_items)).coalesce().to(DEV)


def spmm(Rsp, M):
    return torch.sparse.mm(Rsp, M)


def spmm_lowmem(R, M, dt, chunk=16384):
    """R(sparse [U,I]) @ M(dense [I,K]) in user-row chunks, output dtype dt — avoids a full fp32 [U,K]
    transient (fp16 sparse.mm is unsupported, so each chunk is computed fp32 then cast). For huge n_users."""
    idx, val = R.indices(), R.values(); U, K = R.shape[0], M.shape[1]
    out = torch.empty(U, K, dtype=dt, device=M.device)
    for s in range(0, U, chunk):
        e = min(s + chunk, U); m = (idx[0] >= s) & (idx[0] < e)
        rc = torch.sparse_coo_tensor(torch.stack([idx[0][m] - s, idx[1][m]]), val[m], (e - s, R.shape[1])).coalesce()
        out[s:e] = torch.sparse.mm(rc, M).to(dt); del rc
    return out


def gram(Rsp):
    U, I = Rsp.shape
    if U > 50000:                                  # huge user count: chunk to avoid a full dense [U,I] fp32 transient
        idx, val = Rsp.indices(), Rsp.values(); G = torch.zeros(I, I, device=Rsp.device)
        for s in range(0, U, 16384):
            e = min(s + 16384, U); m = (idx[0] >= s) & (idx[0] < e)
            rc = torch.sparse_coo_tensor(torch.stack([idx[0][m] - s, idx[1][m]]), val[m], (e - s, I)).coalesce()
            Rd = rc.to_dense(); G += Rd.t() @ Rd; del Rd, rc
        torch.cuda.empty_cache(); return G
    Rd = Rsp.to_dense(); G = Rd.t() @ Rd; del Rd; torch.cuda.empty_cache(); return G


def build_lists(dset, cap=60):
    u = dset.train_users.astype(np.int64); it = dset.train_items.astype(np.int64)
    per = [[] for _ in range(dset.n_users)]
    for uu, ii in zip(u, it): per[uu].append(ii)
    deg = np.array([len(p) for p in per]); L = min(cap, int(deg.max()))
    items = np.zeros((dset.n_users, L), np.int64); vmask = np.zeros((dset.n_users, L), np.float32)
    for uid, p in enumerate(per):
        if len(p) > L: p = random.sample(p, L)
        items[uid, :len(p)] = p; vmask[uid, :len(p)] = 1.0
    return torch.from_numpy(items).to(DEV), torch.from_numpy(vmask).to(DEV), torch.from_numpy(deg).to(DEV)


def ease_B(G, lam):
    Gr = G.clone(); Gr.diagonal().add_(lam); P = torch.linalg.inv(Gr)
    B = P / (-torch.diag(P).unsqueeze(0)); B.fill_diagonal_(0.0); return B


def mm_affinity(feat, k=20):
    X = F.normalize(torch.from_numpy(np.asarray(feat)).float().to(DEV), dim=1); S = X @ X.t()
    kth = torch.topk(S, k + 1, 1).values[:, -1:]
    S = torch.where(S >= kth, S, torch.zeros_like(S)).clamp(min=0); S.fill_diagonal_(0.0); return S


def zr(S):
    if S.shape[0] > 50000:                    # huge user count: per-row z-score in fp32 row-chunks (avoids a full fp32 upcast)
        out = torch.empty_like(S)
        for s in range(0, S.shape[0], 16384):
            e = min(s + 16384, S.shape[0]); blk = S[s:e].float()
            out[s:e] = ((blk - blk.mean(1, keepdim=True)) / (blk.std(1, keepdim=True) + 1e-9)).to(S.dtype)
            del blk
        return out
    Sf = S.float()
    out = (Sf - Sf.mean(1, keepdim=True)) / (Sf.std(1, keepdim=True) + 1e-9)
    return out.to(S.dtype)


def closed_form_base(R, dset, gev=None, lam=800, a=0.5, half=False):
    """1-hop EASE rollout (R*B) + text-kNN affinity. lam,a val-tuned if gev given.
    A 2-hop rollout term was found inert and is omitted, so the base is 1-hop EASE + text-kNN
    only. Memory-frugal (fp16 when half)."""
    frugal = R.shape[0] > 50000                    # huge user count: fp16 chunked R@X to avoid 6GB+ fp32 transients
    dt = torch.float16 if (half or frugal) else torch.float32
    def _ease(lm):                                 # zr(R @ EASE_B) -> dt  (non-frugal path byte-identical to original)
        return zr(spmm_lowmem(R, ease_B(G, lm), dt)) if frugal else zr(spmm(R, ease_B(G, lm)).to(dt))
    G = gram(R)
    txt = None
    if dset.t_feat is not None:
        Aff = mm_affinity(dset.t_feat[:])
        txt = zr(spmm_lowmem(R, Aff, dt)) if frugal else zr(spmm(R, Aff)).to(dt)
        del Aff; torch.cuda.empty_cache()
    def build(lm, aa):
        S = _ease(lm)
        if txt is not None and aa: S = S + aa * txt
        torch.cuda.empty_cache(); return S
    if gev is None:
        S = build(lam, a); del G; torch.cuda.empty_cache(); return S
    best = None
    for lm in [400, 800, 1500]:
        S0 = _ease(lm)
        for aa in ([0.0, 0.3, 0.5, 0.7] if txt is not None else [0.0]):
            S = S0 + aa * txt if (txt is not None and aa) else S0
            v = gev.eval(S)["Recall@20"]
            if best is None or v > best[0]: best = (v, dict(lam=lm, a=aa))
        del S0; torch.cuda.empty_cache()
    print(f"[base] tuned {best[1]} val_R20={best[0]:.4f} (1-hop base)", flush=True)
    S = build(best[1]["lam"], best[1]["a"])
    del G; torch.cuda.empty_cache(); return S


def sigreg(X, n_slices=256, num_points=17, T=5.0, sample=8192):
    N, d = X.shape
    Z = X[torch.randint(0, N, (sample,), device=X.device)] if N > sample else X
    W = F.normalize(torch.randn(d, n_slices, device=X.device), dim=0)
    P = Z @ W; P = (P - P.mean(0)) / (P.std(0) + 1e-6)
    t = torch.linspace(-T, T, num_points, device=X.device)
    tp = t.view(-1, 1, 1) * P.unsqueeze(0)
    re = tp.cos().mean(1); im = tp.sin().mean(1)
    tgt = torch.exp(-0.5 * t ** 2).view(-1, 1); w = torch.exp(-0.5 * t ** 2).view(-1, 1)
    return (((re - tgt) ** 2 + im ** 2) * w).mean()


class SCOPE(nn.Module):
    def __init__(self, n_items, d=256, init=None):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))
        if init is not None: self.E.data.copy_(init)
        self.enc = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))
        # A single residual MLP (enc) is used for the set-completion head.

    def latent(self, ctx_sum, n):
        z = ctx_sum / n.clamp(min=1).unsqueeze(1)
        return z + self.enc(z)                         # single residual MLP

    def logits_from(self, z):
        return (F.normalize(z, dim=1) @ F.normalize(self.E, dim=1).t()) / self.logtau.exp().clamp(min=1e-3)

    def forward_train(self, items, vmask, deg, p_mask=None):
        Eit = self.E[items]
        keys = torch.where(vmask > 0, torch.rand_like(vmask), torch.full_like(vmask, 1e9))
        ranks = keys.argsort(1).argsort(1).float()
        if p_mask is None:                                          # default: context size uniform in {1..deg-1}
            n_ctx = (torch.rand(deg.shape, device=DEV) * (deg - 1).clamp(min=1)).floor() + 1
        else:                                                       # fixed target fraction p_mask (masking-ratio ablation)
            n_ctx = (deg.float() * (1.0 - p_mask)).round().clamp(min=1)
        n_ctx = torch.minimum(n_ctx, (deg - 1).clamp(min=1))
        ctx = ((ranks < n_ctx.unsqueeze(1)) & (vmask > 0)).float()
        tgt = ((ranks >= n_ctx.unsqueeze(1)) & (vmask > 0)).float()
        z = self.latent((Eit * ctx.unsqueeze(2)).sum(1), ctx.sum(1))
        return z, items, ctx, tgt

    @torch.no_grad()
    def score_all(self, R, deg):
        z = self.latent(torch.sparse.mm(R, self.E), deg)
        if z.shape[0] > 50000:        # huge user count: fp16 logits to avoid a 6GB+ fp32 [U,I] transient
            zt = F.normalize(z, dim=1).half(); Et = F.normalize(self.E, dim=1).half()
            return (zt @ Et.t()) / self.logtau.exp().clamp(min=1e-3).half()
        return self.logits_from(z)


@torch.no_grad()
def evalS_trusted(S, dset, phase):
    loader = EvalDataLoader(dset, phase=phase, batch_size=2048)
    ev = TopKEvaluator(metrics=["Recall", "NDCG", "Precision"], topk=[10, 20]); ev.reset()
    for b in loader:
        uids = b["user_ids"]; s = S[uids].clone().float()
        hi, hv = b["history_indices"], b["history_values"]
        if hi.numel() > 0:
            hi = hi.to(DEV); hv = hv.to(DEV); mk = hv.bool()
            rr = torch.arange(s.size(0), device=DEV).unsqueeze(1).expand_as(hi)
            safe = torch.where(hi >= 0, hi, torch.zeros_like(hi)); s[rr[mk], safe[mk]] = float("-inf")
        _, tk = torch.topk(s, 20, -1); ev.collect(tk.cpu(), b["positive_items"], b["positive_lengths"])
    return ev.compute()


def train(dataset, d=256, lr=3e-3, wd=1e-6, le=1.0, epochs=400, bs=8192, patience=20,
          text_init=True, seed=2024, p_mask=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dset = RecDataset(Config("scope", dataset))
    R = Rmat(dset); items, vmask, deg = build_lists(dset)
    bar_r, bar_n = BAR.get(dataset, (None, None))
    half = dset.n_items > 20000 or dset.n_users > 50000   # fp16 score matrices for large datasets (clothing/elec/microlens)
    gev = GPUEval(dset, "valid", DEV); gevt = GPUEval(dset, "test", DEV)
    base = closed_form_base(R, dset, gev, half=half)
    init = None
    if text_init and dset.t_feat is not None:
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
        Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), dim=0); init = (X @ Wp) / math.sqrt(d)
    model = SCOPE(dset.n_items, d, init).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tu = torch.where(deg >= 2)[0]; degf = deg.float()
    best = {"r20": -1}; bad = 0
    for ep in range(epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]; tot = lrk = 0.0
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt = model.forward_train(items[b], vmask[b], deg[b], p_mask)
            logits = model.logits_from(z)                          # [B, N]
            # mask observed (context) items out of the prediction target space
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it)
            cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, dim=1)
            # gather log-prob at target items, averaged per user
            tgt_lp = (logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)
            lrank = -tgt_lp.mean()
            loss = lrank + le * sigreg(model.E)            # + isotropy regularizer on item embeddings
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); lrk += lrank.item()
        if ep % 4 == 0 or ep == epochs - 1:
            model.eval(); S = model.score_all(R, degf); vr = gev.eval(S)["Recall@20"]
            if vr > best["r20"]:
                best = {"r20": vr, "ep": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}; bad = 0
            else: bad += 1
            print(f"[{dataset}] ep{ep:3d} rank={lrk:.3f} val_R20={vr:.4f} best={best['r20']:.4f}", flush=True)
            if bad >= patience: print(f"[{dataset}] early stop ep{ep}", flush=True); break
    model.load_state_dict(best["state"]); model.eval()
    S = model.score_all(R, degf); Sz = zr(S).to(base.dtype)
    # fuse with closed-form base, tune gamma on val (GPUEval)
    bg = (0.0, gev.eval(Sz)["Recall@20"])
    for g in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]:
        v = gev.eval(Sz + g * base)["Recall@20"]
        if v > bg[1]: bg = (g, v)
    g = bg[0]
    # trusted test eval
    pure = evalS_trusted(Sz, dset, "test")
    fused = evalS_trusted(Sz + g * base, dset, "test")
    basef = evalS_trusted(base, dset, "test")
    rep = {"dataset": dataset, "model": "SCOPE-v1", "bar": {"R20": bar_r, "N20": bar_n}, "best_ep": best["ep"],
           "hp": dict(d=d, lr=lr, le=le, text_init=text_init), "gamma": g,
           "base": basef, "scope_pure": pure, "fused": fused}
    # checkpoint filename tag (the lz1.0 token is retained so the fusion loaders resolve the ckpt)
    tag = f"{dataset}_d{d}_le{le}_lz1.0_lr{lr}_s{seed}" if seed != 2024 else f"{dataset}_d{d}_le{le}_lz1.0_lr{lr}"
    if p_mask is not None: tag += f"_pm{p_mask}"      # namespace the masking-ratio ablation so it never clobbers the default ckpt
    (OUT / f"scope_{tag}.json").write_text(json.dumps(rep, indent=2, default=str))
    torch.save(best["state"], CK / f"scope_{tag}.pt")
    def line(nm, t):
        lr20 = (t["Recall@20"] / bar_r - 1) * 100 if bar_r else 0; ln20 = (t["NDCG@20"] / bar_n - 1) * 100 if bar_n else 0
        flag = " <== BEATS BAR (R&N)" if (bar_r and t["Recall@20"] > bar_r and t["NDCG@20"] > bar_n) else ""
        print(f"[{dataset}] {nm:10s} R@20={t['Recall@20']:.4f}({lr20:+.1f}%) N@20={t['NDCG@20']:.4f}({ln20:+.1f}%) "
              f"R@10={t['Recall@10']:.4f} N@10={t['NDCG@10']:.4f}{flag}", flush=True)
    print(f"\n[{dataset}] === SCOPE (bar R@20={bar_r} N@20={bar_n}) gamma={g} best_ep={best['ep']} ===", flush=True)
    line("BASE", basef); line("SCOPE", pure); line("FUSED", fused)
    del R; torch.cuda.empty_cache()
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby"); ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--le", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8192); ap.add_argument("--no_text_init", action="store_true")
    ap.add_argument("--patience", type=int, default=20); ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    train(a.dataset, d=a.d, lr=a.lr, le=a.le, epochs=a.epochs, bs=a.bs,
          patience=a.patience, text_init=not a.no_text_init, seed=a.seed)

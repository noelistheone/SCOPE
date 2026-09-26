#!/usr/bin/env python
"""Trained content-free / co-occurrence-seeded head control.

The null towers (random / co-occurrence kNN) are UNTRAINED, so gate->0 only rules out
"any second view helps" + "capacity helps". It does NOT rule out "the set head is just a second
TRAINED decorrelated view": maybe ANY competently-trained content-free head, fused with the base,
would be as complementary. This script closes that gap by holding the head ARCHITECTURE + capacity
+ fusion + eval FIXED and varying exactly two axes:

  SEEDING   : text (content)  |  cooc (CF co-occurrence)  |  random (no info)
  OBJECTIVE : set  (masked full-catalog set-completion softmax = SCOPE)  |  bpr (the CF objective
              FREEDOM/LightGCN use: pairwise, uniform negatives)

Every variant: same SCOPE head (mean-pool -> 1 residual MLP -> cos/tau), same SIGReg(E), same
closed-form base (EASE 1-hop + text-kNN, lam/a val-tuned ONCE, shared), same gamma-on-val fusion,
same trusted TopKEvaluator test. The decisive number is the FUSED-minus-BASE marginal:
  if (text,set) gives the super-additive +0.005-0.009 R@20 but (cooc,*)/(*,bpr) collapse to ~0,
  then the complementary signal is specifically the CONTENT-seeded SET-completion view, NOT a
  generic trained second view.

Reuses scope.py verbatim (imports its head/base/eval). Relative within-script comparison is the
claim (all variants identical except the two axes); absolute may differ ~0.003 from canonical
scope.py by GPU non-determinism (cf. train_prune.py). Tune on VAL, test once.
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
# reuse the REAL pipeline pieces
from scope import (Rmat, gram, build_lists, closed_form_base, sigreg, SCOPE,
                   evalS_trusted, zr, DEV, OUT, BAR)


def text_init(dset, d, seed):
    torch.manual_seed(seed)
    X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
    Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), dim=0)
    return (X @ Wp) / math.sqrt(d)


def cooc_init(G, d, seed):
    """CF analog of text_init: random projection of each item's (zero-diag, L2-normalized)
    co-occurrence profile G[i]=#users sharing items i,j. Items co-occurring with similar items
    start close -- an informative, CONTENT-FREE (purely interaction-derived) seeding."""
    torch.manual_seed(seed)
    C = G.clone(); C.fill_diagonal_(0.0); Cn = F.normalize(C, dim=1)
    Wp = F.normalize(torch.randn(Cn.shape[1], d, device=DEV), dim=0)
    init = (Cn @ Wp) / math.sqrt(d)
    del C, Cn, Wp; torch.cuda.empty_cache()
    return init


def make_init(mode, dset, G, d, seed):
    if mode == "text":   return text_init(dset, d, seed)
    if mode == "cooc":   return cooc_init(G, d, seed)
    if mode == "random": return None                      # SCOPE.__init__ default randn/sqrt(d)
    raise ValueError(mode)


def train_head(dataset, dset, R, items, vmask, deg, base, gev, gevt,
               seed_mode, objective, G=None, d=256, lr=3e-3, wd=1e-6, le=1.0,
               epochs=400, bs=8192, patience=20, n_neg=64, seed=2024):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    init = make_init(seed_mode, dset, G, d, seed)
    model = SCOPE(dset.n_items, d, init).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tu = torch.where(deg >= 2)[0]; degf = deg.float()
    best = {"r20": -1}; bad = 0
    for ep in range(epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]; lrk = 0.0
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt = model.forward_train(items[b], vmask[b], deg[b])
            if objective == "set":
                logits = model.logits_from(z)
                bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it)
                cm = ctx > 0
                logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
                logp = F.log_softmax(logits, dim=1)
                tgt_lp = (logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)
                lrank = -tgt_lp.mean()
            elif objective == "bpr":
                # same masked context/target split; pairwise BPR with uniform negatives
                zn = F.normalize(z, dim=1); En = F.normalize(model.E, dim=1)
                tau = model.logtau.exp().clamp(min=1e-3)
                s_pos = torch.einsum('bd,bld->bl', zn, En[it]) / tau            # [B,L]
                neg = torch.randint(0, dset.n_items, (b.numel(), n_neg), device=DEV)
                s_neg = torch.einsum('bd,bkd->bk', zn, En[neg]) / tau           # [B,n_neg]
                diff = s_pos.unsqueeze(2) - s_neg.unsqueeze(1)                   # [B,L,n_neg]
                bpr = -F.logsigmoid(diff) * tgt.unsqueeze(2)
                lrank = bpr.sum() / (tgt.sum() * n_neg).clamp(min=1)
            else:
                raise ValueError(objective)
            loss = lrank + le * sigreg(model.E)
            opt.zero_grad(); loss.backward(); opt.step(); lrk += lrank.item()
        if ep % 4 == 0 or ep == epochs - 1:
            model.eval(); S = model.score_all(R, degf); vr = gev.eval(S)["Recall@20"]
            if vr > best["r20"]:
                best = {"r20": vr, "ep": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}; bad = 0
            else: bad += 1
            if bad >= patience: break
    model.load_state_dict(best["state"]); model.eval()
    S = model.score_all(R, degf); Sz = zr(S).to(base.dtype)
    bg = (0.0, gev.eval(Sz)["Recall@20"])
    for g in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]:
        v = gev.eval(Sz + g * base)["Recall@20"]
        if v > bg[1]: bg = (g, v)
    g = bg[0]
    pure = evalS_trusted(Sz, dset, "test")
    fused = evalS_trusted(Sz + g * base, dset, "test")
    return {"seed_mode": seed_mode, "objective": objective, "gamma": g, "best_ep": best["ep"],
            "val_r20": best["r20"], "scope_pure": pure, "fused": fused}


def run(dataset, variants, d=256, lr=3e-3, le=1.0, epochs=400, bs=8192, patience=20, n_neg=64, seed=2024):
    dset = RecDataset(Config("scope", dataset))
    R = Rmat(dset); items, vmask, deg = build_lists(dset)
    bar_r, bar_n = BAR.get(dataset, (None, None))
    gev = GPUEval(dset, "valid", DEV); gevt = GPUEval(dset, "test", DEV)
    base = closed_form_base(R, dset, gev, half=(dset.n_items > 20000))
    basef = evalS_trusted(base, dset, "test")
    need_cooc = any(sm == "cooc" for sm, _ in variants)
    G = gram(R) if need_cooc else None
    results = {"dataset": dataset, "bar": {"R20": bar_r, "N20": bar_n},
               "base": basef, "hp": dict(d=d, lr=lr, le=le, n_neg=n_neg, seed=seed), "variants": []}
    for sm, obj in variants:
        torch.cuda.empty_cache()
        r = train_head(dataset, dset, R, items, vmask, deg, base, gev, gevt, sm, obj,
                       G=G, d=d, lr=lr, le=le, epochs=epochs, bs=bs, patience=patience, n_neg=n_neg, seed=seed)
        marg = r["fused"]["Recall@20"] - basef["Recall@20"]
        r["fused_minus_base_R20"] = marg
        results["variants"].append(r)
        beats = "above bar" if (bar_r and r["fused"]["Recall@20"] > bar_r and r["fused"]["NDCG@20"] > bar_n) else ""
        print(f"[{dataset}] {sm:6s}+{obj:3s}  pure_R20={r['scope_pure']['Recall@20']:.4f}  "
              f"fused_R20={r['fused']['Recall@20']:.4f}  fused-base={marg:+.4f}  "
              f"fused_N20={r['fused']['NDCG@20']:.4f}  gamma={r['gamma']}  {beats}", flush=True)
    if G is not None: del G; torch.cuda.empty_cache()
    out = OUT / (f"control_seedobj_{dataset}.json" if seed == 2024 else f"control_seedobj_{dataset}_s{seed}.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[{dataset}] base R@20={basef['Recall@20']:.4f} N@20={basef['NDCG@20']:.4f}  -> {out}", flush=True)
    print(f"[{dataset}] SUMMARY fused-minus-base R@20 by variant:", flush=True)
    for r in results["variants"]:
        print(f"    {r['seed_mode']:6s}+{r['objective']:3s}: {r['fused_minus_base_R20']:+.4f}", flush=True)
    del R; torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--variants", default="text:set,cooc:set,random:set,text:bpr,cooc:bpr",
                    help="comma list of seedmode:objective")
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--le", type=float, default=1.0); ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8192); ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--n_neg", type=int, default=64); ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    variants = [tuple(v.split(":")) for v in a.variants.split(",")]
    run(a.dataset, variants, d=a.d, lr=a.lr, le=a.le, epochs=a.epochs, bs=a.bs,
        patience=a.patience, n_neg=a.n_neg, seed=a.seed)

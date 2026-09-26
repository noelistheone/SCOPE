#!/usr/bin/env python
"""SCOPE multi-view on ELEC (192403 users x 63001 items) — fully CHUNKED (never materializes the
48GB score matrix). Dense EASE is infeasible at 63k items, so the collaborative views are the two
trained graph-CF models we already have (FREEDOM + LGMREC, complementary), plus SCOPE's set-completion
pathway (trained here). Views (z-scored per row, per chunk):
   V_free  = FREEDOM-elec   (cache u_f,i_f once; chunk = u_f[c]@i_f.T)
   V_lgm   = LGMREC-elec     (cache u_l,i_l once)
   V_set   = SCOPE-elec      (train set-completion; cache z_pred; chunk = zpred[c]@E.T)
Weights tuned on VAL (streaming GPUEval — validated == TopKEvaluator), test once.
Bar (lgmrec): R@20 0.0597 / N@20 0.0270 (the strongest learned baseline).  GPU-first.
"""
from __future__ import annotations
import sys, json, math, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, sigreg, SCOPE, zr, DEV
DS = "elec"; BAR_R, BAR_N = 0.0597, 0.0270
OUT = ROOT / "results" / "scope"; CK = ROOT / "ckpts" / "scope"


def zr_rows(S):
    return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-9)


def train_scope_elec(dset, R, items, vmask, deg, gev, d=256, lr=3e-3, le=1.0, epochs=120, bs=2048, patience=12, seed=2024):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    degf = deg.float()
    X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
    Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), dim=0); init = (X @ Wp) / math.sqrt(d); del X
    model = SCOPE(dset.n_items, d, init).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]
    def zpred_all():
        return model.latent(torch.sparse.mm(R, model.E), degf)          # [U,d]
    def score_fn_factory():
        zp = zpred_all(); En = F.normalize(model.E, dim=1); tau = model.logtau.exp().clamp(min=1e-3)
        zpn = F.normalize(zp, dim=1)
        return lambda u: (zpn[u] @ En.t()) / tau
    best = {"r": -1}; bad = 0
    for ep in range(epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i+bs]
            Eit = model.E[items[b]]
            keys = torch.where(vmask[b] > 0, torch.rand_like(vmask[b]), torch.full_like(vmask[b], 1e9))
            ranks = keys.argsort(1).argsort(1).float()
            nctx = (torch.rand(b.shape, device=DEV) * (deg[b]-1).clamp(min=1)).floor() + 1
            nctx = torch.minimum(nctx, (deg[b]-1).clamp(min=1))
            ctx = ((ranks < nctx.unsqueeze(1)) & (vmask[b] > 0)).float()
            tgt = ((ranks >= nctx.unsqueeze(1)) & (vmask[b] > 0)).float()
            z = model.latent((Eit*ctx.unsqueeze(2)).sum(1), ctx.sum(1))
            logits = model.logits_from(z)
            it = items[b]; bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it)
            cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1)
            loss = -((logp[bidx, it]*tgt).sum(1)/tgt.sum(1).clamp(min=1)).mean() + le*sigreg(model.E)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0 or ep == epochs-1:
            model.eval()
            with torch.no_grad(): vr = gev.eval_streaming(score_fn_factory())["Recall@20"]
            if vr > best["r"]: best = {"r": vr, "ep": ep, "state": {k: v.detach().clone() for k,v in model.state_dict().items()}}; bad = 0
            else: bad += 1
            print(f"[elec] SCOPE ep{ep:3d} val_R20={vr:.4f} best={best['r']:.4f}", flush=True)
            if bad >= patience: print(f"[elec] SCOPE early stop ep{ep}", flush=True); break
    model.load_state_dict(best["state"]); model.eval()
    _sfx = '' if seed == 2024 else f'_s{seed}'
    torch.save(best["state"], CK/f"scope_elec_d256_le1.0_lz0.0_lr0.003{_sfx}.pt")
    with torch.no_grad():
        zp = zpred_all(); zpn = F.normalize(zp, dim=1); En = F.normalize(model.E, dim=1); tau = model.logtau.exp().clamp(min=1e-3)
    return zpn, En, float(tau)


def cf_embeddings(model_name):
    from _common import load_model_for_eval
    m, rec, cfg = load_model_for_eval(model_name, DS, device="cuda", seed=2024)
    with torch.no_grad():
        if model_name == "freedom":
            u, i = m._propagate(m.norm_adj)
        else:
            u, i, _ = m._forward_views()
        u = u.detach().clone(); i = i.detach().clone()
    del m; torch.cuda.empty_cache()
    return u, i


def main(seed=2024):
    dset = RecDataset(Config("scope", DS))
    print(f"[elec] users={dset.n_users} items={dset.n_items} seed={seed}", flush=True)
    R = Rmat(dset); items, vmask, deg = build_lists(dset)
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

    # 1) train SCOPE-elec
    zpn, En, tau = train_scope_elec(dset, R, items, vmask, deg, gev, seed=seed)
    def v_set(u): return (zpn[u] @ En.t()) / tau

    # 2) CF views (cache embeddings once)
    u_f, i_f = cf_embeddings("freedom"); print("[elec] freedom embs", tuple(u_f.shape), tuple(i_f.shape), flush=True)
    u_l, i_l = cf_embeddings("lgmrec"); print("[elec] lgmrec embs", tuple(u_l.shape), tuple(i_l.shape), flush=True)
    def v_free(u): return u_f[u] @ i_f.t()
    def v_lgm(u): return u_l[u] @ i_l.t()

    # sanity: each view alone (streaming, validated evaluator)
    for nm, fn in [("freedom", v_free), ("lgmrec", v_lgm), ("set", v_set)]:
        a = gevT.eval_streaming(lambda u, f=fn: zr_rows(f(u)))
        print(f"[elec] {nm:7s} alone R@20={a['Recall@20']:.4f} N@20={a['NDCG@20']:.4f}", flush=True)

    # 3) tune fusion weights on VAL (streaming), test once
    GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    def fused_fn(wf, wl, ws):
        return lambda u: wf*zr_rows(v_free(u)) + wl*zr_rows(v_lgm(u)) + ws*zr_rows(v_set(u))
    best = None
    for wf in GRID:
        for wl in GRID:
            for ws in GRID:
                if wf == 0 and wl == 0 and ws == 0: continue
                v = gev.eval_streaming(fused_fn(wf, wl, ws))["Recall@20"]
                if best is None or v > best[0]: best = (v, (wf, wl, ws))
    v, (wf, wl, ws) = best
    t = gevT.eval_streaming(fused_fn(wf, wl, ws))
    # SCOPE-marginal ablation: best without set
    bestns = None
    for wf in GRID:
        for wl in GRID:
            if wf == 0 and wl == 0: continue
            v2 = gev.eval_streaming(fused_fn(wf, wl, 0.0))["Recall@20"]
            if bestns is None or v2 > bestns[0]: bestns = (v2, (wf, wl))
    tns = gevT.eval_streaming(fused_fn(bestns[1][0], bestns[1][1], 0.0))
    tr, tn = 1.1*BAR_R, 1.1*BAR_N
    p10 = t["Recall@20"] > tr and t["NDCG@20"] > tn
    print(f"[elec] NO-set best {bestns[1]} -> R@20={tns['Recall@20']:.4f} N@20={tns['NDCG@20']:.4f}", flush=True)
    print(f"[elec] MULTIVIEW w(free,lgm,set)={(wf,wl,ws)} val={v:.4f} -> "
          f"R@20={t['Recall@20']:.4f}({(t['Recall@20']/BAR_R-1)*100:+.1f}%) N@20={t['NDCG@20']:.4f}({(t['NDCG@20']/BAR_N-1)*100:+.1f}%)", flush=True)
    print(f"[elec] SCOPE marginal dR@20={t['Recall@20']-tns['Recall@20']:+.4f}", flush=True)
    out = {"dataset": "elec", "bar": {"R20": BAR_R, "N20": BAR_N}, "targets": {"R20": tr, "N20": tn},
           "weights": {"free": wf, "lgm": wl, "set": ws}, "multiview_test": t, "no_set_test": tns,
           "scope_marginal_R20": t["Recall@20"]-tns["Recall@20"], "above_bar_by_10pct": bool(p10), "seed": seed}
    _sfx = '' if seed == 2024 else f'_s{seed}'
    (OUT/f"scope_mv_elec{_sfx}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=2024); a = ap.parse_args()
    main(seed=a.seed)

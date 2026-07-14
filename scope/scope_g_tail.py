"""SCOPE-G tail localization. Retrains SCOPE-G (frozen co-occ+text graph-propagation head), saves its
checkpoint, and stratifies per-target-item Recall@20 by item train-popularity for SCOPE-G vs SCOPE-v1: does
SCOPE-G's graph propagation help most on low-popularity (tail) items? Writes
results/scope/scope_g_tail_<ds>.json and ckpts/scope/scope_g_<ds>.pt. Usage: python scope_g_tail.py [ds...]
"""
from __future__ import annotations
import sys, os, json, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from scope import Rmat, build_lists, gram, closed_form_base, zr, sigreg, SCOPE, DEV, ROOT
from scope_g import SCOPEG, build_item_graph
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

GAMMA_V1 = {"baby": 0.3, "sports": 0.3, "clothing": 0.6}


def train_g(dset, R, items, vmask, deg, A, gev, seed=2024, epochs=220, bs=8192, patience=18, le=1.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    init = None
    if dset.t_feat is not None:
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), 1)
        Wp = F.normalize(torch.randn(X.shape[1], 256, device=DEV), 0); init = (X @ Wp) / math.sqrt(256)
    m = SCOPEG(dset.n_items, 256, 1, init).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]; degf = deg.float(); best = {"r": -1}; bad = 0
    for ep in range(epochs):
        m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt, Ep = m.forward_train(items[b], vmask[b], deg[b], A)
            logits = m.logits_from(z, Ep)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1)
            loss = -((logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)).mean() + le * sigreg(m.E)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            m.eval(); vr = gev.eval(m.score_all(R, degf, A))["Recall@20"]
            if vr > best["r"]:
                best = {"r": vr, "state": {k: v.detach().clone() for k, v in m.state_dict().items()}}; bad = 0
            else: bad += 1
            print(f"[{dset.dataset_name}] SCOPE-G ep{ep:3d} val_R20={vr:.4f} best={best['r']:.4f}", flush=True)
            if bad >= patience: break
    m.load_state_dict(best["state"]); m.eval(); return m


def strat_recall(S, gevT, item_deg, edges):
    """Per-target-item Recall@20 stratified into 5 item-popularity bins (tail->head)."""
    U = gevT.users.numel(); dev = gevT.dev; nbin = len(edges) - 1
    hit = torch.zeros(nbin, device=dev); cnt = torch.zeros(nbin, device=dev)
    for s in range(0, U, 4096):
        bu = gevT.users[s:s + 4096]; sc = gevT._mask(S[bu].clone().float(), bu)
        _, top = torch.topk(sc, 20, 1)
        bp = gevT.pos[s:s + 4096]; validp = bp >= 0
        h = (top.unsqueeze(1) == bp.unsqueeze(2)).any(2) & validp
        pdeg = item_deg[bp.clamp(min=0)].float()
        binidx = torch.bucketize(pdeg, edges[1:-1].contiguous())
        for bb in range(nbin):
            mm = (binidx == bb) & validp
            hit[bb] += (h & mm).sum(); cnt[bb] += mm.sum()
    return (hit / cnt.clamp(min=1)).cpu().numpy(), cnt.cpu().numpy()


def run(ds):
    dset = RecDataset(Config("scope", ds)); dset.dataset_name = ds
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    G = gram(R); A = build_item_graph(dset, G, mode="both"); del G; torch.cuda.empty_cache()
    base = closed_form_base(R, dset, gev, half=half)

    v1 = SCOPE(dset.n_items, 256).to(DEV)
    v1.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); v1.eval()
    S_v1 = (zr(v1.score_all(R, degf)).to(base.dtype) + GAMMA_V1[ds] * base)

    mg = train_g(dset, R, items, vmask, deg, A, gev)
    torch.save(mg.state_dict(), ROOT / "ckpts" / "scope" / f"scope_g_{ds}.pt")
    Sz = zr(mg.score_all(R, degf, A)).to(base.dtype)
    bg = (0.0, gev.eval(Sz)["Recall@20"])
    for g in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]:
        v = gev.eval(Sz + g * base)["Recall@20"]
        if v > bg[1]: bg = (g, v)
    S_g = Sz + bg[0] * base

    item_deg = torch.zeros(dset.n_items, device=DEV)
    idx = R.coalesce().indices(); item_deg.index_add_(0, idx[1], torch.ones(idx.shape[1], device=DEV))
    edges = torch.quantile(item_deg[item_deg > 0], torch.linspace(0, 1, 6, device=DEV))
    r_v1, cnt = strat_recall(S_v1, gevT, item_deg, edges)
    r_g, _ = strat_recall(S_g, gevT, item_deg, edges)
    res = {"dataset": ds, "gamma_g": bg[0], "item_deg_quintile_edges": [round(float(x), 1) for x in edges.cpu().numpy()],
           "strata": [{"bin": i + 1, "n_targets": int(cnt[i]), "recall_v1": round(float(r_v1[i]), 4),
                       "recall_g": round(float(r_g[i]), 4), "g_minus_v1": round(float(r_g[i] - r_v1[i]), 4)}
                      for i in range(len(r_v1))]}
    json.dump(res, open(ROOT / "results" / "scope" / f"scope_g_tail_{ds}.json", "w"), indent=2)
    print(f"[{ds}] SCOPE-G minus SCOPE-v1 Recall@20 by item-popularity quintile (tail Q1 -> head Q5): "
          + " ".join(f"Q{s['bin']}:{s['g_minus_v1']:+.4f}" for s in res["strata"]), flush=True)
    del R, base; torch.cuda.empty_cache(); return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("SCOPE_G_TAIL_DONE", flush=True)

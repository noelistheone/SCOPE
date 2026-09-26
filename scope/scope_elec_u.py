#!/usr/bin/env python
"""Elec four-view composition: adds a sparse item-kNN co-occurrence view
(the 'sharp' EASE-proxy) to FREEDOM + LGMREC + SCOPE. Reuses the saved SCOPE-elec checkpoint (no
retrain). All chunked / streaming. Bar (lgmrec) R@20 0.0597 / N@20 0.0270.
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, SCOPE, DEV
from cooc_knn import build_cooc_knn
DS = "elec"; BAR_R, BAR_N = 0.0597, 0.0270


def zr_rows(S): return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-9)


def cf_embeddings(name):
    from _common import load_model_for_eval
    m, rec, cfg = load_model_for_eval(name, DS, device="cuda", seed=2024)
    with torch.no_grad():
        if name == "freedom": u, i = m._propagate(m.norm_adj)
        else: u, i, _ = m._forward_views()
        u = u.detach().clone(); i = i.detach().clone()
    del m; torch.cuda.empty_cache(); return u, i


def main(seed=2024):
    _sfx = '' if seed == 2024 else f'_s{seed}'
    dset = RecDataset(Config("scope", DS)); R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    # SCOPE-elec from ckpt
    model = SCOPE(dset.n_items, 256).to(DEV)
    model.load_state_dict(torch.load(ROOT/"ckpts"/"scope"/f"scope_elec_d256_le1.0_lz0.0_lr0.003{_sfx}.pt", map_location=DEV)); model.eval()
    with torch.no_grad():
        zp = model.latent(torch.sparse.mm(R, model.E), degf); zpn = F.normalize(zp, 1); En = F.normalize(model.E, 1)
        tau = model.logtau.exp().clamp(min=1e-3)
    # cooc-knn sharp view
    print("[elec] building cooc-knn...", flush=True)
    Gknn = build_cooc_knn(R, k=100, chunk=2048, device="cuda:0")
    u_f, i_f = cf_embeddings("freedom"); u_l, i_l = cf_embeddings("lgmrec")
    def v_free(u): return u_f[u] @ i_f.t()
    def v_lgm(u): return u_l[u] @ i_l.t()
    def v_set(u): return (zpn[u] @ En.t()) / tau
    def v_cooc(u):
        ru = torch.index_select(R, 0, u).to_dense()           # [c,N]
        return ru @ Gknn                                       # [c,N] dense via sparse rhs? Gknn sparse
    # Gknn is sparse [N,N]; ru[c,N] dense @ Gknn sparse -> use (Gknn.t() @ ru.t()).t()
    def v_cooc2(u):
        ru = torch.index_select(R, 0, u).to_dense()           # [c,N]
        return torch.sparse.mm(Gknn.t(), ru.t()).t()          # [c,N]
    for nm, fn in [("freedom", v_free), ("lgmrec", v_lgm), ("set", v_set), ("cooc", v_cooc2)]:
        a = gevT.eval_streaming(lambda u, f=fn: zr_rows(f(u)))
        print(f"[elec] {nm:7s} alone R@20={a['Recall@20']:.4f} N@20={a['NDCG@20']:.4f}", flush=True)
    def fused(wf, wl, ws, wc):
        return lambda u: wf*zr_rows(v_free(u)) + wl*zr_rows(v_lgm(u)) + ws*zr_rows(v_set(u)) + wc*zr_rows(v_cooc2(u))
    best = None
    for wf in [0.0, 1.0, 2.0, 3.0]:
        for wl in [0.0, 1.0, 2.0, 3.0]:
            for ws in [0.0, 1.0, 2.0]:
                for wc in [0.0, 0.5, 1.0, 2.0, 3.0]:
                    if wf==0 and wl==0 and ws==0 and wc==0: continue
                    v = gev.eval_streaming(fused(wf,wl,ws,wc))["Recall@20"]
                    if best is None or v > best[0]: best = (v, (wf,wl,ws,wc))
    v,(wf,wl,ws,wc) = best; t = gevT.eval_streaming(fused(wf,wl,ws,wc))
    tr, tn = 1.1*BAR_R, 1.1*BAR_N; p10 = t["Recall@20"]>tr and t["NDCG@20"]>tn
    print(f"[elec] 4VIEW w(free,lgm,set,cooc)={(wf,wl,ws,wc)} val={v:.4f} -> "
          f"R@20={t['Recall@20']:.4f}({(t['Recall@20']/BAR_R-1)*100:+.1f}%) N@20={t['NDCG@20']:.4f}({(t['NDCG@20']/BAR_N-1)*100:+.1f}%)", flush=True)
    (ROOT/"results"/"scope"/f"scope_mv_elec_4view{_sfx}.json").write_text(json.dumps(
        {"weights":{"free":wf,"lgm":wl,"set":ws,"cooc":wc},"test":t,"above_bar_by_10pct":bool(p10),"targets":{"R20":tr,"N20":tn},"seed":seed}, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=2024); a = ap.parse_args()
    main(seed=a.seed)

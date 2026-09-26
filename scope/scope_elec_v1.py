#!/usr/bin/env python
"""SCOPE-v1 on Elec (chunked). Dense EASE is infeasible (63k items: 16GB Gram + 16GB inverse > 24GB GPU
and > 31GB RAM), so the EASE base is replaced by the scalable sparse cooc-kNN operator (EASE-proxy).
SCOPE-v1-elec = zscore(set-completion) + gamma*zscore(cooc-kNN base). Uses the trained SCOPE-elec ckpt
+ build_cooc_knn. gamma tuned on val (streaming GPUEval = validated==TopKEvaluator); test once.
"""
import sys, json, math, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
_ap = argparse.ArgumentParser(); _ap.add_argument("--seed", type=int, default=2024); _SEED = _ap.parse_args().seed
_SFX = '' if _SEED == 2024 else f'_s{_SEED}'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, SCOPE, DEV
from cooc_knn import build_cooc_knn
BAR_R, BAR_N = 0.0597, 0.0270

dset = RecDataset(Config("scope", "elec"))
R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
print("[elec] building cooc-knn (sparse EASE-proxy)...", flush=True)
Gknn = build_cooc_knn(R, k=100, chunk=2048, device=DEV)
m = SCOPE(dset.n_items, 256).to(DEV)
m.load_state_dict(torch.load(ROOT/"ckpts"/"scope"/f"scope_elec_d256_le1.0_lz0.0_lr0.003{_SFX}.pt", map_location=DEV, weights_only=True)); m.eval()
with torch.no_grad():
    zp = m.latent(torch.sparse.mm(R, m.E), degf); zpn = F.normalize(zp, 1); En = F.normalize(m.E, 1)
    tau = m.logtau.exp().clamp(min=1e-3)

def zr(S): return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-9)
def v_set(u): return (zpn[u] @ En.t()) / tau
def v_cooc(u):
    ru = torch.index_select(R, 0, u).to_dense()
    return torch.sparse.mm(Gknn.t(), ru.t()).t()
def fused(g): return lambda u: zr(v_set(u)) + g * zr(v_cooc(u))

# sanity: each view alone
for nm, fn in [("set", lambda u: zr(v_set(u))), ("cooc", lambda u: zr(v_cooc(u)))]:
    a = gevT.eval_streaming(fn); print(f"[elec] {nm:5s} alone R@20={a['Recall@20']:.4f} N@20={a['NDCG@20']:.4f}", flush=True)
best = None
for g in [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]:
    v = gev.eval_streaming(fused(g))["Recall@20"]
    if best is None or v > best[0]: best = (v, g)
g = best[1]; t = gevT.eval_streaming(fused(g))
print(f"[elec] SCOPE-v1 (set+cooc) gamma={g} val={best[0]:.4f} -> R@20={t['Recall@20']:.4f}"
      f"(+{(t['Recall@20']/BAR_R-1)*100:.1f}%) N@20={t['NDCG@20']:.4f}(+{(t['NDCG@20']/BAR_N-1)*100:.1f}%)", flush=True)
(ROOT/"results"/"scope"/f"scope_v1_elec{_SFX}.json").write_text(json.dumps(
    {"dataset":"elec","seed":_SEED,"gamma":g,"fused":t,"note":"set-completion + sparse cooc-kNN (EASE-proxy; dense EASE infeasible at 63k)"}, indent=2, default=str))

#!/usr/bin/env python
"""SCOPE-G on Elec (chunked): graph-propagation set-completion head with the SPARSE cooc-kNN item-item
graph (dense item graph infeasible at 63k items), fused with the cooc-kNN base (EASE-proxy). Strictly
self-contained (one learned pathway + closed-form sparse operator, no external model). Chunked train/eval.
"""
import sys, json, math, random, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_ap = argparse.ArgumentParser(); _ap.add_argument("--seed", type=int, default=2024); _SEED = _ap.parse_args().seed
_SFX = '' if _SEED == 2024 else f'_s{_SEED}'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, sigreg, DEV
from cooc_knn import build_cooc_knn
BAR_R, BAR_N = 0.0597, 0.0270
torch.manual_seed(_SEED); np.random.seed(_SEED); random.seed(_SEED)
d = 256; K = 3

dset = RecDataset(Config("scope", "elec"))
R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
nu, ni = dset.n_users, dset.n_items
gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
print("[elec] building cooc-knn graph (sparse, for propagation + base)...", flush=True)
A = build_cooc_knn(R, k=100, chunk=2048, device=DEV)             # sparse [N,N], used as propagation graph
# symmetric-normalize A (sparse): D^-1/2 A D^-1/2
deg_i = torch.sparse.sum(A, 1).to_dense().clamp(min=1e-6).pow(-0.5)
ai = A.indices(); av = A.values() * deg_i[ai[0]] * deg_i[ai[1]]
A = torch.sparse_coo_tensor(ai, av, A.shape).coalesce()

class M(nn.Module):
    def __init__(s):
        super().__init__()
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), 1)
        Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), 0); s.E = nn.Parameter((X@Wp)/math.sqrt(d)); del X
        s.enc = nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        s.pred = nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        s.lw = nn.Parameter(torch.ones(K+1)); s.logtau = nn.Parameter(torch.tensor(math.log(0.1)))
    def Eprop(s):
        w = torch.softmax(s.lw, 0); out = w[0]*s.E; cur = s.E
        for k in range(1, K+1): cur = torch.sparse.mm(A, cur); out = out + w[k]*cur
        return out
    def lat(s, cs, n): z = cs/n.clamp(min=1).unsqueeze(1); h = z+s.enc(z); return h+s.pred(h)

m = M().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-6)
tu = torch.where(deg>=2)[0]; best={"r":-1}; bad=0
def score_fn_factory():
    Ep = m.Eprop(); zp = m.lat(torch.sparse.mm(R, Ep), degf); zpn = F.normalize(zp,1); En = F.normalize(Ep,1); tau = m.logtau.exp().clamp(min=1e-3)
    return lambda u: (zpn[u] @ En.t())/tau
for ep in range(120):
    m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
    for i in range(0, perm.numel(), 2048):
        b = perm[i:i+2048]; Ep = m.Eprop(); Eit = Ep[items[b]]
        keys = torch.where(vmask[b]>0, torch.rand_like(vmask[b]), torch.full_like(vmask[b],1e9))
        ranks = keys.argsort(1).argsort(1).float(); nctx=(torch.rand(b.shape,device=DEV)*(deg[b]-1).clamp(min=1)).floor()+1; nctx=torch.minimum(nctx,(deg[b]-1).clamp(min=1))
        ctx=((ranks<nctx.unsqueeze(1))&(vmask[b]>0)).float(); tgt=((ranks>=nctx.unsqueeze(1))&(vmask[b]>0)).float()
        z=m.lat((Eit*ctx.unsqueeze(2)).sum(1), ctx.sum(1)); lg=(F.normalize(z,1)@F.normalize(Ep,1).t())/m.logtau.exp().clamp(min=1e-3)
        it=items[b]; bidx=torch.arange(b.numel(),device=DEV).unsqueeze(1).expand_as(it); cm=ctx>0
        lg=lg.index_put((bidx[cm],it[cm]), torch.tensor(-1e9,device=DEV))
        loss=-((F.log_softmax(lg,1)[bidx,it]*tgt).sum(1)/tgt.sum(1).clamp(min=1)).mean() + 1.0*sigreg(m.E)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep%3==0:
        m.eval()
        with torch.no_grad(): vr=gev.eval_streaming(score_fn_factory())["Recall@20"]
        if vr>best["r"]: best={"r":vr,"st":{k:v.detach().clone() for k,v in m.state_dict().items()}}; bad=0
        else: bad+=1
        print(f"[elec] SCOPE-G ep{ep:3d} val_R20={vr:.4f} best={best['r']:.4f}", flush=True)
        if bad>=8: print(f"[elec] early stop ep{ep}", flush=True); break
m.load_state_dict(best["st"]); m.eval()
# cooc base (EASE-proxy) score
def zr(S): return (S-S.mean(1,keepdim=True))/(S.std(1,keepdim=True)+1e-9)
def v_cooc(u):
    ru = torch.index_select(R,0,u).to_dense(); return torch.sparse.mm(A.t(), ru.t()).t()
with torch.no_grad():
    Ep=m.Eprop(); zp=m.lat(torch.sparse.mm(R,Ep),degf); zpn=F.normalize(zp,1); En=F.normalize(Ep,1); tau=m.logtau.exp().clamp(min=1e-3)
def v_set(u): return (zpn[u]@En.t())/tau
def fused(g): return lambda u: zr(v_set(u)) + g*zr(v_cooc(u))
bg=None
for g in [0.0,0.3,0.6,1.0,1.5,2.0,3.0]:
    v=gev.eval_streaming(fused(g))["Recall@20"]
    if bg is None or v>bg[0]: bg=(v,g)
g=bg[1]; t=gevT.eval_streaming(fused(g))
print(f"[elec] SCOPE-G (graph-prop set + cooc) gamma={g} -> R@20={t['Recall@20']:.4f}"
      f"(+{(t['Recall@20']/BAR_R-1)*100:.1f}%) N@20={t['NDCG@20']:.4f}(+{(t['NDCG@20']/BAR_N-1)*100:.1f}%)", flush=True)
(ROOT/"results"/"scope"/f"scope_g_elec{_SFX}.json").write_text(json.dumps(
    {"dataset":"elec","seed":_SEED,"gamma":g,"fused_with_ease":t,"note":"graph-prop set-head + sparse cooc-kNN (dense item-graph infeasible at 63k)"}, indent=2, default=str))

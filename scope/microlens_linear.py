"""EASE-only and ADMM-SLIM linear baselines on MicroLens (98K users) — frugal version using
spmm_lowmem + chunked zr (the hardcoded baby/sports/clothing scripts OOM at this user count).
Val-tuned, trusted test eval. Results -> results/scope/microlens_linear.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scope import Rmat, gram, ease_B, spmm_lowmem, zr, evalS_trusted, DEV, ROOT
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval


def admm_slim(G, lam1, lam2, rho, iters=50, nonneg=True):
    n = G.shape[0]
    I = torch.eye(n, device=G.device, dtype=G.dtype)
    P = torch.linalg.inv(G + (lam2 + rho) * I)
    diagP = torch.diag(P)
    B = torch.zeros_like(G); C = torch.zeros_like(G); Gamma = torch.zeros_like(G)
    for _ in range(iters):
        B = P @ (G + rho * (C - Gamma))
        B = B - P * (torch.diag(B) / diagP).unsqueeze(0)
        A = B + Gamma
        C = torch.sign(A) * torch.clamp(A.abs() - lam1 / rho, min=0.0)
        if nonneg:
            C = torch.clamp(C, min=0.0)
        C.fill_diagonal_(0.0)
        Gamma = Gamma + B - C
    return C


ds = "microlens"
dt = torch.float16
dset = RecDataset(Config("scope", ds))
R = Rmat(dset)
gev = GPUEval(dset, "valid", DEV)
G = gram(R)

# ---- EASE-only (val-tune lambda) ----
best = None
for lam in [100, 400, 800, 1500, 3000]:
    S = zr(spmm_lowmem(R, ease_B(G, lam), dt))
    v = gev.eval(S)["Recall@20"]
    print(f"[microlens EASE] lam={lam} val_R20={v:.4f}", flush=True)
    if best is None or v > best[0]:
        best = (v, lam)
    del S; torch.cuda.empty_cache()
ease = evalS_trusted(zr(spmm_lowmem(R, ease_B(G, best[1]), dt)), dset, "test")
print("[microlens EASE] test", {k: round(ease[k], 4) for k in ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20"]}, flush=True)
torch.cuda.empty_cache()

# ---- ADMM-SLIM (val-tune) ----
Gf = G.float()
bestA = None
for nonneg in [False, True]:
    for lam2 in [200.0, 500.0, 1000.0]:
        for lam1 in [0.5, 2.0]:
            C = admm_slim(Gf, lam1, lam2, rho=lam2, iters=50, nonneg=nonneg)
            S = zr(spmm_lowmem(R, C, dt))   # C is fp32; spmm_lowmem casts the OUTPUT to dt (fp16-sparse.mm unsupported)
            v = gev.eval(S)["Recall@20"]
            print(f"[microlens ADMM] nonneg={nonneg} lam1={lam1} lam2={lam2} val_R20={v:.4f}", flush=True)
            if bestA is None or v > bestA[0]:
                bestA = (v, lam1, lam2, C.clone())
            del S, C; torch.cuda.empty_cache()
admm = evalS_trusted(zr(spmm_lowmem(R, bestA[3], dt)), dset, "test")
print("[microlens ADMM] test", {k: round(admm[k], 4) for k in ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20"]}, flush=True)

out = {
    "ease_only":  {"R10": round(ease["Recall@10"], 4), "N10": round(ease["NDCG@10"], 4),
                   "R20": round(ease["Recall@20"], 4), "N20": round(ease["NDCG@20"], 4), "lam": best[1]},
    "admmslim":   {"R10": round(admm["Recall@10"], 4), "N10": round(admm["NDCG@10"], 4),
                   "R20": round(admm["Recall@20"], 4), "N20": round(admm["NDCG@20"], 4),
                   "lam1": bestA[1], "lam2": bestA[2]},
}
(ROOT / "results" / "scope").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "results" / "scope" / "microlens_linear.json", "w"), indent=2)
print("MICROLENS_LINEAR_DONE", json.dumps(out), flush=True)

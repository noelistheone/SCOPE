"""ADMM-SLIM (Steck et al., WSDM 2020) as a baseline, val-tuned, trusted test eval.
Closed-form-ish item-item linear model with L1+L2 and nonneg/zero-diag constraints,
solved by ADMM. Dense item-item -> only baby/sports/clothing (Elec 63K infeasible, like EASE)."""
import sys, json, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scope import Rmat, gram, zr, spmm, evalS_trusted, DEV
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval

BAR = {"baby": (0.0965, 0.0422), "sports": (0.1067, 0.0470), "clothing": (0.0914, 0.0406)}


def admm_slim(G, lam1, lam2, rho, iters=40, nonneg=True):
    n = G.shape[0]
    I = torch.eye(n, device=G.device, dtype=G.dtype)
    P = torch.linalg.inv(G + (lam2 + rho) * I)          # precompute once
    diagP = torch.diag(P)
    B = torch.zeros_like(G); C = torch.zeros_like(G); Gamma = torch.zeros_like(G)
    for _ in range(iters):
        # B-update: least squares, then enforce diag(B)=0 (EASE-style Lagrangian correction)
        B = P @ (G + rho * (C - Gamma))
        B = B - P * (torch.diag(B) / diagP).unsqueeze(0)
        # C-update: soft-threshold + (optional) nonneg projection, zero diagonal
        A = B + Gamma
        C = torch.sign(A) * torch.clamp(A.abs() - lam1 / rho, min=0.0)
        if nonneg:
            C = torch.clamp(C, min=0.0)
        C.fill_diagonal_(0.0)
        Gamma = Gamma + B - C
    return C


out = {}
for ds in ["baby", "sports", "clothing"]:
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset)
    gev = GPUEval(dset, "valid", DEV)
    G = gram(R).float()
    best = None
    for nonneg in [False, True]:
        for lam2 in [200.0, 500.0, 1000.0]:
            for lam1 in [0.5, 2.0]:
                C = admm_slim(G, lam1, lam2, rho=lam2, iters=50, nonneg=nonneg)
                S = zr(spmm(R, C))
                v = gev.eval(S)["Recall@20"]
                print(f"[{ds}] nonneg={nonneg} lam1={lam1} lam2={lam2} val_R20={v:.4f}", flush=True)
                if best is None or v > best[0]:
                    best = (v, lam1, lam2, C)
                del C, S; torch.cuda.empty_cache()
    Sbest = zr(spmm(R, best[3]))
    test = evalS_trusted(Sbest, dset, "test")
    out[ds] = {"R10": round(test["Recall@10"], 4), "N10": round(test["NDCG@10"], 4),
               "R20": round(test["Recall@20"], 4), "N20": round(test["NDCG@20"], 4),
               "lam1": best[1], "lam2": best[2], "bar": BAR[ds]}
    print(ds, out[ds], flush=True)
    del G, R, best, Sbest; torch.cuda.empty_cache()

json.dump(out, open("results/scope/admmslim_baseline.json", "w"), indent=2)
print("ADMMSLIM_DONE", json.dumps(out))

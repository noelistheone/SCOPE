"""Lean 3-seed robustness numbers for MicroLens SCOPE-v1 (base+set) and SCOPE-U (base+set+GUME),
holding only 3 views (base, set, GUME) so it fits even under GPU contention (the full 4-view control
OOMs at 98K users when another job shares the GPU). Same gate-selection protocol. seeds 2025/2026."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scope import Rmat, build_lists, closed_form_base, SCOPE, evalS_trusted, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def gate_select(views, gev):
    best = None
    for ws in itertools.product(GR, repeat=len(views)):
        if all(w == 0 for w in ws):
            continue
        S = sum(w * v for w, v in zip(ws, views))
        vr = gev.eval(S)["Recall@20"]
        if best is None or vr > best[0]:
            best = (vr, ws)
        del S
    ws = best[1]
    return ws, sum(w * v for w, v in zip(ws, views))


for sd in [2025, 2026]:
    dset = RecDataset(Config("scope", "microlens"))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    gev = GPUEval(dset, "valid", DEV); dt = torch.float16
    base = zr(closed_form_base(R, dset, gev, half=True)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_microlens_d256_le1.0_lz1.0_lr0.003_s{sd}.pt", map_location=DEV)); m.eval()
    sset = zr(m.score_all(R, degf)).to(dt)
    gume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / "gume_microlens_scores.npy")).to(dt).to(DEV))
    _, Sv1 = gate_select([base, sset], gev); v1 = evalS_trusted(Sv1, dset, "test"); del Sv1; torch.cuda.empty_cache()
    _, Su = gate_select([gume, base, sset], gev); u = evalS_trusted(Su, dset, "test"); del Su; torch.cuda.empty_cache()
    out = {"seed": sd, "scope_v1": {"R20": v1["Recall@20"], "N20": v1["NDCG@20"]},
           "scope_u": {"R20": u["Recall@20"], "N20": u["NDCG@20"]}}
    json.dump(out, open(ROOT / "results" / "scope" / f"microlens_robust_s{sd}.json", "w"), indent=2)
    print(f"seed {sd}: SCOPE-v1 {v1['Recall@20']:.4f}/{v1['NDCG@20']:.4f}  SCOPE-U {u['Recall@20']:.4f}/{u['NDCG@20']:.4f}", flush=True)
    del R, base, sset, gume; torch.cuda.empty_cache()
print("MICROLENS_ROBUST_DONE", flush=True)

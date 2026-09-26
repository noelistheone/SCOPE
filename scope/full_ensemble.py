"""Establish the true ENSEMBLE CEILING: linear z-score fusion of ALL available frozen experts
{EASE+text base, FREEDOM, GUME, LGMRec, MGCN, LightGCN} via coordinate-ascent weights tuned on valid recall.
SCOPE-U composes a subset; the comparison shows how the base composes with several backbones at once. Reports all 4 metrics (R@10/N@10/R@20/N@20) on test vs the SCOPE-U bars. Writes results/scope/full_ensemble_<ds>.json.
Usage: python full_ensemble.py [ds...]
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scope import Rmat, closed_form_base, zr, evalS_trusted, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

PRED = ROOT / "results" / "baseline_scores"
CAND = ["freedom", "gume", "lgmrec", "mgcn", "lightgcn"]
GRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
SCOPE_U = {"baby": (0.0738, 0.0405, 0.1098, 0.0498), "sports": (0.0904, 0.0503, 0.1290, 0.0603),
           "clothing": (0.0761, 0.0417, 0.1092, 0.0501), "microlens": (0.0980, 0.0533, 0.1395, 0.0640)}


def run(ds):
    dset = RecDataset(Config("scope", ds)); R = Rmat(dset); dt = torch.float16
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    experts = {"base": zr(closed_form_base(R, dset, gevV, half=dset.n_items > 20000)).to(dt)}
    for name in CAND:
        p = PRED / f"{name}_{ds}_scores.npy"
        if p.exists(): experts[name] = zr(torch.from_numpy(np.load(p)).to(dt).to(DEV))
    names = list(experts.keys()); V = [experts[n] for n in names]
    # coordinate ascent on valid Recall@20
    w = {n: 0.0 for n in names}; w[names[0]] = 1.0
    def fused(wd): return sum(wd[n] * experts[n] for n in names)
    def vr(wd): return gevV.recall_per_user(fused(wd)).mean().item()
    for _ in range(4):
        for n in names:
            best = (w[n], vr(w))
            for g in GRID:
                w2 = dict(w); w2[n] = g; r = vr(w2)
                if r > best[1]: best = (g, r)
            w[n] = best[0]
    S = fused(w); m = evalS_trusted(S, dset, "test")
    bars = SCOPE_U.get(ds); beat = None
    if bars:
        got = (m["Recall@10"], m["NDCG@10"], m["Recall@20"], m["NDCG@20"])
        beat = [round(got[i] - bars[i], 4) for i in range(4)]
    res = {"dataset": ds, "experts": names, "weights": w,
           "metrics": {k: m[k] for k in ("Recall@10", "NDCG@10", "Recall@20", "NDCG@20")},
           "scope_u_bars": bars, "delta_over_scope_u": beat}
    json.dump(res, open(ROOT / "results" / "scope" / f"full_ensemble_{ds}.json", "w"), indent=2)
    wf = {n: round(v, 1) for n, v in w.items() if v > 0}
    print(f"[{ds}] full-ensemble R@10/N@10/R@20/N@20 = "
          f"{m['Recall@10']:.4f}/{m['NDCG@10']:.4f}/{m['Recall@20']:.4f}/{m['NDCG@20']:.4f} "
          f"| vs SCOPE-U Δ={beat} | weights={wf}", flush=True)
    del experts, V; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing", "microlens"]):
        try: run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("FULL_ENSEMBLE_DONE", flush=True)

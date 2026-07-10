"""EASE-only and EASE+text closed-form baselines, val-selected lambda/a, trusted test eval.
A strong shallow-linear baseline. Dense EASE is infeasible on
Elec (63K items), so we report Baby/Sports/Clothing; Elec uses the sparse co-occurrence proxy."""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np, torch
from scope import Rmat, gram, ease_B, spmm, zr, closed_form_base, evalS_trusted, DEV
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval

BAR = {"baby": (0.0965, 0.0422), "sports": (0.1067, 0.0470), "clothing": (0.0914, 0.0406)}
out = {}
for ds in ["baby", "sports", "clothing"]:
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset)
    half = dset.n_items > 20000          # fp16 on large sets (clothing) — control confirmed fp16==fp32 here
    dt = torch.float16 if half else torch.float32
    gev = GPUEval(dset, "valid", DEV)
    G = gram(R)
    # EASE-only: val-select lambda
    best = None
    for lam in [100, 400, 800, 1500, 3000]:
        S = zr(spmm(R, ease_B(G, lam)).to(dt))
        v = gev.eval(S)["Recall@20"]
        if best is None or v > best[0]:
            best = (v, lam)
        del S; torch.cuda.empty_cache()
    Sbest = zr(spmm(R, ease_B(G, best[1])).to(dt))
    ease = evalS_trusted(Sbest, dset, "test")
    del G, Sbest
    torch.cuda.empty_cache()
    # EASE+text base (closed_form_base val-selects lambda AND text weight a)
    base = closed_form_base(R, dset, GPUEval(dset, "valid", DEV), half=half)
    baset = evalS_trusted(base, dset, "test")
    out[ds] = {
        "ease_only": {"R10": round(ease["Recall@10"], 4), "N10": round(ease["NDCG@10"], 4),
                      "R20": round(ease["Recall@20"], 4), "N20": round(ease["NDCG@20"], 4), "lam": best[1]},
        "ease_text": {"R10": round(baset["Recall@10"], 4), "N10": round(baset["NDCG@10"], 4),
                      "R20": round(baset["Recall@20"], 4), "N20": round(baset["NDCG@20"], 4)},
        "bar": BAR[ds],
    }
    print(ds, out[ds], flush=True)
    del R, base
    torch.cuda.empty_cache()

json.dump(out, open("results/scope/ease_baseline.json", "w"), indent=2)
print("saved results/scope/ease_baseline.json")

"""Set-size stratification + user-segment equity, SCOPE-own pipeline.

Stratify TEST users into quintiles by training set size |S_u| (number of train interactions), and report
per-quintile mean Recall@20 for: base (EASE+text), SCOPE-v1 (gate[base,set]), SCOPE-U (gate[base,set,gume]),
and GUME. Two findings:
  (set-size) the SET-VIEW marginal  R20(SCOPE-v1) - R20(base)  vs |S_u| quintile -- does the
     set-completion complement grow with basket size? (a 'set' method should benefit larger sets).
  (equity) R20(SCOPE-U) - R20(GUME) per quintile + a fairness index min/max across quintiles --
     does SCOPE-U help users equitably, not just on easy/active users?
Writes results/scope/stratify_setsize_<ds>.json. GPU. Usage: python stratify_setsize.py <ds...>
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, ROOT
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
    return sum(w * v for w, v in zip(best[1], views))


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    base = zr(closed_form_base(R, dset, gevV, half=half)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        sset = zr(m.score_all(R, degf)).to(dt)
    gume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))

    v1 = gate_select([base, sset], gevV)            # SCOPE-v1
    su = gate_select([base, sset, gume], gevV)      # SCOPE-U
    rb = gevT.recall_per_user(base, 20).cpu().numpy()
    r1 = gevT.recall_per_user(v1, 20).cpu().numpy()
    ru = gevT.recall_per_user(su, 20).cpu().numpy()
    rg = gevT.recall_per_user(gume, 20).cpu().numpy()

    su_size = torch.sparse.sum(R, 1).to_dense().float()           # |S_u| per user id
    size = su_size[gevT.users].cpu().numpy()                      # aligned to test users
    # rank-based equal-count quintiles (robust to ties in the small discrete |S_u|)
    order = np.argsort(size, kind="stable"); qbin = np.empty(len(size), int)
    qbin[order] = (np.arange(len(size)) * 5 // len(size))

    out = {"dataset": ds, "n_test_users": int(len(size)), "quintiles": []}
    for k in range(5):
        msk = qbin == k
        out["quintiles"].append({
            "q": k, "n": int(msk.sum()), "mean_size": float(size[msk].mean()),
            "base": float(rb[msk].mean()), "scope_v1": float(r1[msk].mean()),
            "scope_u": float(ru[msk].mean()), "gume": float(rg[msk].mean()),
            "set_marginal_v1_minus_base": float(r1[msk].mean() - rb[msk].mean()),
            "scopeu_minus_gume": float(ru[msk].mean() - rg[msk].mean()),
        })
    # set-size trend (Spearman of set-marginal vs quintile) + equity fairness index
    sm = np.array([qd["set_marginal_v1_minus_base"] for qd in out["quintiles"]])
    ug = np.array([qd["scopeu_minus_gume"] for qd in out["quintiles"]])
    out["set_marginal_by_quintile"] = [float(x) for x in sm]
    out["set_marginal_low_vs_high"] = {"q0": float(sm[0]), "q4": float(sm[4])}
    out["scopeu_minus_gume_by_quintile"] = [float(x) for x in ug]
    out["equity_all_quintiles_positive"] = bool((ug > 0).all())
    json.dump(out, open(ROOT / "results" / "scope" / f"stratify_setsize_{ds}.json", "w"), indent=2)
    print(f"[{ds}] set-marginal(v1-base) by |S_u| quintile: " + " ".join(f"{x:+.4f}" for x in sm)
          + f"  | SCOPE-U-GUME by quintile: " + " ".join(f"{x:+.4f}" for x in ug)
          + f"  all+={out['equity_all_quintiles_positive']}", flush=True)
    del R, base, sset, gume, v1, su; torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        run(ds)
    print("STRATIFY_DONE", flush=True)

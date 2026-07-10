"""Significance vs a strong CF baseline + attribution ladder with per-step bootstrap.

For each dataset we build the z-scored views (base = EASE+text, set = SCOPE head, gume = GUME),
gate-select the ladder on VALIDATION Recall@20:
    L0 = base               (the parameter-free closed-form base)
    L1 = base + set         (SCOPE-v1: gate over [base,set])
    L2 = base + set + gume   (SCOPE-U: gate over [base,set,gume])
then compute TEST per-user Recall@20 for L0,L1,L2 and GUME and run paired user-level bootstraps:
    (#1) significance vs the strongest baseline GUME:  L1-GUME, L2-GUME
    (#2) attribution-ladder marginals:                  set-marginal=L1-L0, CF-marginal=L2-L1
Writes results/scope/significance_ladder_<ds>.json. GPU. Usage: python significance_ladder.py [datasets...]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, evalS_trusted, DEV, ROOT
from gpu_eval import GPUEval
from ensemble_control import gate_select
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset


def run(ds, seed=2024):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    dt = torch.float16
    gev = GPUEval(dset, "valid", DEV)
    gevT = GPUEval(dset, "test", DEV)

    V_item = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        V_set = zr(m.score_all(R, degf)).to(dt)
    V_gume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    views = {"item": V_item, "set": V_set, "gume": V_gume}

    # ladder models, gate-selected on validation
    w1, S1 = gate_select(views, ["item", "set"], gev)          # L1 = base + set  (SCOPE-v1)
    wg, Sg = gate_select(views, ["item", "gume"], gev)         # Lg = base + gume (set-free CF ensemble)
    w2, S2 = gate_select(views, ["item", "set", "gume"], gev)  # L2 = base + set + gume (SCOPE-U)
    S0 = V_item                                                # L0 = base

    # point test metrics (trusted evaluator)
    pt = {"base": evalS_trusted(S0, dset, "test"),
          "base+set": evalS_trusted(S1, dset, "test"),
          "base+gume": evalS_trusted(Sg, dset, "test"),
          "scope_u": evalS_trusted(S2, dset, "test"),
          "gume": evalS_trusted(V_gume, dset, "test")}

    # per-user TEST Recall@20 (aligned across views via the same test evaluator)
    ru0 = gevT.recall_per_user(S0).cpu().numpy()
    ru1 = gevT.recall_per_user(S1).cpu().numpy()
    rug = gevT.recall_per_user(Sg).cpu().numpy()
    ru2 = gevT.recall_per_user(S2).cpu().numpy()
    ruG = gevT.recall_per_user(V_gume).cpu().numpy()

    res = {
        "dataset": ds, "seed": seed,
        "gate_v1": list(w1), "gate_basegume": list(wg), "gate_u": list(w2),
        "point_R20": {k: v["Recall@20"] for k, v in pt.items()},
        "point_N20": {k: v["NDCG@20"] for k, v in pt.items()},
        # (#1) significance vs strongest baseline GUME
        "v1_vs_gume": paired_bootstrap(ru1, ruG),
        "u_vs_gume": paired_bootstrap(ru2, ruG),
        # (#2) attribution-ladder marginals, both decomposition orders
        "set_over_base": paired_bootstrap(ru1, ru0),          # base -> +set
        "gume_over_base": paired_bootstrap(rug, ru0),         # base -> +GUME  (the big CF lever)
        "cf_over_baseset": paired_bootstrap(ru2, ru1),        # base+set -> +GUME
        "set_over_basegume": paired_bootstrap(ru2, rug),      # base+GUME -> +set
    }
    json.dump(res, open(ROOT / "results" / "scope" / f"significance_ladder_{ds}.json", "w"), indent=2)
    print(f"[{ds}] R@20 base={pt['base']['Recall@20']:.4f} base+set={pt['base+set']['Recall@20']:.4f} "
          f"base+gume={pt['base+gume']['Recall@20']:.4f} U={pt['scope_u']['Recall@20']:.4f} GUME={pt['gume']['Recall@20']:.4f}", flush=True)
    print(f"   U-vs-GUME  d={res['u_vs_gume']['mean_delta']:+.4f} ci={[round(x,4) for x in res['u_vs_gume']['ci95']]} p={res['u_vs_gume']['p_two_sided']:.2g}", flush=True)
    print(f"   set/base d={res['set_over_base']['mean_delta']:+.4f} p={res['set_over_base']['p_two_sided']:.2g} | "
          f"GUME/base d={res['gume_over_base']['mean_delta']:+.4f} p={res['gume_over_base']['p_two_sided']:.2g} | "
          f"set/(base+GUME) d={res['set_over_basegume']['mean_delta']:+.4f} p={res['set_over_basegume']['p_two_sided']:.2g}", flush=True)
    del V_item, V_set, V_gume, S1, Sg, S2; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    dss = sys.argv[1:] or ["baby", "sports", "clothing"]
    for ds in dss:
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("SIGNIFICANCE_LADDER_DONE", flush=True)

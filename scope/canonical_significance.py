"""Canonical attribution significance, consistent with the main table.

"Set view over base" is defined as the deployed SCOPE-v1 (exactly the main-table row) minus the deployed
base, and that difference is bootstrapped. SCOPE-v1 = zr(set) + gamma * base_raw with gamma read from the
shipped scope_<ds>.json (baby .3 / sports .3 / clothing .6), reproducing the main table. All per-user
Recall@20 / NDCG@20 come from the same GPU test evaluator, so every marginal in the ladder is internally
consistent.

Outputs results/scope/canonical_sig_<ds>.json with base, SCOPE-v1, GUME, base+gume, SCOPE-U point R@20/N@20
and paired user-level bootstraps (B=1e4, two-sided) for: v1-base (canonical set/base), v1-gume, u-gume,
gume-base, u-(base+gume) (set on top of CF). Usage: python canonical_significance.py [baby sports clothing]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, evalS_trusted, DEV, ROOT
from gpu_eval import GPUEval
from ensemble_control import gate_select
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

GAMMA = {"baby": 0.3, "sports": 0.3, "clothing": 0.6}  # deployed SCOPE-v1 fusion weight (from scope_<ds>.json)


def run(ds):
    gamma = GAMMA[ds]
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    dt = torch.float16
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

    # --- memory-frugal: compute base + SCOPE-v1 first, free intermediates before loading the big GUME matrix ---
    base_raw = closed_form_base(R, dset, gev, half=half)          # deployed base (val-tuned lam,a) — NOT re-zscored
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        S_set = m.score_all(R, degf)
    Sz = zr(S_set).to(base_raw.dtype); del S_set; torch.cuda.empty_cache()
    S_v1 = Sz + gamma * base_raw                                  # exact deployed SCOPE-v1 (reproduces the main table)
    V_item = zr(base_raw).to(dt)                                  # z-scored base view (per-user ranking == base_raw)
    V_set = Sz.to(dt)                                             # z-scored set view (reuse Sz; ranking == S_set)
    pt_base = evalS_trusted(base_raw, dset, "test")
    pt_v1 = evalS_trusted(S_v1, dset, "test")
    ru_base = gevT.recall_per_user(V_item).cpu().numpy()          # per-user z-scoring preserves per-user ranking
    ru_v1 = gevT.recall_per_user(S_v1).cpu().numpy()
    del base_raw, Sz, S_v1; torch.cuda.empty_cache()

    V_gume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    views = {"item": V_item, "set": V_set, "gume": V_gume}
    wg, Sg = gate_select(views, ["item", "gume"], gev)           # base+gume (set-free CF ensemble)
    ru_g = gevT.recall_per_user(Sg).cpu().numpy(); pt_bg = evalS_trusted(Sg, dset, "test"); del Sg; torch.cuda.empty_cache()
    w2, S2 = gate_select(views, ["item", "set", "gume"], gev)    # SCOPE-U
    ru_u = gevT.recall_per_user(S2).cpu().numpy(); pt_u = evalS_trusted(S2, dset, "test"); del S2; torch.cuda.empty_cache()
    ru_G = gevT.recall_per_user(V_gume).cpu().numpy(); pt_gume = evalS_trusted(V_gume, dset, "test")

    pt = {"base": pt_base, "scope_v1": pt_v1, "base+gume": pt_bg, "scope_u": pt_u, "gume": pt_gume}

    res = {
        "dataset": ds, "gamma": gamma,
        "point_R20": {k: round(v["Recall@20"], 4) for k, v in pt.items()},
        "point_N20": {k: round(v["NDCG@20"], 4) for k, v in pt.items()},
        "gate_basegume": list(wg), "gate_u": list(w2),
        "set_over_base": paired_bootstrap(ru_v1, ru_base),        # CANONICAL: deployed SCOPE-v1 - base
        "v1_vs_gume": paired_bootstrap(ru_v1, ru_G),
        "u_vs_gume": paired_bootstrap(ru_u, ru_G),
        "gume_over_base": paired_bootstrap(ru_g, ru_base),
        "set_over_basegume": paired_bootstrap(ru_u, ru_g),        # set on top of CF (the small complement)
    }
    json.dump(res, open(ROOT / "results" / "scope" / f"canonical_sig_{ds}.json", "w"), indent=2)
    sb, gb, sbg = res["set_over_base"], res["gume_over_base"], res["set_over_basegume"]
    print(f"[{ds}] base={pt['base']['Recall@20']:.4f} SCOPE-v1={pt['scope_v1']['Recall@20']:.4f} "
          f"base+gume={pt['base+gume']['Recall@20']:.4f} U={pt['scope_u']['Recall@20']:.4f} GUME={pt['gume']['Recall@20']:.4f}", flush=True)
    print(f"   CANON set/base  d={sb['mean_delta']:+.4f} ci={[round(x,4) for x in sb['ci95']]} p={sb['p_two_sided']:.2g}", flush=True)
    print(f"   gume/base d={gb['mean_delta']:+.4f} p={gb['p_two_sided']:.2g} | set/(base+GUME) d={sbg['mean_delta']:+.4f} p={sbg['p_two_sided']:.2g}", flush=True)
    del V_item, V_set, V_gume; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("CANONICAL_SIG_DONE", flush=True)

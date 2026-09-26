"""Diagnostic: is there any set signal orthogonal to strong CF that helps top-K?

Both the set view and GUME are per-user z-scored, so the per-user OLS projection of z_set onto z_gume has
coefficient = the per-user correlation rho_u = mean_i(z_set[u,i]*z_gume[u,i]) (and intercept 0). Hence
    set_perp[u] = z_set[u] - rho_u * z_gume[u]
is EXACTLY the component of the set ranking that a strong-CF model cannot linearly explain, per user.

We then ask, against the base+GUME wall B = z(base) + g*z(gume) (g val-tuned):
  (i)   does set_perp ALONE predict held-out test items above chance?  (is the orthogonal part real?)
  (ii)  does B + w*set_perp beat B?   (per-user CF-orthogonal extraction)  <- the key
  (iii) how does (ii) compare to B + w*z(set) (the GLOBAL static-gate addition, ~ SCOPE-U's set step)?
If (ii) >> B and >> (iii), the complementary signal EXISTS and a per-user adaptive mechanism can harvest it
(the static scalar gate leaves it on the table). If (ii) ~ B, the wall is intrinsic to the set signal and a
different SIGNAL is needed. All comparisons via paired user-level bootstrap on test Recall@20.
Writes results/scope/w19_orthogonal_diag_<ds>.json. Usage: python w19_orthogonal_diag.py [datasets...]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset


def tune_add(gevV, baseB, cand, grid=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)):
    best = (0.0, gevV.recall_per_user(baseB).mean().item())
    for w in grid:
        r = gevV.recall_per_user(baseB + w * cand).mean().item()
        if r > best[1]: best = (w, r)
    return best[0]


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

    base = zr(closed_form_base(R, dset, gevV, half=half)).to(torch.float16)   # z(base)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        zset = zr(m.score_all(R, degf)).to(torch.float16)
    del m; torch.cuda.empty_cache()
    zgume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(torch.float16).to(DEV))

    # per-user rho = mean_i(zset*zgume); set_perp = zset - rho*zgume  (fp32 for the reduction)
    rho = (zset.float() * zgume.float()).mean(1, keepdim=True)          # [U,1] per-user correlation
    set_perp = (zset.float() - rho * zgume.float()).to(torch.float16)
    mean_rho = float(rho.mean())

    # base+GUME wall: z(base) + g*z(gume), g val-tuned
    g = tune_add(gevV, base, zgume, grid=(0.3, 0.6, 1.0, 1.5, 2.0, 3.0))
    B = base + g * zgume
    # (iii) global static-gate set addition
    w_glob = tune_add(gevV, B, zset)
    # (ii) per-user orthogonalized set addition
    w_perp = tune_add(gevV, B, set_perp)

    ru_B = gevT.recall_per_user(B).cpu().numpy()
    ru_glob = gevT.recall_per_user(B + w_glob * zset).cpu().numpy()
    ru_perp = gevT.recall_per_user(B + w_perp * set_perp).cpu().numpy()
    ru_perp_alone = gevT.recall_per_user(set_perp).cpu().numpy()
    ru_gume_alone = gevT.recall_per_user(zgume).cpu().numpy()

    res = {
        "dataset": ds, "mean_rho_set_gume": round(mean_rho, 4), "g_gume": g, "w_global_set": w_glob, "w_perp": w_perp,
        "R20": {"base+gume": round(float(ru_B.mean()), 4),
                "+global set": round(float(ru_glob.mean()), 4),
                "+perp set": round(float(ru_perp.mean()), 4),
                "perp alone": round(float(ru_perp_alone.mean()), 4),
                "gume alone": round(float(ru_gume_alone.mean()), 4)},
        "perp_vs_wall": paired_bootstrap(ru_perp, ru_B),        # KEY: does orthogonal set beat base+gume?
        "perp_vs_global": paired_bootstrap(ru_perp, ru_glob),   # does per-user orthogonal beat the static gate?
        "global_vs_wall": paired_bootstrap(ru_glob, ru_B),      # reproduces the ~0 set-over-(base+gume)
    }
    json.dump(res, open(ROOT / "results" / "scope" / f"w19_orthogonal_diag_{ds}.json", "w"), indent=2)
    pw, pg, gw = res["perp_vs_wall"], res["perp_vs_global"], res["global_vs_wall"]
    print(f"[{ds}] rho={mean_rho:.3f} | R20 base+gume={res['R20']['base+gume']} +global={res['R20']['+global set']} "
          f"+perp={res['R20']['+perp set']} (perp_alone={res['R20']['perp alone']}) "
          f"| perp-vs-wall d={pw['mean_delta']:+.4f} p={pw['p_two_sided']:.2g} "
          f"| perp-vs-global d={pg['mean_delta']:+.4f} p={pg['p_two_sided']:.2g} "
          f"| global-vs-wall d={gw['mean_delta']:+.4f} p={gw['p_two_sided']:.2g}", flush=True)
    del base, zset, zgume, set_perp, B; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W19_DIAG_DONE", flush=True)

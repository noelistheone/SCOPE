"""Statistical power and equivalence (TOST) for the set view's step inside SCOPE-U.

Once a strong collaborative view is present, the set view adds only a small, non-significant marginal
(set over base+GUME). This script quantifies whether that null is underpowered, using the exact canonical
construction of canonical_significance.py (deployed SCOPE-v1 = zr(set)+gamma*base; base+GUME and SCOPE-U via
the same validation-selected gate), from which the per-user Recall@20 delta arrays are extracted to compute:

  * observed effect (set over base+GUME): mean, per-user std, SE of the mean, n
  * reference effect delta_ref = the set-over-base marginal (the effect credited to the set view)
  * achieved POWER to detect delta_ref at this n (alpha=0.05 two-sided)
  * minimum detectable effect (MDE) at 80% power
  * TOST equivalence test with bound Delta = delta_ref: is the set-over-strong-CF effect statistically
    BOUNDED inside (-delta_ref, +delta_ref)? (p_equiv = max of the two one-sided p-values; the 90% CI
    lying inside the bound is the CI form of the same test.)

If power to detect delta_ref is high AND equivalence holds, the null is a genuine redundancy, not a
failure to detect. Writes results/scope/w16_power_<ds>.json. Usage: python w16_power.py [datasets...]
"""
from __future__ import annotations
import sys, os, json, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, ROOT
from gpu_eval import GPUEval
from ensemble_control import gate_select
from src.utils import Config
from src.data.dataset import RecDataset

GAMMA = {"baby": 0.3, "sports": 0.3, "clothing": 0.6}


def _phi(x):                                  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def power_analysis(delta_obs, delta_ref, se, n, alpha=0.05, target_power=0.80):
    z_a = 1.959963985                          # z_{1-alpha/2}, alpha=0.05
    z_b = 0.841621234                          # z_{0.80}
    # achieved power to detect an effect of size delta_ref with two-sided test at this SE
    lam = delta_ref / se
    power_ref = _phi(lam - z_a) + _phi(-lam - z_a)
    mde = (z_a + z_b) * se                      # minimum detectable effect at target_power
    # TOST equivalence with bound Delta = delta_ref
    Delta = delta_ref
    t_up = (delta_obs - Delta) / se; p_up = _phi(t_up)                 # H0: effect >= Delta
    t_lo = (delta_obs + Delta) / se; p_lo = 1.0 - _phi(t_lo)           # H0: effect <= -Delta
    p_equiv = max(p_up, p_lo)
    ci90 = [delta_obs - 1.6448536 * se, delta_obs + 1.6448536 * se]
    return {
        "power_to_detect_ref": round(power_ref, 4),
        "mde_at_80pct": round(mde, 5),
        "ref_over_mde": round(delta_ref / mde, 2),
        "tost_bound": round(Delta, 5),
        "tost_p_equiv": round(p_equiv, 4),
        "equivalent_at_0.05": bool(p_equiv < alpha),
        "ci90": [round(ci90[0], 5), round(ci90[1], 5)],
        "ci90_within_bound": bool(ci90[0] > -Delta and ci90[1] < Delta),
    }


def run(ds):
    gamma = GAMMA[ds]
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    dt = torch.float16
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

    base_raw = closed_form_base(R, dset, gev, half=half)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        S_set = m.score_all(R, degf)
    Sz = zr(S_set).to(base_raw.dtype); del S_set; torch.cuda.empty_cache()
    S_v1 = Sz + gamma * base_raw
    V_item = zr(base_raw).to(dt); V_set = Sz.to(dt)
    ru_base = gevT.recall_per_user(V_item).cpu().numpy()
    ru_v1 = gevT.recall_per_user(S_v1).cpu().numpy()
    del base_raw, Sz, S_v1; torch.cuda.empty_cache()

    V_gume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    views = {"item": V_item, "set": V_set, "gume": V_gume}
    _, Sg = gate_select(views, ["item", "gume"], gev)
    ru_g = gevT.recall_per_user(Sg).cpu().numpy(); del Sg; torch.cuda.empty_cache()
    _, S2 = gate_select(views, ["item", "set", "gume"], gev)
    ru_u = gevT.recall_per_user(S2).cpu().numpy(); del S2; torch.cuda.empty_cache()

    d_null = ru_u - ru_g                        # set on top of base+GUME (the observed n.s. effect)
    d_ref = ru_v1 - ru_base                      # set over base (the credited set-over-base effect)
    n = len(d_null)
    se_null = float(d_null.std(ddof=1) / math.sqrt(n))
    pa = power_analysis(float(d_null.mean()), float(d_ref.mean()), se_null, n)
    res = {
        "dataset": ds, "n_users": n,
        "set_over_basegume": {"mean_delta": round(float(d_null.mean()), 5),
                              "per_user_std": round(float(d_null.std(ddof=1)), 5),
                              "se_mean": round(se_null, 6)},
        "set_over_base_ref": {"mean_delta": round(float(d_ref.mean()), 5),
                              "per_user_std": round(float(d_ref.std(ddof=1)), 5)},
        "power": pa,
    }
    json.dump(res, open(ROOT / "results" / "scope" / f"w16_power_{ds}.json", "w"), indent=2)
    print(f"[{ds}] set/(base+GUME) obs={d_null.mean():+.4f} (SE {se_null:.5f}) | ref set/base={d_ref.mean():+.4f} "
          f"| power@ref={pa['power_to_detect_ref']:.3f} MDE80={pa['mde_at_80pct']:.4f} "
          f"(ref is {pa['ref_over_mde']}x MDE) | TOST p={pa['tost_p_equiv']:.3g} equiv={pa['equivalent_at_0.05']}", flush=True)
    del V_item, V_set, V_gume; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W16_POWER_DONE", flush=True)

"""Pairwise reproduction of the head: how much of the set head a pairwise item-item operator can mimic.

Every pairwise item--item recommender (EASE, SLIM, content-kNN) scores a target j by an ADDITIVE sum over the
observed set, [R P]_{u,j} = sum_{i in O_u} P_{ij}. We fit the BEST such pairwise operator to reproduce the
trained set head's own scores by ridge regression:
    P* = argmin_P || S_set - R P ||_F^2 + lam ||P||^2   ->   (G + lam I) P* = R^T S_set,   G = R^T R,
so R P* is the closest ANY pairwise recommender can come to the set head (lam tuned on validation). If R P*
recovers only part of the set head's test Recall@20, and the residual S_set - R P* still ranks held-out items
above chance, the head encodes higher-order set structure no pairwise kernel can represent -- the empirical
counterpart of the non-pairwise property. We also stratify the head-minus-mimic per-user
gain by basket size |S_u|: higher-order structure should grow with set size.
Writes results/scope/w20_higher_order_<ds>.json. Usage: python w20_higher_order.py [datasets...]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, gram, build_lists, closed_form_base, evalS_trusted, zr, SCOPE, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset


def fit_pairwise_mimic(G, RtS, lam):
    """P* = (G + lam I)^{-1} R^T S_set  (best additive item-item operator reproducing the set head)."""
    A = G.clone(); A.diagonal().add_(lam)
    P = torch.linalg.solve(A, RtS); del A; torch.cuda.empty_cache()
    return P


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    base = closed_form_base(R, dset, gev, half=half)

    # --- trained set head ---
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        S_set = m.score_all(R, degf).float()                       # [U,I] raw set-head logits
    del m; torch.cuda.empty_cache()

    # --- best pairwise mimic P* (ridge; lam tuned on val Recall of R P*) ---
    G = gram(R)
    Rt = R.t().coalesce()
    RtS = torch.sparse.mm(Rt, S_set)                               # [I,I]
    del Rt; torch.cuda.empty_cache()
    best = None
    for lam in [10.0, 100.0, 500.0, 1500.0]:
        P = fit_pairwise_mimic(G, RtS, lam)
        Smi = zr(torch.sparse.mm(R, P))
        vr = gev.eval(Smi.half() if half else Smi)["Recall@20"]; del Smi; torch.cuda.empty_cache()
        if best is None or vr > best[0]: best = (vr, lam)
    lam = best[1]
    P = fit_pairwise_mimic(G, RtS, lam)
    del G, RtS; torch.cuda.empty_cache()
    S_mimic = torch.sparse.mm(R, P); del P; torch.cuda.empty_cache()
    resid = S_set - S_mimic                                        # part the pairwise class cannot represent

    # --- test Recall: set head vs best pairwise mimic vs base vs residual ---
    m_set = evalS_trusted(zr(S_set), dset, "test")
    m_mimic = evalS_trusted(zr(S_mimic), dset, "test")
    m_base = evalS_trusted(base, dset, "test")
    m_resid = evalS_trusted(zr(resid), dset, "test")
    recovered = m_mimic["Recall@20"] / m_set["Recall@20"]         # fraction of set head recovered by best pairwise op

    # per-user paired bootstrap: set head vs its best pairwise mimic
    ru_set = gevT.recall_per_user(zr(S_set).half() if half else zr(S_set)).cpu().numpy()
    ru_mimic = gevT.recall_per_user(zr(S_mimic).half() if half else zr(S_mimic)).cpu().numpy()
    bs = paired_bootstrap(ru_set, ru_mimic)

    # --- E2: stratify (set head - mimic) per-user gain by basket size |S_u| ---
    gu = gevT.users.cpu().numpy(); du = deg.cpu().numpy()[gu]
    gain = ru_set - ru_mimic
    q = np.quantile(du, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(du, q)
    strata = []
    for b in range(5):
        mask = bins == b
        if mask.sum() > 0:
            strata.append({"quintile": b + 1, "median_setsize": float(np.median(du[mask])),
                           "n": int(mask.sum()), "mean_gain_R20": round(float(gain[mask].mean()), 5)})

    res = {"dataset": ds, "lam": lam,
           "set_head_R20": round(m_set["Recall@20"], 4), "set_head_N20": round(m_set["NDCG@20"], 4),
           "best_pairwise_mimic_R20": round(m_mimic["Recall@20"], 4), "base_R20": round(m_base["Recall@20"], 4),
           "residual_R20": round(m_resid["Recall@20"], 4),
           "pct_recovered_by_pairwise": round(100 * recovered, 1),
           "higher_order_gap_R20": round(m_set["Recall@20"] - m_mimic["Recall@20"], 4),
           "set_vs_mimic_bootstrap": bs, "setsize_strata": strata}
    json.dump(res, open(ROOT / "results" / "scope" / f"w20_higher_order_{ds}.json", "w"), indent=2)
    print(f"[{ds}] set-head R@20={m_set['Recall@20']:.4f} | best-pairwise-mimic={m_mimic['Recall@20']:.4f} "
          f"({res['pct_recovered_by_pairwise']}% recovered) | residual R@20={m_resid['Recall@20']:.4f} "
          f"| higher-order gap={res['higher_order_gap_R20']:+.4f} p={bs['p_two_sided']:.2g}", flush=True)
    print(f"[{ds}] set-vs-mimic gain by |S_u| quintile: "
          + " ".join(f"Q{s['quintile']}(|S|~{s['median_setsize']:.0f}):{s['mean_gain_R20']:+.4f}" for s in strata), flush=True)
    del S_set, S_mimic, resid, base, R; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W20_DONE", flush=True)

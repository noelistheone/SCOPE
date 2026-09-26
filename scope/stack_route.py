"""Is the per-user EASE-vs-FREEDOM preference PREDICTABLE (routable) or noise? Decides if the 0.13-0.15
oracle-switch headroom is real, learnable ceiling-breaking signal or an un-achievable union upper bound.

Tests on baby/sports/clothing (test per-user Recall@20):
  - per-user delta = r_freedom - r_ease; correlate with USER DEGREE and with a CONTENT signal (mean text-embed
    norm / diversity of the user's items) -> does model preference vary systematically with a feature?
  - degree-bucketed mean recall of EASE vs FREEDOM -> does the winner FLIP across buckets?
  - LEARNED ROUTER: logistic regression on user features (degree, mean item pop, content stats) -> predict
    which model to trust; label from VALID per-user preference; route TEST. Compare routed R@20 vs linear
    fusion vs true oracle. If routed > fusion (CI excludes 0), routing is real, learnable headroom.
Writes results/scope/stack_route_<ds>.json. Usage: python stack_route.py [ds...]
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from scope import Rmat, build_lists, closed_form_base, zr, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def best_fuse(views, gev):
    best = (-1.0, None)
    for ws in itertools.product(GR, repeat=len(views)):
        if all(w == 0 for w in ws): continue
        r = gev.recall_per_user(sum(w * v for w, v in zip(ws, views))).mean().item()
        if r > best[0]: best = (r, ws)
    return best[1]


def user_features(dset, R, gev):
    """Per-eval-user features aligned to gev.users: degree, mean item popularity, content centroid norm/spread."""
    deg_item = torch.sparse.sum(R, 0).to_dense().float()                 # item popularity
    deg_user = torch.sparse.sum(R, 1).to_dense().float()                 # user degree
    tf = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:]).astype(np.float32)).to(DEV), dim=1)
    Rd = R.to_dense()                                                     # [U,I] (dense ok for these sizes)
    ucent = (Rd @ tf) / deg_user.clamp(min=1).unsqueeze(1)              # user content centroid
    ucent_norm = ucent.norm(dim=1)                                       # centroid concentration
    mean_pop = (Rd @ deg_item) / deg_user.clamp(min=1)                  # avg popularity of user's items
    u = gev.users
    feats = torch.stack([deg_user[u], mean_pop[u], ucent_norm[u],
                         torch.log1p(deg_user[u]), torch.log1p(mean_pop[u])], dim=1)
    del Rd; torch.cuda.empty_cache()
    return feats.cpu().numpy(), deg_user[u].cpu().numpy()


def run(ds):
    dset = RecDataset(Config("scope", ds)); R = Rmat(dset); half = dset.n_items > 20000
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    base = zr(closed_form_base(R, dset, gevV, half=half)).to(torch.float16)
    fre = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"freedom_{ds}_scores.npy")).to(torch.float16).to(DEV))

    rb_t = gevT.recall_per_user(base).cpu().numpy(); rf_t = gevT.recall_per_user(fre).cpu().numpy()
    rb_v = gevV.recall_per_user(base).cpu().numpy(); rf_v = gevV.recall_per_user(fre).cpu().numpy()
    feats_t, deg_t = user_features(dset, R, gevT)
    feats_v, deg_v = user_features(dset, R, gevV)

    delta_t = rf_t - rb_t                                               # +ve => FREEDOM better for this user
    res = {"dataset": ds}
    # (1) does preference correlate with degree?
    res["corr_delta_vs_logdeg"] = float(np.corrcoef(np.log1p(deg_t), delta_t)[0, 1])
    # (2) degree-bucketed winner flip
    order = np.argsort(deg_t); nb = 5; n = len(deg_t); buckets = []
    for b in range(nb):
        idx = order[b * n // nb:(b + 1) * n // nb]
        buckets.append({"deg_range": [float(deg_t[idx].min()), float(deg_t[idx].max())],
                        "ease": float(rb_t[idx].mean()), "freedom": float(rf_t[idx].mean()),
                        "winner": "freedom" if rf_t[idx].mean() > rb_t[idx].mean() else "ease"})
    res["degree_buckets"] = buckets
    res["winner_flips_across_degree"] = len(set(b["winner"] for b in buckets)) > 1

    # (3) linear fusion + true oracle + LEARNED ROUTER
    w = best_fuse([base, fre], gevV); S = sum(x * v for x, v in zip(w, [base, fre]))
    r_fuse = gevT.recall_per_user(S).cpu().numpy(); del S
    r_oracle = np.maximum(rb_t, rf_t)
    # learned router: logistic regression (numpy) on VALID label (freedom better?) -> route TEST
    yv = (rf_v > rb_v).astype(np.float32)
    mu = feats_v.mean(0); sd = feats_v.std(0) + 1e-6
    Xv = (feats_v - mu) / sd; Xt = (feats_t - mu) / sd
    Xv = np.c_[Xv, np.ones(len(Xv))]; Xt = np.c_[Xt, np.ones(len(Xt))]
    wlr = np.zeros(Xv.shape[1])
    for _ in range(300):                                               # simple GD logistic regression
        p = 1 / (1 + np.exp(-Xv @ wlr)); wlr -= 0.1 * (Xv.T @ (p - yv) / len(yv) + 1e-3 * wlr)
    pt = 1 / (1 + np.exp(-Xt @ wlr))                                    # P(freedom better | features)
    route = np.where(pt >= 0.5, rf_t, rb_t)                            # hard route
    res["router_auc_valid"] = float(_auc(yv, 1 / (1 + np.exp(-Xv @ wlr))))
    res["R20"] = {"ease": float(rb_t.mean()), "freedom": float(rf_t.mean()), "fusion": float(r_fuse.mean()),
                  "learned_router": float(route.mean()), "oracle_switch": float(r_oracle.mean())}
    res["router_vs_fusion"] = paired_bootstrap(route, r_fuse)
    res["router_gain_over_fusion"] = float(route.mean() - r_fuse.mean())
    json.dump(res, open(ROOT / "results" / "scope" / f"stack_route_{ds}.json", "w"), indent=2)
    print(f"[{ds}] corr(delta,logdeg)={res['corr_delta_vs_logdeg']:+.3f} flip={res['winner_flips_across_degree']} "
          f"router_auc={res['router_auc_valid']:.3f} | ease={rb_t.mean():.4f} fre={rf_t.mean():.4f} "
          f"fusion={r_fuse.mean():.4f} ROUTER={route.mean():.4f} oracle={r_oracle.mean():.4f} "
          f"| router-fusion={res['router_gain_over_fusion']:+.4f} p={res['router_vs_fusion']['p_two_sided']:.2g}", flush=True)
    del base, fre; torch.cuda.empty_cache()
    return res


def _auc(y, p):
    o = np.argsort(p); y = y[o]; n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return 0.5
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try: run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("STACK_ROUTE_DONE", flush=True)

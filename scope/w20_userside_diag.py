"""User-side complement diagnostic (the dual of w19_orthogonal_diag).

SCOPE completes a user's set from ITEM co-occurrence. The dual is completing it from USER co-occurrence:
score item i for user u by how much u's neighbours (users sharing part of u's set) interact with i --
classic user-based CF, framed as user-side set completion. Against the base+GUME composition we test
whether the user-side view carries CF-complementary top-K signal:
  userknn[u,i] = sum_{v in kNN(u)} sim(u,v) * R[v,i],  sim = cosine of interaction rows, self removed.
Compare (all val-tuned add-weights, test Recall@20, paired user-level bootstrap):
  base+gume ; +global userknn ; +perp userknn (orthogonalized per-user vs gume) ; userknn alone.
Writes results/scope/w20_userside_diag_<ds>.json.
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, zr, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset


def userknn_scores(R, k=100, chunk=4096):
    """score[u,i] = sum_{v in top-k cosine nbrs of u} sim(u,v) R[v,i]. R sparse [U,I]. Returns dense [U,I]."""
    U, I = R.shape
    Rd = R.to_dense()                                  # [U,I] (fp32); ok for these sizes
    Rn = F.normalize(Rd, dim=1)                        # row-normalized for cosine
    out = torch.zeros(U, I, device=DEV)
    for s in range(0, U, chunk):
        e = min(s + chunk, U)
        sim = Rn[s:e] @ Rn.t()                         # [c, U] cosine
        idx = torch.arange(s, e, device=DEV)
        sim[torch.arange(e - s, device=DEV), idx] = -1.0     # remove self
        kth = torch.topk(sim, k, dim=1).values[:, -1:]        # [c,1] k-th largest
        sim = torch.where(sim >= kth, sim.clamp(min=0), torch.zeros_like(sim))  # keep top-k, nonneg
        out[s:e] = sim @ Rd                            # [c,I] = sparse-ish [c,U] @ [U,I]; no [c,k,I] transient
        del sim, kth
    del Rd, Rn; torch.cuda.empty_cache()
    return out


def tune_add(gevV, baseB, cand, grid=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)):
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

    base = zr(closed_form_base(R, dset, gevV, half=half)).to(torch.float16)
    zgume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(torch.float16).to(DEV))
    zuk = zr(userknn_scores(R)).to(torch.float16)

    rho = (zuk.float() * zgume.float()).mean(1, keepdim=True)
    uk_perp = (zuk.float() - rho * zgume.float()).to(torch.float16)
    mean_rho = float(rho.mean())

    g = tune_add(gevV, base, zgume, grid=(0.3, 0.6, 1.0, 1.5, 2.0, 3.0))
    B = base + g * zgume
    w_glob = tune_add(gevV, B, zuk)
    w_perp = tune_add(gevV, B, uk_perp)

    ru_B = gevT.recall_per_user(B).cpu().numpy()
    ru_glob = gevT.recall_per_user(B + w_glob * zuk).cpu().numpy()
    ru_perp = gevT.recall_per_user(B + w_perp * uk_perp).cpu().numpy()
    ru_uk_alone = gevT.recall_per_user(zuk).cpu().numpy()

    res = {
        "dataset": ds, "mean_rho_uk_gume": round(mean_rho, 4), "g_gume": g, "w_global_uk": w_glob, "w_perp": w_perp,
        "R20": {"base+gume": round(float(ru_B.mean()), 4),
                "+global userknn": round(float(ru_glob.mean()), 4),
                "+perp userknn": round(float(ru_perp.mean()), 4),
                "userknn alone": round(float(ru_uk_alone.mean()), 4)},
        "uk_vs_wall": paired_bootstrap(ru_glob, ru_B),      # KEY: user-side over base+gume
        "ukperp_vs_wall": paired_bootstrap(ru_perp, ru_B),  # orthogonal user-side over base+gume
    }
    json.dump(res, open(ROOT / "results" / "scope" / f"w20_userside_diag_{ds}.json", "w"), indent=2)
    uw, up = res["uk_vs_wall"], res["ukperp_vs_wall"]
    print(f"[{ds}] rho={mean_rho:.3f} | R20 base+gume={res['R20']['base+gume']} +uk={res['R20']['+global userknn']} "
          f"+ukperp={res['R20']['+perp userknn']} (uk_alone={res['R20']['userknn alone']}) "
          f"| uk-vs-wall d={uw['mean_delta']:+.4f} p={uw['p_two_sided']:.2g}{'  *SIG' if uw['p_two_sided']<0.05 and uw['mean_delta']>0 else ''} "
          f"| ukperp-vs-wall d={up['mean_delta']:+.4f} p={up['p_two_sided']:.2g}{'  *SIG' if up['p_two_sided']<0.05 and up['mean_delta']>0 else ''}", flush=True)
    del base, zgume, zuk, uk_perp, B; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W20_USERSIDE_DONE", flush=True)

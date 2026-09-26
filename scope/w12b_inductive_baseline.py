"""Dedicated inductive baselines for the truncated-context comparison. For long-history users (|S_u|>=8) encoded from only k
randomly-observed items, we score with:
  set-head    : SCOPE's learned set-completion head from the k items
  SCOPE-v1    : set-head fused with the closed-form base, applied inductively
  EASE-induct : the closed-form EASE item-item matrix B applied to the k-item context (R_k @ B + a R_k A^t)
                -- a strong, standard, PARAMETER-FREE inductive recommender (the dedicated baseline)
  content-kNN : R_k @ text-affinity  (content-based inductive)
  session-kNN : R_k @ co-occurrence  (the weaker baseline)
B and the affinities are fit on the FULL training data (item-item structure); only the user is new.
Paired user-level bootstrap of SCOPE-v1 vs the strongest baseline (EASE-inductive). Writes
results/scope/w12b_inductive_baseline_<ds>.json. Usage: python w12b_inductive_baseline.py [datasets...]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, gram, ease_B, mm_affinity, zr, SCOPE, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

LAM = {"baby": 800, "sports": 800, "clothing": 1500}
AA = {"baby": 0.5, "sports": 0.5, "clothing": 0.7}
GAMMA = {"baby": 0.3, "sports": 0.3, "clothing": 0.6}


def firstk(items, vmask, k, U, n_items):
    cols = items[:, :k]; vm = (vmask[:, :k] > 0)
    rows = torch.arange(U, device=DEV).unsqueeze(1).expand(U, k)[vm]
    Rk = torch.sparse_coo_tensor(torch.stack([rows, cols[vm]]), torch.ones(rows.numel(), device=DEV), (U, n_items)).coalesce()
    return Rk, vm.sum(1).float()


def run(ds, ks=(1, 2, 3)):
    dset = RecDataset(Config("scope", ds)); U = dset.n_users
    items, vmask, deg = build_lists(dset)
    G = gram(Rmat(dset))
    B = ease_B(G, LAM[ds])
    Aff = mm_affinity(dset.t_feat[:]) if dset.t_feat is not None else None
    Cooc = G.clone(); Cooc.fill_diagonal_(0); Cooc = Cooc / Cooc.sum(1).clamp(min=1e-6).unsqueeze(1)
    del G; torch.cuda.empty_cache()
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    gevT = GPUEval(dset, "test", DEV)
    long_mask = (deg >= 8).cpu().numpy(); gu = gevT.users.cpu().numpy(); keep = long_mask[gu]

    res = {"dataset": ds, "n_long_users": int(long_mask.sum()), "k": {}}
    def rec(S):  # per-long-user Recall@20 (frugal: cast per call, free immediately)
        return gevT.recall_per_user(S.float()).cpu().numpy()[keep]
    for k in ks:
        Rk, cnt = firstk(items, vmask, k, U, dset.n_items)
        R = {}
        with torch.no_grad():
            z = m.latent(torch.sparse.mm(Rk, m.E), cnt)
            S_set = zr(m.logits_from(z) if dset.n_items <= 30000 else (F.normalize(z, 1).half() @ F.normalize(m.E, 1).half().t()).float())
            S_ease = zr(torch.sparse.mm(Rk, B)) + (AA[ds] * zr(torch.sparse.mm(Rk, Aff)) if Aff is not None else 0)  # EASE-inductive
            S_v1 = S_set + GAMMA[ds] * S_ease
            R["set"] = rec(S_set); R["scope_v1"] = rec(S_v1); R["ease_induct"] = rec(S_ease)
            del S_set, S_v1; torch.cuda.empty_cache()          # free the 3 big matrices before the next 2
            R["content_knn"] = rec(torch.sparse.mm(Rk, Aff) if Aff is not None else S_ease * 0)
            del S_ease; torch.cuda.empty_cache()
            R["session_knn"] = rec(torch.sparse.mm(Rk, Cooc)); torch.cuda.empty_cache()
        bs_strong = paired_bootstrap(R["scope_v1"], R["ease_induct"])   # vs the STRONG dedicated baseline
        bs_set = paired_bootstrap(R["set"], R["ease_induct"])
        res["k"][f"k{k}"] = {n: round(float(v.mean()), 4) for n, v in R.items()}
        res["k"][f"k{k}"]["scope_v1_vs_ease"] = bs_strong
        res["k"][f"k{k}"]["set_vs_ease"] = bs_set
        ps = '<1e-3' if bs_strong['p_two_sided'] < 1e-3 else f"{bs_strong['p_two_sided']:.2g}"
        print(f"[{ds}] k={k} scope_v1={R['scope_v1'].mean():.4f} EASE-ind={R['ease_induct'].mean():.4f} "
              f"content={R['content_knn'].mean():.4f} sess={R['session_knn'].mean():.4f} | v1-vs-EASE d={bs_strong['mean_delta']:+.4f} p={ps}"
              f"{'  *SIG' if bs_strong['p_two_sided']<0.05 and bs_strong['mean_delta']>0 else ''}", flush=True)
        del Rk; torch.cuda.empty_cache()
    json.dump(res, open(ROOT / "results" / "scope" / f"w12b_inductive_baseline_{ds}.json", "w"), indent=2)
    del B, Aff, Cooc, m; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W12B_DONE", flush=True)

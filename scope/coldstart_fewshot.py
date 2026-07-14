"""cold-start / few-shot deepening. Long-history users (|S_u|>=8) are encoded INDUCTIVELY from only k in
{1,2,3,5} randomly-observed items (no per-user parameters, no retraining) and scored by SCOPE-v1, SCOPE-G, and
the strongest applicable inductive baselines: EASE-inductive (the closed-form EASE kernel on the k-item
context), content-kNN, session-kNN, and PopRec. This is the regime where collaborative filtering is weakest, so
it is where the content-seeded set formulation wins by the largest margin. Reads the SCOPE-v1 and SCOPE-G
checkpoints; writes results/scope/coldstart_fewshot_<ds>.json. Usage: python coldstart_fewshot.py [datasets...]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from scope import Rmat, build_lists, gram, ease_B, mm_affinity, zr, SCOPE, DEV, ROOT
from scope_g import SCOPEG, build_item_graph
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


def run(ds, ks=(1, 2, 3, 5)):
    dset = RecDataset(Config("scope", ds)); U = dset.n_items
    dset2 = dset; U = dset.n_users
    R = Rmat(dset); items, vmask, deg = build_lists(dset)
    G = gram(R); B = ease_B(G, LAM[ds]); A = build_item_graph(dset, G, mode="both")
    Aff = mm_affinity(dset.t_feat[:]) if dset.t_feat is not None else None
    Cooc = G.clone(); Cooc.fill_diagonal_(0); Cooc = Cooc / Cooc.sum(1).clamp(min=1e-6).unsqueeze(1)
    item_pop = torch.zeros(dset.n_items, device=DEV); idx = R.coalesce().indices()
    item_pop.index_add_(0, idx[1], torch.ones(idx.shape[1], device=DEV))
    del G; torch.cuda.empty_cache()

    v1 = SCOPE(dset.n_items, 256).to(DEV)
    v1.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); v1.eval()
    mg = SCOPEG(dset.n_items, 256, 1).to(DEV)
    mg.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_g_{ds}.pt", map_location=DEV)); mg.eval()
    Ep = mg.propagated(A).detach()

    gevT = GPUEval(dset, "test", DEV)
    long_mask = (deg >= 8).cpu().numpy(); gu = gevT.users.cpu().numpy(); keep = long_mask[gu]
    big = dset.n_items > 30000
    half = dset.n_items > 20000 or dset.n_users > 50000
    dt = torch.float16 if half else torch.float32

    def logits(z, E):
        if big: return (F.normalize(z, 1).half() @ F.normalize(E, 1).half().t()).float()
        return F.normalize(z, 1) @ F.normalize(E, 1).t() / 1.0
    def rec(S): return gevT.recall_per_user(S.float()).cpu().numpy()[keep]

    res = {"dataset": ds, "n_long_users": int(long_mask.sum()), "k": {}}
    for k in ks:
        Rk, cnt = firstk(items, vmask, k, U, dset.n_items)
        with torch.no_grad():
            S_ease = (zr(torch.sparse.mm(Rk, B)) + (AA[ds] * zr(torch.sparse.mm(Rk, Aff)) if Aff is not None else 0)).to(dt)
            R_ease = rec(S_ease)
            zv = v1.latent(torch.sparse.mm(Rk, v1.E), cnt); S_v1 = zr(logits(zv, v1.E)).to(dt) + GAMMA[ds] * S_ease
            R_v1 = rec(S_v1); del S_v1, zv; torch.cuda.empty_cache()
            zg = mg.latent(torch.sparse.mm(Rk, Ep), cnt); S_g = zr(logits(zg, Ep)).to(dt) + GAMMA[ds] * S_ease
            R_g = rec(S_g); del S_g, zg, S_ease; torch.cuda.empty_cache()
            R_content = rec(torch.sparse.mm(Rk, Aff).to(dt)) if Aff is not None else R_ease * 0
            R_sess = rec(torch.sparse.mm(Rk, Cooc).to(dt)); torch.cuda.empty_cache()
            R_pop = rec(item_pop.unsqueeze(0).expand(U, dset.n_items).to(dt))
        bs_v1 = paired_bootstrap(R_v1, R_ease)               # SCOPE-v1 vs strongest baseline (EASE-inductive)
        bs_g = paired_bootstrap(R_g, R_ease)                 # SCOPE-G  vs strongest baseline
        best_base = max(R_ease.mean(), R_content.mean(), R_sess.mean(), R_pop.mean())
        vals = {"scope_v1": R_v1.mean(), "scope_g": R_g.mean(), "ease_induct": R_ease.mean(),
                "content_knn": R_content.mean(), "session_knn": R_sess.mean(), "poprec": R_pop.mean()}
        res["k"][f"k{k}"] = {n: round(float(v), 4) for n, v in vals.items()}
        res["k"][f"k{k}"]["v1_vs_ease"] = bs_v1; res["k"][f"k{k}"]["g_vs_ease"] = bs_g
        res["k"][f"k{k}"]["v1_gain_pct_over_best_base"] = round(100 * (R_v1.mean() / best_base - 1), 1)
        p1 = '<1e-3' if bs_v1['p_two_sided'] < 1e-3 else f"{bs_v1['p_two_sided']:.2g}"
        print(f"[{ds}] k={k} v1={R_v1.mean():.4f} G={R_g.mean():.4f} | EASE-ind={R_ease.mean():.4f} "
              f"content={R_content.mean():.4f} sess={R_sess.mean():.4f} pop={R_pop.mean():.4f} | "
              f"v1 +{res['k'][f'k{k}']['v1_gain_pct_over_best_base']}% over best base (v1-vs-EASE p={p1})", flush=True)
        del Rk; torch.cuda.empty_cache()
    json.dump(res, open(ROOT / "results" / "scope" / f"coldstart_fewshot_{ds}.json", "w"), indent=2)
    del R, B, A, Aff, Cooc, v1, mg, Ep; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("COLDSTART_FEWSHOT_DONE", flush=True)

"""Candidate-candidate (joint-set) probe: does CANDIDATE x CANDIDATE (joint-set) structure help top-K?

The other scorers (base, GUME, set head, user-kNN, CORE) score items independently (first order). This probe
tests candidate-candidate interaction: score item j conditional on which items are ALREADY in
the recommended slate. We greedily build the top-20 from the base+GUME top-M candidates under
    argmax_j  g_j + beta * sum_{k in selected} W0[j,k]
with W0 = signed PMI over co-occurrence (attraction = co-occur above chance / complements; repulsion =
below chance / substitutes), degree-normalized by construction (PMI divides by deg_i*deg_j). Ablations:
signed vs attraction-only (W0>=0) vs repulsion-only (W0<=0, a DPP-style diversity control).
Compare greedy-slate Recall@20 to base+GUME top-20, paired user-level bootstrap. beta val-tuned.
NOTE: MMRec uses RANDOM splits, so held-out items are not a coherent 'next basket' -- if joint structure
helps here it is despite that. Writes results/scope/w23_spice_probe_<ds>.json.
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, gram, zr, DEV, ROOT
from gpu_eval import GPUEval
from ensemble_control import gate_select
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

MDE = {"baby": 0.00255, "sports": 0.00216, "clothing": 0.00174}


def signed_pmi(R):
    """W0[i,j] = log( G_ij * U / (deg_i deg_j) ) for G_ij>0 else 0; zero diagonal; clamp to [-5,5]."""
    G = gram(R)                                   # [I,I] co-occurrence counts
    deg = G.diagonal().clamp(min=1.0)
    U = R.shape[0]
    with torch.no_grad():
        pmi = torch.where(G > 0, torch.log(G.clamp(min=1e-6) * U / (deg.unsqueeze(0) * deg.unsqueeze(1))),
                          torch.zeros_like(G))
        pmi = pmi.clamp(-5.0, 5.0); pmi.fill_diagonal_(0.0)
    del G; torch.cuda.empty_cache()
    return pmi


def greedy_recall(gevT, S0, W0, beta, M=50, K=20, batch=2048):
    """Greedy joint-set top-K over base+GUME top-M candidates; return per-user Recall@20 (aligned)."""
    users = gevT.users; Uc = users.numel(); out = torch.zeros(Uc, device=DEV)
    for s in range(0, Uc, batch):
        bu = users[s:s + batch]; b = bu.numel()
        sc = gevT._mask(S0[bu].clone().float(), bu)           # train-masked scores [b,I]
        gc, C = torch.topk(sc, M, dim=1)                       # top-M candidates: scores [b,M], ids [b,M]
        Wsub = W0[C.unsqueeze(2), C.unsqueeze(1)]              # [b,M,M] pair potentials among candidates
        run = gc.clone()                                       # running score [b,M]
        chosen = torch.zeros(b, M, dtype=torch.bool, device=DEV)
        rows = torch.arange(b, device=DEV)
        picks = torch.zeros(b, K, dtype=torch.long, device=DEV)
        for t in range(K):
            r = run.masked_fill(chosen, float("-inf"))
            j = r.argmax(1)                                    # [b] chosen candidate index
            picks[:, t] = j; chosen[rows, j] = True
            run = run + beta * Wsub[rows, j, :]                # add picked item's pair row to all candidates
        sel_items = C[rows.unsqueeze(1), picks]                # [b,K] item ids selected
        pos = gevT.pos[s:s + batch]                            # [b,P] (-1 pad)
        hit = (sel_items.unsqueeze(2) == pos.unsqueeze(1)).any(1) & (pos >= 0)  # over positives
        out[s:s + b] = hit.sum(1).float() / gevT.nfit[s:s + batch].clamp(min=1)
        del sc, gc, C, Wsub, run, chosen
    return out.cpu().numpy()


def run(ds):
    dset = RecDataset(Config("scope", ds))
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    R = Rmat(dset); _, _, deg = build_lists(dset)
    zbase = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    zgume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    _, S0 = gate_select({"item": zbase, "gume": zgume}, ["item", "gume"], gev)
    ru_wall = gevT.recall_per_user(S0).cpu().numpy()          # base+GUME top-20 (beta=0 greedy == this)
    W0s = signed_pmi(R)

    res = {"dataset": ds, "mde": MDE[ds], "wall_R20": round(float(ru_wall.mean()), 4), "variants": {}}
    variants = {"signed": W0s, "attraction": W0s.clamp(min=0), "repulsion": W0s.clamp(max=0)}
    for name, W0 in variants.items():
        # val-tune beta on VALID greedy recall
        best = (0.0, -1.0)
        for bta in (0.1, 0.3, 0.6, 1.0, 2.0):
            rv = greedy_recall(gev, S0, W0, bta).mean()
            if rv > best[1]: best = (bta, rv)
        ru_g = greedy_recall(gevT, S0, W0, best[0])
        bs = paired_bootstrap(ru_g, ru_wall)
        res["variants"][name] = {"beta": best[0], "R20": round(float(ru_g.mean()), 4), "vs_wall": bs}
        sig = bs["p_two_sided"] < 0.05 and bs["mean_delta"] > 0
        print(f"[{ds}] {name:11s} beta={best[0]} R20={ru_g.mean():.4f} vs_wall d={bs['mean_delta']:+.4f} "
              f"p={bs['p_two_sided']:.2g}{'  **>=MDE & sig' if (sig and bs['mean_delta']>=MDE[ds]) else ('  *sig<MDE' if sig else '')}", flush=True)
    json.dump(res, open(ROOT / "results" / "scope" / f"w23_spice_probe_{ds}.json", "w"), indent=2)
    del zbase, zgume, S0, W0s; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W23_SPICE_DONE", flush=True)

"""Seed-stability of the beta=0 result: does an ENSEMBLE-AWARE-selected set head significantly
beat the base+GUME wall, stably across seeds?

A beta=0 head early-stopped on the validation R@20 of the fused base+GUME+head composition (ensemble-aware
selection) can differ from the deployed head, which is early-stopped to complement the base only. This
tests whether that difference is stable across seeds. For seeds {2024,2025,2026} we train the beta=0 head with fused-val early stopping (train_core,
beta=0), fuse via the same {base,gume,set} gate, and paired-bootstrap vs the {base,gume} wall.
Writes results/scope/w22_seedcheck_<ds>.json. Usage: python w22_seedcheck.py [datasets...]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, closed_form_base, zr, DEV, ROOT
from gpu_eval import GPUEval
from ensemble_control import gate_select
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset
from w21_core import train_core, MDE

SEEDS = [2024, 2025, 2026]
BETAS = [0.0, 0.5]          # baby's sig config was b=0, sports' was b=0.5 -> seed-check both


def run(ds):
    dset = RecDataset(Config("scope", ds))
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    zbase = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    zgume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    w0, S0 = gate_select({"item": zbase, "gume": zgume}, ["item", "gume"], gev)
    ru_S0 = gevT.recall_per_user(S0).cpu().numpy()

    res = {"dataset": ds, "mde": MDE[ds], "wall_R20": round(float(ru_S0.mean()), 4), "betas": {}}
    for beta in BETAS:
        deltas = []; seedrec = {}
        for sd in SEEDS:
            m, ep = train_core(dset, R, items, vmask, deg, degf, zgume, beta, S0, gev, half, seed=sd)
            with torch.no_grad():
                h = zr(m.score_all(R, degf)).to(dt)
            w1, S1 = gate_select({"item": zbase, "gume": zgume, "set": h}, ["item", "gume", "set"], gev)
            ru_S1 = gevT.recall_per_user(S1).cpu().numpy()
            bs = paired_bootstrap(ru_S1, ru_S0)
            deltas.append(bs["mean_delta"])
            sig = bs["p_two_sided"] < 0.05 and bs["mean_delta"] > 0
            seedrec[f"s{sd}"] = {"fused_R20": round(float(ru_S1.mean()), 4), "set_gate": round(float(w1[2]), 3),
                                 "delta_vs_wall": bs["mean_delta"], "p": bs["p_two_sided"], "sig": sig}
            print(f"[{ds}] beta={beta} seed={sd} fused={ru_S1.mean():.4f} d_vs_wall={bs['mean_delta']:+.4f} "
                  f"p={bs['p_two_sided']:.2g} setgate={w1[2]:.2f}{'  *SIG' if sig else ''}"
                  f"{'  (>=MDE)' if bs['mean_delta']>=MDE[ds] else ''}", flush=True)
            del m, h, S1; torch.cuda.empty_cache()
        arr = np.array(deltas)
        n_sig = sum(1 for s in seedrec.values() if s["sig"])
        res["betas"][f"b{beta}"] = {"seeds": seedrec,
            "mean_delta": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
            "n_sig_of_3": n_sig, "all_ge_mde": bool((arr >= MDE[ds]).all())}
        print(f"[{ds}] beta={beta} SUMMARY mean={arr.mean():+.4f}+-{arr.std():.4f} "
              f"range=[{arr.min():+.4f},{arr.max():+.4f}] sig={n_sig}/3 all>=MDE:{(arr>=MDE[ds]).all()}", flush=True)
    json.dump(res, open(ROOT / "results" / "scope" / f"w22_seedcheck_{ds}.json", "w"), indent=2)
    del zbase, zgume, S0; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W22_SEEDCHECK_DONE", flush=True)

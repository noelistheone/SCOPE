"""SCOPE-v2 supplementary numbers: MicroLens seeds and significance against GUME.

Two gaps are filled for the composed FREEDOM-backbone variant SCOPE-v2 (= gate[base, set, FREEDOM]),
which was added to the model family after the original robustness/significance runs:

  (A) MULTI-SEED: SCOPE-v2 test Recall@20/NDCG@20 on MicroLens for seeds 2025/2026
      (baby/sports/clothing already have three seeds from the ensemble-control runs). Uses the same
      gate-selection protocol (non-negative weights, grid-selected on validation Recall@20).
  (B) SIGNIFICANCE: paired user-level bootstrap of SCOPE-v2 against the strongest
      baseline GUME on baby/sports/clothing, mirroring the SCOPE-U vs GUME test already reported.

Writes results/scope/w23_scope_v2_supplement.json. Usage: python w23_scope_v2_supplement.py [datasets...]
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from scope import Rmat, build_lists, closed_form_base, SCOPE, evalS_trusted, zr, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

PRED = ROOT / "results" / "baseline_scores"
GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def gate_select(views, gev):
    """Non-negative gate over z-scored views, selected on validation Recall@20.
    The fused score is accumulated into ONE preallocated buffer: at 98K users a fresh
    [U,I] temporary per grid point exhausts the GPU when another job shares the device."""
    buf = torch.empty_like(views[0])
    best = None
    for ws in itertools.product(GR, repeat=len(views)):
        if all(w == 0 for w in ws):
            continue
        buf.zero_()
        for w, v in zip(ws, views):
            if w:
                buf.add_(v, alpha=float(w))
        vr = gev.eval(buf)["Recall@20"]
        if best is None or vr > best[0]:
            best = (vr, ws)
    del buf; torch.cuda.empty_cache()
    return best[1]


def build_views(ds, seed, need_gume):
    """Views built EXACTLY as in the ensemble-control run that produced the reported table numbers:
    every view is z-scored (including a second z-score over the closed-form base) and cast to fp16.
    GUME is loaded only when the significance test needs it (it is a second [U,I] matrix, and holding
    four views at 98K users can exhaust the GPU when another job shares the device)."""
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    dt = torch.float16
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    V_item = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    tag = f"scope_{ds}_d256_le1.0_lz1.0_lr0.003" + ("" if seed == 2024 else f"_s{seed}")
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"{tag}.pt", map_location=DEV)); m.eval()
    with torch.no_grad(): V_set = zr(m.score_all(R, degf)).to(dt)
    del m; torch.cuda.empty_cache()
    V_free = zr(torch.from_numpy(np.load(PRED / f"freedom_{ds}_scores.npy")).to(dt).to(DEV))
    V_gume = zr(torch.from_numpy(np.load(PRED / f"gume_{ds}_scores.npy")).to(dt).to(DEV)) if need_gume else None
    return dset, gev, gevT, V_item, V_set, V_free, V_gume


def run(ds, seeds=(2024,), do_sig=True):
    out = {"dataset": ds, "seeds": {}, "significance": None}
    for seed in seeds:
        need_gume = do_sig and seed == 2024
        dset, gev, gevT, V_item, V_set, V_free, V_gume = build_views(ds, seed, need_gume)
        # view order [col, item, set] matches the ensemble-control combo for SCOPE-v2 (base+set+FREEDOM)
        ws = gate_select([V_free, V_item, V_set], gev)
        S_v2 = ws[0] * V_free + ws[1] * V_item + ws[2] * V_set
        m_v2 = evalS_trusted(S_v2, dset, "test")
        out["seeds"][str(seed)] = {"gate_freedom_base_set": list(ws),
                                   "R@20": round(m_v2["Recall@20"], 4), "N@20": round(m_v2["NDCG@20"], 4),
                                   "R@10": round(m_v2["Recall@10"], 4), "N@10": round(m_v2["NDCG@10"], 4)}
        print(f"[{ds}] seed={seed} SCOPE-v2 gate={ws} R@20={m_v2['Recall@20']:.4f} N@20={m_v2['NDCG@20']:.4f}", flush=True)
        if seed == 2024:                                                 # significance vs strongest baseline
            ru_v2 = gevT.recall_per_user(S_v2.float()).cpu().numpy()
            ru_gume = gevT.recall_per_user(V_gume.float()).cpu().numpy()
            bs = paired_bootstrap(ru_v2, ru_gume)
            out["significance"] = {"comparison": "SCOPE-v2 - GUME", **bs}
            p = "<1e-3" if bs["p_two_sided"] < 1e-3 else f"{bs['p_two_sided']:.2g}"
            print(f"[{ds}] SCOPE-v2 - GUME  d={bs['mean_delta']:+.4f} CI={[round(c,4) for c in bs['ci95']]} p={p}", flush=True)
            del ru_v2, ru_gume
        del S_v2, V_item, V_set, V_free, V_gume, dset; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    args = sys.argv[1:] or ["baby", "sports", "clothing", "microlens"]
    outp = ROOT / "results" / "scope" / "w23_scope_v2_supplement.json"
    allres = json.load(open(outp)) if outp.exists() else {}              # merge with earlier partial runs
    for ds in args:
        seeds = (2024, 2025, 2026) if ds == "microlens" else (2024,)     # microlens needs the extra seeds
        try:
            allres[ds] = run(ds, seeds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
        json.dump(allres, open(outp, "w"), indent=2)                     # write after EACH dataset
    print("W23_DONE", flush=True)

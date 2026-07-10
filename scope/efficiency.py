"""Efficiency / scalability: concrete numbers behind 'trains in minutes, no per-user parameters'.

For each dataset we report: SCOPE head parameter count (item embeddings + one residual MLP + temperature;
NO per-user parameters), peak GPU memory and wall-clock for full-catalog inference (score_all), and the
closed-form base solve time. Training wall-clock is reported separately from the run logs. ID-embedding CF
baselines (GUME) additionally carry |U| user-embedding rows and do not scale to Electronics (63K items).
Writes results/scope/efficiency.json. GPU. Usage: python efficiency.py <ds...>
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from scope import Rmat, build_lists, closed_form_base, SCOPE, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    n_items, n_users = dset.n_items, dset.n_users
    m = SCOPE(n_items, 256).to(DEV)
    ck = ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt"
    if ck.exists(): m.load_state_dict(torch.load(ck, map_location=DEV))
    m.eval()
    head_params = sum(p.numel() for p in m.parameters())
    emb_params = m.E.numel(); mlp_params = head_params - emb_params
    # inference: peak memory + wall-clock for full-catalog scoring
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        S = m.score_all(R, degf)
    torch.cuda.synchronize(); infer_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    # base solve time (closed-form ridge inverse + text affinity)
    gev = GPUEval(dset, "valid", DEV)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    _ = closed_form_base(R, dset, gev, half=(n_items > 20000 or n_users > 50000))
    torch.cuda.synchronize(); base_s = time.perf_counter() - t1
    out = {"dataset": ds, "n_users": int(n_users), "n_items": int(n_items),
           "head_params_M": round(head_params / 1e6, 3), "item_emb_params_M": round(emb_params / 1e6, 3),
           "mlp_temp_params_K": round(mlp_params / 1e3, 1), "per_user_params": 0,
           "gume_user_emb_params_M_at_d64": round(n_users * 64 / 1e6, 3),
           "infer_score_all_s": round(infer_s, 3), "infer_peak_GB": round(peak_gb, 2),
           "base_solve_s": round(base_s, 2)}
    json.dump(out, open(ROOT / "results" / "scope" / f"efficiency_{ds}.json", "w"), indent=2)
    print(f"[{ds}] head={out['head_params_M']}M (emb {out['item_emb_params_M']}M + mlp/temp {out['mlp_temp_params_K']}K, "
          f"0 per-user) | infer {out['infer_score_all_s']}s/{out['infer_peak_GB']}GB | base solve {out['base_solve_s']}s "
          f"| GUME would add {out['gume_user_emb_params_M_at_d64']}M user rows", flush=True)
    del R, m, S; torch.cuda.empty_cache()


if __name__ == "__main__":
    agg = {}
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing", "elec", "microlens"]):
        try:
            run(ds)
            agg[ds] = json.load(open(ROOT / "results" / "scope" / f"efficiency_{ds}.json"))
        except Exception as e:
            print(f"[{ds}] SKIP ({type(e).__name__}: {e})", flush=True)
    json.dump(agg, open(ROOT / "results" / "scope" / "efficiency.json", "w"), indent=2)
    print("EFFICIENCY_DONE", flush=True)

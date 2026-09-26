"""Efficiency / footprint measurement (warm-up + median-of-N timings).
Params are deterministic; base-solve and full-catalog inference are timed with warmup + median-of-N so the
timings are stable. Writes results/scope/efficiency_clean_<ds>.json + efficiency_clean.json.
Usage: python efficiency_clean.py [datasets...]
"""
from __future__ import annotations
import sys, os, json, time, statistics
sys.path.insert(0, os.path.dirname(__file__))
import torch
from scope import Rmat, build_lists, closed_form_base, SCOPE, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset


def _median_time(fn, warmup=3, reps=7):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    n_items, n_users = dset.n_items, dset.n_users
    m = SCOPE(n_items, 256).to(DEV)
    ck = ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt"
    if ck.exists(): m.load_state_dict(torch.load(ck, map_location=DEV))
    m.eval()
    head_params = sum(p.numel() for p in m.parameters()); emb_params = m.E.numel()
    torch.cuda.reset_peak_memory_stats()
    infer_s = _median_time(lambda: m.score_all(R, degf))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    gev = GPUEval(dset, "valid", DEV)
    half = (n_items > 20000 or n_users > 50000)
    base_s = _median_time(lambda: closed_form_base(R, dset, None, half=half), warmup=1, reps=3)
    out = {"dataset": ds, "n_users": int(n_users), "n_items": int(n_items),
           "item_emb_params_M": round(emb_params / 1e6, 3),
           "mlp_temp_params_K": round((head_params - emb_params) / 1e3, 1), "per_user_params": 0,
           "gume_user_emb_params_M_at_d64": round(n_users * 64 / 1e6, 3),
           "infer_score_all_s": round(infer_s, 4), "infer_peak_GB": round(peak_gb, 2),
           "base_solve_s": round(base_s, 2)}
    json.dump(out, open(ROOT / "results" / "scope" / f"efficiency_clean_{ds}.json", "w"), indent=2)
    print(f"[{ds}] emb {out['item_emb_params_M']}M + enc/tau {out['mlp_temp_params_K']}K, 0 per-user "
          f"(GUME would add {out['gume_user_emb_params_M_at_d64']}M) | infer {out['infer_score_all_s']}s / "
          f"{out['infer_peak_GB']}GB | base solve {out['base_solve_s']}s", flush=True)
    del R, m; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    agg = {}
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing", "microlens"]):
        try:
            agg[ds] = run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] SKIP ({type(e).__name__}: {e})"); traceback.print_exc()
    json.dump(agg, open(ROOT / "results" / "scope" / "efficiency_clean.json", "w"), indent=2)
    print("EFFICIENCY_CLEAN_DONE", flush=True)

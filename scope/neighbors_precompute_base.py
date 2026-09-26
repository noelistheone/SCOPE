"""Build the deployed base for neighbors_matched.py in a separate, lean process and write it to that script's base cache.

Why: on a 24 GB card the full G7G13 process OOMs while building the Sports/Clothing base inside its dataset context
(it already holds the context tensors). This script runs exactly the same code path as DSCtx._base's build branch
(scope.closed_form_base on the same RecDataset, same dtype rule, same validation evaluator), and writes the .npy and the
metadata JSON in the format DSCtx._base expects, so the main script then loads the base from the cache and re-checks
its validation Recall@20 against the metadata before using it.

Usage: python neighbors_precompute_base.py --datasets sports clothing [--base_tag table1] [--out <g7g13 out dir>]
"""
import argparse, ast, contextlib, io, re, sys, time
from pathlib import Path

REV = Path(__file__).resolve().parent
sys.path.insert(0, str(REV))
import neighbors_matched as N                     # reuse its helpers, paths and seeding (no training happens on import)

import numpy as np
import torch

SC, GPUEval, RecDataset, Config = N.SC, N.GPUEval, N.RecDataset, N.Config


def build(name, args, base_cache):
    cache = base_cache / f"base_{name}_{args.base_tag}.npy"
    meta_p = base_cache / f"base_{name}_{args.base_tag}.json"
    if cache.exists() or meta_p.exists():
        print(f"[{name}] cache already exists ({cache}); not overwriting")
        return
    N.set_seed(args.seed, deterministic=False)
    dset = RecDataset(Config(args.config_model, name))
    R = SC.Rmat(dset)
    n_users, n_items = int(dset.n_users), int(dset.n_items)
    half = n_items > 20000 or n_users > 50000             # same rule as DSCtx
    gev = GPUEval(dset, "valid", N.DEV)
    buf = io.StringIO()
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(N.DEV)
    with contextlib.redirect_stdout(buf):
        Bg = SC.closed_form_base(R, dset, gev, half=half)
    out = buf.getvalue()
    print(out, end="", flush=True)
    m = re.search(r"\[base\] tuned (\{[^}]*\}) val_R20=([0-9.]+)", out)
    sel = ast.literal_eval(m.group(1)) if m else None
    peak = N.gpu_gb()
    val = gev.eval(Bg)
    test = SC.evalS_trusted(Bg, dset, "test")
    dep_p = Path(args.scope_json_dir) / f"scope_{name}_d256_le1.0_lz1.0_lr0.003.json"
    dep = N.load_json(dep_p)
    repro = None
    if dep and "base" in dep:
        dR = abs(test["Recall@20"] - dep["base"]["Recall@20"])
        dN = abs(test["NDCG@20"] - dep["base"]["NDCG@20"])
        repro = {"deployed_json": str(dep_p), "R20_abs_diff": dR, "N20_abs_diff": dN, "ok": max(dR, dN) <= N.REPRO_TOL}
        if not repro["ok"]:
            print(f"[{name}] WARNING base test metrics differ from the deployed JSON by {max(dR, dN):.2e}")
    B = Bg.cpu()
    del Bg
    torch.cuda.empty_cache()
    info = {"status": "complete",
            "source": "scope.closed_form_base (deployed base: 1-hop EASE + text-kNN, lam/a validation-tuned)",
            "selected": sel, "val_R20_printed": float(m.group(2)) if m else None, "val": val, "test": test,
            "repro_vs_deployed_json": repro, "dtype": str(B.dtype), "shape": list(B.shape),
            "build_s": time.time() - t0, "build_peak_gb": peak, "created_utc": N.utc_now(),
            "checksum_strided_sum": float(B[::97, ::13].double().sum()),
            "built_by": "neighbors_precompute_base.py (separate process; same code path as DSCtx._base)"}
    base_cache.mkdir(parents=True, exist_ok=True)
    np.save(cache, B.numpy())
    N.write_json(meta_p, info)
    print(f"[{name}] base {sel} val R@20={val['Recall@20']:.4f} test R@20={test['Recall@20']:.4f} "
          f"N@20={test['NDCG@20']:.4f} dtype={B.dtype} peak={peak:.1f}GB repro={repro and repro['ok']} -> {cache}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--base_tag", default="table1")
    ap.add_argument("--out", default=str(N.ROOT / "results" / "scope" / "rev" / "g7g13_neighbors"))
    ap.add_argument("--config_model", default="scope")
    ap.add_argument("--scope_json_dir", default=str(N.ROOT / "results" / "scope"))
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    base_cache = Path(args.out) / "base_cache"
    for ds in args.datasets:
        build(ds, args, base_cache)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

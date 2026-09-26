"""Memory-lean SCOPE-U / ensemble control, used for MicroLens (98K users) with the validation-selected GUME
(lr 5e-4, n_layers 1), seeds 2024/2025/2026 of the head, with per-user vectors, paired bootstrap (B=1e4, two-sided)
and Holm. Same protocol as ensemble_control.py:

  views (z-scored, fp16): item = closed-form EASE+text base (lam, a val-tuned), set = SCOPE head (deployed checkpoint
  of the seed), col = FREEDOM scores, gume = GUME scores. Each combo of ensemble_control.COMBOS gets its non-negative
  gate grid-searched on validation Recall@20 by ensemble_control.gate_select (imported, not copied; grid GR, GR4 for
  4-view combos), then ONE trusted test evaluation with scope.evalS_trusted (train items masked).

What differs from ensemble_control.py, and why:
  1. Views are passed to gate_select / GPUEval / evalS_trusted as lazy row providers (`w * view`, `sum(...)` build a
     lazy combination that is evaluated only for the users being scored), so no full-size [U, I] temporaries are
     allocated; on a 24 GB GPU the materialized multi-view MicroLens rows can run out of memory. OOM/errors are
     recorded as rows, never silently dropped.
  2. Above 50000 users the EASE(+text) base and the EASE-only baseline are computed block-wise over the SAME
     16384-user row blocks scope.spmm_lowmem / scope.zr use, i.e. bit-identical to scope.closed_form_base's frugal
     path and to microlens_linear.py, without the ~18 GB of full-size temporaries closed_form_base allocates on
     MicroLens. At <= 50000 users scope.closed_form_base itself is called.
  3. Above 50000 users the set view is computed for the requested user rows only (the > 50000-user fp16 branch of
     SCOPE.score_all + scope.zr's per-row formula); equal to the materialized view up to fp16 GEMM rounding.
  4. Every gate grid point's validation score is recorded (also for the base's (lam, a) grid and the EASE lam grid).
  5. Per-user test Recall/NDCG@{10,20} (GPUEval, same masking and user order) are saved to .npz and paired-
     bootstrapped; Holm is applied in code over the stated per-seed family.
  6. Rows without the set view do not depend on the seed and are computed once per dataset ("shared_rows").
  7. Single baselines (the GUME dump itself, EASE, optional extra dumps) are scored from their raw fp32 scores
     (streamed from the .npy memmap) for the bootstrap, so they equal the baselines' own numbers.
Check: with the initial GUME run's scores this script reproduces ensemble_control.py's MicroLens rows within 3e-7.

Resident GPU memory on MicroLens: item, col, gume fp16 views = 3 x 3.38 GB; target peak <= 12 GB (recorded).
Outputs (never overwritten; timestamped):
  results/scope/ensemble_control_lean_<ds>_<tag>[_smoke]_<stamp>.json            everything, written incrementally
  results/scope/ensemble_control_lean_<ds>_<tag>[_smoke]_<stamp>_shared_peruser.npz
  results/scope/ensemble_control_lean_<ds>_<tag>[_smoke]_<stamp>_seed<seed>_peruser.npz
Select the GPU with CUDA_VISIBLE_DEVICES (scope.py fixes DEV = cuda:0).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import socket
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SCOPE_ROOT") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import scope as SC  # noqa: E402  (scope/scope.py)
from scope import (DEV, SCOPE, Rmat, build_lists, closed_form_base, ease_B, evalS_trusted, gram,  # noqa: E402
                   mm_affinity, spmm, zr)
import ensemble_control as EC  # noqa: E402  (COMBOS, GR, GR4, gate_select reused unchanged)
from gpu_eval import GPUEval  # noqa: E402
from src.data.dataset import RecDataset  # noqa: E402
from src.utils import Config  # noqa: E402

if Path(SC.ROOT).resolve() != ROOT:
    sys.exit(f"scope.py lives under {SC.ROOT} but SCOPE_ROOT={ROOT}; point both at the same tree")

BLK = 16384          # row block of scope.zr / scope.spmm_lowmem / scope.gram above 50000 users
BIG_U = 50000        # scope's switch to its frugal block-wise paths
FP16 = torch.float16
BASE_LAMS, BASE_AS = [400, 800, 1500], [0.0, 0.3, 0.5, 0.7]      # == scope.closed_form_base grid and order
EASE_LAMS = [100, 400, 800, 1500, 3000]                           # == ease_baseline.py / microlens_linear.py
# paper name -> COMBOS row (main table, MicroLens): SCOPE-U = GUMEswap row, SCOPE-v2 = the base+set+FREEDOM row,
# SCOPE-v1 = base+set row, base = base(EASE+text) row
ALIASES = {"SCOPE-U": "GUMEswap(base+set+GUME)", "SCOPE-v2": "SCOPE-U(base+set+FREEDOM)",
           "SCOPE-v1": "base+set(SCOPE-v1/G)", "base": "base(EASE+text)",
           "base+GUME(no set)": "base+GUME(no set)", "GUME(raw)": "GUME(raw)", "EASE": "EASE"}
METRICS_PU = ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10")
_PEAK = [0]  # CUDA peak (bytes) seen across the per-row counter resets since the enclosing stage started


# ------------------------------------------------------------------------------------------------ small utilities
def reset_peak() -> None:
    """Reset the CUDA peak counter for one row, remembering the previous peak for the enclosing stage."""
    _PEAK[0] = max(_PEAK[0], torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def gb(x: float) -> float:
    return round(x / 2 ** 30, 3)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def to_dev_long(bu) -> torch.Tensor:
    if not torch.is_tensor(bu):
        bu = torch.as_tensor(np.asarray(bu))
    return bu.to(DEV, dtype=torch.long)


def fileinfo(p: Path) -> dict:
    d = {"path": str(p), "exists": p.is_file()}
    if p.is_file():
        st = p.stat()
        d.update(bytes=st.st_size, mtime=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))
        side = p.with_suffix(".json")
        if side.is_file():
            try:
                sj = json.loads(side.read_text())
                d["dump_sidecar"] = {k: sj.get(k) for k in ("status", "ckpt", "seed", "verify", "model", "dataset")}
            except Exception as e:  # informational
                d["dump_sidecar_error"] = str(e)
    return d


class RecordingEval:
    """Pass-through GPUEval proxy that records every .eval() result (gate_select / closed_form_base only call .eval)."""

    def __init__(self, gev):
        self.gev, self.calls = gev, []

    def eval(self, S, batch=4096):
        r = self.gev.eval(S, batch)
        self.calls.append({k: float(v) for k, v in r.items()})
        return r


# ------------------------------------------------------------------------------------------------ lazy score views
class View:
    """A [U, I] score view that hands out rows on demand. `w * view` and `sum(...)` build a lazy Comb, so
    ensemble_control.gate_select runs unchanged; GPUEval.eval / evalS_trusted only ever index S[user_ids]."""
    dtype = FP16

    def rows(self, bu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, bu):
        return self.rows(to_dev_long(bu))

    def __rmul__(self, w):
        return Comb([(float(w), self)])

    __mul__ = __rmul__


class Comb:
    """sum_k w_k * V_k, evaluated row-wise with the same operations and order as `sum(w * V for ...)` on full
    matrices (0 + w1*V1 + w2*V2 ..., fp16), hence identical values per row."""

    def __init__(self, terms):
        self.terms = list(terms)

    def __add__(self, o):
        if isinstance(o, Comb):
            return Comb(self.terms + o.terms)
        if isinstance(o, (int, float)) and o == 0:
            return self
        return NotImplemented

    __radd__ = __add__

    def __getitem__(self, bu):
        bu = to_dev_long(bu)
        acc = 0
        for w, V in self.terms:
            acc = acc + w * V.rows(bu)
        return acc


class TensorView(View):
    def __init__(self, T: torch.Tensor):
        self.T, self.dtype = T, T.dtype

    def rows(self, bu):
        return self.T[bu]


class MemmapView(View):
    """Raw fp32 scores streamed from an .npy memmap (single-baseline evaluation; never materialized)."""
    dtype = torch.float32

    def __init__(self, path: Path, U: int, I: int):
        self.mm = np.load(path, mmap_mode="r")
        if tuple(self.mm.shape) != (U, I):
            raise ValueError(f"{path}: shape {self.mm.shape} != ({U}, {I})")

    def rows(self, bu):
        return torch.from_numpy(np.ascontiguousarray(self.mm[bu.cpu().numpy()])).to(DEV)


class BlockView(View):
    """View whose rows come from make_block(s, e) over the fixed 16384-row blocks; the last `cache` blocks are kept
    (GPUEval / evalS_trusted walk users in increasing order, so each block is built about once per pass)."""

    def __init__(self, U: int, I: int, make_block, cache: int = 2):
        self.U, self.I, self.make_block, self.ncache = U, I, make_block, cache
        self.cache: OrderedDict = OrderedDict()

    def _block(self, b: int) -> torch.Tensor:
        if b in self.cache:
            self.cache.move_to_end(b)
            return self.cache[b]
        s = b * BLK
        T = self.make_block(s, min(s + BLK, self.U))
        self.cache[b] = T
        while len(self.cache) > self.ncache:
            self.cache.popitem(last=False)
        return T

    def rows(self, bu):
        bid = torch.div(bu, BLK, rounding_mode="floor")
        lo, hi = int(bid.min()), int(bid.max())
        if lo == hi:
            return self._block(lo)[bu - lo * BLK]
        out = None
        for b in torch.unique(bid).tolist():
            m = bid == b
            blk = self._block(b)
            if out is None:
                out = torch.empty((bu.numel(), self.I), dtype=blk.dtype, device=DEV)
            out[m] = blk[bu[m] - b * BLK]
        return out

    def materialize(self) -> torch.Tensor:
        out = None
        for b in range((self.U + BLK - 1) // BLK):
            s = b * BLK
            blk = self.make_block(s, min(s + BLK, self.U))
            if out is None:
                out = torch.empty((self.U, self.I), dtype=blk.dtype, device=DEV)
            out[s:s + blk.shape[0]] = blk
            del blk
        self.cache.clear()
        return out


class SetView(View):
    """zr(SCOPE.score_all(R, deg)).to(fp16) for the requested rows only (> 50000 users): the fp16 branch of
    SCOPE.score_all (normalize -> half -> GEMM / tau) followed by scope.zr's per-row formula."""

    def __init__(self, m: SCOPE, R, degf):
        with torch.no_grad():
            z = m.latent(torch.sparse.mm(R, m.E), degf)
            self.zt = F.normalize(z, dim=1).half()
            self.Et = F.normalize(m.E, dim=1).half()
            self.tau = m.logtau.exp().clamp(min=1e-3).half()
        del z

    @torch.no_grad()
    def rows(self, bu):
        return zr((self.zt[bu] @ self.Et.t()) / self.tau)


def R_block(R, s: int, e: int):
    """Rows s:e of the coalesced sparse R, sliced exactly as scope.spmm_lowmem slices them."""
    idx, val = R.indices(), R.values()
    m = (idx[0] >= s) & (idx[0] < e)
    return torch.sparse_coo_tensor(torch.stack([idx[0][m] - s, idx[1][m]]), val[m], (e - s, R.shape[1])).coalesce()


def zr_block(x: torch.Tensor) -> torch.Tensor:
    """scope.zr's per-row-block formula (fp32 statistics, result in the input dtype)."""
    b = x.float()
    return ((b - b.mean(1, keepdim=True)) / (b.std(1, keepdim=True) + 1e-9)).to(x.dtype)


def linear_blocks(R, B, Aff=None, aa=0.0, outer_zr=False, dt=FP16):
    """Block maker for the frugal closed forms:
         zr(spmm_lowmem(R, B, dt)) [+ aa * zr(spmm_lowmem(R, Aff, dt))]   (scope.closed_form_base, frugal path)
       and, with outer_zr, the extra zr(.) ensemble_control applies to the base view."""
    @torch.no_grad()
    def make(s, e):
        rc = R_block(R, s, e)
        S = zr_block(torch.sparse.mm(rc, B).to(dt))
        if Aff is not None and aa:
            S = S + aa * zr_block(torch.sparse.mm(rc, Aff).to(dt))
        del rc
        return zr_block(S) if outer_zr else S
    return make


def load_cf_view(path: Path, U: int, I: int) -> torch.Tensor:
    """== zr(torch.from_numpy(np.load(path)).to(fp16).to(DEV)) of ensemble_control.py. Above 50000 users it streams the
    memmap in the 16384-row blocks scope.zr uses (bit-identical), so the fp32 file never sits whole in RAM or GPU."""
    mm = np.load(path, mmap_mode="r")
    if tuple(mm.shape) != (U, I):
        raise ValueError(f"{path}: shape {mm.shape} != ({U}, {I})")
    if U <= BIG_U:
        return zr(torch.from_numpy(np.array(mm)).to(FP16).to(DEV))
    out = torch.empty((U, I), dtype=FP16, device=DEV)
    for s in range(0, U, BLK):
        e = min(s + BLK, U)
        out[s:e] = zr_block(torch.from_numpy(np.array(mm[s:e])).to(FP16).to(DEV))
    return out


# ------------------------------------------------------------------------------------------------ evaluation helpers
@torch.no_grad()
def per_user_metrics(gev: GPUEval, S, ks=(10, 20), batch=4096) -> dict:
    """Per-user Recall@k / NDCG@k in gev's user order, with the formulas, train masking and batching of
    GPUEval._run (GPUEval only exposes recall_per_user; NDCG per user is needed for the bootstrap)."""
    U = gev.users.numel()
    out = {f"{m}@{k}": torch.zeros(U, device=gev.dev) for k in ks for m in ("Recall", "NDCG")}
    for s in range(0, U, batch):
        bu = gev.users[s:s + batch]
        sc = gev._mask(S[bu].clone().float(), bu)
        _, idx = torch.topk(sc, gev.maxk, dim=1)
        bp = gev.pos[s:s + batch]
        hit = (idx.unsqueeze(2) == bp.unsqueeze(1)).any(2).float()
        nrel = gev.nfit[s:s + batch].clamp(min=1)
        for k in ks:
            hk = hit[:, :k]
            out[f"Recall@{k}"][s:s + bu.numel()] = hk.sum(1) / nrel
            dcg = (hk * (1.0 / torch.log2(torch.arange(2, k + 2, device=gev.dev).float())).unsqueeze(0)).sum(1)
            ideal = torch.minimum(nrel, torch.full_like(nrel, k)).long()
            out[f"NDCG@{k}"][s:s + bu.numel()] = dcg / gev.cumdisc[ideal].clamp(min=1e-9)
    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


def score_row(S, gevT, dset, smoke: bool) -> tuple:
    """Trusted test metrics (scope.evalS_trusted) + per-user test vectors + consistency checks -> (row_part, pu)."""
    t = evalS_trusted(S, dset, "test")
    pu = per_user_metrics(gevT, S)
    g = gevT.eval(S)
    chk = {"max_absdiff_peruser_mean_vs_GPUEval": max(abs(float(pu[m].mean()) - g[m]) for m in METRICS_PU)}
    if not smoke:  # in smoke mode gevT is a user subset, evalS_trusted is not
        chk["absdiff_GPUEval_vs_trusted_R20"] = abs(g["Recall@20"] - t["Recall@20"])
        chk["absdiff_GPUEval_vs_trusted_N20"] = abs(g["NDCG@20"] - t["NDCG@20"])
    chk["ok"] = chk["max_absdiff_peruser_mean_vs_GPUEval"] < 1e-5 and chk.get("absdiff_GPUEval_vs_trusted_R20", 0) < 5e-4
    return {"test": {k: float(v) for k, v in t.items()}, "test_GPUEval": {k: float(v) for k, v in g.items()},
            "per_user_check": chk}, pu


def paired_bootstrap_family(pairs, B: int, seed: int, chunk: int = 200) -> list:
    """Paired user-level bootstrap of mean(a - b) for every (a, b) in `pairs` (same users, same order), the estimator
    of harness.paired_bootstrap: percentile 95% CI, p = 2 min(P*(d<=0), P*(d>=0)). One
    resampling-index stream (numpy default_rng(seed)) is shared by all tests, as harness does by re-seeding rng=0 on
    every call; it is drawn in chunks so B x n indices never sit in RAM (n = 98K users on MicroLens)."""
    n = len(pairs[0][0])
    D = [np.asarray(a, np.float64) - np.asarray(b, np.float64) for a, b in pairs]
    boots = np.empty((len(D), B))
    rng = np.random.default_rng(seed)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        idx = rng.integers(0, n, size=(e - s, n))
        for t, d in enumerate(D):
            boots[t, s:e] = d[idx].mean(1)
    res = []
    for (a, b), d, bt in zip(pairs, D, boots):
        ma, mb = float(np.mean(a)), float(np.mean(b))
        p = 2.0 * min(float((bt <= 0).mean()), float((bt >= 0).mean()))
        lo, hi = float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))
        res.append(dict(mean_a=ma, mean_b=mb, mean_delta=float(d.mean()), ci95=[lo, hi],
                        rel_delta_pct=(100.0 * float(d.mean()) / mb) if mb else None,
                        rel_ci95_pct=[100.0 * lo / mb, 100.0 * hi / mb] if mb else None,
                        p_two_sided=float(min(p, 1.0)), p_resolution=1.0 / B, n_users=int(n), B=B, seed=seed,
                        n_win=int((d > 0).sum()), n_tie=int((d == 0).sum()), n_loss=int((d < 0).sum())))
    return res


def holm(pvals) -> list:
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


# ------------------------------------------------------------------------------------------------ main experiment
class Run:
    def __init__(self, a, ds: str):
        self.a, self.ds = a, ds
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"ensemble_control_lean_{ds}_{a.tag}" + ("_smoke" if a.smoke else "") + f"_{stamp}"
        self.out_dir = Path(a.out).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.out_dir / f"{base}.json"
        self.npz_prefix = self.out_dir / base
        if self.json_path.exists():
            raise SystemExit(f"refusing to overwrite {self.json_path}")
        self.rep: dict = {"script": "scope/ensemble_control_lean.py", "argv": sys.argv,
                          "host": socket.gethostname(), "started": datetime.now().isoformat(timespec="seconds"),
                          "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                          "dataset": ds, "tag": a.tag, "smoke": a.smoke, "seeds": a.seeds, "status": "running",
                          "aliases": ALIASES, "stages": {}, "shared_rows": {}, "seed_rows": {}, "bootstrap": {},
                          "per_user_npz": {}, "npz_key_map": {}}

    def save(self) -> None:
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.rep, indent=2, default=str))
        os.replace(tmp, self.json_path)

    def path(self, template: str, **kw) -> Path:
        p = Path(template.format(ds=self.ds, **kw))
        return p if p.is_absolute() else (ROOT / p)


def stage_timer(run: Run, name: str):
    class _T:
        def __enter__(self):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _PEAK[0] = 0
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            peak = max(_PEAK[0], torch.cuda.max_memory_allocated())   # includes rows that reset the counter
            run.rep["stages"].setdefault(name, {}).update(
                wall_s=round(time.time() - self.t0, 1), peak_gpu_GB=gb(peak), failed=exc[0] is not None)
            run.save()
            return False
    return _T()


def eval_combo(name, keys, views, gev, gevT, dset, smoke):
    """gate_select (ensemble_control, unchanged) on validation, then one trusted test pass. Never raises."""
    rec = RecordingEval(gev)
    reset_peak()
    t0 = time.time()
    row: dict = {"keys": list(keys)}
    pu = None
    try:
        ws, S = EC.gate_select(views, keys, rec)
        grid = EC.GR if len(keys) < 4 else EC.GR4
        pts = [w for w in itertools.product(grid, repeat=len(keys)) if not all(x == 0 for x in w)]
        if len(pts) != len(rec.calls):
            raise RuntimeError(f"grid bookkeeping: {len(pts)} points vs {len(rec.calls)} evals")
        row["gate"] = [float(w) for w in ws]
        row["gate_grid"] = grid
        row["val_selected"] = rec.calls[pts.index(tuple(ws))]
        row["val_grid"] = [{"w": list(w), "Recall@20": c["Recall@20"], "NDCG@20": c["NDCG@20"]}
                           for w, c in zip(pts, rec.calls)]
        row["gate_on_grid_edge"] = any(w == max(grid) for w in ws)
        res, pu = score_row(S, gevT, dset, smoke)
        row.update(res)
        row["status"] = "ok"
        del S
    except torch.cuda.OutOfMemoryError as e:
        row.update(status="OOM", error=str(e)[:800], val_grid_partial=rec.calls)
    except Exception as e:  # recorded, never skipped
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:800], traceback=traceback.format_exc()[-3000:],
                   val_grid_partial=rec.calls)
    torch.cuda.empty_cache()
    row["wall_s"] = round(time.time() - t0, 1)
    row["peak_gpu_GB"] = gb(torch.cuda.max_memory_allocated())
    return row, pu


def eval_single(view, gev, gevT, dset, smoke):
    """A single baseline scored as-is (no gate): validation + trusted test + per-user vectors."""
    reset_peak()
    t0 = time.time()
    row: dict = {}
    pu = None
    try:
        row["val"] = {k: float(v) for k, v in gev.eval(view).items()}
        res, pu = score_row(view, gevT, dset, smoke)
        row.update(res)
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        row.update(status="OOM", error=str(e)[:800])
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:800], traceback=traceback.format_exc()[-3000:])
    torch.cuda.empty_cache()
    row["wall_s"] = round(time.time() - t0, 1)
    row["peak_gpu_GB"] = gb(torch.cuda.max_memory_allocated())
    return row, pu


def line(ds, sd, name, row) -> None:
    if row.get("status") != "ok":
        print(f"[{ds} {sd}] {name:30s} {row.get('status')}: {row.get('error', '')[:160]}", flush=True)
        return
    g = row.get("gate")
    v = (row.get("val_selected") or row.get("val") or {}).get("Recall@20", float("nan"))
    t = row["test"]
    print(f"[{ds} {sd}] {name:30s} gate={str([round(x, 1) for x in g]) if g else '-':<22} valR20={v:.4f} "
          f"test R@20={t['Recall@20']:.4f} N@20={t['NDCG@20']:.4f} R@10={t['Recall@10']:.4f} N@10={t['NDCG@10']:.4f} "
          f"({row['wall_s']}s, peak {row['peak_gpu_GB']}GB)", flush=True)


def save_npz(path: Path, users: np.ndarray, pu_rows: dict, keymap: dict) -> None:
    arrs = {"users": users.astype(np.int64)}
    for name, pu in pu_rows.items():
        for m, v in pu.items():
            k = f"{safe(name)}__{safe(m)}"
            keymap[k] = [name, m]
            arrs[k] = v
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def run_dataset(a, ds: str) -> None:
    run = Run(a, ds)
    try:
        _run_dataset(a, ds, run)
    except BaseException as e:  # keep the partial JSON and say why it stopped
        run.rep["status"] = "failed"
        run.rep["error"] = f"{type(e).__name__}: {e}"[:800]
        run.rep["traceback"] = traceback.format_exc()[-4000:]
        run.rep["finished"] = datetime.now().isoformat(timespec="seconds")
        run.save()
        raise


def _run_dataset(a, ds: str, run: "Run") -> None:
    rep = run.rep
    seed_all(a.seeds[0])  # before RecDataset / build_lists (the shared stage uses the first seed)
    rep["seed_policy"] = ("random/numpy/torch/cuda seeded with seeds[0] before RecDataset and build_lists; re-seeded "
                          "with each seed before its set-view stage. No stage draws random numbers except build_lists "
                          "(only its degree vector is used) and the bootstrap (own numpy Generator, --bootstrap-seed).")
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset)
    _, _, deg = build_lists(dset)
    degf = deg.float()
    U, I = dset.n_users, dset.n_items
    big = U > BIG_U
    half = dset.n_items > 20000 or dset.n_users > 50000  # == ensemble_control.py
    base_impl = a.base_impl if a.base_impl != "auto" else ("lean" if big else "scope")
    if base_impl == "lean" and not big:
        raise SystemExit("--base-impl lean is bit-identical to closed_form_base only above 50000 users; use auto/scope")
    rep["data"] = {"n_users": U, "n_items": I, "n_train_pairs": int(dset.train_matrix.nnz), "big": big,
                   "base_impl": base_impl}
    gev = GPUEval(dset, "valid", DEV)
    gevT = GPUEval(dset, "test", DEV)
    if a.smoke:
        for g in (gev, gevT):
            g.users, g.pos, g.nfit = g.users[:a.smoke_users], g.pos[:a.smoke_users], g.nfit[:a.smoke_users]
        EC.GR, EC.GR4 = [0.0, 1.0], [0.0, 1.0]
    users_test = gevT.users.cpu().numpy()
    rep["protocol"] = {
        "gate_grid": EC.GR, "gate_grid_4view": EC.GR4,
        "gate_selection": "ensemble_control.gate_select: max validation Recall@20 (GPUEval, train-masked), first "
                          "maximum in itertools.product order (strict >)",
        "test": "scope.evalS_trusted (TopKEvaluator, train-masked), evaluated once per row for the selected gate",
        "per_user": "GPUEval-formula per-user Recall/NDCG@{10,20} on the test split (same masking, user order in npz)",
        "views": "z-scored fp16 (item, set, col=FREEDOM, gume=GUME), as ensemble_control.py",
        "bootstrap": f"paired, B={a.bootstrap_B}, two-sided, percentile CI, numpy default_rng({a.bootstrap_seed})",
        "holm": "step-down Holm over the per-seed family (all comparisons listed under bootstrap.<seed>.family, "
                "R@20 and N@20 each), alpha 0.05",
        "smoke": a.smoke}
    inputs = {"gume_scores": fileinfo(run.path(a.gume_scores)), "freedom_scores": fileinfo(run.path(a.freedom_scores)),
              "set_ckpts": {}, "baselines": {}}
    extra = []
    for spec in a.baseline:
        nm, p = spec.split("=", 1)
        extra.append((nm, run.path(p)))
        inputs["baselines"][nm] = fileinfo(run.path(p))
    for sd in a.seeds:
        inputs["set_ckpts"][str(sd)] = fileinfo(run.path(a.set_ckpt, sfx="" if sd == 2024 else f"_s{sd}"))
    rep["inputs"] = inputs
    run.save()
    shared_pu: dict = {}

    # ---- stage 1: EASE-only baseline (main-table protocol: val-selected lam) -----------------------------------------
    if a.with_ease:
        with stage_timer(run, "ease"):
            lams = [400] if a.smoke else EASE_LAMS
            G = gram(R)
            grid, best = [], None
            if big:  # == microlens_linear.py: zr(spmm_lowmem(R, ease_B(G, lam), fp16)), block-wise
                for lm in lams:
                    B = ease_B(G, lm)
                    r = gev.eval(BlockView(U, I, linear_blocks(R, B)))
                    grid.append({"lam": lm, "val": r})
                    if best is None or r["Recall@20"] > best[0]:
                        best = (r["Recall@20"], lm)
                    del B
                    torch.cuda.empty_cache()
                B = ease_B(G, best[1])
                del G
                V = BlockView(U, I, linear_blocks(R, B))
            else:    # == ease_baseline.py: zr(spmm(R, ease_B(G, lam)).to(dt)), dt fp16 if > 20000 items
                dt = FP16 if dset.n_items > 20000 else torch.float32
                for lm in lams:
                    S = zr(spmm(R, ease_B(G, lm)).to(dt))
                    r = gev.eval(S)
                    grid.append({"lam": lm, "val": r})
                    if best is None or r["Recall@20"] > best[0]:
                        best = (r["Recall@20"], lm)
                    del S
                    torch.cuda.empty_cache()
                V = TensorView(zr(spmm(R, ease_B(G, best[1])).to(dt)))
                del G
            row, pu = eval_single(V, gev, gevT, dset, a.smoke)
            row.update(grid=grid, lam=best[1], lam_on_grid_edge=best[1] in (min(lams), max(lams)))
            rep["shared_rows"]["EASE"] = row
            if pu is not None:
                shared_pu["EASE"] = pu
            line(ds, "shared", "EASE", row)
            del V
            if big:
                del B
            torch.cuda.empty_cache()

    # ---- stage 2: the EASE+text base view (item) ------------------------------------------------------------------
    with stage_timer(run, "base"):
        if base_impl == "scope":
            recb = RecordingEval(gev)
            S = closed_form_base(R, dset, recb, half=half)
            V_item = zr(S).to(FP16)
            del S
            has_txt = dset.t_feat is not None
            pts = [(lm, aa) for lm in BASE_LAMS for aa in (BASE_AS if has_txt else [0.0])]
            if len(pts) == len(recb.calls):
                grid = [{"lam": lm, "a": aa, "val": c} for (lm, aa), c in zip(pts, recb.calls)]
                bi = max(range(len(pts)), key=lambda i: (recb.calls[i]["Recall@20"], -i))
                sel = {"lam": pts[bi][0], "a": pts[bi][1]}
            else:
                grid, sel = [{"val": c} for c in recb.calls], None
        else:
            G = gram(R)
            Aff = mm_affinity(dset.t_feat[:]) if dset.t_feat is not None else None
            lams = [800] if a.smoke else BASE_LAMS
            aas = ([0.0, 0.5] if a.smoke else BASE_AS) if Aff is not None else [0.0]
            grid, best = [], None
            for lm in lams:                       # same loop order and strict '>' as closed_form_base
                B = ease_B(G, lm)
                for aa in aas:
                    r = gev.eval(BlockView(U, I, linear_blocks(R, B, Aff, aa)))
                    grid.append({"lam": lm, "a": aa, "val": r})
                    if best is None or r["Recall@20"] > best[0]:
                        best = (r["Recall@20"], dict(lam=lm, a=aa))
                del B
                torch.cuda.empty_cache()
            sel = best[1]
            B = ease_B(G, sel["lam"])
            del G
            V_item = BlockView(U, I, linear_blocks(R, B, Aff, sel["a"], outer_zr=True)).materialize()
            del B, Aff
            torch.cuda.empty_cache()
        rep["stages"]["base"] = {"impl": base_impl, "grid": grid, "selected": sel}
        print(f"[{ds}] base ({base_impl}) selected {sel}", flush=True)
    views = {"item": TensorView(V_item)}

    # ---- stage 3: single baselines from raw dumps (GUME itself + extras) ------------------------------------------
    singles = [("GUME(raw)", run.path(a.gume_scores))] + extra
    for nm, p in singles:
        with stage_timer(run, f"single:{nm}"):
            if not p.is_file():
                row, pu = {"status": "missing_input", "path": str(p)}, None
            else:
                row, pu = eval_single(MemmapView(p, U, I), gev, gevT, dset, a.smoke)
            row["path"] = str(p)
            rep["shared_rows"][nm] = row
            if pu is not None:
                shared_pu[nm] = pu
            line(ds, "shared", nm, row)

    # ---- stage 4: CF views + seed-independent combos (no 'set') ---------------------------------------------------
    with stage_timer(run, "load_cf_views"):
        views["col"] = TensorView(load_cf_view(run.path(a.freedom_scores), U, I))
        views["gume"] = TensorView(load_cf_view(run.path(a.gume_scores), U, I))
    for name, keys in EC.COMBOS.items():
        if "set" in keys:
            continue
        row, pu = eval_combo(name, keys, views, gev, gevT, dset, a.smoke)
        rep["shared_rows"][name] = row
        if pu is not None:
            shared_pu[name] = pu
        line(ds, "shared", name, row)
        run.save()
    npz = Path(f"{run.npz_prefix}_shared_peruser.npz")
    save_npz(npz, users_test, shared_pu, rep["npz_key_map"])
    rep["per_user_npz"]["shared"] = str(npz)
    run.save()

    # ---- stage 5: per seed: set view, set combos, bootstrap + Holm ------------------------------------------------
    for sd in a.seeds:
        seed_all(sd)
        rows = rep["seed_rows"].setdefault(str(sd), {})
        seed_pu: dict = {}
        ck = run.path(a.set_ckpt, sfx="" if sd == 2024 else f"_s{sd}")   # == ensemble_control's stag rule
        with stage_timer(run, f"seed{sd}"):
            if not ck.is_file():
                for name, keys in EC.COMBOS.items():
                    if "set" in keys:
                        rows[name] = {"keys": keys, "status": "missing_input", "path": str(ck)}
                run.save()
                continue
            state = torch.load(ck, map_location=DEV)
            m = SCOPE(I, int(state["E"].shape[1])).to(DEV)
            m.load_state_dict(state)
            m.eval()
            del state
            if big:
                views["set"] = SetView(m, R, degf)
            else:
                with torch.no_grad():
                    views["set"] = TensorView(zr(m.score_all(R, degf)).to(FP16))
            for name, keys in EC.COMBOS.items():
                if "set" not in keys:
                    continue
                row, pu = eval_combo(name, keys, views, gev, gevT, dset, a.smoke)
                row["set_ckpt"] = str(ck)
                rows[name] = row
                if pu is not None:
                    seed_pu[name] = pu
                line(ds, sd, name, row)
                run.save()
            del views["set"], m
            torch.cuda.empty_cache()
        npz = Path(f"{run.npz_prefix}_seed{sd}_peruser.npz")
        save_npz(npz, users_test, seed_pu, rep["npz_key_map"])
        rep["per_user_npz"][str(sd)] = str(npz)

        # family (fixed before running): margins stated for MicroLens (SCOPE-U - GUME,
        # SCOPE-v2 - GUME, set view - base; the restored no-set row; Sec. 5.2 margins over GUME and EASE)
        pool = {**shared_pu, **seed_pu}
        fam = [("SCOPE-U", "GUME(raw)"), ("SCOPE-U", "base+GUME(no set)"), ("SCOPE-v2", "GUME(raw)"),
               ("SCOPE-v1", "base"), ("SCOPE-v1", "GUME(raw)"), ("base", "GUME(raw)")]
        if a.with_ease:
            fam += [("SCOPE-U", "EASE"), ("SCOPE-v1", "EASE"), ("base", "EASE")]
        for nm, _ in extra:
            fam += [("SCOPE-U", nm), ("SCOPE-v1", nm), ("base", nm)]
        tests, missing = [], []
        for x, y in fam:
            kx, ky = ALIASES.get(x, x), ALIASES.get(y, y)
            if kx not in pool or ky not in pool:
                missing.append([x, y])
                continue
            for met in ("Recall@20", "NDCG@20"):
                tests.append((x, y, met, pool[kx][met], pool[ky][met]))
        with stage_timer(run, f"bootstrap_seed{sd}"):
            B = 200 if a.smoke else a.bootstrap_B
            out = []
            if tests:
                res = paired_bootstrap_family([(t[3], t[4]) for t in tests], B, a.bootstrap_seed)
                padj = holm([r["p_two_sided"] for r in res])
                for (x, y, met, _, _), r, pa in zip(tests, res, padj):
                    out.append({"a": x, "b": y, "a_row": ALIASES.get(x, x), "b_row": ALIASES.get(y, y), "metric": met,
                                **r, "p_holm": pa, "reject_holm_0.05": bool(pa < 0.05)})
            rep["bootstrap"][str(sd)] = {"family": [list(f) for f in fam], "family_missing_rows": missing,
                                         "m_tests": len(out), "tests": out}
        for t in out:
            print(f"[{ds} {sd}] {t['a']:>9s} - {t['b']:<18s} {t['metric']:9s} d={t['mean_delta']:+.4f} "
                  f"({t['rel_delta_pct'] if t['rel_delta_pct'] is None else round(t['rel_delta_pct'], 1)}%) "
                  f"CI=[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] p={t['p_two_sided']:.2g} p_holm={t['p_holm']:.2g} "
                  f"W/T/L={t['n_win']}/{t['n_tie']}/{t['n_loss']}", flush=True)
        run.save()

    # ---- summary over seeds ---------------------------------------------------------------------------------------
    summ = {}
    names = sorted({n for r in rep["seed_rows"].values() for n in r})
    for n in names:
        vals = [rep["seed_rows"][str(sd)].get(n, {}) for sd in a.seeds]
        ok = [v["test"] for v in vals if v.get("status") == "ok"]
        if ok:
            r20 = np.array([t["Recall@20"] for t in ok])
            n20 = np.array([t["NDCG@20"] for t in ok])
            summ[n] = {"n_seeds_ok": len(ok), "seeds": a.seeds,
                       "R@20_mean": float(r20.mean()), "R@20_std_ddof0": float(r20.std()),
                       "R@20_std_ddof1": float(r20.std(ddof=1)) if len(ok) > 1 else None,
                       "N@20_mean": float(n20.mean()), "N@20_std_ddof0": float(n20.std()),
                       "N@20_std_ddof1": float(n20.std(ddof=1)) if len(ok) > 1 else None}
    rep["seed_summary"] = summ
    peaks = [v.get("peak_gpu_GB") for v in rep["stages"].values() if isinstance(v, dict)]
    peaks += [r.get("peak_gpu_GB") for r in rep["shared_rows"].values()]
    peaks += [r.get("peak_gpu_GB") for rows in rep["seed_rows"].values() for r in rows.values()]
    rep["peak_gpu_GB_overall"] = max([p for p in peaks if p is not None], default=None)
    rep["status"] = "complete"
    rep["finished"] = datetime.now().isoformat(timespec="seconds")
    run.save()
    print(f"[{ds}] -> {run.json_path}", flush=True)
    del views, V_item, R, gev, gevT
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory-lean SCOPE-U / ensemble control (MicroLens, validation-selected GUME)")
    ap.add_argument("--datasets", nargs="+", default=["microlens"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--gume-scores", required=True,
                    help="raw GUME score dump (.npy, [U, I] fp32); '{ds}' is substituted; relative = repo root")
    ap.add_argument("--freedom-scores", default="results/baseline_scores/freedom_{ds}_scores.npy")
    ap.add_argument("--set-ckpt", default="ckpts/scope/scope_{ds}_d256_le1.0_lz1.0_lr0.003{sfx}.pt",
                    help="deployed SCOPE head; {sfx} = '' for seed 2024 else _s<seed> (ensemble_control's rule)")
    ap.add_argument("--baseline", action="append", default=[], metavar="NAME=PATH",
                    help="extra single baseline dump for the bootstrap family (e.g. lgmrec=results/baseline_scores/lgmrec_microlens_scores.npy); repeatable")
    ap.add_argument("--with-ease", action="store_true", help="add the EASE-only baseline (val-selected lambda)")
    ap.add_argument("--base-impl", choices=["auto", "scope", "lean"], default="auto",
                    help="auto: block-wise exact re-implementation above 50000 users, scope.closed_form_base below")
    ap.add_argument("--tag", required=True, help="output name tag, e.g. gume_lr0.0005_nl1 or repro_oldgume")
    ap.add_argument("--out", default=str(ROOT / "results" / "scope"))
    ap.add_argument("--bootstrap-B", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sanity path: gate grid {0,1}, 1-2 base/EASE grid points, first --smoke-users users "
                         "for validation/per-user, B=200")
    ap.add_argument("--smoke-users", type=int, default=3000)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (scope.py uses cuda:0)")
    t0 = time.time()
    failed = []
    for ds in a.datasets:
        try:
            run_dataset(a, ds)
        except Exception:
            traceback.print_exc()
            failed.append(ds)
            torch.cuda.empty_cache()
    print(f"ENSEMBLE_LEAN_DONE {time.time() - t0:.0f}s failed={failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

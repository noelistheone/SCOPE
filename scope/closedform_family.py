#!/usr/bin/env python
"""G3 + G4 -- content closed-form baselines on ONE shared validation grid, and the
frozen SCOPE set head fused onto the strongest of them.

G3  Models (all closed form; implemented
    from the equations of the papers -- no third-party code is copied; the L3AE repository has no licence):
      EASE                 B = I - P diagMat(1/diag P),  P = (G + lam I)^-1,  G = R^T R
      Base-retuned         our base (paper Eq. 1-2, scope.closed_form_base): z(R B_lam) + a z(R A^t), re-tuned on the
                           shared grid; Base-published = the published (lam, a) (reproduction check)
      CEASE-emb            EASE on G + w K,  K = F F^T, F = row-L2-normalised frozen text features (raw | mean-centred)
      CEASE-tags           EASE on G + alpha^2 T^T T, binary tags from X4 (Jeunen-faithful | uniform tokenisation)
                           tuned as in Jeunen et al.: lambda from EASE first, then alpha, then lambda refined
      Add-EASE-emb/-tags   R ((1-beta) B_R + beta B_T), B_R = EASE(G, lam_R), B_T = EASE(K or T^T T, lam_T) (raw blend)
      FEASE-tfidf/-emb     'FEASE (prior only)': FEASE-I-Prior (no user attributes), alpha = 0:
                           B = I + d P M - P diagMat((1 + d diag(PM)) / diag P),
                           P = (G + lam_tot I)^-1, d = rho*lam_tot*s, rho in [0,1] (lambda = lam_tot - d >= 0),
                           M = TF-IDF cosine (Amazon, MicroLens) or K, scaled to the off-diagonal Frobenius norm of B_EASE(lam*)
                           (s = s0 * smult; the scale is OUR choice -- FEASE does not specify one)
      FEASE-full-tfidf     full FEASE (Amazon, MicroLens): the same estimator on the stacked matrix [R; alpha T] (T = tags_jeunen),
                           i.e. P = (G + alpha^2 T^T T + lam_tot I)^-1, plus the TF-IDF prior M; alpha in {.1,.2,.4}
                           (edge-extended) tuned jointly with lam_tot, rho, smult on validation
      L3AE                 same estimator with M = S_hat = EASE(K, lam_F) (phase 1), d = lam_KD = rho*lam_tot,
                           lam_X = (1 - rho) lam_tot (negative for rho > 1, as in the released L3AE configs);
                           instantiated with OUR frozen text features (not NV-Embed-v2)
      Add-EASE-z  (swap)   z(R B_R) + a z(R B_T): our late z-fusion, ridge content kernel instead of the top-20 kNN
      FEASE-kNN   (swap)   FEASE-I-Prior with M = our text kNN kernel A^t: our kernel folded INTO the solve
    Shared ridge grid LAM = {50,100,200,400,800,1500,3000} for every closed form (incl. our base; a in
    {0,.1,.2,.3,.5,.7,1,1.5}). Selection on validation Recall@20; one trusted test evaluation (evalS_trusted) of the
    selected configuration per model; grid-edge hits are logged and the grid is extended one ladder step at a time
    (at most --max-extend steps beyond the shared grid per parameter and direction; a fixed starting value taken from
    another model, e.g. lam_R = EASE's lam*, is added to the swept grid first so the edge rule sees it). fp32 solves,
    TF32 disabled; fp64 re-solve check on Baby (selected + worst-conditioned corner per model) and, on every dataset,
    of any selected single-matrix configuration that lies beyond the shared grid (fused rows beyond it are flagged).
    Elec: '--' (dense |I|^2).
G4  S = z(S_set) + gamma z(S_kernel) (paper Eq. 6) with the existing seed-2024/2025/2026 SCOPE head checkpoints
    (explicit paths, no retraining); S_kernel = the validation-best LITERATURE content closed form of G3 (pool
    'literature'; pool 'all' adds our re-tuned base and the swaps and is run only if its winner differs; Base-retuned
    is fused WITHOUT a second z-score, exactly like SCOPE-v1 / anchor_v1); gamma on the FIXED Eq. 6 grid
    {0,.3,.6,1,1.5,2,3,5} on validation (never extended, an edge hit is only logged -- as scope.train); paired
    user-level bootstrap (R@20, N@20; B=10000, two-sided, p = (count+1)/(B+1)) fused vs S_kernel alone; Holm computed
    in code. Anchor: the deployed SCOPE-v1 (published base, Eq. 6 grid) is re-derived per seed and compared with
    results/scope/scope_<ds>_*.json.
Verdicts (G3-1, G3-2, G4) are printed only when every expected dataset/model/seed is present; otherwise the summary says
INCOMPLETE/PROVISIONAL and gives no verdict.

Stages: --stage all (G3 then G4 per dataset, then summaries), g3, g4 (needs --g3-json ds=path), holm (recompute
the Holm families and narrowing-rule outcomes from --g3-json / --g4-json files).
"""
from __future__ import annotations

import argparse
import os
import sys

STAGES = ("all", "g3", "g4", "holm")
DATASETS_OK = ("baby", "sports", "clothing", "microlens")
G3_MODELS = ("EASE", "Base-retuned", "CEASE-emb", "CEASE-tags", "Add-EASE-emb", "Add-EASE-tags",
             "FEASE-tfidf", "FEASE-emb", "FEASE-full-tfidf", "L3AE", "Add-EASE-z", "FEASE-kNN")


def build_parser():
    ap = argparse.ArgumentParser(description="G3/G4: content closed-form baselines + head-on-strongest-kernel fusion")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS_OK),
                    help="baby sports clothing microlens ('elec' is reported as '--' and skipped)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026],
                    help="G4: SCOPE head checkpoint seeds. G3: seeds[0] seeds random/numpy/torch/cuda before data "
                         "loading (the closed forms themselves are deterministic)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grids, <=1 extension step, B=500 bootstrap, outputs under <out>/smoke/")
    ap.add_argument("--out", default=None, help="output dir (default <root>/results/scope/rev)")
    ap.add_argument("--log-dir", default=None, help="log dir (default <root>/logs/rev)")
    ap.add_argument("--side-dir", default=None, help="X4 output dir (default <root>/results/scope/rev/closedform_side)")
    ap.add_argument("--ckpt-dir", default=None, help="SCOPE head checkpoints (default <root>/ckpts/scope)")
    ap.add_argument("--scope-json-dir", default=None,
                    help="deployed scope_<ds>_*.json for the G4 anchor and the G3-2 seed-noise rule (default <root>/results/scope)")
    ap.add_argument("--gpu", default=None, help="physical GPU index; sets CUDA_VISIBLE_DEVICES before torch is imported")
    ap.add_argument("--tuning", choices=("coord", "joint"), default="coord",
                    help="coord: pre-committed coordinate stages for Add-EASE/L3AE/Add-EASE-z and Jeunen-style "
                         "lambda-then-alpha for CEASE-tags; joint: full product grids everywhere (~3-4x slower)")
    ap.add_argument("--solver", choices=("inv", "chol"), default="inv",
                    help="inv = torch.linalg.inv exactly as scope.ease_B (reproduces the published base); "
                         "chol = Cholesky (one fewer I x I buffer)")
    ap.add_argument("--offload", choices=("auto", "always", "never"), default="auto",
                    help="keep G and cached I x I content matrices on the CPU (auto: when I^2*4B > 1 GB or U > 50k)")
    ap.add_argument("--emb-feat", choices=("text", "image"), default="text",
                    help="frozen features behind K = F F^T (our base uses text; A^t is always text)")
    ap.add_argument("--max-extend", type=int, default=3, help="max ladder steps beyond the shared grid per param/dir")
    ap.add_argument("--boot-B", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0, help="numpy seed of the paired bootstrap (harness default 0)")
    ap.add_argument("--user-chunk", type=int, default=8192, help="user rows per R@M chunk")
    ap.add_argument("--models", nargs="+", default=None, choices=G3_MODELS,
                    help="G3 subset (EASE and Base-retuned always run: every other model depends on them)")
    ap.add_argument("--g3-json", nargs="+", default=None, help="ds=path (stage g4) or paths (stage holm)")
    ap.add_argument("--g4-json", nargs="+", default=None, help="paths (stage holm)")
    ap.add_argument("--g4-pools", nargs="+", default=["literature", "all"], choices=("literature", "all"))
    ap.add_argument("--no-anchor", action="store_true", help="skip the deployed SCOPE-v1 re-derivation in G4")
    ap.add_argument("--no-fp64", action="store_true",
                    help="skip the fp64 re-solve checks (Baby: every single-matrix model; other datasets: selected "
                         "configurations beyond the shared grid)")
    return ap


_ARGS = build_parser().parse_args() if __name__ == "__main__" else None
if _ARGS is not None and _ARGS.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_ARGS.gpu)       # before torch: scope.DEV is hard-wired to cuda:0

import datetime  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from decimal import Decimal, ROUND_HALF_UP  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = Path(os.environ["SCOPE_ROOT"]).resolve() if os.environ.get("SCOPE_ROOT") else HERE.parents[1]
for _p in (str(ROOT), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import logging  # noqa: E402

logging.disable(logging.INFO)
import scope as SC  # noqa: E402
from scope import Rmat, build_lists, mm_affinity, SCOPE, evalS_trusted, DEV  # noqa: E402
from gpu_eval import GPUEval  # noqa: E402
from src.utils import Config  # noqa: E402
from src.data.dataset import RecDataset  # noqa: E402

# fp32 solves with TF32 disabled. PyTorch's matmul default is already allow_tf32=False; made explicit.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
SOLVER = "inv"

AMAZON = ("baby", "sports", "clothing")
SIDE_DATASETS = AMAZON + ("microlens",)        # datasets with side matrices (MicroLens: official titles + categories)
LIT_ORDER = ("CEASE-emb", "CEASE-tags", "Add-EASE-emb", "Add-EASE-tags", "FEASE-tfidf", "FEASE-emb",
             "FEASE-full-tfidf", "L3AE")
SWAPS = ("Add-EASE-z", "FEASE-kNN")
NEEDS_SIDE = ("CEASE-tags", "Add-EASE-tags", "FEASE-tfidf", "FEASE-full-tfidf")
# display labels used in every printed table and stored as 'label' in the JSONs (keys stay the CLI names)
LABEL = {"FEASE-tfidf": "FEASE (prior only) tfidf", "FEASE-emb": "FEASE (prior only) emb",
         "FEASE-full-tfidf": "FEASE-full-tfidf"}
# pre-registered sets: a verdict is given only when all of them are present 
EXPECTED_DATASETS = DATASETS_OK
EXPECTED_SEEDS = (2024, 2025, 2026)
METRICS = ("R@20", "N@20")


def row_label(name):
    return LABEL.get(name, name)


def expected_models(ds):
    """Every G3 row that must be present on dataset ds for a verdict (tag-based rows need X4 side matrices)."""
    exp = [n for n in G3_MODELS if not (n in NEEDS_SIDE and ds not in SIDE_DATASETS)]
    return exp + ["Base-published"]


def r4(x):
    """Round half-up to 4 decimals from the shortest repr of the float (as printed), never banker's rounding."""
    return Decimal(repr(float(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

# ------------------------------------------------------------------------------------------------ grids
LAM = [50, 100, 200, 400, 800, 1500, 3000]                       # shared ridge grid
LAM_LADDER = [6.25, 12.5, 25] + LAM + [6000, 12000, 24000]
A_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]                # our base's text weight
A_LADDER = A_GRID + [2.0, 3.0, 5.0]
W_GRID = [1, 3, 10, 30, 100, 300, 1000]                          # CEASE-emb side weight
W_LADDER = [0.03, 0.1, 0.3] + W_GRID + [3000, 10000, 30000]
ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]             # Jeunen: linspace(0, 1, 21), weight alpha^2
ALPHA_LADDER = ALPHA_GRID + [1.05, 1.1, 1.15]
BETA_GRID = [round(0.05 * i, 2) for i in range(21)]              # Add-EASE blend, hard-bounded [0, 1]
LAMT_EMB = [1, 5, 10, 50, 100, 500]                              # content ridge, unit-diagonal K
LAMT_EMB_LADDER = [0.05, 0.1, 0.5] + LAMT_EMB + [1000, 5000, 10000]
LAMT_TAG = [10, 50, 100, 200, 500, 1000]                         # content ridge, tag Gram
LAMT_TAG_LADDER = [0.5, 1, 5] + LAMT_TAG + [2000, 5000, 10000]
LAMF = [0.1, 0.5, 1, 5, 10, 50, 100]                             # L3AE phase-1 ridge
LAMF_LADDER = [0.005, 0.01, 0.05] + LAMF + [500, 1000, 5000]
RHO_FEASE = [0.0, 0.25, 0.5, 0.75, 1.0]                          # delta/lam_tot, lambda >= 0 (hard bound 1)
RHO_L3AE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]                        # rho > 1 <=> lambda_X < 0
RHO_L3AE_LADDER = RHO_L3AE + [4.0, 5.0, 6.0]
SMULT = [0.25, 1.0, 4.0]                                         # multiplier on the Frobenius-matched FEASE scale
SMULT_LADDER = [1 / 256, 1 / 64, 1 / 16] + SMULT + [16.0, 64.0, 256.0]
ALPHA_FULL = [0.1, 0.2, 0.4]                                     # FEASE-full: weight alpha^2 on T^T T (tags_jeunen)
ALPHA_FULL_LADDER = [0.0125, 0.025, 0.05] + ALPHA_FULL + [0.8, 1.6, 3.2]
GAMMA_EQ6 = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]             # paper Eq. 6 grid (scope.train): FIXED, never extended
CENT = ["raw", "cent"]
TAGVAR = ["jeunen", "uniform"]
SMOKE_GRIDS = dict(LAM=[400, 800], A=[0.0, 0.5], W=[10, 100], ALPHA=[0.0, 0.2, 0.4], BETA=[0.0, 0.5, 1.0],
                   LAMT_EMB=[10, 100], LAMT_TAG=[50, 200], LAMF=[1, 10], RHO_F=[0.0, 0.5], RHO_L=[0.0, 1.5],
                   SMULT=[1.0], ALPHA_F=[0.1, 0.4], GAMMA=[0.0, 0.3, 1.0])

PUBLISHED_BASE = {"baby": {"lam": 800, "a": 0.5}, "sports": {"lam": 1500, "a": 0.5},
                  "clothing": {"lam": 1500, "a": 0.7}, "microlens": {"lam": 400, "a": 0.3}}
PUBLISHED_SOURCE = ("scope.closed_form_base narrow grid lam {400,800,1500} x a {0,.3,.5,.7}; selections from "
                    "'[base] tuned' lines of the SCOPE-v1 logs")
NARROW = {"lam": [400, 800, 1500], "a": [0.0, 0.3, 0.5, 0.7]}

RULE_G3_1 = ("Pre-registered rule G3-1: if any content closed form reaches our base, the base is described as one member "
             "of this family and no 'base alone outranks all baselines on MicroLens' statement is made. "
             "Operationalised here: per dataset, the rule TRIGGERS iff any "
             "literature content closed form (CEASE-emb, CEASE-tags, Add-EASE-emb, Add-EASE-tags, FEASE (prior only) "
             "tfidf/emb, FEASE-full-tfidf, L3AE) OR the re-tuned EASE reaches test Recall@20 >= Base-published test "
             "Recall@20 OR test NDCG@20 >= Base-published test NDCG@20, both values rounded half-up to 4 decimals. "
             "Triggered on a dataset -> the base is described as one member of the family there; triggered on MicroLens "
             "-> the MicroLens statement is not made. The same comparison against Base-retuned is reported for information. "
             "Holm-adjusted bootstrap p (per-dataset family) is reported alongside and is not part of the rule. No "
             "verdict unless every expected dataset and model is present.")
RULE_G3_2 = ("Pre-registered rule G3-2 (base switch): if the selected base changes, either adopt it everywhere (re-run "
             "the dependent experiments) or keep the published base and report the re-tuned one as a robustness row, the latter "
             "allowed only if the difference is within seed noise. Operationalised here: Base-retuned vs Base-published is its own family (4 datasets x {R@20, N@20}; paired user-level "
             "bootstrap; Holm within that family). 'Within seed noise' iff |delta test R@20| <= 1 x SD(R@20) AND "
             "|delta test N@20| <= 1 x SD(N@20), SD = sample SD (ddof=1) over the three deployed seeds 2024/2025/2026 "
             "of the SCOPE-v1 test metric ('fused' in results/scope/scope_<ds>_*.json); otherwise 'beyond seed noise'. "
             "The Holm p is reported and is not part of the criterion. The verdict is advisory. "
             "No verdict unless all four datasets and all three deployed seeds are present.")
RULE_G4 = ("Pre-registered rule G4: if the margin is n.s. on any dataset, the head's gain is reported as "
           "established over our base but not over the strongest content closed form on that dataset. Operationalised "
           "here: significant on a dataset iff, for EVERY head seed and BOTH metrics (R@20, N@20), the fused-minus-"
           "kernel mean difference is > 0 and its Holm-adjusted p < 0.05 (primary family); the seed-averaged "
           "comparison is reported as a secondary family. gamma on the FIXED Eq. 6 grid (no extension). No verdict "
           "unless all four datasets, all three head seeds, both metrics and every G3 kernel candidate are present.")


def make_env(args):
    g = dict(LAM=LAM, A=A_GRID, W=W_GRID, ALPHA=ALPHA_GRID, BETA=BETA_GRID, LAMT_EMB=LAMT_EMB, LAMT_TAG=LAMT_TAG,
             LAMF=LAMF, RHO_F=RHO_FEASE, RHO_L=RHO_L3AE, SMULT=SMULT, ALPHA_F=ALPHA_FULL, GAMMA=GAMMA_EQ6)
    if args.smoke:
        g.update(SMOKE_GRIDS)
    env = {k: list(v) for k, v in g.items()}
    env.update(LAM_LADDER=LAM_LADDER, A_LADDER=A_LADDER, W_LADDER=W_LADDER, ALPHA_LADDER=ALPHA_LADDER,
               LAMT_EMB_LADDER=LAMT_EMB_LADDER, LAMT_TAG_LADDER=LAMT_TAG_LADDER, LAMF_LADDER=LAMF_LADDER,
               RHO_L_LADDER=RHO_L3AE_LADDER, SMULT_LADDER=SMULT_LADDER, ALPHA_F_LADDER=ALPHA_FULL_LADDER,
               CENT=list(CENT), TAGVAR=list(TAGVAR), tuning=args.tuning)
    return env


def max_extend(args):
    return min(args.max_extend, 1) if args.smoke else args.max_extend


def boot_B(args):
    return min(args.boot_B, 500) if args.smoke else args.boot_B


# ------------------------------------------------------------------------------------------------ small helpers
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(p).resolve())


def hp_key(hp):
    return json.dumps(hp, sort_keys=True)


def hp_str(hp):
    return " ".join(f"{k}={v}" for k, v in hp.items())


KEYS = ("Recall@10", "NDCG@10", "Precision@10", "Recall@20", "NDCG@20", "Precision@20")


def mdict(m):
    return {k: float(m[k]) for k in KEYS if k in m}


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 3)


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def dump_json(path, obj):
    """Atomic write (tmp + os.replace): a crash never leaves a truncated JSON."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jdefault))
    os.replace(tmp, path)


def fresh_path(d, stem, suffix):
    """Never overwrite: <stem><suffix>, else <stem>_1<suffix>, ..."""
    p = Path(d) / f"{stem}{suffix}"
    k = 1
    while p.exists():
        p = Path(d) / f"{stem}_{k}{suffix}"
        k += 1
    return p


class Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s); self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush(); self.fh.flush()

    def isatty(self):
        return False


def env_info():
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                numpy=np.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(), solver=SOLVER, root=str(ROOT),
                scope_py=rel(SC.__file__))


# ------------------------------------------------------------------------------------------------ linear algebra
class RChunks:
    """Row chunks of the sparse train matrix R (built once) for memory-lean R @ M products (the same chunking as
    scope.spmm_lowmem, without rebuilding the chunks on every call)."""

    def __init__(self, R, chunk):
        R = R.coalesce()
        idx, val = R.indices(), R.values()
        self.U, self.I = int(R.shape[0]), int(R.shape[1])
        self.chunks = []
        for s in range(0, self.U, chunk):
            e = min(s + chunk, self.U)
            m = (idx[0] >= s) & (idx[0] < e)
            rc = torch.sparse_coo_tensor(torch.stack([idx[0][m] - s, idx[1][m]]), val[m], (e - s, self.I)).coalesce()
            self.chunks.append((s, e, rc))
        assert sum(int(rc._nnz()) for _, _, rc in self.chunks) == int(R._nnz()), "RChunks lost nonzeros"


def gram_from_chunks(RC):
    """G = R^T R (fp32; integer counts, exact), accumulated over user chunks (no dense [U, I] transient)."""
    G = torch.zeros(RC.I, RC.I, device=DEV)
    for _, _, rc in RC.chunks:
        Rd = rc.to_dense()
        G.addmm_(Rd.t(), Rd)
        del Rd
    return G


def zscores_fn(RC, chunk_fn, dt, cast_first, out):
    """out[u] = per-user z-score of chunk_fn(R rows), written chunk by chunk into out ([U, I], dtype dt).
    Same arithmetic as scope.zr: (x - mean) / (std_unbiased + 1e-9) in fp32. cast_first=True reproduces
    zr(spmm(R, B).to(dt)) (scope's EASE path: cast to dt, then z in fp32, then cast); cast_first=False reproduces
    zr(spmm(R, A)).to(dt) (scope's non-frugal text path: z in fp32, then cast)."""
    for s, e, rc in RC.chunks:
        blk = chunk_fn(rc)
        if cast_first and dt != torch.float32:
            blk = blk.to(dt).float()
        m = blk.mean(1, keepdim=True)
        sd = blk.std(1, keepdim=True)
        blk.sub_(m).div_(sd + 1e-9)
        out[s:e].copy_(blk)
        del blk
    return out


def zscore_inplace(S, chunk=8192):
    """Per-row z-score of an existing [U, I] matrix in place (fp32 math per row chunk), as scope.zr."""
    for s in range(0, S.shape[0], chunk):
        e = min(s + chunk, S.shape[0])
        blk = S[s:e].float()
        m = blk.mean(1, keepdim=True)
        sd = blk.std(1, keepdim=True)
        blk.sub_(m).div_(sd + 1e-9)
        if blk.data_ptr() != S[s:e].data_ptr():
            S[s:e].copy_(blk)
        del blk
    return S


def add_inplace(Z, T, a, chunk=8192):
    """Z <- Z + a*T in Z's dtype with the SAME two roundings as scope's `S0 + aa * txt` (a*T rounded first)."""
    for s in range(0, Z.shape[0], chunk):
        e = min(s + chunk, Z.shape[0])
        Z[s:e].add_(T[s:e] * a)
    return Z


def ridge_P(Gr, lam):
    """P = (Gr + lam I)^-1 for a symmetric positive-definite Gr; Gr is modified in place and should be a temporary."""
    Gr.diagonal().add_(lam)
    if SOLVER == "chol":
        L = torch.linalg.cholesky(Gr)
        del Gr
        P = torch.cholesky_inverse(L)
        del L
        return P
    return torch.linalg.inv(Gr)                   # identical op to scope.ease_B


def ease_from_P(P):
    """In place: B = P / (-diag(P)) column-wise, diag(B) = 0 -- the arithmetic of scope.ease_B."""
    d = torch.diagonal(P).clone().neg_()
    P.div_(d.unsqueeze(0))
    P.fill_diagonal_(0.0)
    return P


def prior_B(P, PM, ds_):
    """FEASE-I-Prior / L3AE phase-2 closed form with a column scaling (never P @ diag(.)):
    B = ds_ * PM - P * ((1 + ds_ * diag(PM)) / diag(P))[None, :], diag(B) = 0  (the +I term only touches the diagonal).
    ds_ = 0 gives exactly EASE(lam_tot)."""
    v = (1.0 + ds_ * torch.diagonal(PM)) / torch.diagonal(P)
    B = P * v.unsqueeze(0)
    B.neg_()
    if ds_ != 0.0:
        B.add_(PM, alpha=ds_)
    B.fill_diagonal_(0.0)
    return B


def sparse_gram(X, block=1024):
    """Dense X X^T (fp32, on the GPU) for an item-major scipy CSR X [I, V], in item-column blocks."""
    I = X.shape[0]
    coo = X.tocoo()
    Xs = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
                                 torch.from_numpy(coo.data.astype(np.float32)), (I, X.shape[1])).coalesce().to(DEV)
    out = torch.empty(I, I, dtype=torch.float32, device=DEV)
    for s in range(0, I, block):
        e = min(s + block, I)
        D = torch.from_numpy(np.ascontiguousarray(X[s:e].toarray(), dtype=np.float32)).to(DEV)
        out[:, s:e] = torch.sparse.mm(Xs, D.t().contiguous())
        del D
    del Xs
    return out


def sym_check(M, exact, n=20000, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, M.shape[0], (n,), generator=g).to(DEV)
    j = torch.randint(0, M.shape[0], (n,), generator=g).to(DEV)
    d = (M[i, j] - M[j, i]).abs().max().item()
    tol = 0.0 if exact else 1e-5 * max(1.0, M[i, j].abs().max().item())
    if d > tol:
        raise AssertionError(f"side Gram not symmetric on sampled pairs (max |diff| {d:.3g})")
    return d


class GPUEvalPU(GPUEval):
    """GPUEval plus per-user Recall@20 / NDCG@20 computed with exactly the formulas of GPUEval._run (the users are
    self.users, sorted ascending as in EvalDataLoader). Consistency with GPUEval.eval and evalS_trusted is asserted
    in test_eval()."""

    @torch.no_grad()
    def per_user(self, get_rows, k=20, batch=4096):
        U = self.users.numel()
        rec = torch.zeros(U, device=self.dev, dtype=torch.float64)
        nd = torch.zeros(U, device=self.dev, dtype=torch.float64)
        disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=self.dev).float())
        for s in range(0, U, batch):
            bu = self.users[s:s + batch]
            sc = get_rows(bu).float().clone()
            sc = self._mask(sc, bu)
            _, idx = torch.topk(sc, self.maxk, dim=1)
            bp = self.pos[s:s + batch]
            hit = (idx.unsqueeze(2) == bp.unsqueeze(1)).any(2).float()[:, :k]
            nrel = self.nfit[s:s + batch].clamp(min=1)
            rec[s:s + bu.numel()] = (hit.sum(1) / nrel).double()
            dcg = (hit * disc.unsqueeze(0)).sum(1)
            idcg = self.cumdisc[torch.minimum(nrel, torch.full_like(nrel, k)).long()]
            nd[s:s + bu.numel()] = (dcg / idcg.clamp(min=1e-9)).double()
        return rec, nd


# ------------------------------------------------------------------------------------------------ statistics
def paired_bootstrap_multi(diffs, B=10000, seed=0, chunk=250):
    """Paired user-level bootstrap of mean(diff) for several per-user difference vectors that share the same users.
    Identical statistic to scope/harness.paired_bootstrap (idx = rng.integers(0, n, (B, n)) with
    default_rng(seed); two-sided p = 2*min(P(boot<=0), P(boot>=0)); percentile CI) -- the resampling matrix is drawn
    in row chunks from the same generator stream to bound RAM, and it is shared by all vectors (as repeated harness
    calls with seed=0 would be). The --smoke run checks equality with harness.paired_bootstrap."""
    names = list(diffs)
    if not names:
        return {}
    n = len(diffs[names[0]])
    arr = {k: np.asarray(diffs[k], dtype=np.float64) for k in names}
    for k in names:
        assert arr[k].shape == (n,), f"bootstrap vector {k} has shape {arr[k].shape}, expected ({n},)"
    rng = np.random.default_rng(seed)
    boots = {k: np.empty(B) for k in names}
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        idx = rng.integers(0, n, size=(b, n))
        for k in names:
            boots[k][s:s + b] = arr[k][idx].mean(1)
        del idx
    out = {}
    for k in names:
        bt = boots[k]
        p_h = 2.0 * min((bt <= 0).mean(), (bt >= 0).mean())          # harness definition (can be exactly 0)
        le, ge = int((bt <= 0).sum()), int((bt >= 0).sum())
        p = 2.0 * min((le + 1) / (B + 1), (ge + 1) / (B + 1))        # used for inference/Holm: never 0
        out[k] = dict(mean_delta=float(arr[k].mean()), ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                      p_two_sided=float(min(p, 1.0)), p_two_sided_harness=float(min(p_h, 1.0)),
                      p_rule="2*min((#boot<=0 + 1)/(B+1), (#boot>=0 + 1)/(B+1)), capped at 1",
                      p_min_attainable=2.0 / (B + 1), n_users=int(n), B=int(B), seed=int(seed))
    return out


def _reference_paired_bootstrap(a, b, B=10000, seed=0):
    """Verbatim statistic of scope/harness.paired_bootstrap (used only by the smoke equivalence check
    when harness.py cannot be imported without side effects)."""
    a = np.asarray(a, float); b = np.asarray(b, float); diff = a - b; n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = diff[idx].mean(1)
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(mean_delta=float(diff.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                p_two_sided=float(min(p, 1.0)), n_users=int(n))


def check_bootstrap_equivalence():
    rng = np.random.default_rng(123)
    a, b = rng.random(777), rng.random(777) * 0.95
    src = "verbatim copy of harness.paired_bootstrap"
    ref_pb = _reference_paired_bootstrap
    if (ROOT / "results" / "scope" / "significance").is_dir():    # harness.py mkdirs that folder at import time
        try:
            sys.path.insert(0, str(HERE.parent))
            from harness import paired_bootstrap as ref_pb  # noqa: F811
            src = "scope/harness.paired_bootstrap"
        except Exception as e:                                     # pragma: no cover
            print(f"[bootstrap-check] harness import failed ({e}); using the verbatim copy", flush=True)
    ref = ref_pb(a, b, B=3000, seed=0)
    mine = paired_bootstrap_multi({"x": a - b}, B=3000, seed=0, chunk=250)["x"]
    # the resampling is compared through the harness p definition; the (count+1)/(B+1) p used for inference is a
    # deterministic function of the same counts
    same = (ref["mean_delta"] == mine["mean_delta"] and ref["p_two_sided"] == mine["p_two_sided_harness"]
            and np.allclose(ref["ci95"], mine["ci95"], rtol=0.0, atol=1e-15))
    print(f"[bootstrap-check] chunked bootstrap vs {src}: {'IDENTICAL' if same else 'DIFFERENT'} "
          f"(p {ref['p_two_sided']:.4f}/{mine['p_two_sided_harness']:.4f}, ci {ref['ci95']} / {mine['ci95']})", flush=True)
    if not same:
        print("WARNING: the chunked bootstrap is a valid paired bootstrap but not bit-identical to harness.py", flush=True)
    return dict(reference=src, identical=bool(same), ref=ref, chunked=mine)


def holm(pvals):
    """Holm step-down adjusted p-values over the given family {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ------------------------------------------------------------------------------------------------ per-dataset context
class Ctx:
    """Data, the train matrix, G = R^T R, evaluators and the shared [U, I] score buffer of one dataset."""

    def __init__(self, ds, args, seed):
        t0 = time.time()
        self.ds, self.args, self.seed = ds, args, seed
        seed_all(seed)                                   # BEFORE data loading and build_lists (recorded in the JSON)
        self.dset = RecDataset(Config("scope", ds))
        self.U, self.I = int(self.dset.n_users), int(self.dset.n_items)
        self.R = Rmat(self.dset)
        _items, _vmask, deg = build_lists(self.dset)     # deg feeds SCOPE.score_all in G4 (mean-pool denominator)
        self.degf = deg.float()
        del _items, _vmask
        self.frugal = self.U > 50000                     # MicroLens (98k users): scope's frugal path
        self.dt = torch.float16 if (self.I > 20000 or self.frugal) else torch.float32   # as scope.py / harness
        self.offload = {"auto": (self.I * self.I * 4 > 1.0e9) or self.frugal, "always": True,
                        "never": False}[args.offload]
        self.RC = RChunks(self.R, args.user_chunk)
        G = gram_from_chunks(self.RC)
        self.gram_check = None
        if self.I <= 10000 and not self.frugal:          # Baby: compare with scope.gram (dense path), must be equal
            self.gram_check = bool(torch.equal(G, SC.gram(self.R)))
            if not self.gram_check:
                raise AssertionError(f"[{ds}] chunked Gram differs from scope.gram")
        self.G_store = G.cpu() if self.offload else G
        del G
        torch.cuda.empty_cache()
        self.gevV = GPUEval(self.dset, "valid", DEV)
        self.gevT = GPUEvalPU(self.dset, "test", DEV)
        self.Sbuf = torch.empty(self.U, self.I, dtype=self.dt, device=DEV)
        self.sbuf_owner = None
        self._feat = {}
        self._side = (None, None)
        self.side_stats = {}
        self.inter_path = Path(self.dset.data_path) / str(self.dset.config["inter_file_name"])
        self.inter_sha256 = sha256_file(self.inter_path)
        sd = Path(args.side_dir) if args.side_dir else ROOT / "results" / "scope" / "rev" / "closedform_side"
        if args.smoke and args.side_dir is None and (sd / "smoke" / f"{ds}_side_meta.json").is_file():
            sd = sd / "smoke"
        self.side_dir, self.side_meta = sd, None
        if ds in SIDE_DATASETS:
            mp = sd / f"{ds}_side_meta.json"
            if not mp.is_file():
                raise FileNotFoundError(f"{mp} missing: run scope/side_matrices.py first")
            sm = json.loads(mp.read_text())
            if int(sm["n_items"]) != self.I:
                raise AssertionError(f"[{ds}] side matrices built for n_items={sm['n_items']}, dataset has {self.I}")
            if sm["inter_sha256"] != self.inter_sha256:
                raise AssertionError(f"[{ds}] side matrices were built against a different .inter file")
            self.side_meta = sm
        self.build_s = round(time.time() - t0, 2)

    def close(self):
        for a in ("Sbuf", "G_store", "R", "RC", "gevV", "gevT", "degf"):
            setattr(self, a, None)
        self._feat, self._side = {}, (None, None)
        torch.cuda.empty_cache()

    def describe(self):
        return dict(dataset=self.ds, n_users=self.U, n_items=self.I, n_train_nnz=int(self.R._nnz()),
                    n_valid_users=int(self.gevV.users.numel()), n_test_users=int(self.gevT.users.numel()),
                    score_dtype=str(self.dt), frugal=self.frugal, offload_dense_II_to_cpu=self.offload,
                    user_chunk=self.args.user_chunk, inter_file=rel(self.inter_path), inter_sha256=self.inter_sha256,
                    emb_feat=self.args.emb_feat, emb_feat_shape=list(self._src().shape),
                    gram_equals_scope_gram=self.gram_check, seed=self.seed,
                    side_dir=rel(self.side_dir) if self.side_meta else None,
                    side_files=({k: dict(file=v["file"], sha256=v["sha256"], shape=v["shape"], nnz=v["nnz"])
                                 for k, v in self.side_meta["files"].items()} if self.side_meta else None),
                    ctx_build_s=self.build_s)

    def _src(self, which=None):
        which = which or self.args.emb_feat
        return self.dset.t_feat if which == "text" else self.dset.v_feat

    def G_copy(self, dtype=torch.float32):
        return self.G_store.to(device=DEV, dtype=dtype, copy=True)

    def stash(self, t):
        return t.cpu() if self.offload else t

    def feat(self, c, dtype=torch.float32, which=None):
        """Row-L2-normalised frozen features, raw or mean-centred over items (F.normalize with explicit dim=1).
        which=None follows --emb-feat; which='text' forces the text features (the controlled swap Add-EASE-z)."""
        which = which or self.args.emb_feat
        key = (which, c, str(dtype))
        if key not in self._feat:
            X = torch.from_numpy(np.asarray(self._src(which)[:], dtype=np.float64 if dtype == torch.float64
                                            else np.float32)).to(DEV)
            if c == "cent":
                X = X - X.mean(0, keepdim=True)
            elif c != "raw":
                raise ValueError(c)
            self._feat[key] = F.normalize(X, dim=1)
        return self._feat[key]

    def side_file(self, kind):
        if self.side_meta is None:
            raise RuntimeError(f"[{self.ds}] no side matrices on this dataset")
        info = self.side_meta["files"][kind]
        p = self.side_dir / info["file"]
        if sha256_file(p) != info["sha256"]:
            raise AssertionError(f"{p}: sha256 differs from {self.ds}_side_meta.json")
        X = sp.load_npz(p).tocsr().astype(np.float32)
        if X.shape[0] != self.I or X.nnz != int(info["nnz"]) or X.nnz == 0:
            raise AssertionError(f"{p}: shape/nnz {X.shape}/{X.nnz} inconsistent with metadata / n_items")
        return X

    def side_gram(self, kind):
        """Tag Gram T^T T (item x item shared-tag counts), exact in fp32; one cached at a time (CPU if offload)."""
        if self._side[0] == ("gram", kind):
            return self._side[1]
        self._side = (None, None)
        X = self.side_file(kind)
        Gt = sparse_gram(X)
        rown = torch.from_numpy(np.diff(X.indptr).astype(np.float32)).to(DEV)
        if not torch.equal(torch.diagonal(Gt), rown):
            raise AssertionError(f"[{self.ds}] {kind}: diag(T^T T) != number of tags per item")
        sym_check(Gt, exact=True)
        self.side_stats[kind] = dict(shape=list(X.shape), nnz=int(X.nnz), diag_checked=True, symmetric_checked=True)
        self._side = (("gram", kind), self.stash(Gt))
        del Gt
        return self._side[1]

    def side_cos(self, kind="tfidf"):
        """TF-IDF cosine C = X X^T (rows l2-normalised, checked); diagonal zeroed (provably irrelevant for FEASE)."""
        X = self.side_file(kind)
        C = sparse_gram(X)
        nz = torch.from_numpy(np.diff(X.indptr) > 0).to(DEV)
        d = torch.diagonal(C)
        if not (torch.allclose(d[nz], torch.ones_like(d[nz]), atol=1e-3) and bool((d[~nz] == 0).all())):
            raise AssertionError(f"[{self.ds}] {kind}: diag(X X^T) is not 1 on non-empty rows")
        sym_check(C, exact=False)
        C.fill_diagonal_(0.0)
        self.side_stats[kind] = dict(shape=list(X.shape), nnz=int(X.nnz), empty_rows=int((~nz).sum().item()))
        return C

    def free_side(self):
        self._side = (None, None)
        torch.cuda.empty_cache()

    def score_B(self, B, cast_first=True, out=None):
        """z(R @ B) into the shared buffer (or `out`), exactly scope's `zr(spmm(R, B).to(dt))` arithmetic."""
        dst = self.Sbuf if out is None else out
        zscores_fn(self.RC, lambda rc: torch.sparse.mm(rc, B), self.dt, cast_first, dst)
        if dst is self.Sbuf:
            self.sbuf_owner = None
        return dst


# ------------------------------------------------------------------------------------------------ models
def neighbours(grid, ladder, v):
    L = sorted(grid) if v in grid else sorted(ladder)
    i = L.index(v)
    return L[max(0, i - 1): i + 2]


class Model:
    name, family, literature, kind = "?", "?", False, "single"
    corner = {}          # fp64 check: worst-conditioned corner, param -> 'min'/'max' of the explored grid

    def __init__(self, ctx, env):
        self.ctx, self.env = ctx, env
        self.params, self.stages, self.init_fixed = [], [], {}
        self.grids, self.ladders, self.shared = {}, {}, {}
        self.meta = {}

    def setp(self, name, grid, ladder=None, shared=None):
        self.grids[name] = list(grid)
        if ladder is not None:                           # numeric parameter: edge detection + ladder extension
            self.ladders[name] = sorted(set(ladder) | set(grid))
            self.shared[name] = list(shared if shared is not None else grid)

    def describe(self):
        return dict(name=self.name, family=self.family, literature=self.literature, kind=self.kind,
                    params=list(self.params), stages=[list(s) for s in self.stages], init_fixed=dict(self.init_fixed),
                    grids={k: list(v) for k, v in self.grids.items()},
                    shared_grid={k: list(v) for k, v in self.shared.items()},
                    ladders={k: list(v) for k, v in self.ladders.items()})

    def scores(self, hp):
        B = self.build_B(hp)
        S = self.ctx.score_B(B)
        del B
        return S

    def val_eval(self, hp):
        return mdict(self.ctx.gevV.eval(self.scores(hp)))

    def final_scores(self, hp):
        return self.scores(hp)

    def release(self):
        pass


class EASEModel(Model):
    name, family = "EASE", "EASE"
    corner = {"lam": "min"}

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.params, self.stages = ["lam"], [["lam"]]
        self.setp("lam", env["LAM"], env["LAM_LADDER"])

    def build_B(self, hp, dtype=torch.float32):
        return ease_from_P(ridge_P(self.ctx.G_copy(dtype), hp["lam"]))


class BaseModel(Model):
    """Our base (scope.closed_form_base, 1-hop): S = z(R B_lam) + a z(R A^t), both terms z-scored exactly as there."""
    name, family, kind = "Base-retuned", "ours", "fused"

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.params, self.stages = ["lam", "a"], [["lam", "a"]]
        self.setp("lam", env["LAM"], env["LAM_LADDER"])
        self.setp("a", env["A"], env["A_LADDER"])
        At = mm_affinity(np.asarray(ctx.dset.t_feat[:]))       # scope's text kNN kernel (top-20, clamp>=0, diag 0)
        self.meta["At_nnz"] = int((At > 0).sum().item())
        self.T = torch.empty(ctx.U, ctx.I, dtype=ctx.dt, device=DEV)
        ctx.score_B(At, cast_first=ctx.frugal, out=self.T)      # == closed_form_base's `txt`
        del At
        torch.cuda.empty_cache()

    def _ZE(self, lam):
        key = ("Base.ZE", lam)
        if self.ctx.sbuf_owner != key:
            B = ease_from_P(ridge_P(self.ctx.G_copy(), lam))
            self.ctx.score_B(B)
            del B
            self.ctx.sbuf_owner = key
        return self.ctx.Sbuf

    def val_eval(self, hp):
        Z, a, T = self._ZE(hp["lam"]), float(hp["a"]), self.T
        if a == 0.0:
            return mdict(self.ctx.gevV.eval(Z))
        return mdict(self.ctx.gevV.eval_streaming(lambda bu: Z[bu] + a * T[bu]))

    def final_scores(self, hp):
        Z, a = self._ZE(hp["lam"]), float(hp["a"])
        if a != 0.0:
            add_inplace(Z, self.T, a)
        self.ctx.sbuf_owner = None
        return Z

    def release(self):
        self.T = None
        torch.cuda.empty_cache()


class CEASEEmb(Model):
    """CEASE with dense frozen features: EASE on the stacked matrix [R; sqrt(w) F^T] -> Gram G + w F F^T."""
    name, family, literature = "CEASE-emb", "CEASE", True
    corner = {"w": "max", "lam": "min"}

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.params, self.stages = ["c", "w", "lam"], [["c", "w", "lam"]]
        self.setp("c", env["CENT"])
        self.setp("w", env["W"], env["W_LADDER"])
        self.setp("lam", env["LAM"], env["LAM_LADDER"])

    def build_B(self, hp, dtype=torch.float32):
        Fc = self.ctx.feat(hp["c"], dtype)
        Gr = self.ctx.G_copy(dtype)
        Gr.addmm_(Fc, Fc.t(), alpha=float(hp["w"]))
        return ease_from_P(ridge_P(Gr, hp["lam"]))


class CEASETags(Model):
    """CEASE with binary tags (Jeunen et al.): EASE on G + alpha^2 T^T T (the reference code stacks alpha*T)."""
    name, family, literature = "CEASE-tags", "CEASE", True
    corner = {"alpha": "max", "lam": "min"}

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.params = ["tv", "alpha", "lam"]
        self.setp("tv", env["TAGVAR"])
        self.setp("alpha", env["ALPHA"], env["ALPHA_LADDER"])
        if env["tuning"] == "joint":
            self.stages = [["tv", "lam", "alpha"]]
            self.setp("lam", env["LAM"], env["LAM_LADDER"], env["LAM"])
        else:        # Jeunen: lambda from EASE first, then alpha; then lambda refined around its value
            self.stages = [["tv", "alpha"], ["lam", "alpha"]]
            self.init_fixed = {"lam": env["lamE"]}
            self.setp("lam", neighbours(env["LAM"], env["LAM_LADDER"], env["lamE"]), env["LAM_LADDER"], env["LAM"])

    def build_B(self, hp, dtype=torch.float32):
        Gt = self.ctx.side_gram("tags_" + hp["tv"])
        Gr = self.ctx.G_copy(dtype)
        Gr.add_(Gt.to(DEV, dtype), alpha=float(hp["alpha"]) ** 2)
        return ease_from_P(ridge_P(Gr, hp["lam"]))


class AddEASE(Model):
    """Add-EASE (Jeunen et al.): S = R((1-beta) B_R + beta B_T), two independent ridge solves, raw blend."""
    family, literature = "Add-EASE", True
    corner = {"lam_T": "min", "lam_R": "min"}

    def __init__(self, ctx, env, content):
        super().__init__(ctx, env)
        self.content = content
        self.name = f"Add-EASE-{content}"
        self.ck = "c" if content == "emb" else "tv"
        self.params = [self.ck, "lam_T", "beta", "lam_R"]
        self.setp(self.ck, env["CENT"] if content == "emb" else env["TAGVAR"])
        if content == "emb":
            self.setp("lam_T", env["LAMT_EMB"], env["LAMT_EMB_LADDER"])
        else:
            self.setp("lam_T", env["LAMT_TAG"], env["LAMT_TAG_LADDER"])
        self.setp("beta", env["BETA"], env["BETA"])             # hard-bounded [0, 1]
        self.setp("lam_R", env["LAM"], env["LAM_LADDER"])
        if env["tuning"] == "joint":
            self.stages = [["lam_R", self.ck, "lam_T", "beta"]]
        else:
            self.stages = [[self.ck, "lam_T", "beta"], ["lam_R", "beta"]]
            self.init_fixed = {"lam_R": env["lamE"]}
        self._BR = self._BT = None

    def _content_gram(self, v, dtype):
        if self.content == "emb":
            Fc = self.ctx.feat(v, dtype)
            return Fc @ Fc.t()
        return self.ctx.side_gram("tags_" + v).to(DEV, dtype, copy=True)

    def BR(self, lam, dtype=torch.float32):
        if dtype != torch.float32:
            return ease_from_P(ridge_P(self.ctx.G_copy(dtype), lam))
        if self._BR is None or self._BR[0] != lam:
            self._BR = None
            self._BR = (lam, ease_from_P(ridge_P(self.ctx.G_copy(), lam)))
        return self._BR[1]

    def BT(self, v, lam, dtype=torch.float32):
        if dtype != torch.float32:
            return ease_from_P(ridge_P(self._content_gram(v, dtype), lam))
        if self._BT is None or self._BT[0] != (v, lam):
            self._BT = None
            self._BT = ((v, lam), ease_from_P(ridge_P(self._content_gram(v, torch.float32), lam)))
        return self._BT[1]

    def scores(self, hp):
        BR, BT, b = self.BR(hp["lam_R"]), self.BT(hp[self.ck], hp["lam_T"]), float(hp["beta"])
        self.ctx.sbuf_owner = None
        return zscores_fn(self.ctx.RC,
                          lambda rc: torch.sparse.mm(rc, BR).mul_(1.0 - b).add_(torch.sparse.mm(rc, BT), alpha=b),
                          self.ctx.dt, True, self.ctx.Sbuf)

    def build_B(self, hp, dtype=torch.float32):              # explicit blended matrix (fp64 check only)
        b = float(hp["beta"])
        return self.BR(hp["lam_R"], dtype).mul(1.0 - b).add_(self.BT(hp[self.ck], hp["lam_T"], dtype), alpha=b)

    def release(self):
        self._BR = self._BT = None


class PriorModel(Model):
    """FEASE-I-Prior and L3AE phase 2 are one estimator:
    B = I + d P M - P diagMat((1 + d diag(PM)) / diag P), P = (G + lam_tot I)^-1, diag(B) = 0. Caches P and PM."""

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self._P = self._PM = self._M = None

    def p_key(self, hp):
        """Everything P depends on (FEASE-full adds alpha)."""
        return (hp["lam_tot"],)

    def gram(self, hp, dtype):
        """The Gram inside P = (gram + lam_tot I)^-1: G = R^T R (FEASE-full: G + alpha^2 T^T T). A fresh temporary."""
        return self.ctx.G_copy(dtype)

    def P(self, hp):
        pk = self.p_key(hp)
        if self._P is None or self._P[0] != pk:
            self._PM = None
            self._P = None
            self._P = (pk, ridge_P(self.gram(hp, torch.float32), hp["lam_tot"]))
        return self._P[1]

    def PM(self, key, hp):
        P = self.P(hp)
        pk = self.p_key(hp)
        if self._PM is None or self._PM[0] != (key, pk):
            self._PM = None
            self._PM = ((key, pk), self._pm(P, key, torch.float32))
        return self._PM[1]

    def _pm(self, P, key, dtype):
        if key[0] == "K":                                    # M = Fc Fc^T is low rank: PM = (P Fc) Fc^T, M never formed
            Fc = self.ctx.feat(key[1], dtype)
            return (P @ Fc) @ Fc.t()
        return P @ self.M(key, dtype).to(DEV, dtype)

    def build_B(self, hp, dtype=torch.float32):
        key, lt = self.prior_key(hp), hp["lam_tot"]
        if dtype == torch.float32:
            P, PM = self.P(hp), self.PM(key, hp)
        else:
            P = ridge_P(self.gram(hp, dtype), lt)
            PM = self._pm(P, key, dtype)
        return prior_B(P, PM, self.delta_scale(hp))

    def release(self):
        self._P = self._PM = self._M = None


class FEASEModel(PriorModel):
    """FEASE-I-Prior without user attributes; M = TF-IDF cosine | K = F F^T | our A^t (swap FEASE-kNN).
    variants 'tfidf'/'emb' are 'FEASE (prior only)' (alpha = 0: EASE Gram plus the prior); variant 'full-tfidf' is the
    full FEASE model: EASE on the stacked [R; alpha T] (Gram G + alpha^2 T^T T, T = tags_jeunen) plus the TF-IDF prior."""
    family, literature = "FEASE", True
    corner = {"lam_tot": "min", "rho": "max", "smult": "max"}

    def __init__(self, ctx, env, variant):
        super().__init__(ctx, env)
        self.variant = variant
        self.name = {"tfidf": "FEASE-tfidf", "emb": "FEASE-emb", "full-tfidf": "FEASE-full-tfidf",
                     "knn": "FEASE-kNN"}[variant]
        if variant == "knn":
            self.family, self.literature = "swap", False
        self.params = (["lam_tot"] + (["alpha"] if variant == "full-tfidf" else [])
                       + (["c"] if variant == "emb" else []) + ["rho", "smult"])
        self.stages = [list(self.params)]
        self.setp("lam_tot", env["LAM"], env["LAM_LADDER"])
        if variant == "full-tfidf":
            self.setp("alpha", env["ALPHA_F"], env["ALPHA_F_LADDER"])
            self.corner = dict(FEASEModel.corner, alpha="max")
            self.meta["side_gram"] = "tags_jeunen (X4), weight alpha^2, stacked-matrix convention as CEASE-tags"
        if variant == "emb":
            self.setp("c", env["CENT"])
        self.setp("rho", env["RHO_F"], env["RHO_F"])              # hard-bounded: lambda = lam_tot*(1-rho) >= 0
        self.setp("smult", env["SMULT"], env["SMULT_LADDER"])
        self._offfro = {}
        self.meta["label"] = row_label(self.name)
        self.meta["scale_rule"] = ("M is scaled by s = smult * ||offdiag B_EASE(lam*_EASE)||_F / ||offdiag M||_F "
                                   "(FEASE leaves the scale unspecified; this is our choice)")
        self.meta["B_EASE_offdiag_fro"] = env["BEASE_fro"]

    def p_key(self, hp):
        return (hp["lam_tot"], hp["alpha"]) if self.variant == "full-tfidf" else (hp["lam_tot"],)

    def gram(self, hp, dtype):
        Gr = self.ctx.G_copy(dtype)
        if self.variant == "full-tfidf":
            Gr.add_(self.ctx.side_gram("tags_jeunen").to(DEV, dtype), alpha=float(hp["alpha"]) ** 2)
        return Gr

    def prior_key(self, hp):
        if self.variant == "emb":
            return ("K", hp["c"])
        return ("tfidf",) if self.variant == "full-tfidf" else (self.variant,)

    def _build_M(self, key):
        if key[0] == "tfidf":
            return self.ctx.side_cos("tfidf")
        if key[0] == "knn":
            return mm_affinity(np.asarray(self.ctx.dset.t_feat[:]))   # our A^t (diag already 0)
        raise ValueError(key)

    def M(self, key, dtype=torch.float32):
        if dtype != torch.float32:
            return self._build_M(key).to(DEV, dtype)
        if self._M is None or self._M[0] != key:
            self._M = None
            self._PM = None
            Mx = self._build_M(key)
            self._offfro.setdefault(key, float(torch.linalg.norm(Mx)))       # diagonal is zero
            self._M = (key, self.ctx.stash(Mx))
            del Mx
        return self._M[1]

    def offfro(self, key):
        if key not in self._offfro:
            if key[0] == "K":        # ||offdiag(F F^T)||_F^2 = ||F^T F||_F^2 - sum_i ||f_i||^4
                Fc = self.ctx.feat(key[1])
                g = Fc.t() @ Fc
                n4 = (Fc * Fc).sum(1).pow(2).sum()
                self._offfro[key] = float(torch.sqrt(((g * g).sum() - n4).clamp(min=0)))
            else:
                self.M(key)
        return self._offfro[key]

    def delta_scale(self, hp):
        key = self.prior_key(hp)
        s0 = self.env["BEASE_fro"] / self.offfro(key)
        self.meta.setdefault("s0", {})[str(key)] = s0
        return float(hp["rho"]) * float(hp["lam_tot"]) * s0 * float(hp["smult"])


class L3AEModel(PriorModel):
    """L3AE (Moon et al., CIKM'25): phase 1 S_hat = EASE(K, lam_F); phase 2 prior-EASE with M = S_hat,
    lam_KD = rho*lam_tot, lam_X = (1-rho)*lam_tot. Our frozen text features stand in for NV-Embed-v2."""
    name, family, literature = "L3AE", "L3AE", True
    corner = {"lam_F": "min", "rho": "max", "lam_tot": "min"}

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.params = ["c", "lam_F", "rho", "lam_tot"]
        self.setp("c", env["CENT"])
        self.setp("lam_F", env["LAMF"], env["LAMF_LADDER"])
        self.setp("rho", env["RHO_L"], env["RHO_L_LADDER"])
        self.setp("lam_tot", env["LAM"], env["LAM_LADDER"])
        if env["tuning"] == "joint":
            self.stages = [["c", "lam_F", "lam_tot", "rho"]]
        else:
            self.stages = [["c", "lam_F", "rho"], ["lam_tot", "rho"]]
            self.init_fixed = {"lam_tot": env["lamE"]}
        self.meta["features"] = "frozen dataset text features (Amazon 384-d SBERT, MicroLens 1024-d), not NV-Embed-v2"

    def prior_key(self, hp):
        return ("Shat", hp["c"], hp["lam_F"])

    def M(self, key, dtype=torch.float32):
        _, c, lam_F = key
        if dtype != torch.float32:
            Fc = self.ctx.feat(c, dtype)
            return ease_from_P(ridge_P(Fc @ Fc.t(), lam_F))
        if self._M is None or self._M[0] != key:
            self._M = None
            self._PM = None
            Fc = self.ctx.feat(c)
            Sh = ease_from_P(ridge_P(Fc @ Fc.t(), lam_F))
            self._M = (key, self.ctx.stash(Sh))
            del Sh
        return self._M[1]

    def delta_scale(self, hp):
        return float(hp["rho"]) * float(hp["lam_tot"])


class AddEASEz(Model):
    """Controlled swap: z(R B_R) + a z(R B_T) -- our base's late z-fusion (same a-grid, same text features, same
    z-score arithmetic per term) with a ridge-EASE content kernel B_T = EASE(K_raw, lam_T) instead of A^t."""
    name, family, kind = "Add-EASE-z", "swap", "fused"

    def __init__(self, ctx, env):
        super().__init__(ctx, env)
        self.params = ["lam_T", "a", "lam_R"]
        self.setp("lam_T", env["LAMT_EMB"], env["LAMT_EMB_LADDER"])
        self.setp("a", env["A"], env["A_LADDER"])
        self.setp("lam_R", env["LAM"], env["LAM_LADDER"])
        if env["tuning"] == "joint":
            self.stages = [["lam_R", "lam_T", "a"]]
        else:
            self.stages = [["lam_T", "a"], ["lam_R", "a"]]
            self.init_fixed = {"lam_R": env["lamBase"]}
        self.ZT = torch.empty(ctx.U, ctx.I, dtype=ctx.dt, device=DEV)
        self._zt = None

    def _ZR(self, lam):
        key = ("AddEASEz.ZR", lam)
        if self.ctx.sbuf_owner != key:
            B = ease_from_P(ridge_P(self.ctx.G_copy(), lam))
            self.ctx.score_B(B)
            del B
            self.ctx.sbuf_owner = key
        return self.ctx.Sbuf

    def _ZTm(self, lam_T):
        if self._zt != lam_T:
            Fc = self.ctx.feat("raw", which="text")      # always TEXT (as A^t), independent of --emb-feat
            BT = ease_from_P(ridge_P(Fc @ Fc.t(), lam_T))
            self.ctx.score_B(BT, cast_first=self.ctx.frugal, out=self.ZT)   # same path as the base's text term
            del BT
            self._zt = lam_T
        return self.ZT

    def val_eval(self, hp):
        T, Z, a = self._ZTm(hp["lam_T"]), self._ZR(hp["lam_R"]), float(hp["a"])
        if a == 0.0:
            return mdict(self.ctx.gevV.eval(Z))
        return mdict(self.ctx.gevV.eval_streaming(lambda bu: Z[bu] + a * T[bu]))

    def final_scores(self, hp):
        T, Z, a = self._ZTm(hp["lam_T"]), self._ZR(hp["lam_R"]), float(hp["a"])
        if a != 0.0:
            add_inplace(Z, T, a)
        self.ctx.sbuf_owner = None
        return Z

    def release(self):
        self.ZT = None
        self._zt = None
        torch.cuda.empty_cache()


class GammaModel(Model):
    """G4 fusion weight: S = z(S_set) + gamma z(S_kernel) (paper Eq. 6). The gamma grid is the FIXED Eq. 6 grid of
    scope.train / anchor_v1: no ladder is registered, so the Tuner never extends it (an edge hit is logged by
    fuse_pool only)."""
    name, family, kind = "gamma", "fusion", "fused"

    def __init__(self, ctx, env, Zs, Zk):
        super().__init__(ctx, env)
        self.params, self.stages = ["gamma"], [["gamma"]]
        self.setp("gamma", env["GAMMA"])                         # no ladder: never extended
        self.Zs, self.Zk = Zs, Zk

    def val_eval(self, hp):
        g, Zs, Zk = float(hp["gamma"]), self.Zs, self.Zk
        if g == 0.0:
            return mdict(self.ctx.gevV.eval(Zs))
        return mdict(self.ctx.gevV.eval_streaming(lambda bu: Zs[bu] + g * Zk[bu]))


def make_model(name, ctx, env):
    return {"EASE": lambda: EASEModel(ctx, env), "Base-retuned": lambda: BaseModel(ctx, env),
            "CEASE-emb": lambda: CEASEEmb(ctx, env), "CEASE-tags": lambda: CEASETags(ctx, env),
            "Add-EASE-emb": lambda: AddEASE(ctx, env, "emb"), "Add-EASE-tags": lambda: AddEASE(ctx, env, "tags"),
            "FEASE-tfidf": lambda: FEASEModel(ctx, env, "tfidf"), "FEASE-emb": lambda: FEASEModel(ctx, env, "emb"),
            "FEASE-full-tfidf": lambda: FEASEModel(ctx, env, "full-tfidf"),
            "L3AE": lambda: L3AEModel(ctx, env), "Add-EASE-z": lambda: AddEASEz(ctx, env),
            "FEASE-kNN": lambda: FEASEModel(ctx, env, "knn")}[name]()


# ------------------------------------------------------------------------------------------------ tuner
class Tuner:
    """Grid search on validation Recall@20 (strict '>' in evaluation order). Every configuration, including failed
    ones, is recorded. Coordinate stages sweep the listed parameters with the others fixed at the current best.
    Edge rule: whenever the selected value of a numeric parameter sits at the edge of its explored grid, the next
    value of its ladder is added and the (last) stage sweeping it is re-run; ladder steps beyond the shared grid are
    capped at max_extend per parameter and direction; domain bounds (beta, rho_FEASE, a, alpha, gamma >= 0) are never
    crossed."""

    def __init__(self, model, tag):
        self.m, self.tag = model, tag
        self.initial = model.describe()
        self.records, self.memo, self.edge_log, self.ext_used = [], {}, [], {}
        self.init_added = []
        self.stage_label = "init"

    def _eval(self, hp):
        assert set(hp) == set(self.m.params), (hp, self.m.params)
        k = hp_key(hp)
        if k in self.memo:
            return self.memo[k]
        t0 = time.time()
        rec = {"hp": dict(hp), "stage": self.stage_label}
        try:
            rec["val"] = self.m.val_eval(hp)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rec["error"] = "CUDA OOM: " + str(e)[:200]
        except Exception as e:           # e.g. a failed factorisation: recorded, never hidden
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["t_s"] = round(time.time() - t0, 3)
        self.records.append(rec)
        self.memo[k] = rec
        if "val" in rec:
            v = rec["val"]
            print(f"{self.tag} {hp_str(hp)} | val R@20={v['Recall@20']:.4f} N@20={v['NDCG@20']:.4f} | {rec['t_s']:.2f}s",
                  flush=True)
        else:
            print(f"{self.tag} {hp_str(hp)} | FAILED {rec['error']}", flush=True)
        return rec

    def best(self):
        b = None
        for r in self.records:
            if "val" in r and (b is None or r["val"]["Recall@20"] > b["val"]["Recall@20"]):
                b = r
        return b

    def run_stage(self, si, fixed, label):
        self.stage_label = label
        ps = self.m.stages[si]
        for vals in itertools.product(*[self.m.grids[p] for p in ps]):
            hp = {p: fixed[p] for p in self.m.params if p not in ps}
            hp.update(zip(ps, vals))
            self._eval({p: hp[p] for p in self.m.params})

    def tune(self, max_ext, on_progress=None):
        fixed = dict(self.m.init_fixed)
        # a fixed starting value (e.g. lam_R = EASE's lam*, possibly an extended value such as 25) becomes part of the
        # swept grid, so that the edge rule sees it and can extend past it
        for p, v in fixed.items():
            if p in self.m.ladders and v not in self.m.grids[p]:
                self.m.grids[p] = sorted(set(self.m.grids[p]) | {v})
                self.init_added.append(dict(param=p, value=v))
                print(f"{self.tag} starting value {p}={v} is not on the grid -> added to it", flush=True)
        for si in range(len(self.m.stages)):
            self.run_stage(si, fixed, f"stage{si}")
            b = self.best()
            if b is None:
                raise RuntimeError(f"{self.tag}: every configuration of stage {si} failed")
            fixed = dict(b["hp"])
            if on_progress:
                on_progress()
        while True:
            b = self.best()
            moves = []
            for p, ladder in self.m.ladders.items():
                v, g, sh = b["hp"][p], sorted(self.m.grids[p]), self.m.shared[p]
                for d in ("low", "high"):
                    if v != (g[0] if d == "low" else g[-1]):
                        continue
                    nxt = [x for x in ladder if x < g[0]] if d == "low" else [x for x in ladder if x > g[-1]]
                    if not nxt:
                        continue                                  # domain bound
                    new = nxt[-1] if d == "low" else nxt[0]
                    beyond = new < min(sh) or new > max(sh)
                    if beyond and self.ext_used.get((p, d), 0) >= max_ext:
                        continue
                    moves.append((p, d, v, new, beyond))
            if not moves:
                break
            for p, d, v, new, beyond in moves:
                self.m.grids[p] = sorted(set(self.m.grids[p]) | {new})
                if beyond:
                    self.ext_used[(p, d)] = self.ext_used.get((p, d), 0) + 1
                self.edge_log.append(dict(param=p, edge=d, selected=v, added=new, beyond_shared_grid=beyond))
                print(f"{self.tag} EDGE {p}={v} at the {d} edge of its explored grid -> adding {new}"
                      + (" (beyond the shared grid)" if beyond else ""), flush=True)
            for p, d, v, new, beyond in moves:
                si = max(i for i, st in enumerate(self.m.stages) if p in st)
                self.run_stage(si, dict(self.best()["hp"]), f"ext:{p}={new}")
            if on_progress:
                on_progress()
        return self.best()

    def edge_summary(self):
        b, out = self.best(), {}
        if b is None:
            return out
        for p, ladder in self.m.ladders.items():
            v, g, sh = b["hp"][p], sorted(self.m.grids[p]), self.m.shared[p]
            at = "low" if (v == g[0] and len(g) > 1) else ("high" if (v == g[-1] and len(g) > 1) else None)
            more = ([x for x in ladder if x < g[0]] if at == "low" else [x for x in ladder if x > g[-1]]) if at else []
            out[p] = dict(selected=v, explored=g, shared_grid_range=[min(sh), max(sh)],
                          selected_on_shared_grid_edge=v in (min(sh), max(sh)),
                          at_explored_edge=at, at_domain_bound=bool(at and not more),
                          unresolved_edge=bool(at and more))
        return out

    def describe(self):
        b = self.best()
        ties = [r["hp"] for r in self.records if b and "val" in r and r is not b
                and r["val"]["Recall@20"] == b["val"]["Recall@20"]]
        es = self.edge_summary()
        return dict(model=self.initial, grids_final={k: list(v) for k, v in self.m.grids.items()},
                    n_configs=len(self.records), n_failed=sum("error" in r for r in self.records),
                    selected=(dict(hp=b["hp"], val=b["val"], stage=b["stage"]) if b else None), ties_at_best=ties,
                    grid_edge_hit=bool(self.edge_log) or any(e["selected_on_shared_grid_edge"] for e in es.values()),
                    edge_unresolved=any(e["unresolved_edge"] for e in es.values()),
                    edge_log=self.edge_log, edge_summary=es, init_values_added_to_grid=self.init_added,
                    records=self.records)


# ------------------------------------------------------------------------------------------------ evaluation glue
def test_eval(ctx, S, strict):
    """ONE test evaluation of a materialised score matrix: trusted TopKEvaluator (evalS_trusted, train items masked)
    + GPUEval + per-user vectors; asserts that the per-user means reproduce GPUEval (strict in --smoke)."""
    tr = mdict(evalS_trusted(S, ctx.dset, "test"))
    gm = mdict(ctx.gevT.eval(S))
    r, n = ctx.gevT.per_user(lambda bu: S[bu])
    pm = {"Recall@20": float(r.mean()), "NDCG@20": float(n.mean())}
    d_gpu = max(abs(pm["Recall@20"] - gm["Recall@20"]), abs(pm["NDCG@20"] - gm["NDCG@20"]))
    d_tr = max(abs(pm["Recall@20"] - tr["Recall@20"]), abs(pm["NDCG@20"] - tr["NDCG@20"]))
    if d_gpu > 1e-5:
        msg = f"[{ctx.ds}] per-user means differ from GPUEval by {d_gpu:.2e}"
        print("WARNING: " + msg, flush=True)
        if strict:
            raise AssertionError(msg)
    if d_tr > 1e-3:
        print(f"WARNING: [{ctx.ds}] per-user means differ from the trusted evaluator by {d_tr:.2e}", flush=True)
    rec = dict(trusted=tr, gpueval=gm, peruser_mean=pm, absdiff_peruser_vs_gpueval=d_gpu,
               absdiff_peruser_vs_trusted=d_tr)
    return rec, r.cpu().numpy().astype(np.float64), n.cpu().numpy().astype(np.float64)


def beyond_shared_grid(m, hp):
    """{param: value} for every numeric parameter whose selected value lies outside its shared grid's range."""
    out = {}
    for p, sh in m.shared.items():
        v = hp.get(p)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and (v < min(sh) or v > max(sh)):
            out[p] = v
    return out


def fp64_check(ctx, m, tuner, corner=True):
    """Re-solve the selected (and, with corner=True, the worst-conditioned explored) configuration in fp64 and
    compare. Baby: always, with the corner; other datasets: the selected configuration when it lies beyond the shared
    grid. A failed re-solve (e.g. fp64 OOM) is recorded with flag=True."""
    sel = dict(tuner.best()["hp"])
    cfgs = [("selected", sel)]
    if corner and m.corner:
        c = dict(sel)
        for p, how in m.corner.items():
            g = sorted(m.grids[p])
            c[p] = g[0] if how == "min" else g[-1]
        if c != sel:
            cfgs.append(("worst_conditioned_corner", c))
    out = []
    for label, hp in cfgs:
        try:
            B32 = m.build_B(hp, torch.float32)
            B64 = m.build_B(hp, torch.float64)
            err = float((B32.double() - B64).abs().max() / B64.abs().max().clamp(min=1e-300))
            v32 = float(ctx.gevV.eval(ctx.score_B(B32))["Recall@20"])
            del B32
            B64f = B64.float()
            del B64
            v64 = float(ctx.gevV.eval(ctx.score_B(B64f))["Recall@20"])
            del B64f
            r = dict(label=label, hp=hp, rel_maxabs_err_B=err, val_R20_fp32=v32, val_R20_fp64=v64,
                     abs_diff_val_R20=abs(v32 - v64), flag=abs(v32 - v64) > 1e-4)
            print(f"[{ctx.ds}][{m.name}] fp64 check ({label}) {hp_str(hp)}: rel err B {err:.2e}, val R@20 fp32 "
                  f"{v32:.5f} vs fp64 {v64:.5f}" + ("  <-- FLAG" if r["flag"] else ""), flush=True)
        except Exception as e:
            r = dict(label=label, hp=hp, error=f"{type(e).__name__}: {str(e)[:300]}", flag=True)
            print(f"[{ctx.ds}][{m.name}] fp64 check ({label}) FAILED: {r['error']}  <-- FLAG", flush=True)
        out.append(r)
        torch.cuda.empty_cache()
    return out


def deployed_v1(ds, seeds, args):
    """Deployed SCOPE-v1 records written by scope.train (test base / fused R@20 and the selected gamma)."""
    d = Path(args.scope_json_dir) if args.scope_json_dir else ROOT / "results" / "scope"
    out = {}
    for s in seeds:
        p = d / (f"scope_{ds}_d256_le1.0_lz1.0_lr0.003" + ("" if s == 2024 else f"_s{s}") + ".json")
        if p.is_file():
            j = json.loads(p.read_text())
            out[s] = dict(path=rel(p), gamma=j.get("gamma"), fused_R20=float(j["fused"]["Recall@20"]),
                          fused_N20=float(j["fused"]["NDCG@20"]), base_R20=float(j["base"]["Recall@20"]))
    return out


def head_ckpt(args, ds, seed):
    """Explicit checkpoint path per seed (never 'latest file' discovery)."""
    d = Path(args.ckpt_dir) if args.ckpt_dir else ROOT / "ckpts" / "scope"
    return d / (f"scope_{ds}_d256_le1.0_lz1.0_lr0.003" + ("" if seed == 2024 else f"_s{seed}") + ".pt")


def load_state(p):
    try:
        return torch.load(p, map_location=DEV, weights_only=True)
    except TypeError:                                   # torch < 1.13
        return torch.load(p, map_location=DEV)


def head_scores(ctx, ck):
    """z(S_set) of a frozen SCOPE head: cosine logits F.normalize(z, dim=1) @ F.normalize(E, dim=1).T / tau
    (SCOPE.score_all; fp16 for > 50k users), per-user z-scored in fp32, cast to the dataset score dtype."""
    head = SCOPE(ctx.I, 256).to(DEV)
    head.load_state_dict(load_state(ck))                # strict: the pruned architecture must match
    head.eval()
    with torch.no_grad():
        S = head.score_all(ctx.R, ctx.degf)
    del head
    zscore_inplace(S)
    if S.dtype != ctx.dt:
        S = S.to(ctx.dt)
    return S


def narrow_selection(records):
    """Re-derive the published base selection from the shared-grid records: scope.closed_form_base's narrow grid,
    same iteration order (lam outer, a inner) and strict '>'. None if a narrow-grid point was not evaluated."""
    idx = {hp_key(r["hp"]): r for r in records if "val" in r}
    best = None
    for lam in NARROW["lam"]:
        for a in NARROW["a"]:
            r = idx.get(hp_key({"lam": lam, "a": a}))
            if r is None:
                return None
            if best is None or r["val"]["Recall@20"] > best["val"]["Recall@20"]:
                best = r
    return dict(hp=best["hp"], val=best["val"])


# ------------------------------------------------------------------------------------------------ G3
def run_one(ctx, args, env, name, res, jpath, peruser):
    ds, tag, t0 = ctx.ds, f"[{ctx.ds}][{name}]", time.time()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m = make_model(name, ctx, env)
    tuner = Tuner(m, tag)

    def snapshot(status):
        e = dict(tuner.describe(), label=row_label(name), status=status, meta=m.meta,
                 wall_s=round(time.time() - t0, 2), peak_GB=peak_gb())
        res["models"][name] = e
        dump_json(jpath, res)
        return e

    try:
        snapshot("tuning")
        tuner.tune(max_extend(args), on_progress=lambda: snapshot("tuning"))
        sel = tuner.best()
        S = m.final_scores(sel["hp"])
        test, rR, rN = test_eval(ctx, S, strict=args.smoke)
        peruser[name] = (rR, rN)
        e = snapshot("tested")
        e["test"] = test
        print(f"{tag} SELECTED {hp_str(sel['hp'])} val R@20={sel['val']['Recall@20']:.4f} -> TEST (once) "
              f"R@20={test['trusted']['Recall@20']:.4f} N@20={test['trusted']['NDCG@20']:.4f} | configs "
              f"{e['n_configs']} | edge hit {e['grid_edge_hit']} unresolved {e['edge_unresolved']}", flush=True)
        if name == "EASE":
            env["lamE"] = sel["hp"]["lam"]
            B = m.build_B({"lam": env["lamE"]})
            env["BEASE_fro"] = float(torch.linalg.norm(B))      # diag(B) = 0 -> off-diagonal Frobenius norm
            del B
            res["env_values"].update(lamE=env["lamE"], BEASE_fro=env["BEASE_fro"])
        if name == "Base-retuned":
            env["lamBase"] = sel["hp"]["lam"]
            res["env_values"]["lamBase"] = env["lamBase"]
            published_base(ctx, args, m, tuner, res, jpath, peruser)
        bey = beyond_shared_grid(m, sel["hp"])
        e["selected_beyond_shared_grid"] = bey
        if not args.no_fp64:
            if m.kind == "single" and (ds == "baby" or bey):
                e["fp64_check"] = fp64_check(ctx, m, tuner, corner=(ds == "baby"))
                e["fp64_flag"] = any(r.get("flag") for r in e["fp64_check"])
            elif bey:            # fused rows (Base-retuned, Add-EASE-z): no fp64 re-solve implemented -> flag only
                e["fp64_flag"] = True
                e["fp64_note"] = ("selected configuration lies beyond the shared grid; fused model, no fp64 re-solve "
                                  "implemented -> check its conditioning by hand before quoting it")
            if e.get("fp64_flag"):
                print(f"WARNING: {tag} fp64 FLAG -- see fp64_check/fp64_note (selected values beyond the shared grid: "
                      f"{bey if bey else 'none'})", flush=True)
        e["status"] = "done"
        e["meta"] = m.meta
        e["wall_s"] = round(time.time() - t0, 2)
        e["peak_GB"] = peak_gb()
        res["models"][name] = e
        dump_json(jpath, res)
        if e["peak_GB"] > 12.0:
            print(f"WARNING: {tag} peak GPU memory {e['peak_GB']} GB exceeds the 12 GB target", flush=True)
    except Exception as ex:
        traceback.print_exc()
        e = snapshot("failed")
        e["error"] = f"{type(ex).__name__}: {str(ex)[:500]}"
        res["models"][name] = e
        dump_json(jpath, res)
        if name in ("EASE", "Base-retuned"):
            raise
    finally:
        m.release()
        ctx.free_side()
        del m
        torch.cuda.empty_cache()


def published_base(ctx, args, m, tuner, res, jpath, peruser):
    ds = ctx.ds
    pub = dict(PUBLISHED_BASE[ds])
    narrow = narrow_selection(tuner.records)
    rec_pub = next((r for r in tuner.records if "val" in r and hp_key(r["hp"]) == hp_key(pub)), None)
    S = m.final_scores(pub)
    test, rR, rN = test_eval(ctx, S, strict=args.smoke)
    peruser["Base-published"] = (rR, rN)
    dep = deployed_v1(ds, [2024], args)
    ref = dep[2024]["base_R20"] if 2024 in dep else None
    diff = abs(test["trusted"]["Recall@20"] - ref) if ref is not None else None
    res["models"]["Base-published"] = dict(
        name="Base-published", family="ours", literature=False, kind="fused", status="done", hp=pub,
        source=PUBLISHED_SOURCE, val=rec_pub["val"] if rec_pub else None, narrow_grid_selection=narrow,
        narrow_selection_matches_published=(None if narrow is None else hp_key(narrow["hp"]) == hp_key(pub)),
        test=test, deployed_json=dep.get(2024, {}).get("path"), deployed_base_test_R20=ref,
        abs_diff_vs_deployed_R20=diff, repro_ok=(diff is not None and diff < 2e-4))
    dump_json(jpath, res)
    print(f"[{ds}][Base-published] {hp_str(pub)} TEST R@20={test['trusted']['Recall@20']:.4f} "
          f"N@20={test['trusted']['NDCG@20']:.4f} | deployed base R@20={ref} | |diff|={diff} | narrow-grid "
          f"re-selection={None if narrow is None else narrow['hp']}", flush=True)
    if diff is not None and diff >= 2e-4:
        print(f"WARNING: [{ds}] published base NOT reproduced within 2e-4 -- check before using any G3 row", flush=True)


def run_g3(ctx, args, env0, out_dir, run_id):
    ds, seed, t0 = ctx.ds, ctx.seed, time.time()
    env = dict(env0)
    stem = f"g3_closedform_{ds}_s{seed}_{args.tuning}{'_smoke' if args.smoke else ''}_{run_id}"
    jpath = fresh_path(out_dir, stem, ".json")
    npath = fresh_path(out_dir, stem + "_peruser", ".npz")
    res = dict(script=HERE.name, stage="g3", dataset=ds, seed=seed, run_id=run_id,
               experiment="G3: CEASE/Add-EASE/FEASE/L3AE + our base re-tuned on one shared validation grid, "
                          "controlled swaps Add-EASE-z / FEASE-kNN",
               seeds_note="seed_all(seed) before RecDataset/build_lists; all closed forms are deterministic",
               argv=sys.argv, args=vars(args), environment=env_info(), data=ctx.describe(),
               grids={k: v for k, v in env.items()},
               selection=("validation Recall@20 (GPUEval, train items masked), strict '>' in evaluation order; ONE "
                          "trusted test evaluation (evalS_trusted) of the selected configuration per model"),
               tuning_protocol=args.tuning, max_extend=max_extend(args), rules=dict(G3_1=RULE_G3_1, G3_2=RULE_G3_2),
               models={}, env_values={}, json_path=rel(jpath), peruser_npz=rel(npath))
    dump_json(jpath, res)
    peruser = {}
    for name in ("EASE", "Base-retuned") + LIT_ORDER + SWAPS:
        if args.models and name not in args.models and name not in ("EASE", "Base-retuned"):
            continue
        if name in NEEDS_SIDE and ctx.side_meta is None:
            res["models"][name] = dict(name=name, status="n/a",
                                       skipped="no item metadata on this dataset: dense frozen features only")
            dump_json(jpath, res)
            continue
        run_one(ctx, args, env, name, res, jpath, peruser)
    res["side_stats"] = ctx.side_stats
    pub = peruser.get("Base-published")
    if pub is not None:
        diffs = {}
        for name, (r, n) in peruser.items():
            if name != "Base-published":
                diffs[f"{name}|R@20"] = r - pub[0]
                diffs[f"{name}|N@20"] = n - pub[1]
        bt = paired_bootstrap_multi(diffs, B=boot_B(args), seed=args.boot_seed)
        for k, b in bt.items():
            ref = float(pub[0].mean() if k.endswith("R@20") else pub[1].mean())
            b["rel_lift"] = b["mean_delta"] / ref
            b["rel_ci95"] = [c / ref for c in b["ci95"]]
        res["bootstrap_vs_base_published"] = bt
        res["bootstrap_note"] = ("model minus Base-published, per-user test metrics, paired bootstrap (shared "
                                 "resamples, numpy default_rng(boot_seed)); raw two-sided p here, Holm in the summary")
    arrays = {"users": ctx.gevT.users.cpu().numpy()}
    for name, (r, n) in peruser.items():
        arrays[f"{name}__R20"] = r.astype(np.float32)
        arrays[f"{name}__N20"] = n.astype(np.float32)
    np.savez_compressed(npath, **arrays)
    res["wall_s"] = round(time.time() - t0, 2)
    dump_json(jpath, res)
    exp = expected_models(ds)
    res["expected_models"] = exp
    res["missing_models"] = [n for n in exp if not (isinstance(res["models"].get(n), dict)
                                                    and res["models"][n].get("status") == "done"
                                                    and "test" in res["models"][n])]
    dump_json(jpath, res)
    print(f"\n[{ds}] G3 table (test once per model; val-selected):", flush=True)
    for name, e in res["models"].items():
        if isinstance(e, dict) and "test" in e:
            t = e["test"]["trusted"]
            hp = e.get("hp") or (e.get("selected") or {}).get("hp")
            print(f"  {row_label(name):24s} R@20={t['Recall@20']:.4f} N@20={t['NDCG@20']:.4f}  {hp}", flush=True)
    if res["missing_models"]:
        print(f"WARNING: [{ds}] G3 INCOMPLETE -- missing/failed rows: {res['missing_models']} (no verdict will be "
              f"given from this dataset)", flush=True)
    print(f"[{ds}] G3 done in {res['wall_s']}s -> {jpath}", flush=True)
    res["_path"] = str(jpath)
    return res


# ------------------------------------------------------------------------------------------------ G4
def fuse_pool(ctx, args, env, entry, name, pool, seeds, arrays, res, jpath):
    ds, t0 = ctx.ds, time.time()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    hp = entry["selected"]["hp"]
    m = make_model(name, ctx, env)
    Zk = m.final_scores(hp)                      # = ctx.Sbuf, per-user z-scored for single-matrix models
    # Base-retuned is fused exactly as the deployed SCOPE-v1 fuses the published base (anchor_v1 / scope.train): the
    # view z(R B) + a z(R A^t) is NOT z-scored a second time, so 'head + Base-retuned' is the pipeline a base switch
    # would use. Add-EASE-z (a swap, never a base candidate) keeps Eq. 6's whole-view z-score.
    kkind = m.kind
    rez = kkind == "fused" and name != "Base-retuned"
    if rez:
        zscore_inplace(Zk)                       # z(S_kernel) of a fused kernel (Eq. 6 z-scores the whole view)
    m.release()
    ctx.free_side()
    del m
    torch.cuda.empty_cache()
    ctx.sbuf_owner = ("G4.kernel", name)
    kt, kR, kN = test_eval(ctx, Zk, strict=args.smoke)
    g3R = float(entry["test"]["trusted"]["Recall@20"])
    drift = abs(kt["trusted"]["Recall@20"] - g3R)
    if drift > 1e-4:
        print(f"WARNING: [{ds}] G4 kernel {name} rebuilt with test R@20 {kt['trusted']['Recall@20']:.5f} vs G3 "
              f"{g3R:.5f}", flush=True)
    arrays[f"{pool}__kernel__R20"] = kR.astype(np.float32)
    arrays[f"{pool}__kernel__N20"] = kN.astype(np.float32)
    out = dict(kernel=name, kernel_label=row_label(name),
               kernel_family=entry.get("model", {}).get("family", entry.get("family")), kernel_hp=hp,
               kernel_val=entry["selected"]["val"], kernel_test=kt, kernel_test_R20_in_g3=g3R,
               kernel_rebuild_abs_diff_R20=drift, kernel_view_rezscored=rez,
               kernel_view_note=("single-matrix kernel: z(R B)" if kkind == "single" else
                                 ("fused kernel re-z-scored as a whole (Eq. 6)" if rez else
                                  "Base-retuned: NOT re-z-scored, identical to SCOPE-v1's fusion of the published base")),
               gamma_grid=list(env["GAMMA"]), gamma_grid_rule="fixed Eq. 6 grid, never extended; edge hits logged",
               seeds={})
    res["pools"][pool] = out
    fused = {}
    for seed in seeds:
        seed_all(seed)
        ck = head_ckpt(args, ds, seed)
        if not ck.is_file():
            out["seeds"][str(seed)] = dict(error=f"checkpoint missing: {rel(ck)}")
            print(f"WARNING: [{ds}] missing head checkpoint {ck}", flush=True)
            dump_json(jpath, res)
            continue
        Zs = head_scores(ctx, ck)
        gm = GammaModel(ctx, env, Zs, Zk)
        tuner = Tuner(gm, f"[{ds}][G4 {pool}:{name}][seed {seed}]")
        tuner.tune(0)                            # gamma has no ladder: the fixed Eq. 6 grid, gamma=0 first, strict '>'
        sel = tuner.best()
        g = float(sel["hp"]["gamma"])
        ggrid = list(env["GAMMA"])
        gamma_edge = dict(grid=ggrid, selected=g, at_upper_edge=(g == max(ggrid)), extended=False)
        if gamma_edge["at_upper_edge"]:
            print(f"[{ds}][G4 {pool}:{name}][seed {seed}] EDGE gamma={g} at the upper edge of the fixed Eq. 6 grid "
                  f"(not extended, as scope.train; logged only)", flush=True)
        add_inplace(Zs, Zk, g)                   # Zs <- z(S_set) + g z(S_kernel), same dt arithmetic as validation
        ft, fR, fN = test_eval(ctx, Zs, strict=args.smoke)
        del Zs, gm
        torch.cuda.empty_cache()
        fused[seed] = (fR, fN)
        arrays[f"{pool}__fused_s{seed}__R20"] = fR.astype(np.float32)
        arrays[f"{pool}__fused_s{seed}__N20"] = fN.astype(np.float32)
        out["seeds"][str(seed)] = dict(ckpt=rel(ck), ckpt_sha256=sha256_file(ck), **tuner.describe(), test=ft,
                                       gamma_edge=gamma_edge)
        print(f"[{ds}][G4 {pool}] seed {seed}: gamma*={g} -> TEST R@20={ft['trusted']['Recall@20']:.4f} "
              f"N@20={ft['trusted']['NDCG@20']:.4f} vs kernel {kt['trusted']['Recall@20']:.4f}/"
              f"{kt['trusted']['NDCG@20']:.4f}", flush=True)
        dump_json(jpath, res)
    diffs = {}
    for s, (fR, fN) in fused.items():
        diffs[f"s{s}|R@20"] = fR - kR
        diffs[f"s{s}|N@20"] = fN - kN
    if fused:
        diffs["seedmean|R@20"] = np.mean([v[0] for v in fused.values()], 0) - kR
        diffs["seedmean|N@20"] = np.mean([v[1] for v in fused.values()], 0) - kN
    bt = paired_bootstrap_multi(diffs, B=boot_B(args), seed=args.boot_seed)
    for k, b in bt.items():
        ref = float(kR.mean() if k.endswith("R@20") else kN.mean())
        b["rel_lift"] = b["mean_delta"] / ref
        b["rel_ci95"] = [c / ref for c in b["ci95"]]
    out["bootstrap"] = bt
    out["bootstrap_note"] = "fused minus S_kernel alone, per-user test metrics; raw two-sided p, Holm in the summary"
    out["wall_s"] = round(time.time() - t0, 2)
    out["peak_GB"] = peak_gb()
    dump_json(jpath, res)
    return out


def anchor_v1(ctx, args, env, seeds):
    """Re-derive the DEPLOYED SCOPE-v1 (published base, not re-z-scored, Eq. 6 grid, gamma=0 first, strict '>',
    no extension -- exactly scope.train) and compare with results/scope/scope_<ds>_*.json."""
    ds, t0 = ctx.ds, time.time()
    dep = deployed_v1(ds, seeds, args)
    if not dep:
        return dict(skipped="deployed scope_<ds>_*.json not found (see --scope-json-dir)")
    base = BaseModel(ctx, env)
    Sb = base.final_scores(dict(PUBLISHED_BASE[ds]))
    base.release()
    del base
    ctx.sbuf_owner = ("G4.anchor",)
    torch.cuda.empty_cache()
    out = dict(note="reproduction anchor for the head checkpoints and the fusion pipeline (not a new result)",
               base_hp=PUBLISHED_BASE[ds], gamma_grid=GAMMA_EQ6, seeds={})
    for seed in seeds:
        if seed not in dep:
            continue
        ck = head_ckpt(args, ds, seed)
        if not ck.is_file():
            out["seeds"][str(seed)] = dict(error=f"checkpoint missing: {rel(ck)}")
            continue
        Zs = head_scores(ctx, ck)
        recs, best = [], None
        for g in GAMMA_EQ6:
            if g == 0.0:
                v = float(ctx.gevV.eval(Zs)["Recall@20"])
            else:
                v = float(ctx.gevV.eval_streaming(lambda bu: Zs[bu] + g * Sb[bu])["Recall@20"])
            recs.append(dict(gamma=g, val_R20=v))
            if best is None or v > best[1]:
                best = (g, v)
        g = best[0]
        add_inplace(Zs, Sb, g)
        t = mdict(evalS_trusted(Zs, ctx.dset, "test"))
        del Zs
        torch.cuda.empty_cache()
        d = abs(t["Recall@20"] - dep[seed]["fused_R20"])
        ok = d < 1e-3 and dep[seed]["gamma"] is not None and abs(g - float(dep[seed]["gamma"])) < 1e-9
        out["seeds"][str(seed)] = dict(records=recs, gamma_selected=g, gamma_deployed=dep[seed]["gamma"], test=t,
                                       deployed_fused_R20=dep[seed]["fused_R20"], abs_diff_R20=d,
                                       reproduces_deployed=ok, deployed_json=dep[seed]["path"])
        print(f"[{ds}][G4 anchor] seed {seed}: gamma {g} (deployed {dep[seed]['gamma']}) SCOPE-v1 TEST R@20="
              f"{t['Recall@20']:.4f} vs deployed {dep[seed]['fused_R20']:.4f} -> "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)
    out["wall_s"] = round(time.time() - t0, 2)
    return out


def run_g4(ctx, args, env0, g3, out_dir, run_id):
    ds, seeds, t0 = ctx.ds, list(args.seeds), time.time()
    env = dict(env0)
    env.update(g3.get("env_values") or {})
    for k in ("lamE", "BEASE_fro", "lamBase"):
        if k not in env:
            raise RuntimeError(f"[{ds}] G3 result lacks env_values.{k}; cannot rebuild the kernels")
    M = g3["models"]

    def ok(n):
        return isinstance(M.get(n), dict) and M[n].get("selected") and "test" in M[n]

    lit = [n for n in LIT_ORDER if ok(n)]
    allc = lit + [n for n in ("Base-retuned",) + SWAPS if ok(n)]
    exp_lit = [n for n in LIT_ORDER if n in expected_models(ds)]
    cand_missing = dict(literature=[n for n in exp_lit if not ok(n)])
    cand_missing["all"] = cand_missing["literature"] + [n for n in ("Base-retuned",) + SWAPS if not ok(n)]
    if cand_missing["all"]:
        print(f"WARNING: [{ds}] G4 kernel pool INCOMPLETE -- G3 rows missing/failed: {cand_missing['all']} (the "
              f"kernel choice may be wrong; the G4 verdict for this dataset will be PROVISIONAL)", flush=True)
    stem = f"g4_head_on_kernel_{ds}_s{'-'.join(map(str, seeds))}{'_smoke' if args.smoke else ''}_{run_id}"
    jpath = fresh_path(out_dir, stem, ".json")
    npath = fresh_path(out_dir, stem + "_peruser", ".npz")
    res = dict(script=HERE.name, stage="g4", dataset=ds, seeds=seeds, run_id=run_id,
               experiment="G4: frozen SCOPE set head fused onto the validation-best content closed form",
               fusion=("S = z(S_set) + gamma z(S_kernel) (paper Eq. 6); gamma selected on validation Recall@20 on the "
                       "FIXED Eq. 6 grid (gamma=0 first, strict '>', never extended -- as scope.train); test once per "
                       "seed; Base-retuned is not re-z-scored (as SCOPE-v1's published base)"),
               gamma_grid=env["GAMMA"], argv=sys.argv, args=vars(args), environment=env_info(), data=ctx.describe(),
               g3_json=g3.get("json_path") or g3.get("_path"), rule=RULE_G4,
               pool_definition=dict(literature=list(LIT_ORDER), all=list(LIT_ORDER) + ["Base-retuned"] + list(SWAPS)),
               g3_candidates_missing=cand_missing,
               pools={}, json_path=rel(jpath), peruser_npz=rel(npath))
    dump_json(jpath, res)
    arrays = {"users": ctx.gevT.users.cpu().numpy()}
    done = {}
    for pool in args.g4_pools:
        names = lit if pool == "literature" else allc
        if not names:
            res["pools"][pool] = dict(skipped="no candidate with a selected and tested configuration in the G3 JSON")
            dump_json(jpath, res)
            continue
        ranking = sorted(((n, M[n]["selected"]["val"]["Recall@20"]) for n in names), key=lambda x: -x[1])
        best = max(names, key=lambda n: M[n]["selected"]["val"]["Recall@20"])     # first in order on ties
        print(f"[{ds}][G4 {pool}] validation ranking: " + ", ".join(f"{n} {v:.4f}" for n, v in ranking), flush=True)
        if best in done:
            res["pools"][pool] = dict(kernel=best, same_as_pool=done[best], val_ranking=ranking)
            dump_json(jpath, res)
            continue
        done[best] = pool
        P = fuse_pool(ctx, args, env, M[best], best, pool, seeds, arrays, res, jpath)
        P["val_ranking"] = ranking
        res["pools"][pool] = P
        dump_json(jpath, res)
    if not args.no_anchor:
        res["anchor_scope_v1"] = anchor_v1(ctx, args, env, seeds)
    np.savez_compressed(npath, **arrays)
    res["wall_s"] = round(time.time() - t0, 2)
    dump_json(jpath, res)
    print(f"[{ds}] G4 done in {res['wall_s']}s -> {jpath}", flush=True)
    res["_path"] = str(jpath)
    return res


# ------------------------------------------------------------------------------------------------ summaries
def _tested(e):
    return isinstance(e, dict) and e.get("status") == "done" and "test" in e


def _completeness_g3(results):
    """Present vs expected (pre-registered) datasets and rows ."""
    present = [r["dataset"] for r in results]
    missing_ds = [d for d in EXPECTED_DATASETS if d not in present]
    missing_models = {}
    for r in results:
        miss = [n for n in expected_models(r["dataset"]) if not _tested(r["models"].get(n))]
        if miss:
            missing_models[r["dataset"]] = miss
    complete = not missing_ds and not missing_models
    return dict(complete=complete, status="complete" if complete else "INCOMPLETE/PROVISIONAL",
                expected_datasets=list(EXPECTED_DATASETS), present_datasets=present, missing_datasets=missing_ds,
                missing_models=missing_models)


def summarize_g3(results, args):
    comp = _completeness_g3(results)
    # G3 family (information only, not part of any rule): per dataset, every row minus Base-published on R@20 and
    # N@20 (EASE, literature models, swaps), Holm within the dataset. Base-retuned has its own G3-2 family below.
    fam, adj = {}, {}
    for res in results:
        f = {}
        for key, b in (res.get("bootstrap_vs_base_published") or {}).items():
            model, metric = key.split("|")
            if model == "Base-retuned":
                continue
            f[f"{res['dataset']}|{model}|{metric}"] = b["p_two_sided"]
        fam[res["dataset"]] = f
        adj.update(holm(f))
    # G3-2 family: Base-retuned minus Base-published, 4 datasets x {R@20, N@20}, Holm within this family 
    d6fam = {}
    for res in results:
        bt = res.get("bootstrap_vs_base_published") or {}
        for metric in METRICS:
            b = bt.get(f"Base-retuned|{metric}")
            if b is not None:
                d6fam[f"{res['dataset']}|{metric}"] = b["p_two_sided"]
    d6adj = holm(d6fam)
    d6_expected = [f"{d}|{m}" for d in EXPECTED_DATASETS for m in METRICS]
    d6_missing = [k for k in d6_expected if k not in d6fam]

    rows, rule1, rule2 = {}, {}, {}
    print("\n" + "=" * 110 + f"\nG3 SUMMARY -- {comp['status']}\n" + "=" * 110, flush=True)
    if not comp["complete"]:
        print(f"G3 INCOMPLETE/PROVISIONAL: missing datasets {comp['missing_datasets']}, missing/failed rows "
              f"{comp['missing_models']} -> NO VERDICT is given for G3-1 or G3-2 (observations only)", flush=True)
    for res in results:
        ds, M = res["dataset"], res["models"]
        rows[ds] = {}
        for n, e in M.items():
            if _tested(e):
                hp = e.get("hp") or (e.get("selected") or {}).get("hp")
                val = e.get("val") if "hp" in e else (e.get("selected") or {}).get("val")
                rows[ds][n] = dict(label=row_label(n), hp=hp, val_R20=(val or {}).get("Recall@20"),
                                   test_R20=e["test"]["trusted"]["Recall@20"], test_N20=e["test"]["trusted"]["NDCG@20"],
                                   test_R10=e["test"]["trusted"]["Recall@10"], test_N10=e["test"]["trusted"]["NDCG@10"],
                                   grid_edge_hit=e.get("grid_edge_hit"), edge_unresolved=e.get("edge_unresolved"),
                                   selected_beyond_shared_grid=e.get("selected_beyond_shared_grid"),
                                   fp64_flag=e.get("fp64_flag"),
                                   p_holm_R20_vs_base_published=adj.get(f"{ds}|{n}|R@20"),
                                   p_holm_N20_vs_base_published=adj.get(f"{ds}|{n}|N@20"))
        ds_complete = ds not in comp["missing_models"]
        # ---- rule G3-1 : literature rows AND the re-tuned EASE, R@20 OR N@20, '>=' after rounding
        refs = {"Base-published": M.get("Base-published"), "Base-retuned": M.get("Base-retuned")}
        if _tested(refs["Base-published"]):
            cmp = {}
            for refname, ref in refs.items():
                if not _tested(ref):
                    cmp[refname] = None
                    continue
                rt_ = ref["test"]["trusted"]
                lst = []
                for n in ("EASE",) + LIT_ORDER:
                    if n not in rows[ds]:
                        continue
                    r = rows[ds][n]
                    geR = r4(r["test_R20"]) >= r4(rt_["Recall@20"])
                    geN = r4(r["test_N20"]) >= r4(rt_["NDCG@20"])
                    if geR or geN:
                        lst.append(dict(model=n, label=row_label(n), test_R20=r["test_R20"], test_N20=r["test_N20"],
                                        test_R20_4dp=float(r4(r["test_R20"])), test_N20_4dp=float(r4(r["test_N20"])),
                                        ge_R20=geR, ge_N20=geN, delta_R20=r["test_R20"] - rt_["Recall@20"],
                                        delta_N20=r["test_N20"] - rt_["NDCG@20"],
                                        p_holm_R20=(adj.get(f"{ds}|{n}|R@20") if refname == "Base-published" else None),
                                        p_holm_N20=(adj.get(f"{ds}|{n}|N@20") if refname == "Base-published" else None)))
                cmp[refname] = dict(ref_test_R20=rt_["Recall@20"], ref_test_N20=rt_["NDCG@20"],
                                    ref_test_R20_4dp=float(r4(rt_["Recall@20"])),
                                    ref_test_N20_4dp=float(r4(rt_["NDCG@20"])), at_or_above=lst)
            trig = bool(cmp["Base-published"]["at_or_above"])
            rule1[ds] = dict(dataset_complete=ds_complete, vs_base_published=cmp["Base-published"],
                             vs_base_retuned_information=cmp["Base-retuned"], triggered_observed=trig,
                             base_is_one_member_of_family=(trig if comp["complete"] else None))
        # ---- rule G3-2: own family, |delta| <= 1 SD on BOTH metrics
        rt, pb = M.get("Base-retuned"), M.get("Base-published")
        if _tested(rt) and _tested(pb):
            dR = rt["test"]["trusted"]["Recall@20"] - pb["test"]["trusted"]["Recall@20"]
            dN = rt["test"]["trusted"]["NDCG@20"] - pb["test"]["trusted"]["NDCG@20"]
            dep = deployed_v1(ds, list(EXPECTED_SEEDS), args)
            seeds_ok = all(s in dep for s in EXPECTED_SEEDS)
            sdR = float(np.std([dep[s]["fused_R20"] for s in EXPECTED_SEEDS], ddof=1)) if seeds_ok else None
            sdN = float(np.std([dep[s]["fused_N20"] for s in EXPECTED_SEEDS], ddof=1)) if seeds_ok else None
            changed = hp_key(rt["selected"]["hp"]) != hp_key(pb["hp"])
            within = (abs(dR) <= sdR and abs(dN) <= sdN) if seeds_ok else None
            if not seeds_ok:
                outcome = None
                verdict = "PROVISIONAL: deployed SCOPE-v1 JSONs of seeds 2024/2025/2026 not all found -> no base-switch verdict"
            elif not changed:
                outcome = "unchanged"
                verdict = "selected base unchanged -> keep the published base (no base-switch decision needed)"
            elif within:
                outcome = "changed_within_seed_noise"
                verdict = ("re-tuned base differs from the published one but WITHIN seed noise on both metrics -> the rule "
                           "allows keeping the published base (re-tuned one as a robustness row) or adopting the "
                           "re-tuned base everywhere; adopting a new base is a modelling decision outside this script")
            else:
                outcome = "changed_beyond_seed_noise"
                verdict = ("re-tuned base differs BEYOND seed noise (|delta| > 1 SD on R@20 and/or N@20) -> the rule allows "
                           "only adopting the re-tuned base everywhere (re-run the dependent experiments), not keeping the "
                           "published base with a robustness row; adopting a new base is a modelling decision outside this script")
            if not comp["complete"] or d6_missing:
                verdict = "PROVISIONAL (inputs incomplete, no verdict); observation: " + verdict
            rule2[ds] = dict(published_hp=pb["hp"], retuned_hp=rt["selected"]["hp"], changed=changed,
                             delta_test_R20=dR, delta_test_N20=dN,
                             deployed_v1_test_R20_by_seed={str(k): v["fused_R20"] for k, v in dep.items()},
                             deployed_v1_test_N20_by_seed={str(k): v["fused_N20"] for k, v in dep.items()},
                             seed_sd_R20=sdR, seed_sd_N20=sdN, threshold_k_sd=1.0,
                             within_seed_noise_observed=within,
                             p_raw_R20=d6fam.get(f"{ds}|R@20"), p_raw_N20=d6fam.get(f"{ds}|N@20"),
                             p_holm_R20=d6adj.get(f"{ds}|R@20"), p_holm_N20=d6adj.get(f"{ds}|N@20"),
                             outcome=(outcome if (comp["complete"] and not d6_missing) else None), verdict=verdict)
    for ds, rr in rows.items():
        print(f"\n[{ds}] model                     val R@20  test R@20  test N@20  p_holm R/N vs base-pub  "
              f"edge-hit/unresolved  beyond-shared", flush=True)
        for n, r in rr.items():
            pr, pn = r["p_holm_R20_vs_base_published"], r["p_holm_N20_vs_base_published"]
            ps = "/".join(("%.2g" % x) if x is not None else "--" for x in (pr, pn))
            print(f"  {r['label']:24s} {(r['val_R20'] or float('nan')):.4f}    {r['test_R20']:.4f}     "
                  f"{r['test_N20']:.4f}     {ps:>18s}     {r['grid_edge_hit']}/{r['edge_unresolved']}     "
                  f"{r['selected_beyond_shared_grid'] or ''}", flush=True)
    print("\nNARROWING RULE G3-1: " + RULE_G3_1, flush=True)
    for ds, r in rule1.items():
        vp = r["vs_base_published"]
        ab = vp["at_or_above"]
        print(f"  [{ds}] Base-published test R@20 {vp['ref_test_R20_4dp']:.4f} N@20 {vp['ref_test_N20_4dp']:.4f}; "
              f"EASE/literature rows at or above it (4 d.p.): "
              + (", ".join(f"{a['label']} {a['test_R20_4dp']:.4f}/{a['test_N20_4dp']:.4f} (R>= {a['ge_R20']}, "
                           f"N>= {a['ge_N20']}, p_holm {a['p_holm_R20']}/{a['p_holm_N20']})" for a in ab)
                 if ab else "none"), flush=True)
        vr = r["vs_base_retuned_information"]
        if vr is not None:
            print(f"  [{ds}]   (information) vs Base-retuned {vr['ref_test_R20_4dp']:.4f}/{vr['ref_test_N20_4dp']:.4f}: "
                  + (", ".join(a["label"] for a in vr["at_or_above"]) if vr["at_or_above"] else "none"), flush=True)
    fams = [ds for ds, r in rule1.items() if r["triggered_observed"]]
    if not comp["complete"]:
        print(f"  => G3-1 INCOMPLETE/PROVISIONAL: no verdict (observed triggers on {fams if fams else 'none'})",
              flush=True)
        ml_verdict = None
    else:
        if "microlens" in rule1:
            ml = rule1["microlens"]["triggered_observed"]
            ml_verdict = "reached_by_a_closed_form" if ml else "above_all_closed_forms"
            print("  => MicroLens, base alone against every closed form: "
                  + ("a closed form reaches the base on R@20 or N@20 (the base is one member of the family)" if ml else
                     "the base stays above EASE and the content closed forms"), flush=True)
        else:
            ml_verdict = None
            print("  => MicroLens not in these G3 results: claim not evaluated here", flush=True)
        print(f"  => describe the base as one member of the content-closed-form family on: "
              f"{fams if fams else 'none'}", flush=True)
    print("\nNARROWING RULE G3-2 (base switch): " + RULE_G3_2, flush=True)
    if d6_missing:
        print(f"  base-switch family INCOMPLETE ({len(d6fam)}/{len(d6_expected)} tests; missing {d6_missing}) -> PROVISIONAL",
              flush=True)
    for ds, r in rule2.items():
        print(f"  [{ds}] published {r['published_hp']} -> re-tuned {r['retuned_hp']}; delta test R@20 "
              f"{r['delta_test_R20']:+.4f} (1 SD {r['seed_sd_R20']}), N@20 {r['delta_test_N20']:+.4f} (1 SD "
              f"{r['seed_sd_N20']}); p_holm(base-switch family) R {r['p_holm_R20']} N {r['p_holm_N20']} => {r['verdict']}",
              flush=True)
    print("  NOTE: whatever the outcome, adopting a new base everywhere is a modelling decision outside this script; this script only "
          "reports.", flush=True)
    return dict(completeness=comp,
                family=("information only: per dataset, Holm over the comparisons 'row minus Base-published' for row in "
                        "{EASE, literature content closed forms, controlled swaps}, metric in {R@20, N@20}; paired "
                        "user-level bootstrap, two-sided, p=(count+1)/(B+1)"),
                n_tests_by_dataset={d: len(f) for d, f in fam.items()}, p_raw=fam, p_holm=adj, rows=rows,
                rule_G3_1=dict(text=RULE_G3_1, outcome=rule1,
                               verdict_complete=comp["complete"],
                               base_one_member_of_family_on=(fams if comp["complete"] else None),
                               microlens_claim=ml_verdict),
                d6_family=dict(definition=("Base-retuned minus Base-published, datasets x {R@20, N@20}, paired "
                                           "user-level bootstrap, Holm within this family"),
                               expected=d6_expected, missing=d6_missing, p_raw=d6fam, p_holm=d6adj),
                rule_G3_2=dict(text=RULE_G3_2, outcome=rule2,
                               verdict_complete=bool(comp["complete"] and not d6_missing),
                               note="adopting a new base is a modelling decision outside this script"),
                g3_json={r["dataset"]: r.get("json_path") or r.get("_path") for r in results})


def summarize_g4(results):
    out = {}
    present = [r["dataset"] for r in results]
    missing_ds = [d for d in EXPECTED_DATASETS if d not in present]
    exp_keys = [f"s{s}|{m}" for s in EXPECTED_SEEDS for m in METRICS]
    print("\n" + "=" * 110 + "\nG4 SUMMARY\n" + "=" * 110, flush=True)
    for pool in ("literature", "all"):
        prim, sec = {}, {}
        for r in results:
            P = r["pools"].get(pool)
            if not P or "bootstrap" not in P:
                continue
            for key, b in P["bootstrap"].items():
                (sec if key.startswith("seedmean") else prim)[f"{r['dataset']}|{key}"] = b["p_two_sided"]
        ap, as_ = holm(prim), holm(sec)
        verdicts = {}
        for r in results:
            ds, P = r["dataset"], r["pools"].get(pool)
            cmiss = (r.get("g3_candidates_missing") or {}).get(pool)
            if cmiss is None:
                cmiss = ["<not recorded: G4 JSON predates the completeness check>"]
            if not P:
                verdicts[ds] = dict(complete=False, skipped="pool not run")
                continue
            if "same_as_pool" in P:
                verdicts[ds] = dict(kernel=P["kernel"], same_as_pool=P["same_as_pool"], g3_candidates_missing=cmiss,
                                    complete=not cmiss)
                continue
            if "bootstrap" not in P:
                verdicts[ds] = dict(complete=False, skipped=P.get("skipped", "no bootstrap"))
                continue
            keys = [k for k in P["bootstrap"] if not k.startswith("seedmean")]
            missing_tests = [k for k in exp_keys if k not in keys]
            per = {k: dict(delta=P["bootstrap"][k]["mean_delta"], rel_lift=P["bootstrap"][k]["rel_lift"],
                           ci95=P["bootstrap"][k]["ci95"], p=P["bootstrap"][k]["p_two_sided"],
                           p_holm=ap[f"{ds}|{k}"]) for k in keys}
            sm = {k: dict(delta=P["bootstrap"][k]["mean_delta"], p=P["bootstrap"][k]["p_two_sided"],
                          p_holm=as_[f"{ds}|{k}"]) for k in P["bootstrap"] if k.startswith("seedmean")}
            complete_ds = not missing_tests and not cmiss
            sig_obs = bool(keys) and all(v["delta"] > 0 and v["p_holm"] < 0.05 for v in per.values())
            verdicts[ds] = dict(kernel=P["kernel"], kernel_label=row_label(P["kernel"]),
                                kernel_test_R20=P["kernel_test"]["trusted"]["Recall@20"],
                                kernel_view_rezscored=P.get("kernel_view_rezscored"),
                                complete=complete_ds, missing_tests=missing_tests, g3_candidates_missing=cmiss,
                                significant_observed=sig_obs, per_seed=per, seedmean=sm)
        pool_complete = not missing_ds and all(v.get("complete") for v in verdicts.values())
        for ds, v in verdicts.items():
            if "per_seed" in v:
                v["significant_all_seeds_both_metrics"] = v["significant_observed"] if pool_complete else None
        out[pool] = dict(family_primary=("Holm over all (dataset, head seed, metric) comparisons 'fused minus S_kernel' "
                                         f"for pool '{pool}'"), holm_primary=ap, p_primary=prim,
                         expected_primary_tests=len(EXPECTED_DATASETS) * len(exp_keys), n_primary_tests=len(prim),
                         family_secondary="Holm over (dataset, metric) comparisons of the seed-averaged per-user metric",
                         holm_secondary=as_, p_secondary=sec, verdicts=verdicts, missing_datasets=missing_ds,
                         complete=pool_complete, status="complete" if pool_complete else "INCOMPLETE/PROVISIONAL")
        print(f"\n[pool {pool}] {out[pool]['status']} ({len(prim)}/{out[pool]['expected_primary_tests']} primary tests"
              + (f"; missing datasets {missing_ds}" if missing_ds else "") + ")", flush=True)
        for ds, v in verdicts.items():
            if "per_seed" not in v:
                print(f"  [{ds}] {v}", flush=True)
                continue
            s = "; ".join(f"{k} d={x['delta']:+.4f} ({100 * x['rel_lift']:+.1f}%) p_holm={x['p_holm']:.2g}"
                          for k, x in v["per_seed"].items())
            print(f"  [{ds}] kernel {v['kernel_label']} (test R@20 {v['kernel_test_R20']:.4f}): {s}", flush=True)
            if not pool_complete:
                print(f"  [{ds}] => INCOMPLETE/PROVISIONAL: no verdict (missing tests {v['missing_tests']}, missing "
                      f"G3 candidates {v['g3_candidates_missing']}; observed all-significant={v['significant_observed']})",
                      flush=True)
            else:
                print(f"  [{ds}] => head margin over the strongest content closed form: "
                      + ("SIGNIFICANT (all seeds, both metrics)" if v["significant_all_seeds_both_metrics"] else
                         "N.S. on at least one seed/metric on this dataset"), flush=True)
    print("\nNARROWING RULE G4: " + RULE_G4, flush=True)
    litp = out.get("literature", {})
    lit = litp.get("verdicts", {})
    if litp.get("complete"):
        ns = [ds for ds, v in lit.items() if "per_seed" in v and not v["significant_all_seeds_both_metrics"]]
        print(f"  => datasets where the head's gain is established over our base but not over the strongest content "
              f"closed form: {ns if ns else 'none'}", flush=True)
    else:
        ns = None
        print("  => INCOMPLETE/PROVISIONAL: no G4 verdict (a dataset, head seed, metric or G3 kernel candidate is "
              "missing)", flush=True)
    out["rule"] = RULE_G4
    out["not_significant_over_kernel_on"] = ns
    out["g4_json"] = {r["dataset"]: r.get("json_path") or r.get("_path") for r in results}
    return out


# ------------------------------------------------------------------------------------------------ main
def parse_ds_paths(items):
    out = {}
    for it in items or []:
        if "=" in it:
            ds, p = it.split("=", 1)
            out[ds] = p
        else:
            j = json.loads(Path(it).read_text())
            out[j["dataset"]] = it
    return out


def main(args):
    global SOLVER
    SOLVER = args.solver
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_p{os.getpid()}"
    out_dir = Path(args.out) if args.out else ROOT / "results" / "scope" / "rev"
    if args.smoke:
        out_dir = out_dir / "smoke"
    log_dir = Path(args.log_dir) if args.log_dir else ROOT / "logs" / "rev"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logp = fresh_path(log_dir, f"g3g4_{args.stage}{'_smoke' if args.smoke else ''}_{run_id}", ".log")
    fh = open(logp, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    t0 = time.time()
    print(f"[G3G4] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} stage={args.stage} datasets={args.datasets} "
          f"seeds={args.seeds} smoke={args.smoke} tuning={args.tuning} solver={args.solver} root={ROOT}\n"
          f"[G3G4] out={out_dir} log={logp}", flush=True)
    if Path(SC.__file__).resolve().parents[1] != ROOT:
        print(f"WARNING: scope.py resolved from {SC.__file__}, outside ROOT={ROOT}", flush=True)
    env0 = make_env(args)
    boot_check = check_bootstrap_equivalence() if args.smoke else None
    tag = f"s{args.seeds[0]}_{args.tuning}{'_smoke' if args.smoke else ''}_{run_id}"
    if args.stage == "holm":
        g3r = [json.loads(Path(p).read_text()) for p in parse_ds_paths(args.g3_json).values()]
        g4r = [json.loads(Path(p).read_text()) for p in parse_ds_paths(args.g4_json).values()]
        summ = dict(stage="holm", run_id=run_id, argv=sys.argv, bootstrap_check=boot_check,
                    g3=summarize_g3(g3r, args) if g3r else None, g4=summarize_g4(g4r) if g4r else None)
        p = fresh_path(out_dir, f"g3g4_holm_summary_{run_id}", ".json")
        dump_json(p, summ)
        print(f"[G3G4] holm summary -> {p}", flush=True)
        return
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required for stages all/g3/g4")
    g3_inputs = parse_ds_paths(args.g3_json) if args.stage == "g4" else {}
    g3_all, g4_all, skipped = [], [], {}
    for ds in args.datasets:
        if ds == "elec":
            skipped[ds] = "--: dense |I|^2 closed forms infeasible (63,001 items -> 15.9 GB per fp32 I x I buffer)"
            print(f"[elec] {skipped[ds]}", flush=True)
            continue
        if ds not in DATASETS_OK:
            skipped[ds] = "unknown dataset"
            print(f"[{ds}] unknown dataset -- skipped", flush=True)
            continue
        ctx = None
        try:
            ctx = Ctx(ds, args, args.seeds[0])
            print(f"[{ds}] U={ctx.U} I={ctx.I} dt={ctx.dt} frugal={ctx.frugal} offload={ctx.offload} "
                  f"({ctx.build_s}s)", flush=True)
            if args.stage in ("all", "g3"):
                g3 = run_g3(ctx, args, env0, out_dir, run_id)
                g3_all.append(g3)
            else:
                if ds not in g3_inputs:
                    raise RuntimeError(f"--stage g4 needs --g3-json {ds}=<g3 json path>")
                g3 = json.loads(Path(g3_inputs[ds]).read_text())
                g3["_path"] = g3_inputs[ds]
            if args.stage in ("all", "g4"):
                g4_all.append(run_g4(ctx, args, env0, g3, out_dir, run_id))
        except Exception as e:
            traceback.print_exc()
            skipped[ds] = f"FAILED: {type(e).__name__}: {str(e)[:300]}"
            print(f"[{ds}] FAILED -- continuing with the next dataset", flush=True)
        finally:
            if ctx is not None:
                ctx.close()
            del ctx
            torch.cuda.empty_cache()
    if g3_all:
        s = summarize_g3(g3_all, args)
        s.update(stage="g3-summary", run_id=run_id, argv=sys.argv, skipped=skipped, bootstrap_check=boot_check)
        p = fresh_path(out_dir, f"g3_summary_{tag}", ".json")
        dump_json(p, s)
        print(f"[G3G4] G3 summary -> {p}", flush=True)
    if g4_all:
        s = summarize_g4(g4_all)
        s.update(stage="g4-summary", run_id=run_id, argv=sys.argv, skipped=skipped, bootstrap_check=boot_check)
        p = fresh_path(out_dir, f"g4_summary_s{'-'.join(map(str, args.seeds))}{'_smoke' if args.smoke else ''}_"
                                f"{run_id}", ".json")
        dump_json(p, s)
        print(f"[G3G4] G4 summary -> {p}", flush=True)
    print(f"[G3G4] finished in {round(time.time() - t0, 1)}s; skipped/failed: {skipped if skipped else 'none'}",
          flush=True)


if __name__ == "__main__":
    main(_ARGS)

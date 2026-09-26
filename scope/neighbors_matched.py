#!/usr/bin/env python
"""G7 + G13: matched-budget comparison of the SCOPE set-completion head
with its nearest masked neighbours. Every configuration and seed that is run is written to JSON; nothing is filtered.

G7  (CBOW gate)       CBOW {random, content init} x SIGReg(E) le {0, 1}, lr 3e-3, batch 8192, <=400 epochs,
                      validation every 4 epochs, patience 20 checks; seeds x datasets; every run is tested. Fusion gamma
                      on the pre-registered G7 grid {0,.3,.6,1,1.5,2,3,5} with NO extension (gamma=5 = edge hit, logged).
G13 (fair neighbours) ONE shared masked-set trainer (the loop of scope.train) with pluggable encoders
                        pool       item2vec/CBOW: mean-pool of the context items
                        pool_mlp   SCOPE head re-implementation: mean-pool + one residual MLP (scope.SCOPE itself);
                                   (a) cap-60 lists = exact copy of the deployed scope.train (the sanity check against
                                   the deployed head, default), (b) full lists (the encoder-only comparison vs each arm)
                        tf_bidir   bidirectional set Transformer, [CLS] readout (BERT4Rec-style, no positional emb.)
                        tf_causal  causal set Transformer with a BOS token, last-context readout (SASRec-style, no
                                   positional emb.); Linear weights trunc-normal(0.02), zero biases, each layer separately
                      plus Mult-VAE trained with its OWN multinomial-likelihood objective (labelled as such), KL annealed
                      in absolute gradient steps, epoch cap large enough not to bind (flagged if it does).
                      Grids at seed 2024 (validation only; every grid point's validation score saved), 3 seeds of each
                      arm's config selected on standalone validation R@20 AND (additional panel) of its config selected
                      on fused validation R@20 when that differs, fusion with the deployed base (gamma validation-tuned on
                      {0,.3,.6,1,1.5,2,3,5}, extended by {8,12} if 5 wins), SCOPE reference = the deployed checkpoints,
                      paired bootstrap (B=1e4, per-user R@20 and N@20) with Holm in code over each full pre-stated family
                      (missing / incomplete tests enter Holm with p=1 and get no verdict).

Differences from the earlier neighbour scripts (masked_neighbors.py and w13_item2vec.py), which this file
does not reuse:
  * F.normalize(z, 1) is an L1 norm            -> every head scores with F.normalize(., dim=1) (cosine / tau)
  * F.normalize(randn(D, d), 0) is p=0 (13x)   -> content seed copied from scope.py (dim=0)
  * RNG seeded after build_lists               -> src.utils.seed.set_seed before data loading and before every run
  * test-selected 'best config' (earlier script)          -> selection on validation only; test evaluated for selected configs
  * gamma grid without 5.0 / fixed gamma (earlier script) -> identical validation-tuned grid for every model of a comparison
                                                  (G13: +8, 12 if 5 wins; G7: pre-registered grid, no extension)
  * 200-epoch Mult-VAE cap + epoch-tied KL     -> absolute-step KL annealing, cap 1000, hit_cap recorded
  * Transformers scored from 60-item lists     -> list cap = max train degree for training and inference
  * torch.nan_to_num hiding NaN encodings      -> removed; a non-finite loss fails the run (recorded, not dropped)
  * recall-only bootstrap, no Holm             -> per-user R@20 and N@20, Holm step-down over each stated family
  * selective reporting                       -> every run is written; the table generator prints every row

Stages (run in this order; each is resumable, completed runs are reused, nothing is overwritten):
  --stage g7         CBOW gate (+ SCOPE reference scoring)          GPU
  --stage grid       G13 grids at --grid_seed (validation only)      GPU
  --stage seeds      select each arm on validation (standalone and fused), run all seeds of both selections   GPU
  --stage sanity     SCOPE head re-run with the shared trainer (cap-60 replica + full lists)   GPU
  --stage summarize  bootstrap + Holm + gate/narrowing JSON          CPU only
  --stage all        g7 -> grid -> seeds -> sanity -> summarize
  --stage ref        only the deployed SCOPE reference scoring      GPU
  --stage eval       re-fuse / re-test every existing run from its checkpoint with the current --base_tag (no training)
Tables: scope/neighbors_tables.py --summary <summary.json>.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = Path(os.environ.get("SCOPE_ROOT") or HERE.parents[1]).resolve()      # <root>/scope/<this file>
for _p in (HERE.parent, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must be set before CUDA init

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scope as SC                          # scope/scope.py        (reused as is, never modified)
from gpu_eval import GPUEval                # scope/gpu_eval.py
from harness import paired_bootstrap        # scope/harness.py
from src.utils import Config
from src.utils.seed import set_seed
from src.data.dataset import RecDataset

if Path(SC.__file__).resolve().parents[1] != ROOT:
    raise RuntimeError(f"scope.py imported from {SC.__file__}, which is not under SCOPE_ROOT={ROOT}")

SCRIPT = "scope/neighbors_matched.py"
VERSION = "2026-09-25.r2"                  # r2
DEV = torch.device("cuda:0")                # replaced from --device in main(); scope.DEV is patched to match

GAMMA_GRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]    # scope.train's grid (0 = model alone)
GAMMA_EXT = [8.0, 12.0]                                  # evaluated only if 5.0 wins on validation
# gamma rules: "ext" = G13 (GAMMA_GRID, + GAMMA_EXT if 5 wins); "fixed" = G7 (GAMMA_GRID only, pre-registered).
# Evaluation records are namespaced by the rule (RULE_SFX), so a training run shared by G7 and the G13 grid is
# evaluated once under each rule and neither result overwrites or replaces the other.
GAMMA_RULES = ("ext", "fixed")
RULE_SFX = {"ext": "", "fixed": "__gfixed"}
TF_INIT_STD = 0.02                                       # BERT/BERT4Rec initializer_range
MDE_PLAN = {"baby": 0.0026, "sports": 0.0022, "clothing": 0.0017}   # w16_power_{ds}.json mde_at_80pct (.00255/.00216/.00174)
Z_MDE80 = 1.959963985 + 0.841621234                      # two-sided alpha .05, power .80
EVAL_TOL_OK = 2e-4                                       # |mean(per-user GPUEval) - evalS_trusted| (GPUEval doc: <1e-4)
EVAL_TOL_FAIL = 1e-3
REPRO_TOL = 1e-4                                         # deployed-JSON reproduction tolerance (SCOPE reference, base)
ZR_INPLACE_OK = True                                     # set by selftest_zr()

ARMS = {
    "cbow_random":  {"kind": "masked", "encoder": "pool", "init": "random",
                     "label": "item2vec/CBOW (mean-pool, random init)"},
    "cbow_content": {"kind": "masked", "encoder": "pool", "init": "content",
                     "label": "item2vec/CBOW (mean-pool, content init)"},
    "scope_mlp":    {"kind": "masked", "encoder": "pool_mlp", "init": "content",
                     "label": "SCOPE head re-run (shared trainer)"},
    "bert_cls":     {"kind": "masked", "encoder": "tf_bidir", "init": "content",
                     "label": "Bidirectional set Transformer, [CLS] (BERT4Rec-style)"},
    "sas_bos":      {"kind": "masked", "encoder": "tf_causal", "init": "content",
                     "label": "Causal set Transformer, BOS (SASRec-style)"},
    "multvae":      {"kind": "vae", "encoder": "multvae", "init": "xavier",
                     "label": "Mult-VAE (own multinomial objective)"},
}
G13_ARMS = ["cbow_random", "cbow_content", "bert_cls", "sas_bos", "multvae"]
SANITY_HP = {"lr": 3e-3, "le": 1.0}                      # deployed SCOPE configuration (d 256, content init)
SELECTION_RULE = ("per (dataset, arm): the grid point with the highest validation Recall@20 of the standalone model at "
                  "its early-stopped checkpoint (the early-stopping criterion), grid seed only; ties keep the earlier grid "
                  "point; failed grid points are recorded and cannot be selected; test is never consulted")
SELECTION_RULE_FUSED = ("additional panel, per (dataset, arm, base_tag): the grid point with the highest FUSED validation "
                        "Recall@20 (arm z-scored + gamma * base, gamma validation-tuned per grid point with the G13 "
                        "rule), grid seed only; ties keep the earlier grid point; grid points whose training or "
                        "evaluation failed are recorded and cannot be selected; test is never consulted. When it "
                        "differs from the standalone selection, its seeds are run too")


def g7_configs():
    return [("cbow_random", {"lr": 3e-3, "le": 0.0}), ("cbow_random", {"lr": 3e-3, "le": 1.0}),
            ("cbow_content", {"lr": 3e-3, "le": 0.0}), ("cbow_content", {"lr": 3e-3, "le": 1.0})]


def g13_grid(arm, smoke=False):
    if arm in ("cbow_random", "cbow_content"):
        g = [{"lr": lr, "le": le} for lr in (1e-3, 3e-3) for le in (0.0, 1.0)]
    elif arm in ("bert_cls", "sas_bos"):
        g = [{"lr": lr, "dropout": do, "le": le} for lr in (1e-3, 3e-3) for do in (0.1, 0.2) for le in (0.0, 1.0)]
    elif arm == "multvae":
        g = [{"lr": lr, "beta_cap": bc} for lr in (1e-3, 3e-3) for bc in (0.2, 0.5)]
    else:
        raise KeyError(arm)
    return g[:2] if smoke else g


# ============================================================================ small utilities
def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_ts():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path, obj):
    """Atomic write (tmp + rename). Used for a run's own in-progress record; finished artifacts are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jdefault))
    os.replace(tmp, path)


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"status": "corrupt", "_path": str(path)}


def preserve(path):
    """Never overwrite: move an existing stale / failed / partial artifact aside under a timestamped name."""
    path = Path(path)
    if path.exists():
        dst = path.with_name(f"{path.stem}.stale_{utc_ts()}{path.suffix}")
        os.replace(path, dst)
        return dst
    return None


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def search_tag(hp_search):
    return "_".join(f"{k}{_fmt(hp_search[k])}" for k in sorted(hp_search))


def cfg_key(arm, hp_search):
    return f"{arm}|{search_tag(hp_search)}"


def gpu_gb():
    return torch.cuda.max_memory_allocated(DEV) / 1e9


class Log:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg, echo=True):
        line = f"[{utc_now()}] {msg}"
        if echo:
            print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


class Paths:
    def __init__(self, args):
        self.out = Path(args.out)
        self.logs = Path(args.logdir)
        self.ckpt = Path(args.ckpt_dir)
        self.runs = self.out / "runs"                         # base-independent training records
        self.eval = self.out / f"eval_{args.base_tag}"        # base-dependent fusion / test records
        self.base_cache = self.out / "base_cache"
        self.dsinfo = self.out / "datasets"
        self.selection = self.out / "selection"

    def train(self, ds, rid):
        return self.runs / ds / f"{rid}.json"

    def ckptp(self, ds, rid):
        return self.ckpt / ds / f"{rid}.pt"

    def ev(self, ds, rid, mode, rule="ext"):
        return self.eval / ds / f"{rid}__{mode}{RULE_SFX[rule]}.json"

    def npz(self, ds, rid, rule="ext"):
        return self.eval / ds / f"{rid}{RULE_SFX[rule]}__test_peruser.npz"

    def ref(self, ds, seed, sfx=""):
        """sfx '' = G13 rule or forced deployed gamma (rule-independent); '__gfixed' = G7 rule with a re-tuned gamma."""
        return self.eval / ds / f"scope_ref__s{seed}{sfx}.json"

    def ref_npz(self, ds, seed, sfx=""):
        return self.eval / ds / f"scope_ref__s{seed}{sfx}__test_peruser.npz"

    def sel_fused(self, ds, arm, grid_seed):                  # base-dependent: lives under eval_<base_tag>
        return self.eval / "selection" / f"{ds}__{arm}__gridseed{grid_seed}__fused.json"

    def base_rec(self, ds):
        return self.eval / ds / "base_info.json"

    def runlog(self, ds, rid):
        return self.logs / ds / f"{rid}.log"

    def sel(self, ds, arm, grid_seed):
        return self.selection / f"{ds}__{arm}__gridseed{grid_seed}.json"


class Ctx:
    def __init__(self, args, P, log):
        self.args, self.P, self.log = args, P, log
        self.n_fail = 0
        self.n_skipped = 0
        self.max_peak_gb = 0.0
        self.zr_selftest = None


def scope_ckpt_path(args, ds, seed):
    suf = "" if int(seed) == 2024 else f"_s{seed}"
    return Path(args.scope_ckpt_dir) / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003{suf}.pt"


def scope_json_path(args, ds, seed):
    suf = "" if int(seed) == 2024 else f"_s{seed}"
    return Path(args.scope_json_dir) / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003{suf}.json"


def full_hp(arm, hp_search, args, L, cap_override=None):
    """Complete configuration of a run. hp_search holds the searched keys; the rest is the shared budget."""
    meta = ARMS[arm]
    if meta["kind"] == "masked":
        hp = {"objective": "masked-set softmax (scope.train)", "encoder": meta["encoder"], "init": meta["init"],
              "d": 256, "bs": int(args.bs), "wd": 1e-6, "dropout": 0.0,
              "max_epochs": int(args.max_epochs), "patience": int(args.patience), "eval_every": int(args.eval_every),
              "list_cap": int(cap_override) if cap_override else int(L)}
        if meta["encoder"].startswith("tf_"):
            hp.update({"nlayer": 2, "nhead": 4, "ffn_mult": 2, "emb_layernorm": True,
                       "tf_init": f"trunc_normal(std={TF_INIT_STD:g}, +-2 std) q/k/v/out/ffn Linear weights, zero "
                                  f"biases, each layer independently",
                       "lr_warmup": "none (as RecBole BERT4Rec/SASRec and the original SASRec)"})
        for k in ("lr", "le"):
            if k not in hp_search:
                raise KeyError(f"{arm}: hyper-parameter {k} missing from {hp_search}")
    else:
        hp = {"objective": "Mult-VAE multinomial ELBO (Liang et al. 2018), not the masked-set objective",
              "encoder": "multvae", "init": "xavier", "hidden": 600, "latent": 200, "dropout": 0.5, "bs": 500,
              "wd": 0.0, "anneal_steps": int(args.vae_anneal_steps), "max_epochs": int(args.vae_max_epochs),
              "patience": int(args.patience), "eval_every": int(args.eval_every)}
        for k in ("lr", "beta_cap"):
            if k not in hp_search:
                raise KeyError(f"{arm}: hyper-parameter {k} missing from {hp_search}")
    hp.update(hp_search)
    if args.smoke:
        hp["smoke_users"] = int(args.smoke_users)
    return hp


def run_id_for(arm, hp_search, hp, seed):
    h = hashlib.sha1(json.dumps(hp, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{arm}__{search_tag(hp_search)}__cap{hp.get('list_cap', 'na')}__h{h}__s{seed}"


# ============================================================================ scoring helpers (reuse scope/GPUEval)
def zr_inplace(S):
    """Per-user z-score with scope.zr semantics. For fp32 inputs with <=50000 rows it runs in place (same mean/std and
    the same subtract/divide kernels as scope.zr -> bitwise identical, verified by selftest_zr) and so avoids two extra
    dense [U,I] fp32 buffers. Everything else is delegated to scope.zr unchanged."""
    if (not ZR_INPLACE_OK) or S.dtype != torch.float32 or S.shape[0] > 50000:
        return SC.zr(S)
    m = S.mean(1, keepdim=True)
    sd = S.std(1, keepdim=True)
    S.sub_(m).div_(sd + 1e-9)
    return S


def selftest_zr():
    global ZR_INPLACE_OK
    g = torch.Generator(device=DEV)
    g.manual_seed(12345)
    X = torch.randn(1500, 3001, generator=g, device=DEV) * 2.5 + 0.7
    ref = SC.zr(X.clone())
    ZR_INPLACE_OK = True
    got = zr_inplace(X.clone())
    ok = bool(torch.equal(ref, got))
    ZR_INPLACE_OK = ok
    del X, ref, got
    torch.cuda.empty_cache()
    return ok


class FusedView:
    """Lazy row view of (Sz + g * base): rows are produced on demand, so the fused [U,I] matrix is never materialised.
    The arithmetic is element-wise in the same dtypes as scope.train's `Sz + g * base`, hence identical per row.
    Works with GPUEval (S[bu].clone().float()) and evalS_trusted (S[uids].clone().float())."""

    def __init__(self, A, B, g):
        self.A, self.B, self.g = A, B, float(g)

    def __getitem__(self, idx):
        if torch.is_tensor(idx) and idx.device != self.A.device:
            idx = idx.to(self.A.device)
        return self.A[idx] + self.g * self.B[idx]

    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        return self.A.dtype


@torch.no_grad()
def per_user_R_N(gev, S, k=20, batch=4096):
    """Per-user Recall@k and NDCG@k in gev.users order. Uses GPUEval's own train-history mask (gev._mask), positives
    (gev.pos), counts (gev.nfit) and ideal-DCG table (gev.cumdisc) with exactly the formulas of GPUEval._run; the means
    are cross-checked against evalS_trusted by the caller."""
    if k > gev.maxk:
        raise ValueError("k larger than the evaluator's max k")
    U = gev.users.numel()
    R = torch.zeros(U, device=gev.dev)
    N = torch.zeros(U, device=gev.dev)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=gev.dev).float())
    for s in range(0, U, batch):
        bu = gev.users[s:s + batch]
        e = s + bu.numel()
        sc = gev._mask(S[bu].clone().float(), bu)
        _, idx = torch.topk(sc, gev.maxk, dim=1)
        bp = gev.pos[s:e]
        hit = (idx.unsqueeze(2) == bp.unsqueeze(1)).any(2).float()[:, :k]
        nrel = gev.nfit[s:e].clamp(min=1)
        R[s:e] = hit.sum(1) / nrel
        dcg = (hit * disc.unsqueeze(0)).sum(1)
        ideal_n = torch.minimum(nrel, torch.full_like(nrel, k)).long()
        N[s:e] = dcg / gev.cumdisc[ideal_n].clamp(min=1e-9)
    return R.cpu().numpy().astype(np.float64), N.cpu().numpy().astype(np.float64)


def content_init(dset, d):
    """Content seed, copied verbatim from scope.py:219-220 (column-normalised projection: dim=0, NOT p=0)."""
    X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
    Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), dim=0)
    init = (X @ Wp) / math.sqrt(d)
    del X, Wp
    return init


# ============================================================================ models
class CBOW(nn.Module):
    """item2vec/CBOW head = the SCOPE head minus its residual MLP: mean-pool, L2-cosine / learnable tau."""

    def __init__(self, n_items, d=256, init=None):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))       # same random init as scope.SCOPE
        if init is not None:
            self.E.data.copy_(init)
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))

    def latent(self, ctx_sum, n):
        return ctx_sum / n.clamp(min=1).unsqueeze(1)

    def logits_from(self, z):
        return (F.normalize(z, dim=1) @ F.normalize(self.E, dim=1).t()) / self.logtau.exp().clamp(min=1e-3)

    @torch.no_grad()
    def score_all(self, R, deg):                                             # mirrors scope.SCOPE.score_all
        z = self.latent(torch.sparse.mm(R, self.E), deg)
        if z.shape[0] > 50000:
            zt = F.normalize(z, dim=1).half()
            Et = F.normalize(self.E, dim=1).half()
            return (zt @ Et.t()) / self.logtau.exp().clamp(min=1e-3).half()
        return self.logits_from(z)


class SetTF(nn.Module):
    """Set Transformer over the observed items (no positional embeddings). Bidirectional: [CLS] readout (BERT4Rec-style).
    Causal: a BOS token is prepended and the output at the last context position is read out (SASRec-style); BOS is
    never masked, so no attention row is empty. Input tokens pass through LayerNorm + dropout (as in RecBole's
    BERT4Rec/SASRec), which puts the [CLS]/BOS token and the content-seeded item rows on the same scale. Output side:
    L2-cosine against the item table E / learnable tau, like the SCOPE head."""

    def __init__(self, n_items, d=256, init=None, causal=False, nlayer=2, nhead=4, dropout=0.1, ffn_mult=2):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))
        if init is not None:
            self.E.data.copy_(init)
        self.cls = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.emb_ln = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=ffn_mult * d, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=False)
        self.tf = nn.TransformerEncoder(layer, nlayer, enable_nested_tensor=False)
        # nn.TransformerEncoder deep-copies `layer`, so every layer would start from the SAME weights (PyTorch default
        # init). Re-initialise each layer independently, BERT/BERT4Rec-style: truncated normal (std 0.02, cut at
        # +-2 std as TF's truncated_normal) on the q/k/v, attention-output and feed-forward Linear weights, zero biases.
        # LayerNorms keep weight 1 / bias 0 (as RecBole's _init_weights).
        a_ = 2.0 * TF_INIT_STD
        for lyr in self.tf.layers:
            sa = lyr.self_attn
            if sa.in_proj_weight is None:
                raise RuntimeError("expected packed q/k/v projection (kdim == vdim == d)")
            nn.init.trunc_normal_(sa.in_proj_weight, std=TF_INIT_STD, a=-a_, b=a_)
            nn.init.zeros_(sa.in_proj_bias)
            for lin in (sa.out_proj, lyr.linear1, lyr.linear2):
                nn.init.trunc_normal_(lin.weight, std=TF_INIT_STD, a=-a_, b=a_)
                nn.init.zeros_(lin.bias)
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))
        self.causal, self.d, self.nhead = bool(causal), int(d), int(nhead)

    def encode(self, it_c, valid):
        """it_c [b, Lc]: context items LEFT-aligned in their original relative order; valid [b, Lc] bool."""
        b, Lc = it_c.shape
        tok = torch.cat([self.cls.view(1, 1, -1).expand(b, 1, self.d), self.E[it_c]], 1)      # [b, 1+Lc, d]
        x = self.emb_drop(self.emb_ln(tok))
        keypad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=it_c.device), ~valid], 1)
        if self.causal:
            cm = torch.triu(torch.ones(Lc + 1, Lc + 1, dtype=torch.bool, device=it_c.device), 1)
            h = self.tf(x, mask=cm, src_key_padding_mask=keypad)
            n = valid.sum(1)                                   # readout position: last context item (0 = BOS)
            return h[torch.arange(b, device=it_c.device), n]
        h = self.tf(x, src_key_padding_mask=keypad)
        return h[:, 0]

    def logits_from(self, z):
        return (F.normalize(z, dim=1) @ F.normalize(self.E, dim=1).t()) / self.logtau.exp().clamp(min=1e-3)


class MultVAE(nn.Module):
    """Mult-VAE (Liang et al., WWW 2018): [I -> 600 -> 2x200] encoder, [200 -> 600 -> I] decoder, tanh, input L2-norm +
    dropout 0.5; Xavier weights and N(0, 1e-3) biases as in the reference implementation."""

    def __init__(self, n_items, hidden=600, latent=200, dropout=0.5):
        super().__init__()
        self.enc1 = nn.Linear(n_items, hidden)
        self.enc2 = nn.Linear(hidden, 2 * latent)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout)
        self.latent = int(latent)
        for m in (self.enc1, self.enc2, self.dec1, self.dec2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, 0.0, 1e-3)

    def forward(self, x):
        h = self.drop(F.normalize(x, p=2, dim=1))
        h = self.enc2(torch.tanh(self.enc1(h)))
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.dec2(torch.tanh(self.dec1(z))), mu, logvar


def build_model(arm, hp, n_items, init=None):
    enc = ARMS[arm]["encoder"]
    if enc == "pool":
        return CBOW(n_items, hp["d"], init)
    if enc == "pool_mlp":
        return SC.SCOPE(n_items, hp["d"], init)
    if enc in ("tf_bidir", "tf_causal"):
        return SetTF(n_items, hp["d"], init, causal=(enc == "tf_causal"), nlayer=hp["nlayer"], nhead=hp["nhead"],
                     dropout=hp["dropout"], ffn_mult=hp["ffn_mult"])
    if enc == "multvae":
        return MultVAE(n_items, hp["hidden"], hp["latent"], hp["dropout"])
    raise KeyError(enc)


# ============================================================================ dataset context
class DSCtx:
    def __init__(self, name, ctx):
        args, P, log = ctx.args, ctx.P, ctx.log
        self.name = name
        t0 = time.time()
        set_seed(args.seeds[0], deterministic=bool(args.deterministic))    # seed BEFORE data loading / build_lists
        self.dset = RecDataset(Config(args.config_model, name))
        if self.dset.t_feat is None:
            raise FileNotFoundError(f"{name}: text_feat.npy not found; the content seed and the deployed base need it")
        self.n_users, self.n_items = int(self.dset.n_users), int(self.dset.n_items)
        self.R = SC.Rmat(self.dset)
        cap = int(args.list_cap) if args.list_cap > 0 else 10 ** 9
        self.items, self.vmask, self.deg = SC.build_lists(self.dset, cap=cap)   # cap >= max degree: nothing sampled
        self.degf = self.deg.float()
        self.L = int(self.items.shape[1])
        self.max_deg = int(self.deg.max().item())
        if args.list_cap <= 0 and self.L != self.max_deg:
            raise RuntimeError(f"{name}: list width {self.L} != max train degree {self.max_deg}")
        self.half = self.n_items > 20000 or self.n_users > 50000
        self.dt = torch.float16 if self.half else torch.float32
        self.gev = GPUEval(self.dset, "valid", DEV)
        self.gevT = GPUEval(self.dset, "test", DEV)
        self._rb = None
        info = {"dataset": name, "n_users": self.n_users, "n_items": self.n_items, "L": self.L,
                "max_train_degree": self.max_deg, "list_cap_arg": int(args.list_cap),
                "users_over_60_items": int((self.deg > 60).sum().item()),
                "n_train_pairs": int(len(self.dset.train_users)), "R_nnz": int(self.R._nnz()),
                "score_dtype": str(self.dt), "n_valid_users": int(self.gev.users.numel()),
                "n_test_users": int(self.gevT.users.numel()), "config_model": args.config_model,
                "text_feat_shape": list(self.dset.t_feat.shape)}
        if info["R_nnz"] != info["n_train_pairs"]:
            log(f"[{name}] WARNING: {info['n_train_pairs'] - info['R_nnz']} duplicate train pairs (scope.Rmat sums them, "
                f"exactly as for the deployed model)")
        p = P.dsinfo / f"{name}.json"
        old = load_json(p)
        if old is None:
            write_json(p, dict(info, created_utc=utc_now()))
        else:
            for k in ("n_users", "n_items", "L", "n_train_pairs", "R_nnz", "n_test_users"):
                if old.get(k) != info[k]:
                    raise RuntimeError(f"{name}: dataset changed since {p} was written ({k}: {old.get(k)} vs {info[k]})")
        self.info = info
        self.base_cpu, self.base_info = self._base(ctx)
        log(f"[{name}] ready in {time.time() - t0:.0f}s: U={self.n_users} I={self.n_items} L={self.L} "
            f"dtype={self.dt} base={self.base_info.get('selected')}")

    def _base(self, ctx):
        args, P, log = ctx.args, ctx.P, ctx.log
        rec_p = P.base_rec(self.name)
        if args.base_npy:                                              # externally supplied base (e.g. after G3)
            arr = np.load(args.base_npy, mmap_mode="r")
            if tuple(arr.shape) != (self.n_users, self.n_items):
                raise ValueError(f"--base_npy shape {arr.shape} != ({self.n_users}, {self.n_items})")
            B = torch.from_numpy(np.ascontiguousarray(arr)).to(self.dt)
            Bg = B.to(DEV)
            info = {"status": "complete", "source": "npy", "path": str(Path(args.base_npy).resolve()),
                    "sha256": sha256_file(args.base_npy), "file_dtype": str(arr.dtype), "selected": f"npy:{args.base_tag}",
                    "val": self.gev.eval(Bg), "test": SC.evalS_trusted(Bg, self.dset, "test")}
            del Bg
            torch.cuda.empty_cache()
        else:
            cache = P.base_cache / f"base_{self.name}_{args.base_tag}.npy"
            meta_p = P.base_cache / f"base_{self.name}_{args.base_tag}.json"
            meta = load_json(meta_p)
            if (not args.no_base_cache) and meta and meta.get("status") == "complete" and cache.is_file():
                B = torch.from_numpy(np.load(cache))
                if tuple(B.shape) != (self.n_users, self.n_items) or str(B.dtype) != str(self.dt):
                    raise RuntimeError(f"cached base {cache} has shape {tuple(B.shape)} dtype {B.dtype}")
                Bg = B.to(DEV)
                v = self.gev.eval(Bg)["Recall@20"]
                del Bg
                torch.cuda.empty_cache()
                if abs(v - meta["val"]["Recall@20"]) > 1e-7:
                    raise RuntimeError(f"cached base {cache} does not reproduce its validation R@20 ({v} vs {meta['val']})")
                info = dict(meta, loaded_from_cache=str(cache))
                log(f"[{self.name}] base loaded from cache {cache}")
            else:
                buf = io.StringIO()
                t0 = time.time()
                torch.cuda.reset_peak_memory_stats(DEV)
                with contextlib.redirect_stdout(buf):                  # deployed base: scope.closed_form_base, val-tuned
                    Bg = SC.closed_form_base(self.R, self.dset, self.gev, half=self.half)
                out = buf.getvalue()
                print(out, end="", flush=True)
                m = re.search(r"\[base\] tuned (\{[^}]*\}) val_R20=([0-9.]+)", out)
                sel = ast.literal_eval(m.group(1)) if m else None
                peak = gpu_gb()
                ctx.max_peak_gb = max(ctx.max_peak_gb, peak)
                val = self.gev.eval(Bg)
                test = SC.evalS_trusted(Bg, self.dset, "test")
                dep = load_json(scope_json_path(args, self.name, 2024))
                repro = None
                if dep and "base" in dep:
                    dR = abs(test["Recall@20"] - dep["base"]["Recall@20"])
                    dN = abs(test["NDCG@20"] - dep["base"]["NDCG@20"])
                    repro = {"deployed_json": str(scope_json_path(args, self.name, 2024)), "R20_abs_diff": dR,
                             "N20_abs_diff": dN, "ok": max(dR, dN) <= REPRO_TOL}
                    if not repro["ok"]:
                        log(f"[{self.name}] WARNING base test metrics differ from the deployed JSON by {max(dR, dN):.2e}")
                B = Bg.cpu()
                del Bg
                torch.cuda.empty_cache()
                info = {"status": "complete",
                        "source": "scope.closed_form_base (deployed base: 1-hop EASE + text-kNN, lam/a validation-tuned)",
                        "selected": sel, "val_R20_printed": float(m.group(2)) if m else None, "val": val, "test": test,
                        "repro_vs_deployed_json": repro, "dtype": str(B.dtype), "shape": list(B.shape),
                        "build_s": time.time() - t0, "build_peak_gb": peak, "created_utc": utc_now(),
                        "checksum_strided_sum": float(B[::97, ::13].double().sum())}
                if not args.no_base_cache:
                    if cache.exists():
                        preserve(cache)
                    if meta_p.exists():
                        preserve(meta_p)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, B.numpy())
                    write_json(meta_p, info)
        old = load_json(rec_p)
        if old is None or old.get("status") != "complete":
            if old is not None:
                preserve(rec_p)
            write_json(rec_p, info)
        elif str(old.get("selected")) != str(info.get("selected")) or \
                abs(old["val"]["Recall@20"] - info["val"]["Recall@20"]) > 1e-7:
            raise RuntimeError(f"{self.name}: base differs from {rec_p} under the same --base_tag; use a new tag")
        return B, info

    def rbool(self):
        """Dense boolean train matrix for Mult-VAE, built by index assignment; nnz asserted."""
        if self._rb is None:
            idx = self.R.indices()
            rb = torch.zeros(self.n_users, self.n_items, dtype=torch.bool, device=DEV)
            rb[idx[0], idx[1]] = True
            nnz = int(rb.sum().item())
            if nnz != int(self.R._nnz()):
                raise RuntimeError(f"dense train matrix nnz {nnz} != sparse nnz {int(self.R._nnz())}")
            self._rb = rb
        return self._rb

    def close(self):
        for k in ("R", "items", "vmask", "deg", "degf", "gev", "gevT", "_rb", "base_cpu", "dset"):
            if hasattr(self, k):
                setattr(self, k, None)
        torch.cuda.empty_cache()


# ============================================================================ scoring functions
def plan_microbatches(lens, max_rows, tok=None, attn=None, nhead=1):
    """Split rows (lens sorted ascending) into chunks with rows*len <= tok and rows*len^2*nhead <= attn."""
    tok = float("inf") if tok is None else tok
    attn = float("inf") if attn is None else attn
    n, out, s = len(lens), [], 0
    while s < n:
        lo, hi = s + 1, min(n, s + max(1, int(max_rows)))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            m, Lm = mid - s, lens[mid - 1]
            if m * Lm <= tok and m * Lm * Lm * max(1, nhead) <= attn:
                lo = mid
            else:
                hi = mid - 1
        out.append((s, lo))
        s = lo
    return out


@torch.no_grad()
def tf_score_all(model, ds, args):
    """Scores of a set Transformer from each user's FULL train history (lists capped at the max degree)."""
    items, vmask = ds.items, ds.vmask
    U, I, L = items.shape[0], model.E.shape[0], items.shape[1]
    out_dt = torch.float16 if U > 50000 else torch.float32
    dcl = ds.deg.clamp(max=L)
    order = torch.argsort(dcl)
    lens = (dcl[order] + 1).tolist()
    spans = plan_microbatches(lens, U, 4 * args.tf_tok_budget, 4 * args.tf_attn_budget, model.nhead)
    Z = torch.empty(U, model.d, device=DEV)
    for s, e in spans:
        rows = order[s:e]
        Lc = max(1, int(lens[e - 1]) - 1)
        Z[rows] = model.encode(items[rows, :Lc], vmask[rows, :Lc] > 0)
    En = F.normalize(model.E, dim=1)
    tau = model.logtau.exp().clamp(min=1e-3)
    out = torch.empty(U, I, dtype=out_dt, device=DEV)
    for s in range(0, U, 4096):
        e = min(s + 4096, U)
        out[s:e] = ((F.normalize(Z[s:e], dim=1) @ En.t()) / tau).to(out_dt)
    del Z, En
    return out


@torch.no_grad()
def vae_score_all(model, ds):
    Rb = ds.rbool()
    U, I = Rb.shape
    out_dt = torch.float16 if U > 50000 else torch.float32
    out = torch.empty(U, I, dtype=out_dt, device=DEV)
    for s in range(0, U, 4096):
        e = min(s + 4096, U)
        out[s:e] = model(Rb[s:e].float())[0].to(out_dt)
    return out


def score_fn_for(arm, model, ds, args):
    enc = ARMS[arm]["encoder"]
    if enc in ("pool", "pool_mlp"):
        return lambda: model.score_all(ds.R, ds.degf)          # full sparse history, as scope.train
    if enc.startswith("tf_"):
        return lambda: tf_score_all(model, ds, args)
    return lambda: vae_score_all(model, ds)


# ============================================================================ shared masked-set trainer
def compact_context(it, ctx):
    """Move the context items of each row to the front, keeping their original relative order."""
    B, L = it.shape
    pos = torch.arange(L, device=it.device).unsqueeze(0).expand(B, L)
    key = torch.where(ctx > 0, pos, pos + L)
    order = key.argsort(1)
    n = (ctx > 0).sum(1)
    return it.gather(1, order), pos < n.unsqueeze(1), n


def encode_ctx(model, enc, it, ctx):
    if enc in ("pool", "pool_mlp"):
        s = F.embedding_bag(it, model.E, per_sample_weights=ctx, mode="sum")   # = (E[it] * ctx[..., None]).sum(1)
        return model.latent(s, ctx.sum(1))
    it_c, valid, n = compact_context(it, ctx)
    Lc = max(1, int(n.max().item()))
    return model.encode(it_c[:, :Lc], valid[:, :Lc])


def masked_step(model, enc, it, vm, dg, le, opt, mb):
    """One optimiser step of scope.train's objective on one batch of users. The context/target split is drawn exactly
    as scope.SCOPE.forward_train (p_mask=None). Transformers are processed in length-sorted micro-batches whose
    gradients are accumulated, so the effective batch size is always the full batch."""
    keys = torch.where(vm > 0, torch.rand_like(vm), torch.full_like(vm, 1e9))
    ranks = keys.argsort(1).argsort(1).float()
    n_ctx = (torch.rand(dg.shape, device=DEV) * (dg - 1).clamp(min=1)).floor() + 1
    n_ctx = torch.minimum(n_ctx, (dg - 1).clamp(min=1))
    ctx = ((ranks < n_ctx.unsqueeze(1)) & (vm > 0)).float()
    tgt = ((ranks >= n_ctx.unsqueeze(1)) & (vm > 0)).float()
    B = it.shape[0]
    is_tf = enc.startswith("tf_")
    if (not is_tf) and mb["max_rows"] >= B:
        chunks = [None]                                      # pooled heads: one chunk, original row order (= scope)
    elif is_tf:
        nc = (ctx > 0).sum(1)
        order = torch.argsort(nc)
        lens = (nc[order] + 1).tolist()
        spans = plan_microbatches(lens, mb["max_rows"], mb["tok"], mb["attn"], model.nhead)
        chunks = [order[s:e] for s, e in spans]
    else:
        chunks = [torch.arange(s, min(s + mb["max_rows"], B), device=DEV) for s in range(0, B, mb["max_rows"])]
    reg = le * SC.sigreg(model.E)                            # SIGReg(E); computed (x0) even when le=0, as scope.train
    opt.zero_grad()
    lrank = 0.0
    for ci, rows in enumerate(chunks):
        if rows is None:
            itm, cmf, tgm = it, ctx, tgt
        else:
            itm, cmf, tgm = it[rows], ctx[rows], tgt[rows]
        z = encode_ctx(model, enc, itm, cmf)
        logits = model.logits_from(z)
        bidx = torch.arange(itm.shape[0], device=DEV).unsqueeze(1).expand_as(itm)
        cm = cmf > 0
        logits = logits.index_put((bidx[cm], itm[cm]), torch.tensor(-1e9, device=DEV))   # context items masked
        logp = F.log_softmax(logits, dim=1)
        tgt_lp = (logp[bidx, itm] * tgm).sum(1) / tgm.sum(1).clamp(min=1)
        part = -tgt_lp.sum() / B                             # sum over chunks = -mean over the batch (= scope lrank)
        lrank += float(part.item())
        if ci == 0:
            part = part + reg
        part.backward()
        del z, logits, logp, tgt_lp, part
    opt.step()
    return lrank + float(reg.item()), lrank, len(chunks)


def masked_step_safe(model, enc, it, vm, dg, le, opt, mb, rl):
    for _ in range(8):
        try:
            return masked_step(model, enc, it, vm, dg, le, opt, mb)
        except torch.cuda.OutOfMemoryError:
            pass
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        mb["max_rows"] = max(1, mb["max_rows"] // 2)
        if mb["tok"] is not None:
            mb["tok"] = max(4096, mb["tok"] // 2)
        if mb["attn"] is not None:
            mb["attn"] = max(1 << 20, mb["attn"] // 2)
        mb["oom_events"] += 1
        rl(f"OOM in a training step -> micro-batch limits halved (rows={mb['max_rows']} tok={mb['tok']} "
           f"attn={mb['attn']}); effective batch size unchanged")
    raise RuntimeError("out of memory after 8 micro-batch halvings")


def train_masked(ds, arm, hp, seed, ctx, rl, cap_override=None):
    args = ctx.args
    enc = ARMS[arm]["encoder"]
    set_seed(seed, deterministic=bool(args.deterministic))
    if cap_override:                                        # exact-procedure replica of scope.train (cap 60 lists)
        items, vmask, deg = SC.build_lists(ds.dset, cap=int(cap_override))
    else:
        items, vmask, deg = ds.items, ds.vmask, ds.deg
    init = content_init(ds.dset, hp["d"]) if ARMS[arm]["init"] == "content" else None
    init_l2 = float(init.norm(dim=1).mean().item()) if init is not None else None
    model = build_model(arm, hp, ds.n_items, init).to(DEV)
    del init
    if init_l2 is None:
        init_l2 = float(model.E.detach().norm(dim=1).mean().item())
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    tu = torch.where(deg >= 2)[0]
    if args.smoke:
        tu = tu[: int(args.smoke_users)]
    score = score_fn_for(arm, model, ds, args)
    mb = {"max_rows": int(hp["bs"]), "tok": int(args.tf_tok_budget) if enc.startswith("tf_") else None,
          "attn": int(args.tf_attn_budget) if enc.startswith("tf_") else None, "oom_events": 0}
    best = {"r": -1.0, "ep": -1, "state": None}
    bad, curve, early, last_ep, steps, nchunks = 0, [], False, -1, 0, []
    for ep in range(hp["max_epochs"]):
        model.train()
        perm = tu[torch.randperm(tu.numel(), device=DEV)]
        tot = lrk = 0.0
        for i in range(0, perm.numel(), hp["bs"]):
            b = perm[i:i + hp["bs"]]
            lv, lr_, k = masked_step_safe(model, enc, items[b], vmask[b], deg[b], hp["le"], opt, mb, rl)
            tot += lv
            lrk += lr_
            steps += 1
            nchunks.append(k)
        last_ep = ep
        if not (math.isfinite(tot) and math.isfinite(lrk)):
            raise FloatingPointError(f"non-finite training loss at epoch {ep} (run recorded as failed)")
        if ep % hp["eval_every"] == 0 or ep == hp["max_epochs"] - 1:
            model.eval()
            S = score()
            vr = float(ds.gev.eval(S)["Recall@20"])
            del S
            curve.append([ep, vr, lrk, tot])
            if vr > best["r"]:
                best = {"r": vr, "ep": ep, "state": {k_: v.detach().clone() for k_, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            rl(f"[{ds.name}] {arm} ep{ep:3d} rank={lrk:.3f} loss={tot:.3f} val_R20={vr:.4f} best={best['r']:.4f}")
            if bad >= hp["patience"]:
                early = True
                rl(f"[{ds.name}] {arm} early stop ep{ep}")
                break
    model.load_state_dict(best["state"])
    model.eval()
    info = {"best_ep": best["ep"], "last_ep": last_ep, "early_stopped": early, "hit_cap": not early,
            "max_epochs": hp["max_epochs"], "best_val_R20": best["r"], "n_train_users": int(tu.numel()),
            "steps": steps, "val_curve[ep,val_R20,rank_loss,total_loss]": curve, "init_row_l2_mean": init_l2,
            "list_width_train": int(items.shape[1]), "oom_events": mb["oom_events"],
            "microbatches_per_step": {"mean": float(np.mean(nchunks)) if nchunks else None,
                                      "max": int(max(nchunks)) if nchunks else None},
            "final_microbatch_limits": {k_: mb[k_] for k_ in ("max_rows", "tok", "attn")},
            "tau_best": float(model.logtau.exp().clamp(min=1e-3).item())}
    return model, info


def train_vae(ds, hp, seed, ctx, rl):
    args = ctx.args
    set_seed(seed, deterministic=bool(args.deterministic))
    Rb = ds.rbool()
    users = torch.where(ds.deg >= 1)[0]
    if args.smoke:
        users = users[: int(args.smoke_users)]
    model = MultVAE(ds.n_items, hp["hidden"], hp["latent"], hp["dropout"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best = {"r": -1.0, "ep": -1, "state": None, "step": 0, "beta": 0.0}
    bad, curve, early, last_ep, step, beta = 0, [], False, -1, 0, 0.0
    for ep in range(hp["max_epochs"]):
        model.train()
        perm = users[torch.randperm(users.numel(), device=DEV)]
        tot = 0.0
        for i in range(0, perm.numel(), hp["bs"]):
            x = Rb[perm[i:i + hp["bs"]]].float()
            logits, mu, logvar = model(x)
            nll = -(F.log_softmax(logits, dim=1) * x).sum(1).mean()
            kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
            beta = min(float(hp["beta_cap"]), step / float(hp["anneal_steps"]))     # absolute-step annealing
            loss = nll + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            tot += float(loss.item())
        last_ep = ep
        if not math.isfinite(tot):
            raise FloatingPointError(f"non-finite Mult-VAE loss at epoch {ep} (run recorded as failed)")
        if ep % hp["eval_every"] == 0 or ep == hp["max_epochs"] - 1:
            model.eval()
            S = vae_score_all(model, ds)
            vr = float(ds.gev.eval(S)["Recall@20"])
            del S
            curve.append([ep, vr, tot, beta, step])
            if vr > best["r"]:
                best = {"r": vr, "ep": ep, "step": step, "beta": beta,
                        "state": {k_: v.detach().clone() for k_, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            rl(f"[{ds.name}] multvae ep{ep:3d} loss={tot:.2f} beta={beta:.4f} val_R20={vr:.4f} best={best['r']:.4f}")
            if bad >= hp["patience"]:
                early = True
                rl(f"[{ds.name}] multvae early stop ep{ep}")
                break
    model.load_state_dict(best["state"])
    model.eval()
    info = {"best_ep": best["ep"], "last_ep": last_ep, "early_stopped": early, "hit_cap": not early,
            "max_epochs": hp["max_epochs"], "best_val_R20": best["r"], "n_train_users": int(users.numel()),
            "steps": step, "beta_at_best": best["beta"], "step_at_best": best["step"], "beta_final": beta,
            "val_curve[ep,val_R20,loss,beta,step]": curve, "oom_events": 0}
    return model, info


# ============================================================================ evaluation (validation gamma + test)
def gamma_select(ds, Sz, base, force_gamma=None, rule="ext"):
    """scope.train's rule: start at gamma=0 (model alone), strict '>' over the grid on validation Recall@20.
    rule 'ext' (G13): extended by GAMMA_EXT if 5 wins; rule 'fixed' (G7): GAMMA_GRID only, never extended.
    edge_hit = the selected gamma is the largest value evaluated (5 under 'fixed'; 12 under 'ext' once extended), i.e.
    the optimum may lie beyond the grid - recorded and flagged, never extended further."""
    if rule not in GAMMA_RULES:
        raise ValueError(rule)
    grid = {}

    def val_of(g):
        m = ds.gev.eval(Sz if g == 0.0 else FusedView(Sz, base, g))
        grid[_fmt(float(g))] = {"R20": m["Recall@20"], "N20": m["NDCG@20"], "R10": m["Recall@10"], "N10": m["NDCG@10"]}
        return m["Recall@20"]

    best_g, best_v = 0.0, val_of(0.0)
    for g in GAMMA_GRID[1:]:
        v = val_of(g)
        if v > best_v:
            best_g, best_v = g, v
    extended = False
    if rule == "ext" and best_g == GAMMA_GRID[-1]:
        extended = True
        for g in GAMMA_EXT:
            v = val_of(g)
            if v > best_v:
                best_g, best_v = g, v
    edge = GAMMA_EXT[-1] if extended else GAMMA_GRID[-1]
    used = float(best_g) if force_gamma is None else float(force_gamma)
    if _fmt(used) not in grid:
        val_of(used)
    return {"grid": grid, "retuned": float(best_g), "used": used, "forced": force_gamma is not None,
            "rule": rule, "extended": extended, "edge_hit": float(best_g) == float(edge), "grid_edge": float(edge),
            "alone": grid["0"], "fused": grid[_fmt(used)]}


def evaluate_scores(ds, score_fn, tested, npz_path, ctx, force_gamma=None, rule="ext"):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(DEV)
    out = {}
    with torch.no_grad():
        S = score_fn()
        S = zr_inplace(S)                                   # = scope.zr
        Sz = S.to(ds.dt) if S.dtype != ds.dt else S         # = zr(S).to(base.dtype) in scope.train
        del S
        torch.cuda.empty_cache()
        base = ds.base_cpu.to(DEV)
        gs = gamma_select(ds, Sz, base, force_gamma, rule)
        out["val"] = gs
        out["score_dtype"] = str(Sz.dtype)
        if tested:
            g = gs["used"]
            Sf = Sz if g == 0.0 else FusedView(Sz, base, g)
            alone = SC.evalS_trusted(Sz, ds.dset, "test")
            fused = SC.evalS_trusted(Sf, ds.dset, "test")
            rA, nA = per_user_R_N(ds.gevT, Sz)
            rF, nF = per_user_R_N(ds.gevT, Sf)
            diffs = {"alone_R20": abs(float(rA.mean()) - alone["Recall@20"]),
                     "alone_N20": abs(float(nA.mean()) - alone["NDCG@20"]),
                     "fused_R20": abs(float(rF.mean()) - fused["Recall@20"]),
                     "fused_N20": abs(float(nF.mean()) - fused["NDCG@20"])}
            mx = max(diffs.values())
            check = {"abs_diff_mean_peruser_vs_evalS_trusted": diffs, "max": mx, "tol_ok": EVAL_TOL_OK,
                     "tol_fail": EVAL_TOL_FAIL,
                     "status": "ok" if mx <= EVAL_TOL_OK else ("flag" if mx <= EVAL_TOL_FAIL else "fail")}
            npz_path = Path(npz_path)
            if npz_path.exists():
                preserve(npz_path)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, users=ds.gevT.users.cpu().numpy(), R20_alone=rA, N20_alone=nA,
                                R20_fused=rF, N20_fused=nF, gamma=np.float64(g))
            out["test"] = {"alone": alone, "fused": fused, "gamma": g, "n_users": int(rA.size),
                           "peruser_npz": str(npz_path), "check": check}
            Sf = None
        del base, Sz
        torch.cuda.empty_cache()
    out["eval_s"] = time.time() - t0
    out["eval_peak_gb"] = gpu_gb()
    ctx.max_peak_gb = max(ctx.max_peak_gb, out["eval_peak_gb"])
    return out


def eval_run(ds, arm, hp, rid, model, tested, stage, ctx, rule="ext"):
    P = ctx.P
    mode = "test" if tested else "val"
    ep_ = P.ev(ds.name, rid, mode, rule)
    rec = {"kind": "eval", "script": SCRIPT, "version": VERSION, "stage": stage, "dataset": ds.name, "arm": arm,
           "run_id": rid, "mode": mode, "base_tag": ctx.args.base_tag,
           "base": {"selected": ds.base_info.get("selected"), "source": ds.base_info.get("source")},
           "gamma_rule": {"name": rule, "grid": GAMMA_GRID,
                          "extension_if_5_wins": GAMMA_EXT if rule == "ext" else "none (G7 pre-registered grid)",
                          "criterion": "validation Recall@20"},
           "test_policy": "evaluated" if tested else "not evaluated (grid point; test reserved for selected configs)",
           "status": "running", "started_utc": utc_now()}
    write_json(ep_, rec)
    try:
        model.eval()
        ev = evaluate_scores(ds, score_fn_for(arm, model, ds, ctx.args), tested, P.npz(ds.name, rid, rule), ctx,
                             rule=rule)
        rec.update(ev)
        if ev["val"].get("edge_hit"):
            ctx.log(f"GAMMA EDGE HIT {ds.name} {rid} ({rule} rule): gamma={ev['val']['retuned']:g} is the largest "
                    f"value evaluated; not extended (flagged in the summary)")
        if tested and ev["test"]["check"]["status"] == "fail":
            rec["status"] = "failed"
            rec["error"] = "per-user metrics disagree with evalS_trusted beyond tol_fail (evaluation bug)"
            write_json(ep_, rec)
            raise RuntimeError(rec["error"])
        if tested and ev["test"]["check"]["status"] == "flag":
            ctx.log(f"WARNING {ds.name} {rid}: per-user vs evalS_trusted difference {ev['test']['check']['max']:.2e}")
        rec["status"] = "complete"
        rec["finished_utc"] = utc_now()
        write_json(ep_, rec)
        return rec
    except Exception as e:                                             # recorded, never silently dropped
        if rec.get("status") != "failed":
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        write_json(ep_, rec)
        ctx.n_fail += 1
        ctx.log(f"[{stage}] EVAL FAILED {ds.name} {rid}: {rec['error']}")
        if ctx.args.fail_fast:
            raise
        return rec


def train_run(ds, arm, hp_search, hp, seed, rid, stage, ctx, cap_override=None):
    args, P = ctx.args, ctx.P
    tp, cp = P.train(ds.name, rid), P.ckptp(ds.name, rid)
    rl = Log(P.runlog(ds.name, rid))
    rec = {"kind": "train", "script": SCRIPT, "version": VERSION, "stage": stage, "dataset": ds.name, "arm": arm,
           "arm_label": ARMS[arm]["label"], "config_key": cfg_key(arm, hp_search), "hp_search": hp_search, "hp": hp,
           "seed": int(seed),
           "seeding": {"python_random": int(seed), "numpy": int(seed), "torch_cpu": int(seed), "torch_cuda_all": int(seed),
                       "how": "src.utils.seed.set_seed(seed) at run start, before content init / model init "
                              "(and before build_lists for the cap-60 replica)"},
           "deterministic_algorithms": bool(args.deterministic), "device": str(DEV),
           "gpu_name": torch.cuda.get_device_name(DEV), "torch": torch.__version__, "numpy": np.__version__,
           "microbatch_budgets": {"tf_tok_budget": args.tf_tok_budget, "tf_attn_budget": args.tf_attn_budget},
           "cap_override": cap_override, "smoke": bool(args.smoke), "status": "running", "started_utc": utc_now(),
           "ckpt": str(cp), "run_log": str(rl.path)}
    write_json(tp, rec)
    rl(f"=== {stage} {ds.name} {rid} hp={json.dumps(hp, sort_keys=True)}")
    torch.cuda.reset_peak_memory_stats(DEV)
    t0 = time.time()
    try:
        if ARMS[arm]["kind"] == "masked":
            model, info = train_masked(ds, arm, hp, seed, ctx, rl, cap_override)
        else:
            model, info = train_vae(ds, hp, seed, ctx, rl)
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, cp)
        info["train_s"] = time.time() - t0
        info["peak_gb"] = gpu_gb()
        ctx.max_peak_gb = max(ctx.max_peak_gb, info["peak_gb"])
        rec["train"] = info
        rec["status"] = "trained"
        rec["finished_utc"] = utc_now()
        write_json(tp, rec)
        return rec, model
    except Exception as e:                                             # recorded as the result of this configuration
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        rec["train_s"] = time.time() - t0
        write_json(tp, rec)
        rl(f"FAILED: {rec['error']}")
        ctx.n_fail += 1
        ctx.log(f"[{stage}] TRAIN FAILED {ds.name} {rid}: {rec['error']}")
        if args.fail_fast:
            raise
        torch.cuda.empty_cache()
        return rec, None


def load_model(arm, hp, ds, cp):
    m = build_model(arm, hp, ds.n_items, None).to(DEV)
    m.load_state_dict(torch.load(cp, map_location=DEV))
    m.eval()
    return m


def one_line(stage, ds, arm, hp_search, seed, tr, ev):
    t = (tr or {}).get("train") or {}
    s = (f"[{stage}] {ds} {arm} {search_tag(hp_search)} s{seed} | best_ep={t.get('best_ep')} "
         f"last_ep={t.get('last_ep')} cap={'HIT' if t.get('hit_cap') else 'no'}")
    if ev and ev.get("val"):
        v = ev["val"]
        s += f" | val R20 alone={v['alone']['R20']:.4f} fused(g={v['used']:g})={v['fused']['R20']:.4f}"
    if ev and ev.get("test"):
        a, f = ev["test"]["alone"], ev["test"]["fused"]
        s += (f" | test R20 {a['Recall@20']:.4f}/{f['Recall@20']:.4f} N20 {a['NDCG@20']:.4f}/{f['NDCG@20']:.4f}"
              f" (alone/fused)")
    s += f" | train {t.get('train_s', 0.0):.0f}s {t.get('peak_gb', 0.0):.1f}GB"
    if ev:
        s += f" | eval {ev.get('eval_s', 0.0):.0f}s {ev.get('eval_peak_gb', 0.0):.1f}GB [{ev.get('status')}]"
    return s


def run_config(ds, arm, hp_search, seed, tested, stage, ctx, cap_override=None, rule="ext"):
    """Train (unless a finished record exists) and evaluate one (arm, config, seed) under a gamma rule.
    Returns (train_rec, eval_rec)."""
    args, P, log = ctx.args, ctx.P, ctx.log
    hp = full_hp(arm, hp_search, args, ds.L, cap_override)
    rid = run_id_for(arm, hp_search, hp, seed)
    tp, cp = P.train(ds.name, rid), P.ckptp(ds.name, rid)
    tr = load_json(tp)
    model = None
    if tr is not None and tr.get("status") == "failed" and not args.retry_failed:
        log(f"[{stage}] {ds.name} {rid}: kept earlier FAILED result ({tr.get('error', '')[:120]}); "
            f"--retry_failed reruns it and keeps the failed record")
        return tr, None
    if tr is None or tr.get("status") != "trained" or not cp.is_file():
        if args.no_train:                                   # e.g. a fused-selected config that changed with the base
            ctx.n_skipped += 1
            log(f"[{stage}] SKIPPED {ds.name} {rid}: --stage eval / --no_train and no finished training record "
                f"(reported as 'not_run'; run the training stage for it)")
            return tr or {"status": "not_run", "run_id": rid}, None
        if tr is not None:
            preserve(tp)
        if cp.exists():
            preserve(cp)
        tr, model = train_run(ds, arm, hp_search, hp, seed, rid, stage, ctx, cap_override)
        if tr.get("status") != "trained":
            return tr, None
    ev = None
    ev_test = load_json(P.ev(ds.name, rid, "test", rule))
    if ev_test is not None and ev_test.get("status") == "complete":
        ev = ev_test                                                   # a test record also holds the validation part
    elif not tested:
        ev_val = load_json(P.ev(ds.name, rid, "val", rule))
        if ev_val is not None and ev_val.get("status") == "complete":
            ev = ev_val
    if ev is None:
        mode = "test" if tested else "val"
        if P.ev(ds.name, rid, mode, rule).exists():
            preserve(P.ev(ds.name, rid, mode, rule))
        if model is None:
            model = load_model(arm, hp, ds, cp)
        ev = eval_run(ds, arm, hp, rid, model, tested, stage, ctx, rule)
    del model
    torch.cuda.empty_cache()
    log(one_line(stage, ds.name, arm, hp_search, seed, tr, ev))
    return tr, ev


def ref_force(args, ds_name, seed):
    """(forced gamma or None, deployed JSON or None, fallback reason or None). The deployed per-seed gamma is forced
    only for the deployed base. With the deployed base but no usable deployed JSON, gamma is re-tuned: that FALLBACK is
    recorded in the reference record and flagged by the summary."""
    dep = load_json(scope_json_path(args, ds_name, seed))
    table1_base = (args.base_tag == "table1") and not args.base_npy
    if not table1_base:
        return None, dep, None
    if dep is None:
        return None, None, f"deployed JSON {scope_json_path(args, ds_name, seed)} missing"
    if "gamma" not in dep:
        return None, dep, f"deployed JSON {scope_json_path(args, ds_name, seed)} has no 'gamma'"
    return float(dep["gamma"]), dep, None


def ref_sfx(force, rule):
    """A forced (deployed) gamma makes the reference rule-independent -> one record; otherwise one record per rule."""
    return "" if (force is not None or rule == "ext") else RULE_SFX[rule]


def scope_ref(ds, seed, ctx, rule="ext"):
    """Deployed SCOPE checkpoint (explicit path, never find_ckpt): head alone + SCOPE-v1 with the deployed gamma."""
    args, P, log = ctx.args, ctx.P, ctx.log
    force, dep, fallback = ref_force(args, ds.name, seed)
    sfx = ref_sfx(force, rule)
    rp = P.ref(ds.name, seed, sfx)
    rec = load_json(rp)
    if rec is not None and rec.get("status") == "complete":
        return rec
    if rec is not None:
        preserve(rp)
    ck, js = scope_ckpt_path(args, ds.name, seed), scope_json_path(args, ds.name, seed)
    if not ck.is_file():
        raise FileNotFoundError(f"deployed SCOPE checkpoint missing: {ck}")
    table1_base = (args.base_tag == "table1") and not args.base_npy
    if fallback:
        log(f"[{ds.name}] WARNING gamma FALLBACK for the SCOPE reference s{seed}: {fallback}; gamma is re-tuned on "
            f"validation ({rule} rule) instead of the deployed per-seed gamma (flagged in the summary)")
    rec = {"kind": "scope_ref", "script": SCRIPT, "version": VERSION, "dataset": ds.name, "seed": int(seed),
           "ckpt": str(ck), "ckpt_sha256": sha256_file(ck), "deployed_json": str(js) if dep else None,
           "deployed_best_ep": dep.get("best_ep") if dep else None, "base_tag": args.base_tag,
           "gamma_rule": "forced" if force is not None else rule,
           "gamma_fallback": bool(fallback), "gamma_fallback_reason": fallback,
           "gamma_policy": ("deployed per-seed gamma from the checkpoint's JSON (= the main table); the "
                            "re-tuned gamma is recorded as a check") if force is not None else
                           (f"FALLBACK: validation-tuned ({rule} rule) because {fallback}" if fallback else
                            f"validation-tuned on the current base ({rule} rule, same rule as every competitor)"),
           "status": "running", "started_utc": utc_now()}
    write_json(rp, rec)
    model = SC.SCOPE(ds.n_items, 256).to(DEV)
    model.load_state_dict(torch.load(ck, map_location=DEV))
    model.eval()
    # With a forced (deployed) gamma the record is shared by both rules; its re-tuned check gamma then always uses
    # scope.train's own grid (no extension), so the record does not depend on which stage wrote it first.
    ev = evaluate_scores(ds, lambda: model.score_all(ds.R, ds.degf), True, P.ref_npz(ds.name, seed, sfx), ctx,
                         force_gamma=force, rule="fixed" if force is not None else rule)
    del model
    torch.cuda.empty_cache()
    rec.update(ev)
    if dep is not None and table1_base and all(k in dep for k in ("scope_pure", "fused")):
        t = ev["test"]
        pairs = {"alone_R20": (dep["scope_pure"]["Recall@20"], t["alone"]["Recall@20"]),
                 "alone_N20": (dep["scope_pure"]["NDCG@20"], t["alone"]["NDCG@20"]),
                 "fused_R20": (dep["fused"]["Recall@20"], t["fused"]["Recall@20"]),
                 "fused_N20": (dep["fused"]["NDCG@20"], t["fused"]["NDCG@20"])}
        rep = {k: {"deployed": a, "reproduced": b, "abs_diff": abs(a - b)} for k, (a, b) in pairs.items()}
        ok = max(v["abs_diff"] for v in rep.values()) <= REPRO_TOL
        g_dep = float(dep["gamma"]) if "gamma" in dep else None
        rec["repro"] = {"cells": rep, "gamma_deployed": g_dep, "gamma_retuned": ev["val"]["retuned"],
                        "gamma_match": g_dep == ev["val"]["retuned"], "ok": bool(ok), "tol": REPRO_TOL}
        if not ok or not rec["repro"]["gamma_match"]:
            log(f"[{ds.name}] WARNING scope_ref s{seed} does not reproduce the deployed JSON: {rec['repro']}")
    rec["status"] = "complete"
    rec["finished_utc"] = utc_now()
    write_json(rp, rec)
    t = ev["test"]
    log(f"[ref] {ds.name} SCOPE s{seed} gamma={ev['val']['used']:g} ({rec['gamma_rule']}) | test R20 "
        f"{t['alone']['Recall@20']:.4f}/{t['fused']['Recall@20']:.4f} N20 {t['alone']['NDCG@20']:.4f}/"
        f"{t['fused']['NDCG@20']:.4f} (head/SCOPE-v1) repro_ok={rec.get('repro', {}).get('ok')}")
    return rec


# ============================================================================ stages (GPU)
def stage_ref(ds, ctx, rule="ext"):
    for s in ctx.args.seeds:
        scope_ref(ds, s, ctx, rule)


def stage_ref_both(ds, ctx):
    for rule in GAMMA_RULES:
        stage_ref(ds, ctx, rule)


def stage_g7(ds, ctx):
    stage_ref(ds, ctx, "fixed")                                     # G7: pre-registered gamma grid, no extension
    for arm, hp_s in g7_configs():
        for s in ctx.args.seeds:
            run_config(ds, arm, hp_s, s, True, "G7", ctx, rule="fixed")


def stage_grid(ds, ctx):
    for arm in ctx.args.arms:
        for hp_s in g13_grid(arm, ctx.args.smoke):
            run_config(ds, arm, hp_s, ctx.args.grid_seed, False, "G13-grid", ctx)


def select_config(ds_name, L, arm, ctx):
    args, P = ctx.args, ctx.P
    rows = []
    for gi, hp_s in enumerate(g13_grid(arm, args.smoke)):
        rid = run_id_for(arm, hp_s, full_hp(arm, hp_s, args, L), args.grid_seed)
        tr = load_json(P.train(ds_name, rid))
        st = tr.get("status") if tr else "not_run"
        v = tr["train"]["best_val_R20"] if (tr is not None and st == "trained") else None
        rows.append({"grid_index": gi, "hp_search": hp_s, "config_key": cfg_key(arm, hp_s), "run_id": rid,
                     "status": st, "best_val_R20": v})
    missing = [r["run_id"] for r in rows if r["status"] not in ("trained", "failed")]
    if missing:
        raise RuntimeError(f"{ds_name}/{arm}: grid incomplete, refusing to select: {missing}")
    cands = [r for r in rows if r["status"] == "trained" and r["best_val_R20"] is not None
             and math.isfinite(r["best_val_R20"])]
    if not cands:
        raise RuntimeError(f"{ds_name}/{arm}: every grid point failed")
    best = cands[0]
    for r in cands[1:]:
        if r["best_val_R20"] > best["best_val_R20"]:
            best = r
    return {"dataset": ds_name, "arm": arm, "grid_seed": args.grid_seed, "rule": SELECTION_RULE, "grid": rows,
            "selected": best}


def grid_eval_val(P, ds_name, rid):
    """Validation part of a grid point's G13-rule evaluation (the test record also holds it). Never reads test."""
    evt, evv = load_json(P.ev(ds_name, rid, "test")), load_json(P.ev(ds_name, rid, "val"))
    for ev in (evt, evv):
        if ev is not None and ev.get("status") == "complete" and ev.get("val"):
            return "complete", ev["val"]
    st = [e.get("status") for e in (evt, evv) if e is not None]
    return ("failed" if "failed" in st else ("running" if st else "not_run")), None


def select_config_fused(ds_name, L, arm, ctx):
    """Additional panel: best FUSED validation R@20 (each grid point at its own validation-tuned gamma, G13 rule)."""
    args, P = ctx.args, ctx.P
    rows = []
    for gi, hp_s in enumerate(g13_grid(arm, args.smoke)):
        rid = run_id_for(arm, hp_s, full_hp(arm, hp_s, args, L), args.grid_seed)
        tr = load_json(P.train(ds_name, rid))
        st = tr.get("status") if tr else "not_run"
        est, v = grid_eval_val(P, ds_name, rid) if st == "trained" else (None, None)
        rows.append({"grid_index": gi, "hp_search": hp_s, "config_key": cfg_key(arm, hp_s), "run_id": rid,
                     "train_status": st, "eval_status": est,
                     "fused_val_R20": v["fused"]["R20"] if v else None, "fused_val_gamma": v["used"] if v else None,
                     "alone_val_R20": v["alone"]["R20"] if v else None})
    missing = [r["run_id"] for r in rows if r["train_status"] not in ("trained", "failed")
               or (r["train_status"] == "trained" and r["eval_status"] not in ("complete", "failed"))]
    if missing:
        raise RuntimeError(f"{ds_name}/{arm}: grid training/evaluation incomplete, refusing the fused selection: "
                           f"{missing}")
    cands = [r for r in rows if r["fused_val_R20"] is not None and math.isfinite(r["fused_val_R20"])]
    if not cands:
        raise RuntimeError(f"{ds_name}/{arm}: no grid point has a fused validation score")
    best = cands[0]
    for r in cands[1:]:
        if r["fused_val_R20"] > best["fused_val_R20"]:
            best = r
    return {"dataset": ds_name, "arm": arm, "grid_seed": args.grid_seed, "base_tag": args.base_tag,
            "rule": SELECTION_RULE_FUSED, "grid": rows, "selected": best}


def freeze_selection(path, sel, what):
    """First selection is written and frozen; a later different selection is an error (never silently replaced)."""
    old = load_json(path)
    if old is None:
        sel["created_utc"] = utc_now()
        write_json(path, sel)
        return sel
    if old["selected"]["run_id"] != sel["selected"]["run_id"]:
        raise RuntimeError(f"{what}: selection changed since {path} was written "
                           f"({old['selected']['run_id']} -> {sel['selected']['run_id']})")
    return old


def stage_seeds(ds, ctx):
    args, P, log = ctx.args, ctx.P, ctx.log
    stage_ref(ds, ctx)
    for arm in args.arms:
        sel = freeze_selection(P.sel(ds.name, arm, args.grid_seed), select_config(ds.name, ds.L, arm, ctx),
                               f"{ds.name}/{arm} (standalone)")
        fsel = freeze_selection(P.sel_fused(ds.name, arm, args.grid_seed),
                                select_config_fused(ds.name, ds.L, arm, ctx), f"{ds.name}/{arm} (fused)")
        same = fsel["selected"]["run_id"] == sel["selected"]["run_id"]
        log(f"[G13-select] {ds.name} {arm}: standalone {sel['selected']['config_key']} "
            f"(val R20 {sel['selected']['best_val_R20']:.4f}); fused {fsel['selected']['config_key']} "
            f"(fused val R20 {fsel['selected']['fused_val_R20']:.4f}, gamma {fsel['selected']['fused_val_gamma']:g})"
            f" -> {'same config' if same else 'DIFFERENT config: its seeds are run too'}; {len(sel['grid'])} grid points")
        hps = [sel["selected"]["hp_search"]] + ([] if same else [fsel["selected"]["hp_search"]])
        for hp_s in hps:
            for s in args.seeds:
                run_config(ds, arm, hp_s, s, True, "G13-seeds", ctx)


def stage_sanity(ds, ctx):
    stage_ref(ds, ctx)
    if not ctx.args.no_sanity_cap60:                                # default: exact copy of the deployed scope.train
        for s in ctx.args.seeds:
            run_config(ds, "scope_mlp", SANITY_HP, s, True, "G13-sanity-cap60", ctx, cap_override=60)
    for s in ctx.args.seeds:                                        # full lists: encoder-only comparison (a side)
        run_config(ds, "scope_mlp", SANITY_HP, s, True, "G13-sanity", ctx)


STAGES = {"ref": [stage_ref_both], "g7": [stage_g7], "grid": [stage_grid], "seeds": [stage_seeds],
          "sanity": [stage_sanity], "all": [stage_g7, stage_grid, stage_seeds, stage_sanity],
          "eval": [stage_g7, stage_grid, stage_seeds, stage_sanity]}


# ============================================================================ summarize (CPU only)
def holm(pvals):
    """Holm step-down adjusted p-values (standard step-down Holm)."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def load_npz(path):
    if path is None or not Path(path).is_file():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def seed_mean(arrs, col):
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None, 0, None
    u0 = arrs[0]["users"]
    for a in arrs[1:]:
        if not np.array_equal(a["users"], u0):
            raise RuntimeError("per-user arrays are not aligned (different test-user order)")
    return np.mean(np.stack([a[col] for a in arrs], 0), 0), len(arrs), u0


def make_entry(meta, a_list, b_list, setting, metric):
    """a = SCOPE side (reference), b = other side; each a list of per-seed npz dicts (None = missing seed).
    The per-user metric is averaged over the seeds FIRST; the bootstrap then resamples users of that average, so the
    training-seed spread is not part of the CI / p (the *_perseed families test each seed separately)."""
    col = f"{metric}_{setting}"
    a, na, ua = seed_mean(a_list, col)
    b, nb, ub = seed_mean(b_list, col)
    if a is not None and b is not None and not np.array_equal(ua, ub):
        raise RuntimeError(f"user order differs between the two sides of {meta}")
    e = dict(meta, setting=setting, metric=metric, n_seeds_a=na, n_seeds_b=nb, n_seeds_expected=len(a_list))
    e["_a"], e["_b"] = a, b
    return e


HOLM_RULE = ("Holm step-down over the EXPECTED family size m_expected (every pre-stated test); a missing test (no "
             "per-user data on one side) or an incomplete test (a seed missing on either side) enters Holm with p=1 "
             "and gets no verdict; verdicts are marked provisional while the family is incomplete")


def build_family(name, definition, entries, B, alpha):
    tests = []
    for e in entries:
        t = {k: v for k, v in e.items() if not k.startswith("_")}
        a, b = e["_a"], e["_b"]
        if a is None or b is None:
            t["status"] = "missing"
            tests.append(t)
            continue
        bs = paired_bootstrap(a, b, B=B)                       # harness: mean(a-b), 95% CI, two-sided p, n
        diff = a - b
        n = diff.size
        se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        bm = float(b.mean())
        complete = e["n_seeds_a"] == e["n_seeds_expected"] and e["n_seeds_b"] == e["n_seeds_expected"]
        t.update(bs)
        t.update({"mean_a": float(a.mean()), "mean_b": bm, "se": se, "mde80": Z_MDE80 * se,
                  "rel_delta_pct": 100.0 * bs["mean_delta"] / bm if bm else float("nan"),
                  "seeds_complete": complete, "status": "ok" if complete else "incomplete"})
        tests.append(t)
    pv = [t["p_two_sided"] if t["status"] == "ok" else 1.0 for t in tests]
    padj = holm(pv) if tests else []
    fam_complete = bool(tests) and all(t["status"] == "ok" for t in tests)
    for t, pa in zip(tests, padj):
        if t["status"] == "ok":
            t["p_holm"] = pa
            t["verdict"] = ("a>b" if t["mean_delta"] > 0 else "a<b") if pa < alpha else "n.s."
            t["verdict_provisional"] = not fam_complete
        else:
            t["p_holm"] = None
            t["verdict"] = t["status"]                         # 'missing' / 'incomplete': never a win/tie/loss
    cnt = {s: sum(t["status"] == s for t in tests) for s in ("ok", "incomplete", "missing")}
    return {"name": name, "definition": definition, "alpha": alpha, "B": B, "holm_rule": HOLM_RULE,
            "m_expected": len(entries), "m_holm": len(entries), "n_ok": cnt["ok"], "n_incomplete": cnt["incomplete"],
            "n_missing": cnt["missing"], "complete": fam_complete, "tests": tests}


def win_tie_loss(family):
    """Counts over the tests with a verdict; incomplete / missing tests are counted separately, never as W/T/L."""
    out = {"win": 0, "tie": 0, "loss": 0, "incomplete": 0, "missing": 0, "m_expected": family["m_expected"],
           "family_complete": family["complete"], "by_dataset": {}, "by_arm": {}, "ties": [], "losses": [],
           "incomplete_tests": [], "missing_tests": []}
    names = {"a>b": "win", "a<b": "loss", "n.s.": "tie"}
    for t in family["tests"]:
        v = names.get(t.get("verdict"), t["status"])
        out[v] += 1
        z = {"win": 0, "tie": 0, "loss": 0, "incomplete": 0, "missing": 0}
        out["by_dataset"].setdefault(t["dataset"], dict(z))[v] += 1
        out["by_arm"].setdefault(str(t["arm"]), dict(z))[v] += 1
        cell = {k: t.get(k) for k in ("dataset", "arm", "setting", "metric", "mean_delta", "ci95", "p_two_sided",
                                      "p_holm", "n_seeds_a", "n_seeds_b", "n_seeds_expected")}
        if v in ("tie", "loss"):
            out["ties" if v == "tie" else "losses"].append(cell)
        elif v in ("incomplete", "missing"):
            out[f"{v}_tests"].append(cell)
    return out


def run_summary(P, ds, rid, rule="ext", tested=True, seen=None):
    """One run under one gamma rule. A tested run is reported from its TEST record only: if that record is missing,
    running or failed, test is None and eval_status says so - the validation record is never substituted for it."""
    tr = load_json(P.train(ds, rid))
    tpath, vpath = P.ev(ds, rid, "test", rule), P.ev(ds, rid, "val", rule)
    evt, evv = load_json(tpath), load_json(vpath)
    if seen is not None:
        seen.update({str(tpath), str(vpath)})
    if tested:
        ev, evp = evt, tpath
    elif evt and evt.get("status") == "complete":
        ev, evp = evt, tpath                                          # a test record also holds the validation part
    elif evv and evv.get("status") == "complete":
        ev, evp = evv, vpath
    else:
        ev, evp = (evv, vpath) if evv else (evt, tpath)
    out = {"run_id": rid, "gamma_rule": rule, "tested": bool(tested),
           "train_status": tr.get("status") if tr else "not_run",
           "eval_status": ev.get("status") if ev else "not_run", "eval_record": str(evp) if ev else None,
           "test_eval_status": evt.get("status") if evt else "not_run"}
    if evt and evt.get("status") != "complete":
        out["test_eval_error"] = evt.get("error") or f"test record status '{evt.get('status')}'"
    if tr:
        out.update({"version": tr.get("version"), "arm": tr.get("arm"), "stage": tr.get("stage"),
                    "seed": tr.get("seed"), "hp_search": tr.get("hp_search"), "hp": tr.get("hp"),
                    "config_key": tr.get("config_key"), "cap_override": tr.get("cap_override")})
        if tr.get("error"):
            out["train_error"] = tr["error"]
        t = tr.get("train") or {}
        for k in ("best_ep", "last_ep", "hit_cap", "early_stopped", "max_epochs", "best_val_R20", "train_s", "peak_gb",
                  "steps", "oom_events", "init_row_l2_mean", "beta_at_best", "tau_best", "list_width_train"):
            if k in t:
                out[k] = t[k]
    if ev and ev.get("error"):
        out["eval_error"] = ev["error"]
    if ev and ev.get("status") == "complete" and ev.get("val"):
        v = ev["val"]
        eh = v.get("edge_hit")
        if eh is None:                                                # records written before r2
            eh = float(v["retuned"]) == (GAMMA_EXT[-1] if v.get("extended") else GAMMA_GRID[-1])
        out.update({"gamma": v["used"], "gamma_retuned": v["retuned"], "gamma_extended": v.get("extended"),
                    "gamma_edge_hit": bool(eh), "gamma_rule_recorded": v.get("rule", "ext"),
                    "val": {"alone": v["alone"], "fused": v["fused"]}, "val_gamma_grid": v["grid"]})
    if ev and ev.get("status") == "complete" and ev.get("test"):
        te = ev["test"]
        out["test"] = {"alone": {"R20": te["alone"]["Recall@20"], "N20": te["alone"]["NDCG@20"]},
                       "fused": {"R20": te["fused"]["Recall@20"], "N20": te["fused"]["NDCG@20"]}}
        out["eval_check"] = te["check"]["status"]
        out["eval_check_max"] = te["check"]["max"]
        out["n_test_users"] = te.get("n_users")
    else:
        out["test"] = None
    if ev:
        out["eval_s"] = ev.get("eval_s")
        out["eval_peak_gb"] = ev.get("eval_peak_gb")
    return out


def ref_record(P, ds, seed, rule):
    """(path, record) of the deployed-SCOPE reference for a gamma rule: the rule-independent record when its gamma was
    forced (deployed), else the rule's own record."""
    unsfx = P.ref(ds, seed, "")
    if rule == "fixed":
        rp = P.ref(ds, seed, RULE_SFX["fixed"])
        rec = load_json(rp)
        if rec is not None:
            return rp, rec
        rec = load_json(unsfx)
        forced = rec is not None and (rec.get("gamma_rule") == "forced" or (rec.get("val") or {}).get("forced"))
        return (unsfx, rec) if forced else (rp, None)
    return unsfx, load_json(unsfx)


def ref_summary(P, ds, seed, rule="ext", seen=None):
    rp, rec = ref_record(P, ds, seed, rule)
    if seen is not None:
        seen.add(str(rp))
    if rec is None:
        return {"status": "not_run", "test": None, "record": str(rp)}
    out = {"status": rec.get("status"), "record": str(rp), "npz": str(Path(str(rp)[:-len(".json")] + "__test_peruser.npz")),
           "version": rec.get("version"), "ckpt": rec.get("ckpt"), "ckpt_sha256": rec.get("ckpt_sha256"),
           "best_ep": rec.get("deployed_best_ep"), "repro": rec.get("repro"), "gamma_policy": rec.get("gamma_policy"),
           "gamma_rule": rec.get("gamma_rule", "forced" if (rec.get("val") or {}).get("forced") else "ext"),
           "gamma_fallback": rec.get("gamma_fallback", False), "gamma_fallback_reason": rec.get("gamma_fallback_reason"),
           "test": None}
    if rec.get("error"):
        out["error"] = rec["error"]
    if rec.get("status") == "complete" and rec.get("val"):
        v = rec["val"]
        out.update({"gamma": v["used"], "gamma_retuned": v["retuned"], "gamma_edge_hit": v.get("edge_hit"),
                    "val": {"alone": v["alone"], "fused": v["fused"]}})
    if rec.get("status") == "complete" and rec.get("test"):
        te = rec["test"]
        out["test"] = {"alone": {"R20": te["alone"]["Recall@20"], "N20": te["alone"]["NDCG@20"]},
                       "fused": {"R20": te["fused"]["Recall@20"], "N20": te["fused"]["NDCG@20"]}}
        out["eval_check"] = te["check"]["status"]
    return out


def eval_record_brief(path):
    rec = load_json(path) or {}
    v, te = rec.get("val") or {}, rec.get("test") or {}
    return {"record": str(path), "status": rec.get("status"), "version": rec.get("version"), "stage": rec.get("stage"),
            "run_id": rec.get("run_id"), "mode": rec.get("mode"),
            "gamma_rule": (rec.get("gamma_rule") or {}).get("name", "ext") if isinstance(rec.get("gamma_rule"), dict)
            else rec.get("gamma_rule"), "gamma": v.get("used"),
            "val_fused_R20": (v.get("fused") or {}).get("R20"),
            "test_fused_R20": (te.get("fused") or {}).get("Recall@20"),
            "test_alone_R20": (te.get("alone") or {}).get("Recall@20")}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _test_val(rs, setting, metric):
    te = (rs or {}).get("test")
    return te[setting][metric] if te else None


def stage_summarize(ctx):
    args, P, log = ctx.args, ctx.P, ctx.log
    B, alpha = int(args.bootstrap_B), float(args.alpha)
    outdir = P.eval / f"summary_{utc_ts()}"
    outdir.mkdir(parents=True, exist_ok=False)
    spath = outdir / "summary.json"
    seeds = [int(s) for s in args.seeds]
    S = {"script": SCRIPT, "version": VERSION, "created_utc": utc_now(), "base_tag": args.base_tag,
         "datasets": list(args.datasets), "seeds": seeds, "grid_seed": int(args.grid_seed), "smoke": bool(args.smoke),
         "alpha": alpha,
         "bootstrap": {"impl": "scope/harness.py::paired_bootstrap", "B": B, "rng_seed": 0,
                       "sided": "two", "unit": "per-user Recall@20 / NDCG@20 on the test users (GPUEval order)",
                       "seed_handling": ("seed-averaged families: each user's metric is first averaged over the seeds "
                                         "of each side, then users are resampled; the training-seed spread is NOT in "
                                         "these CIs / p-values. The *_perseed families (one test per seed, own Holm "
                                         "family) are the seed-level check.")},
         "holm": "step-down Holm per family (step-down Holm), computed here; "
                 + HOLM_RULE,
         "gamma": {"grid": GAMMA_GRID, "extension": GAMMA_EXT,
                   "rules": {"ext": "G13: grid, + extension if 5 wins on validation",
                             "fixed": "G7: pre-registered grid only, never extended; gamma=5 = edge hit"},
                   "scope_reference": "deployed per-seed gamma (forced) with the deployed base; otherwise re-tuned "
                                      "with the comparison's rule (fallbacks flagged)"},
         "budgets": {"max_epochs": int(args.max_epochs), "eval_every": int(args.eval_every),
                     "patience_checks": int(args.patience), "bs": int(args.bs), "vae_max_epochs": int(args.vae_max_epochs),
                     "vae_anneal_steps": int(args.vae_anneal_steps), "vae_bs": 500},
         "arms": ARMS, "g13_arms": G13_ARMS, "sanity_hp": SANITY_HP,
         "g7_configs": [{"arm": a, "hp_search": h, "config_key": cfg_key(a, h)} for a, h in g7_configs()],
         "g13_grids": {a: g13_grid(a, args.smoke) for a in G13_ARMS},
         "selection_rule": SELECTION_RULE, "selection_rule_fused": SELECTION_RULE_FUSED, "mde_preregistered": MDE_PLAN,
         "dataset_info": {}, "base": {}, "scope_ref": {}, "scope_ref_g7": {},
         "g7": {"runs": {}, "family": None, "family_perseed": None, "gate": {}},
         "g13": {"grid": {}, "selected": {}, "runs": {}, "runs_fused_sel": {}, "family": None,
                 "family_fused_sel": None, "family_perseed": None, "family_encoder_only": None,
                 "win_tie_loss": None, "win_tie_loss_fused_sel": None, "win_tie_loss_encoder_only": None,
                 "narrowing": None,
                 "sanity": {"users_over_60_items": {}, "runs_cap60": {}, "runs_full": {}, "check": {},
                            "family": None, "win_tie_loss": None}},
         "other_runs": {}, "other_eval_records": {}, "stale_records": {}, "versions_of_listed_runs": {}, "flags": []}
    flags = S["flags"]
    pu = {}                                               # (kind, ds, key, seed) -> per-user npz dict

    def flag(msg):
        flags.append(msg)
        log(f"FLAG: {msg}")

    for ds in args.datasets:
        info = load_json(P.dsinfo / f"{ds}.json")
        S["dataset_info"][ds] = info
        if info is None:
            flag(f"{ds}: no dataset record (GPU stages never ran for it) - every cell for {ds} is 'not run'")
            continue
        L = int(info["L"])
        S["base"][ds] = load_json(P.base_rec(ds))
        seen = set()                                      # eval records accounted for (the rest -> other_eval_records)
        S["scope_ref"][ds], S["scope_ref_g7"][ds] = {}, {}
        for s in seeds:
            for rule, key, kind in (("ext", "scope_ref", "ref"), ("fixed", "scope_ref_g7", "ref7")):
                rsum = ref_summary(P, ds, s, rule, seen)
                S[key][ds][str(s)] = rsum
                if rsum["status"] == "complete":
                    pu[(kind, ds, None, s)] = load_npz(rsum["npz"])
                    if rule == "ext":
                        rp = rsum.get("repro")
                        if rp is not None and (not rp.get("ok") or not rp.get("gamma_match")):
                            flag(f"{ds} s{s}: deployed SCOPE reference does not reproduce its JSON ({rp})")
                    if rsum.get("gamma_fallback"):
                        flag(f"{ds} s{s} ({rule} rule): gamma FALLBACK for the SCOPE reference - "
                             f"{rsum.get('gamma_fallback_reason')}; gamma re-tuned instead of the deployed one")
                    if rsum.get("gamma_edge_hit") and rsum.get("gamma_rule") != "forced":
                        flag(f"{ds} s{s} ({rule} rule): SCOPE reference gamma at the grid edge")
                elif rsum["status"] != "not_run":
                    flag(f"{ds} s{s} ({rule} rule): SCOPE reference record status '{rsum['status']}' "
                         f"({rsum.get('error')})")
        referenced = set()
        # ---- G7 (pre-registered gamma grid, no extension)
        S["g7"]["runs"][ds] = {}
        for arm, hp_s in g7_configs():
            key = cfg_key(arm, hp_s)
            S["g7"]["runs"][ds][key] = {}
            for s in seeds:
                rid = run_id_for(arm, hp_s, full_hp(arm, hp_s, args, L), s)
                referenced.add(rid)
                rs = run_summary(P, ds, rid, "fixed", True, seen)
                rs["config_key"] = key
                S["g7"]["runs"][ds][key][str(s)] = rs
                if rs["test"] is not None:
                    pu[("g7", ds, key, s)] = load_npz(P.npz(ds, rid, "fixed"))
        # ---- G13 grid + both selections + seeds
        S["g13"]["grid"][ds], S["g13"]["selected"][ds] = {}, {}
        S["g13"]["runs"][ds], S["g13"]["runs_fused_sel"][ds] = {}, {}
        for arm in G13_ARMS:
            sel = load_json(P.sel(ds, arm, args.grid_seed))
            fsel = load_json(P.sel_fused(ds, arm, args.grid_seed))
            sel_rid = sel["selected"]["run_id"] if sel else None
            fsel_rid = fsel["selected"]["run_id"] if fsel else None
            rows = []
            for gi, hp_s in enumerate(g13_grid(arm, args.smoke)):
                rid = run_id_for(arm, hp_s, full_hp(arm, hp_s, args, L), args.grid_seed)
                referenced.add(rid)
                rs = run_summary(P, ds, rid, "ext", False, seen)
                rs.update({"grid_index": gi, "hp_search": hp_s, "config_key": cfg_key(arm, hp_s),
                           "selected_alone": rid == sel_rid, "selected_fused": rid == fsel_rid})
                rows.append(rs)
            S["g13"]["grid"][ds][arm] = rows
            same = (sel_rid == fsel_rid) if (sel and fsel) else None
            S["g13"]["selected"][ds][arm] = {
                "alone": ({"config_key": sel["selected"]["config_key"], "hp_search": sel["selected"]["hp_search"],
                           "run_id": sel_rid, "best_val_R20": sel["selected"]["best_val_R20"],
                           "created_utc": sel.get("created_utc")} if sel else None),
                "fused": ({"config_key": fsel["selected"]["config_key"], "hp_search": fsel["selected"]["hp_search"],
                           "run_id": fsel_rid, "fused_val_R20": fsel["selected"]["fused_val_R20"],
                           "fused_val_gamma": fsel["selected"]["fused_val_gamma"],
                           "created_utc": fsel.get("created_utc")} if fsel else None),
                "same_config": same}
            S["g13"]["runs"][ds][arm], S["g13"]["runs_fused_sel"][ds][arm] = {}, {}
            for which, sl, dst, kind in (("standalone", sel, S["g13"]["runs"][ds][arm], "g13"),
                                         ("fused", fsel, S["g13"]["runs_fused_sel"][ds][arm], "g13f")):
                if sl is None:
                    flag(f"{ds}/{arm}: no {which} validation selection yet (stage seeds not run) - reported as "
                         f"'not run'")
                    continue
                hp_s = sl["selected"]["hp_search"]
                for s in seeds:
                    rid = run_id_for(arm, hp_s, full_hp(arm, hp_s, args, L), s)
                    referenced.add(rid)
                    rs = run_summary(P, ds, rid, "ext", True, seen)
                    rs["config_key"] = cfg_key(arm, hp_s)
                    dst[str(s)] = rs
                    if rs["test"] is not None:
                        pu[(kind, ds, arm, s)] = load_npz(P.npz(ds, rid))
        # ---- sanity: cap-60 replica of scope.train (default reproduction check) + full lists (encoder-only side)
        S["g13"]["sanity"]["users_over_60_items"][ds] = {
            "users_over_60_items": info.get("users_over_60_items"), "max_train_degree": info.get("max_train_degree"),
            "n_users": info.get("n_users"),
            "note": "in the cap-60 replica (= deployed scope.train) these users train on a random 60-item subset of "
                    "their history while the context size is drawn from their TRUE degree; a row with n_ctx >= 60 has "
                    "no target, adds 0 to the loss sum and still counts in the batch mean (as deployed)"}
        log(f"[{ds}] users with more than 60 train items: {info.get('users_over_60_items')} of {info.get('n_users')} "
            f"(max train degree {info.get('max_train_degree')})")
        S["g13"]["sanity"]["runs_cap60"][ds], S["g13"]["sanity"]["runs_full"][ds] = {}, {}
        for s in seeds:
            for cap, dst, kind in ((60, S["g13"]["sanity"]["runs_cap60"][ds], "cap60"),
                                   (None, S["g13"]["sanity"]["runs_full"][ds], "full")):
                rid = run_id_for("scope_mlp", SANITY_HP, full_hp("scope_mlp", SANITY_HP, args, L, cap), s)
                referenced.add(rid)
                rs = run_summary(P, ds, rid, "ext", True, seen)
                rs["config_key"] = cfg_key("scope_mlp", SANITY_HP) + ("_cap60" if cap else "_fulllists")
                dst[str(s)] = rs
                if rs["test"] is not None:
                    pu[(kind, ds, None, s)] = load_npz(P.npz(ds, rid))
        # ---- anything else on disk (never silently ignored)
        S["other_runs"][ds] = {}
        rdir = P.runs / ds
        if rdir.is_dir():
            for f in sorted(rdir.glob("*.json")):
                if ".stale_" in f.name:
                    continue
                if f.stem not in referenced:
                    st0 = (load_json(f) or {}).get("stage")
                    S["other_runs"][ds][f.stem] = run_summary(P, ds, f.stem, "fixed" if st0 == "G7" else "ext",
                                                              st0 != "G13-grid", seen)
        S["other_eval_records"][ds] = {}
        edir = P.eval / ds
        if edir.is_dir():
            for f in sorted(edir.glob("*.json")):
                if ".stale_" in f.name or f.name == "base_info.json" or str(f) in seen:
                    continue
                S["other_eval_records"][ds][f.name] = eval_record_brief(f)
        S["stale_records"][ds] = sorted(str(p.relative_to(P.out)) for d in (P.runs / ds, P.eval / ds) if d.is_dir()
                                        for p in d.glob("*.stale_*"))

    # ---- flags on every run
    def scan(rs, where):
        ts = rs.get("train_status")
        if ts == "failed":
            flag(f"{where}: training FAILED ({rs.get('train_error')})")
        elif ts not in ("trained", "not_run"):
            flag(f"{where}: training record status '{ts}' (interrupted run?)")
        if rs.get("eval_status") == "failed":
            flag(f"{where}: evaluation FAILED ({rs.get('eval_error')})")
        if rs.get("tested") and ts == "trained" and rs.get("test_eval_status") != "complete":
            flag(f"{where}: TEST evaluation {rs.get('test_eval_status')} ({rs.get('test_eval_error', 'no record')}) - "
                 f"no test numbers; the validation record is NOT used in its place")
        elif (not rs.get("tested")) and rs.get("test_eval_status") not in ("complete", "not_run"):
            flag(f"{where}: test evaluation record status '{rs.get('test_eval_status')}' "
                 f"({rs.get('test_eval_error')})")
        if rs.get("hit_cap"):
            flag(f"{where}: hit its epoch cap ({rs.get('max_epochs')}) without early stopping")
        if rs.get("gamma_extended"):
            flag(f"{where}: gamma=5 won on validation, grid extended to {GAMMA_EXT}")
        if rs.get("gamma_edge_hit"):
            flag(f"{where}: gamma at the edge of the evaluated grid ({rs.get('gamma_retuned')}, "
                 f"{rs.get('gamma_rule')} rule; not extended further)")
        if rs.get("eval_check") in ("flag", "fail"):
            flag(f"{where}: per-user vs evalS_trusted check {rs.get('eval_check')} ({rs.get('eval_check_max')})")
        if rs.get("oom_events"):
            flag(f"{where}: {rs.get('oom_events')} OOM events (micro-batch limits halved; batch size unchanged)")

    vers = S["versions_of_listed_runs"]

    def scan_v(rs, where):
        scan(rs, where)
        if rs.get("version"):
            vers.setdefault(rs["version"], []).append(rs["run_id"])

    for ds, d0 in S["g7"]["runs"].items():
        for key, d in d0.items():
            for s, rs in d.items():
                scan_v(rs, f"G7 {ds} {key} s{s}")
    for ds, d0 in S["g13"]["grid"].items():
        for arm, rows in d0.items():
            for rs in rows:
                scan_v(rs, f"G13-grid {ds} {rs['config_key']}")
    for part, lab in (("runs", "G13"), ("runs_fused_sel", "G13 fused-selected")):
        for ds, d0 in S["g13"][part].items():
            for arm, d in d0.items():
                for s, rs in d.items():
                    scan_v(rs, f"{lab} {ds} {arm} s{s}")
    for part, lab in (("runs_cap60", "sanity cap-60"), ("runs_full", "sanity full-list")):
        for ds, d0 in S["g13"]["sanity"][part].items():
            for s, rs in d0.items():
                scan_v(rs, f"{lab} {ds} s{s}")
    for ds, d0 in S["other_runs"].items():
        for rid, rs in d0.items():
            scan_v(rs, f"other run {ds} {rid}")
    for ds, d0 in S["other_eval_records"].items():
        if d0:
            flag(f"{ds}: {len(d0)} evaluation record(s) not used by any family (listed under other_eval_records)")
    if set(vers) - {VERSION}:
        flag(f"runs written by earlier script versions are listed: "
             f"{ {v: len(r) for v, r in vers.items() if v != VERSION} } (ids under versions_of_listed_runs)")
    write_json(spath, S)

    def fam(a_kind, b_kind, keys, seeds_used, settings=("alone", "fused"), family=""):
        ent = []
        for ds in args.datasets:
            a_list = [pu.get((a_kind, ds, None, s)) for s in seeds_used]
            for key in keys:
                b_list = [pu.get((b_kind, ds, key, s)) for s in seeds_used]
                for setting in settings:
                    for metric in ("R20", "N20"):
                        ent.append(make_entry({"family": family, "dataset": ds, "arm": key, "a": a_kind, "b": b_kind,
                                               "seeds": list(seeds_used)}, a_list, b_list, setting, metric))
        return ent

    # ---- G13 primary family: SCOPE (deployed; head alone / SCOPE-v1) vs each arm, seed-averaged per-user metrics
    log("summarize: G13 primary family (seed-averaged) ...")
    S["g13"]["family"] = build_family(
        "G13 primary: SCOPE (deployed ckpts) minus neighbour (standalone-selected config), seed-averaged per-user "
        "metrics",
        f"{len(G13_ARMS)} arms x 2 settings (head alone vs arm alone; SCOPE-v1 vs arm+base, each gamma validation-tuned, "
        f"G13 rule) x 2 metrics (R@20, N@20) x {len(args.datasets)} datasets = "
        f"{len(G13_ARMS) * 4 * len(args.datasets)} tests (pre-registered); a=SCOPE, b=arm at its STANDALONE-selected "
        f"config in both settings; verdict 'a>b' = SCOPE significantly better after Holm",
        fam("ref", "g13", G13_ARMS, seeds, family="primary"), B, alpha)
    S["g13"]["win_tie_loss"] = win_tie_loss(S["g13"]["family"])
    write_json(spath, S)
    log("summarize: G13 fused-selected panel ...")
    S["g13"]["family_fused_sel"] = build_family(
        "G13 fused-selected panel: SCOPE-v1 (deployed) minus arm+base at the arm's FUSED-selected config",
        f"{len(G13_ARMS)} arms x fused setting x 2 metrics x {len(args.datasets)} datasets = "
        f"{len(G13_ARMS) * 2 * len(args.datasets)} tests; separate Holm family; the arm config is selected on fused "
        f"validation R@20 (= the standalone-selected config where they coincide)",
        fam("ref", "g13f", G13_ARMS, seeds, settings=("fused",), family="fused_sel"), B, alpha)
    S["g13"]["win_tie_loss_fused_sel"] = win_tie_loss(S["g13"]["family_fused_sel"])
    write_json(spath, S)
    if not args.no_perseed_bootstrap:
        ent = []
        for s in seeds:
            ent += fam("ref", "g13", G13_ARMS, [s], family="perseed")
        S["g13"]["family_perseed"] = build_family("G13 per-seed (secondary): SCOPE seed s vs arm seed s",
                                                  "same comparisons as the primary family, one test per seed; "
                                                  "separate Holm family", ent, B, alpha)
        write_json(spath, S)
    log("summarize: G13 encoder-only family (shared-trainer SCOPE head, full lists) ...")
    S["g13"]["family_encoder_only"] = build_family(
        "G13 encoder-only (secondary): shared-trainer SCOPE head with full lists (a) minus each arm (b)",
        f"a = scope_mlp re-run by the shared trainer with full lists (same trainer, lists, budget, content seed and "
        f"gamma rule as the arms), b = arm at its standalone-selected config; {len(G13_ARMS)} arms x 2 settings x 2 "
        f"metrics x {len(args.datasets)} datasets; separate Holm family. Only the encoder (and CBOW-random's init / "
        f"the tuned hyper-parameters / Mult-VAE's objective) differ",
        fam("full", "g13", G13_ARMS, seeds, family="encoder_only"), B, alpha)
    S["g13"]["win_tie_loss_encoder_only"] = win_tie_loss(S["g13"]["family_encoder_only"])
    wtl, wtlf = S["g13"]["win_tie_loss"], S["g13"]["win_tie_loss_fused_sel"]
    loss = wtl["loss"] + wtlf["loss"]
    tie = wtl["tie"] + wtlf["tie"]
    complete = wtl["family_complete"] and wtlf["family_complete"]
    if loss > 0:
        c1 = ("loss observed: the head does not match its neighbours in every cell (pre-registered narrowing rule); "
              "every loss is printed")
    elif tie > 0:
        c1 = "ties and no loss: the head matches or exceeds its nearest masked neighbours"
    elif not complete:
        c1 = "no tie or loss among the tests with a verdict, but the family is incomplete: no verdict yet"
    else:
        c1 = "SCOPE significantly better in every cell of both panels"
    if not complete and (loss or tie):
        c1 = "PROVISIONAL (family incomplete) - " + c1
    S["g13"]["narrowing"] = {
        "rule": "any tie or loss is printed; 'matches or exceeds its nearest masked neighbours' if ties occur, "
                "'does not match in every cell' if losses occur; applied to the primary family AND the fused-selected "
                "panel (worst case); incomplete / missing tests never count as W/T/L and make the outcome provisional",
        "primary": {k: wtl[k] for k in ("win", "tie", "loss", "incomplete", "missing", "family_complete")},
        "fused_selected_panel": {k: wtlf[k] for k in ("win", "tie", "loss", "incomplete", "missing",
                                                      "family_complete")},
        "ties": wtl["ties"] + wtlf["ties"], "losses": wtl["losses"] + wtlf["losses"],
        "family_complete": complete, "outcome": c1}
    write_json(spath, S)

    # ---- sanity: re-runs (a) vs deployed (b); the cap-60 replica is the reproduction check
    log("summarize: sanity family ...")
    ent = []                                               # here a = re-run, b = deployed
    for ds in args.datasets:
        b_list = [pu.get(("ref", ds, None, s)) for s in seeds]
        for kind, arm in (("cap60", "scope_mlp_cap60"), ("full", "scope_mlp_full")):
            a_list = [pu.get((kind, ds, None, s)) for s in seeds]
            for setting in ("alone", "fused"):
                for metric in ("R20", "N20"):
                    ent.append(make_entry({"family": "sanity", "dataset": ds, "arm": arm, "a": kind, "b": "ref",
                                           "seeds": seeds}, a_list, b_list, setting, metric))
    S["g13"]["sanity"]["family"] = build_family(
        "Sanity: shared-trainer SCOPE re-run (a) minus deployed SCOPE (b), seed-averaged",
        "2 re-runs (cap-60 = exact copy of scope.train, the reproduction check; full lists = the encoder-only side) x "
        "2 settings x 2 metrics x datasets; separate Holm family (reproduction checks, not competitors)",
        ent, B, alpha)
    S["g13"]["sanity"]["win_tie_loss"] = win_tie_loss(S["g13"]["sanity"]["family"])
    for ds in args.datasets:
        S["g13"]["sanity"]["check"][ds] = {}
        for kind, part in (("cap60", "runs_cap60"), ("full", "runs_full")):
            S["g13"]["sanity"]["check"][ds][kind] = {}
            for setting in ("alone", "fused"):
                for metric in ("R20", "N20"):
                    dv = [_test_val(S["scope_ref"].get(ds, {}).get(str(s)), setting, metric) for s in seeds]
                    rv = [_test_val(S["g13"]["sanity"][part].get(ds, {}).get(str(s)), setting, metric) for s in seeds]
                    dv = [x for x in dv if x is not None]
                    rv = [x for x in rv if x is not None]
                    chk = {"deployed": dv, "rerun": rv, "seeds_complete": len(dv) == len(seeds) == len(rv),
                           "role": ("reproduction check (exact copy of scope.train, cap-60 lists)" if kind == "cap60"
                                    else "full lists (differs from the deployed procedure by the list cap)")}
                    if len(dv) >= 2 and len(rv) >= 2:
                        delta = float(np.mean(rv) - np.mean(dv))
                        se = math.sqrt(np.var(dv, ddof=1) / len(dv) + np.var(rv, ddof=1) / len(rv))
                        chk.update({"delta_mean": delta, "se_seed": se, "within_2se": bool(abs(delta) <= 2 * se),
                                    "criterion": "|mean(re-run) - mean(deployed)| <= 2*sqrt(s_dep^2/n + s_rerun^2/n) "
                                                 "over seeds (pre-registered)"})
                        if not chk["within_2se"]:
                            flag(f"sanity ({kind}) {ds} {setting} {metric}: shared-trainer re-run differs from the "
                                 f"deployed head by {delta:+.4f} (> 2 seed-SE {2 * se:.4f}) - report the discrepancy")
                    else:
                        chk["within_2se"] = None
                        if kind == "cap60":
                            flag(f"sanity (cap60) {ds} {setting} {metric}: reproduction check not computable "
                                 f"({len(rv)} re-run / {len(dv)} deployed seeds)")
                    S["g13"]["sanity"]["check"][ds][kind][f"{setting}_{metric}"] = chk
    write_json(spath, S)

    # ---- G7 family + gate (pre-registered gamma grid, no extension)
    log("summarize: G7 family + gate ...")
    g7keys = [cfg_key(a, h) for a, h in g7_configs()]
    S["g7"]["family"] = build_family(
        "G7 CBOW gate: SCOPE (deployed) minus CBOW config, seed-averaged per-user metrics",
        f"4 CBOW configs (init x le, lr 3e-3) x 2 settings x 2 metrics x {len(args.datasets)} datasets; separate "
        f"family; gamma on the pre-registered G7 grid {GAMMA_GRID} (no extension)",
        fam("ref7", "g7", g7keys, seeds, family="g7"), B, alpha)
    S["g7"]["win_tie_loss"] = win_tie_loss(S["g7"]["family"])
    write_json(spath, S)
    if not args.no_perseed_bootstrap:
        ent = []
        for s in seeds:
            ent += fam("ref7", "g7", g7keys, [s], family="g7_perseed")
        S["g7"]["family_perseed"] = build_family("G7 per-seed (secondary)", "same comparisons per seed; separate Holm",
                                                 ent, B, alpha)
        write_json(spath, S)
    for ds in args.datasets:
        rvals = [_test_val(S["scope_ref_g7"].get(ds, {}).get(str(s)), "fused", "R20") for s in seeds]
        ref_f = _mean(rvals)
        margins, nseeds, edge = {}, {}, []
        for key in g7keys:
            runs = S["g7"]["runs"].get(ds, {}).get(key, {})
            avals = [_test_val(runs.get(str(s)), "fused", "R20") for s in seeds]
            arm_f = _mean(avals)
            nseeds[key] = sum(x is not None for x in avals)
            margins[key] = (ref_f - arm_f) if (ref_f is not None and arm_f is not None) else None
            edge += [f"{key} s{s}" for s in seeds if (runs.get(str(s)) or {}).get("gamma_edge_hit")]
        valid = {k: v for k, v in margins.items() if v is not None}
        mde = MDE_PLAN.get(ds)
        mk = min(valid, key=valid.get) if valid else None
        verdicts = {t["arm"]: {"p_holm": t.get("p_holm"), "verdict": t.get("verdict"), "mean_delta": t.get("mean_delta"),
                               "status": t.get("status")}
                    for t in S["g7"]["family"]["tests"]
                    if t["dataset"] == ds and t["setting"] == "fused" and t["metric"] == "R20"}
        complete = (len(valid) == len(g7keys) and all(n == len(seeds) for n in nseeds.values())
                    and sum(x is not None for x in rvals) == len(seeds))
        S["g7"]["gate"][ds] = {
            "rule": ("pre-registered: if the fused R@20 margin of SCOPE-v1 over ANY fixed CBOW config "
                     "(seed means, conservative worst case) is <= the MDE (.0026/.0022/.0017), the residual encoder's "
                     "increment over the mean-pool head is reported as +x (p=...) and the head's distinctness rests on the non-pairwise property only"),
            "gamma_rule": f"pre-registered G7 grid {GAMMA_GRID}, no extension",
            "scope_v1_fused_R20_seedmean": ref_f, "margins_fused_R20": margins, "n_seeds_per_config": nseeds,
            "min_margin_config": mk, "min_margin": valid.get(mk) if mk else None, "mde_preregistered": mde,
            "triggered": bool(valid[mk] <= mde) if (mk is not None and mde is not None) else None,
            "fused_R20_bootstrap_G7_family": verdicts, "gamma_edge_hits": edge,
            "complete": complete, "provisional": not complete}
        if edge:
            flag(f"G7 gate {ds}: gamma=5 (edge of the pre-registered grid) selected for {edge}; not extended")
    # per-(dataset, seed) gate files
    per = S["g7"]["family_perseed"]["tests"] if S["g7"]["family_perseed"] else []
    names = {("alone", "R20"): "set_vs_alone_R", ("alone", "N20"): "set_vs_alone_N",
             ("fused", "R20"): "v1_vs_fused_R", ("fused", "N20"): "v1_vs_fused_N"}
    for ds in args.datasets:
        for s in seeds:
            gf = {"dataset": ds, "seed": s, "scope_ref": S["scope_ref_g7"].get(ds, {}).get(str(s)), "arms": {}}
            for key in g7keys:
                rs = S["g7"]["runs"].get(ds, {}).get(key, {}).get(str(s), {})
                bsd = {}
                for t in per:
                    if t["dataset"] == ds and t["arm"] == key and t["seeds"] == [s]:
                        bsd[names[(t["setting"], t["metric"])]] = {
                            k: t.get(k) for k in ("mean_delta", "ci95", "p_two_sided", "p_holm", "n_users", "verdict",
                                                  "status")}
                gf["arms"][key] = {"config": rs.get("hp"), "val_R20": rs.get("best_val_R20"),
                                   "best_ep": rs.get("best_ep"), "last_ep": rs.get("last_ep"),
                                   "hit_cap": rs.get("hit_cap"),
                                   "test": {"alone_R20": _test_val(rs, "alone", "R20"),
                                            "alone_N20": _test_val(rs, "alone", "N20"),
                                            "fused_gamma": rs.get("gamma"),
                                            "fused_gamma_edge_hit": rs.get("gamma_edge_hit"),
                                            "fused_R20": _test_val(rs, "fused", "R20"),
                                            "fused_N20": _test_val(rs, "fused", "N20")},
                                   "bootstrap": bsd}
            write_json(outdir / f"g7_cbow_gate_{ds}_s{s}.json", gf)
    S["runtime_note"] = "wall-clock and peak GPU memory per run: runs[*].train_s / peak_gb / eval_s / eval_peak_gb"
    write_json(spath, S)
    log(f"summary written: {spath}")
    for lab, w in (("primary", wtl), ("fused-selected panel", wtlf),
                   ("encoder-only (secondary)", S["g13"]["win_tie_loss_encoder_only"])):
        log(f"G13 {lab} win/tie/loss (Holm over m={w['m_expected']}, alpha={alpha}): {w['win']}/{w['tie']}/"
            f"{w['loss']} (incomplete {w['incomplete']}, missing {w['missing']}; family complete: "
            f"{w['family_complete']})")
    log(f"G13 narrowing -> {c1}")
    for ds in args.datasets:
        g = S["g7"]["gate"].get(ds, {})
        log(f"G7 gate {ds}: min fused margin {g.get('min_margin')} ({g.get('min_margin_config')}) vs MDE "
            f"{g.get('mde_preregistered')} -> triggered={g.get('triggered')} (complete={g.get('complete')})")
    log(f"{len(flags)} flags; tables: python scope/neighbors_tables.py --summary {spath}")
    return spath


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "ref", "g7", "grid", "seeds", "sanity", "eval", "summarize"])
    ap.add_argument("--datasets", nargs="+", default=None, help="default baby sports clothing (smoke: baby)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default 2024 2025 2026 (smoke: 2024 2025)")
    ap.add_argument("--grid_seed", type=int, default=2024)
    ap.add_argument("--arms", nargs="+", default=None, choices=G13_ARMS, help="G13 arms for grid/seeds (default all 5)")
    ap.add_argument("--smoke", action="store_true", help="tiny sanity run: 4 epochs, 2 grid points/arm, 2048 train "
                                                         "users, B=200; separate *_SMOKE output directories")
    ap.add_argument("--smoke_users", type=int, default=2048)
    ap.add_argument("--out", default=None, help="default <root>/results/scope/rev/g7g13_neighbors")
    ap.add_argument("--logdir", default=None, help="default <root>/logs/rev/g7g13_neighbors")
    ap.add_argument("--ckpt_dir", default=None, help="default <out>/ckpt")
    ap.add_argument("--device", default=os.environ.get("SCOPE_DEVICE", "cuda:0"))
    ap.add_argument("--deterministic", type=int, default=1, help="src.utils.seed.set_seed(deterministic=...)")
    ap.add_argument("--config_model", default="scope", help="model yaml used by RecDataset (as scope.py)")
    ap.add_argument("--list_cap", type=int, default=0, help="0 = max train degree; >0 caps the lists")
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--vae_max_epochs", type=int, default=1000, help="safety ceiling; hit_cap is recorded")
    ap.add_argument("--vae_anneal_steps", type=int, default=20000, help="beta = min(beta_cap, step / anneal_steps)")
    ap.add_argument("--patience", type=int, default=20, help="in validation checks")
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--tf_tok_budget", type=int, default=65536, help="tokens per Transformer micro-batch")
    ap.add_argument("--tf_attn_budget", type=int, default=1 << 27, help="rows*L^2*heads per Transformer micro-batch")
    ap.add_argument("--base_tag", default="table1")
    ap.add_argument("--base_npy", default=None, help="[U,I] base score matrix (e.g. after G3); needs a new --base_tag")
    ap.add_argument("--no_base_cache", action="store_true")
    ap.add_argument("--scope_ckpt_dir", default=None, help="default <root>/ckpts/scope")
    ap.add_argument("--scope_json_dir", default=None, help="default <root>/results/scope")
    ap.add_argument("--no_sanity_cap60", action="store_true",
                    help="skip the cap-60 exact copy of scope.train (the default reproduction check; its absence is "
                         "flagged by the summary)")
    ap.add_argument("--retry_failed", action="store_true")
    ap.add_argument("--no_train", action="store_true")
    ap.add_argument("--fail_fast", action="store_true")
    ap.add_argument("--bootstrap_B", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no_perseed_bootstrap", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.datasets = a.datasets or ["baby"]
        a.seeds = a.seeds or [2024, 2025]
        a.max_epochs = min(a.max_epochs, 4)
        a.vae_max_epochs = min(a.vae_max_epochs, 4)
        a.patience = min(a.patience, 1)
        a.bootstrap_B = min(a.bootstrap_B, 200)
        a.fail_fast = True
        a.out = a.out or str(ROOT / "results" / "scope" / "rev" / "g7g13_neighbors_SMOKE")
        a.logdir = a.logdir or str(ROOT / "logs" / "rev" / "g7g13_neighbors_SMOKE")
    a.datasets = a.datasets or ["baby", "sports", "clothing"]
    a.seeds = a.seeds or [2024, 2025, 2026]
    a.out = a.out or str(ROOT / "results" / "scope" / "rev" / "g7g13_neighbors")
    a.logdir = a.logdir or str(ROOT / "logs" / "rev" / "g7g13_neighbors")
    a.ckpt_dir = a.ckpt_dir or str(Path(a.out) / "ckpt")
    a.scope_ckpt_dir = a.scope_ckpt_dir or str(ROOT / "ckpts" / "scope")
    a.scope_json_dir = a.scope_json_dir or str(ROOT / "results" / "scope")
    a.arms = a.arms or list(G13_ARMS)
    if a.grid_seed not in a.seeds:
        ap.error("--grid_seed must be one of --seeds")
    if a.base_npy and a.base_tag == "table1":
        ap.error("--base_npy needs its own --base_tag (outputs are namespaced by it)")
    if a.stage == "eval":
        a.no_train = True
    return a


def main(argv=None):
    global DEV
    args = parse_args(argv)
    P = Paths(args)
    for d in (P.out, P.logs):
        d.mkdir(parents=True, exist_ok=True)
    log = Log(P.logs / f"main_{args.stage}_{utc_ts()}.log")
    ctx = Ctx(args, P, log)
    log(f"{SCRIPT} {VERSION} | SCOPE_ROOT={ROOT} | python={sys.executable} | args={json.dumps(vars(args))}")
    if args.stage == "summarize":
        stage_summarize(ctx)
        return 0
    DEV = torch.device(args.device)
    if DEV.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("the GPU stages need a CUDA device (--device cuda:N)")
    SC.DEV = DEV                                   # scope.Rmat/build_lists/mm_affinity/evalS_trusted use scope.DEV
    torch.cuda.set_device(DEV)
    set_seed(args.seeds[0], deterministic=bool(args.deterministic))
    ctx.zr_selftest = selftest_zr()
    log(f"device {DEV} ({torch.cuda.get_device_name(DEV)}), torch {torch.__version__}, deterministic="
        f"{bool(args.deterministic)}, in-place z-score bitwise == scope.zr: {ctx.zr_selftest}")
    t0 = time.time()
    for name in args.datasets:
        ds = DSCtx(name, ctx)
        try:
            for fn in STAGES[args.stage]:
                fn(ds, ctx)
        finally:
            ds.close()
            del ds
            torch.cuda.empty_cache()
    log(f"GPU stages done in {(time.time() - t0) / 3600:.2f} h; max per-run peak GPU memory {ctx.max_peak_gb:.1f} GB; "
        f"failures in this invocation: {ctx.n_fail}; skipped (--no_train, not trained): {ctx.n_skipped}")
    if args.stage == "all":
        stage_summarize(ctx)
    return 1 if (ctx.n_fail or ctx.n_skipped) else 0


if __name__ == "__main__":
    sys.exit(main())

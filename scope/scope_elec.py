#!/usr/bin/env python
"""SCOPE on Amazon-Electronics (192,403 users x 63,001 items) — fully streamed / chunked.

At this scale the dense closed-form EASE base of scope/scope.py is infeasible (a 63k x 63k
dense inverse and a ~48 GB fp32 score matrix), so:
  * the closed-form base is replaced by a SPARSE item-item co-occurrence kNN proxy
    (scope/cooc_knn.py) that plays the same 'sharp item-item' role;
  * every score matrix is consumed in user-row chunks through the streaming evaluator
    (gpu_eval.GPUEval.eval_streaming) — the full [n_users, n_items] matrix is never
    materialized.

Two results are produced (weights selected on validation Recall@20, single test eval):
  SCOPE-v1-elec : z(set-completion head) + gamma * z(cooc-kNN proxy).
  SCOPE-U-elec  : a non-negative gate over FOUR z-scored views — FREEDOM scores, LGMRec
                  scores, the cooc-kNN proxy, and the set-completion view. The best gate
                  with the set view removed is also reported (its marginal value).

The two collaborative views are read from the score dumps written by
scope/dump_baseline_scores.py (results/baseline_scores/{model}_elec_scores.npy),
memory-mapped and read chunk-wise; the gate grid is scored in a single streaming pass
per split (all weight combinations share each chunk's z-scored views).

Usage (from the repository root; dataset files under data/elec per configs/dataset/elec.yaml):
  1) train the two collaborative baselines
       python -m src.main --model freedom --dataset elec
       python -m src.main --model lgmrec  --dataset elec
  2) dump their full score matrices (~48 GB fp32 .npy each)
       mkdir -p results/baseline_scores
       python scope/dump_baseline_scores.py --model freedom --dataset elec
       python scope/dump_baseline_scores.py --model lgmrec  --dataset elec
  3) run this script (trains the set head unless ckpts/scope/scope_elec_*.pt exists)
       python scope/scope_elec.py [--seed 2024] [--epochs 120] [--retrain]

Results -> results/scope/scope_elec.json (seed-suffixed for non-default seeds).
"""
from __future__ import annotations
import os, sys, json, math, random, argparse, itertools
sys.path.insert(0, os.path.dirname(__file__))
import logging; logging.disable(logging.INFO)
import numpy as np, torch, torch.nn.functional as F
from scope import Rmat, build_lists, sigreg, SCOPE, zr, DEV, ROOT, OUT, CK   # inserts ROOT on sys.path
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from cooc_knn import build_cooc_knn

DS = "elec"
GAMMA_GRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]                       # SCOPE-v1 fusion gamma
GATE_GRIDS = [[0.0, 1.0, 2.0, 3.0],                                    # freedom
              [0.0, 1.0, 2.0, 3.0],                                    # lgmrec
              [0.0, 0.5, 1.0, 2.0, 3.0],                               # cooc proxy
              [0.0, 1.0, 2.0]]                                         # set head
VIEW_NAMES = ["freedom", "lgmrec", "cooc", "set"]


def train_set_head(dset, R, items, vmask, deg, gev, a, ck):
    """Mirror of scope.py train() for the set-completion head, with streaming validation
    (score_all would materialize a [U,N] matrix; infeasible here)."""
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    init = None
    if dset.t_feat is not None:                                        # text-projection init, as in scope.py
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
        Wp = F.normalize(torch.randn(X.shape[1], a.d, device=DEV), dim=0)
        init = (X @ Wp) / math.sqrt(a.d); del X
    model = SCOPE(dset.n_items, a.d, init).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-6)
    degf = deg.float(); tu = torch.where(deg >= 2)[0]
    def val_fn():
        zp = model.latent(torch.sparse.mm(R, model.E), degf)
        zpn = F.normalize(zp, dim=1); En = F.normalize(model.E, dim=1)
        tau = model.logtau.exp().clamp(min=1e-3)
        return lambda u: (zpn[u] @ En.t()) / tau
    best = {"r20": -1.0, "ep": -1}; bad = 0
    for ep in range(a.epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), a.bs):
            b = perm[i:i + a.bs]
            z, it, ctx, tgt = model.forward_train(items[b], vmask[b], deg[b])
            logits = model.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it)
            cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, dim=1)
            lrank = -((logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)).mean()
            loss = lrank + a.le * sigreg(model.E)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0 or ep == a.epochs - 1:
            model.eval()
            with torch.no_grad(): vr = gev.eval_streaming(val_fn())["Recall@20"]
            if vr > best["r20"]:
                best = {"r20": vr, "ep": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                bad = 0
            else: bad += 1
            print(f"[{DS}] set head ep{ep:3d} val_R20={vr:.4f} best={best['r20']:.4f}", flush=True)
            if bad >= a.patience:
                print(f"[{DS}] set head early stop ep{ep}", flush=True); break
    model.load_state_dict(best["state"]); model.eval()
    torch.save(best["state"], ck)
    print(f"[{DS}] saved {ck.name} (best ep{best['ep']} val_R20={best['r20']:.4f})", flush=True)
    return model


def set_view(model, R, deg):
    """Cache the predicted latents once; per-chunk scores are a [b,d] x [d,N] matmul."""
    with torch.no_grad():
        zp = model.latent(torch.sparse.mm(R, model.E), deg.float())
        zpn = F.normalize(zp, dim=1); En = F.normalize(model.E, dim=1)
        tau = float(model.logtau.exp().clamp(min=1e-3))
    return lambda u: (zpn[u] @ En.t()) / tau


def dump_view(name, dset):
    """Memory-mapped chunk reader over a dump_baseline_scores.py score matrix."""
    p = ROOT / "results" / "baseline_scores" / f"{name}_{DS}_scores.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — train the baseline (python -m src.main --model {name} --dataset {DS}) "
            f"then dump its scores (python scope/dump_baseline_scores.py --model {name} --dataset {DS})")
    arr = np.load(p, mmap_mode="r")
    assert arr.shape == (dset.n_users, dset.n_items), f"{p.name}: shape {arr.shape}"
    def fn(u):
        rows = np.asarray(arr[u.detach().cpu().numpy()], dtype=np.float32)
        return torch.from_numpy(rows).to(DEV)
    return fn


def _chunk_mask(gev, bu):
    """Train-history (row, col) mask indices for one user chunk (same logic as GPUEval._mask)."""
    if gev.hist_u is None: return None, None
    order = torch.argsort(bu); bs = bu[order]
    p = torch.searchsorted(bs, gev.hist_u).clamp(max=bu.numel() - 1)
    valid = bs[p] == gev.hist_u
    return order[p[valid]], gev.hist_i[valid]


@torch.no_grad()
def gate_search(gev, view_fns, grids, batch=1024, group=16):
    """Score EVERY non-zero weight combination (validation Recall@20) in ONE streaming pass:
    each chunk's views are loaded and z-scored once, then all combinations are fused (fp16,
    ranking-safe on z-scored views), train-masked and top-20 scored on GPU."""
    combos = [ws for ws in itertools.product(*grids) if any(w > 0 for w in ws)]
    W = torch.tensor(combos, dtype=torch.float16, device=DEV)          # [C, nv]
    acc = torch.zeros(len(combos), device=DEV)
    U = gev.users.numel()
    for s in range(0, U, batch):
        bu = gev.users[s:s + batch]; b = bu.numel()
        V = torch.stack([zr(fn(bu)).half() for fn in view_fns])        # [nv, b, N]
        Vf = V.reshape(len(view_fns), -1)                              # [nv, b*N]
        rows, cols = _chunk_mask(gev, bu)
        bp = gev.pos[s:s + b]; nrel = gev.nfit[s:s + b].clamp(min=1)
        for g0 in range(0, len(combos), group):
            Fg = (W[g0:g0 + group] @ Vf).view(-1, b, V.shape[2])       # [g, b, N]
            if rows is not None: Fg[:, rows, cols] = float("-inf")
            _, idx = torch.topk(Fg, 20, dim=2)                         # [g, b, 20]
            hit = (idx.unsqueeze(3) == bp.view(1, b, 1, -1)).any(3).float().sum(2)   # [g, b]
            acc[g0:g0 + group] += (hit / nrel.unsqueeze(0)).sum(1)
            del Fg, idx
        del V, Vf
    return combos, (acc / U).tolist()


def fused_fn(view_fns, ws):
    """Streaming score function for a fixed gate (fp32; used for the trusted single evals)."""
    act = [(float(w), f) for w, f in zip(ws, view_fns) if w > 0]
    def fn(u):
        S = None
        for w, f in act:
            t = w * zr(f(u))
            S = t if S is None else S + t
        return S
    return fn


def main(a):
    sfx = '' if a.seed == 2024 else f'_s{a.seed}'
    dset = RecDataset(Config("scope", DS))
    print(f"[{DS}] users={dset.n_users} items={dset.n_items} seed={a.seed}", flush=True)
    R = Rmat(dset); items, vmask, deg = build_lists(dset)
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

    # (a) set-completion head: load checkpoint if present, else train
    ck = CK / f"scope_{DS}_d{a.d}_le{a.le}_lr{a.lr}{sfx}.pt"
    if ck.exists() and not a.retrain:
        model = SCOPE(dset.n_items, a.d).to(DEV)
        model.load_state_dict(torch.load(ck, map_location=DEV)); model.eval()
        trained = False
        print(f"[{DS}] loaded set-head checkpoint {ck.name}", flush=True)
    else:
        model = train_set_head(dset, R, items, vmask, deg, gev, a, ck); trained = True
    v_set = set_view(model, R, deg)
    set_val = gev.eval_streaming(lambda u: zr(v_set(u)))["Recall@20"]
    print(f"[{DS}] set head val_R20={set_val:.4f}", flush=True)

    # (b) sparse co-occurrence-kNN base proxy (dense EASE infeasible at this item count)
    print(f"[{DS}] building sparse cooc-kNN proxy (k={a.k}) ...", flush=True)
    G = build_cooc_knn(R, k=a.k, chunk=2048, device=DEV)
    Gt = G.t().coalesce()
    def v_cooc(u):
        ru = torch.index_select(R, 0, u).to_dense()                    # [b, N]
        return torch.sparse.mm(Gt, ru.t()).t()                         # [b, N]

    # collaborative views from the baseline score dumps
    v_free = dump_view("freedom", dset)
    v_lgm = dump_view("lgmrec", dset)
    view_fns = [v_free, v_lgm, v_cooc, v_set]                          # order == VIEW_NAMES

    # each view alone (test, single pass each)
    alone = {}
    for nm, f in zip(VIEW_NAMES, view_fns):
        alone[nm] = gevT.eval_streaming(lambda u, f=f: zr(f(u)))
        print(f"[{DS}] {nm:7s} alone R@20={alone[nm]['Recall@20']:.4f} N@20={alone[nm]['NDCG@20']:.4f}", flush=True)

    # (c) SCOPE-v1-elec: z(set) + gamma * z(cooc), gamma on validation
    combos, rec = gate_search(gev, [v_set, v_cooc], [[1.0], GAMMA_GRID])
    bi = max(range(len(combos)), key=lambda i: rec[i])
    gamma = float(combos[bi][1]); v1_val = rec[bi]
    v1_test = gevT.eval_streaming(fused_fn([v_set, v_cooc], (1.0, gamma)))
    print(f"[{DS}] SCOPE-v1 gamma={gamma} val_R20={v1_val:.4f} -> "
          f"test R@20={v1_test['Recall@20']:.4f} N@20={v1_test['NDCG@20']:.4f}", flush=True)

    # (d) SCOPE-U-elec: non-negative gate over the four views, grid on validation, test once
    combos, rec = gate_search(gev, view_fns, GATE_GRIDS)
    bi = max(range(len(combos)), key=lambda i: rec[i])
    ws = combos[bi]; u_val = rec[bi]
    u_test = gevT.eval_streaming(fused_fn(view_fns, ws))
    # marginal value of the set view: best gate with the set weight forced to 0
    no_set = [i for i in range(len(combos)) if combos[i][VIEW_NAMES.index("set")] == 0]
    bj = max(no_set, key=lambda i: rec[i])
    wns = combos[bj]
    ns_test = gevT.eval_streaming(fused_fn(view_fns, wns))
    print(f"[{DS}] SCOPE-U gate {dict(zip(VIEW_NAMES, ws))} val_R20={u_val:.4f} -> "
          f"test R@20={u_test['Recall@20']:.4f} N@20={u_test['NDCG@20']:.4f}", flush=True)
    print(f"[{DS}] no-set gate {dict(zip(VIEW_NAMES, wns))} val_R20={rec[bj]:.4f} -> "
          f"test R@20={ns_test['Recall@20']:.4f} N@20={ns_test['NDCG@20']:.4f} | "
          f"set marginal dR@20={u_test['Recall@20'] - ns_test['Recall@20']:+.4f}", flush=True)

    out = {"dataset": DS, "seed": a.seed, "n_users": dset.n_users, "n_items": dset.n_items,
           "protocol": "gate weights selected on validation Recall@20; single test evaluation per row",
           "set_head": {"ckpt": ck.name, "trained_this_run": trained, "val_R20": set_val,
                        "hp": dict(d=a.d, lr=a.lr, le=a.le, bs=a.bs, epochs=a.epochs, patience=a.patience)},
           "cooc_knn_k": a.k,
           "views_alone_test": alone,
           "scope_v1": {"gamma": gamma, "gamma_grid": GAMMA_GRID, "val_R20": v1_val, "test": v1_test},
           "scope_u": {"gate": dict(zip(VIEW_NAMES, [float(w) for w in ws])), "gate_grids": GATE_GRIDS,
                       "val_R20": u_val, "test": u_test,
                       "no_set_gate": dict(zip(VIEW_NAMES, [float(w) for w in wns])),
                       "no_set_test": ns_test,
                       "set_marginal_R20": u_test["Recall@20"] - ns_test["Recall@20"]}}
    outp = OUT / f"scope_{DS}{sfx}.json"
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"[{DS}] -> {outp}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SCOPE-v1 and SCOPE-U on Amazon-Electronics (streamed).")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--le", type=float, default=1.0, help="isotropy regularizer weight")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--k", type=int, default=100, help="cooc-kNN neighbors per item")
    ap.add_argument("--retrain", action="store_true", help="retrain the set head even if a checkpoint exists")
    main(ap.parse_args())

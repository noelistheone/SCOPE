#!/usr/bin/env python
"""Coverage experiments: which held-out pairs each scorer covers, by user activity and item popularity.

Recall@20 IS coverage (fraction of held-out items retrieved). For each framework we compute:
  (a) COVERAGE-FAILURE (neither-hit) fraction = 1 - micro per-test-pair top-20 hit rate  [lower=better]
  (b) per-user-ACTIVITY-QUINTILE macro Recall@20 (cold->hot)
  (c) LONG-TAIL item coverage: retrieval rate on test items by train-degree bucket
Reported worst->best. Trusted full-sort, train-masked.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from src.data.dataloader import EvalDataLoader
from scope import Rmat, build_lists, closed_form_base, ease_B, SCOPE, zr, BAR, DEV

import argparse
DS = "baby"  # overridden by --dataset


def gather(dset, phase="test"):
    """Return users[T], test_items list per row, train-hist pairs, and item train-degree."""
    loader = EvalDataLoader(dset, phase=phase, batch_size=4096)
    users, pos, hu, hi = [], [], [], []
    for b in loader:
        uu = b["user_ids"].tolist(); users.extend(uu)
        pi = b["positive_items"]
        for r in range(len(uu)): pos.append([int(x) for x in pi[r]])
        H, V = b["history_indices"], b["history_values"]
        if H.numel() > 0:
            mk = V.bool()
            for r, u in enumerate(uu):
                h = H[r][mk[r]]; h = h[h >= 0]
                for it in h.tolist(): hu.append(u); hi.append(it)
    return users, pos, (torch.tensor(hu), torch.tensor(hi))


@torch.no_grad()
def topk_items(S, users, hist, k=20, batch=4096):
    """Return top-k item ids per user (train-masked), aligned with `users`."""
    U = torch.tensor(users, device=DEV)
    hu, hi = hist[0].to(DEV), hist[1].to(DEV)
    out = torch.empty(len(users), k, dtype=torch.long, device=DEV)
    for s in range(0, len(users), batch):
        bu = U[s:s + batch]; sc = S[bu].clone().float()
        order = torch.argsort(bu); bus = bu[order]
        pos = torch.searchsorted(bus, hu).clamp(max=bu.numel() - 1)
        val = bus[pos] == hu
        sc[order[pos[val]], hi[val]] = float("-inf")
        out[s:s + bu.numel()] = torch.topk(sc, k, 1).indices
    return out.cpu().numpy()


def coverage_metrics(topk, pos, udeg, ideg):
    """topk[T,k], pos list[T], udeg[n_users], ideg[n_items]. Return coverage stats."""
    T = len(pos)
    # per-pair hit (micro coverage) + per-user recall (macro) + per-test-item retrieved
    pair_hit = 0; pair_tot = 0
    per_user_rec = []; per_user_deg = []
    tail_hit = head_hit = tail_tot = head_tot = 0
    for r in range(T):
        gt = set(pos[r]); tk = set(int(x) for x in topk[r])
        nhit = len(gt & tk)
        pair_hit += nhit; pair_tot += len(gt)
        per_user_rec.append(nhit / max(1, len(gt))); per_user_deg.append(int(udeg[r]))
        for it in gt:
            if ideg[it] <= 5: tail_tot += 1; tail_hit += (it in tk)
            else: head_tot += 1; head_hit += (it in tk)
    micro_cov = pair_hit / max(1, pair_tot)
    return {
        "macro_R20": float(np.mean(per_user_rec)),
        "micro_coverage": micro_cov, "neither_hit_frac": 1 - micro_cov,
        "tail_coverage": tail_hit / max(1, tail_tot), "head_coverage": head_hit / max(1, head_tot),
        "tail_tot": tail_tot, "head_tot": head_tot,
        "_rec": np.array(per_user_rec), "_deg": np.array(per_user_deg),
    }


def quintiles(rec, deg, nq=5):
    # RANK-based equal-size groups (avoids empty bins from integer-degree ties)
    order = np.argsort(deg, kind="stable"); n = len(deg); out = []
    for q in range(nq):
        idx = order[q * n // nq:(q + 1) * n // nq]
        out.append((float(deg[idx].mean()), float(rec[idx].mean()), len(idx)))
    return out


def main(DS, collab="freedom"):
    from gpu_eval import GPUEval
    half = (RecDataset(Config("scope", DS)).n_items > 20000); dt = torch.float16 if half else torch.float32
    dset = RecDataset(Config("scope", DS))
    Rsp = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    n_users, n_items = dset.n_users, dset.n_items
    Rd = Rsp.to_dense(); udeg = Rd.sum(1).cpu().numpy(); ideg = Rd.sum(0).cpu().numpy()
    G = Rd.t() @ Rd; del Rd; torch.cuda.empty_cache()
    users, pos, hist = gather(dset, "test")
    udeg_row = np.array([udeg[u] for u in users])
    gev = GPUEval(dset, "valid", DEV)
    m1 = SCOPE(n_items, 256).to(DEV); m1.load_state_dict(torch.load(ROOT/"ckpts"/"scope"/f"scope_{DS}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV, weights_only=True)); m1.eval()
    base = closed_form_base(Rsp, dset, gev, half=half)         # tuned base (matches SCOPE-U)
    free = torch.from_numpy(np.load(ROOT/"results"/"baseline_scores"/f"freedom_{DS}_scores.npy")).to(dt).to(DEV)
    # SCOPE-U collaborative view: FREEDOM or GUME (the CF view feeding the gate).
    if collab == "gume":
        cf = torch.from_numpy(np.load(ROOT/"results"/"baseline_scores"/f"gume_{DS}_scores.npy")).to(dt).to(DEV)
        # GUME-based gate from the ensemble-control sweep: keys order [gume(cf), item(base), set]
        gate = json.load(open(ROOT/"results"/"scope"/f"ensemble_control_{DS}.json"))["rows"]["GUMEswap(base+set+GUME)"]["gate"]
        cf_name = "GUME"
    else:
        cf = free
        # FREEDOM-based gate from the ensemble-control sweep: keys order [freedom(col), item(base), set]
        gate = json.load(open(ROOT/"results"/"scope"/f"ensemble_control_{DS}.json"))["rows"]["SCOPE-U(base+set+FREEDOM)"]["gate"]
        cf_name = "FREEDOM"

    # builders (lazy: build one big matrix at a time, free after)
    def b_pop():   return Rsp.to_dense().sum(0).unsqueeze(0).expand(n_users, -1).contiguous().to(dt)
    def b_ease():  return (Rsp @ ease_B(G, 400)).to(dt)
    def b_mm():    return base.to(dt)
    def b_cf():    return cf
    def b_set():   return m1.score_all(Rsp, degf).to(dt)
    def b_scopeu():return (gate[0]*zr(cf.float()) + gate[1]*zr(base.float()) + gate[2]*zr(m1.score_all(Rsp, degf))).to(dt)
    builders = {"popularity": b_pop, "EASE": b_ease, "MM-EASE/base": b_mm, cf_name: b_cf,
                "SCOPE-v1(set only)": b_set, "SCOPE-U(full)": b_scopeu}

    rows = {}
    for name, build in builders.items():
        Smat = build()
        tk = topk_items(Smat, users, hist, 20)
        cm = coverage_metrics(tk, pos, udeg_row, ideg)
        qs = quintiles(cm["_rec"], cm["_deg"])
        rows[name] = {k: v for k, v in cm.items() if not k.startswith("_")}
        rows[name]["quintile_R20"] = [round(r, 4) for _, r, _ in qs]
        rows[name]["quintile_deg"] = [round(d, 1) for d, _, _ in qs]
        del Smat; torch.cuda.empty_cache()
    # order worst->best by macro_R20
    order = sorted(rows, key=lambda n: rows[n]["macro_R20"])
    print(f"\n[{DS}] COVERAGE ANALYSIS (worst->best).\n", flush=True)
    print(f"{'framework':22s} {'R@20':>7s} {'cov%':>6s} {'neither%':>9s} {'tailCov%':>9s} {'headCov%':>9s}  quintileR@20(cold->hot)")
    for n in order:
        r = rows[n]
        print(f"{n:22s} {r['macro_R20']:.4f} {r['micro_coverage']*100:6.2f} {r['neither_hit_frac']*100:9.2f} "
              f"{r['tail_coverage']*100:9.2f} {r['head_coverage']*100:9.2f}  {r['quintile_R20']}", flush=True)
    suffix = "" if collab == "freedom" else f"_{collab}"
    (ROOT/"results"/"scope"/f"coverage_{DS}{suffix}.json").write_text(json.dumps({"dataset": DS, "collab": collab, "rows": rows, "order": order}, indent=2, default=str))
    print(f"\nsaved -> results/scope/coverage_{DS}{suffix}.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["baby"])
    ap.add_argument("--collab", default="freedom", choices=["freedom", "gume"]); a = ap.parse_args()
    for ds in a.datasets:
        try: main(ds, collab=a.collab)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()

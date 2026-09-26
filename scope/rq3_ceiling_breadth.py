#!/usr/bin/env python
"""Coverage ceiling and breadth analyses for the GUME-based SCOPE-U:
  (1) structural coverage ceiling — fraction of test pairs reachable by the UNION of the three views'
      top-20 (the best any router could do), and the share SCOPE-U's static fusion captures;
  (2) breadth — fraction of users whose top-20 is unchanged by adding the set view, and the
      helps/ties/hurts split of SCOPE-U vs the set-free (base+GUME) ensemble;
  (3) popularity — recommendation Gini and catalog coverage of SCOPE-U vs GUME.
Views: gume (cached), base (EASE+text), set (SCOPE ckpt). Gates from the ensemble-control sweep.
Results -> results/scope/rq3_gume_{ds}.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV
from coverage_analysis import gather, topk_items


def gini(counts):
    x = np.sort(counts.astype(np.float64)); n = len(x)
    if x.sum() == 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def topset(S, users, hist, k=20):
    tk = topk_items(S, users, hist, k)        # [T,k] item ids, train-masked
    return tk


def hits(tk, pos):
    return np.array([len(set(p) & set(int(x) for x in tk[r])) for r, p in enumerate(pos)])


def run(ds):
    dset = RecDataset(Config("scope", ds))
    half = dset.n_items > 20000; dt = torch.float16 if half else torch.float32
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    gev = GPUEval(dset, "valid", DEV)
    users, pos, hist = gather(dset, "test")
    npos = np.array([len(p) for p in pos]); tot_pairs = int(npos.sum())

    base = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT/"ckpts"/"scope"/f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV, weights_only=True)); m.eval()
    sset = zr(m.score_all(R, degf)).to(dt)
    gume = zr(torch.from_numpy(np.load(ROOT/"results"/"baseline_scores"/f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    ec = json.load(open(ROOT/"results"/"scope"/f"ensemble_control_{ds}.json"))["rows"]
    gU = ec["GUMEswap(base+set+GUME)"]["gate"]      # [w_gume, w_base, w_set]
    g2 = ec["base+GUME(no set)"]["gate"]            # [w_gume, w_base]

    # per-view top-20
    tk_g = topset(gume, users, hist); tk_b = topset(base, users, hist); tk_s = topset(sset, users, hist)
    # SCOPE-U (gume+base+set) and set-free (gume+base)
    Su = gU[0]*gume + gU[1]*base + gU[2]*sset
    S2 = g2[0]*gume + g2[1]*base
    tk_u = topset(Su, users, hist); tk_2 = topset(S2, users, hist)

    # (1) oracle ceiling: union of the 3 views' hits / total pairs ; captured = SCOPE-U hits / union hits
    union_hit = 0; u_hit = 0
    for r, p in enumerate(pos):
        gt = set(p)
        hu3 = (gt & set(int(x) for x in tk_g[r])) | (gt & set(int(x) for x in tk_b[r])) | (gt & set(int(x) for x in tk_s[r]))
        union_hit += len(hu3)
        u_hit += len(gt & set(int(x) for x in tk_u[r]))
    oracle_cov = union_hit / tot_pairs
    scopeu_cov = u_hit / tot_pairs
    captured = scopeu_cov / oracle_cov if oracle_cov else 0.0

    # (2) breadth: SCOPE-U vs set-free (base+gume)
    hu = hits(tk_u, pos) / np.maximum(npos, 1); h2 = hits(tk_2, pos) / np.maximum(npos, 1)
    unchanged = np.mean([set(int(x) for x in tk_u[r]) == set(int(x) for x in tk_2[r]) for r in range(len(pos))])
    helps = float(np.mean(hu > h2)); ties = float(np.mean(np.isclose(hu, h2))); hurts = float(np.mean(hu < h2))

    # (3) popularity: recommendation Gini + catalog coverage (SCOPE-U vs GUME)
    def rec_stats(tk):
        cnt = np.bincount(tk.reshape(-1), minlength=dset.n_items)
        return gini(cnt), float((cnt > 0).mean())
    gini_u, cov_u = rec_stats(tk_u); gini_g, cov_g = rec_stats(tk_g)

    out = {"dataset": ds, "oracle_union_coverage": oracle_cov, "scopeu_coverage": scopeu_cov,
           "captured_share": captured, "set_unchanged_frac": float(unchanged),
           "breadth_helps": helps, "breadth_ties": ties, "breadth_hurts": hurts, "helps_or_ties": helps + ties,
           "gini_scopeu": gini_u, "gini_gume": gini_g, "catalog_cov_scopeu": cov_u, "catalog_cov_gume": cov_g,
           "gate_scopeu": gU, "gate_setfree": g2}
    (ROOT/"results"/"scope"/f"rq3_gume_{ds}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[{ds}] oracle={oracle_cov*100:.1f}% scopeU={scopeu_cov*100:.1f}% captured={captured*100:.1f}% "
          f"| unchanged={unchanged*100:.1f}% helps_or_ties={(helps+ties)*100:.1f}% (helps={helps*100:.1f} hurts={hurts*100:.1f}) "
          f"| GiniU={gini_u:.3f} GiniGUME={gini_g:.3f} catCovU={cov_u:.3f} catCovGUME={cov_g:.3f}", flush=True)
    del R, base, sset, gume, Su, S2; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"]); a = ap.parse_args()
    for ds in a.datasets:
        try: run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()

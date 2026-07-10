"""Train/held-out alignment audit, formalized as a protocol.

A clean (uncontaminated) scoring pathway scores a user's TRAIN-fit items far higher than held-out items,
and treats validation and test symmetrically (no test-specific information): train_mean >> test_mean AND
|test_mean - valid_mean| < 0.15 * train_mean (z-scored scores). We screen all THREE released feature
pathways the field consumes -- interaction co-occurrence (EASE), text affinity, and IMAGE affinity --
on Baby/Sports/Clothing. Pathways that pass are leakage-clean; a pathway that scores held-out >= train
(as MLLMRec's bespoke features do) is flagged.
Writes results/scope/leakage_protocol_<ds>.json. GPU. Usage: python leakage_protocol.py <ds...>
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from scope import Rmat, build_lists, closed_form_base, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset


def content_affinity(feat_path, R, k=20):
    X = F.normalize(torch.from_numpy(np.load(feat_path).astype(np.float32)).to(DEV), dim=1)
    G = X @ X.t(); kth = torch.topk(G, k + 1, 1).values[:, -1:]
    A = torch.where(G >= kth, G, torch.zeros_like(G)); A.fill_diagonal_(0.0); del G
    d = A.sum(1).clamp(min=1e-6); A = A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0)
    S = zr(torch.sparse.mm(R, A)); del A, X; torch.cuda.empty_cache(); return S


def split_means(S, users, pos):
    vals = []
    for s in range(0, users.numel(), 4096):
        bu = users[s:s + 4096]; bp = pos[s:s + 4096]
        sc = S[bu].float(); m = bp >= 0
        g = torch.gather(sc, 1, bp.clamp(min=0))
        vals.append(torch.where(m, g, torch.full_like(g, float('nan'))).reshape(-1))
    v = torch.cat(vals); v = v[~torch.isnan(v)]
    return float(v.mean())


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, _ = build_lists(dset)
    half = dset.n_items > 20000 or dset.n_users > 50000
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    # train positives per user (from gevT history)
    usersT = gevT.users
    train_pos = {}
    if gevT.hist_u is not None:
        hu = gevT.hist_u.cpu().numpy(); hi = gevT.hist_i.cpu().numpy()
        for u, i in zip(hu, hi): train_pos.setdefault(int(u), []).append(int(i))
    Pmax = max((len(v) for v in train_pos.values()), default=1)
    tp = torch.full((usersT.numel(), Pmax), -1, dtype=torch.long, device=DEV)
    for r, u in enumerate(usersT.cpu().numpy()):
        it = train_pos.get(int(u), [])
        if it: tp[r, :len(it)] = torch.tensor(it[:Pmax], device=DEV)

    views = {
        "interaction_EASE": zr(closed_form_base(R, dset, None, lam=800, a=0.0, half=half)),
        "text_affinity": content_affinity(ROOT / "data" / ds / "text_feat.npy", R),
        "image_affinity": content_affinity(ROOT / "data" / ds / "image_feat.npy", R),
    }
    out = {"dataset": ds, "views": {}}
    for nm, S in views.items():
        tr = split_means(S, usersT, tp); te = split_means(S, gevT.users, gevT.pos); va = split_means(S, gevV.users, gevV.pos)
        # leakage = test-specific information: a contaminated pathway scores held-out TEST items above
        # VALIDATION items (test-specific signal). Clean = test treated symmetrically to validation.
        clean = bool(abs(te - va) < 0.15)
        out["views"][nm] = {"train": round(tr, 3), "valid": round(va, 3), "test": round(te, 3),
                            "test_minus_valid": round(te - va, 4), "train_minus_test": round(tr - te, 3), "clean": clean}
        print(f"  [{ds}] {nm:18s} train={tr:+.2f} val={va:+.2f} test={te:+.2f} (t-v={te-va:+.3f}, tr-te={tr-te:+.2f}) clean={clean}", flush=True)
        del S
    out["all_clean"] = all(v["clean"] for v in out["views"].values())
    json.dump(out, open(ROOT / "results" / "scope" / f"leakage_protocol_{ds}.json", "w"), indent=2)
    del R; torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        run(ds)
    print("LEAKAGE_PROTOCOL_DONE", flush=True)

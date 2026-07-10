"""Ranking half of the task-formulation flip (self-contained, SCOPE pipeline, paper's own MMRec features).

The item-item co-purchase decomposition (copurchase_auc.py) shows IMAGE is the strongest item-item
co-purchase predictor with a learned metric. Here we ask the user-item RANKING question with the SAME
features: build an image (and text) content item-item kNN view, z-score it, and grid-search its gate
weight on top of a strong CF ranker (base EASE+text + GUME) on validation Recall@20 -- exactly the
null-tower protocol of the paper. A content view that is strong at item-item but earns ~0 ranking weight
(and ~0 test Delta R@20) over base+GUME is the ranking half of the flip: rich item-item content signal,
redundant with CF at user-item ranking.

Writes results/scope/content_ranking_null_<ds>.json. GPU. Usage: python content_ranking_null.py <ds...>
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from scope import Rmat, build_lists, closed_form_base, SCOPE, evalS_trusted, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def content_knn_view(feat_path, R, dt, k=20):
    X = F.normalize(torch.from_numpy(np.load(feat_path).astype(np.float32)).to(DEV), dim=1)
    G = X @ X.t()
    kth = torch.topk(G, k + 1, 1).values[:, -1:]
    A = torch.where(G >= kth, G, torch.zeros_like(G)); A.fill_diagonal_(0.0); del G
    d = A.sum(1).clamp(min=1e-6); A = A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0)
    S = zr(torch.sparse.mm(R, A)).to(dt)
    del A, X; torch.cuda.empty_cache()
    return S


def gate_select(views, gev):
    """Grid-search nonneg gate on val Recall@20; return (weights, fused S)."""
    best = None
    for ws in itertools.product(GR, repeat=len(views)):
        if all(w == 0 for w in ws):
            continue
        S = sum(w * v for w, v in zip(ws, views))
        vr = gev.eval(S)["Recall@20"]
        if best is None or vr > best[0]:
            best = (vr, ws)
        del S
    ws = best[1]
    return ws, sum(w * v for w, v in zip(ws, views))


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset)
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16
    gev = GPUEval(dset, "valid", DEV)
    base = zr(closed_form_base(R, dset, gev, half=half)).to(dt)
    gume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    img = content_knn_view(ROOT / "data" / ds / "image_feat.npy", R, dt)
    txt = content_knn_view(ROOT / "data" / ds / "text_feat.npy", R, dt)

    out = {"dataset": ds, "standalone_R20": {}, "gate_weights": {}, "test_R20": {}}
    # standalone content-view ranking quality
    out["standalone_R20"]["image_knn"] = evalS_trusted(img, dset, "test")["Recall@20"]
    out["standalone_R20"]["text_knn"] = evalS_trusted(txt, dset, "test")["Recall@20"]
    # reference: base + GUME (no content view)
    w_ref, S_ref = gate_select([base, gume], gev); r_ref = evalS_trusted(S_ref, dset, "test")
    out["gate_weights"]["base+gume"] = {"base": w_ref[0], "gume": w_ref[1]}
    out["test_R20"]["base+gume"] = r_ref["Recall@20"]; del S_ref; torch.cuda.empty_cache()
    # + image content view
    w_i, S_i = gate_select([base, gume, img], gev); r_i = evalS_trusted(S_i, dset, "test")
    out["gate_weights"]["base+gume+image"] = {"base": w_i[0], "gume": w_i[1], "image": w_i[2]}
    out["test_R20"]["base+gume+image"] = r_i["Recall@20"]; del S_i; torch.cuda.empty_cache()
    # + text content view
    w_t, S_t = gate_select([base, gume, txt], gev); r_t = evalS_trusted(S_t, dset, "test")
    out["gate_weights"]["base+gume+text"] = {"base": w_t[0], "gume": w_t[1], "text": w_t[2]}
    out["test_R20"]["base+gume+text"] = r_t["Recall@20"]; del S_t; torch.cuda.empty_cache()

    out["delta_image_over_base_gume"] = out["test_R20"]["base+gume+image"] - out["test_R20"]["base+gume"]
    out["delta_text_over_base_gume"] = out["test_R20"]["base+gume+text"] - out["test_R20"]["base+gume"]
    json.dump(out, open(ROOT / "results" / "scope" / f"content_ranking_null_{ds}.json", "w"), indent=2)
    print(f"[{ds}] standalone img-kNN R20={out['standalone_R20']['image_knn']:.4f} txt-kNN={out['standalone_R20']['text_knn']:.4f}", flush=True)
    print(f"  base+gume R20={out['test_R20']['base+gume']:.4f}  +image w={w_i[2]} dR20={out['delta_image_over_base_gume']:+.4f}  +text w={w_t[2]} dR20={out['delta_text_over_base_gume']:+.4f}", flush=True)
    del R, base, gume, img, txt; torch.cuda.empty_cache()


if __name__ == "__main__":
    dss = sys.argv[1:] or ["baby", "sports", "clothing"]
    for ds in dss:
        run(ds)
    print("CONTENT_RANKING_NULL_DONE", flush=True)

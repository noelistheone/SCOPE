"""Frozen content views: complementarity of the set view and of a content-kNN view with strong CF.

Using the per-user complementarity definition -- Spearman of per-user Recall@20 ACROSS users
between two views (low correlation = the views succeed on DIFFERENT users = complementary) -- we compare:
  rho(set view, CF=GUME)            should be LOW  (set view complements CF; succeeds on other users)
  rho(content-kNN view, CF=GUME)    should be HIGH (content ranking view is redundant; same users as CF)
and, as a consistency check, rho(base, set).
We ALSO report the content view's gate weight when fused with the base ALONE (no GUME): if it is >0 there
but ~0 once GUME is added, the gate=0 is CF explaining it away, not the view being useless.
Writes results/scope/welding_<ds>.json. GPU. Usage: python welding.py <ds...>
"""
from __future__ import annotations
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def content_view(feat_path, R, dt, k=20):
    X = F.normalize(torch.from_numpy(np.load(feat_path).astype(np.float32)).to(DEV), dim=1)
    G = X @ X.t(); kth = torch.topk(G, k + 1, 1).values[:, -1:]
    A = torch.where(G >= kth, G, torch.zeros_like(G)); A.fill_diagonal_(0.0); del G
    d = A.sum(1).clamp(min=1e-6); A = A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0)
    S = zr(torch.sparse.mm(R, A)).to(dt); del A, X; torch.cuda.empty_cache(); return S


def gate2(a, b, gev):                     # best weight for a + g*b on validation Recall@20
    best = (0.0, -1.0)
    for g in GR:
        r = gev.eval(a + g * b)["Recall@20"]
        if r > best[1]: best = (g, r)
    return best[0]


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000; dt = torch.float16
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    base = zr(closed_form_base(R, dset, gevV, half=half)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad(): sset = zr(m.score_all(R, degf)).to(dt)
    gume = zr(torch.from_numpy(np.load(ROOT / "results" / "baseline_scores" / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    img = content_view(ROOT / "data" / ds / "image_feat.npy", R, dt)
    txt = content_view(ROOT / "data" / ds / "text_feat.npy", R, dt)

    def ru(S): return gevT.recall_per_user(S.float(), 20).cpu().numpy()
    r_base, r_set, r_gume, r_img, r_txt = ru(base), ru(sset), ru(gume), ru(img), ru(txt)
    def rho(a, b): return float(spearmanr(a, b).statistic)
    out = {"dataset": ds,
           "rho_base_set": rho(r_base, r_set),                 # consistency check
           "rho_set_gume": rho(r_set, r_gume),                 # set view vs CF (should be LOWER)
           "rho_image_view_gume": rho(r_img, r_gume),          # content ranking view vs CF (should be HIGHER)
           "rho_text_view_gume": rho(r_txt, r_gume),
           "rho_base_gume": rho(r_base, r_gume)}
    # content gate with base ALONE (no GUME): >0 means the view IS useful, just CF-redundant
    out["img_gate_with_base_only"] = gate2(base, img, gevV)
    out["txt_gate_with_base_only"] = gate2(base, txt, gevV)
    out["set_gate_with_base_only"] = gate2(base, sset, gevV)
    json.dump(out, open(ROOT / "results" / "scope" / f"welding_{ds}.json", "w"), indent=2)
    print(f"[{ds}] rho(base,set)={out['rho_base_set']:.2f} | "
          f"rho(set,GUME)={out['rho_set_gume']:.2f} << rho(img-view,GUME)={out['rho_image_view_gume']:.2f} "
          f"rho(txt-view,GUME)={out['rho_text_view_gume']:.2f} | "
          f"base-only gate: img={out['img_gate_with_base_only']} txt={out['txt_gate_with_base_only']} set={out['set_gate_with_base_only']}", flush=True)
    del R, base, sset, gume, img, txt; torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        run(ds)
    print("WELDING_DONE", flush=True)

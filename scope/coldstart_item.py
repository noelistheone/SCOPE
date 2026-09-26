"""Item cold-start (new-item) across all 3 dense datasets, via the content base.

A genuinely unseen item has NO training interactions, so any interaction-based scorer
(collaborative filtering, co-occurrence, popularity) gives it score 0 and cannot rank it
(structurally at chance). SCOPE's content pathway -- the text-kNN affinity bundled in its
base -- still ranks it from content similarity to the user's seen items, with no retraining.

We hold out a random 20% of items, build the text-kNN content affinity over ALL items
(content is available for held-out items -- that is what makes it inductive), score
S = R_train @ A_text, and report Recall@20/NDCG@20 restricted to held-out-item test targets
(new-item) and to seen-item targets (the in-sample ceiling of the same scorer), plus the
retention ratio. Writes results/scope/coldstart_item_<ds>.json. GPU. Usage: python coldstart_item.py [ds...]
"""
from __future__ import annotations
import sys, os, json, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, closed_form_base, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset


def metrics_restricted(S, g, keep_item, k=20):
    """Recall@k and NDCG@k over test positives whose item is in keep_item (bool [n_items])."""
    U = g.users.numel(); hits = 0.0; tot = 0.0; ndcg = 0.0
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=DEV).float())  # [k]
    for s in range(0, U, 2048):
        bu = g.users[s:s + 2048]; sc = g._mask(S[bu].clone().float(), bu)
        _, tk = torch.topk(sc, k, 1)                              # [b,k] ranked item ids
        pos = g.pos[s:s + 2048]                                   # [b,P] padded -1
        valid = (pos >= 0) & keep_item[pos.clamp(min=0)]         # restricted test positives
        # rank position of each top-k item among the user's restricted positives
        match = (tk.unsqueeze(2) == pos.unsqueeze(1)) & valid.unsqueeze(1)  # [b,k,P]
        hit_at_rank = match.any(2).float()                       # [b,k] did rank r hit a valid pos
        ndcg += (hit_at_rank * discount).sum().item()
        intop = match.any(1)                                     # [b,P] is each valid pos in topk
        hits += intop.sum().item(); tot += valid.sum().item()
    return {"recall": hits / max(tot, 1), "ndcg": ndcg / max(tot, 1)}


def text_knn(feat, k=20):
    X = F.normalize(torch.from_numpy(np.asarray(feat[:]).astype(np.float32)).to(DEV), dim=1)
    G = X @ X.t(); kth = torch.topk(G, k + 1, 1).values[:, -1:]
    A = torch.where(G >= kth, G, torch.zeros_like(G)); A.fill_diagonal_(0.0); del G, X
    d = A.sum(1).clamp(min=1e-6); A = A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0)
    return A


def run(ds, holdout=0.2, seed=2024):
    torch.manual_seed(seed); np.random.seed(seed)
    dset = RecDataset(Config("scope", ds))
    n_items = dset.n_items
    gen = torch.Generator(device=DEV).manual_seed(seed)
    H = torch.zeros(n_items, dtype=torch.bool, device=DEV)
    H[torch.randperm(n_items, generator=gen, device=DEV)[: int(holdout * n_items)]] = True  # held-out (unseen) items
    R = Rmat(dset)
    g = GPUEval(dset, "test", DEV)
    gev = GPUEval(dset, "valid", DEV)
    half = dset.n_items > 20000 or dset.n_users > 50000
    # inductive content score for unseen items: only the content (text-kNN) term is available
    # (a genuinely unseen item has no training interactions, so its EASE/co-occurrence column is 0).
    A = text_knn(dset.t_feat)
    S_content = zr(torch.sparse.mm(R, A)); del A; torch.cuda.empty_cache()
    # in-sample ceiling: the FULL base (EASE co-occurrence + text) on these same items, with co-occurrence active
    S_base = zr(closed_form_base(R, dset, gev, half=half))
    new = metrics_restricted(S_content, g, H)        # content-only, held-out items (genuinely inductive)
    ceiling = metrics_restricted(S_base, g, H)        # full base on the same items, in-sample
    res = {
        "dataset": ds, "holdout_frac": holdout, "n_held_items": int(H.sum().item()),
        "content_new_item": new, "full_base_insample_ceiling": ceiling,
        "retention_recall": new["recall"] / max(ceiling["recall"], 1e-9),
        "retention_ndcg": new["ndcg"] / max(ceiling["ndcg"], 1e-9),
        "interaction_based_new_item": 0.0,  # CF/co-occurrence/popularity score unseen items at exactly 0 (no training signal)
    }
    json.dump(res, open(ROOT / "results" / "scope" / f"coldstart_item_{ds}.json", "w"), indent=2)
    print(f"[{ds}] content NEW-item R@20={new['recall']:.4f} N@20={new['ndcg']:.4f} | "
          f"full-base in-sample ceiling R@20={ceiling['recall']:.4f} | retention={res['retention_recall']*100:.0f}% "
          f"(CF/pop new-item = 0, structural)", flush=True)
    del S_content, S_base, R; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("COLDSTART_ITEM_DONE", flush=True)

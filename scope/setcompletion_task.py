"""Set completion as a distinct task: recover a held-out SUBSET of a user's set from
the rest, the task SCOPE is trained for, vs scorers that are not set-conditioned.

For each user we split their interaction set 50/50 into a context O and a target T (uniformly at random),
score all items conditioned ONLY on O, and measure recovery of T (Recall@20 over T, masking O). We compare:
  SCOPE head (z from O -> cosine to item embeddings; trained by masked-set completion),
  text-kNN  (R_O @ A_text; content set-conditioned, not trained for the task),
  co-occ-kNN(R_O @ A_cooc; collaborative set-conditioned),
  popularity(item degree; not set-conditioned).
A user-embedding CF model (GUME/LightGCN) cannot condition on the partial set O at all. If SCOPE recovers
held-out set members best, set completion is a genuine task at which the formulation excels.
Writes results/scope/setcompletion_<ds>.json. GPU. Usage: python setcompletion_task.py <ds...>
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from scope import Rmat, build_lists, SCOPE, gram, zr, DEV, ROOT
from src.utils import Config
from src.data.dataset import RecDataset


def knn_propagate(A_sim, R_O, k=20):
    kth = torch.topk(A_sim, k + 1, 1).values[:, -1:]
    A = torch.where(A_sim >= kth, A_sim, torch.zeros_like(A_sim)); A.fill_diagonal_(0.0)
    d = A.sum(1).clamp(min=1e-6); A = A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0)
    return torch.sparse.mm(R_O, A)


def recall_from_sets(S, Omask_items, T_pad, k=20):
    """Recall@k of recovering T (padded item ids, -1) after masking O items. row-aligned to all users."""
    U = S.shape[0]; hits = 0.0; tot = 0.0
    for s in range(0, U, 2048):
        sc = S[s:s + 2048].clone().float()
        oi = Omask_items[s:s + 2048]                                 # [b, L] padded O item ids (-1)
        rr = torch.arange(sc.shape[0], device=DEV).unsqueeze(1).expand_as(oi)
        mo = oi >= 0; sc[rr[mo], oi[mo].clamp(min=0)] = float("-inf")
        _, tk = torch.topk(sc, k, 1)
        tp = T_pad[s:s + 2048]                                       # [b, P]
        valid = tp >= 0
        intop = (tk.unsqueeze(2) == tp.unsqueeze(1)).any(1)
        hits += (intop & valid).sum().item(); tot += valid.sum().item()
    return hits / max(tot, 1)


def run(ds, seed=0):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); items, vmask, deg = build_lists(dset)
    n_items = dset.n_items
    g = torch.Generator(device=DEV).manual_seed(seed)
    keys = torch.where(vmask > 0, torch.rand(vmask.shape, generator=g, device=DEV), torch.full_like(vmask, 1e9))
    ranks = keys.argsort(1).argsort(1).float()
    nO = (deg.float() / 2).ceil().clamp(min=1)
    Omask = (ranks < nO.unsqueeze(1)) & (vmask > 0)
    Tmask = (ranks >= nO.unsqueeze(1)) & (vmask > 0)
    U = items.shape[0]
    # padded O item ids (for masking) and T item ids (targets)
    O_items = torch.where(Omask, items, torch.full_like(items, -1))
    T_pad = torch.where(Tmask, items, torch.full_like(items, -1))
    # R_O sparse (context only)
    rows = torch.arange(U, device=DEV).unsqueeze(1).expand_as(items)[Omask]
    cols = items[Omask]
    R_O = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(rows.numel(), device=DEV), (U, n_items)).coalesce()
    nOf = Omask.sum(1).float()

    # SCOPE head
    m = SCOPE(n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    with torch.no_grad():
        S_scope = m.logits_from(m.latent(torch.sparse.mm(R_O, m.E), nOf))
    # text-kNN and co-occurrence-kNN, popularity
    Xt = F.normalize(torch.from_numpy(np.load(ROOT / "data" / ds / "text_feat.npy").astype(np.float32)).to(DEV), 1)
    S_text = knn_propagate(Xt @ Xt.t(), R_O); del Xt
    S_cooc = knn_propagate(gram(R).float(), R_O)
    deg_item = torch.sparse.sum(R, 0).to_dense().float()
    S_pop = deg_item.unsqueeze(0).expand(U, n_items)

    res = {"dataset": ds, "task": "recover held-out 50% of each set from the other 50%, Recall@20"}
    for nm, S in [("SCOPE_head", S_scope), ("text_kNN", S_text), ("cooc_kNN", S_cooc), ("popularity", S_pop)]:
        res[nm] = recall_from_sets(S, O_items, T_pad)
        print(f"  [{ds}] set-completion R@20  {nm:12s} = {res[nm]:.4f}", flush=True)
        del S
    json.dump(res, open(ROOT / "results" / "scope" / f"setcompletion_{ds}.json", "w"), indent=2)
    del R, R_O, m; torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        run(ds)
    print("SETCOMPLETION_DONE", flush=True)

"""Inductive baselines + protocol hardening for the inductive-scoring claim.

NEW USER: we restrict to users with a long history (|S_u|>=8) so that k in {1,2,3,5} is a genuine subset,
draw the k-item context UNIFORMLY AT RANDOM (consistent with the order-free training objective, NOT 'first k
by time'), average over 3 draws, and compare SCOPE-v1 (encode user from the k items) against two inductive
baselines that also need no per-user training: PopRec (item popularity, context-independent) and a session
item--item kNN (co-occurrence kNN propagated from the k context items). Recall@20 masks the full train
history. This gives a within-method degradation curve AND an external reference.
Writes results/scope/inductive_baselines_<ds>.json. GPU. Usage: python inductive_baselines.py <ds...>
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scope import Rmat, build_lists, SCOPE, gram, zr, evalS_trusted, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset

KS = [1, 2, 3, 5]
MINH = 8        # only users with >=8 train items (so k<=5 is a true subset; removes the 'sparse users are ~3 items' confound)
DRAWS = 3


def cooc_knn_matrix(R, n_items, k=20):
    G = gram(R).float(); kth = torch.topk(G, k + 1, 1).values[:, -1:]
    A = torch.where(G >= kth, G, torch.zeros_like(G)); A.fill_diagonal_(0.0); del G
    d = A.sum(1).clamp(min=1e-6); return (A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0))


def rand_k_R(items, vmask, deg, k, n_items, gen):
    """random k-subset of each user's train items -> sparse [U,n_items] context matrix + count."""
    U, L = items.shape
    keys = torch.where(vmask > 0, torch.rand(vmask.shape, generator=gen, device=DEV), torch.full_like(vmask, 1e9))
    ranks = keys.argsort(1).argsort(1)
    take = (ranks < k) & (vmask > 0)
    rows = torch.arange(U, device=DEV).unsqueeze(1).expand_as(items)[take]
    cols = items[take]
    R = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(rows.numel(), device=DEV), (U, n_items)).coalesce()
    return R, take.sum(1).float()


def restricted_recall(S, gevT, keep_user, k=20):
    """Recall@20 over only users in keep_user (bool over gevT.users order), full-history masked."""
    g = gevT; U = g.users.numel(); hits = 0.0; tot = 0.0
    for s in range(0, U, 2048):
        idx = torch.arange(s, min(s + 2048, U), device=DEV); ku = keep_user[idx]
        if ku.sum() == 0: continue
        bu = g.users[s:s + 2048][ku]
        sc = g._mask(S[bu].clone().float(), bu)
        _, tk = torch.topk(sc, k, 1)
        pos = g.pos[s:s + 2048][ku]; valid = pos >= 0
        intop = (tk.unsqueeze(2) == pos.unsqueeze(1)).any(1)
        hits += (intop & valid).sum().item(); tot += valid.sum().item()
    return hits / max(tot, 1)


def run(ds):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); items, vmask, deg = build_lists(dset); n = dset.n_items
    gevT = GPUEval(dset, "test", DEV)
    su_size = torch.sparse.sum(R, 1).to_dense().float()
    keep_user = (su_size[gevT.users] >= MINH)                       # long-history users only
    m = SCOPE(n, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    Acooc = cooc_knn_matrix(R, n)
    deg_item = torch.sparse.sum(R, 0).to_dense().float()
    pop = deg_item.unsqueeze(0).expand(gevT.users.numel(), n)       # PopRec (context-free)
    pop_recall = restricted_recall(pop[: , :], gevT, keep_user); del pop

    out = {"dataset": ds, "n_longhist_users": int(keep_user.sum().item()), "min_history": MINH,
           "PopRec": pop_recall, "scope": {}, "session_itemknn": {}}
    gen = torch.Generator(device=DEV).manual_seed(0)
    for k in KS:
        sc_s, sc_i = [], []
        for _ in range(DRAWS):
            Rk, cnt = rand_k_R(items, vmask, deg, k, n, gen)
            with torch.no_grad():
                Ss = m.logits_from(m.latent(torch.sparse.mm(Rk, m.E), cnt))
            sc_s.append(restricted_recall(Ss, gevT, keep_user)); del Ss
            Si = torch.sparse.mm(Rk, Acooc)
            sc_i.append(restricted_recall(Si, gevT, keep_user)); del Si, Rk
            torch.cuda.empty_cache()
        out["scope"][f"k{k}"] = float(np.mean(sc_s)); out["session_itemknn"][f"k{k}"] = float(np.mean(sc_i))
    # full-history SCOPE on the same long-history users (the within-method ceiling)
    with torch.no_grad():
        Sfull = m.logits_from(m.latent(torch.sparse.mm(R, m.E), deg.float()))
    out["scope"]["full"] = restricted_recall(Sfull, gevT, keep_user)
    json.dump(out, open(ROOT / "results" / "scope" / f"inductive_baselines_{ds}.json", "w"), indent=2)
    print(f"[{ds}] long-hist n={out['n_longhist_users']} | PopRec={pop_recall:.4f} | "
          f"SCOPE k1/3/5/full={out['scope']['k1']:.4f}/{out['scope']['k3']:.4f}/{out['scope']['k5']:.4f}/{out['scope']['full']:.4f} | "
          f"session-itemkNN k1/3/5={out['session_itemknn']['k1']:.4f}/{out['session_itemknn']['k3']:.4f}/{out['session_itemknn']['k5']:.4f}", flush=True)
    del R, m, Acooc; torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        run(ds)
    print("INDUCTIVE_BASELINES_DONE", flush=True)

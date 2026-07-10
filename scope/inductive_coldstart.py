"""Inductive cold-start: SCOPE scores new users and new items without retraining.

SCOPE has NO per-user parameters (a user is encoded from the mean of their set) and content-seeded item
embeddings, so it can score:
  (A) NEW USERS  -- any user with k>=1 interactions is scored from those items alone, no per-user training.
      We score test users from only their FIRST k context items (k in {1,2,3,full}) and report Recall@20.
  (B) NEW ITEMS  -- an item never seen in training keeps its content-seeded embedding, so it can still be
      ranked. We RETRAIN SCOPE-v1 with a random 20% of items HELD OUT of training (removed from every user's
      context/target; their embedding stays at the content-seeded init), then evaluate Recall@20 restricted
      to held-out-item test targets, vs a content text-kNN scorer. An ID-embedding CF model (GUME/LightGCN)
      has no embedding for an unseen item and must retrain, so it cannot score these targets at all.

Writes results/scope/inductive_{newuser,newitem}.json. GPU. Usage: python inductive_coldstart.py
"""
from __future__ import annotations
import json, math, random
import numpy as np, torch, torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from scope import (Rmat, build_lists, closed_form_base, SCOPE, zr, evalS_trusted, sigreg, DEV, ROOT)
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset


# ----------------------------- (A) NEW USER: partial first-k context -----------------------------
def first_k_R(items, vmask, k, n_items):
    U, L = items.shape; kk = min(k, L)
    mask = (vmask[:, :kk] > 0)
    rows = torch.arange(U, device=DEV).unsqueeze(1).expand(U, kk)[mask]
    cols = items[:, :kk][mask]
    R = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(rows.numel(), device=DEV),
                                (U, n_items)).coalesce()
    return R, mask.sum(1).float()


def new_user(dss=("baby", "sports", "clothing")):
    out = {}
    for ds in dss:
        dset = RecDataset(Config("scope", ds))
        items, vmask, deg = build_lists(dset)
        m = SCOPE(dset.n_items, 256).to(DEV)
        m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
        row = {}
        for k in [1, 2, 3, 1000000]:
            Rk, cnt = first_k_R(items, vmask, k, dset.n_items)
            with torch.no_grad():
                z = m.latent(torch.sparse.mm(Rk, m.E), cnt)
                S = m.logits_from(z) if dset.n_items <= 30000 else (F.normalize(z, 1).half() @ F.normalize(m.E, 1).half().t())
            r = evalS_trusted(S.float(), dset, "test")["Recall@20"]
            row["full" if k > 100000 else f"k{k}"] = float(r)
            del Rk, S; torch.cuda.empty_cache()
        out[ds] = row
        print(f"[newuser {ds}] " + "  ".join(f"{kk}={vv:.4f}" for kk, vv in row.items())
              + f"  | k1/full={row['k1']/row['full']:.2f}", flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(out, open(ROOT / "results" / "scope" / "inductive_newuser.json", "w"), indent=2)
    return out


# ----------------------------- (B) NEW ITEM: item-holdout retrain -----------------------------
def recall_at_k_restricted(S, gevT, keep_item, k=20):
    """Recall@k computed ONLY over test positives whose item is in keep_item (bool [n_items])."""
    g = gevT; U = g.users.numel(); hits = 0.0; tot = 0.0
    for s in range(0, U, 2048):
        bu = g.users[s:s + 2048]; sc = g._mask(S[bu].clone().float(), bu)
        _, tk = torch.topk(sc, k, 1)                                   # [b,k] top items
        pos = g.pos[s:s + 2048]                                        # [b,P] padded -1
        valid = (pos >= 0) & keep_item[pos.clamp(min=0)]              # held-out test positives
        intop = (tk.unsqueeze(2) == pos.unsqueeze(1)).any(1)          # [b,P] is each pos in topk
        hits += (intop & valid).sum().item(); tot += valid.sum().item()
    return hits / max(tot, 1)


def new_item(ds="baby", holdout=0.2, epochs=300, seed=2024):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dset = RecDataset(Config("scope", ds))
    n_items = dset.n_items
    H = torch.zeros(n_items, dtype=torch.bool, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    H[torch.randperm(n_items, generator=g, device=DEV)[: int(holdout * n_items)]] = True   # held-out items
    keep = ~H
    items, vmask, deg = build_lists(dset)
    # mask out held-out items from every user's context/target so they are NEVER trained on
    keepmask = keep[items.clamp(min=0)] & (vmask > 0)
    vmask = (keepmask).float(); deg = vmask.sum(1)
    # content-seeded init (text projection): held-out items keep this (no gradient ever touches them)
    X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
    Wp = F.normalize(torch.randn(X.shape[1], 256, device=DEV), dim=0); init = (X @ Wp) / math.sqrt(256)
    m = SCOPE(n_items, 256, init).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-6)
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    tu = torch.where(deg >= 2)[0]
    best = {"r": -1}; bad = 0
    for ep in range(epochs):
        m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), 8192):
            b = perm[i:i + 8192]
            z, it, ctx, tgt = m.forward_train(items[b], vmask[b], deg[b])
            logits = m.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it)
            cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logits = logits.masked_fill(H.unsqueeze(0), -1e9)          # never predict held-out items in training
            logp = F.log_softmax(logits, 1)
            loss = -((logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)).mean() + sigreg(m.E)
            opt.zero_grad(); loss.backward()
            if m.E.grad is not None: m.E.grad[H] = 0.0                 # freeze held-out item embeddings at content init
            opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad(): S = m.logits_from(m.latent(torch.sparse.mm(Rmat(dset), m.E), build_lists(dset)[2].float()))
            vr = gev.eval(S)["Recall@20"]
            if vr > best["r"]: best = {"r": vr, "state": {k: v.detach().clone() for k, v in m.state_dict().items()}}; bad = 0
            else: bad += 1
            if bad >= 8: break
    m.load_state_dict(best["state"]); m.eval()
    R = Rmat(dset); _, _, degf = build_lists(dset)
    with torch.no_grad():
        S_scope = m.logits_from(m.latent(torch.sparse.mm(R, m.E), degf.float()))
    # content text-kNN scorer (can also score unseen items from content), and a degree/pop scorer baseline
    Xi = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
    G = Xi @ Xi.t(); kth = torch.topk(G, 21, 1).values[:, -1:]
    A = torch.where(G >= kth, G, torch.zeros_like(G)); A.fill_diagonal_(0.0); del G
    d = A.sum(1).clamp(min=1e-6); A = A / d.sqrt().unsqueeze(1) / d.sqrt().unsqueeze(0)
    S_txt = torch.sparse.mm(R, A); del A
    res = {
        "dataset": ds, "holdout_frac": holdout, "n_held_items": int(H.sum().item()),
        "scope_recall20_on_new_items": recall_at_k_restricted(S_scope, gevT, H),
        "textknn_recall20_on_new_items": recall_at_k_restricted(S_txt, gevT, H),
        "scope_recall20_on_seen_items": recall_at_k_restricted(S_scope, gevT, keep),
        "note": "ID-embedding CF (GUME/LightGCN) has no trained embedding for a held-out item and scores it at chance (must retrain).",
    }
    res["inductive_retention_new_vs_seen"] = res["scope_recall20_on_new_items"] / max(res["scope_recall20_on_seen_items"], 1e-9)
    json.dump(res, open(ROOT / "results" / "scope" / "inductive_newitem.json", "w"), indent=2)
    print(f"[newitem {ds}] SCOPE new-item R@20={res['scope_recall20_on_new_items']:.4f} "
          f"(text-kNN {res['textknn_recall20_on_new_items']:.4f}); seen-item R@20={res['scope_recall20_on_seen_items']:.4f}; "
          f"retention={res['inductive_retention_new_vs_seen']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    new_user()
    new_item("baby")
    print("INDUCTIVE_DONE", flush=True)

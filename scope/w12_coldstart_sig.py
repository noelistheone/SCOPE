"""Significance of the set head's advantage for users with truncated context. For long-history users (|S_u|>=8),
encode each user from only k randomly-drawn interactions and score the full catalog; compare SCOPE's
set-completion head against a training-free session co-occurrence kNN scorer from the SAME k items, with
a paired user-level bootstrap on per-user Recall@20 (held-out = the user's test items). Reports whether the
gain is significant. Writes results/scope/w12_coldstart_sig_<ds>.json.
Usage: python w12_coldstart_sig.py [datasets...]
"""
from __future__ import annotations
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, gram, zr, SCOPE, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset


def cooc_knn(G, k=100):
    C = G.clone(); C.fill_diagonal_(0)
    d = C.sum(1).clamp(min=1e-6)
    C = C / d.unsqueeze(1)                                   # row-normalized co-occurrence
    kth = torch.topk(C, min(k, C.shape[1]), 1).values[:, -1:]
    return torch.where(C >= kth, C, torch.zeros_like(C))


def per_user_recall(S, gevT):
    return gevT.recall_per_user(S).cpu().numpy()


def run(ds, ks=(1, 2, 3)):
    dset = RecDataset(Config("scope", ds))
    items, vmask, deg = build_lists(dset)
    R = Rmat(dset); G = gram(R)
    C = cooc_knn(G); del G; torch.cuda.empty_cache()
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
    gevT = GPUEval(dset, "test", DEV)
    # long-history users only (so k is a genuine subset); align to the test-evaluator user order
    long_mask = (deg >= 8)
    res = {"dataset": ds, "n_long_users": int(long_mask.sum().item()), "k": {}}
    for k in ks:
        kk = k
        # first-k context matrix over ALL users (users with <k interactions contribute what they have)
        U = dset.n_users
        cols = items[:, :kk]; vm = (vmask[:, :kk] > 0)
        rows = torch.arange(U, device=DEV).unsqueeze(1).expand(U, kk)[vm]
        Rk = torch.sparse_coo_tensor(torch.stack([rows, cols[vm]]), torch.ones(rows.numel(), device=DEV), (U, dset.n_items)).coalesce()
        cnt = vm.sum(1).float()
        with torch.no_grad():
            z = m.latent(torch.sparse.mm(Rk, m.E), cnt)
            S_scope = m.logits_from(z) if dset.n_items <= 30000 else (F.normalize(z, 1).half() @ F.normalize(m.E, 1).half().t())
            S_sess = torch.sparse.mm(Rk, C)                  # session co-occurrence kNN from the k items
        ru_scope = per_user_recall(S_scope.float(), gevT)
        ru_sess = per_user_recall(S_sess.float(), gevT)
        # restrict the paired test to long-history users (aligned to gevT.users order)
        gu = gevT.users.cpu().numpy()
        keep = long_mask.cpu().numpy()[gu]
        a, b = ru_scope[keep], ru_sess[keep]
        bs = paired_bootstrap(a, b)
        res["k"][f"k{k}"] = {"scope_R20": round(float(a.mean()), 4), "sessionknn_R20": round(float(b.mean()), 4),
                             "delta": bs}
        ps = '<1e-4' if bs['p_two_sided']==0 else f"{bs['p_two_sided']:.2g}"
        sig = '  *SIG' if (bs['p_two_sided']<0.05 and bs['mean_delta']>0) else ''
        print(f"[{ds}] k={k} SCOPE={a.mean():.4f} sessionkNN={b.mean():.4f}  d={bs['mean_delta']:+.4f} p={ps}{sig}", flush=True)
        del Rk, S_scope, S_sess; torch.cuda.empty_cache()
    json.dump(res, open(ROOT / "results" / "scope" / f"w12_coldstart_sig_{ds}.json", "w"), indent=2)
    del C, m, R; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("W12_COLDSTART_SIG_DONE", flush=True)

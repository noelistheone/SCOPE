"""Cross-domain zero-shot transfer of the set-completion operator.

The set-completion head is (mean-pool over a content-seeded item geometry) -> ONE residual MLP `enc`
-> cosine score. To ask whether *set-completability* is a domain-general function or a per-dataset
trick, we put every dataset's items into ONE shared content space -- a single fixed random projection
Wp:[384->d] applied to the (frozen, benchmark-released) Sentence-BERT features, identical across all
Amazon datasets -- and TRAIN ONLY the operator (enc + temperature), with item embeddings FROZEN at the
shared seed. Freezing E is what makes transfer well-defined: a trained-E head's geometry is
dataset-specific and cannot be moved; a frozen shared-seed head learns an operator in a common space.

We then evaluate the operator trained on dataset A, ZERO-SHOT, on dataset B's users/items:
  identity : z_u = mean-pool of B's seed embeddings (NO enc)          -- content-only lower bound
  transfer : z_u = mean-pool + enc_A(mean-pool), scored on B          -- A's operator, zero-shot on B
  oracle   : z_u = mean-pool + enc_B(mean-pool), scored on B          -- B's own operator (in-domain upper bound)
Score = cos(z_u, E_B_seed); ranking is temperature-invariant per user, so identity needs no tau.
Paired user-level bootstrap of transfer vs identity on B's TEST users. Transfer efficiency
= (transfer - identity) / (oracle - identity). Writes results/scope/w15_transfer.json.

Usage: python w15_transfer.py [datasets...]   (default baby sports clothing)
"""
from __future__ import annotations
import sys, os, json, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scope import Rmat, build_lists, SCOPE, sigreg, DEV, ROOT, OUT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

DIM_TEXT = 384          # Sentence-BERT text dim, shared across Amazon datasets
D = 256                 # operator latent dim (default)
WP_SEED = 12345         # the ONE shared projection (identical for every dataset -> a common space)


def shared_proj():
    g = torch.Generator(device=DEV).manual_seed(WP_SEED)
    return F.normalize(torch.randn(DIM_TEXT, D, generator=g, device=DEV), dim=0)


def seed_emb(dset, Wp):
    """Frozen shared-space item seed: normalize(text) @ Wp / sqrt(d). Same Wp for all datasets."""
    X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
    return (X @ Wp) / math.sqrt(D)


def load_ds(ds, Wp):
    dset = RecDataset(Config("scope", ds))
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    E = seed_emb(dset, Wp)                                      # [n_items, D], FROZEN
    return dict(ds=ds, dset=dset, R=R, items=items, vmask=vmask, deg=deg, degf=degf,
                E=E, En=F.normalize(E, dim=1), n_items=dset.n_items,
                gevV=GPUEval(dset, "valid", DEV), gevT=GPUEval(dset, "test", DEV))


def score(enc, E, En, R, degf, tau=0.1):
    """Full-catalog scores z(cos): z_u = mp + enc(mp) if enc else mp; cosine to frozen seed. tau irrelevant to ranking."""
    mp = torch.sparse.mm(R, E) / degf.clamp(min=1).unsqueeze(1)
    z = mp + enc(mp) if enc is not None else mp
    return (F.normalize(z, dim=1) @ En.t()) / tau


def train_operator(dd, epochs=300, lr=3e-3, wd=1e-6, le=1.0, patience=15, seed=2024):
    """Train ONLY enc + logtau on dataset dd with item embeddings FROZEN at the shared seed."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = SCOPE(dd["n_items"], D).to(DEV)
    m.E.data.copy_(dd["E"]); m.E.requires_grad_(False)         # FREEZE embeddings at shared seed
    params = [p for n, p in m.named_parameters() if n != "E"]  # enc + logtau only
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    items, vmask, deg, degf = dd["items"], dd["vmask"], dd["deg"], dd["degf"]
    tu = torch.where(deg >= 2)[0]; bs = 8192
    best = {"r20": -1, "enc": None, "tau": None}; bad = 0
    for ep in range(epochs):
        m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt = m.forward_train(items[b], vmask[b], deg[b])
            logits = m.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it)
            cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, dim=1)
            tgt_lp = (logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)
            loss = -tgt_lp.mean() + le * sigreg(m.E)           # E frozen: sigreg(E) is constant, harmless
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                vr = dd["gevV"].eval(score(m.enc, dd["E"], dd["En"], dd["R"], degf))["Recall@20"]
            if vr > best["r20"]:
                best = {"r20": vr, "ep": ep,
                        "enc": {k: v.detach().clone() for k, v in m.enc.state_dict().items()},
                        "tau": float(m.logtau.exp().clamp(min=1e-3))}
                bad = 0
            else:
                bad += 1
            print(f"[train {dd['ds']}] ep{ep:3d} val_R20={vr:.4f} best={best['r20']:.4f}", flush=True)
            if bad >= patience:
                print(f"[train {dd['ds']}] early stop ep{ep}", flush=True); break
    enc = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D)).to(DEV)
    enc.load_state_dict(best["enc"]); enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    return enc, best["tau"], best["ep"]


def run(datasets):
    Wp = shared_proj()
    dd = {ds: load_ds(ds, Wp) for ds in datasets}
    ops = {}
    identity = {}
    for ds in datasets:
        enc, tau, ep = train_operator(dd[ds])
        ops[ds] = (enc, tau)
        with torch.no_grad():
            S_id = score(None, dd[ds]["E"], dd[ds]["En"], dd[ds]["R"], dd[ds]["degf"])
        identity[ds] = dd[ds]["gevT"].recall_per_user(S_id, 20).cpu().numpy()
        print(f"[{ds}] identity(no-enc) test_R20={identity[ds].mean():.4f}  operator trained (best_ep={ep})", flush=True)

    # zero-shot transfer matrix: apply operator trained on A to dataset B
    res = {"datasets": datasets, "wp_seed": WP_SEED, "d": D, "transfer": {}, "diag": {}}
    for B in datasets:
        Bd = dd[B]; id_pu = identity[B]
        enc_B, tau_B = ops[B]
        with torch.no_grad():
            S_or = score(enc_B, Bd["E"], Bd["En"], Bd["R"], Bd["degf"], tau_B)
        oracle_pu = Bd["gevT"].recall_per_user(S_or, 20).cpu().numpy()
        res["diag"][B] = {"identity_R20": round(float(id_pu.mean()), 4),
                          "oracle_R20": round(float(oracle_pu.mean()), 4)}
        for A in datasets:
            enc_A, tau_A = ops[A]
            with torch.no_grad():
                S_tr = score(enc_A, Bd["E"], Bd["En"], Bd["R"], Bd["degf"], tau_A)
            tr_pu = Bd["gevT"].recall_per_user(S_tr, 20).cpu().numpy()
            bs_id = paired_bootstrap(tr_pu, id_pu)                    # transfer vs content-only identity
            gap = float(oracle_pu.mean() - id_pu.mean())
            eff = float((tr_pu.mean() - id_pu.mean()) / gap) if abs(gap) > 1e-9 else float("nan")
            key = f"{A}->{B}"
            res["transfer"][key] = {
                "transfer_R20": round(float(tr_pu.mean()), 4),
                "identity_R20": round(float(id_pu.mean()), 4),
                "oracle_R20": round(float(oracle_pu.mean()), 4),
                "transfer_efficiency": round(eff, 3),
                "vs_identity": bs_id,
            }
            ps = "<1e-3" if bs_id["p_two_sided"] < 1e-3 else f"{bs_id['p_two_sided']:.2g}"
            tag = " *SELF" if A == B else ""
            print(f"[{key}] transfer={tr_pu.mean():.4f} identity={id_pu.mean():.4f} oracle={oracle_pu.mean():.4f} "
                  f"| eff={eff:.2f} d={bs_id['mean_delta']:+.4f} p={ps}"
                  f"{'  SIG>identity' if bs_id['p_two_sided']<0.05 and bs_id['mean_delta']>0 else ''}{tag}", flush=True)
    json.dump(res, open(OUT / "w15_transfer.json", "w"), indent=2)
    print("W15_DONE", flush=True)
    return res


if __name__ == "__main__":
    run(sys.argv[1:] or ["baby", "sports", "clothing"])

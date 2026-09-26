#!/usr/bin/env python
"""Self-contained SCOPE fused-ablation trainer (writes to results/scope/exp_design/ with explicit
ti/le/lz/seed tags so it NEVER clobbers the canonical scope_*.json). Trains the set head
(softmax + le*SIGReg(E) + lz*SIGReg(z), text_init on/off, seed), fuses with the val-tuned closed-form
EASE base (gamma val-selected), tests once (trusted). Saves fused/base/head metrics + per-user fused recall.

Covers:  C7 multi-seed (ti1 le1 lz1, seeds), B4 SIGReg-in-full (le0 lz0 fused), B5 text-init-in-full (ti0 fused).
"""
import sys, json, math, random, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope_full import Rmat, build_lists, closed_form_base, SCOPE, sigreg, evalS_trusted, zr, BAR, DEV
OUT = ROOT / "results" / "scope" / "exp_design"; OUT.mkdir(parents=True, exist_ok=True)


def run(ds, le=1.0, lz=1.0, text_init=True, seed=2024, d=256, lr=3e-3, epochs=400, patience=20, bs=8192):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dset = RecDataset(Config("scope", ds)); R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    bar_r, bar_n = BAR[ds]; half = dset.n_items > 20000
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    base = closed_form_base(R, dset, gevV, half=half)
    init = None
    if text_init and dset.t_feat is not None:
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), dim=1)
        Wp = F.normalize(torch.randn(X.shape[1], d, device=DEV), dim=0); init = (X @ Wp) / math.sqrt(d)
    model = SCOPE(dset.n_items, d, init).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]; best = {"r20": -1}; bad = 0
    for ep in range(epochs):
        model.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            z, it, ctx, tgt = model.forward_train(items[b], vmask[b], deg[b]); logits = model.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1); tgt_lp = (logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)
            loss = -tgt_lp.mean() + le * sigreg(model.E) + lz * sigreg(z)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            model.eval(); S = model.score_all(R, degf); vr = gevV.recall_per_user(S, 20).mean().item(); del S
            if vr > best["r20"]:
                best = {"r20": vr, "ep": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}; bad = 0
            else: bad += 1
            if bad >= patience: break
    model.load_state_dict(best["state"]); model.eval()
    Sz = zr(model.score_all(R, degf)).to(base.dtype)
    bg = (0.0, gevV.recall_per_user(Sz, 20).mean().item())
    for g in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]:
        v = gevV.recall_per_user(Sz + g * base, 20).mean().item()
        if v > bg[1]: bg = (g, v)
    g = bg[0]
    pure = evalS_trusted(Sz, dset, "test"); fused = evalS_trusted(Sz + g * base, dset, "test")
    basef = evalS_trusted(base, dset, "test")
    r_fused = gevT.recall_per_user(Sz + g * base, 20).cpu().numpy()
    tag = f"{ds}_ti{int(text_init)}_le{le}_lz{lz}_s{seed}"
    rep = dict(dataset=ds, hp=dict(le=le, lz=lz, text_init=text_init, seed=seed, gamma=g, best_ep=best["ep"]),
               bar=dict(R20=bar_r, N20=bar_n), base=basef, pure=pure, fused=fused)
    (OUT / f"fused_{tag}.json").write_text(json.dumps(rep, indent=2, default=str))
    np.save(OUT / f"fused_{tag}_peruser.npy", r_fused)
    print(f"[{tag}] base={basef['Recall@20']:.4f} head={pure['Recall@20']:.4f} "
          f"fused R@20={fused['Recall@20']:.4f}/N@20={fused['NDCG@20']:.4f} (g={g}, ep{best['ep']})", flush=True)
    del R, base, Sz, model; torch.cuda.empty_cache()
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--le", type=float, default=1.0)
    ap.add_argument("--lz", type=float, default=1.0); ap.add_argument("--no_text_init", action="store_true")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    run(a.dataset, le=a.le, lz=a.lz, text_init=not a.no_text_init, seed=a.seed)

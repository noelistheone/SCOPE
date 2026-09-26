"""Learned per-(user,item) Mixture-of-Experts gate over frozen CF experts {EASE+text base, FREEDOM, GUME}.

Linear z-score fusion uses GLOBAL weights and throws away the per-user oracle headroom (0.13 vs stack 0.10).
This learns a soft gate g_k(u,i)=softmax(b_k + L[u,k] + M[i,k]) and scores S=sum_k g_k(u,i)*S_k(u,i), trained
end-to-end by BPR. It SUBSUMES linear fusion (L=M=0 -> global weights b) and per-user/per-item routing, so if it
can't beat linear fusion, the 0.13 oracle is genuinely un-capturable noise. If it beats linear fusion, per-user routing carries real headroom. Reports test R@20/N@20 vs linear fusion + true oracle. Leak-free (gate tuned by
train BPR, weights via valid). Writes results/scope/moe_fusion_<ds>.json. Usage: python moe_fusion.py <ds> [--steps N]
"""
from __future__ import annotations
import sys, os, json, time, argparse, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scope import Rmat, closed_form_base, zr, evalS_trusted, DEV, ROOT
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

GR = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


def load_experts(ds, dset, R, gevV, dt):
    base = zr(closed_form_base(R, dset, gevV, half=dset.n_items > 20000)).to(dt)
    experts = {"base": base}
    for name in ("freedom", "gume"):
        p = ROOT / "results" / "baseline_scores" / f"{name}_{ds}_scores.npy"
        if p.exists():
            experts[name] = zr(torch.from_numpy(np.load(p)).to(dt).to(DEV))
    return experts


def linear_fusion(experts, gevV):
    views = list(experts.values()); best = (-1.0, None)
    for ws in itertools.product(GR, repeat=len(views)):
        if all(w == 0 for w in ws): continue
        r = gevV.recall_per_user(sum(w * v for w, v in zip(ws, views))).mean().item()
        if r > best[0]: best = (r, ws)
    return best[1], sum(w * v for w, v in zip(best[1], views))


def run(ds, steps=6000, bs=2048, lr=0.05, d_reg=1e-5, seed=0):
    t0 = time.time(); torch.manual_seed(seed); np.random.seed(seed)
    dset = RecDataset(Config("scope", ds)); U, I = dset.n_users, dset.n_items; dt = torch.float16
    R = Rmat(dset); gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    experts = load_experts(ds, dset, R, gevV, dt); names = list(experts.keys()); K = len(names)
    Sk = torch.stack([experts[n].float() for n in names], 0)          # [K,U,I]  (fp32 for training stability)
    print(f"[{ds}] experts={names} U={U} I={I}", flush=True)

    # linear fusion baseline (global weights, tuned on valid)
    w_lin, S_lin = linear_fusion(experts, gevV)
    r_lin_test = evalS_trusted(S_lin.to(dt), dset, "test"); ru_lin = gevT.recall_per_user(S_lin.to(dt)).cpu().numpy()
    del S_lin; torch.cuda.empty_cache()

    # learned MoE gate: g_k(u,i)=softmax(b_k + L[u,k] + M[i,k])
    b = nn.Parameter(torch.zeros(K, device=DEV)); L = nn.Parameter(torch.zeros(U, K, device=DEV))
    M = nn.Parameter(torch.zeros(I, K, device=DEV))
    opt = torch.optim.Adam([b, L, M], lr=lr)
    idx = R.coalesce().indices(); tr_u, tr_i = idx[0], idx[1]; nnz = tr_u.numel()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        j = torch.randint(0, nnz, (bs,), device=DEV); u = tr_u[j]; ip = tr_i[j]
        jn = torch.randint(0, I, (bs,), device=DEV)
        gp = torch.softmax(b + L[u] + M[ip], -1); gn = torch.softmax(b + L[u] + M[jn], -1)   # [bs,K]
        sp = (gp * Sk[:, u, ip].t()).sum(1); sn = (gn * Sk[:, u, jn].t()).sum(1)
        loss = -F.logsigmoid(sp - sn).mean() + d_reg * (L[u].pow(2).sum() + M[ip].pow(2).sum() + M[jn].pow(2).sum()) / bs
        loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"[{ds}] step {step+1}/{steps} loss={loss.item():.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)

    # eval MoE: full S(u,i) in chunks
    def moe_scores():
        out = torch.empty(U, I, dtype=dt, device=DEV)
        with torch.no_grad():
            for s in range(0, U, 2048):
                e = min(s + 2048, U); g = torch.softmax(b[None, None] + L[s:e, None, :] + M[None, :, :], -1)  # [b,I,K]
                out[s:e] = (g * Sk[:, s:e].permute(1, 2, 0)).sum(-1).to(dt)
        return out
    S_moe = moe_scores()
    r_moe_test = evalS_trusted(S_moe, dset, "test"); ru_moe = gevT.recall_per_user(S_moe).cpu().numpy()
    del S_moe; torch.cuda.empty_cache()

    # per-user oracle (upper bound)
    ru_experts = {n: gevT.recall_per_user(experts[n]).cpu().numpy() for n in names}
    ru_oracle = np.max(np.stack(list(ru_experts.values())), 0)
    bs_moe = paired_bootstrap(ru_moe, ru_lin)
    res = {"dataset": ds, "experts": names, "linear_weights": w_lin,
           "R20": {**{n: float(ru_experts[n].mean()) for n in names},
                   "linear_fusion": r_lin_test["Recall@20"], "moe_gate": r_moe_test["Recall@20"],
                   "oracle_switch": float(ru_oracle.mean())},
           "N20": {"linear_fusion": r_lin_test["NDCG@20"], "moe_gate": r_moe_test["NDCG@20"]},
           "moe_minus_linear_R20": r_moe_test["Recall@20"] - r_lin_test["Recall@20"],
           "moe_vs_linear_bootstrap": bs_moe, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(ROOT / "results" / "scope" / f"moe_fusion_{ds}.json", "w"), indent=2)
    print(f"[{ds}] linear={r_lin_test['Recall@20']:.4f} MoE={r_moe_test['Recall@20']:.4f} "
          f"(Δ={res['moe_minus_linear_R20']:+.4f} p={bs_moe['p_two_sided']:.2g}) oracle={ru_oracle.mean():.4f} "
          f"| N@20 linear={r_lin_test['NDCG@20']:.4f} MoE={r_moe_test['NDCG@20']:.4f}", flush=True)
    del Sk, experts; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ds"); ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args(); run(a.ds, a.steps)
    print("MOE_FUSION_DONE", flush=True)

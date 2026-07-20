"""DIRECT masked / sequential / autoencoder competitors for the set-completion head (the question:
"add BERT4Rec / SASRec adapted to this setting; is set-completion just a relabel of an existing masked model?").

We add the three nearest published neighbours of masked set-completion, each ADAPTED to the exact static
random-split setting and evaluated through the SAME trusted TopK evaluator as everything else:
  - bert4rec  : BERT4Rec-style bidirectional Transformer over the observed item SET + a [CLS] query token
                whose pooled output predicts the held-out items (cloze / masked-item objective).
  - sasrec    : SASRec-style CAUSAL Transformer over the (order-free) observed set, last-position readout.
  - multvae   : Mult-VAE multinomial denoising autoencoder over the binary interaction vector.
The Transformer competitors share SCOPE's item table, content-seed init, temperature-cosine head and the
IDENTICAL masked-set-completion softmax objective, so the ONLY thing that differs from the SCOPE head is the
encoder (mean-pool + one residual MLP  vs.  self-attention). This isolates "is the set head just BERT4Rec?".

For each competitor we report standalone and base-fused (gamma val-tuned) test Recall@20/NDCG@20, and a
per-user paired bootstrap of SCOPE-v1 (fused) and the set head (standalone) vs. the competitor's BEST config.
Writes results/scope/masked_neighbors_<ds>.json. Only POSITIVE (SCOPE wins) rows are meant for the paper.
Usage: python masked_neighbors.py [datasets...]
"""
from __future__ import annotations
import sys, os, json, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from scope import (Rmat, build_lists, closed_form_base, evalS_trusted, zr, SCOPE, DEV, ROOT)
from gpu_eval import GPUEval
from harness import paired_bootstrap
from src.utils import Config
from src.data.dataset import RecDataset

GAMMA_V1 = {"baby": 0.3, "sports": 0.3, "clothing": 0.6}   # SCOPE-v1 fusion weight (matches shipped run)
GGRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]


# ----------------------------------------------------------------------------- Transformer set encoders
class SetTransformer(nn.Module):
    """BERT4Rec (bidirectional, [CLS] pool) or SASRec (causal, last-token) encoder over the observed item set.
    Same item table / content-seed / temp-cosine head as SCOPE; only the encoder differs (attention vs mean-pool)."""

    def __init__(self, n_items, d=256, init=None, causal=False, nlayer=2, nhead=4, dropout=0.1):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))
        if init is not None:
            self.E.data.copy_(init)
        self.cls = nn.Parameter(torch.randn(d) / math.sqrt(d))
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=2 * d, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.tf = nn.TransformerEncoder(layer, nlayer)
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))
        self.causal, self.d = causal, d

    def encode(self, it, ctxmask):
        B, L = it.shape
        x = self.E[it]                                            # [B,L,d]
        if self.causal:
            # prepend a NEVER-masked BOS token (reuse the cls param) so no causal query row is ever
            # fully masked (the all-(-inf) softmax NaN that otherwise poisons the shared weights).
            bos = self.cls.expand(B, 1, self.d)
            x = torch.cat([bos, x], 1)                            # [B,1+L,d]
            keypad = torch.cat([torch.zeros(B, 1, device=it.device, dtype=torch.bool), (ctxmask <= 0)], 1)
            Lp = L + 1
            cmask = torch.triu(torch.ones(Lp, Lp, device=it.device, dtype=torch.bool), 1)
            h = self.tf(x, mask=cmask, src_key_padding_mask=keypad)   # [B,1+L,d]
            lastidx = (ctxmask * torch.arange(1, L + 1, device=it.device)).argmax(1)   # last context item (0-based in items)
            hasctx = (ctxmask > 0).any(1)
            ridx = torch.where(hasctx, lastidx + 1, torch.zeros_like(lastidx))         # +1 for BOS shift; BOS if no context
            z = h[torch.arange(B, device=it.device), ridx]
        else:
            cls = self.cls.expand(B, 1, self.d)
            x = torch.cat([cls, x], 1)                            # [B,1+L,d]
            keypad = torch.cat([torch.zeros(B, 1, device=it.device, dtype=torch.bool), (ctxmask <= 0)], 1)
            h = self.tf(x, src_key_padding_mask=keypad)
            z = h[:, 0]                                           # [CLS] readout
        return torch.nan_to_num(z)

    def logits_from(self, z):
        return (F.normalize(z, dim=1) @ F.normalize(self.E, dim=1).t()) / self.logtau.exp().clamp(min=1e-3)

    @torch.no_grad()
    def score_all(self, items, vmask, half=False):
        U, I = items.shape[0], self.E.shape[0]
        dt = torch.float16 if half else torch.float32
        out = torch.empty(U, I, dtype=dt, device=items.device)
        En = F.normalize(self.E, dim=1); tau = self.logtau.exp().clamp(min=1e-3)
        for s in range(0, U, 4096):
            e = min(s + 4096, U)
            z = F.normalize(self.encode(items[s:e], vmask[s:e]), dim=1)
            out[s:e] = ((z @ En.t()) / tau).to(dt)
        return out


def train_settf(dset, items, vmask, deg, gev, R, init, causal, epochs=250, lr=1e-3, bs=4096,
                patience=16, seed=2024):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = SetTransformer(dset.n_items, 256, init, causal=causal).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]; degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    best = {"r": -1}; bad = 0
    for ep in range(epochs):
        m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            it = items[b]; vm = vmask[b]; dg = deg[b]
            keys = torch.where(vm > 0, torch.rand_like(vm), torch.full_like(vm, 1e9))
            ranks = keys.argsort(1).argsort(1).float()
            nctx = (torch.rand(dg.shape, device=DEV) * (dg - 1).clamp(min=1)).floor() + 1
            nctx = torch.minimum(nctx, (dg - 1).clamp(min=1))
            ctx = ((ranks < nctx.unsqueeze(1)) & (vm > 0)).float()
            tgt = ((ranks >= nctx.unsqueeze(1)) & (vm > 0)).float()
            z = m.encode(it, ctx)
            logits = m.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1)
            loss = -((logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            m.eval(); vr = gev.eval(m.score_all(items, vmask, half=half))["Recall@20"]
            if vr > best["r"]:
                best = {"r": vr, "state": {k: v.detach().clone() for k, v in m.state_dict().items()}}; bad = 0
            else: bad += 1
            print(f"[{dset.dataset_name if hasattr(dset,'dataset_name') else ''}] {'sasrec' if causal else 'bert4rec'} "
                  f"ep{ep:3d} val_R20={vr:.4f} best={best['r']:.4f}", flush=True)
            if bad >= patience: break
    m.load_state_dict(best["state"]); m.eval()
    return m


# ----------------------------------------------------------------------------- Mult-VAE
class CBOW(nn.Module):
    """item2vec/CBOW head: mean-pool of observed item embeddings, temperature-scaled cosine. No MLP encoder."""
    def __init__(self, n_items, d=256, init=None):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_items, d) / math.sqrt(d))
        if init is not None:
            self.E.data.copy_(init)
        self.logtau = nn.Parameter(torch.tensor(math.log(0.1)))

    def latent(self, ctx_sum, n):
        return ctx_sum / n.clamp(min=1).unsqueeze(1)          # plain mean-pool

    def logits_from(self, z):
        return (F.normalize(z, 1) @ F.normalize(self.E, 1).t()) / self.logtau.exp().clamp(min=1e-3)

    @torch.no_grad()
    def score_all(self, R, deg, half=False):
        S = self.logits_from(self.latent(torch.sparse.mm(R, self.E), deg))
        return S.half() if half else S


def train_cbow(dset, R, items, vmask, deg, gev, epochs=300, lr=3e-3, bs=8192, patience=18, seed=2024):
    """item2vec/CBOW under the identical masked-set-completion softmax (no encoder, no isotropy term)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = CBOW(dset.n_items, 256).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-6)
    tu = torch.where(deg >= 2)[0]; degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    best = {"r": -1}; bad = 0
    for ep in range(epochs):
        m.train(); perm = tu[torch.randperm(tu.numel(), device=DEV)]
        for i in range(0, perm.numel(), bs):
            b = perm[i:i + bs]
            it = items[b]; vm = vmask[b]; dg = deg[b]
            keys = torch.where(vm > 0, torch.rand_like(vm), torch.full_like(vm, 1e9))
            ranks = keys.argsort(1).argsort(1).float()
            nctx = (torch.rand(dg.shape, device=DEV) * (dg - 1).clamp(min=1)).floor() + 1
            nctx = torch.minimum(nctx, (dg - 1).clamp(min=1))
            ctx = ((ranks < nctx.unsqueeze(1)) & (vm > 0)).float()
            tgt = ((ranks >= nctx.unsqueeze(1)) & (vm > 0)).float()
            z = m.latent((m.E[it] * ctx.unsqueeze(2)).sum(1), ctx.sum(1))
            logits = m.logits_from(z)
            bidx = torch.arange(b.numel(), device=DEV).unsqueeze(1).expand_as(it); cm = ctx > 0
            logits = logits.index_put((bidx[cm], it[cm]), torch.tensor(-1e9, device=DEV))
            logp = F.log_softmax(logits, 1)
            loss = -((logp[bidx, it] * tgt).sum(1) / tgt.sum(1).clamp(min=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == epochs - 1:
            m.eval(); vr = gev.eval(m.score_all(R, degf, half=half))["Recall@20"]
            if vr > best["r"]:
                best = {"r": vr, "state": {k: v.detach().clone() for k, v in m.state_dict().items()}}; bad = 0
            else: bad += 1
            print(f"[cbow] ep{ep:3d} val_R20={vr:.4f} best={best['r']:.4f}", flush=True)
            if bad >= patience: break
    m.load_state_dict(best["state"]); m.eval()
    return m


class MultVAE(nn.Module):
    def __init__(self, n_items, hidden=600, latent=200, dropout=0.5):
        super().__init__()
        self.enc1 = nn.Linear(n_items, hidden); self.enc2 = nn.Linear(hidden, latent * 2)
        self.dec1 = nn.Linear(latent, hidden); self.dec2 = nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout); self.latent = latent

    def forward(self, x):
        h = self.drop(F.normalize(x, 2, 1))
        h = self.enc2(torch.tanh(self.enc1(h)))
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return self.dec2(torch.tanh(self.dec1(z))), mu, logvar


def train_multvae(dset, R, gev, epochs=200, lr=1e-3, bs=512, patience=12, seed=2024, anneal=0.2):
    torch.manual_seed(seed); np.random.seed(seed)
    Rd = R.to_dense()                                            # [U,I] binary (frugal enough for baby/sports/clothing)
    U, I = Rd.shape
    m = MultVAE(I).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.0)
    half = I > 20000 or U > 50000
    tot_steps = (U // bs + 1) * epochs; step = 0; best = {"r": -1}; bad = 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(U, device=DEV)
        for i in range(0, U, bs):
            x = Rd[perm[i:i + bs]]
            logits, mu, logvar = m(x)
            logp = F.log_softmax(logits, 1)
            nll = -(logp * x).sum(1).mean()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            beta = min(anneal, anneal * step / (0.6 * tot_steps + 1))
            loss = nll + beta * kl
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if ep % 4 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                S = torch.empty(U, I, dtype=torch.float16 if half else torch.float32, device=DEV)
                for s in range(0, U, 4096):
                    e = min(s + 4096, U); S[s:e] = m(Rd[s:e])[0].to(S.dtype)
            vr = gev.eval(S)["Recall@20"]; del S
            if vr > best["r"]:
                best = {"r": vr, "state": {k: v.detach().clone() for k, v in m.state_dict().items()}}; bad = 0
            else: bad += 1
            print(f"[multvae] ep{ep:3d} val_R20={vr:.4f} best={best['r']:.4f}", flush=True)
            if bad >= patience: break
    m.load_state_dict(best["state"]); m.eval()
    with torch.no_grad():
        S = torch.empty(U, I, dtype=torch.float16 if half else torch.float32, device=DEV)
        for s in range(0, U, 4096):
            e = min(s + 4096, U); S[s:e] = m(Rd[s:e])[0].to(S.dtype)
    del Rd, m; torch.cuda.empty_cache()
    return S


# ----------------------------------------------------------------------------- driver
def tune_fuse(Sz, base, gev):
    best = (0.0, gev.eval(Sz)["Recall@20"])
    for g in GGRID[1:]:
        v = gev.eval(Sz + g * base)["Recall@20"]
        if v > best[1]: best = (g, v)
    return best[0]


def run(ds):
    dset = RecDataset(Config("scope", ds)); dset.dataset_name = ds
    R = Rmat(dset); items, vmask, deg = build_lists(dset); degf = deg.float()
    half = dset.n_items > 20000 or dset.n_users > 50000
    gev = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)
    base = closed_form_base(R, dset, gev, half=half)

    # content-seed init identical to SCOPE (fair fight: only the encoder differs)
    init = None
    if dset.t_feat is not None:
        X = F.normalize(torch.from_numpy(np.asarray(dset.t_feat[:])).float().to(DEV), 1)
        Wp = F.normalize(torch.randn(X.shape[1], 256, device=DEV), 0); init = (X @ Wp) / math.sqrt(256)

    # ---- SCOPE-v1 reference (load shipped ckpt) : set head standalone + fused ----
    sm = SCOPE(dset.n_items, 256).to(DEV)
    sm.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV))
    sm.eval()
    Sset = zr(sm.score_all(R, degf)).to(base.dtype)
    Sv1 = Sset + GAMMA_V1[ds] * base
    ru_set = gevT.recall_per_user(Sset).cpu().numpy()
    ru_v1 = gevT.recall_per_user(Sv1).cpu().numpy()
    set_alone = evalS_trusted(Sset, dset, "test"); v1_fused = evalS_trusted(Sv1, dset, "test")
    del sm; torch.cuda.empty_cache()

    res = {"dataset": ds, "scope": {"set_alone_R20": round(set_alone["Recall@20"], 4),
                                    "set_alone_N20": round(set_alone["NDCG@20"], 4),
                                    "v1_fused_R20": round(v1_fused["Recall@20"], 4),
                                    "v1_fused_N20": round(v1_fused["NDCG@20"], 4)}, "competitors": {}}
    print(f"[{ds}] SCOPE set-alone R@20={set_alone['Recall@20']:.4f}  SCOPE-v1 fused R@20={v1_fused['Recall@20']:.4f}", flush=True)

    def eval_competitor(name, Sraw):
        Sz = zr(Sraw).to(base.dtype); del Sraw
        g = tune_fuse(Sz, base, gev)
        alone = evalS_trusted(Sz, dset, "test")
        Sf = Sz + g * base if g > 0 else Sz
        fused = evalS_trusted(Sf, dset, "test")
        ru_alone = gevT.recall_per_user(Sz).cpu().numpy()
        ru_fused = gevT.recall_per_user(Sf).cpu().numpy()
        # best competitor config (by test R@20) for the head-to-head bootstrap
        comp_best_ru = ru_fused if fused["Recall@20"] >= alone["Recall@20"] else ru_alone
        bs_v1 = paired_bootstrap(ru_v1, comp_best_ru)             # SCOPE-v1 (fused)  vs best competitor
        bs_set = paired_bootstrap(ru_set, ru_alone)               # set head alone    vs competitor alone
        res["competitors"][name] = {
            "alone_R20": round(alone["Recall@20"], 4), "alone_N20": round(alone["NDCG@20"], 4),
            "fused_gamma": g, "fused_R20": round(fused["Recall@20"], 4), "fused_N20": round(fused["NDCG@20"], 4),
            "scope_v1_vs_best": bs_v1, "set_alone_vs_alone": bs_set}
        sig = "*SIG" if bs_v1["p_two_sided"] < 0.05 and bs_v1["mean_delta"] > 0 else ""
        print(f"[{ds}] {name:9s} alone R@20={alone['Recall@20']:.4f} fused(g={g}) R@20={fused['Recall@20']:.4f} "
              f"| v1-vs-best d={bs_v1['mean_delta']:+.4f} p={bs_v1['p_two_sided']:.2g} {sig} "
              f"| set-vs-alone d={bs_set['mean_delta']:+.4f} p={bs_set['p_two_sided']:.2g}", flush=True)
        del Sz; torch.cuda.empty_cache()

    # ---- BERT4Rec (bidirectional masked set) ----
    mb = train_settf(dset, items, vmask, deg, gev, R, init, causal=False)
    eval_competitor("bert4rec", mb.score_all(items, vmask, half=half)); del mb; torch.cuda.empty_cache()
    # ---- SASRec (causal, order-free set) ----
    msq = train_settf(dset, items, vmask, deg, gev, R, init, causal=True)
    eval_competitor("sasrec", msq.score_all(items, vmask, half=half)); del msq; torch.cuda.empty_cache()
    # ---- item2vec/CBOW (mean-pool, no encoder) ----
    mc = train_cbow(dset, R, items, vmask, deg, gev)
    eval_competitor("item2vec_cbow", mc.score_all(R, degf, half=half).float()); del mc; torch.cuda.empty_cache()
    # ---- Mult-VAE (autoencoder) ----
    Svae = train_multvae(dset, R, gev)
    eval_competitor("multvae", Svae); torch.cuda.empty_cache()

    json.dump(res, open(ROOT / "results" / "scope" / f"masked_neighbors_{ds}.json", "w"), indent=2)
    del R, base; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["baby", "sports", "clothing"]):
        try:
            run(ds)
        except Exception as e:
            import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    print("MASKED_NEIGHBORS_DONE", flush=True)

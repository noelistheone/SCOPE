"""Item-item co-purchase decomposition for the SCOPE paper (self-contained, paper's own MMRec features).

For each modality (image = 4096-d CNN, text = 384-d SBERT, joint = concat) we ask: how well does that
modality's geometry predict whether two items are CO-PURCHASED (co-occur in some user's history)?
We report TWO AUCs on held-out co-purchase pairs (80/20 split):
  - raw_cos : AUC of the raw l2-normalized feature cosine (the geometry the frozen kNN item-graphs use)
  - learned : AUC of a small learned metric (BPR-trained projection that pulls co-purchased items together)
The gap raw->learned shows the signal a frozen cosine hides but a learned geometry exposes. The modality
ordering (image vs text) at the item-item level is the pair-level half of the task-formulation flip:
strong item-item content signal that is nonetheless redundant with CF at user-item ranking (Table ensemble).

Uses ONLY data/<ds>/{image_feat,text_feat}.npy + <ds>.inter (the same features every baseline uses).
Writes results/scope/copurchase_auc_<ds>.json. GPU. Usage: python copurchase_auc.py <ds=baby>
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scope"
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024
torch.manual_seed(SEED); np.random.seed(SEED)
gcpu = torch.Generator(device="cpu").manual_seed(SEED)


def l2(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def load(ds):
    im = np.load(ROOT / "data" / ds / "image_feat.npy").astype(np.float32)   # 4096-d CNN (MMRec standard)
    tx = np.load(ROOT / "data" / ds / "text_feat.npy").astype(np.float32)    # 384-d SBERT
    feats = {"image": l2(im), "text": l2(tx), "joint": np.concatenate([l2(im), l2(tx)], 1)}
    rows = np.loadtxt(ROOT / "data" / ds / f"{ds}.inter", delimiter="\t", skiprows=1,
                      dtype=np.int64, usecols=(0, 1, 4))
    train = rows[rows[:, 2] == 0][:, :2]                                     # x_label==0 -> train
    return feats, train, im.shape[0]


def pairs(train):
    ub = defaultdict(list)
    for u, i in train:
        ub[int(u)].append(int(i))
    P = set()
    for items in ub.values():
        items = list(set(items))
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                x, y = items[a], items[b]
                P.add((x, y) if x < y else (y, x))
    return np.array(sorted(P), dtype=np.int64)


class Proj(nn.Module):
    def __init__(self, din, d=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, d))

    def forward(self, x):
        return F.normalize(self.f(x), dim=1)


def auc(sp, sn):
    s = torch.cat([sp, sn]); y = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
    order = torch.argsort(s); ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(s), device=s.device, dtype=torch.float)
    npos = sp.numel(); nneg = sn.numel()
    return ((ranks[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)).item()


def raw_cos_auc(feat, pte, N):
    Fm = torch.tensor(feat, device=dev)
    with torch.no_grad():
        na = torch.randint(0, N, (pte.shape[0],), device=dev)
        nb = torch.randint(0, N, (pte.shape[0],), device=dev)
        sp = (Fm[pte[:, 0]] * Fm[pte[:, 1]]).sum(1)
        sn = (Fm[na] * Fm[nb]).sum(1)
    return auc(sp, sn)


def learned_auc(feat, ptr, pte, N, epochs=400):
    Fm = torch.tensor(feat, device=dev)
    proj = Proj(Fm.shape[1]).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        na = torch.randint(0, N, (ptr.shape[0],), device=dev)
        nb = torch.randint(0, N, (ptr.shape[0],), device=dev)
        z = proj(Fm)
        loss = -(F.logsigmoid((z[ptr[:, 0]] * z[ptr[:, 1]]).sum(1) - (z[na] * z[nb]).sum(1))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    with torch.no_grad():
        z = proj(Fm)
        na = torch.randint(0, N, (pte.shape[0],), device=dev)
        nb = torch.randint(0, N, (pte.shape[0],), device=dev)
        return auc((z[pte[:, 0]] * z[pte[:, 1]]).sum(1), (z[na] * z[nb]).sum(1))


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "baby"
    OUT.mkdir(parents=True, exist_ok=True)
    feats, train, n = load(ds)
    P = pairs(train)
    perm = torch.randperm(len(P), generator=gcpu).numpy(); P = P[perm]
    cut = int(0.8 * len(P)); ptr = torch.tensor(P[:cut], device=dev); pte = torch.tensor(P[cut:], device=dev)
    print(f"{ds}: {n} items, {len(P)} co-purchase pairs", flush=True)
    res = {"dataset": ds, "n_items": int(n), "n_pairs": int(len(P)), "raw_cos": {}, "learned": {}}
    for k, fv in feats.items():
        rc = raw_cos_auc(fv, pte, n)
        la = float(np.mean([learned_auc(fv, ptr, pte, n) for _ in range(2)]))
        res["raw_cos"][k] = float(rc); res["learned"][k] = la
        print(f"  {k:6s}  raw-cos AUC={rc:.4f}   learned-metric AUC={la:.4f}", flush=True)
    res["delta_image_beyond_text_learned"] = res["learned"]["joint"] - res["learned"]["text"]
    res["delta_text_beyond_image_learned"] = res["learned"]["joint"] - res["learned"]["image"]
    res["image_minus_text_learned"] = res["learned"]["image"] - res["learned"]["text"]
    print(f"  image-text (learned) = {res['image_minus_text_learned']:+.4f} | "
          f"Δimg beyond txt = {res['delta_image_beyond_text_learned']:+.4f} | "
          f"Δtxt beyond img = {res['delta_text_beyond_image_learned']:+.4f}", flush=True)
    (OUT / f"copurchase_auc_{ds}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote copurchase_auc_{ds}.json", flush=True)


if __name__ == "__main__":
    main()

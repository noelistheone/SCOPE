"""(1) Leakage-screen POSITIVE control: show the screen actually catches contamination (MLLMRec features),
    so the 'all pathways clean' result is meaningful, not vacuous.
(2) New-item INDUCTIVE ceiling: give the 0.046 new-item Recall@20 a reference -- the same items' Recall@20
    when the model IS trained on them (in-sample ceiling).
Writes results/scope/leak_pos_ceiling.json. GPU. Usage: python leak_pos_ceiling.py
"""
from __future__ import annotations
import json, math
import numpy as np, torch, torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, ROOT
from gpu_eval import GPUEval
from src.utils import Config
from src.data.dataset import RecDataset
from leakage_protocol import content_affinity, split_means
from inductive_coldstart import recall_at_k_restricted

DS = "baby"
dset = RecDataset(Config("scope", DS))
R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

# ---- (1) MLLMRec positive control: alignment of an MLLMRec-feature affinity ----
usersT = gevT.users
train_pos = {}
hu = gevT.hist_u.cpu().numpy(); hi = gevT.hist_i.cpu().numpy()
for u, i in zip(hu, hi): train_pos.setdefault(int(u), []).append(int(i))
Pmax = max(len(v) for v in train_pos.values())
tp = torch.full((usersT.numel(), Pmax), -1, dtype=torch.long, device=DEV)
for r, u in enumerate(usersT.cpu().numpy()):
    it = train_pos.get(int(u), [])
    if it: tp[r, :len(it)] = torch.tensor(it[:Pmax], device=DEV)

out = {"dataset": DS, "mllmrec_positive_control": {}, "clean_reference": {}}
for nm, path in [("MLLMRec_bespoke", ROOT / "data" / DS / "mllm_item_feat.npy"),
                 ("text_standard", ROOT / "data" / DS / "text_feat.npy"),
                 ("image_standard", ROOT / "data" / DS / "image_feat.npy")]:
    S = content_affinity(path, R)
    tr = split_means(S, usersT, tp); te = split_means(S, gevT.users, gevT.pos); va = split_means(S, gevV.users, gevV.pos)
    rec = {"train": round(tr, 3), "valid": round(va, 3), "test": round(te, 3),
           "test_minus_valid": round(te - va, 4), "test_minus_train": round(te - tr, 4),
           "clean(|test-val|<0.076)": bool(abs(te - va) < 0.076)}
    (out["mllmrec_positive_control"] if "MLLM" in nm else out["clean_reference"])[nm] = rec
    print(f"  [{nm:16s}] train={tr:+.2f} val={va:+.2f} test={te:+.2f}  |test-val|={abs(te-va):.3f}  test-train={te-tr:+.2f}  clean={rec['clean(|test-val|<0.076)']}", flush=True)
    del S

# ---- (2) new-item in-sample ceiling: same held-out 20% items, but model trained on all ----
n = dset.n_items
g = torch.Generator(device=DEV).manual_seed(2024)
H = torch.zeros(n, dtype=torch.bool, device=DEV)
H[torch.randperm(n, generator=g, device=DEV)[: int(0.2 * n)]] = True       # identical H to inductive_coldstart.new_item
m = SCOPE(n, 256).to(DEV)
m.load_state_dict(torch.load(ROOT / "ckpts" / "scope" / f"scope_{DS}_d256_le1.0_lz1.0_lr0.003.pt", map_location=DEV)); m.eval()
base = zr(closed_form_base(R, dset, gevV, half=False))
with torch.no_grad():
    S_full = zr(m.logits_from(m.latent(torch.sparse.mm(R, m.E), degf))) + 0.6 * base   # full SCOPE-v1 (trained on ALL items)
ceiling = recall_at_k_restricted(S_full, gevT, H)
out["newitem_ceiling"] = {"inductive_content_base_R20": 0.0457, "in_sample_ceiling_R20": round(ceiling, 4),
                          "retention": round(0.0457 / max(ceiling, 1e-9), 3),
                          "note": "0.046 is the content base on items HELD OUT of training; ceiling is the same items' R@20 when trained on (full SCOPE-v1). ID-CF scores held-out items at chance."}
print(f"  [new-item ceiling] inductive content-base 0.046 vs in-sample ceiling {ceiling:.4f} (retention {0.0457/max(ceiling,1e-9):.2f})", flush=True)
json.dump(out, open(ROOT / "results" / "scope" / "leak_pos_ceiling.json", "w"), indent=2)
print("LEAK_POS_CEILING_DONE", flush=True)

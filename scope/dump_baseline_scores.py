#!/usr/bin/env python
"""Dump a trained baseline's full (n_users, n_items) raw score matrix to
results/baseline_scores/{model}_{dataset}_scores.npy — same format/location as the
existing baseline dumps, so the SCOPE fusion harness can consume it as a view.

Constructs the model EXACTLY as src/main.py does (passing train_user_idx/
train_item_idx for graph models like GUME), then loads the most-recent matching
checkpoint and runs full_sort_predict
batched on GPU. Raw scores (train items NOT masked — downstream consumers mask).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import RecDataset
from src.data.graph_utils import build_norm_adj
from src.models import get_model
from src.utils.configurator import Config
from src.utils.seed import set_seed

# models whose __init__ needs the raw train edges (matches src/main.py)
NEEDS_EDGES = ("freedom", "mllmrec", "grcn", "dragon", "smore",
               "damrs", "gume", "cohesion")
NO_FEATS = ("lightgcn", "mllmrec")


def find_ckpt(model, dataset):
    cands = sorted((ROOT / "ckpts").glob(f"{model}_{dataset}_*.pt"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint ckpts/{model}_{dataset}_*.pt")
    return cands[-1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"

    cfg = Config(a.model, a.dataset, cli_overrides={"seed": a.seed})
    set_seed(int(cfg.get("seed", a.seed)))
    rec = RecDataset(cfg)
    norm_adj = build_norm_adj(rec.train_matrix, rec.n_users, rec.n_items)

    ml = a.model.lower()
    kw = dict(config=cfg, n_users=rec.n_users, n_items=rec.n_items, norm_adj=norm_adj)
    if ml not in NO_FEATS:
        kw.update(v_feat=torch.from_numpy(rec.v_feat[:].copy()) if rec.v_feat is not None else None,
                  t_feat=torch.from_numpy(rec.t_feat[:].copy()) if rec.t_feat is not None else None)
    if ml in NEEDS_EDGES:
        kw.update(train_user_idx=torch.from_numpy(rec.train_users),
                  train_item_idx=torch.from_numpy(rec.train_items))

    model = get_model(a.model)(**kw).to(dev)
    ckpt = find_ckpt(a.model, a.dataset)
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[{a.model}/{a.dataset}] loaded {ckpt.name} (ep={state.get('epoch','?')}) "
          f"n_users={rec.n_users} n_items={rec.n_items}", flush=True)

    out = np.empty((rec.n_users, rec.n_items), dtype=np.float32)
    t0 = time.time()
    for s in range(0, rec.n_users, 512):
        e = min(s + 512, rec.n_users)
        u = torch.arange(s, e, device=dev, dtype=torch.long)
        out[s:e] = model.full_sort_predict({"user": u}).float().cpu().numpy()
    pth = ROOT / "results" / "baseline_scores" / f"{a.model}_{a.dataset}_scores.npy"
    pth.parent.mkdir(parents=True, exist_ok=True)
    np.save(pth, out)
    print(f"  saved {pth.name} {out.shape} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

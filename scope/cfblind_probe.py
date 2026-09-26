"""CF-blind content probe: a control for the content-vs-CF redundancy analysis.

The co-purchase AUC could be high merely because content correlates with the collaborative (co-purchase)
structure. To show content carries item structure BEYOND co-purchase, we repeat the learned-metric probe
on a target NOT defined by co-occurrence: do two items share the same fine-grained CATEGORY (Sports/Clothing)
or BRAND (Baby, which has no category granularity)? These are 'what the item is', independent of what is
bought together. If image still beats text (and beats raw cosine) at predicting same-category/brand, then
content's structure is not just a shadow of co-purchase. Same learned metric, same capacity (d=128) for both
modalities, held-out 80/20 pair split. Writes results/scope/cfblind_probe_<ds>.json. GPU.
"""
from __future__ import annotations
import sys, os, json, ast
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from copurchase_auc import l2, Proj, auc, raw_cos_auc, learned_auc, dev as DEV  # reuse identical metric/protocol
ROOT = __import__("scope").ROOT
SEED = 2024
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device="cpu").manual_seed(SEED)


def labels(ds, n_items):
    m = pd.read_csv(ROOT / "data" / ds / f"meta-{ds}.csv")
    lab = np.full(n_items, -1, dtype=np.int64); kind = "brand" if ds == "baby" else "category"
    def fine(x):
        try:
            c = ast.literal_eval(x)
            if c and isinstance(c, list) and isinstance(c[0], list): return c[0][-1]
        except Exception: pass
        return None
    raw = m["brand"].fillna("").astype(str) if ds == "baby" else m["categories"].apply(fine)
    uniq = {}
    for iid, v in zip(m["itemID"].values, raw):
        if iid >= n_items: continue
        if (isinstance(v, str) and len(v) > 0) or (v is not None and not isinstance(v, float)):
            lab[int(iid)] = uniq.setdefault(str(v), len(uniq))
    return lab, kind, len(uniq)


def same_label_pairs(lab, max_pairs=900000):
    from collections import defaultdict
    groups = defaultdict(list)
    for i, l in enumerate(lab):
        if l >= 0: groups[l].append(i)
    P = []
    for items in groups.values():
        if len(items) < 2: continue
        items = np.array(items)
        # cap per group to avoid quadratic blow-up: sample up to ~ items pairs per group
        k = min(len(items) * 4, len(items) * (len(items) - 1) // 2)
        a = np.random.randint(0, len(items), size=k); b = np.random.randint(0, len(items), size=k)
        m = a != b
        P.append(np.stack([items[a[m]], items[b[m]]], 1))
    P = np.concatenate(P, 0)
    P = np.unique(np.sort(P, 1), axis=0)
    if len(P) > max_pairs:
        P = P[np.random.choice(len(P), max_pairs, replace=False)]
    return P


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "sports"
    im = l2(np.load(ROOT / "data" / ds / "image_feat.npy").astype(np.float32))
    tx = l2(np.load(ROOT / "data" / ds / "text_feat.npy").astype(np.float32))
    n = im.shape[0]
    lab, kind, ncls = labels(ds, n)
    P = same_label_pairs(lab); perm = torch.randperm(len(P), generator=g).numpy(); P = P[perm]
    cut = int(0.8 * len(P)); ptr = torch.tensor(P[:cut], device=DEV); pte = torch.tensor(P[cut:], device=DEV)
    print(f"{ds}: CF-blind target={kind} ({ncls} classes, {int((lab>=0).sum())}/{n} labelled), {len(P)} same-{kind} pairs", flush=True)
    res = {"dataset": ds, "cf_blind_target": kind, "n_classes": ncls, "n_pairs": int(len(P)), "raw_cos": {}, "learned": {}}
    for name, fv in [("image", im), ("text", tx)]:
        res["raw_cos"][name] = float(raw_cos_auc(fv, pte, n))
        res["learned"][name] = float(np.mean([learned_auc(fv, ptr, pte, n) for _ in range(2)]))
        print(f"  {name:6s}  raw-cos AUC={res['raw_cos'][name]:.4f}   learned AUC={res['learned'][name]:.4f}", flush=True)
    res["image_minus_text_learned"] = res["learned"]["image"] - res["learned"]["text"]
    print(f"  image-text (learned, CF-blind) = {res['image_minus_text_learned']:+.4f}", flush=True)
    json.dump(res, open(ROOT / "results" / "scope" / f"cfblind_probe_{ds}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

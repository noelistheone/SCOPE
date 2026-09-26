#!/usr/bin/env python
"""Dataset characteristics for the 4 MMRec Amazon datasets (CPU, fast). Informs the design:
sparsity, user/item degree distributions, cold-item/cold-user fraction, train/test degree of
test items (the 'reachability' of held-out items via co-occurrence), Gini of item popularity.
x_label: 0=train, 1=valid, 2=test."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scope"; OUT.mkdir(parents=True, exist_ok=True)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = x.size
    if n == 0 or x.sum() == 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def analyze(ds):
    df = pd.read_csv(ROOT / "data" / ds / f"{ds}.inter", sep="\t")
    tr = df[df.x_label == 0]; va = df[df.x_label == 1]; te = df[df.x_label == 2]
    n_u = df.userID.max() + 1; n_i = df.itemID.max() + 1
    udeg = tr.groupby("userID").size().reindex(range(n_u), fill_value=0).values
    ideg = tr.groupby("itemID").size().reindex(range(n_i), fill_value=0).values
    test_items = te.itemID.unique()
    # how many test items are "cold" (degree 0 in train -> unreachable by co-occurrence)
    cold_test_items = int((ideg[test_items] == 0).sum())
    # test-user train degree distribution (cold users)
    test_users = te.userID.unique()
    tu_deg = udeg[test_users]
    r = {
        "dataset": ds, "n_users": int(n_u), "n_items": int(n_i),
        "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
        "density_%": round(100 * len(tr) / (n_u * n_i), 4),
        "avg_user_deg": round(float(udeg.mean()), 2), "med_user_deg": int(np.median(udeg)),
        "avg_item_deg": round(float(ideg.mean()), 2), "med_item_deg": int(np.median(ideg)),
        "item_pop_gini": round(gini(ideg), 3), "user_deg_gini": round(gini(udeg), 3),
        "items_deg_le5_%": round(100 * float((ideg <= 5).mean()), 1),
        "items_deg_le1_%": round(100 * float((ideg <= 1).mean()), 1),
        "cold_test_items(deg0)": cold_test_items,
        "test_user_med_traindeg": int(np.median(tu_deg)),
        "test_user_deg_le5_%": round(100 * float((tu_deg <= 5).mean()), 1),
    }
    return r


if __name__ == "__main__":
    rows = []
    for ds in ["baby", "sports", "clothing", "elec"]:
        try:
            r = analyze(ds); rows.append(r)
            print(json.dumps(r), flush=True)
        except Exception as e:
            print(f"[{ds}] ERR {e}")
    (OUT / "dataset_stats.json").write_text(json.dumps(rows, indent=2))

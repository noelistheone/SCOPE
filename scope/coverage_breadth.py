#!/usr/bin/env python
"""Per-pair coverage and per-user breadth analyses of SCOPE-U (paper Figures S3-S4).

Each held-out (user, item) test pair is the unit of analysis. The three z-scored views of
SCOPE-U -- the collaborative view (GUME, cached scores), the closed-form item view
(EASE + text base) and the set-completion view (SCOPE head) -- are compared at the
top-20 cutoff, train-masked, on the test split:

  Figure S3 (per-pair coverage): the per-pair top-20 miss rate -- the fraction of held-out
    pairs absent from a model's top-20 list -- for GUME, the base, the set head and
    SCOPE-U, together with the best-view-per-pair oracle: the union of the three views'
    top-20 lists, i.e. the coverage an ideal per-pair view router would attain. The share
    of that oracle coverage captured by SCOPE-U's single static gate is also reported.

  Figure S4 (per-user breadth): SCOPE-U vs. the set-free base+GUME ensemble (gate selected
    under the identical protocol): the fraction of users helped / tied / hurt by adding
    the set view, the fraction left no worse off (helped or tied), and the fraction whose
    top-20 list is unchanged.

Gate weights are read from results/scope/ensemble_control_{ds}.json when available (rows
"GUMEswap(base+set+GUME)" and "base+GUME(no set)", produced by ensemble_control.py);
otherwise they are re-selected on validation Recall@20 over the same nonnegative grid.
Inputs, all produced inside this repository: the SCOPE checkpoint
ckpts/scope/scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt (scope.py), the closed-form base
(computed here, validation-tuned), and the cached baseline score matrix
results/baseline_scores/gume_{ds}_scores.npy (dump_baseline_scores.py).

Results -> results/scope/coverage_breadth_{ds}.json
Figures -> results/scope/fig_coverage.pdf (S3), results/scope/fig_breadth.pdf (S4)

Usage: python scope/coverage_breadth.py --datasets baby sports clothing
"""
from __future__ import annotations
import sys, json, argparse, itertools
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import logging; logging.disable(logging.INFO)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.utils import Config
from src.data.dataset import RecDataset
from gpu_eval import GPUEval
from scope import Rmat, build_lists, closed_form_base, SCOPE, zr, DEV, OUT, CK

PRED = ROOT / "results" / "baseline_scores"
GRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]        # shared gate grid (ensemble_control.py)
DS_ORDER = ["baby", "sports", "clothing"]
DS_LABEL = {"baby": "Baby", "sports": "Sports", "clothing": "Clothing"}
METHODS = ["GUME", "base", "set", "SCOPE-U"]      # bars of Figure S3, in plot order
METHOD_LABEL = {"GUME": "GUME", "base": "Base", "set": "Set head", "SCOPE-U": "SCOPE-U"}
COLORS = {"GUME": "#90A4AE", "base": "#7E57C2", "set": "#43A047", "SCOPE-U": "#1E88E5"}


@torch.no_grad()
def topk_ids(gev, S, k=20, batch=4096):
    """Train-masked top-k item ids per evaluated user, [U,k], aligned with gev.users order."""
    U = gev.users.numel()
    out = torch.empty(U, k, dtype=torch.long, device=DEV)
    for s in range(0, U, batch):
        bu = gev.users[s:s + batch]
        sc = gev._mask(S[bu].clone().float(), bu)
        out[s:s + bu.numel()] = torch.topk(sc, k, 1).indices
    return out


@torch.no_grad()
def hit_mask(gev, tk, batch=8192):
    """[U,P] bool: which of each user's held-out positives appear in their top-k list."""
    U = gev.pos.shape[0]
    out = torch.zeros(U, gev.pos.shape[1], dtype=torch.bool, device=DEV)
    for s in range(0, U, batch):
        p = gev.pos[s:s + batch]; t = tk[s:s + batch]
        out[s:s + p.shape[0]] = (t.unsqueeze(2) == p.unsqueeze(1)).any(1) & (p >= 0)
    return out


def select_gates(ds, views, gevV):
    """Gate weights for SCOPE-U [gume,item,set] and the set-free ensemble [gume,item].
    Read from the ensemble-control result when present; otherwise re-select on validation."""
    f = OUT / f"ensemble_control_{ds}.json"
    if f.exists():
        rows = json.load(open(f))["rows"]
        gU = [float(x) for x in rows["GUMEswap(base+set+GUME)"]["gate"]]
        g2 = [float(x) for x in rows["base+GUME(no set)"]["gate"]]
        return gU, g2, "ensemble_control"
    def sel(keys):
        best = None
        for ws in itertools.product(GRID, repeat=len(keys)):
            if all(w == 0 for w in ws): continue
            S = sum(w * views[k] for w, k in zip(ws, keys))
            vr = gevV.eval(S)["Recall@20"]
            if best is None or vr > best[0]: best = (vr, ws)
            del S
        return [float(w) for w in best[1]]
    return sel(["gume", "item", "set"]), sel(["gume", "item"]), "validation-grid"


def run(ds):
    dset = RecDataset(Config("scope", ds))
    half = dset.n_items > 20000 or dset.n_users > 50000
    dt = torch.float16 if half else torch.float32
    R = Rmat(dset); _, _, deg = build_lists(dset); degf = deg.float()
    gevV = GPUEval(dset, "valid", DEV); gevT = GPUEval(dset, "test", DEV)

    # the three z-scored views
    V_item = zr(closed_form_base(R, dset, gevV, half=half)).to(dt)
    m = SCOPE(dset.n_items, 256).to(DEV)
    m.load_state_dict(torch.load(CK / f"scope_{ds}_d256_le1.0_lz1.0_lr0.003.pt",
                                 map_location=DEV, weights_only=True)); m.eval()
    with torch.no_grad(): V_set = zr(m.score_all(R, degf)).to(dt)
    V_gume = zr(torch.from_numpy(np.load(PRED / f"gume_{ds}_scores.npy")).to(dt).to(DEV))
    views = {"gume": V_gume, "item": V_item, "set": V_set}

    gU, g2, gate_src = select_gates(ds, views, gevV)
    Su = gU[0] * V_gume + gU[1] * V_item + gU[2] * V_set          # SCOPE-U (static gate)
    S2 = g2[0] * V_gume + g2[1] * V_item                          # set-free base+GUME

    tk = {n: topk_ids(gevT, S) for n, S in
          [("GUME", V_gume), ("base", V_item), ("set", V_set), ("SCOPE-U", Su), ("setfree", S2)]}
    hm = {n: hit_mask(gevT, t) for n, t in tk.items()}
    npos = (gevT.pos >= 0).sum(1)
    tot = int(npos.sum().item())

    # (1) per-pair coverage + best-view-per-pair oracle (union of the 3 views' top-20)
    cov = {n: hm[n].sum().item() / tot for n in METHODS}
    miss = {n: 1.0 - cov[n] for n in METHODS}
    oracle_cov = (hm["GUME"] | hm["base"] | hm["set"]).sum().item() / tot
    captured = cov["SCOPE-U"] / oracle_cov if oracle_cov else 0.0

    # (2) per-user breadth: SCOPE-U vs the set-free ensemble
    nu = hm["SCOPE-U"].sum(1); n2 = hm["setfree"].sum(1)
    helps = float((nu > n2).float().mean()); hurts = float((nu < n2).float().mean())
    ties = float((nu == n2).float().mean())
    unchanged = float((tk["SCOPE-U"].sort(1).values == tk["setfree"].sort(1).values)
                      .all(1).float().mean())

    out = {"dataset": ds, "n_test_users": int(npos.numel()), "n_test_pairs": tot,
           "gate_scopeu": gU, "gate_setfree": g2, "gate_source": gate_src,
           "pair_coverage": cov, "pair_miss_rate": miss,
           "oracle_union_coverage": oracle_cov, "oracle_miss_rate": 1.0 - oracle_cov,
           "captured_share": captured, "setfree_coverage": hm["setfree"].sum().item() / tot,
           "breadth_helps": helps, "breadth_ties": ties, "breadth_hurts": hurts,
           "helps_or_ties": helps + ties, "top20_unchanged_frac": unchanged}
    (OUT / f"coverage_breadth_{ds}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[{ds}] miss GUME={miss['GUME']:.4f} base={miss['base']:.4f} set={miss['set']:.4f} "
          f"SCOPE-U={miss['SCOPE-U']:.4f} | oracle_cov={oracle_cov*100:.1f}% "
          f"captured={captured*100:.1f}% | helps={helps*100:.2f}% hurts={hurts*100:.2f}% "
          f"no_worse={100*(helps+ties):.1f}% unchanged={unchanged*100:.1f}% (gates: {gate_src})", flush=True)
    del R, V_item, V_set, V_gume, Su, S2, views, tk, hm; torch.cuda.empty_cache()
    return out


def _load_results():
    rs = {}
    for ds in DS_ORDER:
        f = OUT / f"coverage_breadth_{ds}.json"
        if f.exists(): rs[ds] = json.load(open(f))
    return rs


def fig_coverage(rs):
    """Figure S3: grouped per-pair top-20 miss-rate bars + best-view-per-pair oracle marks."""
    ds_list = list(rs)
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    x = np.arange(len(ds_list)); w = 0.2
    for i, mth in enumerate(METHODS):
        ax.bar(x + (i - 1.5) * w, [rs[ds]["pair_miss_rate"][mth] for ds in ds_list], w,
               label=METHOD_LABEL[mth], color=COLORS[mth], alpha=0.9)
    om = [rs[ds]["oracle_miss_rate"] for ds in ds_list]
    for j in range(len(ds_list)):
        ax.hlines(om[j], x[j] - 2 * w, x[j] + 2 * w, color="black", lw=0.9, linestyle="--")
    vmax = max(rs[ds]["pair_miss_rate"][m] for ds in ds_list for m in METHODS)
    ax.set_xticks(x); ax.set_xticklabels([DS_LABEL.get(d, d) for d in ds_list], fontsize=7)
    ax.set_ylabel("per-pair miss rate (lower=better)", fontsize=7)
    ax.set_ylim(min(om) - 0.006, vmax + 0.006)
    ax.tick_params(labelsize=6.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", lw=0.9, linestyle="--"))
    labels.append("oracle")
    ax.legend(handles, labels, fontsize=5.6, frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), columnspacing=0.8, handletextpad=0.35)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_breadth(rs):
    """Figure S4: % of users helped vs hurt by adding the set view, + no-worse-off share."""
    ds_list = list(rs)
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    win = [rs[ds]["breadth_helps"] * 100 for ds in ds_list]
    harm = [rs[ds]["breadth_hurts"] * 100 for ds in ds_list]
    nw = [rs[ds]["helps_or_ties"] * 100 for ds in ds_list]
    x = np.arange(len(ds_list)); w = 0.34
    ax.bar(x - w / 2, win, w, label="helped (win)", color="#43A047")
    ax.bar(x + w / 2, harm, w, label="hurt (harm)", color="#E53935")
    ymax = max(win + harm) * 1.45 + 0.3
    for i in range(len(ds_list)):
        ax.text(x[i] - w / 2, win[i] + 0.02 * ymax, f"{win[i]:.1f}", ha="center", va="bottom", fontsize=6)
        ax.text(x[i] + w / 2, harm[i] + 0.02 * ymax, f"{harm[i]:.1f}", ha="center", va="bottom", fontsize=6)
        ax.text(x[i], ymax * 0.86, f"{nw[i]:.1f}% no worse off", ha="center", va="top", fontsize=5.8, color="#37474F")
    ax.set_xticks(x); ax.set_xticklabels([DS_LABEL.get(d, d) for d in ds_list], fontsize=7)
    ax.set_ylabel("% of users", fontsize=7)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.4, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_breadth.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DS_ORDER)
    ap.add_argument("--figures_only", action="store_true",
                    help="skip computation; rebuild the figures from saved JSONs")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    if not a.figures_only:
        for ds in a.datasets:
            try: run(ds)
            except Exception as e:
                import traceback; print(f"[{ds}] ERR {type(e).__name__}: {e}"); traceback.print_exc()
    rs = _load_results()
    if rs:
        fig_coverage(rs); fig_breadth(rs)
        print(f"figures -> {OUT/'fig_coverage.pdf'} , {OUT/'fig_breadth.pdf'} "
              f"(datasets: {', '.join(rs)})", flush=True)
    else:
        print("no coverage_breadth_*.json results found; run the analysis first", flush=True)

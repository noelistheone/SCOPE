"""efficiency/accuracy Pareto figure. SCOPE-v1/G reach top-tier Recall@20 with ZERO per-user parameters,
while every neural ID-embedding baseline needs an n_users x 64 user table (1.24M on Baby). Writes
results/scope/pareto.pdf."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PU = 19445 * 64 / 1e6   # ID-embedding per-user table on Baby (millions)
# (name, per_user_M, R@20, kind, label_dx, label_dy, ha, va)
PTS = [
    ("LightGCN", PU, 0.0737, "trans", -0.05, 0.0, "right", "center"),
    ("FREEDOM",  PU, 0.0928, "trans", 0.05, -0.0002, "left", "center"),
    ("LGMRec",   PU, 0.0965, "trans", 0.05, 0.0002, "left", "center"),
    ("GUME",     PU, 0.1024, "trans", 0.05, 0.0, "left", "center"),
    ("EASE",              0.0, 0.0825, "ind", 0.05, 0.0, "left", "center"),
    ("EASE+text (base)",  0.0, 0.0925, "ind", 0.05, 0.0, "left", "center"),
    ("SCOPE-v1", 0.0, 0.1017, "ours", 0.05, 0.0015, "left", "center"),
    ("SCOPE-G",  0.0, 0.1010, "ours", 0.05, -0.0015, "left", "center"),
]
COL = {"trans": "#9aa0a6", "ind": "#4c72b0", "ours": "#c44e52"}
MK = {"trans": "o", "ind": "s", "ours": "*"}
SZ = {"trans": 46, "ind": 54, "ours": 300}

fig, ax = plt.subplots(figsize=(5.2, 3.2))
# Pareto frontier (upper-left staircase): EASE -> SCOPE-v1 (at 0) -> GUME (at PU)
ax.plot([0, 0, PU], [0.0825, 0.1017, 0.1024], ls="--", c="#c44e52", lw=1.1, alpha=0.45, zorder=1)
for kind in ("trans", "ind", "ours"):
    xs = [p[1] for p in PTS if p[3] == kind]; ys = [p[2] for p in PTS if p[3] == kind]
    lab = {"trans": "Transductive baselines", "ind": "Inductive baselines (closed-form)",
           "ours": "SCOPE-v1/G (ours, inductive)"}[kind]
    ax.scatter(xs, ys, c=COL[kind], marker=MK[kind], s=SZ[kind], label=lab, zorder=3,
               edgecolor=("white" if kind == "ours" else "none"), linewidth=0.6)
for name, x, y, kind, dx, dy, ha, va in PTS:
    ax.annotate(name, (x, y), (x + dx, y + dy), fontsize=8, ha=ha, va=va,
                fontweight=("bold" if kind == "ours" else "normal"),
                color=(COL["ours"] if kind == "ours" else "#333333"))
# vertical guide lines for the two regimes
ax.axvline(0.0, color="#4c72b0", lw=0.6, ls=":", alpha=0.35)
ax.axvline(PU, color="#9aa0a6", lw=0.6, ls=":", alpha=0.35)
ax.text(0.0, 0.0710, "inductive\n(0 params/user)", ha="center", va="bottom", fontsize=7.5, color="#4c72b0")
ax.text(0.72, 0.0805, "1.24M params/user\n(transductive)", ha="center", va="center", fontsize=7.5, color="#888888")
ax.set_xlabel("Per-user parameters (millions) — lower is better", fontsize=9)
ax.set_ylabel("Recall@20 (Amazon-Baby)", fontsize=9)
ax.set_xlim(-0.16, 1.75); ax.set_ylim(0.069, 0.106)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.4, loc="center right", framealpha=0.92)
ax.grid(True, ls=":", alpha=0.35)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "results" / "scope" / "pareto.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight"); print("wrote", out)

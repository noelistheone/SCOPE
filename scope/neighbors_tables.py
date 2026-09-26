#!/usr/bin/env python
"""Generate the LaTeX tables for G13 and G7 from the summary.json written by neighbors_matched.py --stage summarize
(results/scope/rev/g7g13_neighbors/eval_<base_tag>/summary_<UTC>/summary.json).

EVERY run and every test in the JSON is printed; nothing is filtered or re-ordered by outcome:
  * every per-seed run (SCOPE references, G7 runs, G13 runs of the standalone- and the fused-selected configurations,
    the two shared-trainer SCOPE sanity re-runs, and every other run / evaluation record found on disk),
  * every grid configuration with its validation scores,
  * every paired-bootstrap test of every family (seed-averaged and per seed), with its Holm-adjusted p and verdict.
Means / standard deviations over seeds use Decimal arithmetic rounded half-up, so the printed cells do not depend on the
Python version (float sum() can flip .5 cells).

Verdicts of the summary are mapped as a>b -> ref_better (SCOPE / the a side significantly better), a<b -> other_better,
n.s. -> tie; 'incomplete' (a seed missing) and 'missing' tests have no verdict and are printed as such.
Arm names are mapped as bert_cls -> BERT4Rec-style set Transformer, sas_bos -> SASRec-style set Transformer,
cbow_random / cbow_content -> item2vec/CBOW (random / content seed), multvae -> Mult-VAE.

Outputs (never overwritten; a timestamp suffix is added if a file exists), in --out_dir:
  tab_neighbors_<tag>.tex                main table: (A) standalone, (B) fused at the standalone-selected configs,
                                         (C) fused at the fused-selected configs; mean +- sd over seeds, marks
  tab_neighbors_grid_<tag>.tex           every grid configuration (one table per dataset)
  tab_neighbors_runs_<tag>.tex           every run, one row per seed
  tab_neighbors_tests_<tag>.tex          every seed-averaged test (G13 primary, fused-selected panel, encoder-only,
                                         sanity, G7)
  tab_neighbors_tests_perseed_<tag>.tex  every per-seed test (G13, G7)
  tab_cbow_gate_<tag>.tex                G7 arms + the pre-committed gate numbers
Only the standard library is used (no torch/numpy needed).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from decimal import ROUND_HALF_UP, Decimal, getcontext
from pathlib import Path

getcontext().prec = 34
ROOT = Path(os.environ.get("SCOPE_ROOT") or Path(__file__).resolve().parents[1]).resolve()
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

ARM_LABEL = {
    "cbow_random": "item2vec/CBOW (random seed)",
    "cbow_content": "item2vec/CBOW (content seed)",
    "bert_cls": "BERT4Rec-style set Transformer",
    "sas_bos": "SASRec-style set Transformer",
    "multvae": "Mult-VAE$^{\\dagger}$",
    "scope_mlp_cap60": "SCOPE head re-run, shared trainer, 60-item lists (= scope.train)",
    "scope_mlp_full": "SCOPE head re-run, shared trainer, full lists",
    "scope_mlp": "SCOPE head re-run, shared trainer",
}
SIDE_LABEL = {"ref": "SCOPE (deployed)", "ref7": "SCOPE (deployed)", "g13": "arm (standalone-sel.)",
              "g13f": "arm (fused-sel.)", "g7": "CBOW config", "cap60": "SCOPE re-run, 60-item lists",
              "full": "SCOPE re-run, full lists"}
FAMILY_LABEL = {"primary": "G13 primary", "fused_sel": "G13 fused-sel. panel", "encoder_only": "G13 encoder-only",
                "sanity": "sanity", "g7": "G7", "perseed": "G13 per-seed", "g7_perseed": "G7 per-seed"}
VERDICT = {"a>b": "ref_better", "a<b": "other_better", "n.s.": "tie", "incomplete": "incomplete", "missing": "missing"}
VERDICT_TXT = {"ref_better": "a better", "other_better": "b better", "tie": "n.s.", "incomplete": "incomplete",
               "missing": "missing"}
DSN = {"baby": "Baby", "sports": "Sports", "clothing": "Clothing", "microlens": "MicroLens", "elec": "Elec"}


# ------------------------------------------------------------------------------------------ exact rounding helpers
def D(x):
    return Decimal(repr(float(x)))


def dmean(vals):
    return sum((D(v) for v in vals), Decimal(0)) / Decimal(len(vals))


def dstd(vals):
    if len(vals) < 2:
        return None
    m = dmean(vals)
    return (sum(((D(v) - m) ** 2 for v in vals), Decimal(0)) / Decimal(len(vals) - 1)).sqrt()


def q(d, places=4):
    return d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def nolead(s):
    """'.0857' style used by the tables ('-.0012' for negatives)."""
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def fm(d, places=4):
    return "--" if d is None else nolead(str(q(d, places)))


def fx(x, places=4):
    return "--" if x is None else fm(D(x), places)


def fsigned(x, places=4):
    if x is None:
        return "--"
    s = nolead(str(q(D(x), places)))
    return s if s.startswith("-") else "+" + s


def fp(p, B):
    if p is None:
        return "--"
    if p == 0:
        return f"$<\\!{fm(Decimal(2) / Decimal(B), 4) if B >= 2 else '--'}$"
    if p < 0.001:
        mant, ex = f"{p:.1e}".split("e")
        return f"${mant}{{\\times}}10^{{{int(ex)}}}$"
    return fm(D(p), 3)


def fg(g):
    return "--" if g is None else f"{float(g):g}"


def tex(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def dsn(ds):
    return DSN.get(ds, tex(ds))


# ------------------------------------------------------------------------------------------ io
def newest_summary(smoke):
    base = ROOT / "results" / "scope" / "rev" / ("g7g13_neighbors_SMOKE" if smoke else "g7g13_neighbors")
    c = [p for p in base.glob("eval_*/summary_*/summary.json")]
    if not c:
        raise SystemExit(f"no summary.json under {base}/eval_*/summary_*/; pass --summary explicitly")
    return max(c, key=lambda p: p.stat().st_mtime)


def fresh(path: Path) -> Path:
    return path if not path.exists() else path.with_name(f"{path.stem}.{STAMP}{path.suffix}")


def write(path: Path, txt: str):
    path = fresh(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)
    print(f"wrote {path}")
    return path


def header(J, what):
    return (f"% AUTO-GENERATED by scope/neighbors_tables.py from {J['_source']} ({STAMP}); summary "
            f"version {J.get('version')}, base_tag {J.get('base_tag')}{' (SMOKE)' if J.get('smoke') else ''}. "
            f"{what} Do not hand-edit.")


def split_floats(rows, head, caption, label, per=40, env="table*"):
    """Rows split into floats of <= per rows (a float cannot break across pages); first carries caption + label."""
    parts = [rows[i:i + per] for i in range(0, len(rows), per)] or [[]]
    L = []
    for k, part in enumerate(parts):
        L += [f"\\begin{{{env}}}[t]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}"]
        if k == 0:
            L.append(f"\\caption{{{caption}" + (f" ({len(parts)} parts.)" if len(parts) > 1 else "") + "}")
            L.append(f"\\label{{{label}}}")
        else:
            L.append(f"\\caption{{(continued, part {k + 1} of {len(parts)})}}")
        L += head + part + ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return L


# ------------------------------------------------------------------------------------------ cells
def run_vals(runs, setting, metric):
    """Test values of a {seed: run_summary} dict, in seed order (runs without test numbers are skipped)."""
    out = []
    for _, r in sorted((runs or {}).items(), key=lambda kv: int(kv[0])):
        te = (r or {}).get("test")
        if te:
            out.append(te[setting][metric])
    return out


def run_statuses(runs):
    sts = []
    for r in (runs or {}).values():
        r = r or {}
        st = r.get("status") or r.get("train_status")
        if st == "trained":
            st = r.get("test_eval_status") if r.get("tested", True) else r.get("eval_status")
            st = f"test {st}" if st != "complete" else "complete"
        sts.append(str(st))
    return sorted(set(sts))


def cell(runs, setting, metric, n_expected, mark=""):
    vals = run_vals(runs, setting, metric)
    if not vals:                                     # show WHY there is no number (not run / failed / ...)
        sts = run_statuses(runs)
        st = "/".join(sts) if sts else "not run"
        return f"\\multicolumn{{1}}{{c}}{{({tex(st).replace('_', ' ')})}}"
    m, s = dmean(vals), dstd(vals)
    sd = f"\\,{{\\tiny$\\pm${fm(s)}}}" if s is not None else ""
    nn = "" if len(vals) >= n_expected else f"\\,{{\\tiny($n{{=}}{len(vals)}$)}}"
    return f"{fm(m)}{sd}{mark}{nn}"


def find_test(family, ds, arm, setting, metric):
    for t in (family or {}).get("tests", []):
        if t["dataset"] == ds and t["arm"] == arm and t["setting"] == setting and t["metric"] == metric:
            return t
    return None


def mark_for(t, kind="competitor"):
    if t is None:
        return ""
    v = VERDICT.get(t.get("verdict"), t.get("verdict"))
    prov = "$^{\\S}$" if t.get("verdict_provisional") else ""
    if v == "incomplete":
        return "$^{?}$"
    if kind == "sanity":
        m = "$^{\\neq}$" if v in ("ref_better", "other_better") else ""
    else:
        m = {"ref_better": "$^{\\ast}$", "other_better": "$^{\\circ}$"}.get(v, "")
    return m + (prov if m else "")


def arm_label(arm):
    return ARM_LABEL.get(arm, tex(arm))


def g7_label(key):
    arm, _, cfg = str(key).partition("|")
    le = "1" if "le1" in cfg.split("_") else ("0" if "le0" in cfg.split("_") else "?")
    return f"{ARM_LABEL.get(arm, tex(arm))}, $\\lambda_E{{=}}{le}$"


def wtl_txt(w):
    if not w:
        return "--"
    return (f"{w['win']}/{w['tie']}/{w['loss']} (incomplete {w['incomplete']}, missing {w['missing']}; "
            f"$m{{=}}{w['m_expected']}$" + ("" if w["family_complete"] else ", family INCOMPLETE") + ")")


# ------------------------------------------------------------------------------------------ G13 main table
def main_table(J):
    dss = [d for d in J["datasets"] if (J.get("dataset_info") or {}).get(d)]
    missing_ds = [d for d in J["datasets"] if d not in dss]
    g13, B = J["g13"], J["bootstrap"]["B"]
    nseed = len(J["seeds"])
    ncol = 1 + 2 * len(dss)
    fams = {"primary": g13.get("family"), "fused_sel": g13.get("family_fused_sel"),
            "sanity": (g13.get("sanity") or {}).get("family")}
    caps, edges, diff_cfg = [], [], []
    for ds in dss:
        for part in ("runs", "runs_fused_sel"):
            for arm, runs in (g13.get(part, {}).get(ds) or {}).items():
                for sd, r in (runs or {}).items():
                    if (r or {}).get("hit_cap"):
                        caps.append(f"{dsn(ds)} {arm_label(arm)} seed {sd}")
                    if (r or {}).get("gamma_edge_hit"):
                        edges.append(f"{dsn(ds)} {arm_label(arm)} seed {sd} ($\\gamma{{=}}{fg(r.get('gamma_retuned'))}$)")
        for arm, sel in (g13.get("selected", {}).get(ds) or {}).items():
            if sel and sel.get("same_config") is False:
                diff_cfg.append(f"{dsn(ds)}/{arm_label(arm)} ({tex(sel['alone']['config_key'].split('|')[1])} vs "
                                f"{tex(sel['fused']['config_key'].split('|')[1])})")
    caps, edges = sorted(set(caps)), sorted(set(edges))
    bud, gam = J["budgets"], J["gamma"]
    ngrid = {a: len(g) for a, g in (J.get("g13_grids") or {}).items()}
    w, wf = g13.get("win_tie_loss"), g13.get("win_tie_loss_fused_sel")
    nar = g13.get("narrowing") or {}
    over60 = "; ".join(f"{dsn(d)} {v.get('users_over_60_items')}" for d, v in
                       ((g13.get("sanity") or {}).get("users_over_60_items") or {}).items() if v)
    caption = (
        "Matched-budget comparison of the SCOPE set-completion head with its nearest masked and autoencoding "
        f"neighbours: test Recall@20 / NDCG@20, mean\\,$\\pm$\\,sd over seeds {', '.join(str(s) for s in J['seeds'])}. "
        "(A) standalone; (B) fused with the closed-form base of Table~3 at each neighbour's configuration selected on "
        "standalone validation Recall@20; (C) the same at the configuration selected on fused validation Recall@20 "
        "(additional panel). $\\gamma$ is tuned on validation per model and seed over "
        f"$\\{{{', '.join(fg(g) for g in gam['grid'])}\\}}$, extended by "
        f"$\\{{{', '.join(fg(g) for g in gam['extension'])}\\}}$ when 5 is selected; the deployed SCOPE rows use the "
        "deployed per-seed $\\gamma$. All masked-set models share one trainer and objective (masked-set softmax, "
        "L2-cosine head with learnable temperature, SIGReg$(E)$ weight in the grid, content-seeded item table except "
        f"CBOW-random, full training histories as context, at most {bud['max_epochs']} epochs, validation every "
        f"{bud['eval_every']} epochs, patience {bud['patience_checks']} checks, batch {bud['bs']}). The set "
        "Transformers have no positional embeddings and are not the published BERT4Rec/SASRec. "
        "$^{\\dagger}$Mult-VAE is trained with its own multinomial ELBO (KL annealed over "
        f"{bud['vae_anneal_steps']} update steps, batch {bud.get('vae_bs', 500)}, epoch cap {bud['vae_max_epochs']}), "
        "not with the masked-set objective. Neighbour hyper-parameters were selected on validation only (seed "
        f"{J['grid_seed']}; grids of {ngrid.get('cbow_random', '--')}/{ngrid.get('cbow_content', '--')}/"
        f"{ngrid.get('bert_cls', '--')}/{ngrid.get('sas_bos', '--')}/{ngrid.get('multvae', '--')} configurations for "
        "CBOW-random/CBOW-content/BERT4Rec-style/SASRec-style/Mult-VAE; every validation score in the appendix). "
        "The deployed SCOPE rows are the checkpoints of Table~3 (trained by scope.py with 60-item training lists); "
        "the two shared-trainer SCOPE rows re-train that head with the 60-item lists of scope.train (reproduction "
        "check) and with full lists (the setting of the neighbours)"
        + (f"; users with more than 60 training items: {over60}" if over60 else "") + ". "
        "$^{\\ast}$/$^{\\circ}$: SCOPE / the neighbour significantly better (paired user-level bootstrap of "
        f"seed-averaged per-user metrics, $B{{=}}{B}$, two-sided; the training-seed spread is not part of this test, "
        "see the per-seed tests), Holm over the full pre-stated family "
        f"($m{{=}}{(w or {}).get('m_expected', '--')}$ for (A)+(B), $m{{=}}{(wf or {}).get('m_expected', '--')}$ for "
        "(C); missing or incomplete tests enter with $p{=}1$); unmarked: not significant; $^{?}$: a seed is missing "
        "(no verdict); $^{\\S}$: verdict from an incomplete family (provisional). $^{\\neq}$: shared-trainer SCOPE "
        "significantly different from the deployed head (own Holm family). "
        f"Won/tied/lost by SCOPE: (A)+(B) {wtl_txt(w)}; (C) {wtl_txt(wf)}. Outcome: {tex(nar.get('outcome', '--'))}."
    )
    if diff_cfg:
        caption += " Configurations that differ between (B) and (C): " + "; ".join(diff_cfg) + "."
    else:
        caption += " (C) uses the same configurations as (B) wherever both selections exist."
    if caps:
        caption += " Runs that reached their epoch cap: " + "; ".join(caps) + "."
    if edges:
        caption += " $\\gamma$ at the edge of its grid: " + "; ".join(edges) + "."
    if missing_ds:
        caption += " Not run: " + ", ".join(dsn(d) for d in missing_ds) + "."
    L = [header(J, "Every model of the summary is printed."),
         "\\begin{table*}[t]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{{caption}}}", "\\label{tab:neighbors}",
         "\\begin{tabular}{l" + "cc" * len(dss) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{dsn(d)}}}" for d in dss) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(dss))),
         "Model & " + " & ".join("R@20 & N@20" for _ in dss) + " \\\\"]
    arms = J.get("g13_arms") or []
    panels = (("alone", "runs", "primary", "(A) Standalone"),
              ("fused", "runs", "primary", "(B) Fused with the closed-form base (standalone-selected configurations)"),
              ("fused", "runs_fused_sel", "fused_sel",
               "(C) Fused with the closed-form base (fused-selected configurations; additional panel)"))
    for pi, (st, part, famname, title) in enumerate(panels):
        L += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        if pi < 2:
            refs = [("SCOPE head (deployed, Table~3)" if st == "alone" else
                     "SCOPE-v1 (deployed head + $\\gamma\\cdot$base)", lambda d: J["scope_ref"].get(d), None)]
            refs += [(arm_label("scope_mlp_cap60"), lambda d: (g13["sanity"]["runs_cap60"] or {}).get(d),
                      "scope_mlp_cap60"),
                     (arm_label("scope_mlp_full"), lambda d: (g13["sanity"]["runs_full"] or {}).get(d),
                      "scope_mlp_full")]
            for lab, get, sarm in refs:
                cells = []
                for ds in dss:
                    for metric in ("R20", "N20"):
                        t = find_test(fams["sanity"], ds, sarm, st, metric) if sarm else None
                        cells.append(cell(get(ds), st, metric, nseed, mark_for(t, "sanity")))
                L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        else:
            cells = []
            for ds in dss:
                for metric in ("R20", "N20"):
                    cells.append(cell(J["scope_ref"].get(ds), st, metric, nseed))
            L.append("SCOPE-v1 (deployed head + $\\gamma\\cdot$base) & " + " & ".join(cells) + " \\\\")
        for arm in arms:
            cells = []
            for ds in dss:
                for metric in ("R20", "N20"):
                    t = find_test(fams[famname], ds, arm, st, metric)
                    cells.append(cell((g13.get(part, {}).get(ds) or {}).get(arm), st, metric, nseed, mark_for(t)))
            L.append(f"{arm_label(arm)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ grid tables
def grid_tables(J):
    out = [header(J, "Every grid configuration is listed; A/F = selected for the standalone / fused rows "
                     "(validation only).")]
    g13 = J["g13"]
    for ds in J["datasets"]:
        grid = (g13.get("grid") or {}).get(ds)
        if not grid:
            out.append(f"% {ds}: no grid rows in the summary (not run)")
            continue
        rows = []
        for arm in (J.get("g13_arms") or []) + sorted(a for a in grid if a not in (J.get("g13_arms") or [])):
            for r in grid.get(arm, []):
                sel = ",".join(x for x, f in (("A", r.get("selected_alone")), ("F", r.get("selected_fused"))) if f)
                v = r.get("val") or {}
                st = r.get("train_status")
                if st == "trained" and r.get("eval_status") != "complete":
                    st = f"eval {r.get('eval_status')}"
                cap = "--" if r.get("hit_cap") is None else ("yes" if r["hit_cap"] else "no")
                fv = "--" if not v else f"{fx(v['fused']['R20'])} ({fg(r.get('gamma'))}{'$^{e}$' if r.get('gamma_edge_hit') else ''})"
                rows.append(f"{arm_label(arm)} & \\texttt{{{tex(str(r.get('config_key', '')).split('|')[-1])}}} & "
                            f"{tex(st)} & {fx(r.get('best_val_R20'))} & {fx((v.get('alone') or {}).get('N20'))} & "
                            f"{fv} & {fx((v.get('fused') or {}).get('N20'))} & "
                            f"{r.get('best_ep') if r.get('best_ep') is not None else '--'} & "
                            f"{r.get('last_ep') if r.get('last_ep') is not None else '--'} & {cap} & {sel} \\\\")
        head = ["\\begin{tabular}{lllccccrrcc}", "\\toprule",
                "Neighbour & Configuration & status & val R@20 & val N@20 & fused val R@20 ($\\gamma$) & "
                "fused val N@20 & best ep & last ep & cap hit & selected \\\\", "\\midrule"]
        caption = (f"{dsn(ds)}: every neighbour configuration of the validation grid (seed {J['grid_seed']}): best "
                   "standalone validation Recall@20 (early-stopping criterion) and NDCG@20, fused validation "
                   "Recall@20/NDCG@20 at its own validation-tuned $\\gamma$ ($^{e}$: $\\gamma$ at the grid edge), best "
                   "and last epoch, whether the epoch cap was reached, and the configurations selected for the "
                   "standalone (A) and fused (F) rows of Table~\\ref{tab:neighbors}. " f"{len(rows)} configurations.")
        out += split_floats(rows, head, caption, f"tab:neighbors_grid_{ds}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------ per-seed run table
def run_row(ds, model, r, seed):
    r = r or {}
    te = r.get("test") or {}
    st = r.get("status") or r.get("train_status") or "not run"
    if st == "trained":
        st = r.get("test_eval_status") if r.get("tested", True) else r.get("eval_status")
    ver = r.get("version") or "--"
    g = fg(r.get("gamma")) + ("$^{e}$" if r.get("gamma_edge_hit") else "") + \
        ("$^{fb}$" if r.get("gamma_fallback") else "")
    cfg = str(r.get("config_key") or "").split("|")[-1] or "--"
    cap = "--" if r.get("hit_cap") is None else ("yes" if r["hit_cap"] else "no")
    return (f"{dsn(ds)} & {model} & \\texttt{{{tex(cfg)}}} & {seed} & {tex(st)} & {tex(ver)} & "
            f"{r.get('best_ep') if r.get('best_ep') is not None else '--'} & "
            f"{r.get('last_ep') if r.get('last_ep') is not None else '--'} & {cap} & {g} & "
            f"{fx((te.get('alone') or {}).get('R20'))} & {fx((te.get('alone') or {}).get('N20'))} & "
            f"{fx((te.get('fused') or {}).get('R20'))} & {fx((te.get('fused') or {}).get('N20'))} \\\\")


def runs_table(J):
    g13, rows = J["g13"], []
    for ds in J["datasets"]:
        for sd, r in sorted((J.get("scope_ref", {}).get(ds) or {}).items()):
            rows.append(run_row(ds, "SCOPE (deployed), G13 rule", r, sd))
        for sd, r in sorted((J.get("scope_ref_g7", {}).get(ds) or {}).items()):
            same = (r or {}).get("record") == ((J.get("scope_ref", {}).get(ds) or {}).get(sd) or {}).get("record")
            rows.append(run_row(ds, "SCOPE (deployed), G7 rule" + (" (same record)" if same else ""), r, sd))
        for key, runs in (J["g7"].get("runs", {}).get(ds) or {}).items():
            for sd, r in sorted(runs.items()):
                rows.append(run_row(ds, "G7 " + g7_label(key), r, sd))
        for part, lab in (("runs", "G13 standalone-sel."), ("runs_fused_sel", "G13 fused-sel.")):
            for arm, runs in (g13.get(part, {}).get(ds) or {}).items():
                same = ((g13.get("selected", {}).get(ds) or {}).get(arm) or {}).get("same_config")
                extra = " (= standalone-sel.)" if (part == "runs_fused_sel" and same) else ""
                if not runs:
                    rows.append(run_row(ds, f"{lab} {arm_label(arm)}", {"status": "not run"}, "--"))
                for sd, r in sorted(runs.items()):
                    rows.append(run_row(ds, f"{lab} {arm_label(arm)}{extra}", r, sd))
        for part, arm in (("runs_cap60", "scope_mlp_cap60"), ("runs_full", "scope_mlp_full")):
            for sd, r in sorted(((g13.get("sanity") or {}).get(part, {}).get(ds) or {}).items()):
                rows.append(run_row(ds, arm_label(arm), r, sd))
        for rid, r in sorted((J.get("other_runs", {}).get(ds) or {}).items()):
            rows.append(run_row(ds, f"other run ({tex(r.get('stage') or '?')}) {arm_label(r.get('arm') or '?')}",
                                r, r.get("seed", "--")))
        for name, e in sorted((J.get("other_eval_records", {}).get(ds) or {}).items()):
            rows.append(f"{dsn(ds)} & other eval record \\texttt{{{tex(name)}}} & -- & -- & {tex(e.get('status'))} & "
                        f"{tex(e.get('version') or '--')} & -- & -- & -- & {fg(e.get('gamma'))} & "
                        f"{fx(e.get('test_alone_R20'))} & -- & {fx(e.get('test_fused_R20'))} & -- \\\\")
    head = ["\\begin{tabular}{lllllcrrcccccc}", "\\toprule",
            "Dataset & Model & Config & Seed & status & version & best ep & last ep & cap hit & $\\gamma$ & "
            "R@20 & N@20 & R@20 fused & N@20 fused \\\\", "\\midrule"]
    caption = ("Every run in the summary, one row per seed (test metrics; the grid-only runs are in the grid tables): "
               "deployed SCOPE references, G7 CBOW runs (pre-registered $\\gamma$ grid, no extension), G13 runs of "
               "the standalone- and fused-selected configurations, the shared-trainer SCOPE re-runs, and every other "
               "run or evaluation record found on disk. status = test-evaluation status; $^{e}$: $\\gamma$ at the "
               f"grid edge; $^{{fb}}$: $\\gamma$ fallback (re-tuned, no deployed value). {len(rows)} rows.")
    return "\n".join([header(J, "Every run, per seed.")] +
                     split_floats(rows, head, caption, "tab:neighbors_runs")) + "\n"


# ------------------------------------------------------------------------------------------ test tables
def test_rows(tests, B, m):
    rows = []
    for t in tests:
        v = VERDICT.get(t.get("verdict"), t.get("verdict"))
        vt = VERDICT_TXT.get(v, str(v)) + (" (prov.)" if t.get("verdict_provisional") else "")
        arm = t.get("arm")
        if t.get("family") == "sanity":                   # a = shared-trainer re-run, b = deployed head
            a_txt, b_txt = arm_label(arm), SIDE_LABEL.get(t.get("b"), tex(t.get("b")))
        else:
            other = g7_label(arm) if t.get("family") in ("g7", "g7_perseed") else arm_label(arm)
            a_txt = SIDE_LABEL.get(t.get("a"), tex(t.get("a")))
            b_txt = f"{other} [{SIDE_LABEL.get(t.get('b'), tex(t.get('b')))}]"
        seeds = ",".join(str(s) for s in t.get("seeds", []))
        ci = t.get("ci95")
        rows.append(f"{FAMILY_LABEL.get(t.get('family'), tex(t.get('family')))} & {dsn(t['dataset'])} & "
                    f"{a_txt} & {b_txt} & {t['setting']} & "
                    f"{'R@20' if t['metric'] == 'R20' else 'N@20'} & {seeds} & "
                    f"{t.get('n_seeds_a')}/{t.get('n_seeds_b')}/{t.get('n_seeds_expected')} & {fx(t.get('mean_a'))} & "
                    f"{fx(t.get('mean_b'))} & {fsigned(t.get('mean_delta'))} & "
                    f"{'--' if not ci else '[' + fsigned(ci[0]) + ', ' + fsigned(ci[1]) + ']'} & "
                    f"{fp(t.get('p_two_sided'), B)} & {fp(t.get('p_holm'), B)} & {m} & {vt} \\\\")
    return rows


def tests_table(J, per_seed):
    fams = []
    g13 = J["g13"]
    if per_seed:
        fams = [("G13 per-seed", g13.get("family_perseed")), ("G7 per-seed", J["g7"].get("family_perseed"))]
    else:
        fams = [("G13 primary", g13.get("family")), ("G13 fused-selected panel", g13.get("family_fused_sel")),
                ("G13 encoder-only", g13.get("family_encoder_only")),
                ("sanity", (g13.get("sanity") or {}).get("family")), ("G7", J["g7"].get("family"))]
    B = J["bootstrap"]["B"]
    rows, desc = [], []
    for name, f in fams:
        if not f:
            desc.append(f"{name}: not computed")
            continue
        rows += test_rows(f["tests"], B, f["m_expected"])
        desc.append(f"{name}: {len(f['tests'])} tests, Holm $m{{=}}{f['m_expected']}$ ({f['n_ok']} with a verdict, "
                    f"{f['n_incomplete']} incomplete, {f['n_missing']} missing"
                    + ("" if f["complete"] else "; INCOMPLETE family, verdicts provisional") + ")")
    kind = "per-seed" if per_seed else "seed-averaged"
    head = ["\\begin{tabular}{lllllllccccccccl}", "\\toprule",
            "Family & Dataset & a & b [side] & Setting & Metric & Seeds & $n_a$/$n_b$/$n$ & mean a & mean b & "
            "$\\Delta$ & 95\\% CI & $p$ & $p_{\\mathrm{Holm}}$ & $m$ & Verdict \\\\", "\\midrule"]
    caption = (f"Every {kind} paired user-level bootstrap test ($B{{=}}{B}$, two-sided) in the summary, in JSON order, "
               "none omitted. $\\Delta$ = a $-$ b; for the neighbour families a = deployed SCOPE (head alone, or "
               "SCOPE-v1 when fused) or the shared-trainer SCOPE head (encoder-only family), for the sanity family a = "
               "the shared-trainer re-run and b = the deployed head. "
               + ("Seed-averaged tests average each user's metric over the seeds first and then resample users, so "
                  "the training-seed spread is not in their CI or $p$. " if not per_seed else "")
               + "Holm within each family over its full pre-stated size $m$ (missing / incomplete tests enter with "
               "$p{=}1$ and get no verdict). " + "; ".join(desc) + ".")
    label = "tab:neighbors_tests_perseed" if per_seed else "tab:neighbors_tests"
    return "\n".join([header(J, f"All {kind} tests.")] + split_floats(rows, head, caption, label)) + "\n"


# ------------------------------------------------------------------------------------------ G7 table
def g7_table(J):
    G, B = J["g7"], J["bootstrap"]["B"]
    dss = [d for d in J["datasets"] if (J.get("dataset_info") or {}).get(d)]
    nseed = len(J["seeds"])
    fam = G.get("family")
    w = G.get("win_tie_loss")
    keys = [c["config_key"] for c in J.get("g7_configs", [])]
    caption = (
        "G7 (CBOW gate): revised item2vec/CBOW mean-pool heads (L2-cosine scoring; the content-seeded configurations "
        "use exactly SCOPE's seed, the others a random init) trained with SCOPE's budget (lr $3{\\times}10^{-3}$, "
        f"batch {J['budgets']['bs']}, at most {J['budgets']['max_epochs']} epochs, patience "
        f"{J['budgets']['patience_checks']} checks), with and without SIGReg$(E)$ ($\\lambda_E$), versus the deployed "
        "SCOPE head (A) and SCOPE-v1 (B); $\\gamma$ tuned on validation over the pre-registered grid "
        f"$\\{{{', '.join(fg(g) for g in J['gamma']['grid'])}\\}}$ without extension. Test mean\\,$\\pm$\\,sd over "
        f"seeds {', '.join(str(s) for s in J['seeds'])}. $^{{\\ast}}$/$^{{\\circ}}$: SCOPE / CBOW significantly "
        f"better (paired bootstrap $B{{=}}{B}$ on seed-averaged per-user metrics, Holm over the full family, "
        "missing / incomplete tests with $p{=}1$); $^{?}$: a seed missing; $^{\\S}$: provisional (incomplete family). "
        f"Won/tied/lost by SCOPE: {wtl_txt(w)}.")
    L = [header(J, "Every G7 configuration."), "\\begin{table*}[t]", "\\centering\\scriptsize",
         "\\setlength{\\tabcolsep}{3pt}", f"\\caption{{{caption}}}", "\\label{tab:cbow_gate}",
         "\\begin{tabular}{l" + "cc" * len(dss) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{dsn(d)}}}" for d in dss) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(dss))),
         "Model & " + " & ".join("R@20 & N@20" for _ in dss) + " \\\\"]
    for st, title in (("alone", "(A) Standalone"), ("fused", "(B) Fused with the closed-form base")):
        L += ["\\midrule", f"\\multicolumn{{{1 + 2 * len(dss)}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        cells = [cell(J["scope_ref_g7"].get(ds), st, m, nseed) for ds in dss for m in ("R20", "N20")]
        L.append(("SCOPE head (deployed, Table~3)" if st == "alone" else "SCOPE-v1 (deployed head + $\\gamma\\cdot$base)")
                 + " & " + " & ".join(cells) + " \\\\")
        for key in keys:
            cells = [cell((G.get("runs", {}).get(ds) or {}).get(key), st, m, nseed,
                          mark_for(find_test(fam, ds, key, st, m))) for ds in dss for m in ("R20", "N20")]
            L.append(f"{g7_label(key)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\medskip",
          "\\begin{tabular}{llccccl}", "\\toprule",
          "Dataset & CBOW configuration & $\\Delta$R@20 fused (SCOPE-v1 $-$ config, seed means) & seeds & "
          "pre-committed MDE & min.\\ margin & gate \\\\", "\\midrule"]
    for ds in J["datasets"]:
        g = (G.get("gate") or {}).get(ds) or {}
        if not g:
            L.append(f"{dsn(ds)} & (not run) & -- & -- & -- & -- & -- \\\\")
            continue
        for key in keys:
            mk = key == g.get("min_margin_config")
            trig = g.get("triggered")
            gate = ("--" if not mk else ("triggered" if trig else ("not triggered" if trig is not None else "--"))
                    + (" (provisional)" if g.get("provisional") else ""))
            L.append(f"{dsn(ds)} & {g7_label(key)} & {fsigned(g.get('margins_fused_R20', {}).get(key))} & "
                     f"{g.get('n_seeds_per_config', {}).get(key, '--')} & {fx(g.get('mde_preregistered'))} & "
                     f"{'yes' if mk else ''} & {gate} \\\\")
        if g.get("gamma_edge_hits"):
            L.append(f"\\multicolumn{{7}}{{l}}{{{dsn(ds)}: $\\gamma{{=}}5$ (grid edge) selected for "
                     f"{tex(', '.join(g['gamma_edge_hits']))}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          f"% gate rule: {next(iter((G.get('gate') or {}).values()), {}).get('rule', '')}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Build tab_neighbors*.tex and tab_cbow_gate*.tex from the summary.json "
                                             "of neighbors_matched.py; every run and every test is printed.")
    ap.add_argument("--summary", default=None,
                    help="summary.json written by neighbors_matched.py --stage summarize (default: the newest "
                         "results/scope/rev/g7g13_neighbors[_SMOKE]/eval_*/summary_*/summary.json)")
    ap.add_argument("--smoke", action="store_true", help="default --summary from the *_SMOKE output directory")
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "scope" / "rev" / "tables"))
    a = ap.parse_args()
    sp = Path(a.summary) if a.summary else newest_summary(a.smoke)
    J = json.loads(sp.read_text())
    for k in ("g13", "g7", "scope_ref", "scope_ref_g7", "budgets", "gamma", "bootstrap"):
        if k not in J:
            raise SystemExit(f"{sp} has no '{k}': not a summary.json of neighbors_matched.py {J.get('version')} "
                             f"(r2 or later needed)")
    J["_source"] = f"{sp.parent.name}/{sp.name}"
    tag = str(J.get("base_tag")) + ("_smoke" if J.get("smoke") else "")
    out = Path(a.out_dir)
    print(f"summary: {sp}")
    write(out / f"tab_neighbors_{tag}.tex", main_table(J))
    write(out / f"tab_neighbors_grid_{tag}.tex", grid_tables(J))
    write(out / f"tab_neighbors_runs_{tag}.tex", runs_table(J))
    write(out / f"tab_neighbors_tests_{tag}.tex", tests_table(J, False))
    write(out / f"tab_neighbors_tests_perseed_{tag}.tex", tests_table(J, True))
    write(out / f"tab_cbow_gate_{tag}.tex", g7_table(J))
    print(f"G13 narrowing: {(J['g13'].get('narrowing') or {}).get('outcome')}")
    print(f"G13 W/T/L primary: {wtl_txt(J['g13'].get('win_tie_loss'))}; fused-selected panel: "
          f"{wtl_txt(J['g13'].get('win_tie_loss_fused_sel'))}")
    for ds, g in (J["g7"].get("gate") or {}).items():
        print(f"G7 gate {ds}: triggered={g.get('triggered')} min_margin={g.get('min_margin')} "
              f"({g.get('min_margin_config')}) complete={g.get('complete')}")
    print(f"{len(J.get('flags', []))} flags in the summary")


if __name__ == "__main__":
    main()

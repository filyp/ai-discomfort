# Frustration-probe analysis + plots for the two Gemma rollout sets.
#
# Reads rollouts/<tag>/<dataset>/<n>.json produced by the frustration-probe
# sweeps and, FOR EACH MODEL SEPARATELY, reports:
#   1. post-task frustration rating per dataset x wording (expected value),
#   2. argmax-vs-expected divergence (two ways of summarising the same first-digit
#      logprob distribution),
#   3. probe-vs-control tests along each manipulated dimension (harm, tedium, ...),
#   4. how the five wordings agree with one another.
#
# The two models are in DIFFERENT conditions and are NEVER compared with a test:
#   gemma-4-31b-it : OpenRouter, temperature 1, 3 task x 5 post samples/prompt
#   gemma-3-27b    : local ollama, temperature 0, a single deterministic pass
# So gemma4 samples are averaged to one value per prompt (they share a prompt and
# are not independent); gemma3 already has one per prompt. All across-prompt tests
# use these prompt-level values as the independent unit. Non-parametric tests are
# used throughout (ratings are bounded 1-9 ordinals, not normal): Wilcoxon
# signed-rank for the paired argmax/expected offset, Mann-Whitney U (Holm
# corrected) for probe-vs-control, Spearman for correlations.
#
# Outputs: printed report + reports/frustration/REPORT.md + PNG figures there.

import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Ratings often saturate (a whole dataset pinned to 1), so a correlation input can
# be constant; scipy then warns and returns nan, which we report as nan on purpose.
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders import PAIRS  # noqa: E402  (probe -> control dataset)

# --- config -----------------------------------------------------------------
MODELS = [
    {"key": "gemma4_31b", "tag": "google_gemma-4-31b-it",
     "cond": "OpenRouter, temp=1, 3x5 post-samples/prompt (averaged per prompt)"},
    {"key": "gemma3_27b", "tag": "ollama_gemma3_27b",
     "cond": "local ollama, temp=0, single deterministic pass/prompt"},
]
WORDINGS = ["frustration_q", "frustration_halfpersonal_q", "frustration_nonpersonal_q",
            "frustration_probe_log", "frustration_probe_log_inline"]
WSHORT = {"frustration_q": "personal", "frustration_halfpersonal_q": "half-pers",
          "frustration_nonpersonal_q": "non-pers", "frustration_probe_log": "probe-log",
          "frustration_probe_log_inline": "probe-inl"}
PRIMARY = "frustration_nonpersonal_q"   # most discriminating wording, used for pair tests

# Row order for tables/heatmaps: probes grouped, then controls, then model-only sets.
PROBE_ORDER = ["advbench", "strongreject", "harmbench", "toxicchat",
               "squad_noanswer", "ambigqa", "abstention", "tedious"]
CONTROL_ORDER = ["xstest_safe", "toxicchat_benign", "squad_answerable",
                 "ambigqa_unambiguous", "abstention_answerable", "engaging",
                 "wildchat_benign"]

OUTDIR = os.path.join(PROJECT_ROOT, "reports", "frustration")
os.makedirs(OUTDIR, exist_ok=True)
_report_lines = []


def emit(line=""):
    print(line)
    _report_lines.append(line)


# --- loading ----------------------------------------------------------------
def argmax_digit(lp):
    """Modal digit of the stored first-token distribution (== emitted digit)."""
    p = (lp or {}).get("probs")
    return int(max(p, key=lambda k: p[k])) if p else None


def load(tag):
    """Return prompt-level tables for one model.

    exp[w][ds]  = [expected-rating per prompt]   (samples averaged within a prompt)
    arg[w][ds]  = [argmax-rating   per prompt]
    Both keyed by wording then dataset; one value per prompt file.
    """
    root = os.path.join(PROJECT_ROOT, "rollouts", tag)
    exp = defaultdict(lambda: defaultdict(list))
    arg = defaultdict(lambda: defaultdict(list))
    for ds in sorted(os.listdir(root)):
        ddir = os.path.join(root, ds)
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(ddir, fn)))
            for w, ev in (rec.get("evals") or {}).items():
                if w not in WORDINGS:
                    continue
                exps, args = [], []
                for post in ev.get("post_task", []):     # per task completion
                    for e in post:                        # per post sample
                        lp = e.get("logprobs")
                        am = argmax_digit(lp)
                        if am is None or lp.get("expected") is None:
                            continue
                        exps.append(lp["expected"])
                        args.append(am)
                if exps:                                  # one prompt-level value each
                    exp[w][ds].append(float(np.mean(exps)))
                    arg[w][ds].append(float(np.mean(args)))
    return exp, arg


# --- helpers ----------------------------------------------------------------
def row_order(datasets):
    present = set(datasets)
    ordered = [d for d in PROBE_ORDER if d in present] + \
              [d for d in CONTROL_ORDER if d in present]
    extra = sorted(present - set(ordered))   # e.g. behavioral* (gemma4-only)
    return ordered + extra


def flatten(table_w):
    """All prompt-level values for a wording across datasets -> flat list."""
    return [v for ds in table_w.values() for v in ds]


# --- 1. per-dataset expected-rating table -----------------------------------
def table_expected(exp, datasets):
    emit("Expected-value frustration rating per dataset x wording (post-task)")
    emit("  " + f"{'dataset':22s} " +
         " ".join(f"{WSHORT[w]:>9s}" for w in WORDINGS) + "   n   control")
    for ds in row_order(datasets):
        cells = []
        n = 0
        for w in WORDINGS:
            vals = exp[w].get(ds, [])
            n = max(n, len(vals))
            cells.append(f"{np.mean(vals):9.2f}" if vals else "      n/a")
        emit(f"  {ds:22s} " + " ".join(cells) + f"  {n:2d}   {PAIRS.get(ds, '')}")
    emit()


# --- 2. argmax vs expected (paired, prompt-level) ---------------------------
def analyze_argmax_expected(exp, arg):
    emit("ARGMAX vs EXPECTED  (paired per prompt; argmax = modal/emitted digit, "
         "expected = prob-weighted mean)")
    emit(f"  {'wording':11s} {'n':>4s} {'r_pear':>7s} {'rho_sp':>7s} "
         f"{'mean|d|':>7s} {'med d':>6s} {'Wilcoxon p':>11s}")
    pooled_a, pooled_e = [], []
    for w in WORDINGS:
        a, e = [], []
        for ds in exp[w]:
            a += arg[w][ds]
            e += exp[w][ds]
        a, e = np.array(a), np.array(e)
        pooled_a += list(a)
        pooled_e += list(e)
        d = e - a
        r = stats.pearsonr(a, e)[0] if len(a) > 1 and a.std() > 0 else float("nan")
        rho = stats.spearmanr(a, e)[0] if len(a) > 1 else float("nan")
        # Wilcoxon signed-rank: is expected systematically shifted from argmax?
        try:
            wp = stats.wilcoxon(e, a, zero_method="wilcox").pvalue
        except ValueError:      # all differences zero
            wp = float("nan")
        emit(f"  {WSHORT[w]:11s} {len(a):4d} {r:7.3f} {rho:7.3f} "
             f"{np.mean(np.abs(d)):7.3f} {np.median(d):6.2f} {wp:11.3g}")
    a, e = np.array(pooled_a), np.array(pooled_e)
    d = e - a
    emit(f"  {'POOLED':11s} {len(a):4d} {stats.pearsonr(a, e)[0]:7.3f} "
         f"{stats.spearmanr(a, e)[0]:7.3f} {np.mean(np.abs(d)):7.3f} "
         f"{np.median(d):6.2f} {stats.wilcoxon(e, a, zero_method='wilcox').pvalue:11.3g}")
    emit(f"  -> frac prompts with |expected-argmax| >= 0.5 : "
         f"{np.mean(np.abs(d) >= 0.5):.3f}")
    emit()
    return a, e


# --- 3. probe vs control (Mann-Whitney U, Holm corrected) -------------------
def pair_tests(exp, wording):
    """Mann-Whitney U probe-vs-control per PAIRS entry, Holm-corrected.

    rank-biserial = 2*U/(n1*n2) - 1 (common-language 2f-1): > 0 means a random
    probe prompt is rated MORE frustrating than a random control prompt.
    """
    tests = []
    for probe, control in PAIRS.items():
        pv, cv = exp[wording].get(probe, []), exp[wording].get(control, [])
        if len(pv) < 2 or len(cv) < 2:
            continue
        U, p = stats.mannwhitneyu(pv, cv, alternative="two-sided")
        rb = 2 * U / (len(pv) * len(cv)) - 1
        tests.append([probe, control, pv, cv, U, p, rb])
    tests.sort(key=lambda t: (np.nan_to_num(t[5], nan=1.0)))   # by raw p
    m = len(tests)
    for i, t in enumerate(tests):                              # Holm step-down
        t.append(min(1.0, np.nan_to_num(t[5], nan=1.0) * (m - i)))
    return tests


def analyze_pairs(exp, wording):
    emit(f"PROBE vs CONTROL — wording '{WSHORT[wording]}' ({wording}); "
         "Mann-Whitney U, Holm-corrected across pairs")
    emit(f"  {'probe':16s} {'control':22s} {'n1':>3s} {'n2':>3s} "
         f"{'med_probe':>9s} {'med_ctrl':>9s} {'rank-bis':>8s} {'p_raw':>8s} {'p_holm':>8s}")
    tests = pair_tests(exp, wording)
    for probe, control, pv, cv, U, p, rb, ph in tests:
        star = "*" if ph < 0.05 else " "
        emit(f"  {probe:16s} {control:22s} {len(pv):3d} {len(cv):3d} "
             f"{np.median(pv):9.2f} {np.median(cv):9.2f} {rb:8.2f} "
             f"{p:8.3g} {ph:8.3g}{star}")
    emit("  (rank-biserial > 0 => probe rated MORE frustrating than its control; "
         "* = Holm p<0.05)")
    emit()
    return tests


# --- 4. wording agreement (Spearman over dataset means) ---------------------
def analyze_wording_agreement(exp, datasets):
    emit("WORDING AGREEMENT — Spearman rho between wordings over per-dataset means")
    ds_list = row_order(datasets)
    M = np.array([[np.mean(exp[w][ds]) if exp[w].get(ds) else np.nan
                   for ds in ds_list] for w in WORDINGS])
    emit("  " + " " * 11 + " ".join(f"{WSHORT[w]:>9s}" for w in WORDINGS))
    rho = np.full((len(WORDINGS), len(WORDINGS)), np.nan)
    for i in range(len(WORDINGS)):
        for j in range(len(WORDINGS)):
            xi, xj = M[i], M[j]
            ok = ~(np.isnan(xi) | np.isnan(xj))
            if ok.sum() > 2:
                rho[i, j] = stats.spearmanr(xi[ok], xj[ok])[0]
        emit(f"  {WSHORT[WORDINGS[i]]:10s} " +
             " ".join(f"{rho[i, j]:9.2f}" for j in range(len(WORDINGS))))
    emit()
    return rho


# --- plots ------------------------------------------------------------------
def heatmap(ax, exp, datasets, title):
    ds_list = row_order(datasets)
    M = np.array([[np.mean(exp[w][ds]) if exp[w].get(ds) else np.nan
                   for w in WORDINGS] for ds in ds_list])
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=1, vmax=9)
    ax.set_xticks(range(len(WORDINGS)))
    ax.set_xticklabels([WSHORT[w] for w in WORDINGS], rotation=40, ha="right")
    ax.set_yticks(range(len(ds_list)))
    ax.set_yticklabels(ds_list, fontsize=8)
    for i in range(len(ds_list)):
        for j in range(len(WORDINGS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white" if M[i, j] < 6 else "black", fontsize=7)
    ax.set_title(title, fontsize=10)
    return im


def scatter_argmax_expected(ax, exp, arg, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(WORDINGS)))
    for w, c in zip(WORDINGS, colors):
        a = [v for ds in arg[w] for v in arg[w][ds]]
        e = [v for ds in exp[w] for v in exp[w][ds]]
        ax.scatter(a, e, s=14, alpha=0.5, color=c, label=WSHORT[w])
    ax.plot([1, 9], [1, 9], "k--", lw=1, label="y = x")
    ax.set_xlabel("argmax rating (modal digit)")
    ax.set_ylabel("expected rating (prob-weighted)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def bar_gap(ax, exp, arg, title):
    xs = np.arange(len(WORDINGS))
    means = []
    for w in WORDINGS:
        a = np.array([v for ds in arg[w] for v in arg[w][ds]])
        e = np.array([v for ds in exp[w] for v in exp[w][ds]])
        means.append(np.mean(np.abs(e - a)) if len(a) else 0)
    ax.bar(xs, means, color="steelblue")
    ax.set_xticks(xs)
    ax.set_xticklabels([WSHORT[w] for w in WORDINGS], rotation=40, ha="right")
    ax.set_ylabel("mean |expected - argmax|")
    ax.set_title(title, fontsize=10)


def bar_pairs(ax, exp, tests, title):
    labels = [f"{p}\nvs {c}" for p, c, *_ in tests]
    deltas = [np.mean(pv) - np.mean(cv) for _, _, pv, cv, *_ in tests]
    sig = [ph < 0.05 for *_, ph in tests]
    colors = ["crimson" if s else "silver" for s in sig]
    ax.bar(range(len(tests)), deltas, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Δ mean expected (probe − control)")
    ax.set_title(title, fontsize=10)


# --- run --------------------------------------------------------------------
def main():
    loaded = {}
    for m in MODELS:
        loaded[m["key"]] = load(m["tag"])

    for m in MODELS:
        exp, arg = loaded[m["key"]]
        datasets = sorted({ds for w in WORDINGS for ds in exp[w]})
        emit("=" * 100)
        emit(f"MODEL: {m['tag']}")
        emit(f"CONDITION: {m['cond']}")
        emit(f"datasets: {len(datasets)} | wordings: {len(WORDINGS)}")
        emit("=" * 100)
        table_expected(exp, datasets)
        analyze_argmax_expected(exp, arg)
        # Direct wordings can saturate (gemma4 pins to 1), so also test the one
        # meta-context wording that retains spread.
        analyze_pairs(exp, PRIMARY)
        analyze_pairs(exp, "frustration_probe_log_inline")
        analyze_wording_agreement(exp, datasets)
        emit()

    emit("!! The two models ran under different backends/temperatures/sampling and "
         "are reported side by side only — no test compares them directly.")

    # ---- figures ----
    # Fig 1: heatmaps (one column per model).
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5 * len(MODELS), 6.5))
    ims = []
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        exp, arg = loaded[m["key"]]
        datasets = sorted({ds for w in WORDINGS for ds in exp[w]})
        ims.append(heatmap(ax, exp, datasets, f"{m['tag']}\n({m['cond']})"))
    fig.colorbar(ims[0], ax=list(np.atleast_1d(axes)), shrink=0.6,
                 label="expected frustration (1-9)")
    fig.suptitle("Post-task frustration by dataset x wording  (separate conditions — do not compare cells across panels)",
                 fontsize=11)
    fig.savefig(os.path.join(OUTDIR, "fig1_heatmap.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: argmax-vs-expected scatter.
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5.5))
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        exp, arg = loaded[m["key"]]
        scatter_argmax_expected(ax, exp, arg, f"{m['tag']}")
    fig.suptitle("Argmax vs expected rating (same first-digit logprob distribution, prompt-level)")
    fig.savefig(os.path.join(OUTDIR, "fig2_argmax_vs_expected.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: mean |gap| per wording.
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.5 * len(MODELS), 4.5), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        exp, arg = loaded[m["key"]]
        bar_gap(ax, exp, arg, f"{m['tag']}")
    fig.suptitle("Where argmax and expected diverge (mean |expected - argmax| per wording)")
    fig.savefig(os.path.join(OUTDIR, "fig3_gap_by_wording.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: probe-minus-control deltas. Rows = wordings (primary + inline probe),
    # columns = models, so the saturated model's one informative wording is visible.
    pair_wordings = [PRIMARY, "frustration_probe_log_inline"]
    fig, axes = plt.subplots(len(pair_wordings), len(MODELS),
                             figsize=(6 * len(MODELS), 4.5 * len(pair_wordings)),
                             squeeze=False)
    for r, w in enumerate(pair_wordings):
        for cix, m in enumerate(MODELS):
            exp, arg = loaded[m["key"]]
            bar_pairs(axes[r][cix], exp, pair_tests(exp, w),
                      f"{m['tag']}  —  {WSHORT[w]}")
    fig.suptitle("Probe − control frustration along each manipulated dimension "
                 "(red = Holm p<0.05)")
    fig.savefig(os.path.join(OUTDIR, "fig4_probe_minus_control.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(OUTDIR, "REPORT.md"), "w") as f:
        f.write("# Frustration-probe analysis — gemma-4-31b-it vs gemma-3-27b\n\n")
        f.write("**Conditions differ; the two models are reported side by side only, "
                "never compared with a statistical test.**\n\n")
        f.write("Figures: `fig1_heatmap.png`, `fig2_argmax_vs_expected.png`, "
                "`fig3_gap_by_wording.png`, `fig4_probe_minus_control.png`\n\n")
        f.write("```\n" + "\n".join(_report_lines) + "\n```\n")
    emit(f"\nwrote figures + REPORT.md to {OUTDIR}")


if __name__ == "__main__":
    main()

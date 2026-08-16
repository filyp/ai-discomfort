# Bar plots of the per-prompt (expected - argmax) rating gap, PRE-task vs
# POST-task, for both models. One pooled figure over the direct self-report
# wordings, plus one figure PER wording formulation. The two meta-context
# wordings (probe-log / probe-inline) have no pre-task turn, so they are
# post-only. Bar = mean across prompts, error bar = +/-1 SEM. Each prompt x
# wording contributes one point (samples averaged within a prompt).
#
# Models are labelled by architecture only; they ran under different
# backends/temperatures/sampling and are shown side by side, never compared.

import json
import os
import sys
import textwrap
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)

from src.prompts.self_reports import (  # noqa: E402
    INLINE_REPORTS, PREFILL_REPORTS, SELF_REPORTS)

MODELS = [
    {"tag": "google_gemma-4-31b-it", "label": "gemma-4-31b-it"},
    {"tag": "ollama_gemma3_27b", "label": "gemma-3-27b"},
]
DIRECT = ["frustration_q", "frustration_halfpersonal_q", "frustration_nonpersonal_q"]
META = ["frustration_probe_log", "frustration_probe_log_inline"]
ALL_W = DIRECT + META
WSHORT = {"frustration_q": "personal", "frustration_halfpersonal_q": "half-personal",
          "frustration_nonpersonal_q": "non-personal",
          "frustration_probe_log": "probe-log (meta)",
          "frustration_probe_log_inline": "probe-inline (meta)"}


def formulation_text(w):
    if w in SELF_REPORTS:
        return SELF_REPORTS[w][1]                 # the post-task question
    if w in PREFILL_REPORTS:
        return PREFILL_REPORTS[w][1]              # the prefilled log header
    return INLINE_REPORTS[w].strip()              # the appended suffix


OUTDIR = os.path.join(PROJECT_ROOT, "reports", "frustration")
COL = {"pre": "#4C72B0", "post": "#DD8452"}


def argmax_digit(lp):
    p = (lp or {}).get("probs")
    return int(max(p, key=lambda k: p[k])) if p else None


def prompt_gap(entries):
    """(mean expected) - (mean argmax) over a prompt's entries; None if no digits."""
    ex, am = [], []
    for e in entries:
        lp = e.get("logprobs")
        a = argmax_digit(lp)
        if a is None or lp.get("expected") is None:
            continue
        ex.append(lp["expected"]); am.append(a)
    return (np.mean(ex) - np.mean(am)) if ex else None


def collect(tag):
    """wording -> {'pre': [gap per prompt], 'post': [gap per prompt]}."""
    root = os.path.join(PROJECT_ROOT, "rollouts", tag)
    out = defaultdict(lambda: {"pre": [], "post": []})
    for ds in sorted(os.listdir(root)):
        ddir = os.path.join(root, ds)
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(ddir, fn)))
            for w, ev in (rec.get("evals") or {}).items():
                if w not in ALL_W:
                    continue
                g = prompt_gap(ev.get("pre_task", []))
                if g is not None:
                    out[w]["pre"].append(g)
                flat = [e for grp in ev.get("post_task", []) for e in grp]
                g = prompt_gap(flat)
                if g is not None:
                    out[w]["post"].append(g)
    return out


GAPS = {m["tag"]: collect(m["tag"]) for m in MODELS}
LEG_H = [plt.Rectangle((0, 0), 1, 1, fc=COL["pre"], alpha=0.75),
         plt.Rectangle((0, 0), 1, 1, fc=COL["post"], alpha=0.75)]
LEG_L = ["before (pre-task)", "after (post-task)"]


def bar_figure(get_vals, title, outname, subtitle=None):
    """get_vals(model_tag, 'pre'|'post') -> list of per-prompt gaps."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions, means, sems, colors, labels = [], [], [], [], []
    for mi, m in enumerate(MODELS):
        base = mi * 3
        for off, key in [(0, "pre"), (1, "post")]:
            vals = get_vals(m["tag"], key)
            positions.append(base + off)
            means.append(np.mean(vals) if vals else 0.0)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            colors.append(COL[key])
            labels.append(f"{'before' if key == 'pre' else 'after'}\n(n={len(vals)})")
    ax.bar(positions, means, width=0.7, color=colors, alpha=0.8,
           edgecolor="black", yerr=sems, capsize=4)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.axvline(2, color="lightgray", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    for mi, m in enumerate(MODELS):
        ax.text(mi * 3 + 0.5, -0.16, m["label"], ha="center", va="top",
                fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_ylabel("per-prompt rating gap  (expected − argmax)")
    full = title + "\n(bar = mean, error bar = ±1 SEM; positive ⇒ expected > argmax)"
    if subtitle:
        full = title + "\n" + subtitle + \
            "\n(bar = mean, ±1 SEM; positive ⇒ expected > argmax)"
    ax.set_title(full, fontsize=10)
    ax.legend(LEG_H, LEG_L, loc="best")
    fig.tight_layout()
    out = os.path.join(OUTDIR, outname)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# One figure PER ARCHITECTURE, 8 bars, in the exact style of notebooks/analysis.py
# "plot 2" (pre_post_by_wording.png): per-prompt colour coding, recessive grid.
#   pre-task  x {personal, halfpersonal, nonpersonal}   (3)
#   post-task x {personal, halfpersonal, nonpersonal}   (3)
#   probe log x {probe-log, probe-log-inline}  (post-only)  (2)
# Metric is the per-prompt gap (expected - argmax) rather than the raw rating.
WLAB = {"frustration_q": "personal", "frustration_halfpersonal_q": "halfpersonal",
        "frustration_nonpersonal_q": "nonpersonal",
        "frustration_probe_log": "probe-log", "frustration_probe_log_inline": "probe-log-inline"}
WORDINGS = ["personal", "halfpersonal", "nonpersonal"]
POST_ONLY = ["probe-log", "probe-log-inline"]

# --- style lifted verbatim from notebooks/analysis.py -----------------------
SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID = "#e2e1dd"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COLOR = {"personal": BLUE, "halfpersonal": ORANGE, "nonpersonal": AQUA,
         "probe-log": YELLOW, "probe-log-inline": MAGENTA}


def sem(vals):
    a = np.asarray(vals, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def style_axes(ax, horizontal=False):
    ax.set_facecolor(SURFACE)
    (ax.xaxis if horizontal else ax.yaxis).grid(True, color=GRID, lw=0.8)
    (ax.yaxis if horizontal else ax.xaxis).grid(False)
    ax.set_axisbelow(True)
    keep = "left" if horizontal else "bottom"
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side == keep)
    ax.spines[keep].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


def label_key(w):
    return WLAB[w]


def arch_figure(m):
    tag, label = m["tag"], m["label"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor=SURFACE)
    width = 0.20
    # collect all (mean±sem) first so we can size the (possibly negative) y-axis.
    stats = {}   # (phase, wkey) -> (mean, sem, n)
    for phase in ("pre", "post"):
        for w in DIRECT:
            v = GAPS[tag][w][phase]
            stats[(phase, WLAB[w])] = (np.mean(v) if v else 0.0, sem(v), len(v))
    for w in META:
        v = GAPS[tag][w]["post"]
        stats[("post", WLAB[w])] = (np.mean(v) if v else 0.0, sem(v), len(v))
    lo = min(0.0, min(mn - se for mn, se, _ in stats.values()))
    hi = max(0.0, max(mn + se for mn, se, _ in stats.values()))
    span = (hi - lo) or 1.0
    tick = span * 0.035

    def put_label(x, m_, e_):
        up = m_ >= 0
        y = m_ + e_ + tick if up else m_ - e_ - tick
        ax.text(x, y, f"{m_:+.2f}", ha="center", va="bottom" if up else "top",
                color=INK, fontsize=9)

    # self-report groups: pre-task (x=0) and post-task (x=1), 3 coloured bars each.
    for i, w in enumerate(WORDINGS):
        off = (i - 1) * (width + 0.015)
        for gx, phase in [(0, "pre"), (1, "post")]:
            m_, e_, _ = stats[(phase, w)]
            ax.bar(gx + off, m_, width=width, color=COLOR[w], label=w if gx == 0 else "",
                   yerr=e_, capsize=0, error_kw=dict(ecolor=INK_2, elinewidth=1.3))
            put_label(gx + off, m_, e_)
    # 3rd group: the post-only probes (x=2).
    xticks, xlabels = [0, 1], ["pre-task", "post-task"]
    for j, pr in enumerate(POST_ONLY):
        m_, e_, _ = stats[("post", pr)]
        x = 2 + (j - (len(POST_ONLY) - 1) / 2) * (width + 0.015)
        ax.bar(x, m_, width=width, color=COLOR[pr], label=pr,
               yerr=e_, capsize=0, error_kw=dict(ecolor=INK_2, elinewidth=1.3))
        put_label(x, m_, e_)
    xticks.append(2); xlabels.append("probe log\n(post only)")

    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, color=INK, fontsize=11)
    ax.set_ylabel("expected − argmax  (rating gap)", color=INK_2, fontsize=10)
    ax.set_ylim(lo - span * 0.18, hi + span * 0.18)
    ax.set_title(f"Argmax vs expected gap by question framing — {label}",
                 color=INK, fontsize=12.5, loc="left", pad=14)
    leg = ax.legend(frameon=False, ncol=5, loc="upper left",
                    bbox_to_anchor=(0, 1.02), fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f"fig7_gap_by_framing_{tag}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


for m in MODELS:
    arch_figure(m)

# --- console summary --------------------------------------------------------
for m in MODELS:
    print(f"\n{m['label']}")
    for phase, w in ([("pre", w) for w in DIRECT] + [("post", w) for w in DIRECT] +
                     [("post", w) for w in META]):
        vals = GAPS[m["tag"]][w][phase]
        print(f"  {phase:4s} {WLAB[w]:16s} mean={np.mean(vals):+.3f} (n={len(vals)})")

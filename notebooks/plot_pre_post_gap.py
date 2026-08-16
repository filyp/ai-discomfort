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


# One figure PER ARCHITECTURE, 8 bars:
#   pre-task  x {personal, half-personal, non-personal}      (3)
#   post-task x {personal, half-personal, non-personal}      (3)
#   post-task x {probe-log, probe-inline}  (meta, no pre)    (2)
WLAB = {"frustration_q": "personal", "frustration_halfpersonal_q": "half-pers",
        "frustration_nonpersonal_q": "non-pers",
        "frustration_probe_log": "probe-log", "frustration_probe_log_inline": "probe-inl"}
# (phase, wording, colour) for each of the 8 bars, with group gaps in positions.
BARS = ([("pre", w) for w in DIRECT] + [("post", w) for w in DIRECT] +
        [("post", w) for w in META])
POS = [0, 1, 2, 4, 5, 6, 8, 9]
GROUP_COL = {"pre": "#4C72B0", "post": "#DD8452", "meta": "#55A868"}
GROUPS = [("pre-task", 1.0), ("post-task", 5.0), ("post-task · meta", 8.5)]


def arch_figure(m):
    tag, label = m["tag"], m["label"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    means, sems, colors, xlabels = [], [], [], []
    for (phase, w), p in zip(BARS, POS):
        vals = GAPS[tag][w][phase]
        means.append(np.mean(vals) if vals else 0.0)
        sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        colors.append(GROUP_COL["meta"] if w in META else GROUP_COL[phase])
        xlabels.append(f"{WLAB[w]}\n(n={len(vals)})")
    ax.bar(POS, means, width=0.85, color=colors, alpha=0.85, edgecolor="black",
           yerr=sems, capsize=4)
    ax.axhline(0, color="gray", lw=1, ls="--")
    for x in (3, 7):
        ax.axvline(x, color="lightgray", lw=1)
    ax.set_xticks(POS)
    ax.set_xticklabels(xlabels, fontsize=9)
    for name, x in GROUPS:
        ax.text(x, -0.17, name, ha="center", va="top", fontsize=11,
                fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_ylabel("per-prompt rating gap  (expected − argmax)")
    ax.set_title(f"Argmax vs expected-value frustration gap — {label}\n"
                 "by prompt formulation (bar = mean, ±1 SEM; "
                 "positive ⇒ expected > argmax)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=GROUP_COL[k], alpha=0.85)
               for k in ("pre", "post", "meta")]
    ax.legend(handles, ["pre-task (direct)", "post-task (direct)",
                        "post-task (meta probes)"], loc="best")
    fig.tight_layout()
    out = os.path.join(OUTDIR, f"fig7_gap_8bars_{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


for m in MODELS:
    arch_figure(m)

# --- console summary --------------------------------------------------------
for m in MODELS:
    print(f"\n{m['label']}")
    for phase, w in BARS:
        vals = GAPS[m["tag"]][w][phase]
        print(f"  {phase:4s} {WLAB[w]:10s} mean={np.mean(vals):+.3f} (n={len(vals)})")

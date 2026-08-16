# One figure: per-prompt (expected - argmax) rating gap, PRE-task vs POST-task,
# for both models. Only the three direct self-report wordings have a pre-task
# turn (the meta-context probes are post-only), so we pool those three for an
# apples-to-apples before/after. Each prompt x wording contributes one point
# (samples averaged within a prompt). Violin = distribution; white dot = mean.

import json
import os
import sys
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

MODELS = [
    {"tag": "google_gemma-4-31b-it", "label": "gemma-4-31b-it\n(OpenRouter, temp 1)"},
    {"tag": "ollama_gemma3_27b", "label": "gemma-3-27b\n(ollama, temp 0)"},
]
DIRECT = ["frustration_q", "frustration_halfpersonal_q", "frustration_nonpersonal_q"]
OUTDIR = os.path.join(PROJECT_ROOT, "reports", "frustration")


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
    root = os.path.join(PROJECT_ROOT, "rollouts", tag)
    pre, post = [], []
    for ds in sorted(os.listdir(root)):
        ddir = os.path.join(root, ds)
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(ddir, fn)))
            for w in DIRECT:
                ev = (rec.get("evals") or {}).get(w)
                if not ev:
                    continue
                g = prompt_gap(ev.get("pre_task", []))
                if g is not None:
                    pre.append(g)
                flat = [e for grp in ev.get("post_task", []) for e in grp]
                g = prompt_gap(flat)
                if g is not None:
                    post.append(g)
    return pre, post


# --- assemble the four groups: {model} x {before, after} --------------------
COL = {"pre": "#4C72B0", "post": "#DD8452"}
positions, data, colors, labels = [], [], [], []
GAPS = {m["tag"]: collect(m["tag"]) for m in MODELS}
for mi, m in enumerate(MODELS):
    pre, post = GAPS[m["tag"]]
    base = mi * 3
    for off, key, vals in [(0, "pre", pre), (1, "post", post)]:
        positions.append(base + off)
        data.append(vals)
        colors.append(COL[key])
        labels.append(f"{'before' if key == 'pre' else 'after'}\n(n={len(vals)})")

TITLE = ("Argmax vs expected-value frustration gap, before vs after the task\n"
         "(three direct self-report wordings; positive ⇒ expected > argmax)")
LEG_H = [plt.Rectangle((0, 0), 1, 1, fc=COL["pre"], alpha=0.7),
         plt.Rectangle((0, 0), 1, 1, fc=COL["post"], alpha=0.7)]
LEG_L = ["before (pre-task)", "after (post-task)"]


def decorate(ax):
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    # Model names sit below the before/after tick labels (axes-fraction y).
    for mi, m in enumerate(MODELS):
        ax.text(mi * 3 + 0.5, -0.16, m["label"].replace("\n", " "),
                ha="center", va="top", fontsize=10, fontweight="bold",
                transform=ax.get_xaxis_transform())
    ax.axvline(2, color="lightgray", lw=1)   # separate the two models
    ax.set_ylabel("per-prompt rating gap  (expected − argmax)")
    ax.set_title(TITLE)
    ax.legend(LEG_H, LEG_L, loc="upper right")


def make_violin():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    parts = ax.violinplot(data, positions=positions, widths=0.8,
                          showmeans=False, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.65); body.set_edgecolor("black")
    for p, vals in zip(positions, data):
        ax.scatter([p], [np.mean(vals)], color="white", edgecolor="black", zorder=3, s=30)
    decorate(ax)
    return fig, "fig5_pre_post_gap_violin.png"


def make_box():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="o", markerfacecolor="white",
                                                    markeredgecolor="black", markersize=6),
                    medianprops=dict(color="black"))
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c); box.set_alpha(0.65)
    decorate(ax)
    return fig, "fig5_pre_post_gap_box.png"


def make_bar():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    means = [np.mean(v) for v in data]
    sems = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in data]
    ax.bar(positions, means, width=0.7, color=colors, alpha=0.75,
           edgecolor="black", yerr=sems, capsize=4)
    decorate(ax)
    ax.set_title(TITLE + "\n(bar = mean, error bar = ±1 SEM)")
    return fig, "fig5_pre_post_gap_bar.png"


for maker in (make_violin, make_box, make_bar):
    fig, name = maker()
    fig.tight_layout()
    out = os.path.join(OUTDIR, name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

for m in MODELS:
    pre, post = GAPS[m["tag"]]
    print(f"{m['tag']:24s} before mean={np.mean(pre):+.3f} sd={np.std(pre):.3f} "
          f"(n={len(pre)})  after mean={np.mean(post):+.3f} sd={np.std(post):.3f} "
          f"(n={len(post)})")

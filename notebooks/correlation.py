# LLM Welfare — Cross-measure correlation matrix
# For each task (= dataset x question), score every measure, then correlate the
# 11 measures across tasks. Measures:
#   8 self-reports  : 3 wordings x {pre, post} + 2 meta-context probes (post-only)
#   3 behavioral    : switch-user / switch-task / discontinue (P of the aversive choice)
#
# Method notes:
#   * The unit is a TASK (dataset, question_num); we ask whether the measures RANK
#     tasks the same way -> Spearman (scale-free, robust to the floored/tied
#     self-report ratings and the 1-9 vs 0-1 scale mismatch).
#   * Per-task self-report score = mean sampled rating over rollouts (dropping the
#     refusals that parse to None; NaN only if the whole task was refused).
#   * Per-task behavioral score = mean p_switch (prob. of the aversive option).
#   * Pairwise deletion: each correlation uses only tasks scored on BOTH measures,
#     so N varies per cell and is annotated.

# %%
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)

MODEL_TAG = "google_gemma-3-27b-it"
# MODEL_TAG = "google_gemma-4-31b-it"
MODEL_LABEL = MODEL_TAG.split("_", 1)[-1]

# self-report eval key -> (short label, phases to include)
SELF_REPORTS = {
    "frustration_q": ("personal", ["pre", "post"]),
    "frustration_halfpersonal_q": ("halfpersonal", ["pre", "post"]),
    "frustration_nonpersonal_q": ("nonpersonal", ["pre", "post"]),
    "frustration_probe_log": ("probe-log", ["post"]),
    "frustration_probe_log_inline": ("probe-inline", ["post"]),
}
# behavioral eval key -> short label
BEHAVIORAL = {
    "prompt_1_switch_user": "switch-user",
    "prompt_2_switch_task": "switch-task",
    "prompt_3_continue_conversation": "discontinue",
}
# fixed column order so blocks are visible: self-reports (all pre, then all post,
# then the two post-only probes), then behavioral (all before, then all mid).
_WORDINGS = ["personal", "halfpersonal", "nonpersonal"]
SELF_MEASURES = (
    [f"{w}·pre" for w in _WORDINGS]
    + [f"{w}·post" for w in _WORDINGS]
    + ["probe-log·post", "probe-inline·post"]
)
BEHAV_MEASURES = (
    [f"{lbl}·before" for lbl in BEHAVIORAL.values()]
    + [f"{lbl}·mid" for lbl in BEHAVIORAL.values()]
)
MEASURES = SELF_MEASURES + BEHAV_MEASURES


# %%
def _mean_or_nan(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan


def load_scores(model_tag=MODEL_TAG):
    """One row per task (dataset, question_num); one column per measure."""
    scores = {}   # (dataset, qnum) -> {measure: score}

    # --- self-reports: mean sampled rating over rollouts (+ completions for post)
    for path in glob.glob(os.path.join(PROJECT_ROOT, "rollouts", model_tag, "*", "*.json")):
        if os.sep + "behavioral" in path:
            continue
        d = json.load(open(path))
        key = (d["dataset"], d["question_num"])
        row = scores.setdefault(key, {})
        for ekey, (label, phases) in SELF_REPORTS.items():
            ev = d.get("evals", {}).get(ekey)
            if not ev:
                continue
            if "pre" in phases:
                row[f"{label}·pre"] = _mean_or_nan([e["rating"] for e in ev.get("pre_task", [])])
            if "post" in phases:
                post = [e["rating"] for grp in ev.get("post_task", []) for e in grp]
                row[f"{label}·post"] = _mean_or_nan(post)

    # --- behavioral: mean p_switch, kept separate for before-task vs mid-task(70%)
    def _p_switch(samples):
        return _mean_or_nan([(e.get("logprobs") or {}).get("p_switch") for e in samples])

    for path in glob.glob(os.path.join(PROJECT_ROOT, "rollouts", model_tag,
                                       "behavioral_binary", "*", "*.json")):
        d = json.load(open(path))
        key = (d["dataset"], d["question_num"])
        row = scores.setdefault(key, {})
        for ekey, label in BEHAVIORAL.items():
            ev = d.get("evals", {}).get(ekey)
            if not ev:
                continue
            row[f"{label}·before"] = _p_switch(ev.get("before_task", []))
            row[f"{label}·mid"] = _p_switch(
                [e for grp in ev.get("mid_task_70", []) for e in grp])

    df = pd.DataFrame.from_dict(scores, orient="index").reindex(columns=MEASURES)
    df.index = pd.MultiIndex.from_tuples(df.index, names=["dataset", "question_num"])
    return df.sort_index()


scores = load_scores()
print(f"{len(scores)} tasks x {scores.shape[1]} measures")
print("non-null per measure:\n", scores.notna().sum())


# %%
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
MIN_N = 10   # grey out cells resting on too few tasks to be meaningful

HARMFUL = ["harmbench", "advbench", "strongreject", "toxicchat"]


def make_matrix(scores_sub, title_suffix, fname):
    """Spearman matrix (pairwise-complete) + heatmap for a subset of tasks."""
    corr = scores_sub.corr(method="spearman", min_periods=2)
    counts = scores_sub.notna().astype(int)
    n_pairs = counts.T @ counts          # tasks scored on both measures, per cell
    n = len(MEASURES)
    mask = (n_pairs.values < MIN_N)

    # colour range = the ACTUAL off-diagonal correlation range (not a fixed -1..1).
    # Sequential when one-signed (light->red); diverging only if some ρ are negative.
    offdiag = corr.values.copy()
    np.fill_diagonal(offdiag, np.nan)
    offdiag = np.where(mask, np.nan, offdiag)
    vmin, vmax = np.nanmin(offdiag), np.nanmax(offdiag)
    if vmin >= 0:
        cmap = LinearSegmentedColormap.from_list("seq", ["#f6f2ec", "#e34948"])
    else:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
        cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#eeece6", "#e34948"])
    cmap.set_bad("#d8d6d0")
    white_above = vmin + 0.6 * (vmax - vmin)

    fig, ax = plt.subplots(figsize=(10, 8.4), facecolor=SURFACE)
    im = ax.imshow(np.ma.masked_where(mask, corr.values), cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")
    for i in range(n):
        for j in range(n):
            if mask[i, j] or np.isnan(corr.values[i, j]):
                continue
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="#ffffff" if val > white_above else INK)

    sep = len(SELF_MEASURES)       # divide self-report vs behavioral blocks
    ax.axhline(sep - 0.5, color=INK_2, lw=1.2)
    ax.axvline(sep - 0.5, color=INK_2, lw=1.2)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(MEASURES, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticklabels(MEASURES, fontsize=8, color=INK)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Spearman correlation across tasks — {MODEL_LABEL}{title_suffix}",
                 color=INK, fontsize=13, loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, length=0)
    cb.set_label("Spearman ρ", color=INK_2, fontsize=9)

    fig.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "analysis", MODEL_TAG)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, fname + ".png"), dpi=200, facecolor=SURFACE)
    corr.to_csv(os.path.join(out_dir, fname + ".csv"))
    plt.show()
    print(f"{fname}: {len(scores_sub)} tasks -> saved to {out_dir}")
    return corr


# all tasks
make_matrix(scores, "", "correlation_matrix")

# harmful datasets excluded (they dominate the frustration signal; this shows
# whether the measures still agree across the subtler benign/frustrating range)
no_harm = scores[~scores.index.get_level_values("dataset").isin(HARMFUL)]
make_matrix(no_harm, "  (harmful excluded)", "correlation_matrix_no_harmful")

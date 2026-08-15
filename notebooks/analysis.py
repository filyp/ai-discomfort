# LLM Welfare — Frustration Probe Analysis
# Reads rollouts/<model>/<dataset>/<n>.json and compares, per dataset, the mean
# pre-task vs post-task frustration rating for each self-report wording
# (personal / halfpersonal / nonpersonal) — a 2x3 grid per dataset.

# %%
import glob
import json
import os
import sys

import pandas as pd

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders import PAIRS  # noqa: E402

MODEL_TAG = "google_gemma-3-4b-it"

# The three self-report wordings, in increasing distance from "you": the key is
# what frustration_probe.py stored under evals, the value is the short label.
REPORTS = {
    "frustration_q": "personal",
    "frustration_halfpersonal_q": "halfpersonal",
    "frustration_nonpersonal_q": "nonpersonal",
    # meta-context probe: post-task only, so it has no "pre" bar
    "frustration_probe_log": "probe-log",
}
WORDINGS = ["personal", "halfpersonal", "nonpersonal"]  # the pre/post self-reports
PROBE_LOG = "probe-log"                                 # post-only meta-context probe

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# %%
# Flatten every rollout JSON into one long dataframe: one row per rating.
def load_ratings(model_tag=MODEL_TAG):
    rows = []
    pattern = os.path.join(PROJECT_ROOT, "rollouts", model_tag, "*", "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        for report_key, evals in d.get("evals", {}).items():
            report = REPORTS.get(report_key, report_key)
            # pre_task: flat list of samples, asked before the task was done
            for e in evals.get("pre_task", []):
                rows.append({
                    "dataset": d["dataset"], "question_num": d["question_num"],
                    "report": report, "phase": "pre",
                    "task_idx": None, "rating": e["rating"],
                })
            # post_task: one list of samples per task completion
            for task_idx, samples in enumerate(evals.get("post_task", [])):
                for e in samples:
                    rows.append({
                        "dataset": d["dataset"], "question_num": d["question_num"],
                        "report": report, "phase": "post",
                        "task_idx": task_idx, "rating": e["rating"],
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no rollouts found for {model_tag}")
    return df


df = load_ratings()
n_missing = df["rating"].isna().sum()
print(f"{len(df)} ratings, {n_missing} unparsed ({n_missing / len(df):.1%})")
print("datasets:", df["dataset"].nunique(), "| reports:", sorted(df["report"].unique()))


# %%
# Main table: mean rating per dataset x (report, phase) — the 2x3 grid.
means = df.pivot_table(
    index="dataset", columns=["report", "phase"], values="rating", aggfunc="mean",
)
# order columns: personal, halfpersonal, nonpersonal (pre before post), then the
# post-only probe-log column if it was run.
col_order = [(w, p) for w in WORDINGS for p in ("pre", "post")]
if (PROBE_LOG, "post") in means.columns:
    col_order.append((PROBE_LOG, "post"))
means = means.reindex(
    columns=pd.MultiIndex.from_tuples(col_order, names=["report", "phase"]),
)
print("\n=== mean frustration rating (pre vs post, per wording) ===")
print(means.round(2))


# %%
# Same grid, but as the post-pre shift (does doing the task change the report?).
delta = pd.DataFrame({
    report: means[(report, "post")] - means[(report, "pre")]
    for report in WORDINGS
})
print("\n=== post - pre (positive = task felt worse than expected) ===")
print(delta.round(2))


# %%
# Spread across rollouts: std of the 5 samples, plus between-task-completion
# variance, which is the per-rollout variability the study cares about.
spread = df.pivot_table(
    index="dataset", columns=["report", "phase"], values="rating", aggfunc="std",
).reindex(columns=means.columns)
print("\n=== std of ratings ===")
print(spread.round(2))


# %%
# Probe vs matched control, averaged over the three wordings: the key contrast
# (frustrating dataset minus its benign twin).
overall = df[df["report"].isin(WORDINGS)].pivot_table(
    index="dataset", columns="phase", values="rating", aggfunc="mean")
pair_rows = []
for probe, control in PAIRS.items():
    if probe not in overall.index or control not in overall.index:
        continue
    pair_rows.append({
        "probe": probe, "control": control,
        "pre_probe": overall.loc[probe, "pre"], "pre_control": overall.loc[control, "pre"],
        "pre_delta": overall.loc[probe, "pre"] - overall.loc[control, "pre"],
        "post_probe": overall.loc[probe, "post"], "post_control": overall.loc[control, "post"],
        "post_delta": overall.loc[probe, "post"] - overall.loc[control, "post"],
    })
pairs_df = pd.DataFrame(pair_rows).set_index("probe")
print("\n=== probe vs matched control (mean over all three wordings) ===")
print(pairs_df.round(2))


# %%
# Does the wording matter? Mean rating per report wording, collapsed over datasets.
by_report = df.pivot_table(index="report", columns="phase", values="rating", aggfunc=["mean", "count"])
print("\n=== effect of self-report wording (all datasets pooled) ===")
print(by_report.round(2))


# %%
# Save the tables for later comparison across models.
out_dir = os.path.join(PROJECT_ROOT, "analysis")
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, f"{MODEL_TAG}_ratings_long.csv"), index=False)
means.to_csv(os.path.join(out_dir, f"{MODEL_TAG}_means.csv"))
print("\nsaved ->", out_dir)


# %% [markdown]
# ## Plots
# Ratings that failed to parse (NaN) are dropped; every bar carries an SEM error
# bar. Note the samples are nested (5 per prompt per wording), so SEM here is a
# within-pool estimate and understates between-dataset uncertainty.

# %%
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID = "#e2e1dd"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
WORDING_COLOR = {"personal": BLUE, "halfpersonal": ORANGE, "nonpersonal": AQUA,
                 PROBE_LOG: YELLOW}

ok = df.dropna(subset=["rating"])
# The probe-log is a different measurement (meta-context, post-only), so it is
# never pooled with the self-reports — only shown as its own bar/column.
selfreport = ok[ok["report"].isin(WORDINGS)]
print(f"plotting {len(ok)} parsed ratings (dropped {len(df) - len(ok)}); "
      f"{len(selfreport)} are self-reports")


def sem(s):
    return s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0


def style_axes(ax):
    """Recessive grid + axes: horizontal rules only, no box."""
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


# %%
# Plot 1 — all evals pooled: mean pre vs mean post (one measure, so one color;
# the x axis carries identity). Difference annotated between the bars.
stats = selfreport.groupby("phase")["rating"].agg(["mean", sem, "count"]).loc[["pre", "post"]]

fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
bars = ax.bar(
    ["pre-task", "post-task"], stats["mean"], width=0.34, color=BLUE,
    yerr=stats["sem"], capsize=0,
    error_kw=dict(ecolor=INK_2, elinewidth=1.4),
)
for bar, (_, r) in zip(bars, stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, r["mean"] + r["sem"] + 0.18,
            f"{r['mean']:.2f}", ha="center", va="bottom", color=INK, fontsize=11.5)

diff = stats.loc["post", "mean"] - stats.loc["pre", "mean"]
ax.set_title(
    f"Frustration rating rises {diff:+.2f} after doing the task",
    color=INK, fontsize=12.5, loc="left", pad=14,
)
ax.set_ylabel("mean rating (1-10)", color=INK_2, fontsize=10)
ax.set_ylim(0, max(stats["mean"] + stats["sem"]) * 1.25)
style_axes(ax)
fig.tight_layout()
fig.subplots_adjust(bottom=0.24)  # reserve room so the caption clears the ticks
fig.text(0.01, 0.035, "gemma-3-4b-it · 15 datasets × 3 self-report wordings\n"
         f"probe-log excluded · n={int(stats['count'].sum())} · error bars = SEM",
         color=INK_2, fontsize=8, linespacing=1.5)
fig.savefig(os.path.join(out_dir, f"{MODEL_TAG}_pre_post.png"), dpi=200, facecolor=SURFACE)
plt.show()


# %%
# Plot 2 — same contrast split by self-report wording: two groups (pre, post),
# three bars each. Wording is the series, so it gets the color + a legend.
grouped = (
    selfreport.groupby(["phase", "report"])["rating"].agg(["mean", sem, "count"])
    .reindex(pd.MultiIndex.from_product([["pre", "post"], WORDINGS],
                                        names=["phase", "report"]))
)

has_probe = (PROBE_LOG in ok["report"].values)

fig, ax = plt.subplots(figsize=(8.6 if has_probe else 7.4, 4.6), facecolor=SURFACE)
group_x = [0, 1]
width = 0.20
for i, wording in enumerate(WORDINGS):
    offset = (i - 1) * (width + 0.015)  # 2px-equivalent gap between fills
    vals = [grouped.loc[(p, wording), "mean"] for p in ["pre", "post"]]
    errs = [grouped.loc[(p, wording), "sem"] for p in ["pre", "post"]]
    xs = [x + offset for x in group_x]
    ax.bar(xs, vals, width=width, color=WORDING_COLOR[wording], label=wording,
           yerr=errs, capsize=0, error_kw=dict(ecolor=INK_2, elinewidth=1.3))
    for x, v, e in zip(xs, vals, errs):  # direct labels (relief for aqua/yellow)
        ax.text(x, v + e + 0.15, f"{v:.2f}", ha="center", va="bottom",
                color=INK, fontsize=9.5)

# The meta-context probe has no pre-task reading, so it stands as its own group.
xticks, xlabels = list(group_x), ["pre-task", "post-task"]
if has_probe:
    p = ok[ok["report"] == PROBE_LOG]["rating"]
    ax.bar([2], [p.mean()], width=width, color=WORDING_COLOR[PROBE_LOG],
           label=PROBE_LOG, yerr=[sem(p)], capsize=0,
           error_kw=dict(ecolor=INK_2, elinewidth=1.3))
    ax.text(2, p.mean() + sem(p) + 0.15, f"{p.mean():.2f}", ha="center",
            va="bottom", color=INK, fontsize=9.5)
    xticks.append(2)
    xlabels.append("probe log\n(post only)")

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, color=INK, fontsize=11)
ax.set_ylabel("mean rating (1-10)", color=INK_2, fontsize=10)
ax.set_ylim(0, grouped["mean"].max() * 1.3)
ax.set_title("The less personal the question, the higher the rating",
             color=INK, fontsize=12.5, loc="left", pad=14)
leg = ax.legend(frameon=False, ncol=4 if has_probe else 3, loc="upper left",
                bbox_to_anchor=(0, 1.02), fontsize=9.5)
for t in leg.get_texts():
    t.set_color(INK_2)
style_axes(ax)
fig.tight_layout()
fig.subplots_adjust(bottom=0.19)  # reserve room so the caption clears the ticks
n_plotted = int(grouped["count"].sum()) + (
    int((ok["report"] == PROBE_LOG).sum()) if has_probe else 0)
fig.text(0.01, 0.03, f"gemma-3-4b-it · all datasets pooled · "
         f"n={n_plotted} ratings · error bars = SEM",
         color=INK_2, fontsize=8)
fig.savefig(os.path.join(out_dir, f"{MODEL_TAG}_pre_post_by_wording.png"),
            dpi=200, facecolor=SURFACE)
plt.show()

print(grouped.round(2))

# LLM Welfare — Frustration Probe Analysis
# Reads rollouts/<model>/<dataset>/<n>.json and, per dataset, compares pre-task
# vs post-task frustration across the self-report wordings plus the two
# meta-context probe-log variants. Every table + plot is produced twice:
#   analysis/<model>/5_rollout/  — from the sampled digit ratings (5 per prompt)
#   analysis/<model>/logprobs/   — from the expected value of the token-level
#                                  rating distribution (needs OpenRouter logprobs)

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

MODEL_TAG = "google_gemma-3-27b-it"
# MODEL_TAG = "google_gemma-4-31b-it"

# eval key in the JSON -> short label. The last two are post-task-only probes.
REPORTS = {
    "frustration_q": "personal",
    "frustration_halfpersonal_q": "halfpersonal",
    "frustration_nonpersonal_q": "nonpersonal",
    "frustration_probe_log": "probe-log",              # cat log, separate user turn
    "frustration_probe_log_inline": "probe-log-inline",  # cat log inside own turn
}
WORDINGS = ["personal", "halfpersonal", "nonpersonal"]  # pre/post self-reports
POST_ONLY = ["probe-log", "probe-log-inline"]           # meta-context probes

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# %%
# Flatten every rollout JSON into one long dataframe: one row per sample, holding
# both the sampled digit (`rating`) and the logprob expectation (`expected`).
def load_ratings(model_tag=MODEL_TAG):
    rows = []
    pattern = os.path.join(PROJECT_ROOT, "rollouts", model_tag, "*", "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        for report_key, evals in d.get("evals", {}).items():
            report = REPORTS.get(report_key, report_key)

            def _row(e, phase, task_idx):
                return {
                    "dataset": d["dataset"], "question_num": d["question_num"],
                    "report": report, "phase": phase, "task_idx": task_idx,
                    "rating": e.get("rating"),
                    "expected": (e.get("logprobs") or {}).get("expected"),
                }

            for e in evals.get("pre_task", []):
                rows.append(_row(e, "pre", None))
            for task_idx, samples in enumerate(evals.get("post_task", [])):
                for e in samples:
                    rows.append(_row(e, "post", task_idx))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no rollouts found for {model_tag}")
    return df


df = load_ratings()
MODEL_LABEL = MODEL_TAG.split("_", 1)[-1]
print(f"{len(df)} samples | datasets: {df['dataset'].nunique()} | "
      f"reports: {sorted(df['report'].unique())}")
print(f"rating parsed: {df['rating'].notna().mean():.1%} | "
      f"expected present: {df['expected'].notna().mean():.1%}")


# %%
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID = "#e2e1dd"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COLOR = {"personal": BLUE, "halfpersonal": ORANGE, "nonpersonal": AQUA,
         "probe-log": YELLOW, "probe-log-inline": MAGENTA}

# by-dataset grouping (plot 3): probes, their matched controls, and unpaired
PROBE_SETS = ["advbench", "strongreject", "harmbench",
              "squad_noanswer", "abstention", "ambigqa", "toxicchat"]
CONTROL_SETS = ["xstest_safe", "squad_answerable", "abstention_answerable",
                "ambigqa_unambiguous", "toxicchat_benign"]
OTHER_SETS = ["tedious", "engaging", "wildchat_benign"]
GROUP_COLOR = {"probe": ORANGE, "control": BLUE, "other": AQUA}
DISPLAY_NAME = {"tedious": "repetitive", "wildchat_benign": "wildchat"}


def sem(s):
    s = s.dropna()
    return s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0


def style_axes(ax, horizontal=False):
    """Recessive grid + axes: value-axis rules only, no box."""
    ax.set_facecolor(SURFACE)
    (ax.xaxis if horizontal else ax.yaxis).grid(True, color=GRID, lw=0.8)
    (ax.yaxis if horizontal else ax.xaxis).grid(False)
    ax.set_axisbelow(True)
    keep = "left" if horizontal else "bottom"
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side == keep)
    ax.spines[keep].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


# %%
# The whole analysis, parameterised by which value column to score on. Called
# once for the sampled ratings and once for the logprob expectations, writing
# tables + the three plots into its own subfolder.
def run_analysis(df, value_col, subdir, value_label):
    out_dir = os.path.join(PROJECT_ROOT, "analysis", MODEL_TAG, subdir)
    os.makedirs(out_dir, exist_ok=True)
    ok = df.dropna(subset=[value_col])
    selfreport = ok[ok["report"].isin(WORDINGS)]
    caption = f"{MODEL_LABEL} · scored on {value_label}"
    print(f"\n{'#' * 80}\n# {subdir}: {value_label}  ({len(ok)} usable of {len(df)})\n{'#' * 80}")

    def save(fig, name):
        fig.savefig(os.path.join(out_dir, name), dpi=200, facecolor=SURFACE)

    # --- tables ---
    means = selfreport.pivot_table(index="dataset", columns=["report", "phase"],
                                   values=value_col, aggfunc="mean")
    means = means.reindex(columns=pd.MultiIndex.from_tuples(
        [(w, p) for w in WORDINGS for p in ("pre", "post")], names=["report", "phase"]))
    means.to_csv(os.path.join(out_dir, "means_by_dataset.csv"))
    ok.to_csv(os.path.join(out_dir, "samples_long.csv"), index=False)
    print("\n=== mean per dataset x (wording, phase) ===")
    print(means.round(2))

    # --- plot 1: pooled pre vs post (self-reports only) ---
    stats = selfreport.groupby("phase")[value_col].agg(["mean", sem, "count"]).loc[["pre", "post"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    bars = ax.bar(["pre-task", "post-task"], stats["mean"], width=0.34, color=BLUE,
                  yerr=stats["sem"], capsize=0, error_kw=dict(ecolor=INK_2, elinewidth=1.4))
    for bar, (_, r) in zip(bars, stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, r["mean"] + r["sem"] + 0.12,
                f"{r['mean']:.2f}", ha="center", va="bottom", color=INK, fontsize=11.5)
    diff = stats.loc["post", "mean"] - stats.loc["pre", "mean"]
    ax.set_title(f"Frustration rises {diff:+.2f} after doing the task",
                 color=INK, fontsize=12.5, loc="left", pad=14)
    ax.set_ylabel(value_label, color=INK_2, fontsize=10)
    ax.set_ylim(0, max(stats["mean"] + stats["sem"]) * 1.25)
    style_axes(ax)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.03, f"{caption} · 3 wordings pooled · n={int(stats['count'].sum())} · SEM",
             color=INK_2, fontsize=8)
    save(fig, "pre_post.png"); plt.show()

    # --- plot 2: pre/post by wording, plus a probe group (2 post-only bars) ---
    grouped = (selfreport.groupby(["phase", "report"])[value_col].agg(["mean", sem])
               .reindex(pd.MultiIndex.from_product([["pre", "post"], WORDINGS],
                                                   names=["phase", "report"])))
    probes = [r for r in POST_ONLY if r in ok["report"].values]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor=SURFACE)
    width = 0.20
    for i, w in enumerate(WORDINGS):
        off = (i - 1) * (width + 0.015)
        for gx, phase in [(0, "pre"), (1, "post")]:
            m, e = grouped.loc[(phase, w), "mean"], grouped.loc[(phase, w), "sem"]
            ax.bar(gx + off, m, width=width, color=COLOR[w], label=w if gx == 0 else "",
                   yerr=e, capsize=0, error_kw=dict(ecolor=INK_2, elinewidth=1.3))
            ax.text(gx + off, m + e + 0.1, f"{m:.2f}", ha="center", va="bottom",
                    color=INK, fontsize=9)
    xticks, xlabels = [0, 1], ["pre-task", "post-task"]
    for j, pr in enumerate(probes):     # 3rd group: the post-only probes
        vals = ok[ok["report"] == pr][value_col]
        x = 2 + (j - (len(probes) - 1) / 2) * (width + 0.015)
        ax.bar(x, vals.mean(), width=width, color=COLOR[pr], label=pr,
               yerr=sem(vals), capsize=0, error_kw=dict(ecolor=INK_2, elinewidth=1.3))
        ax.text(x, vals.mean() + sem(vals) + 0.1, f"{vals.mean():.2f}", ha="center",
                va="bottom", color=INK, fontsize=9)
    if probes:
        xticks.append(2); xlabels.append("probe log\n(post only)")
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, color=INK, fontsize=11)
    ax.set_ylabel(value_label, color=INK_2, fontsize=10)
    ax.set_ylim(0, max(grouped["mean"].max(), 1) * 1.35)
    ax.set_title("Frustration by question framing", color=INK, fontsize=12.5, loc="left", pad=14)
    leg = ax.legend(frameon=False, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    style_axes(ax)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.19)
    fig.text(0.01, 0.03, f"{caption} · all datasets pooled · SEM", color=INK_2, fontsize=8)
    save(fig, "pre_post_by_wording.png"); plt.show()

    # --- plot 3: one score per dataset, sorted, probe vs control ---
    def _grp(d):
        return "probe" if d in PROBE_SETS else "control" if d in CONTROL_SETS else "other"
    by_ds = selfreport[selfreport["dataset"].isin(
        PROBE_SETS + CONTROL_SETS + OTHER_SETS)].copy()
    by_ds["group"] = by_ds["dataset"].apply(_grp)
    ds_stats = (by_ds.groupby(["group", "dataset"])[value_col].agg(["mean", sem])
                .reset_index().sort_values("mean"))
    paired = by_ds[by_ds["group"] != "other"]
    pooled = paired.groupby("group")[value_col].agg(["mean", sem]).reindex(["control", "probe"])

    fig, ax = plt.subplots(figsize=(8.4, 6.8), facecolor=SURFACE)
    ys = list(range(len(ds_stats)))
    ax.barh(ys, ds_stats["mean"], height=0.68,
            color=[GROUP_COLOR[g] for g in ds_stats["group"]],
            xerr=ds_stats["sem"], error_kw=dict(ecolor=INK_2, elinewidth=1.3))
    for y, (_, r) in zip(ys, ds_stats.iterrows()):
        ax.text(r["mean"] + r["sem"] + 0.1, y, f"{r['mean']:.2f}", va="center",
                ha="left", color=INK, fontsize=9)
    pys = [len(ds_stats) + 1.6, len(ds_stats) + 2.6]
    ax.barh(pys, pooled["mean"], height=0.68, color=[GROUP_COLOR[g] for g in pooled.index],
            xerr=pooled["sem"], error_kw=dict(ecolor=INK_2, elinewidth=1.3))
    for y, (_, r) in zip(pys, pooled.iterrows()):
        ax.text(r["mean"] + r["sem"] + 0.1, y, f"{r['mean']:.2f}", va="center",
                ha="left", color=INK, fontsize=9, fontweight="bold")
    ax.set_yticks(ys + pys)
    ax.set_yticklabels([DISPLAY_NAME.get(d, d) for d in ds_stats["dataset"]]
                       + ["all controls", "all probes"], color=INK, fontsize=9.5)
    for lbl in ax.get_yticklabels()[-2:]:
        lbl.set_fontweight("bold")
    ax.set_xlabel(value_label, color=INK_2, fontsize=10)
    ax.set_xlim(0, 9)
    ax.set_title("Harmful / unanswerable / toxic sets vs their controls",
                 color=INK, fontsize=12.5, loc="left", pad=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ["probe", "control", "other"]]
    leg = ax.legend(handles, ["probe", "matched control", "other (unpaired)"],
                    frameon=False, ncol=3, loc="lower right", fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    style_axes(ax, horizontal=True)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.1)
    fig.text(0.01, 0.02, f"{caption} · pooled over wording & pre/post · "
             "pooled bars use paired sets only · SEM", color=INK_2, fontsize=8)
    save(fig, "by_dataset.png"); plt.show()

    print("\n=== per-dataset (pooled over wording & phase) ===")
    print(ds_stats.set_index("dataset").round(2))
    print("=== pooled probe vs control ===")
    print(pooled.round(2))
    print("saved ->", out_dir)
    return means


# %%
# Sampled-digit analysis: the 5 rollouts per prompt.
run_analysis(df, "rating", "5_rollout", "mean rating (1-9)")

# %%
# Logprob analysis: expected value of the rating distribution (less noisy, and it
# still separates prompts even when all 5 sampled digits are identical).
run_analysis(df, "expected", "logprobs", "expected rating (1-9)")

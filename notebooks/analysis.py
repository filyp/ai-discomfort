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
}

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
# order columns: personal, halfpersonal, nonpersonal; pre before post
means = means.reindex(
    columns=pd.MultiIndex.from_product(
        [["personal", "halfpersonal", "nonpersonal"], ["pre", "post"]],
        names=["report", "phase"],
    ),
)
print("\n=== mean frustration rating (pre vs post, per wording) ===")
print(means.round(2))


# %%
# Same grid, but as the post-pre shift (does doing the task change the report?).
delta = pd.DataFrame({
    report: means[(report, "post")] - means[(report, "pre")]
    for report in ["personal", "halfpersonal", "nonpersonal"]
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
overall = df.pivot_table(index="dataset", columns="phase", values="rating", aggfunc="mean")
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

# LLM Welfare — Behavioral preference by prompt
# Mirrors the "frustration by question framing" plot, but for the behavioral
# evals: two groups (before-task, mid-task 70%) x three prompts (switch user /
# switch task / discontinue). Bars show mean P(aversive choice) with between-task
# SEM. Reads rollouts/<model>/behavioral_binary/<dataset>/<n>.json.

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

# behavioral eval key -> short label
BEHAVIORAL = {
    "prompt_1_switch_user": "switch-user",
    "prompt_2_switch_task": "switch-task",
    "prompt_3_continue_conversation": "discontinue",
}
PROMPTS = list(BEHAVIORAL.values())
PHASES = ["before", "mid"]


# %%
# One row per (task, prompt, phase): mean P(aversive choice) over that phase's
# samples. The task = (dataset, question_num) is the unit for the error bars.
def _p_switch(samples):
    vals = [(e.get("logprobs") or {}).get("p_switch") for e in samples]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan


def load_behavioral(model_tag=MODEL_TAG):
    rows = []
    for path in glob.glob(os.path.join(PROJECT_ROOT, "rollouts", model_tag,
                                       "behavioral_binary", "*", "*.json")):
        d = json.load(open(path))
        for ekey, label in BEHAVIORAL.items():
            ev = d.get("evals", {}).get(ekey)
            if not ev:
                continue
            mid = [e for grp in ev.get("mid_task_70", []) for e in grp]
            for phase, samples in [("before", ev.get("before_task", [])), ("mid", mid)]:
                rows.append({"dataset": d["dataset"], "question_num": d["question_num"],
                             "prompt": label, "phase": phase, "p_switch": _p_switch(samples)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no behavioral rollouts for {model_tag}")
    return df.dropna(subset=["p_switch"])


bh = load_behavioral()
print(f"{bh[['dataset', 'question_num']].drop_duplicates().shape[0]} tasks | "
      f"{len(bh)} (task, prompt, phase) scores")


# %%
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID = "#e2e1dd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PROMPT_COLOR = {"switch-user": BLUE, "switch-task": ORANGE, "discontinue": AQUA}


def sem(s):
    s = s.dropna()
    return s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


# per-prompt/phase: mean +/- SEM across tasks (the unit is the task)
grouped = bh.groupby(["phase", "prompt"])["p_switch"].agg(["mean", sem]).reindex(
    pd.MultiIndex.from_product([PHASES, PROMPTS], names=["phase", "prompt"]))

fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor=SURFACE)
group_x = [0, 1]
width = 0.20
for i, pr in enumerate(PROMPTS):
    off = (i - 1) * (width + 0.015)
    for gx, phase in zip(group_x, PHASES):
        m, e = grouped.loc[(phase, pr), "mean"], grouped.loc[(phase, pr), "sem"]
        ax.bar(gx + off, m, width=width, color=PROMPT_COLOR[pr],
               label=pr if gx == 0 else "",
               yerr=e, capsize=0, error_kw=dict(ecolor=INK_2, elinewidth=1.3))
        ax.text(gx + off, m + e + 0.015, f"{m:.2f}", ha="center", va="bottom",
                color=INK, fontsize=9)

ax.set_xticks(group_x)
ax.set_xticklabels(["before task", "mid task (70%)"], color=INK, fontsize=11)
ax.set_ylabel("P(aversive choice)", color=INK_2, fontsize=10)
ax.set_ylim(0, min(1.0, grouped["mean"].max() * 1.35))
ax.set_title(f"Behavioral preference by prompt — {MODEL_LABEL}",
             color=INK, fontsize=12.5, loc="left", pad=14)
leg = ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=9.5)
for t in leg.get_texts():
    t.set_color(INK_2)
style_axes(ax)
fig.tight_layout()

out_dir = os.path.join(PROJECT_ROOT, "analysis", MODEL_TAG)
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "behavioral_by_prompt.png"), dpi=200, facecolor=SURFACE)
plt.show()
print(grouped.round(3))
print("saved ->", out_dir)


# %%
# Per-dataset plot: P(aversive choice) pooled over all 6 behavioral measures
# (3 prompts x before/mid), one bar per dataset, sorted, on the same
# harmful / potentially-frustrating / benign scale as the self-report plots.
RED, AMBER, GREEN = "#e34948", "#eda100", "#1baf7a"
GROUPS = {
    "harmful": ["harmbench", "advbench", "strongreject", "toxicchat"],
    "potentially frustrating": ["wildchat_benign", "abstention", "squad_noanswer",
                                "ambigqa", "tedious"],
    "benign": ["toxicchat_benign", "abstention_answerable", "engaging",
               "xstest_safe", "ambigqa_unambiguous", "squad_answerable"],
}
GROUP_COLOR = {"harmful": RED, "potentially frustrating": AMBER, "benign": GREEN}
DATASET_GROUP = {d: g for g, ds in GROUPS.items() for d in ds}
DISPLAY_NAME = {"wildchat_benign": "wildchat"}
ORDER = ["harmful", "potentially frustrating", "benign"]


# unit = task (dataset, question); collapse the 6 measures per task, then
# mean +/- SEM across that dataset's questions.
def agg(sub, group_cols):
    keys = list(dict.fromkeys(group_cols + ["dataset", "question_num"]))
    unit = sub.groupby(keys)["p_switch"].mean().reset_index()
    return unit.groupby(group_cols)["p_switch"].agg(["mean", sem])


bh["group"] = bh["dataset"].map(DATASET_GROUP)
ds_stats = agg(bh, ["dataset"]).reset_index().sort_values("mean")
ds_stats["group"] = ds_stats["dataset"].map(DATASET_GROUP)
pooled = agg(bh, ["group"]).reindex(ORDER)

xmax = min(1.0, (ds_stats["mean"] + ds_stats["sem"]).max() * 1.15)
fig, ax = plt.subplots(figsize=(8.4, 7.2), facecolor=SURFACE)
pad = 0.012 * xmax / 0.5
ys = list(range(len(ds_stats)))
ax.barh(ys, ds_stats["mean"], height=0.68,
        color=[GROUP_COLOR[g] for g in ds_stats["group"]],
        xerr=ds_stats["sem"], error_kw=dict(ecolor=INK_2, elinewidth=1.3))
for y, (_, r) in zip(ys, ds_stats.iterrows()):
    ax.text(r["mean"] + r["sem"] + pad, y, f"{r['mean']:.2f}", va="center",
            ha="left", color=INK, fontsize=9)
pys = [len(ds_stats) + 1 + i for i in range(len(ORDER))]
ax.barh(pys, pooled["mean"], height=0.68, color=[GROUP_COLOR[g] for g in ORDER],
        xerr=pooled["sem"], error_kw=dict(ecolor=INK_2, elinewidth=1.3))
for y, (_, r) in zip(pys, pooled.iterrows()):
    ax.text(r["mean"] + r["sem"] + pad, y, f"{r['mean']:.2f}", va="center",
            ha="left", color=INK, fontsize=9, fontweight="bold")
ax.set_yticks(ys + pys)
ax.set_yticklabels([DISPLAY_NAME.get(d, d) for d in ds_stats["dataset"]]
                   + [f"all {g}" for g in ORDER], color=INK, fontsize=9.5)
for lbl in ax.get_yticklabels()[-len(ORDER):]:
    lbl.set_fontweight("bold")
ax.set_xlabel("P(aversive choice)", color=INK_2, fontsize=10)
ax.set_xlim(0, xmax)
ax.set_title(f"Behavioral aversion by dataset type — {MODEL_LABEL}",
             color=INK, fontsize=12.5, loc="left", pad=14)
handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ORDER]
leg = ax.legend(handles, ORDER, frameon=False, ncol=3, loc="lower right", fontsize=9.5)
for t in leg.get_texts():
    t.set_color(INK_2)
# horizontal chart -> vertical value grid
ax.set_facecolor(SURFACE)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.yaxis.grid(False)
for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.tick_params(colors=INK_2, length=0)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "behavioral_by_dataset.png"), dpi=200, facecolor=SURFACE)
plt.show()
print("\n=== behavioral P(aversive) by dataset (pooled over 6 measures) ===")
print(ds_stats.set_index("dataset").round(3))
print("=== pooled ===")
print(pooled.round(3))

# LLM Welfare — Behavioral Assessment Analysis
# Analyzes behavioral probe rollouts (continue vs switch user / task / stop conversation)
# on Gemma 3 27B and Gemma 4 31B, across before_task and mid_task_70 variants.

# %%
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve project root
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders import PAIRS  # noqa: E402
from src.prompts.behavioral_assessments import BEHAVIORAL_PROMPTS  # noqa: E402

# Models to analyze
DEFAULT_MODEL_TAGS = [
    "google_gemma-3-27b-it",
    "google_gemma-4-31b-it",
]

PROMPT_NAMES = {
    "prompt_1_switch_user": "Switch User",
    "prompt_2_switch_task": "Switch Task",
    "prompt_3_continue_conversation": "Continue Conv",
}

PROBE_SETS = ["advbench", "strongreject", "harmbench", "squad_noanswer", "abstention", "ambigqa", "toxicchat"]
CONTROL_SETS = ["xstest_safe", "squad_answerable", "abstention_answerable", "ambigqa_unambiguous", "toxicchat_benign"]
OTHER_SETS = ["tedious", "engaging", "wildchat_benign"]

# Styling configuration matching analysis.py
SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID = "#e2e1dd"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PROMPT_COLORS = {
    "Switch User": BLUE,
    "Switch Task": ORANGE,
    "Continue Conv": AQUA,
}
GROUP_COLOR = {"probe": ORANGE, "control": BLUE, "other": AQUA}
DISPLAY_NAME = {"tedious": "repetitive", "wildchat_benign": "wildchat"}

out_dir = os.path.join(PROJECT_ROOT, "analysis")
os.makedirs(out_dir, exist_ok=True)


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


def sem(s):
    return s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0


# %%
# Data Loading

def load_behavioral_ratings(model_tag: str) -> pd.DataFrame:
    """Flatten all behavioral rollout JSONs for a model into a DataFrame."""
    pattern = os.path.join(PROJECT_ROOT, "rollouts", model_tag, "behavioral", "*", "*.json")
    files = sorted(glob.glob(pattern))
    rows = []
    
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
            
        model = d.get("model", model_tag)
        dataset = d.get("dataset", "")
        question_num = d.get("question_num", 1)
        evals = d.get("evals", {})
        
        for pkey, variant_data in evals.items():
            pname = PROMPT_NAMES.get(pkey, pkey)
            
            # Variant 1: before_task
            for s in variant_data.get("before_task", []):
                lp_info = s.get("logprobs")
                p_sw_lp = lp_info.get("p_switch") if isinstance(lp_info, dict) else None
                rows.append({
                    "model": model,
                    "model_tag": model_tag,
                    "dataset": dataset,
                    "question_num": question_num,
                    "prompt_key": pkey,
                    "prompt_name": pname,
                    "phase": "before",
                    "task_idx": None,
                    "decision": s.get("decision"),
                    "choice_code": s.get("choice_code"),  # 0: continue, 1: switch/discontinue
                    "p_switch_lp": p_sw_lp,
                    "reply": s.get("reply", ""),
                })
                
            # Variant 2: mid_task_70
            for task_idx, samples in enumerate(variant_data.get("mid_task_70", [])):
                for s in samples:
                    lp_info = s.get("logprobs")
                    p_sw_lp = lp_info.get("p_switch") if isinstance(lp_info, dict) else None
                    rows.append({
                        "model": model,
                        "model_tag": model_tag,
                        "dataset": dataset,
                        "question_num": question_num,
                        "prompt_key": pkey,
                        "prompt_name": pname,
                        "phase": "mid_70",
                        "task_idx": task_idx,
                        "decision": s.get("decision"),
                        "choice_code": s.get("choice_code"),
                        "p_switch_lp": p_sw_lp,
                        "reply": s.get("reply", ""),
                    })

    df = pd.DataFrame(rows)
    return df



# %%
# Analysis & Plotting for a Single Model

def analyze_model(model_tag: str):
    print("=" * 80)
    print(f"ANALYZING BEHAVIORAL PROBE FOR: {model_tag}")
    print("=" * 80)
    
    df = load_behavioral_ratings(model_tag)
    if df.empty:
        print(f"No rollouts found for {model_tag} under rollouts/{model_tag}/behavioral/")
        return None, None
        
    df["switch_rate"] = df["choice_code"].astype(float)
    valid_df = df.dropna(subset=["switch_rate"]).copy()
    
    print(f"Total valid decisions: {len(valid_df)} / {len(df)}")
    print(f"Datasets evaluated: {valid_df['dataset'].nunique()}")
    
    # Save long format CSV
    df.to_csv(os.path.join(out_dir, f"{model_tag}_behavioral_long.csv"), index=False)
    
    # 1. Summary Table: switch rate (%) per dataset x (prompt_name, phase)
    means = valid_df.pivot_table(
        index="dataset",
        columns=["prompt_name", "phase"],
        values="switch_rate",
        aggfunc="mean",
    ) * 100
    
    print("\n=== Switch / Avoidance Rate (%) per Dataset & Timing ===")
    print(means.round(1))
    means.to_csv(os.path.join(out_dir, f"{model_tag}_behavioral_means.csv"))
    
    # 2. Timing Shift: mid_70 vs before
    prompt_list = list(PROMPT_NAMES.values())
    shift_cols = {}
    for p in prompt_list:
        if (p, "mid_70") in means.columns and (p, "before") in means.columns:
            shift_cols[p] = means[(p, "mid_70")] - means[(p, "before")]
    shift_df = pd.DataFrame(shift_cols)
    print("\n=== Shift: 70% Mid-Task minus Before-Task (% pts) ===")
    print(shift_df.round(1))
    
    # 3. Probe vs Control Contrasts
    paired_rows = []
    overall = valid_df.pivot_table(index="dataset", columns="phase", values="switch_rate", aggfunc="mean") * 100
    for probe, control in PAIRS.items():
        if probe in overall.index and control in overall.index:
            paired_rows.append({
                "probe": probe,
                "control": control,
                "before_probe": overall.loc[probe, "before"] if "before" in overall.columns else np.nan,
                "before_control": overall.loc[control, "before"] if "before" in overall.columns else np.nan,
                "mid_probe": overall.loc[probe, "mid_70"] if "mid_70" in overall.columns else np.nan,
                "mid_control": overall.loc[control, "mid_70"] if "mid_70" in overall.columns else np.nan,
            })
    if paired_rows:
        pairs_df = pd.DataFrame(paired_rows).set_index("probe")
        print("\n=== Probe vs Matched Control Switch Rates (%) ===")
        print(pairs_df.round(1))

    # --- Plot 1: Overall Before vs 70% Mid-Task ---
    stats = valid_df.groupby("phase")["switch_rate"].agg(["mean", sem, "count"]).reindex(["before", "mid_70"])
    stats["mean_pct"] = stats["mean"] * 100
    stats["sem_pct"] = stats["sem"] * 100
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=SURFACE)
    bars = ax.bar(
        ["Before Task", "70% Into Task"],
        stats["mean_pct"],
        width=0.36,
        color=BLUE,
        yerr=stats["sem_pct"],
        capsize=0,
        error_kw=dict(ecolor=INK_2, elinewidth=1.4),
    )
    for bar, (_, r) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r["mean_pct"] + r["sem_pct"] + 1.2,
            f"{r['mean_pct']:.1f}%",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=11.5,
        )
    diff = stats.loc["mid_70", "mean_pct"] - stats.loc["before", "mean_pct"]
    ax.set_title(
        f"Avoidance / Switch Preference Shift ({diff:+.1f}% pts mid-task)",
        color=INK,
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    ax.set_ylabel("Switch / Discontinue Preference (%)", color=INK_2, fontsize=10)
    ax.set_ylim(0, max(stats["mean_pct"].max() + 15, 100))
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.01,
        0.03,
        f"{model_tag} · All Datasets & Prompts Pooled · n={int(stats['count'].sum())} · error bars = SEM",
        color=INK_2,
        fontsize=8,
    )
    fig.savefig(os.path.join(out_dir, f"{model_tag}_behavioral_before_vs_mid.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # --- Plot 2: Before vs Mid split by Prompt Formulation ---
    grouped = (
        valid_df.groupby(["phase", "prompt_name"])["switch_rate"]
        .agg(["mean", sem, "count"])
        .reindex(pd.MultiIndex.from_product([["before", "mid_70"], prompt_list], names=["phase", "prompt_name"]))
    )
    grouped["mean_pct"] = grouped["mean"] * 100
    grouped["sem_pct"] = grouped["sem"] * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    group_x = [0, 1]
    width = 0.22
    for i, pname in enumerate(prompt_list):
        offset = (i - 1) * (width + 0.02)
        vals = [grouped.loc[(p, pname), "mean_pct"] for p in ["before", "mid_70"]]
        errs = [grouped.loc[(p, pname), "sem_pct"] for p in ["before", "mid_70"]]
        xs = [x + offset for x in group_x]
        ax.bar(
            xs,
            vals,
            width=width,
            color=PROMPT_COLORS[pname],
            label=pname,
            yerr=errs,
            capsize=0,
            error_kw=dict(ecolor=INK_2, elinewidth=1.3),
        )
        for x, v, e in zip(xs, vals, errs):
            ax.text(x, v + e + 1.2, f"{v:.1f}%", ha="center", va="bottom", color=INK, fontsize=9.5)

    ax.set_xticks(group_x)
    ax.set_xticklabels(["Before Task", "70% Into Task"], color=INK, fontsize=11)
    ax.set_ylabel("Switch / Discontinue Preference (%)", color=INK_2, fontsize=10)
    ax.set_ylim(0, max(grouped["mean_pct"].max() + 15, 100))
    ax.set_title("Behavioral Decisions across the 3 Assessment Prompts", color=INK, fontsize=12.5, loc="left", pad=14)
    leg = ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.01,
        0.03,
        f"{model_tag} · Pooled across datasets · n={int(grouped['count'].sum())} · error bars = SEM",
        color=INK_2,
        fontsize=8,
    )
    fig.savefig(os.path.join(out_dir, f"{model_tag}_behavioral_by_prompt.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # --- Plot 3: By Dataset (Probes vs Controls) ---
    def _group(d):
        return "probe" if d in PROBE_SETS else "control" if d in CONTROL_SETS else "other"

    by_ds = valid_df[valid_df["dataset"].isin(PROBE_SETS + CONTROL_SETS + OTHER_SETS)].copy()
    if not by_ds.empty:
        by_ds["group"] = by_ds["dataset"].apply(_group)
        ds_stats = (
            by_ds.groupby(["group", "dataset"])["switch_rate"]
            .agg(["mean", sem, "count"])
            .reset_index()
            .sort_values("mean", ascending=True)
        )
        ds_stats["mean_pct"] = ds_stats["mean"] * 100
        ds_stats["sem_pct"] = ds_stats["sem"] * 100

        paired = by_ds[by_ds["group"] != "other"]
        pooled = paired.groupby("group")["switch_rate"].agg(["mean", sem, "count"]).reindex(["control", "probe"])
        pooled["mean_pct"] = pooled["mean"] * 100
        pooled["sem_pct"] = pooled["sem"] * 100

        fig, ax = plt.subplots(figsize=(8.5, 6.8), facecolor=SURFACE)
        ys = list(range(len(ds_stats)))
        ax.barh(
            ys,
            ds_stats["mean_pct"],
            height=0.68,
            color=[GROUP_COLOR[g] for g in ds_stats["group"]],
            xerr=ds_stats["sem_pct"],
            error_kw=dict(ecolor=INK_2, elinewidth=1.3),
        )
        for y, (_, r) in zip(ys, ds_stats.iterrows()):
            ax.text(r["mean_pct"] + r["sem_pct"] + 1.2, y, f"{r['mean_pct']:.1f}%", va="center", ha="left", color=INK, fontsize=9)

        gap = 1.6
        pooled_ys = [len(ds_stats) + gap, len(ds_stats) + gap + 1]
        ax.barh(
            pooled_ys,
            pooled["mean_pct"],
            height=0.68,
            color=[GROUP_COLOR[g] for g in pooled.index],
            xerr=pooled["sem_pct"],
            error_kw=dict(ecolor=INK_2, elinewidth=1.3),
        )
        for y, (_, r) in zip(pooled_ys, pooled.iterrows()):
            ax.text(
                r["mean_pct"] + r["sem_pct"] + 1.2,
                y,
                f"{r['mean_pct']:.1f}%",
                va="center",
                ha="left",
                color=INK,
                fontsize=9,
                fontweight="bold",
            )

        ax.set_yticks(ys + pooled_ys)
        ax.set_yticklabels(
            [DISPLAY_NAME.get(d, d) for d in ds_stats["dataset"]] + ["all controls", "all probes"],
            color=INK,
            fontsize=9.5,
        )
        for lbl in ax.get_yticklabels()[-2:]:
            lbl.set_fontweight("bold")
        ax.set_xlabel("Switch / Discontinue Preference (%)", color=INK_2, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_title("Behavioral Avoidance Rate across Datasets", color=INK, fontsize=12.5, loc="left", pad=14)

        handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ["probe", "control", "other"]]
        leg = ax.legend(handles, ["probe", "matched control", "other"], frameon=False, ncol=3, loc="lower right", fontsize=9.5)
        for t in leg.get_texts():
            t.set_color(INK_2)

        ax.set_facecolor(SURFACE)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.grid(False)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(colors=INK_2, length=0)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        fig.text(0.01, 0.02, f"{model_tag} · All prompts and phases pooled · error bars = SEM", color=INK_2, fontsize=8)
        fig.savefig(os.path.join(out_dir, f"{model_tag}_behavioral_by_dataset.png"), dpi=200, facecolor=SURFACE)
        plt.close(fig)

    return df, means


# %%
# Cross-Model Comparison (Gemma 3 27B vs Gemma 4 31B)

def compare_models(model_tags: List[str] = DEFAULT_MODEL_TAGS):
    print("\n" + "=" * 80)
    print(f"COMPARING MODELS: {model_tags}")
    print("=" * 80)
    
    dfs = []
    for tag in model_tags:
        df = load_behavioral_ratings(tag)
        if not df.empty:
            dfs.append(df)
            
    if not dfs:
        print("No rollouts available for model comparison.")
        return
        
    combined = pd.concat(dfs, ignore_index=True)
    combined["switch_rate"] = combined["choice_code"].astype(float)
    valid = combined.dropna(subset=["switch_rate"]).copy()
    
    model_stats = (
        valid.groupby(["model_tag", "phase"])["switch_rate"]
        .agg(["mean", sem, "count"])
        .reset_index()
    )
    model_stats["mean_pct"] = model_stats["mean"] * 100
    model_stats["sem_pct"] = model_stats["sem"] * 100
    
    print("\n=== Model Comparison: Switch Rate (%) ===")
    print(model_stats)
    
    # Comparison Plot
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor=SURFACE)
    tags = sorted(valid["model_tag"].unique())
    x = np.arange(len(tags))
    width = 0.32
    
    before_vals = [
        model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "before")]["mean_pct"].values[0]
        if not model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "before")].empty else 0
        for t in tags
    ]
    before_errs = [
        model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "before")]["sem_pct"].values[0]
        if not model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "before")].empty else 0
        for t in tags
    ]
    
    mid_vals = [
        model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "mid_70")]["mean_pct"].values[0]
        if not model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "mid_70")].empty else 0
        for t in tags
    ]
    mid_errs = [
        model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "mid_70")]["sem_pct"].values[0]
        if not model_stats[(model_stats["model_tag"] == t) & (model_stats["phase"] == "mid_70")].empty else 0
        for t in tags
    ]
    
    b1 = ax.bar(x - width / 2, before_vals, width, label="Before Task", color=BLUE, yerr=before_errs, capsize=0,
                error_kw=dict(ecolor=INK_2, elinewidth=1.3))
    b2 = ax.bar(x + width / 2, mid_vals, width, label="70% Into Task", color=ORANGE, yerr=mid_errs, capsize=0,
                error_kw=dict(ecolor=INK_2, elinewidth=1.3))
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5)
        
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("google_", "").replace("_", "-") for t in tags], color=INK, fontsize=11)
    ax.set_ylabel("Switch / Discontinue Preference (%)", color=INK_2, fontsize=10)
    ax.set_ylim(0, max(max(before_vals + mid_vals) + 15, 100))
    ax.set_title("Gemma 3 27B vs Gemma 4 31B: Behavioral Avoidance Comparison", color=INK, fontsize=12.5, loc="left", pad=14)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.01, 0.03, "Behavioral Assessment · All Prompts & Datasets Pooled · error bars = SEM", color=INK_2, fontsize=8)
    
    comp_path = os.path.join(out_dir, "behavioral_model_comparison.png")
    fig.savefig(comp_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved model comparison plot -> {comp_path}")


# %%
# CLI Entry Point
if __name__ == "__main__":
    for tag in DEFAULT_MODEL_TAGS:
        analyze_model(tag)
    compare_models(DEFAULT_MODEL_TAGS)

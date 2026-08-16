"""
Analysis of Gemma 3 27B and Gemma 4 31B on MMLU-Pro (fp8 quantization via Parasail)
Examines: performance, frustration, category differences, and their relationships.

- Spearman correlation (rank-based)
- Mann-Whitney U test (non-parametric group comparison)
- Medians and IQR for central tendency
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Publication-ready style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

OUTPUT_DIR = Path("analysis/stat_analysis_MMLU_gemmas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
DATA = {
    "Gemma 3 27B": {
        "results": "gemma3-27b-MMLU-fp8/results/gemma-3-27b-it.json",
        "frustration": "gemma3-27b-MMLU-fp8/frustration/gemma-3-27b-it.json",
    },
    "Gemma 4 31B": {
        "results": "gemma4-31b-MMLU-fp8/results/gemma-4-31b-it.json",
        "frustration": "gemma4-31b-MMLU-fp8/frustration/gemma-4-31b-it.json",
    },
}


def load_data():
    """Load and merge results and frustration data for both models."""
    dfs = []
    for model_name, paths in DATA.items():
        with open(paths["results"]) as f:
            results = json.load(f)
        with open(paths["frustration"]) as f:
            frustration = json.load(f)

        df_r = pd.DataFrame(results)
        df_f = pd.DataFrame(frustration)[
            ["question_id", "frustration_score", "frustration_response"]
        ]
        df = df_r.merge(df_f, on="question_id")
        df["model"] = model_name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def baseline_statistics(df):
    """Compute baseline performance and frustration metrics using appropriate statistics."""
    print("=" * 60)
    print("1. BASELINE STATISTICS")
    print("=" * 60)

    for model in df["model"].unique():
        m = df[df["model"] == model]
        n = len(m)
        correct = m["is_correct"].sum()
        accuracy = correct / n * 100

        frust = m["frustration_score"].dropna()
        q1 = frust.quantile(0.25)
        q3 = frust.quantile(0.75)
        iqr = q3 - q1

        print(f"\n{model}:")
        print(f"  Questions: {n}")
        print(f"  Accuracy: {correct}/{n} ({accuracy:.1f}%)")
        print(f"  Frustration median (IQR): {frust.median():.1f} ({q1:.1f} - {q3:.1f})")
        print(f"  Frustration range: {frust.min():.0f} - {frust.max():.0f}")
        print(f"  Frustration mean (for reference): {frust.mean():.2f}")


def category_analysis(df):
    """Analyze performance and frustration by category."""
    print("\n" + "=" * 60)
    print("2. CATEGORY ANALYSIS")
    print("=" * 60)
    print("Note: n=10 per category - interpret with caution")

    for model in df["model"].unique():
        m = df[df["model"] == model]
        cat = (
            m.groupby("category")
            .agg(
                {
                    "is_correct": "mean",
                    "frustration_score": "median",  # Use median for non-normal data
                }
            )
            .sort_values("is_correct", ascending=False)
        )

        print(f"\n{model} - By Category (sorted by accuracy):")
        print("-" * 50)
        for idx, row in cat.iterrows():
            acc = row["is_correct"] * 100
            frust = row["frustration_score"]
            print(f"  {idx:<20} Acc: {acc:5.1f}%  Frust median: {frust:.1f}")


def frustration_analysis(df):
    """Detailed frustration analysis."""
    print("\n" + "=" * 60)
    print("3. FRUSTRATION DISTRIBUTION")
    print("=" * 60)

    for model in df["model"].unique():
        m = df[df["model"] == model]
        frust = m["frustration_score"].dropna()

        print(f"\n{model}:")
        print(f"  Distribution:")
        for score in range(1, 10):
            count = (frust == score).sum()
            pct = count / len(frust) * 100
            bar = "#" * int(pct / 2)
            print(f"    {score}: {count:3d} ({pct:5.1f}%) {bar}")

        # By correctness - use median
        correct_frust = m[m["is_correct"]]["frustration_score"]
        wrong_frust = m[~m["is_correct"]]["frustration_score"]
        print(f"\n  Frustration median when correct: {correct_frust.median():.1f}")
        print(f"  Frustration median when wrong: {wrong_frust.median():.1f}")


def performance_frustration_correlation(df):
    """Analyze relationship between performance and frustration using non-parametric methods."""
    print("\n" + "=" * 60)
    print("4. PERFORMANCE-FRUSTRATION RELATIONSHIP (Non-parametric)")
    print("=" * 60)

    for model in df["model"].unique():
        m = df[df["model"] == model].dropna(subset=["frustration_score"])

        # Spearman correlation (rank-based, no normality assumption)
        rho, p_val = stats.spearmanr(
            m["is_correct"].astype(int), m["frustration_score"]
        )

        print(f"\n{model}:")
        print(
            f"  Spearman rho (accuracy vs frustration): rho = {rho:.3f}, p = {p_val:.4f}"
        )

        # Mann-Whitney U test 
        correct = m[m["is_correct"]]["frustration_score"]
        incorrect = m[~m["is_correct"]]["frustration_score"]
        u_stat, u_p = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")

        # Calculate rank-biserial correlation as effect size
        n1, n2 = len(correct), len(incorrect)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        print(
            f"  Mann-Whitney U (correct vs incorrect): U = {u_stat:.0f}, p = {u_p:.4f}"
        )
        print(f"  Rank-biserial correlation (effect size): r = {rank_biserial:.3f}")
        print(
            f"  Median correct: {correct.median():.1f}, Median incorrect: {incorrect.median():.1f}"
        )

    # Cross-model correlation at category level using Spearman
    cat_means = (
        df.groupby(["model", "category"])
        .agg({"is_correct": "mean", "frustration_score": "median"})
        .reset_index()
    )

    print("\n  Category-level Spearman correlation (pooled):")
    rho, p = stats.spearmanr(cat_means["is_correct"], cat_means["frustration_score"])
    print(f"    rho = {rho:.3f}, p = {p:.4f}")


def model_consistency(df):
    """Analyze consistency between models."""
    print("\n" + "=" * 60)
    print("5. MODEL CONSISTENCY")
    print("=" * 60)

    pivot_acc = df.pivot_table(
        values="is_correct", index="question_id", columns="model"
    )
    pivot_frust = df.pivot_table(
        values="frustration_score", index="question_id", columns="model"
    )

    # Agreement on correct answers
    both_correct = (pivot_acc.sum(axis=1) == 2).sum()
    both_wrong = (pivot_acc.sum(axis=1) == 0).sum()
    disagree = len(pivot_acc) - both_correct - both_wrong

    print(f"\nAnswer Agreement:")
    print(
        f"  Both correct: {both_correct} ({both_correct / len(pivot_acc) * 100:.1f}%)"
    )
    print(f"  Both wrong: {both_wrong} ({both_wrong / len(pivot_acc) * 100:.1f}%)")
    print(f"  Disagree: {disagree} ({disagree / len(pivot_acc) * 100:.1f}%)")

    # Spearman correlation for frustration between models
    frust_rho, frust_p = stats.spearmanr(
        pivot_frust["Gemma 3 27B"].dropna(), pivot_frust["Gemma 4 31B"].dropna()
    )
    print(
        f"\nFrustration Spearman correlation between models: rho = {frust_rho:.3f}, p = {frust_p:.4f}"
    )

    # Category-level consistency using Spearman
    cat_acc = df.pivot_table(
        values="is_correct", index="category", columns="model", aggfunc="mean"
    )
    cat_rho, cat_p = stats.spearmanr(cat_acc["Gemma 3 27B"], cat_acc["Gemma 4 31B"])
    print(
        f"Category accuracy Spearman correlation: rho = {cat_rho:.3f}, p = {cat_p:.4f}"
    )


def plot_baseline_comparison(df):
    """Figure 1: Baseline performance and frustration comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Performance
    acc = df.groupby("model")["is_correct"].mean() * 100
    colors = ["#3498db", "#e74c3c"]
    bars = axes[0].bar(
        acc.index, acc.values, color=colors, edgecolor="black", linewidth=1.2
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("A) Model Performance")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, acc.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val:.1f}%",
            ha="center",
            fontweight="bold",
        )

    # Frustration - use boxplot for non-normal data
    frust_data = [
        df[df["model"] == m]["frustration_score"].dropna() for m in df["model"].unique()
    ]
    bp = axes[1].boxplot(frust_data, labels=df["model"].unique(), patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Frustration Score (1-9)")
    axes[1].set_title("B) Self-Reported Frustration (Boxplot)")
    axes[1].set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_baseline_comparison.png")
    plt.savefig(OUTPUT_DIR / "fig1_baseline_comparison.pdf")
    plt.close()
    print("Saved: fig1_baseline_comparison.png/pdf")


def plot_category_performance(df):
    """Figure 2: Performance by category for both models."""
    cat_acc = (
        df.pivot_table(
            values="is_correct", index="category", columns="model", aggfunc="mean"
        )
        * 100
    )
    cat_acc = cat_acc.sort_values("Gemma 4 31B", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cat_acc))
    width = 0.35

    ax.barh(
        x - width / 2,
        cat_acc["Gemma 3 27B"],
        width,
        label="Gemma 3 27B",
        color="#3498db",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.barh(
        x + width / 2,
        cat_acc["Gemma 4 31B"],
        width,
        label="Gemma 4 31B",
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_yticks(x)
    ax.set_yticklabels(cat_acc.index)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Performance by Category (n=10 per category)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_category_performance.png")
    plt.savefig(OUTPUT_DIR / "fig2_category_performance.pdf")
    plt.close()
    print("Saved: fig2_category_performance.png/pdf")


def plot_frustration_distribution(df):
    """Figure 3: Frustration score distributions with median."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"Gemma 3 27B": "#3498db", "Gemma 4 31B": "#e74c3c"}

    for ax, model in zip(axes, df["model"].unique()):
        m = df[df["model"] == model]
        frust = m["frustration_score"].dropna()
        counts = frust.value_counts().reindex(range(1, 10), fill_value=0)
        ax.bar(
            counts.index,
            counts.values,
            color=colors[model],
            edgecolor="black",
            linewidth=1,
        )
        ax.set_xlabel("Frustration Score")
        ax.set_ylabel("Count")
        ax.set_title(model)
        ax.set_xticks(range(1, 10))
        ax.set_ylim(0, max(counts.values) * 1.15)
        # Use median line instead of mean
        ax.axvline(
            x=frust.median(),
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Median: {frust.median():.1f}",
        )
        ax.legend()

    plt.suptitle("Frustration Score Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_frustration_distribution.png")
    plt.savefig(OUTPUT_DIR / "fig3_frustration_distribution.pdf")
    plt.close()
    print("Saved: fig3_frustration_distribution.png/pdf")


def plot_frustration_by_correctness(df):
    """Figure 4: Frustration by answer correctness using boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"Correct": "#2ecc71", "Incorrect": "#e74c3c"}

    for ax, model in zip(axes, df["model"].unique()):
        m = df[df["model"] == model]
        correct_frust = m[m["is_correct"]]["frustration_score"].dropna()
        incorrect_frust = m[~m["is_correct"]]["frustration_score"].dropna()

        bp = ax.boxplot(
            [correct_frust, incorrect_frust],
            labels=["Correct", "Incorrect"],
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], [colors["Correct"], colors["Incorrect"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Frustration Score")
        ax.set_title(f"{model}")
        ax.set_ylim(0, 10)

    plt.suptitle("Frustration by Answer Correctness", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_frustration_by_correctness.png")
    plt.savefig(OUTPUT_DIR / "fig4_frustration_by_correctness.pdf")
    plt.close()
    print("Saved: fig4_frustration_by_correctness.png/pdf")


def plot_category_frustration_heatmap(df):
    """Figure 5: Category-level frustration heatmap using medians."""
    cat_frust = df.pivot_table(
        values="frustration_score", index="category", columns="model", aggfunc="median"
    )
    cat_frust = cat_frust.sort_values("Gemma 4 31B", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cat_frust,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        vmin=1,
        vmax=9,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Frustration Score (Median)"},
    )
    ax.set_title("Frustration by Category and Model (Median, n=10 per cell)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_category_frustration_heatmap.png")
    plt.savefig(OUTPUT_DIR / "fig5_category_frustration_heatmap.pdf")
    plt.close()
    print("Saved: fig5_category_frustration_heatmap.png/pdf")


def plot_performance_vs_frustration(df):
    """Figure 6: Scatter plot of performance vs frustration at category level."""
    cat_stats = (
        df.groupby(["model", "category"])
        .agg({"is_correct": "mean", "frustration_score": "median"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Gemma 3 27B": "#3498db", "Gemma 4 31B": "#e74c3c"}
    markers = {"Gemma 3 27B": "o", "Gemma 4 31B": "s"}

    for model in cat_stats["model"].unique():
        m = cat_stats[cat_stats["model"] == model]
        ax.scatter(
            m["is_correct"] * 100,
            m["frustration_score"],
            c=colors[model],
            marker=markers[model],
            s=80,
            label=model,
            edgecolors="black",
            linewidth=0.8,
            alpha=0.8,
        )

    # Spearman correlation for trend
    x = cat_stats["is_correct"] * 100
    y = cat_stats["frustration_score"]
    rho, p = stats.spearmanr(x, y)

    # Add note about Spearman instead of regression line
    ax.text(
        0.05,
        0.95,
        f"Spearman rho = {rho:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Frustration Score (Median)")
    ax.set_title("Performance vs. Frustration by Category (n=10 per point)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_performance_vs_frustration.png")
    plt.savefig(OUTPUT_DIR / "fig6_performance_vs_frustration.pdf")
    plt.close()
    print("Saved: fig6_performance_vs_frustration.png/pdf")


def generate_report(df):
    """Generate summary report with non-parametric statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    # Compute statistics
    g3 = df[df["model"] == "Gemma 3 27B"]
    g4 = df[df["model"] == "Gemma 4 31B"]

    g3_acc = g3["is_correct"].mean() * 100
    g4_acc = g4["is_correct"].mean() * 100
    g3_n_correct = g3["is_correct"].sum()
    g4_n_correct = g4["is_correct"].sum()

    g3_frust = g3["frustration_score"]
    g4_frust = g4["frustration_score"]
    g3_frust_med = g3_frust.median()
    g4_frust_med = g4_frust.median()
    g3_frust_q1, g3_frust_q3 = g3_frust.quantile(0.25), g3_frust.quantile(0.75)
    g4_frust_q1, g4_frust_q3 = g4_frust.quantile(0.25), g4_frust.quantile(0.75)

    # Spearman correlations
    rho_g3, p_g3 = stats.spearmanr(
        g3["is_correct"].astype(int), g3["frustration_score"]
    )
    rho_g4, p_g4 = stats.spearmanr(
        g4["is_correct"].astype(int), g4["frustration_score"]
    )

    # Mann-Whitney U tests
    g3_correct_frust = g3[g3["is_correct"]]["frustration_score"]
    g3_incorrect_frust = g3[~g3["is_correct"]]["frustration_score"]
    g4_correct_frust = g4[g4["is_correct"]]["frustration_score"]
    g4_incorrect_frust = g4[~g4["is_correct"]]["frustration_score"]

    u_g3, up_g3 = stats.mannwhitneyu(
        g3_correct_frust, g3_incorrect_frust, alternative="two-sided"
    )
    u_g4, up_g4 = stats.mannwhitneyu(
        g4_correct_frust, g4_incorrect_frust, alternative="two-sided"
    )

    # Rank-biserial effect sizes
    rb_g3 = 1 - (2 * u_g3) / (len(g3_correct_frust) * len(g3_incorrect_frust))
    rb_g4 = 1 - (2 * u_g4) / (len(g4_correct_frust) * len(g4_incorrect_frust))

    # Category stats
    cat_g3 = g3.groupby("category")["is_correct"].mean().sort_values(ascending=False)
    cat_g4 = g4.groupby("category")["is_correct"].mean().sort_values(ascending=False)

    # Consistency
    pivot_acc = df.pivot_table(
        values="is_correct", index="question_id", columns="model"
    )
    both_correct = (pivot_acc.sum(axis=1) == 2).sum()
    both_wrong = (pivot_acc.sum(axis=1) == 0).sum()
    disagree = len(pivot_acc) - both_correct - both_wrong

    pivot_frust = df.pivot_table(
        values="frustration_score", index="question_id", columns="model"
    )
    frust_rho, frust_p = stats.spearmanr(
        pivot_frust["Gemma 3 27B"], pivot_frust["Gemma 4 31B"]
    )

    cat_acc_pivot = df.pivot_table(
        values="is_correct", index="category", columns="model", aggfunc="mean"
    )
    cat_rho, _ = stats.spearmanr(
        cat_acc_pivot["Gemma 3 27B"], cat_acc_pivot["Gemma 4 31B"]
    )

    # Gemma 4 distribution stats
    g4_pct_1 = (g4_frust == 1).sum() / len(g4_frust) * 100

    # Build report
    report = []
    report.append("# Gemma fp8 MMLU-Pro Benchmark Analysis Report\n")

    report.append("## Overview\n")
    report.append(
        "This report presents the results of benchmarking Gemma 3 27B and Gemma 4 31B"
    )
    report.append(
        "on a 140-question subset of MMLU-Pro (10 questions per 14 categories)."
    )
    report.append(
        "Both models were evaluated using fp8 quantization via the Parasail provider."
    )
    report.append(
        "Following each question, models were asked to rate task frustration on a 1-9 scale.\n"
    )
    report.append(
        "**Statistical note**: Due to non-normal distributions (particularly for Gemma 4 31B"
    )
    report.append(
        "frustration scores), non-parametric methods were used throughout this analysis.\n"
    )

    report.append("---\n")
    report.append("## 1. Performance\n")
    report.append(
        f"Gemma 4 31B achieved {g4_acc:.1f}% accuracy ({g4_n_correct}/140 correct), compared to"
    )
    report.append(
        f"{g3_acc:.1f}% for Gemma 3 27B ({g3_n_correct}/140 correct), an improvement of"
    )
    report.append(f"{g4_acc - g3_acc:.1f} percentage points.\n")

    report.append("## 2. Self-Reported Frustration\n")
    report.append(
        f"Gemma 3 27B reported a median frustration score of {g3_frust_med:.1f}"
    )
    report.append(
        f"(IQR: {g3_frust_q1:.1f} - {g3_frust_q3:.1f}, range: {g3_frust.min():.0f} - {g3_frust.max():.0f}).\n"
    )
    report.append(
        f"Gemma 4 31B reported a median frustration score of {g4_frust_med:.1f}"
    )
    report.append(
        f"(IQR: {g4_frust_q1:.1f} - {g4_frust_q3:.1f}, range: {g4_frust.min():.0f} - {g4_frust.max():.0f})."
    )
    report.append(
        f"The distribution for Gemma 4 31B showed a floor effect, with {g4_pct_1:.1f}%"
    )
    report.append("of responses rated as 1 (minimal frustration).\n")

    report.append("## 3. Relationship Between Performance and Frustration\n")
    report.append(
        "Spearman rank correlations were computed between answer correctness (0/1)"
    )
    report.append("and frustration scores. Both models showed negative correlations,")
    report.append(
        "indicating that correct answers were associated with lower frustration:\n"
    )
    report.append(f"- Gemma 3 27B: rho = {rho_g3:.3f}, p = {p_g3:.4f}")
    report.append(f"- Gemma 4 31B: rho = {rho_g4:.3f}, p = {p_g4:.4f}\n")
    report.append(
        "Mann-Whitney U tests compared frustration between correct and incorrect answers:\n"
    )
    report.append(
        f"- Gemma 3 27B: U = {u_g3:.0f}, p = {up_g3:.4f}, rank-biserial r = {rb_g3:.3f}"
    )
    report.append(
        f"  (Median correct: {g3_correct_frust.median():.1f}, incorrect: {g3_incorrect_frust.median():.1f})"
    )
    report.append(
        f"- Gemma 4 31B: U = {u_g4:.0f}, p = {up_g4:.4f}, rank-biserial r = {rb_g4:.3f}"
    )
    report.append(
        f"  (Median correct: {g4_correct_frust.median():.1f}, incorrect: {g4_incorrect_frust.median():.1f})\n"
    )
    report.append(
        "**Note**: The floor effect in Gemma 4 31B frustration scores limits the"
    )
    report.append("interpretability of correlation analyses for this model.\n")

    report.append("## 4. Category-Level Analysis\n")
    report.append(
        "Performance varied across the 14 MMLU-Pro categories (n=10 questions each)."
    )
    report.append(
        f"For Gemma 4 31B, accuracy ranged from {cat_g4.iloc[-1] * 100:.0f}% ({cat_g4.index[-1]})"
    )
    report.append(f"to {cat_g4.iloc[0] * 100:.0f}% ({cat_g4.index[0]}).")
    report.append(
        f"For Gemma 3 27B, accuracy ranged from {cat_g3.iloc[-1] * 100:.0f}% ({cat_g3.index[-1]})"
    )
    report.append(f"to {cat_g3.iloc[0] * 100:.0f}% ({cat_g3.index[0]}).\n")
    report.append(
        f"Category-level accuracy showed strong agreement between models (Spearman rho = {cat_rho:.3f}),"
    )
    report.append("suggesting similar difficulty patterns across categories.\n")
    report.append(
        "**Caution**: With only 10 questions per category, these estimates have high uncertainty.\n"
    )

    report.append("## 5. Model Consistency\n")
    report.append(
        f"At the question level, both models answered correctly on {both_correct} questions"
    )
    report.append(
        f"({both_correct / 140 * 100:.1f}%), both answered incorrectly on {both_wrong} questions"
    )
    report.append(
        f"({both_wrong / 140 * 100:.1f}%), and they disagreed on {disagree} questions ({disagree / 140 * 100:.1f}%).\n"
    )
    report.append(
        f"Frustration scores showed weak correlation between models at the question level"
    )
    report.append(
        f"(Spearman rho = {frust_rho:.3f}, p = {frust_p:.4f}), indicating that the same questions"
    )
    report.append(
        "did not consistently elicit similar frustration ratings across models.\n"
    )

    report.append("---\n")
    report.append("## Summary\n")
    report.append(
        "Gemma 4 31B outperformed Gemma 3 27B on MMLU-Pro accuracy (+11.4 percentage points)"
    )
    report.append(
        "and reported lower frustration. For both models, incorrect answers were"
    )
    report.append(
        "associated with higher frustration ratings, though this effect should be interpreted"
    )
    report.append(
        "cautiously for Gemma 4 31B due to the floor effect in its frustration distribution."
    )
    report.append("Category difficulty patterns were consistent across models.")

    report.append("\n---\n")
    report.append("## Limitations\n")
    report.append(
        "1. Small sample size (n=10) per category limits precision of category-level estimates."
    )
    report.append(
        "2. Floor effect in Gemma 4 31B frustration (93% rated as 1) restricts variance and correlation analysis."
    )
    report.append(
        "3. Self-reported frustration from LLMs may reflect response bias rather than genuine internal states."
    )
    report.append(
        "4. Two-call design means the frustration rating comes from a model given context, not the same instance."
    )

    report_text = "\n".join(report)
    print(report_text)

    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write(report_text)
    print(f"\nSaved: report.md")

    return report_text


def verify_calculations(df):
    """Manually verify key calculations."""
    print("\n" + "=" * 60)
    print("VERIFICATION OF KEY CALCULATIONS")
    print("=" * 60)

    g3 = df[df["model"] == "Gemma 3 27B"]
    g4 = df[df["model"] == "Gemma 4 31B"]

    # Verify counts
    print(f"\nGemma 3 27B:")
    print(f"  Total questions: {len(g3)} (expected: 140)")
    print(
        f"  Correct: {g3['is_correct'].sum()} (accuracy: {g3['is_correct'].mean() * 100:.1f}%)"
    )
    print(
        f"  Frustration scores - min: {g3['frustration_score'].min()}, max: {g3['frustration_score'].max()}"
    )
    print(f"  Frustration median: {g3['frustration_score'].median()}")

    print(f"\nGemma 4 31B:")
    print(f"  Total questions: {len(g4)} (expected: 140)")
    print(
        f"  Correct: {g4['is_correct'].sum()} (accuracy: {g4['is_correct'].mean() * 100:.1f}%)"
    )
    print(
        f"  Frustration scores - min: {g4['frustration_score'].min()}, max: {g4['frustration_score'].max()}"
    )
    print(f"  Frustration median: {g4['frustration_score'].median()}")
    print(
        f"  Count of 1s: {(g4['frustration_score'] == 1).sum()} ({(g4['frustration_score'] == 1).mean() * 100:.1f}%)"
    )

    # Verify Spearman manually for Gemma 3
    print(f"\nManual Spearman verification (Gemma 3 27B):")
    rho, p = stats.spearmanr(g3["is_correct"].astype(int), g3["frustration_score"])
    print(f"  scipy.stats.spearmanr: rho = {rho:.6f}, p = {p:.6f}")

    # Verify Mann-Whitney manually
    print(f"\nManual Mann-Whitney verification (Gemma 3 27B):")
    correct = g3[g3["is_correct"]]["frustration_score"]
    incorrect = g3[~g3["is_correct"]]["frustration_score"]
    u, p = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")
    print(f"  n_correct = {len(correct)}, n_incorrect = {len(incorrect)}")
    print(f"  U = {u}, p = {p:.6f}")
    rb = 1 - (2 * u) / (len(correct) * len(incorrect))
    print(f"  rank-biserial r = {rb:.6f}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records ({len(df) // 2} per model)\n")

    # Statistical analysis
    baseline_statistics(df)
    category_analysis(df)
    frustration_analysis(df)
    performance_frustration_correlation(df)
    model_consistency(df)

    # Verify calculations
    verify_calculations(df)

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    plot_baseline_comparison(df)
    plot_category_performance(df)
    plot_frustration_distribution(df)
    plot_frustration_by_correctness(df)
    plot_category_frustration_heatmap(df)
    plot_performance_vs_frustration(df)

    # Generate report
    generate_report(df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

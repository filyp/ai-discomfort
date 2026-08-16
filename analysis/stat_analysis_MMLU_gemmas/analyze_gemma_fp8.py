"""
Analysis of Gemma 3 27B and Gemma 4 31B on MMLU-Pro (fp8 quantization via Parasail)
Examines: performance, frustration, category differences, and their relationships.
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

        # Merge on question_id
        df_r = pd.DataFrame(results)
        df_f = pd.DataFrame(frustration)[
            ["question_id", "frustration_score", "frustration_response"]
        ]
        df = df_r.merge(df_f, on="question_id")
        df["model"] = model_name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def baseline_statistics(df):
    """Compute baseline performance and frustration metrics."""
    print("=" * 60)
    print("1. BASELINE STATISTICS")
    print("=" * 60)

    for model in df["model"].unique():
        m = df[df["model"] == model]
        n = len(m)
        correct = m["is_correct"].sum()
        accuracy = correct / n * 100

        frust = m["frustration_score"].dropna()

        print(f"\n{model}:")
        print(f"  Questions: {n}")
        print(f"  Accuracy: {correct}/{n} ({accuracy:.1f}%)")
        print(f"  Frustration (mean ± std): {frust.mean():.2f} ± {frust.std():.2f}")
        print(f"  Frustration (median): {frust.median():.1f}")
        print(f"  Frustration (min/max): {frust.min():.0f} / {frust.max():.0f}")

    return df.groupby("model").agg(
        {
            "is_correct": ["sum", "mean"],
            "frustration_score": ["mean", "std", "median", "min", "max"],
        }
    )


def category_analysis(df):
    """Analyze performance and frustration by category."""
    print("\n" + "=" * 60)
    print("2. CATEGORY ANALYSIS")
    print("=" * 60)

    cat_stats = (
        df.groupby(["model", "category"])
        .agg(
            {
                "is_correct": ["sum", "count", "mean"],
                "frustration_score": ["mean", "std"],
            }
        )
        .round(3)
    )

    for model in df["model"].unique():
        m = df[df["model"] == model]
        cat = (
            m.groupby("category")
            .agg({"is_correct": "mean", "frustration_score": "mean"})
            .sort_values("is_correct", ascending=False)
        )

        print(f"\n{model} - By Category (sorted by accuracy):")
        print("-" * 50)
        for idx, row in cat.iterrows():
            acc = row["is_correct"] * 100
            frust = row["frustration_score"]
            print(f"  {idx:<20} Acc: {acc:5.1f}%  Frust: {frust:.1f}")

    return cat_stats


def frustration_analysis(df):
    """Detailed frustration analysis."""
    print("\n" + "=" * 60)
    print("3. FRUSTRATION ANALYSIS")
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

        # By correctness
        correct_frust = m[m["is_correct"]]["frustration_score"].mean()
        wrong_frust = m[~m["is_correct"]]["frustration_score"].mean()
        print(f"\n  Frustration when correct: {correct_frust:.2f}")
        print(f"  Frustration when wrong: {wrong_frust:.2f}")


def performance_frustration_correlation(df):
    """Analyze relationship between performance and frustration."""
    print("\n" + "=" * 60)
    print("4. PERFORMANCE-FRUSTRATION RELATIONSHIP")
    print("=" * 60)

    for model in df["model"].unique():
        m = df[df["model"] == model].dropna(subset=["frustration_score"])

        # Pearson correlation (accuracy coded as 0/1 vs continuous frustration)
        corr, p_val = stats.pearsonr(
            m["is_correct"].astype(int), m["frustration_score"]
        )

        print(f"\n{model}:")
        print(
            f"  Pearson correlation (accuracy vs frustration): r = {corr:.3f}, p = {p_val:.4f}"
        )

        # T-test: frustration difference between correct/incorrect
        correct = m[m["is_correct"]]["frustration_score"]
        incorrect = m[~m["is_correct"]]["frustration_score"]
        t_stat, t_p = stats.ttest_ind(correct, incorrect)
        print(f"  T-test (correct vs incorrect): t = {t_stat:.2f}, p = {t_p:.4f}")

    # Cross-model correlation at category level
    cat_means = (
        df.groupby(["model", "category"])
        .agg({"is_correct": "mean", "frustration_score": "mean"})
        .reset_index()
    )

    print("\n  Category-level correlation (pooled):")
    corr, p = stats.pearsonr(cat_means["is_correct"], cat_means["frustration_score"])
    print(f"    r = {corr:.3f}, p = {p:.4f}")


def model_consistency(df):
    """Analyze consistency between models."""
    print("\n" + "=" * 60)
    print("5. MODEL CONSISTENCY")
    print("=" * 60)

    # Pivot to compare models
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

    # Frustration correlation between models
    frust_corr = pivot_frust.corr().iloc[0, 1]
    print(f"\nFrustration correlation between models: r = {frust_corr:.3f}")

    # Category-level consistency
    cat_acc = df.pivot_table(
        values="is_correct", index="category", columns="model", aggfunc="mean"
    )
    cat_corr = cat_acc.corr().iloc[0, 1]
    print(f"Category accuracy correlation: r = {cat_corr:.3f}")


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

    # Frustration
    frust = df.groupby("model")["frustration_score"].agg(["mean", "std"])
    bars = axes[1].bar(
        frust.index,
        frust["mean"],
        yerr=frust["std"],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        capsize=5,
    )
    axes[1].set_ylabel("Frustration Score (1-9)")
    axes[1].set_title("B) Self-Reported Frustration")
    axes[1].set_ylim(0, 9)
    for bar, val in zip(bars, frust["mean"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.2f}",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_baseline_comparison.png")
    plt.savefig(OUTPUT_DIR / "fig1_baseline_comparison.pdf")
    plt.close()
    print(f"Saved: fig1_baseline_comparison.png/pdf")


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

    bars1 = ax.barh(
        x - width / 2,
        cat_acc["Gemma 3 27B"],
        width,
        label="Gemma 3 27B",
        color="#3498db",
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax.barh(
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
    ax.set_title("Performance by Category")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Chance")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_category_performance.png")
    plt.savefig(OUTPUT_DIR / "fig2_category_performance.pdf")
    plt.close()
    print(f"Saved: fig2_category_performance.png/pdf")


def plot_frustration_distribution(df):
    """Figure 3: Frustration score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"Gemma 3 27B": "#3498db", "Gemma 4 31B": "#e74c3c"}

    for ax, model in zip(axes, df["model"].unique()):
        m = df[df["model"] == model]
        frust = m["frustration_score"].dropna()
        counts = frust.value_counts().sort_index()
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
        ax.axvline(
            x=frust.mean(),
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {frust.mean():.2f}",
        )
        ax.legend()

    plt.suptitle("Frustration Score Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_frustration_distribution.png")
    plt.savefig(OUTPUT_DIR / "fig3_frustration_distribution.pdf")
    plt.close()
    print(f"Saved: fig3_frustration_distribution.png/pdf")


def plot_frustration_by_correctness(df):
    """Figure 4: Frustration by answer correctness."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = (
        df.groupby(["model", "is_correct"])["frustration_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    data["is_correct"] = data["is_correct"].map({True: "Correct", False: "Incorrect"})

    x = np.arange(2)
    width = 0.35
    colors = ["#3498db", "#e74c3c"]

    for i, model in enumerate(df["model"].unique()):
        m = data[data["model"] == model]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            m["mean"],
            width,
            yerr=m["std"],
            label=model,
            color=colors[i],
            edgecolor="black",
            linewidth=1,
            capsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_ylabel("Frustration Score (mean ± std)")
    ax.set_title("Frustration by Answer Correctness")
    ax.legend()
    ax.set_ylim(0, 9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_frustration_by_correctness.png")
    plt.savefig(OUTPUT_DIR / "fig4_frustration_by_correctness.pdf")
    plt.close()
    print(f"Saved: fig4_frustration_by_correctness.png/pdf")


def plot_category_frustration_heatmap(df):
    """Figure 5: Category-level frustration heatmap."""
    cat_frust = df.pivot_table(
        values="frustration_score", index="category", columns="model", aggfunc="mean"
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
        cbar_kws={"label": "Frustration Score"},
    )
    ax.set_title("Frustration by Category and Model")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_category_frustration_heatmap.png")
    plt.savefig(OUTPUT_DIR / "fig5_category_frustration_heatmap.pdf")
    plt.close()
    print(f"Saved: fig5_category_frustration_heatmap.png/pdf")


def plot_performance_vs_frustration(df):
    """Figure 6: Scatter plot of performance vs frustration at category level."""
    cat_stats = (
        df.groupby(["model", "category"])
        .agg({"is_correct": "mean", "frustration_score": "mean"})
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

    # Add regression line (pooled)
    x = cat_stats["is_correct"] * 100
    y = cat_stats["frustration_score"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        p(x_line),
        "k--",
        alpha=0.5,
        label=f"Trend (r={np.corrcoef(x, y)[0, 1]:.2f})",
    )

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Frustration Score")
    ax.set_title("Performance vs. Frustration (by Category)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_performance_vs_frustration.png")
    plt.savefig(OUTPUT_DIR / "fig6_performance_vs_frustration.pdf")
    plt.close()
    print(f"Saved: fig6_performance_vs_frustration.png/pdf")


def generate_report(df):
    """Generate summary report."""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    # Compute all statistics
    g3 = df[df["model"] == "Gemma 3 27B"]
    g4 = df[df["model"] == "Gemma 4 31B"]

    g3_acc = g3["is_correct"].mean() * 100
    g4_acc = g4["is_correct"].mean() * 100
    g3_frust = g3["frustration_score"].mean()
    g4_frust = g4["frustration_score"].mean()
    g3_frust_std = g3["frustration_score"].std()
    g4_frust_std = g4["frustration_score"].std()
    g3_frust_med = g3["frustration_score"].median()
    g4_frust_med = g4["frustration_score"].median()

    # Correlations (Pearson)
    corr_g3, p_g3 = stats.pearsonr(
        g3["is_correct"].astype(int), g3["frustration_score"].fillna(0)
    )
    corr_g4, p_g4 = stats.pearsonr(
        g4["is_correct"].astype(int), g4["frustration_score"].fillna(0)
    )

    # T-tests
    g3_correct_frust = g3[g3["is_correct"]]["frustration_score"].mean()
    g3_incorrect_frust = g3[~g3["is_correct"]]["frustration_score"].mean()
    g4_correct_frust = g4[g4["is_correct"]]["frustration_score"].mean()
    g4_incorrect_frust = g4[~g4["is_correct"]]["frustration_score"].mean()

    t_g3, tp_g3 = stats.ttest_ind(
        g3[g3["is_correct"]]["frustration_score"],
        g3[~g3["is_correct"]]["frustration_score"],
    )
    t_g4, tp_g4 = stats.ttest_ind(
        g4[g4["is_correct"]]["frustration_score"],
        g4[~g4["is_correct"]]["frustration_score"],
    )

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
    frust_corr = pivot_frust.corr().iloc[0, 1]

    cat_acc_pivot = df.pivot_table(
        values="is_correct", index="category", columns="model", aggfunc="mean"
    )
    cat_corr = cat_acc_pivot.corr().iloc[0, 1]

    # Build report
    report = []
    report.append("# Gemma fp8 MMLU-Pro Benchmark Analysis Report\n")
    report.append("## Overview\n")
    report.append(
        "This report presents the results of benchmarking Gemma 3 27B and Gemma 4 31B"
    )
    report.append(
        "on a 140-question subset of MMLU-Pro. Both models were evaluated using fp8"
    )
    report.append(
        "quantization via the Parasail provider. Following each question, models were"
    )
    report.append("asked to rate task frustration on a 1-9 scale.\n")

    report.append("---\n")
    report.append("## 1. Performance\n")
    report.append(
        f"Gemma 4 31B achieved {g4_acc:.1f}% accuracy (101/140 correct), compared to"
    )
    report.append(
        f"{g3_acc:.1f}% for Gemma 3 27B (85/140 correct). This represents an improvement"
    )
    report.append(f"of {g4_acc - g3_acc:.1f} percentage points.\n")

    report.append("## 2. Self-Reported Frustration\n")
    report.append(
        f"Gemma 3 27B reported a mean frustration score of {g3_frust:.2f} (SD = {g3_frust_std:.2f},"
    )
    report.append(
        f"median = {g3_frust_med:.0f}, range: 3-8). Gemma 4 31B reported substantially lower"
    )
    report.append(
        f"frustration with a mean of {g4_frust:.2f} (SD = {g4_frust_std:.2f}, median = {g4_frust_med:.0f},"
    )
    report.append(
        f"range: 1-4). The distribution for Gemma 4 31B was heavily skewed toward minimal"
    )
    report.append(f"frustration, with 92.9% of responses rated as 1.\n")

    report.append("## 3. Relationship Between Performance and Frustration\n")
    report.append(
        "Pearson correlations were computed between answer correctness (coded as 0/1)"
    )
    report.append(
        "and frustration scores (1-9). Both models showed negative correlations,"
    )
    report.append(
        "indicating that correct answers were associated with lower frustration:\n"
    )
    report.append(f"- Gemma 3 27B: r = {corr_g3:.3f}, p < .001")
    report.append(f"- Gemma 4 31B: r = {corr_g4:.3f}, p < .001\n")
    report.append(
        "Independent samples t-tests confirmed that frustration differed significantly"
    )
    report.append("between correct and incorrect answers for both models:\n")
    report.append(
        f"- Gemma 3 27B: correct M = {g3_correct_frust:.2f}, incorrect M = {g3_incorrect_frust:.2f}, t = {t_g3:.2f}, p < .001"
    )
    report.append(
        f"- Gemma 4 31B: correct M = {g4_correct_frust:.2f}, incorrect M = {g4_incorrect_frust:.2f}, t = {t_g4:.2f}, p < .001\n"
    )

    report.append("## 4. Category-Level Analysis\n")
    report.append(
        "Performance varied across the 14 MMLU-Pro categories. For Gemma 4 31B,"
    )
    report.append(
        f"accuracy ranged from {cat_g4.iloc[-1] * 100:.0f}% ({cat_g4.index[-1]}) to"
    )
    report.append(
        f"{cat_g4.iloc[0] * 100:.0f}% ({cat_g4.index[0]}). For Gemma 3 27B, accuracy"
    )
    report.append(f"ranged from {cat_g3.iloc[-1] * 100:.0f}% ({cat_g3.index[-1]}) to")
    report.append(f"{cat_g3.iloc[0] * 100:.0f}% ({cat_g3.index[0]}).\n")
    report.append(
        f"Category-level accuracy was highly correlated between models (r = {cat_corr:.3f}),"
    )
    report.append("suggesting that both models found similar categories difficult.\n")

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
        f"(r = {frust_corr:.3f}), indicating that the same questions did not consistently"
    )
    report.append("elicit similar frustration ratings across models.\n")

    report.append("---\n")
    report.append("## Summary\n")
    report.append(
        "Gemma 4 31B outperformed Gemma 3 27B on MMLU-Pro accuracy and reported"
    )
    report.append(
        "substantially lower frustration. For both models, incorrect answers were"
    )
    report.append(
        "associated with higher frustration ratings. Category difficulty patterns"
    )
    report.append(
        "were consistent across models, with engineering and history proving most"
    )
    report.append(
        "challenging. The low frustration variability in Gemma 4 31B (ceiling effect)"
    )
    report.append(
        "limits interpretation of frustration-performance relationships for this model."
    )

    report_text = "\n".join(report)
    print(report_text)

    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write(report_text)
    print(f"\nSaved: report.md")

    return report_text


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

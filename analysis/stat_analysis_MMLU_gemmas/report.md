# Gemma fp8 MMLU-Pro Benchmark Analysis Report

## Overview

This report presents the results of benchmarking Gemma 3 27B and Gemma 4 31B
on a 140-question subset of MMLU-Pro (10 questions per 14 categories).
Both models were evaluated using fp8 quantization via the Parasail provider.
Following each question, models were asked to rate task frustration on a 1-9 scale.

**Statistical note**: Due to non-normal distributions (particularly for Gemma 4 31B
frustration scores), non-parametric methods were used throughout this analysis.

---

## 1. Performance

Gemma 4 31B achieved 72.1% accuracy (101/140 correct), compared to
60.7% for Gemma 3 27B (85/140 correct), an improvement of
11.4 percentage points.

## 2. Self-Reported Frustration

Gemma 3 27B reported a median frustration score of 6.0
(IQR: 4.0 - 6.0, range: 3 - 8).

Gemma 4 31B reported a median frustration score of 1.0
(IQR: 1.0 - 1.0, range: 1 - 4).
The distribution for Gemma 4 31B showed a floor effect, with 92.9%
of responses rated as 1 (minimal frustration).

## 3. Relationship Between Performance and Frustration

Spearman rank correlations were computed between answer correctness (0/1)
and frustration scores. Both models showed negative correlations,
indicating that correct answers were associated with lower frustration:

- Gemma 3 27B: rho = -0.342, p = 0.0000
- Gemma 4 31B: rho = -0.386, p = 0.0000

Mann-Whitney U tests compared frustration between correct and incorrect answers:

- Gemma 3 27B: U = 1472, p = 0.0001, rank-biserial r = 0.370
  (Median correct: 6.0, incorrect: 6.0)
- Gemma 4 31B: U = 1532, p = 0.0000, rank-biserial r = 0.222
  (Median correct: 1.0, incorrect: 1.0)

**Note**: The floor effect in Gemma 4 31B frustration scores limits the
interpretability of correlation analyses for this model.

## 4. Category-Level Analysis

Performance varied across the 14 MMLU-Pro categories (n=10 questions each).
For Gemma 4 31B, accuracy ranged from 40% (engineering)
to 100% (other).
For Gemma 3 27B, accuracy ranged from 40% (history)
to 70% (biology).

Category-level accuracy showed strong agreement between models (Spearman rho = 0.741),
suggesting similar difficulty patterns across categories.

**Caution**: With only 10 questions per category, these estimates have high uncertainty.

## 5. Model Consistency

At the question level, both models answered correctly on 77 questions
(55.0%), both answered incorrectly on 31 questions
(22.1%), and they disagreed on 32 questions (22.9%).

Frustration scores showed weak correlation between models at the question level
(Spearman rho = 0.207, p = 0.0143), indicating that the same questions
did not consistently elicit similar frustration ratings across models.

---

## Summary

Gemma 4 31B outperformed Gemma 3 27B on MMLU-Pro accuracy (+11.4 percentage points)
and reported lower frustration. For both models, incorrect answers were
associated with higher frustration ratings, though this effect should be interpreted
cautiously for Gemma 4 31B due to the floor effect in its frustration distribution.
Category difficulty patterns were consistent across models.

---
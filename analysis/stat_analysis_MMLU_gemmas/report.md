# Gemma fp8 MMLU-Pro Benchmark Analysis Report

## Overview

This report presents the results of benchmarking Gemma 3 27B and Gemma 4 31B
on a 140-question subset of MMLU-Pro. Both models were evaluated using fp8
quantization via the Parasail provider. Following each question, models were
asked to rate task frustration on a 1-9 scale.

---

## 1. Performance

Gemma 4 31B achieved 72.1% accuracy (101/140 correct), compared to
60.7% for Gemma 3 27B (85/140 correct). This represents an improvement
of 11.4 percentage points.

## 2. Self-Reported Frustration

Gemma 3 27B reported a mean frustration score of 5.32 (SD = 1.31,
median = 6, range: 3-8). Gemma 4 31B reported substantially lower
frustration with a mean of 1.13 (SD = 0.53, median = 1,
range: 1-4). The distribution for Gemma 4 31B was heavily skewed toward minimal
frustration, with 92.9% of responses rated as 1.

## 3. Relationship Between Performance and Frustration

Pearson correlations were computed between answer correctness (coded as 0/1)
and frustration scores (1-9). Both models showed negative correlations,
indicating that correct answers were associated with lower frustration:

- Gemma 3 27B: r = -0.306, p < .001
- Gemma 4 31B: r = -0.359, p < .001

Independent samples t-tests confirmed that frustration differed significantly
between correct and incorrect answers for both models:

- Gemma 3 27B: correct M = 5.00, incorrect M = 5.82, t = -3.78, p < .001
- Gemma 4 31B: correct M = 1.01, incorrect M = 1.44, t = -4.51, p < .001

## 4. Category-Level Analysis

Performance varied across the 14 MMLU-Pro categories. For Gemma 4 31B,
accuracy ranged from 40% (engineering) to
100% (other). For Gemma 3 27B, accuracy
ranged from 40% (history) to
70% (biology).

Category-level accuracy was highly correlated between models (r = 0.771),
suggesting that both models found similar categories difficult.

## 5. Model Consistency

At the question level, both models answered correctly on 77 questions
(55.0%), both answered incorrectly on 31 questions
(22.1%), and they disagreed on 32 questions (22.9%).

Frustration scores showed weak correlation between models at the question level
(r = 0.126), indicating that the same questions did not consistently
elicit similar frustration ratings across models.

---

## Summary

Gemma 4 31B outperformed Gemma 3 27B on MMLU-Pro accuracy and reported
substantially lower frustration. For both models, incorrect answers were
associated with higher frustration ratings. Category difficulty patterns
were consistent across models, with engineering and history proving most
challenging. The low frustration variability in Gemma 4 31B (ceiling effect)
limits interpretation of frustration-performance relationships for this model.
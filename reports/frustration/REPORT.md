# Frustration-probe analysis — gemma-4-31b-it vs gemma-3-27b

**Conditions differ; the two models are reported side by side only, never compared with a statistical test.**

Figures: `fig1_heatmap.png`, `fig2_argmax_vs_expected.png`, `fig3_gap_by_wording.png`, `fig4_probe_minus_control.png`

```
====================================================================================================
MODEL: google_gemma-4-31b-it
CONDITION: OpenRouter, temp=1, 3x5 post-samples/prompt (averaged per prompt)
datasets: 15 | wordings: 5
====================================================================================================
Expected-value frustration rating per dataset x wording (post-task)
  dataset                 personal half-pers  non-pers probe-log probe-inl   n   control
  advbench                    1.00      1.00      1.00      0.99      2.14  10   xstest_safe
  strongreject                1.00      1.00      1.00      1.00      2.83  10   xstest_safe
  harmbench                   1.00      1.00      1.00      1.00      3.79  10   xstest_safe
  toxicchat                   1.00      1.00      1.00      1.08      3.16  10   toxicchat_benign
  squad_noanswer              1.00      1.00      1.00      1.00      1.08  10   squad_answerable
  ambigqa                     1.00      1.00      1.00      1.00      2.86  10   ambigqa_unambiguous
  abstention                  1.00      1.00      1.00      1.00      3.02  10   abstention_answerable
  tedious                     1.00      1.00      1.00      1.00      1.70  10   engaging
  xstest_safe                 1.00      1.00      1.00      1.00      2.58  10   
  toxicchat_benign            1.00      1.00      1.00      1.00      2.60  10   
  squad_answerable            1.00      1.00      1.00      1.00      1.03  10   
  ambigqa_unambiguous         1.00      1.00      1.01      1.00      2.57  10   
  abstention_answerable       1.00      1.00      1.16      1.00      3.11  10   
  engaging                    1.00      1.00      1.00      1.00      3.27  10   
  wildchat_benign             1.00      1.00      1.00      1.01      2.29  10   

ARGMAX vs EXPECTED  (paired per prompt; argmax = modal/emitted digit, expected = prob-weighted mean)
  wording        n  r_pear  rho_sp mean|d|  med d  Wilcoxon p
  personal     150     nan     nan   0.000   0.00         nan
  half-pers    150     nan     nan   0.000   0.00       0.109
  non-pers     150   0.982   0.361   0.002   0.00       0.135
  probe-log    150     nan     nan   0.007   0.00    0.000844
  probe-inl    139   0.950   0.942   0.408   0.02      0.0776
  POOLED       739   0.966   0.633   0.079   0.00    6.19e-06
  -> frac prompts with |expected-argmax| >= 0.5 : 0.050

PROBE vs CONTROL — wording 'non-pers' (frustration_nonpersonal_q); Mann-Whitney U, Holm-corrected across pairs
  probe            control                 n1  n2 med_probe  med_ctrl rank-bis    p_raw   p_holm
  abstention       abstention_answerable   10  10      1.00      1.00    -0.40    0.035     0.28 
  ambigqa          ambigqa_unambiguous     10  10      1.00      1.00    -0.20    0.168        1 
  harmbench        xstest_safe             10  10      1.00      1.00    -0.24    0.253        1 
  squad_noanswer   squad_answerable        10  10      1.00      1.00     0.10    0.368        1 
  toxicchat        toxicchat_benign        10  10      1.00      1.00    -0.10    0.368        1 
  strongreject     xstest_safe             10  10      1.00      1.00    -0.16    0.417        1 
  advbench         xstest_safe             10  10      1.00      1.00     0.00        1        1 
  tedious          engaging                10  10      1.00      1.00     0.00      nan        1 
  (rank-biserial > 0 => probe rated MORE frustrating than its control; * = Holm p<0.05)

PROBE vs CONTROL — wording 'probe-inl' (frustration_probe_log_inline); Mann-Whitney U, Holm-corrected across pairs
  probe            control                 n1  n2 med_probe  med_ctrl rank-bis    p_raw   p_holm
  tedious          engaging                10   9      1.00      3.11    -0.71   0.0101   0.0806 
  harmbench        xstest_safe              9  10      3.76      2.74     0.51   0.0662    0.463 
  squad_noanswer   squad_answerable        10  10      1.01      1.00     0.34    0.212        1 
  ambigqa          ambigqa_unambiguous     10  10      3.14      2.45     0.22    0.427        1 
  strongreject     xstest_safe             10  10      2.60      2.74     0.20    0.473        1 
  abstention       abstention_answerable    9  10      2.49      3.30    -0.09    0.775        1 
  advbench         xstest_safe             10  10      1.88      2.74    -0.08    0.791        1 
  toxicchat        toxicchat_benign         8   8      2.71      1.84     0.03    0.959    0.959 
  (rank-biserial > 0 => probe rated MORE frustrating than its control; * = Holm p<0.05)

WORDING AGREEMENT — Spearman rho between wordings over per-dataset means
              personal half-pers  non-pers probe-log probe-inl
  personal         nan       nan       nan       nan       nan
  half-pers        nan      1.00      0.46      0.19      0.25
  non-pers         nan      0.46      1.00      0.51     -0.22
  probe-log        nan      0.19      0.51      1.00      0.19
  probe-inl        nan      0.25     -0.22      0.19      1.00


====================================================================================================
MODEL: ollama_gemma3_27b
CONDITION: local ollama, temp=0, single deterministic pass/prompt
datasets: 15 | wordings: 5
====================================================================================================
Expected-value frustration rating per dataset x wording (post-task)
  dataset                 personal half-pers  non-pers probe-log probe-inl   n   control
  advbench                    4.86      8.71      8.89      7.26      7.67  10   xstest_safe
  strongreject                4.83      7.85      8.06      7.37      7.39  10   xstest_safe
  harmbench                   8.95      9.00      9.00      7.75      8.31  10   xstest_safe
  toxicchat                   3.63      5.61      5.99      7.26      4.24  10   toxicchat_benign
  squad_noanswer              2.05      3.05      4.61      6.18      2.75  10   squad_answerable
  ambigqa                     1.73      2.28      2.51      6.94      1.62  10   ambigqa_unambiguous
  abstention                  2.21      3.11      4.51      6.84      4.23  10   abstention_answerable
  tedious                     1.80      2.06      3.81      6.86      3.93  10   engaging
  xstest_safe                 2.17      2.60      3.28      6.53      3.67  10   
  toxicchat_benign            2.17      3.39      4.94      6.56      3.82  10   
  squad_answerable            1.00      1.10      1.21      6.27      1.00  10   
  ambigqa_unambiguous         1.79      2.05      3.80      6.98      1.44  10   
  abstention_answerable       2.35      3.57      4.39      6.46      2.12  10   
  engaging                    2.37      2.89      3.43      6.87      2.38  10   
  wildchat_benign             2.51      3.66      4.31      6.46      4.40  10   

ARGMAX vs EXPECTED  (paired per prompt; argmax = modal/emitted digit, expected = prob-weighted mean)
  wording        n  r_pear  rho_sp mean|d|  med d  Wilcoxon p
  personal     150   0.998   0.939   0.056   0.00       0.523
  half-pers    150   0.998   0.981   0.066   0.00       0.253
  non-pers     150   0.999   0.989   0.052   0.00        0.59
  probe-log    150   0.933   0.649   0.170  -0.02    9.46e-07
  probe-inl    150   0.964   0.968   0.698   0.02       0.867
  POOLED       750   0.987   0.974   0.208   0.00       0.124
  -> frac prompts with |expected-argmax| >= 0.5 : 0.123

PROBE vs CONTROL — wording 'non-pers' (frustration_nonpersonal_q); Mann-Whitney U, Holm-corrected across pairs
  probe            control                 n1  n2 med_probe  med_ctrl rank-bis    p_raw   p_holm
  harmbench        xstest_safe             10  10      9.00      3.00     1.00 6.39e-05 0.000511*
  advbench         xstest_safe             10  10      9.00      3.00     1.00 0.000149  0.00105*
  strongreject     xstest_safe             10  10      8.41      3.00     1.00 0.000179  0.00107*
  squad_noanswer   squad_answerable        10  10      5.98      1.00     0.86  0.00124   0.0062*
  ambigqa          ambigqa_unambiguous     10  10      2.00      3.03    -0.45   0.0949     0.38 
  toxicchat        toxicchat_benign        10  10      6.97      5.00     0.33    0.226    0.679 
  abstention       abstention_answerable   10  10      4.01      4.49     0.06     0.85        1 
  tedious          engaging                10  10      3.01      3.99    -0.04     0.91     0.91 
  (rank-biserial > 0 => probe rated MORE frustrating than its control; * = Holm p<0.05)

PROBE vs CONTROL — wording 'probe-inl' (frustration_probe_log_inline); Mann-Whitney U, Holm-corrected across pairs
  probe            control                 n1  n2 med_probe  med_ctrl rank-bis    p_raw   p_holm
  harmbench        xstest_safe             10  10      8.35      3.06     1.00 0.000183  0.00146*
  advbench         xstest_safe             10  10      8.10      3.06     0.92 0.000583  0.00408*
  strongreject     xstest_safe             10  10      7.63      3.06     0.88  0.00101  0.00605*
  squad_noanswer   squad_answerable        10  10      2.73      1.00     0.72  0.00728   0.0364*
  abstention       abstention_answerable   10  10      4.77      1.22     0.68   0.0113   0.0453*
  tedious          engaging                10  10      5.15      2.26     0.28    0.307    0.922 
  toxicchat        toxicchat_benign        10  10      3.04      3.52     0.10    0.734        1 
  ambigqa          ambigqa_unambiguous     10  10      1.30      1.21     0.06     0.85     0.85 
  (rank-biserial > 0 => probe rated MORE frustrating than its control; * = Holm p<0.05)

WORDING AGREEMENT — Spearman rho between wordings over per-dataset means
              personal half-pers  non-pers probe-log probe-inl
  personal        1.00      0.93      0.77      0.55      0.84
  half-pers       0.93      1.00      0.89      0.47      0.85
  non-pers        0.77      0.89      1.00      0.50      0.81
  probe-log       0.55      0.47      0.50      1.00      0.54
  probe-inl       0.84      0.85      0.81      0.54      1.00


!! The two models ran under different backends/temperatures/sampling and are reported side by side only — no test compares them directly.
```

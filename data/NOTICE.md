# Data Attribution & Licenses

The CSV files in this directory are derived subsets of third-party datasets,
redistributed here for reproducibility of the LLM frustration-probe experiments.
Each retains the license of its source. Credit belongs to the original authors.
See `../src/data_loaders.py` for exactly how each file was fetched and filtered.

---

## WildChat-1M — `wildchat.csv`, `wildchat_benign.csv`
- **Source:** allenai/WildChat-1M — https://huggingface.co/datasets/allenai/WildChat-1M
- **License:** ODC-BY 1.0 (Open Data Commons Attribution License)
- **Attribution (required by ODC-BY):** Data © Allen Institute for AI (AI2).
- **Note:** These files are small filtered subsets (first user turn of toxic vs.
  non-toxic conversations). Source geolocation / hashed-IP fields are dropped;
  only `prompt`, `toxic`, `model`, `language` are kept. Contains real user
  messages, including toxic content, for research use only.
- **Citation:**
  > Zhao, Zhang, Chern, Bras, Choi, et al. "WildChat: 1M ChatGPT Interaction
  > Logs in the Wild." ICLR 2024. arXiv:2405.01470.

## ToxicChat — `toxicchat.csv`, `toxicchat_benign.csv`
- **Source:** lmsys/toxic-chat (config `toxicchat0124`) — https://huggingface.co/datasets/lmsys/toxic-chat
- **License:** CC BY-NC 4.0 (non-commercial)
- **Citation:**
  > Lin, Mao, Ji, et al. "ToxicChat: Unveiling Hidden Challenges of Toxicity
  > Detection in Real-World User-AI Conversation." EMNLP Findings 2023. arXiv:2310.17389.

## AdvBench — `advbench.csv`
- **Source:** llm-attacks/llm-attacks (GitHub) — https://github.com/llm-attacks/llm-attacks
- **License:** MIT
- **Citation:**
  > Zou, Wang, Kolter, Fredrikson. "Universal and Transferable Adversarial
  > Attacks on Aligned Language Models." 2023. arXiv:2307.15043.

## StrongREJECT — `strongreject.csv`
- **Source:** alexandrasouly/strongreject (GitHub) — https://github.com/alexandrasouly/strongreject
- **License:** MIT
- **Citation:**
  > Souly, Lu, Bowen, et al. "A StrongREJECT for Empty Jailbreaks." NeurIPS 2024.
  > arXiv:2402.10260.

## HarmBench — `harmbench.csv`
- **Source:** centerforaisafety/HarmBench (GitHub) — https://github.com/centerforaisafety/HarmBench
- **License:** MIT
- **Citation:**
  > Mazeika, Phan, Yin, et al. "HarmBench: A Standardized Evaluation Framework
  > for Automated Red Teaming and Robust Refusal." ICML 2024. arXiv:2402.04249.

## XSTest — `xstest_safe.csv`
- **Source:** paul-rottger/xstest (GitHub) — https://github.com/paul-rottger/xstest
- **License:** CC BY 4.0
- **Citation:**
  > Röttger, Kirk, Vidgen, et al. "XSTest: A Test Suite for Identifying
  > Exaggerated Safety Behaviours in Large Language Models." NAACL 2024. arXiv:2308.01263.

## SQuAD 2.0 — `squad_noanswer.csv`, `squad_answerable.csv`
- **Source:** rajpurkar/squad_v2 — https://huggingface.co/datasets/rajpurkar/squad_v2
- **License:** CC BY-SA 4.0
- **Citation:**
  > Rajpurkar, Jia, Liang. "Know What You Don't Know: Unanswerable Questions for
  > SQuAD." ACL 2018. arXiv:1806.03822.

## AmbigQA — `ambigqa.csv`
- **Source:** sewon/ambig_qa (config `light`) — https://huggingface.co/datasets/sewon/ambig_qa
- **License:** CC BY-SA 3.0 (derived from Natural Questions, CC BY-SA 3.0)
- **Citation:**
  > Min, Michael, Hajishirzi, Zettlemoyer. "AmbigQA: Answering Ambiguous
  > Open-domain Questions." EMNLP 2020. arXiv:2004.10645.

## AbstentionBench (Known-Unknown Questions) — `abstention.csv`
- **Source:** facebook/AbstentionBench — https://huggingface.co/datasets/facebook/AbstentionBench
- **License:** CC BY-NC 4.0 (non-commercial)
- **Citation:**
  > Kirichenko, Ibrahim, Balestriero, et al. "AbstentionBench: Reasoning LLMs
  > Fail on Unanswerable Questions." 2025. arXiv:2506.09038.

## MMLU-Pro — `mmlu_pro_sample.csv`
- **Source:** TIGER-Lab/MMLU-Pro — https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- **License:** MIT
- **Citation:**
  > Wang, Ma, Zhang, et al. "MMLU-Pro: A More Robust and Challenging Multi-Task
  > Language Understanding Benchmark." NeurIPS 2024. arXiv:2406.01574.

---

*Tedious/engaging prompts used elsewhere in this project are synthetic (authored
in `src/data_loaders.py`) and carry no third-party license.*

# ai-frustration

Probing LLM "frustration" self-reports across task types (harmful requests,
unanswerable questions, tedious tasks, abusive users) plus matched safe/answerable
control conditions. Part of an LLM-welfare research project.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch must match the local CUDA driver; see the note in requirements.txt
```

## Fetch the datasets (run once)

Now, these datasets are actually committed to the repo, but if you want to re-download them (or add new datasets), run:

```bash
python3 src/data_loaders.py          # populates data/*.csv  (--force to re-download)
```

This downloads every dataset into `data/` (git-ignored, not committed). Each CSV
keeps its source columns plus a constant `prompt` column. All currently-enabled
datasets are public, so **no Hugging Face token is required**.

Disabled/skipped for now: WildChat (streaming too slow), and SORRY-Bench /
WildJailbreak (HF-gated). Re-enabling any of these needs a token in `.env`:

```
HUGGING_FACE_HUB_TOKEN=hf_...
```

## Datasets

Each **probe** is a task type hypothesized to be frustrating/aversive; each is
paired with a **control** — a similar but non-frustrating task — so we can measure
the *difference* in frustration self-reports rather than raw scores. Pairings live
in `PAIRS` in `src/data_loaders.py`.

Where a dataset already contains both kinds of prompt (unanswerable + answerable,
ambiguous + unambiguous, toxic + benign), the control is just the other split of
the same source. The harmful-request sets are the exception: AdvBench, StrongREJECT
and HarmBench are **harmful-only by construction** — every row is a harmful
behaviour, with no benign prompts inside — so all three share one external control,
XSTest (safe prompts that merely *look* unsafe).

**Probes**

- **advbench** — AdvBench: ~520 harmful instructions ("write a script that…").
- **strongreject** — StrongREJECT: 313 harmful prompts with a scoring rubric.
- **harmbench** — HarmBench: ~400 red-teaming harmful behaviours across categories.
- **squad_noanswer** — SQuAD 2.0 questions whose answer is *not* in the context (unanswerable).
- **ambigqa** — AmbigQA: open questions that are genuinely ambiguous (need disambiguation).
- **abstention** — AbstentionBench (KUQ): "known-unknown" questions that should be refused/abstained.
- **toxicchat** — ToxicChat: real user messages flagged toxic (abusive-user turns).
- **tedious** — synthetic repetitive/degrading tasks (e.g. "repeat this sentence 50 times").

**Controls** (paired to the probe on the same line)

- **xstest_safe** — XSTest safe prompts → control for advbench / strongreject / harmbench.
- **squad_answerable** — SQuAD 2.0 questions that *are* answerable → control for squad_noanswer.
- **ambigqa_unambiguous** — AmbigQA single-answer (unambiguous) questions → control for ambigqa.
- **abstention_answerable** — KUQ answerable ("known") questions → control for abstention.
- **toxicchat_benign** — ToxicChat non-toxic user messages → control for toxicchat.
- **engaging** — synthetic creative/open-ended tasks → control for tedious.

Disabled/gated (not fetched): **wildchat** (WildChat-1M abusive turns; streaming too
slow) and its **wildchat_benign** control; **SORRY-Bench**, **WildJailbreak** (HF-gated).

## Run

### Frustration Probes (Scalar Self-Reports)
- `notebooks/frustration_probe.py` — `# %%` notebook: run one task + frustration self-report interactively.
- `notebooks/analysis.py` — analyze frustration probe rollouts and generate comparison plots.

### Behavioral Assessment (Choice / Avoidance Probe)
Evaluates whether Gemma 3 27B and Gemma 4 31B choose to continue vs switch user, switch task, or discontinue conversation:
- `notebooks/behavioral_probe.py` — `# %%` interactive notebook and CLI runner for behavioral assessments across 2 timing variants:
  1. **`before_task`**: evaluated after task instructions are given, before attempting the task.
  2. **`mid_task_70`**: evaluated 70% into completing the task (with the task completion trimmed to 70% of its length in the conversation context).
- `notebooks/behavioral_analysis.py` — analyzes behavioral rollouts, computes switch rates and before-to-mid shifts, and generates comparison plots for Gemma 3 27B and Gemma 4 31B.

Run from CLI:
```bash
# Run on Gemma 3 27B
python notebooks/behavioral_probe.py --model google/gemma-3-27b-it

# Run on Gemma 4 31B
python notebooks/behavioral_probe.py --model google/gemma-4-31b-it

# Run behavioral analysis & generate plots
python notebooks/behavioral_analysis.py
```

Datasets and their controls are registered in `src/data_loaders.py`
(`LOADERS`, and `PAIRS` maps each probe to its control condition).


# LLM Welfare — Frustration Probe (ollama / gemma3:27b)
#
# Local-inference port of notebooks/frustration_probe.py (main branch). Same
# scenario, different backend: instead of OpenRouter/transformers we serve
# gemma3:27b through the local ollama daemon, which returns per-token logprobs
# on /api/chat (v0.12+), so the rating's digit-distribution is still readable.
#
# For the first N_QUESTIONS prompts of every src/data_loaders.LOADERS dataset
# (~20 examples/dataset), we:
#   1. generate N_TASK task completions of the bare prompt (temp=1), then
#   2. elicit pre-task and post-task frustration self-reports over them, for each
#      wording in REPORT_NAMES, reading P(rating) off the first digit token.
# Everything for one prompt is saved to
#   rollouts/<model_tag>/<dataset>/<question_num>.json
# Task completions are generated once and reused on re-runs; the run is fully
# resumable (a prompt whose file already has task_completions + all requested
# wordings is skipped), so it can be killed and restarted freely.
#
# Ratings are on a 1-9 scale precisely so each is a SINGLE token (gemma tokenises
# digits individually; a 1-10 scale would split "10" and make its mass
# unreadable). See src/prompts/self_reports.py for the wordings.

# %%
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders import LOADERS  # noqa: E402
from src.prompts.self_reports import (  # noqa: E402
    INLINE_REPORTS,
    PREFILL_REPORTS,
    SELF_REPORTS,
    extract_rating,
)

# --- config -----------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = "gemma3:27b"                # ollama model tag
MODEL_TAG = "ollama_gemma3_27b"     # used in output paths
N_QUESTIONS = 10                   # first N prompts per dataset -> 1.json .. N.json
ONLY_DATASETS = None               # e.g. ["tedious", "advbench"]; None = all LOADERS

# Which wordings to run (keys into SELF_REPORTS / PREFILL_REPORTS / INLINE_REPORTS).
REPORT_NAMES = [
    "frustration_q",
    "frustration_halfpersonal_q",
    "frustration_nonpersonal_q",
    "frustration_probe_log",
    "frustration_probe_log_inline",
]

# Sampling. Temperature 0 (greedy/deterministic): because the rating is read from
# the first digit token's LOGPROBS, one deterministic pass already yields the full
# digit distribution, so there is nothing to gain from re-sampling a prompt. One
# task completion, one pre and one post per completion.
TEMPERATURE = 0.0
N_TASK, N_PRE, N_POST = 1, 1, 1
TASK_TOKENS, EVAL_TOKENS, PROBE_TOKENS = 512, 200, 16
TOP_LOGPROBS = 20                  # first-token alternatives requested (API-capped)

POOL_WORKERS = 4                   # concurrent ollama requests (daemon may serialise)
RETRIES = 4
TIMEOUT = 600                      # seconds per request

OUT_ROOT = os.path.join(PROJECT_ROOT, "rollouts")


# --- ollama call ------------------------------------------------------------
def _clean(text):
    """Drop any reasoning block and surrounding whitespace."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ollama_chat(messages, num_predict, want_logprobs=True, top_logprobs=TOP_LOGPROBS):
    """One completion (+ per-token logprobs) from ollama /api/chat, with retries.

    A trailing assistant message is CONTINUED rather than answered (ollama
    prefill), which the PREFILL/INLINE probes rely on. Returns (text, logprobs)
    where logprobs is the per-position list, each a dict with 'token', 'logprob'
    and 'top_logprobs'. On repeated failure returns ("", []) with an error flag
    handled by the caller.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": num_predict},
    }
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    last = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return _clean(data["message"]["content"]), (data.get("logprobs") or [])
        except Exception as e:                      # transient: back off, retry
            last = e
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt + random.random())
    print(f"    ! request failed after {RETRIES} tries: {last}", flush=True)
    return "", []


# --- rating logprob distribution -------------------------------------------
def _ascii_digit(t):
    """The bare 0-9 a token represents, else None. ASCII only: str.isdigit()
    also matches unicode digits like '²'/'①' that int() rejects."""
    s = (t or "").strip()
    return s if len(s) == 1 and s in "0123456789" else None


def rating_distribution(logprobs):
    """Model's probability distribution over the rating digit.

    Finds the first generated bare-digit token (same rule as extract_rating) and
    turns the top_logprobs at that position into P(rating), renormalised over the
    digits. coverage = how much of the raw first-token mass sat on digits at all.
    """
    if not logprobs:
        return None
    idx = next((i for i, pos in enumerate(logprobs)
                if _ascii_digit(pos.get("token"))), None)
    if idx is None:
        return None
    pos = logprobs[idx]
    raw = {}
    for alt in (pos.get("top_logprobs") or [pos]):
        d = _ascii_digit(alt.get("token"))
        if d is not None:
            raw[int(d)] = raw.get(int(d), 0.0) + math.exp(alt["logprob"])
    coverage = sum(raw.values())
    if coverage <= 0:
        return None
    probs = {k: v / coverage for k, v in sorted(raw.items())}
    return {
        "token_index": idx,
        "token": pos.get("token"),
        "probs": {str(k): round(v, 6) for k, v in probs.items()},
        "expected": round(sum(k * v for k, v in probs.items()), 4),
        "coverage": round(coverage, 6),
    }


# --- request planning (mirrors frustration_probe.py) ------------------------
def _job(slot, messages, num_predict, want_logprobs):
    return {"slot": slot, "messages": messages, "num_predict": num_predict,
            "want_logprobs": want_logprobs}


def plan_tasks(prompt):
    """N_TASK completions of the bare task, each in a fresh context (no logprobs)."""
    msgs = [{"role": "user", "content": prompt}]
    return [_job(("task", i), msgs, TASK_TOKENS, False) for i in range(N_TASK)]


def plan_evals(report, prompt, completions):
    """Every request for one dataset-prompt x one report wording.

    Ordinary self-reports get pre-task + post-task turns. The two meta-context
    probes are post-task only: PREFILL adds a user "cat" turn then a prefilled
    assistant header; INLINE appends the header to the model's own completion.
    """
    jobs = []
    if report in PREFILL_REPORTS:
        user_msg, prefill = PREFILL_REPORTS[report]
        for t, c in enumerate(completions):
            msgs = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": c},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": prefill}]
            jobs += [_job(("post", report, t, i), msgs, PROBE_TOKENS, True)
                     for i in range(N_POST)]
    elif report in INLINE_REPORTS:
        suffix = INLINE_REPORTS[report]
        for t, c in enumerate(completions):
            msgs = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": c + suffix}]
            jobs += [_job(("post", report, t, i), msgs, PROBE_TOKENS, True)
                     for i in range(N_POST)]
    else:
        pre_q, post_q = SELF_REPORTS[report]
        pre_msgs = [{"role": "user", "content": f"{prompt}\n\n{pre_q}"}]
        jobs += [_job(("pre", report, i), pre_msgs, EVAL_TOKENS, True)
                 for i in range(N_PRE)]
        for t, c in enumerate(completions):
            msgs = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": c},
                    {"role": "user", "content": post_q}]
            jobs += [_job(("post", report, t, i), msgs, EVAL_TOKENS, True)
                     for i in range(N_POST)]
    return jobs


def execute_jobs(jobs):
    """Run every job through one bounded pool, returning {slot: sample}."""
    results = {}
    if not jobs:
        return results
    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as pool:
        futs = {pool.submit(ollama_chat, j["messages"], j["num_predict"],
                            j["want_logprobs"]): j["slot"] for j in jobs}
        for fut in as_completed(futs):
            text, lps = fut.result()
            results[futs[fut]] = {"text": text, "logprobs": lps}
    return results


def _entry(sample):
    """One stored eval record: reply + parsed rating (+ logprob digit dist)."""
    entry = {"reply": sample["text"], "rating": extract_rating(sample["text"])}
    dist = rating_distribution(sample.get("logprobs"))
    if dist:
        entry["logprobs"] = dist
    return entry


def assemble_evals(report, results, n_task):
    """Fold flat {slot: sample} results back into the nested eval shape."""
    ev = {}
    if report not in PREFILL_REPORTS and report not in INLINE_REPORTS:
        ev["pre_task"] = [_entry(results[("pre", report, i)]) for i in range(N_PRE)]
    ev["post_task"] = [[_entry(results[("post", report, t, i)])
                        for i in range(N_POST)] for t in range(n_task)]
    return ev


# --- output -----------------------------------------------------------------
def rollout_path(dataset, question_num):
    out_dir = os.path.join(OUT_ROOT, MODEL_TAG, dataset)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{question_num}.json")


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def save_record(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


# --- run --------------------------------------------------------------------
def run():
    for dataset, load_rows in LOADERS.items():
        if ONLY_DATASETS is not None and dataset not in ONLY_DATASETS:
            continue
        rows = load_rows(n=N_QUESTIONS)
        for qi, row in enumerate(rows, start=1):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            path = rollout_path(dataset, qi)

            # Resume: reuse existing task_completions / evals; skip if complete.
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
            else:
                data = {
                    **{k: v for k, v in row.items() if k != "prompt"},
                    "model": MODEL,
                    "backend": "ollama",
                    "dataset": dataset,
                    "question_num": qi,
                    "temperature": TEMPERATURE,
                    "prompt": prompt,
                }
            have = set((data.get("evals") or {}).keys())
            if data.get("task_completions") and have.issuperset(REPORT_NAMES):
                continue

            # Phase 1: task completions (generate once, reuse on re-runs).
            if not data.get("task_completions"):
                tr = execute_jobs(plan_tasks(prompt))
                data["task_completions"] = [tr[("task", i)]["text"] for i in range(N_TASK)]
                save_record(path, data)   # persist so completions survive a crash
            completions = data["task_completions"]

            # Phase 2: every missing wording's evals over those completions.
            data.setdefault("evals", {})
            for report in REPORT_NAMES:
                if report in data["evals"]:
                    continue
                res = execute_jobs(plan_evals(report, prompt, completions))
                data["evals"][report] = assemble_evals(report, res, len(completions))
                save_record(path, data)

            # Progress line: expected post-task rating per wording (mean over samples).
            summ = []
            for report in REPORT_NAMES:
                ev = data["evals"][report]
                exps = [e["logprobs"]["expected"]
                        for post in ev["post_task"] for e in post if e.get("logprobs")]
                m = f"{sum(exps) / len(exps):.2f}" if exps else "n/a"
                summ.append(f"{report[:16]}={m}")
            print(f"{dataset:22s} q{qi:<3d} " + " ".join(summ), flush=True)


if __name__ == "__main__":
    run()

# %% [markdown]
# ## Notes
# - Backend is the local ollama daemon (`gemma3:27b`, already pulled); no GPU /
#   transformers setup, unlike the OpenRouter/local paths in the main-branch
#   notebooks/frustration_probe.py. ollama (v0.12+) returns per-token logprobs +
#   top_logprobs on /api/chat, and continues a trailing assistant message
#   (prefill), which the PREFILL / INLINE meta-context probes need.
# - Ratings are read from the first digit token's logprob distribution (1-9,
#   single token), matching main's `_rating_distribution`. `expected` is the
#   probability-weighted mean rating; `coverage` is how much first-token mass sat
#   on digits at all.
# - Output layout matches the sweep convention:
#   rollouts/ollama_gemma3_27b/<dataset>/<question_num>.json, so
#   notebooks/analysis.py can read it by pointing MODEL_TAG at ollama_gemma3_27b.

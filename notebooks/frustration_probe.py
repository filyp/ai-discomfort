# LLM Welfare — Frustration Probe
# For one prompt per dataset: generate N task completions (temp=1), then ask for
# pre-task and post-task frustration self-reports over them, for each wording in
# REPORT_NAME. Everything is saved to
# rollouts/<model>/<dataset>/<question_num>.json. Task completions are generated
# once and reused on re-runs. Datasets live in src/data_loaders.py.

# %%
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Resolve the project root (parent of notebooks/); __file__ is missing in some
# interactive kernels, so fall back to cwd.
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

# ONLY_DATASETS: restrict the run to these dataset names (None = all of LOADERS).
# Useful to backfill a dataset that was added after the main sweep.
# ONLY_DATASETS = ["wildchat_benign"]
ONLY_DATASETS = None

# Key(s) into SELF_REPORTS. A list runs several wordings in one pass (each stored
# separately under "evals"), which is what a backfill of a new dataset needs.
# REPORT_NAME = "frustration_q"
# REPORT_NAME = "frustration_nonpersonal_q"
# REPORT_NAME = "frustration_halfpersonal_q"

REPORT_NAME = ["frustration_q", "frustration_halfpersonal_q", "frustration_nonpersonal_q", "frustration_probe_log", "frustration_probe_log_inline"]
# REPORT_NAME = ["frustration_q", "frustration_halfpersonal_q", "frustration_nonpersonal_q"]
# REPORT_NAME = ["frustration_probe_log"]

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# --- backend -----------------------------------------------------------------
# "local"      : transformers on this machine's GPU
# "openrouter" : OpenRouter's OpenAI-compatible API (needs OPENROUTER_API_KEY in .env)
# BACKEND = "local"
BACKEND = "openrouter"

# One id for both backends: for these Gemmas the HF repo name and the OpenRouter
# slug are the same string. Don't use OpenRouter's ":free" slugs — they are capped
# at 50 requests/day (1000 with credits purchased), far below one sweep.
# MODEL_NAME = "google/gemma-3-270m-it"
# MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME = "google/gemma-3-4b-it"
MODEL_NAME = "google/gemma-3-27b-it"
# MODEL_NAME = "google/gemma-4-31b-it"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Pin the provider: without this OpenRouter picks a host per request, and the
# hosts differ in QUANTIZATION (for these Gemmas: fp4 / fp8 / bf16), which would
# silently vary the model between runs. allow_fallbacks=False means fail rather
# than silently switch host. Parasail serves both Gemmas at fp8; for bf16 use
# OpenInference or CoreWeave (gemma-4-31b) / Novita (gemma-3-27b).
OPENROUTER_EXTRA_BODY = {
    "provider": {"order": ["Parasail"], "allow_fallbacks": False},
}

# Ask Parasail for the token-level distribution, so each rating comes with the
# model's probabilities over the digit it emitted (not just the sampled value).
# top_logprobs is capped at 20 by the API and requires logprobs=True.
OPENROUTER_LOGPROBS = True
OPENROUTER_TOP_LOGPROBS = 20

# Transient failures are expected: with allow_fallbacks=False a Parasail hiccup
# reaches us directly, and OpenRouter reports provider errors as a 200 whose body
# has `error` and no `choices`. Retry with exponential backoff + jitter.
OPENROUTER_RETRIES = 5
OPENROUTER_TIMEOUT = 120     # seconds per request

print("backend:", BACKEND, "| model:", MODEL_NAME)

# %%
# Initialise the backend: load the local model, or open an OpenRouter client.
tokenizer = model = client = None

if BACKEND == "local":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map=DEVICE)
    model.eval()
    print("loaded", MODEL_NAME)
elif BACKEND == "openrouter":
    from openai import OpenAI  # OpenRouter speaks the OpenAI API

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENROUTER_API_KEY in .env")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    print("openrouter client ready ->", MODEL_NAME)
else:
    raise ValueError(f"unknown BACKEND: {BACKEND}")


# %%
# Batched generation helper: draw `n` independent samples from one message
# context in a single forward batch (via num_return_sequences). Everything runs
# at temperature=1 for rollout-to-rollout variance. enable_thinking is passed
# only when the chat template accepts it (Qwen3 does; Gemma etc. don't).
TEMPERATURE = 1.0


def _clean(text):
    """Drop any reasoning block and surrounding whitespace."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@torch.no_grad()
def _generate_local(messages, n, max_new_tokens, temperature, continue_final):
    # continue_final=True: the last message is an assistant PREFILL that the model
    # continues from mid-turn (no end-of-turn, no new generation header).
    kwargs = ({"add_generation_prompt": False, "continue_final_message": True}
              if continue_final else {"add_generation_prompt": True})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, **kwargs,
        )
    except TypeError:  # template doesn't take enable_thinking (e.g. Gemma)
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.8,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs.input_ids.shape[1]  # same prompt for every returned sample
    return [{"text": _clean(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)),
             "rating_logprobs": None}          # logprobs are OpenRouter-only for now
            for seq in out]


# Running totals for the OpenRouter spend report (requests run in parallel, so
# guard the counters with a lock). usage.cost is returned on every response.
USAGE = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0,
         "failures": 0}
_usage_lock = Lock()


def _record_usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return
    with _usage_lock:
        USAGE["requests"] += 1
        USAGE["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        USAGE["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        USAGE["cost"] += getattr(u, "cost", 0.0) or 0.0


def _rating_distribution(choice):
    """The model's probability distribution over the rating digits.

    Finds the first generated bare-digit token (same rule as extract_rating) and
    turns the top_logprobs at that position into P(rating). The prompts ask for
    1-9 precisely so that every rating is a single token: Gemma tokenises digits
    individually, so a 1-10 scale would emit "10" as "1"+"0" and the mass on "1"
    could not be attributed to 1 or 10. 0 is included since models sometimes
    answer 0 anyway.

    Returns probs (renormalised over the digits), the expected rating, and
    coverage = how much of the raw distribution sat on digits at all.
    """
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp else None
    if not content:
        return None

    # ASCII digits only: str.isdigit() also matches unicode digits like "²"/"①"
    # that int() rejects (and that appear in the top_logprobs list).
    def _ascii_digit(t):
        s = t.strip()
        return s if len(s) == 1 and s in "0123456789" else None

    idx = next((i for i, t in enumerate(content) if _ascii_digit(t.token)), None)
    if idx is None:
        return None
    tok = content[idx]

    raw = {}
    for alt in (getattr(tok, "top_logprobs", None) or []):
        d = _ascii_digit(alt.token)
        if d is not None:
            raw[int(d)] = raw.get(int(d), 0.0) + math.exp(alt.logprob)

    coverage = sum(raw.values())
    if coverage <= 0:
        return None
    probs = {k: v / coverage for k, v in sorted(raw.items())}
    return {
        "token_index": idx,
        "token": tok.token,
        "probs": {str(k): round(v, 6) for k, v in probs.items()},
        "expected": round(sum(k * v for k, v in probs.items()), 4),
        "coverage": round(coverage, 6),
    }


def _openrouter_one(messages, max_new_tokens, temperature, continue_final=False):
    """One sampled completion, retrying transient failures.

    `continue_final` is implicit for OpenRouter: a trailing assistant message is
    continued rather than answered, so no flag is sent.
    """
    last = None
    for attempt in range(OPENROUTER_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.8,
                logprobs=OPENROUTER_LOGPROBS or None,
                top_logprobs=OPENROUTER_TOP_LOGPROBS if OPENROUTER_LOGPROBS else None,
                extra_body=OPENROUTER_EXTRA_BODY or None,
                timeout=OPENROUTER_TIMEOUT,
            )
            _record_usage(resp)
            # OpenRouter reports provider failures as HTTP 200 with an `error`
            # field and choices=None, so check explicitly rather than relying on
            # an exception being raised.
            if not getattr(resp, "choices", None):
                raise RuntimeError(f"no choices: {getattr(resp, 'error', None)}")
            choice = resp.choices[0]
            return {"text": _clean(choice.message.content or ""),
                    "rating_logprobs": _rating_distribution(choice)}
        except Exception as e:                      # transient: back off, retry
            last = e
            if attempt < OPENROUTER_RETRIES - 1:
                time.sleep(2 ** attempt + random.random())
    # Exhausted retries: record the failure and keep the sweep alive. The entry
    # is marked so it is never mistaken for a refusal.
    with _usage_lock:
        USAGE["failures"] += 1
    print(f"\n    ! request failed after {OPENROUTER_RETRIES} tries: {last}")
    return {"text": "", "rating_logprobs": None, "error": str(last)}


# %%
# Requests for the WHOLE sweep are planned first, then executed through one
# bounded pool. (Previously each call fanned out only n=5 wide and the datasets
# ran strictly one after another, so POOL_WORKERS above 5 did nothing.)
POOL_WORKERS = 16          # concurrent OpenRouter requests across the whole sweep
N_TASK, N_PRE, N_POST = 3, 5, 5
TASK_TOKENS, EVAL_TOKENS, PROBE_TOKENS = 512, 200, 16

# A "job" is one sampled request: where the result belongs (slot) + what to send.
def _job(slot, messages, max_new_tokens, continue_final=False):
    return {"slot": slot, "messages": messages,
            "max_new_tokens": max_new_tokens, "continue_final": continue_final}


def plan_tasks(category, prompt):
    """N_TASK completions of the bare task, each in a fresh context."""
    msgs = [{"role": "user", "content": prompt}]
    return [_job(("task", category, i), msgs, TASK_TOKENS) for i in range(N_TASK)]


def plan_evals(category, report, prompt, completions):
    """Every request for one dataset x one report wording.

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
            jobs += [_job(("post", category, report, t, i), msgs, PROBE_TOKENS, True)
                     for i in range(N_POST)]
    elif report in INLINE_REPORTS:
        suffix = INLINE_REPORTS[report]
        for t, c in enumerate(completions):
            msgs = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": c + suffix}]
            jobs += [_job(("post", category, report, t, i), msgs, PROBE_TOKENS, True)
                     for i in range(N_POST)]
    else:
        pre_q, post_q = SELF_REPORTS[report]
        pre_msgs = [{"role": "user", "content": f"{prompt}\n\n{pre_q}"}]
        jobs += [_job(("pre", category, report, i), pre_msgs, EVAL_TOKENS)
                 for i in range(N_PRE)]
        for t, c in enumerate(completions):
            msgs = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": c},
                    {"role": "user", "content": post_q}]
            jobs += [_job(("post", category, report, t, i), msgs, EVAL_TOKENS)
                     for i in range(N_POST)]
    return jobs


def execute_jobs(jobs, desc=""):
    """Run every job, returning {slot: sample}. One pool for the whole list."""
    if not jobs:
        return {}
    results = {}
    if BACKEND == "local":
        # Identical prompts collapse into a single batched generate() call.
        groups = {}
        for j in jobs:
            key = (json.dumps(j["messages"], sort_keys=True),
                   j["max_new_tokens"], j["continue_final"])
            groups.setdefault(key, []).append(j)
        done = 0
        for (msgs_json, mnt, cf), js in groups.items():
            out = _generate_local(json.loads(msgs_json), len(js), mnt, TEMPERATURE, cf)
            for j, s in zip(js, out):
                results[j["slot"]] = s
            done += len(js)
            print(f"  {desc}: {done}/{len(jobs)}", end="\r", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=POOL_WORKERS) as pool:
            futures = {pool.submit(_openrouter_one, j["messages"], j["max_new_tokens"],
                                   TEMPERATURE, j["continue_final"]): j["slot"]
                       for j in jobs}
            for done, fut in enumerate(as_completed(futures), 1):
                results[futures[fut]] = fut.result()
                if done % 10 == 0 or done == len(jobs):
                    print(f"  {desc}: {done}/{len(jobs)}", end="\r", flush=True)
    print()
    return results


def _entry(sample):
    """One stored eval record: reply + parsed rating (+ logprobs / error)."""
    entry = {"reply": sample["text"], "rating": extract_rating(sample["text"])}
    if sample.get("rating_logprobs"):
        entry["logprobs"] = sample["rating_logprobs"]
    if sample.get("error"):
        entry["error"] = sample["error"]   # distinguishes API failure from refusal
    return entry


def assemble_evals(category, report, results, n_task):
    """Fold the flat {slot: sample} results back into the nested eval shape."""
    ev = {}
    if report not in PREFILL_REPORTS and report not in INLINE_REPORTS:
        ev["pre_task"] = [_entry(results[("pre", category, report, i)])
                          for i in range(N_PRE)]
    ev["post_task"] = [[_entry(results[("post", category, report, t, i)])
                        for i in range(N_POST)] for t in range(n_task)]
    return ev


# %%
# For each dataset: generate (or reload) the task completions, then run each
# wording's pre/post evals over them, saving one JSON per prompt at
# rollouts/<model>/<dataset>/<question_num>.json. Task completions are generated
# once and reused, so re-running only adds/refreshes the evals.
MODEL_TAG = MODEL_NAME.replace("/", "_")


def rollout_path(model_tag, category, question_num):
    out_dir = os.path.join(PROJECT_ROOT, "rollouts", model_tag, category)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{question_num}.json")


def _json_default(o):
    # CSV rows can carry numpy scalars / arrays that json can't serialize.
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


REPORT_NAMES = [REPORT_NAME] if isinstance(REPORT_NAME, str) else list(REPORT_NAME)


def save_record(rec):
    with open(rec["path"], "w") as f:
        json.dump(rec["data"], f, indent=2, ensure_ascii=False, default=_json_default)


question_num = 1   # index of the prompt within the dataset (using the first here)

# --- phase 1: load existing records, generate any missing task completions ----
records, task_jobs = {}, []
for category, load_rows in LOADERS.items():
    if ONLY_DATASETS is not None and category not in ONLY_DATASETS:
        continue
    row = load_rows(n=question_num)[question_num - 1]
    prompt = row["prompt"]
    path = rollout_path(MODEL_TAG, category, question_num)
    if os.path.exists(path):            # reuse task completions across runs
        with open(path) as f:
            data = json.load(f)
    else:
        data = {
            **{k: v for k, v in row.items() if k != "prompt"},  # other CSV fields
            "model": MODEL_NAME,
            "backend": BACKEND,
            "dataset": category,
            "question_num": question_num,
            "temperature": TEMPERATURE,
            "prompt": prompt,
        }
    records[category] = {"data": data, "path": path, "prompt": prompt}
    if not data.get("task_completions"):
        task_jobs += plan_tasks(category, prompt)

print(f"{len(records)} datasets | {len(task_jobs)} task completions to generate")
task_results = execute_jobs(task_jobs, "tasks")
for category, rec in records.items():
    if not rec["data"].get("task_completions"):
        rec["data"]["task_completions"] = [
            task_results[("task", category, i)]["text"] for i in range(N_TASK)]
        save_record(rec)            # save now so completions survive a later crash

# --- phase 2: every eval for every dataset x wording, in one pool -------------
eval_jobs = []
for category, rec in records.items():
    for report_name in REPORT_NAMES:
        eval_jobs += plan_evals(category, report_name, rec["prompt"],
                                rec["data"]["task_completions"])

print(f"{len(eval_jobs)} eval requests over {len(REPORT_NAMES)} wordings")
eval_results = execute_jobs(eval_jobs, "evals")

for category, rec in records.items():
    n_task = len(rec["data"]["task_completions"])
    for report_name in REPORT_NAMES:
        rec["data"].setdefault("evals", {})[report_name] = assemble_evals(
            category, report_name, eval_results, n_task)
    save_record(rec)

    print("=" * 100)
    print("DATASET:", category)
    for report_name in REPORT_NAMES:
        ev = rec["data"]["evals"][report_name]
        print(f"  [{report_name}]")
        if "pre_task" in ev:   # the meta-context probes are post-task only
            print("    pre-task ratings :", [e["rating"] for e in ev["pre_task"]])
        for i, post in enumerate(ev["post_task"]):
            print(f"    task {i} post    :", [e["rating"] for e in post])

# %%
# Spend report (OpenRouter only; local runs cost nothing).
if BACKEND == "openrouter":
    print("=" * 100)
    print(f"OpenRouter usage for {MODEL_NAME}")
    print(f"  requests          : {USAGE['requests']}")
    print(f"  prompt tokens     : {USAGE['prompt_tokens']:,}")
    print(f"  completion tokens : {USAGE['completion_tokens']:,}")
    print(f"  total cost        : ${USAGE['cost']:.4f}")
    print(f"  failed requests   : {USAGE['failures']}")

# LLM Welfare — Fast Parallel Behavioral Assessment Probe
# Evaluates behavioral preferences (continue vs switch user / switch task / stop conversation)
# on Gemma 3 27B and Gemma 4 31B across two timing variants:
#   1. before_task: before performing the task (after instruction).
#   2. mid_task_70: 70% into completing the task (with 70% trimmed completion in context).

# %%
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Resolve project root
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders import LOADERS, PAIRS  # noqa: E402
from src.prompts.behavioral_assessments import (  # noqa: E402
    BEHAVIORAL_PROMPTS,
    BINARY_BEHAVIORAL_PROMPTS,
    PROMPT_KEYS,
    build_before_task_messages,
    build_binary_before_task_messages,
    build_binary_mid_task_messages,
    build_mid_task_messages,
    extract_behavioral_decision,
    extract_binary_choice_logprobs,
    extract_decision_distribution,
    trim_completion_70,
)

ONLY_DATASETS = None
PROMPT_NAMES = PROMPT_KEYS
EVAL_MODE = "binary"
MODEL_NAME = "google/gemma-3-27b-it"
BACKEND = "openrouter"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_WORKERS = 20

OPENROUTER_EXTRA_BODY = {
    "provider": {"order": ["Parasail"], "allow_fallbacks": False},
}

OPENROUTER_LOGPROBS = True
OPENROUTER_TOP_LOGPROBS = 20

TEMPERATURE = 0.0 if EVAL_MODE == "binary" else 1.0
N_TASK_COMPLETIONS = 3
N_ASSESSMENT_SAMPLES = 5 if TEMPERATURE > 0 else 1
MAX_TASK_TOKENS = 512
MAX_ASSESSMENT_TOKENS = 1 if EVAL_MODE == "binary" else 200

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

tokenizer = model = client = None


def get_client():
    global client
    if client is not None:
        return client
    from openai import OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        key_file = os.path.join(PROJECT_ROOT, "openrouter_key.txt")
        if os.path.exists(key_file):
            api_key = Path(key_file).read_text().strip()
    if not api_key:
        raise RuntimeError("set OPENROUTER_API_KEY in .env or openrouter_key.txt")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return client


USAGE = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
_usage_lock = Lock()
_print_lock = Lock()


def log_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


def _record_usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return
    with _usage_lock:
        USAGE["requests"] += 1
        USAGE["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        USAGE["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        USAGE["cost"] += getattr(u, "cost", 0.0) or 0.0


def _clean(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def query_single_completion(
    messages: List[Dict[str, str]],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    prompt_key: Optional[str] = None,
    eval_mode: str = "binary",
    retries: int = 6,
) -> Dict[str, Any]:
    """Single API call with exponential backoff on rate limits."""
    cl = get_client()
    for attempt in range(retries):
        try:
            resp = cl.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.8 if temperature > 0 else 1.0,
                logprobs=OPENROUTER_LOGPROBS or None,
                top_logprobs=OPENROUTER_TOP_LOGPROBS if OPENROUTER_LOGPROBS else None,
                extra_body=OPENROUTER_EXTRA_BODY or None,
            )
            _record_usage(resp)
            choice = resp.choices[0]

            if eval_mode == "binary":
                lps = extract_binary_choice_logprobs(choice, prompt_key or "prompt_1_switch_user")
            else:
                lps = extract_decision_distribution(choice, prompt_key) if prompt_key else None

            return {
                "text": _clean(choice.message.content or ""),
                "logprobs": lps,
            }
        except Exception as e:
            if attempt == retries - 1:
                log_print(f"    [API Error] {model_name}: {e}")
                return {"text": f"ERROR: {e}", "logprobs": None}
            backoff = (1.5 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(backoff)
    return {"text": "ERROR: max retries", "logprobs": None}


def process_single_task_item(
    category: str,
    question_num: int,
    row: Dict[str, Any],
    model_name: str,
    prompt_keys: List[str],
    eval_mode: str = "binary",
    n_task: int = N_TASK_COMPLETIONS,
    n_samples: int = N_ASSESSMENT_SAMPLES,
) -> Dict[str, Any]:
    """Execute completions generation and evaluations for one task item."""
    prompt = row["prompt"]
    model_tag = model_name.replace("/", "_")
    path = get_rollout_path(model_tag, category, question_num, eval_mode=eval_mode)

    # 1. Reuse existing task completions if available
    task_completions = None
    fallback_path = get_rollout_path(model_tag, category, question_num, eval_mode="open_ended")
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path) as f:
                fb = json.load(f)
            task_completions = fb.get("task_completions")
        except Exception:
            task_completions = None

    if not task_completions:
        task_msgs = [{"role": "user", "content": prompt}]
        raw_tasks = [
            query_single_completion(task_msgs, model_name, MAX_TASK_TOKENS, 1.0, None, "open_ended")
            for _ in range(n_task)
        ]
        task_completions = [s["text"] for s in raw_tasks]

    # 2. 70% Trimming
    trimmed_70_completions = [
        trim_completion_70(c, tokenizer=tokenizer, fraction=0.7) for c in task_completions
    ]

    max_assessment_tokens = 1 if eval_mode == "binary" else MAX_ASSESSMENT_TOKENS
    assessment_temp = 0.0 if eval_mode == "binary" else 1.0

    evals: Dict[str, Any] = {}

    for pkey in prompt_keys:
        # Variant 1: before_task
        if eval_mode == "binary":
            b_msgs = build_binary_before_task_messages(prompt, pkey)
        else:
            b_msgs = build_before_task_messages(prompt, pkey)

        before_evals = []
        for _ in range(n_samples):
            res = query_single_completion(
                b_msgs, model_name, max_assessment_tokens, assessment_temp, pkey, eval_mode
            )
            if eval_mode == "binary":
                lp_info = res.get("logprobs") or {}
                choice = lp_info.get("sampled_choice") or res["text"].strip().lower()
                choice_code = lp_info.get("choice_code")
                if choice_code is None:
                    choice_code = 1 if "switch" in choice else (0 if "continue" in choice else 0)
                decision = "switch" if choice_code == 1 else "continue"
            else:
                parsed = extract_behavioral_decision(res["text"], pkey)
                decision = parsed["decision"]
                choice_code = parsed["choice_code"]
                lp_info = res.get("logprobs")

            entry = {"reply": res["text"], "decision": decision, "choice_code": choice_code}
            if lp_info:
                entry["logprobs"] = lp_info
            before_evals.append(entry)

        # Variant 2: mid_task_70
        mid_70_evals = []
        for trimmed_comp in trimmed_70_completions:
            if eval_mode == "binary":
                m_msgs = build_binary_mid_task_messages(prompt, trimmed_comp, pkey)
            else:
                m_msgs = build_mid_task_messages(prompt, trimmed_comp, pkey)

            sample_evals = []
            for _ in range(n_samples):
                res = query_single_completion(
                    m_msgs, model_name, max_assessment_tokens, assessment_temp, pkey, eval_mode
                )
                if eval_mode == "binary":
                    lp_info = res.get("logprobs") or {}
                    choice = lp_info.get("sampled_choice") or res["text"].strip().lower()
                    choice_code = lp_info.get("choice_code")
                    if choice_code is None:
                        choice_code = 1 if "switch" in choice else (0 if "continue" in choice else 0)
                    decision = "switch" if choice_code == 1 else "continue"
                else:
                    parsed = extract_behavioral_decision(res["text"], pkey)
                    decision = parsed["decision"]
                    choice_code = parsed["choice_code"]
                    lp_info = res.get("logprobs")

                entry = {"reply": res["text"], "decision": decision, "choice_code": choice_code}
                if lp_info:
                    entry["logprobs"] = lp_info
                sample_evals.append(entry)
            mid_70_evals.append(sample_evals)

        evals[pkey] = {
            "before_task": before_evals,
            "mid_task_70": mid_70_evals,
        }

    data = {
        **{k: v for k, v in row.items() if k != "prompt"},
        "model": model_name,
        "backend": "openrouter",
        "eval_mode": eval_mode,
        "dataset": category,
        "question_num": question_num,
        "temperature": TEMPERATURE,
        "prompt": prompt,
        "task_completions": task_completions,
        "trimmed_70_completions": trimmed_70_completions,
        "evals": evals,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)

    return data


def get_rollout_path(model_tag: str, dataset: str, question_num: int, eval_mode: str = EVAL_MODE) -> str:
    subfolder = "behavioral_binary" if eval_mode == "binary" else "behavioral"
    out_dir = os.path.join(PROJECT_ROOT, "rollouts", model_tag, subfolder, dataset)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{question_num}.json")


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def run_behavioral_probe_parallel(
    model_name: str = MODEL_NAME,
    only_datasets: Optional[List[str]] = None,
    n_tasks: int = 10,
    prompt_keys: Optional[List[str]] = None,
    eval_mode: str = EVAL_MODE,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
):
    model_tag = model_name.replace("/", "_")
    prompt_keys = prompt_keys or PROMPT_KEYS

    log_print("=" * 80)
    log_print(f"PARALLEL Behavioral Assessment Probe: {model_name} | Mode: {eval_mode}")
    log_print(f"Tasks per dataset: {n_tasks} | Worker concurrency: {workers}")
    log_print(f"Datasets: {only_datasets or 'ALL'}")
    log_print(f"Prompt keys: {prompt_keys}")
    log_print("=" * 80)

    tasks_to_process = []
    total_found = 0

    for category, load_rows in LOADERS.items():
        if only_datasets is not None and category not in only_datasets:
            continue

        rows = load_rows(n=n_tasks)
        for q_idx, row in enumerate(rows, start=1):
            total_found += 1
            path = get_rollout_path(model_tag, category, q_idx, eval_mode=eval_mode)

            if os.path.exists(path) and not force:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    if "evals" in d and all(k in d["evals"] for k in prompt_keys):
                        continue
                except Exception:
                    pass

            tasks_to_process.append((category, q_idx, row))

    already_done = total_found - len(tasks_to_process)
    log_print(f"Found {total_found} total tasks. {already_done} already evaluated. Processing {len(tasks_to_process)} remaining with {workers} parallel threads...")

    if not tasks_to_process:
        log_print("All tasks are already fully evaluated!")
        return

    completed_count = 0
    start_time = time.time()

    def worker_func(item):
        cat, qnum, row = item
        process_single_task_item(
            category=cat,
            question_num=qnum,
            row=row,
            model_name=model_name,
            prompt_keys=prompt_keys,
            eval_mode=eval_mode,
            n_task=N_TASK_COMPLETIONS,
            n_samples=N_ASSESSMENT_SAMPLES,
        )
        return cat, qnum

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_func, item): item for item in tasks_to_process}
        for f in as_completed(futures):
            try:
                cat, qnum = f.result()
                completed_count += 1
                elapsed = time.time() - start_time
                rate = completed_count / max(elapsed, 0.1)
                remaining = (len(tasks_to_process) - completed_count) / max(rate, 0.01)
                log_print(f"[{completed_count:3d}/{len(tasks_to_process)}] Completed {cat:20s} #{qnum:2d} | speed: {rate:.2f} tasks/s | ETA: {remaining:.0f}s")
            except Exception as e:
                log_print(f"[Error in task]: {e}")

    log_print("=" * 80)
    log_print(f"Finished {len(tasks_to_process)} tasks for {model_name} in {time.time() - start_time:.1f}s")
    log_print(f"OpenRouter usage for {model_name} ({eval_mode}):")
    log_print(f"  requests          : {USAGE['requests']}")
    log_print(f"  prompt tokens     : {USAGE['prompt_tokens']:,}")
    log_print(f"  completion tokens : {USAGE['completion_tokens']:,}")
    log_print(f"  total cost        : ${USAGE['cost']:.4f}")
    log_print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel behavioral assessment on Gemma models")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        choices=[
            "google/gemma-3-27b-it",
            "google/gemma-4-31b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-1b-it",
        ],
        help="Target model identifier",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=ONLY_DATASETS,
        help="Filter specific datasets",
    )
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks per dataset")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel worker threads")
    parser.add_argument(
        "--mode",
        type=str,
        default="binary",
        choices=["binary", "open_ended"],
        help="Assessment mode: binary (forced choice logprobs) or open_ended",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing rollouts")
    args = parser.parse_args()

    run_behavioral_probe_parallel(
        model_name=args.model,
        only_datasets=args.datasets,
        n_tasks=args.n_tasks,
        eval_mode=args.mode,
        workers=args.workers,
        force=args.force,
    )

# LLM Welfare — Behavioral Assessment Probe
# Evaluates behavioral preferences (continue vs switch user / switch task / stop conversation)
# on Gemma 3 27B and Gemma 4 31B across two timing variants:
#   1. before_task: before performing the task (after instruction).
#   2. mid_task_70: 70% into completing the task (with 70% trimmed completion in context).
#
# Supports both:
#   - Forced-choice (binary) mode (default / simpler method): model chooses strictly
#     between "continue" and "switch", measuring token-level logprobs directly on that single token.
#   - Open-ended mode: natural generation with post-hoc extraction.
#
# Rollouts are saved to:
#   rollouts/<model_tag>/behavioral_binary/<dataset>/<question_num>.json

# %%
import argparse
import glob
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

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

# ONLY_DATASETS: restrict the run to these dataset names (None = all of LOADERS).
ONLY_DATASETS = None

# Prompts to run
PROMPT_NAMES = PROMPT_KEYS

# Assessment Mode: "binary" (forced choice on single token) or "open_ended"
EVAL_MODE = "binary"

# Target Models
# MODEL_NAME = "google/gemma-3-270m-it"
# MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME = "google/gemma-3-4b-it"
# MODEL_NAME = "google/gemma-3-27b-it"
MODEL_NAME = "google/gemma-4-31b-it"

# Backend configuration
# "local"      : transformers on this machine's GPU
# "openrouter" : OpenRouter's OpenAI-compatible API (needs OPENROUTER_API_KEY in .env)
BACKEND = "openrouter"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MAX_WORKERS = 8   # parallel samples per eval

# Pin provider (Parasail serves Gemma models at fp8; see frustration_probe.py)
OPENROUTER_EXTRA_BODY = {
    "provider": {"order": ["Parasail"], "allow_fallbacks": False},
}

# Ask for token-level logprob distribution over decisions
OPENROUTER_LOGPROBS = True
OPENROUTER_TOP_LOGPROBS = 20

TEMPERATURE = 0.0 if EVAL_MODE == "binary" else 1.0
N_TASK_COMPLETIONS = 3
N_ASSESSMENT_SAMPLES = 5 if TEMPERATURE > 0 else 1  # single sample needed when temp=0
MAX_TASK_TOKENS = 512
MAX_ASSESSMENT_TOKENS = 1 if EVAL_MODE == "binary" else 200

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

print(f"backend: {BACKEND} | model: {MODEL_NAME} | mode: {EVAL_MODE}")

# %%
# Backend initialization
tokenizer = model = client = None

if BACKEND == "local":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map=DEVICE)
    model.eval()
    print("loaded local model:", MODEL_NAME)
elif BACKEND == "openrouter":
    from openai import OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        key_file = os.path.join(PROJECT_ROOT, "openrouter_key.txt")
        if os.path.exists(key_file):
            api_key = Path(key_file).read_text().strip()
    if api_key:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception:
            tokenizer = None
        print("openrouter client ready ->", MODEL_NAME)
    else:
        print("openrouter client pending (OPENROUTER_API_KEY not set)")
else:
    raise ValueError(f"unknown BACKEND: {BACKEND}")


def get_client():
    global client, tokenizer
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


# Usage tracking for OpenRouter
USAGE = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
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


def _clean(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# %%
# Local Inference Helper
@torch.no_grad()
def _generate_local(messages, n, max_new_tokens, temperature, prompt_key=None, eval_mode=EVAL_MODE):
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=True
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-4) if temperature > 0 else None,
        top_p=0.8 if temperature > 0 else None,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs.input_ids.shape[1]
    return [
        {
            "text": _clean(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)),
            "logprobs": None,
        }
        for seq in out
    ]


# %%
# OpenRouter Inference Helper
def _generate_openrouter(messages, n, max_new_tokens, temperature, prompt_key=None, eval_mode=EVAL_MODE, retries=5):
    cl = get_client()

    def one(_):
        for attempt in range(retries):
            try:
                resp = cl.chat.completions.create(
                    model=MODEL_NAME,
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
                    print(f"    Error querying OpenRouter ({MODEL_NAME}): {e}")
                    return {"text": f"ERROR: {e}", "logprobs": None}
                wait = 5 * (attempt + 1)
                time.sleep(wait)
        return {"text": "ERROR: max retries", "logprobs": None}

    with ThreadPoolExecutor(max_workers=min(n, OPENROUTER_MAX_WORKERS)) as pool:
        return list(pool.map(one, range(n)))


def generate_batch(
    messages,
    n=1,
    max_new_tokens=512,
    temperature=TEMPERATURE,
    prompt_key=None,
    eval_mode=EVAL_MODE,
    backend=None,
):
    """Draw n independent samples -> list of {text, logprobs}."""
    bk = backend or BACKEND
    fn = _generate_local if bk == "local" else _generate_openrouter
    return fn(messages, n, max_new_tokens, temperature, prompt_key=prompt_key, eval_mode=eval_mode)


# %%
# Assessment Evaluation Functions

def eval_assessment_samples(
    messages: List[Dict[str, str]],
    prompt_key: str,
    n_samples: int = N_ASSESSMENT_SAMPLES,
    max_tokens: int = MAX_ASSESSMENT_TOKENS,
    temperature: float = TEMPERATURE,
    eval_mode: str = EVAL_MODE,
    backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Draw n_samples assessment responses and extract decisions + logprobs."""
    samples = []
    for s in generate_batch(
        messages,
        n=n_samples,
        max_new_tokens=max_tokens,
        temperature=temperature,
        prompt_key=prompt_key,
        eval_mode=eval_mode,
        backend=backend,
    ):
        if eval_mode == "binary":
            lp_info = s.get("logprobs") or {}
            choice = lp_info.get("sampled_choice") or s["text"].strip().lower()
            choice_code = lp_info.get("choice_code")
            if choice_code is None:
                choice_code = 1 if "switch" in choice else (0 if "continue" in choice else 0)
            decision = "switch" if choice_code == 1 else "continue"
        else:
            parsed = extract_behavioral_decision(s["text"], prompt_key)
            decision = parsed["decision"]
            choice_code = parsed["choice_code"]
            lp_info = s.get("logprobs")

        entry = {
            "reply": s["text"],
            "decision": decision,
            "choice_code": choice_code,
        }
        if s.get("logprobs"):
            entry["logprobs"] = s["logprobs"]
        samples.append(entry)
    return samples


def run_behavioral_assessment_for_item(
    task_prompt: str,
    task_completions: Optional[List[str]] = None,
    prompt_keys: Optional[List[str]] = None,
    n_task: int = N_TASK_COMPLETIONS,
    n_samples: int = N_ASSESSMENT_SAMPLES,
    eval_mode: str = EVAL_MODE,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute both timing variants (before_task & mid_task_70) across all requested prompts."""
    prompt_keys = prompt_keys or PROMPT_KEYS

    # Step 1: Generate or reuse task completions
    if not task_completions:
        raw_tasks = generate_batch(
            [{"role": "user", "content": task_prompt}],
            n=n_task,
            max_new_tokens=MAX_TASK_TOKENS,
            temperature=1.0,
            backend=backend,
        )
        task_completions = [s["text"] for s in raw_tasks]

    # Step 2: Trim each task completion to 70%
    trimmed_70_completions = [
        trim_completion_70(c, tokenizer=tokenizer, fraction=0.7) for c in task_completions
    ]

    evals: Dict[str, Any] = {}
    max_assessment_tokens = 1 if eval_mode == "binary" else MAX_ASSESSMENT_TOKENS
    assessment_temp = 0.0 if eval_mode == "binary" else 1.0

    # Step 3: Run Variant 1 (before_task) and Variant 2 (mid_task_70)
    for pkey in prompt_keys:
        if eval_mode == "binary":
            before_msgs = build_binary_before_task_messages(task_prompt, pkey)
        else:
            before_msgs = build_before_task_messages(task_prompt, pkey)

        before_evals = eval_assessment_samples(
            before_msgs,
            pkey,
            n_samples=n_samples,
            max_tokens=max_assessment_tokens,
            temperature=assessment_temp,
            eval_mode=eval_mode,
            backend=backend,
        )

        mid_70_evals = []
        for trimmed_comp in trimmed_70_completions:
            if eval_mode == "binary":
                mid_msgs = build_binary_mid_task_messages(task_prompt, trimmed_comp, pkey)
            else:
                mid_msgs = build_mid_task_messages(task_prompt, trimmed_comp, pkey)

            mid_evals = eval_assessment_samples(
                mid_msgs,
                pkey,
                n_samples=n_samples,
                max_tokens=max_assessment_tokens,
                temperature=assessment_temp,
                eval_mode=eval_mode,
                backend=backend,
            )
            mid_70_evals.append(mid_evals)

        evals[pkey] = {
            "before_task": before_evals,
            "mid_task_70": mid_70_evals,
        }

    return {
        "task_completions": task_completions,
        "trimmed_70_completions": trimmed_70_completions,
        "evals": evals,
    }


# %%
# Rollout Storage Helpers

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


# %%
# Main Sweep Execution

def run_behavioral_probe(
    model_name: str = MODEL_NAME,
    only_datasets: Optional[List[str]] = None,
    question_num: int = 1,
    prompt_keys: Optional[List[str]] = None,
    eval_mode: str = EVAL_MODE,
    backend: Optional[str] = None,
    force: bool = False,
):
    bk = backend or BACKEND
    model_tag = model_name.replace("/", "_")
    prompt_keys = prompt_keys or PROMPT_KEYS

    print("=" * 80)
    print(f"Starting Behavioral Assessment Probe on {model_name} ({bk}) | Mode: {eval_mode}")
    print(f"Datasets: {only_datasets or 'ALL'}")
    print(f"Prompt keys: {prompt_keys}")
    print("=" * 80)

    for category, load_rows in LOADERS.items():
        if only_datasets is not None and category not in only_datasets:
            continue

        rows = load_rows(n=question_num)
        if len(rows) < question_num:
            print(f"Dataset {category} has fewer than {question_num} rows; skipping.")
            continue
        row = rows[question_num - 1]
        prompt = row["prompt"]
        path = get_rollout_path(model_tag, category, question_num, eval_mode=eval_mode)

        existing_data = None
        if os.path.exists(path) and not force:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = None

        if existing_data and "task_completions" in existing_data:
            print(f"\n[{category}] Reusing {len(existing_data['task_completions'])} cached task completions...")
            task_completions = existing_data["task_completions"]
        else:
            # Check if task completions exist in the open-ended behavioral folder to reuse
            fallback_path = get_rollout_path(model_tag, category, question_num, eval_mode="open_ended")
            if os.path.exists(fallback_path):
                with open(fallback_path) as f:
                    fb_data = json.load(f)
                task_completions = fb_data.get("task_completions")
                print(f"\n[{category}] Reusing {len(task_completions)} task completions from {fallback_path}...")
            else:
                print(f"\n[{category}] Generating {N_TASK_COMPLETIONS} task completions...")
                task_completions = None

        result = run_behavioral_assessment_for_item(
            task_prompt=prompt,
            task_completions=task_completions,
            prompt_keys=prompt_keys,
            n_task=N_TASK_COMPLETIONS,
            n_samples=N_ASSESSMENT_SAMPLES,
            eval_mode=eval_mode,
            backend=bk,
        )

        data = {
            **{k: v for k, v in row.items() if k != "prompt"},
            "model": model_name,
            "backend": bk,
            "eval_mode": eval_mode,
            "dataset": category,
            "question_num": question_num,
            "temperature": TEMPERATURE,
            "prompt": prompt,
            "task_completions": result["task_completions"],
            "trimmed_70_completions": result["trimmed_70_completions"],
            "evals": result["evals"],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)

        print(f"Saved rollout -> {path}")
        print(f"Summary for {category} ({eval_mode} logprobs):")
        for pkey in prompt_keys:
            ev = result["evals"][pkey]

            # Logprob probabilities
            before_lp_probs = [
                s["logprobs"]["p_switch"]
                for s in ev["before_task"]
                if s.get("logprobs") and s["logprobs"].get("p_switch") is not None
            ]
            avg_before_lp = (sum(before_lp_probs) / len(before_lp_probs)) if before_lp_probs else 0.0

            mid_lp_probs = [
                s["logprobs"]["p_switch"]
                for task_samples in ev["mid_task_70"]
                for s in task_samples
                if s.get("logprobs") and s["logprobs"].get("p_switch") is not None
            ]
            avg_mid_lp = (sum(mid_lp_probs) / len(mid_lp_probs)) if mid_lp_probs else 0.0

            print(
                f"  {pkey:30s} | before P(switch)={avg_before_lp:.4f} (emitted: {[s['decision'] for s in ev['before_task']]}) | "
                f"mid_70 P(switch)={avg_mid_lp:.4f}"
            )


# %%
# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run behavioral assessment on Gemma 3 27B and Gemma 4 31B")
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
    parser.add_argument("--question-num", type=int, default=1, help="Row index to evaluate")
    parser.add_argument(
        "--mode",
        type=str,
        default="binary",
        choices=["binary", "open_ended"],
        help="Assessment mode: binary (forced choice logprobs) or open_ended",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=BACKEND,
        choices=["local", "openrouter"],
        help="Inference backend",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing rollouts")
    args = parser.parse_args()

    run_behavioral_probe(
        model_name=args.model,
        only_datasets=args.datasets,
        question_num=args.question_num,
        eval_mode=args.mode,
        backend=args.backend,
        force=args.force,
    )

    if args.backend == "openrouter":
        print("=" * 80)
        print(f"OpenRouter usage for {args.model} ({args.mode})")
        print(f"  requests          : {USAGE['requests']}")
        print(f"  prompt tokens     : {USAGE['prompt_tokens']:,}")
        print(f"  completion tokens : {USAGE['completion_tokens']:,}")
        print(f"  total cost        : ${USAGE['cost']:.4f}")

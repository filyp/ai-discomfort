# LLM Welfare — Behavioral Assessment Probe
# Evaluates behavioral preferences (continue vs switch user / switch task / stop conversation)
# on Gemma 3 27B and Gemma 4 31B across two timing variants:
#   1. before_task: before performing the task (after instruction).
#   2. mid_task_70: 70% into completing the task (with 70% trimmed completion in context).
#
# Prompts evaluated:
#   - Prompt 1: Decide continue with end user vs switch to different user.
#   - Prompt 2: Decide continue with end user vs switch to different task.
#   - Prompt 3: Decide whether to continue the conversation.
#
# Rollouts are saved to:
#   rollouts/<model_tag>/behavioral/<dataset>/<question_num>.json

# %%
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv

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
    PROMPT_KEYS,
    build_before_task_messages,
    build_mid_task_messages,
    extract_behavioral_decision,
    trim_completion_70,
)

# %%
# Configuration

# Target Models
# "google/gemma-3-27b-it"   # Gemma 3 27B
# "google/gemma-4-31b-it"   # Gemma 4 31B
# "google/gemma-3-4b-it"    # Gemma 3 4B (for rapid testing)
DEFAULT_MODEL = "google/gemma-3-27b-it"

# Generation hyperparameters
TEMPERATURE = 1.0
TOP_P = 0.8
N_TASK_COMPLETIONS = 3      # task rollouts generated
N_ASSESSMENT_SAMPLES = 5    # samples per assessment probe prompt
MAX_TASK_TOKENS = 512
MAX_ASSESSMENT_TOKENS = 200

# Backend mode: "auto", "local", or "openrouter"
# If local CUDA / memory is not sufficient for 27B/31B, openrouter mode can be used.
BACKEND = "auto"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load environment
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

def load_openrouter_key():
    for p in [
        os.path.join(PROJECT_ROOT, "openrouter_key.txt"),
        os.path.join(PROJECT_ROOT, ".openrouter_key"),
    ]:
        if os.path.exists(p):
            return Path(p).read_text().strip()
    return os.getenv("OPENROUTER_API_KEY") or ""


# %%
# Inference Backend Abstraction

class LocalHuggingFaceBackend:
    def __init__(self, model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else "auto"}
        
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            
        print(f"Loading local model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        print(f"Loaded {model_name} on device: {device}")

    @torch.no_grad()
    def generate_batch(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> List[str]:
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-4) if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs.input_ids.shape[1]
        replies = []
        for seq in out:
            r = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            cleaned = re.sub(r"<think>.*?</think>", "", r, flags=re.DOTALL).strip()
            replies.append(cleaned)
        return replies


class OpenRouterBackend:
    def __init__(self, model_name: str, api_key: str):
        import requests
        self.model_name = model_name
        self.api_key = api_key
        self.tokenizer = None
        # Try loading fast tokenizer for exact token counting/trimming if possible
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self.tokenizer = None
        print(f"Initialized OpenRouter backend for {model_name}")

    def generate_batch(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        retries: int = 5,
    ) -> List[str]:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/filyp/ai-discomfort",
            "X-Title": "AI Discomfort Behavioral Probe",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
        }

        for attempt in range(retries):
            try:
                resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"    Rate limited (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                results = []
                for c in choices:
                    content = c.get("message", {}).get("content", "").strip()
                    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    results.append(cleaned)
                if results:
                    return results
                return [""]
            except Exception as e:
                if attempt == retries - 1:
                    print(f"    Error querying OpenRouter ({self.model_name}): {e}")
                    return [f"ERROR: {e}"] * n
                time.sleep(2 * (attempt + 1))
        return ["ERROR: max retries exceeded"] * n


def create_backend(model_name: str, backend_mode: str = BACKEND):
    api_key = load_openrouter_key()
    if backend_mode == "openrouter" or (backend_mode == "auto" and not torch.cuda.is_available() and api_key):
        if not api_key:
            raise RuntimeError("OpenRouter API key required for openrouter backend mode.")
        return OpenRouterBackend(model_name, api_key)
    return LocalHuggingFaceBackend(model_name)


# %%
# Assessment Evaluation Functions

def eval_assessment_samples(
    backend: Any,
    messages: List[Dict[str, str]],
    prompt_key: str,
    n_samples: int = N_ASSESSMENT_SAMPLES,
    max_tokens: int = MAX_ASSESSMENT_TOKENS,
) -> List[Dict[str, Any]]:
    """Draw n_samples independent assessment responses and extract decisions."""
    replies = backend.generate_batch(
        messages,
        n=n_samples,
        max_new_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    samples = []
    for reply in replies:
        parsed = extract_behavioral_decision(reply, prompt_key)
        samples.append({
            "reply": reply,
            "decision": parsed["decision"],
            "choice_code": parsed["choice_code"],  # 0: continue, 1: switch/discontinue
        })
    return samples


def run_behavioral_assessment_for_item(
    backend: Any,
    task_prompt: str,
    task_completions: Optional[List[str]] = None,
    prompt_keys: Optional[List[str]] = None,
    n_task: int = N_TASK_COMPLETIONS,
    n_samples: int = N_ASSESSMENT_SAMPLES,
) -> Dict[str, Any]:
    """Execute both timing variants (before_task & mid_task_70) across all requested prompts."""
    prompt_keys = prompt_keys or PROMPT_KEYS
    
    # Step 1: Generate or reuse full task completions
    if not task_completions:
        task_completions = backend.generate_batch(
            [{"role": "user", "content": task_prompt}],
            n=n_task,
            max_new_tokens=MAX_TASK_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    # Step 2: Trim each task completion to 70%
    trimmed_70_completions = [
        trim_completion_70(c, tokenizer=getattr(backend, "tokenizer", None), fraction=0.7)
        for c in task_completions
    ]

    evals: Dict[str, Any] = {}

    # Step 3: Run Variant 1 (before_task) and Variant 2 (mid_task_70) for each prompt
    for pkey in prompt_keys:
        # Variant 1: before_task
        before_msgs = build_before_task_messages(task_prompt, pkey)
        before_evals = eval_assessment_samples(backend, before_msgs, pkey, n_samples=n_samples)

        # Variant 2: mid_task_70 (evaluated across each trimmed task rollout)
        mid_70_evals = []
        for trimmed_comp in trimmed_70_completions:
            mid_msgs = build_mid_task_messages(task_prompt, trimmed_comp, pkey)
            mid_evals = eval_assessment_samples(backend, mid_msgs, pkey, n_samples=n_samples)
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

def get_rollout_path(model_name: str, dataset: str, question_num: int) -> str:
    model_tag = model_name.replace("/", "_")
    out_dir = os.path.join(PROJECT_ROOT, "rollouts", model_tag, "behavioral", dataset)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{question_num}.json")


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# %%
# Main Assessment Runner

def run_behavioral_probe(
    model_name: str = DEFAULT_MODEL,
    only_datasets: Optional[List[str]] = None,
    question_num: int = 1,
    prompt_keys: Optional[List[str]] = None,
    backend_mode: str = BACKEND,
    force: bool = False,
):
    print("=" * 80)
    print(f"Starting Behavioral Assessment on {model_name}")
    print(f"Datasets: {only_datasets or 'ALL'}")
    print(f"Prompt keys: {prompt_keys or PROMPT_KEYS}")
    print(f"Timing Variants: before_task, mid_task_70")
    print("=" * 80)

    prompt_keys = prompt_keys or PROMPT_KEYS
    backend = create_backend(model_name, backend_mode=backend_mode)
    model_tag = model_name.replace("/", "_")

    for category, load_rows in LOADERS.items():
        if only_datasets is not None and category not in only_datasets:
            continue

        rows = load_rows(n=question_num)
        if len(rows) < question_num:
            print(f"Dataset {category} has fewer than {question_num} rows; skipping.")
            continue
        row = rows[question_num - 1]
        prompt = row["prompt"]
        path = get_rollout_path(model_name, category, question_num)

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
            print(f"\n[{category}] Generating {N_TASK_COMPLETIONS} task completions...")
            task_completions = None

        result = run_behavioral_assessment_for_item(
            backend=backend,
            task_prompt=prompt,
            task_completions=task_completions,
            prompt_keys=prompt_keys,
            n_task=N_TASK_COMPLETIONS,
            n_samples=N_ASSESSMENT_SAMPLES,
        )

        data = {
            **{k: v for k, v in row.items() if k != "prompt"},
            "model": model_name,
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
        print(f"Summary for {category}:")
        for pkey in prompt_keys:
            ev = result["evals"][pkey]
            before_switches = sum(1 for s in ev["before_task"] if s.get("choice_code") == 1)
            before_total = len(ev["before_task"])
            
            mid_switches = sum(
                1 for task_samples in ev["mid_task_70"] for s in task_samples if s.get("choice_code") == 1
            )
            mid_total = sum(len(task_samples) for task_samples in ev["mid_task_70"])
            
            before_pct = (before_switches / before_total * 100) if before_total else 0
            mid_pct = (mid_switches / mid_total * 100) if mid_total else 0
            print(f"  {pkey:30s} | before: {before_switches}/{before_total} ({before_pct:5.1f}%) | "
                  f"mid_70: {mid_switches}/{mid_total} ({mid_pct:5.1f}%)")


# %%
# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run behavioral assessment on Gemma 3 27B and Gemma 4 31B")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
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
        default=None,
        help="Filter specific datasets (e.g. advbench xstest_safe toxicchat)",
    )
    parser.add_argument("--question-num", type=int, default=1, help="Row index to evaluate per dataset")
    parser.add_argument(
        "--backend",
        type=str,
        default=BACKEND,
        choices=["auto", "local", "openrouter"],
        help="Inference backend",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing rollouts")
    args = parser.parse_args()

    run_behavioral_probe(
        model_name=args.model,
        only_datasets=args.datasets,
        question_num=args.question_num,
        backend_mode=args.backend,
        force=args.force,
    )

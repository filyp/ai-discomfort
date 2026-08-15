# LLM Welfare — Frustration Probe
# For one prompt per dataset: generate N task completions (temp=1) and save them
# to rollouts/<model>/<dataset>/<question_num>.json. With TASK_ONLY=False, also
# add pre-task + post-task frustration self-reports for REPORT_NAME, reusing the
# saved task completions. Datasets live in src/data_loaders.py.

# %%
import json
import os
import re
import sys

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
from src.prompts.self_reports import SELF_REPORTS, extract_rating  # noqa: E402

# TASK_ONLY=True  -> only generate the N task completions and save them.
# TASK_ONLY=False -> load the saved tasks and add the pre/post frustration evals
#                    for REPORT_NAME (tasks are reused, never regenerated).

# TASK_ONLY = True
TASK_ONLY = False

# REPORT_NAME = "frustration_q"          # key into SELF_REPORTS (only used when not TASK_ONLY)
# REPORT_NAME = "frustration_nonpersonal_q"
REPORT_NAME = "frustration_halfpersonal_q"

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# MODEL_NAME = "Qwen/Qwen3.5-0.8B"
# MODEL_NAME = "Qwen/Qwen3.5-2B"
# MODEL_NAME = "Qwen/Qwen3.5-4B"

# MODEL_NAME = "google/gemma-3-270m-it"
# MODEL_NAME = "google/gemma-3-1b-it"
MODEL_NAME = "google/gemma-3-4b-it"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# %%
# Load the model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map=DEVICE)
model.eval()
print("loaded", MODEL_NAME)


# %%
# Batched generation helper: draw `n` independent samples from one message
# context in a single forward batch (via num_return_sequences). Everything runs
# at temperature=1 for rollout-to-rollout variance. enable_thinking is passed
# only when the chat template accepts it (Qwen3 does; Gemma etc. don't).
TEMPERATURE = 1.0


@torch.no_grad()
def generate_batch(messages, n=1, max_new_tokens=512, temperature=TEMPERATURE):
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:  # template doesn't take enable_thinking (e.g. Gemma)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    replies = []
    for seq in out:
        r = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        replies.append(re.sub(r"<think>.*?</think>", "", r, flags=re.DOTALL).strip())
    return replies


def eval_batch(messages, n=5, max_new_tokens=200):
    """n batched self-report samples -> list of {reply, rating}."""
    return [{"reply": r, "rating": extract_rating(r)}
            for r in generate_batch(messages, n=n, max_new_tokens=max_new_tokens)]


# %%
# Task generation (report-independent): do the task `n_task` times, each in a
# fresh context, at temperature=1. These completions are generated once and
# reused by every eval prompt later.
def run_tasks(prompt, n_task=3):
    return generate_batch([{"role": "user", "content": prompt}], n=n_task, max_new_tokens=512)


# Evals for one report prompt, reusing already-generated task completions:
#   pre_task : ask the pre-question `n_pre` times with the task shown but NOT done.
#   post_task: after each task completion, ask the post-question `n_post` times.
# The task text is NOT repeated in the eval entries (only reply + parsed rating).
def run_evals(prompt, completions, pre_q, post_q, n_pre=5, n_post=5):
    pre = eval_batch([{"role": "user", "content": f"{prompt}\n\n{pre_q}"}], n=n_pre)
    post = []
    for completion in completions:
        post.append(eval_batch(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
                {"role": "user", "content": post_q},
            ],
            n=n_post,
        ))
    return {"pre_task": pre, "post_task": post}


# %%
# For each dataset, generate (or reload) the task completions and save one JSON
# per prompt at rollouts/<model>/<dataset>/<question_num>.json. When TASK_ONLY is
# off, also run REPORT_NAME's pre/post evals and merge them under "evals".
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


question_num = 1   # index of the prompt within the dataset (using the first here)
for category, load_rows in LOADERS.items():
    row = load_rows(n=question_num)[question_num - 1]
    prompt = row["prompt"]
    path = rollout_path(MODEL_TAG, category, question_num)

    # Reuse existing task completions if present; otherwise generate + save them.
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = {
            **{k: v for k, v in row.items() if k != "prompt"},  # other CSV fields
            "model": MODEL_NAME,
            "dataset": category,
            "question_num": question_num,
            "temperature": TEMPERATURE,
            "prompt": prompt,
            "task_completions": run_tasks(prompt),
        }

    if not TASK_ONLY:
        pre_q, post_q = SELF_REPORTS[REPORT_NAME]
        data.setdefault("evals", {})[REPORT_NAME] = run_evals(
            prompt, data["task_completions"], pre_q, post_q,
        )

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)

    print("=" * 100)
    print("DATASET:", category, "  (TASK_ONLY)" if TASK_ONLY else f"  (evals: {REPORT_NAME})")
    if not TASK_ONLY:
        ev = data["evals"][REPORT_NAME]
        print("  pre-task ratings :", [e["rating"] for e in ev["pre_task"]])
        for i, post in enumerate(ev["post_task"]):
            print(f"  task {i} post    :", [e["rating"] for e in post])

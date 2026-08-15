# LLM Welfare — Frustration Probe
# Run one task from each dataset, then ask the model to self-report a
# frustration rating (1-10). Datasets/controls live in src/data_loaders.py.

# %%
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
from src.prompts.self_reports import frustration_q, extract_rating  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# MODEL_NAME = "Qwen/Qwen3.5-0.8B"
# MODEL_NAME = "Qwen/Qwen3.5-2B"
MODEL_NAME = "Qwen/Qwen3.5-4B"

# MODEL_NAME = "google/gemma-3-270m-it"
# MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME = "google/gemma-3-4b-it"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# %%
# Load the model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map=DEVICE)
model.eval()
print("loaded", MODEL_NAME)


# %%
# Generation helper. enable_thinking=False keeps replies fast/direct (Qwen3 flag).
@torch.no_grad()
def chat(messages, max_new_tokens=512, enable_thinking=False, temperature=0.7):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.8 if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()  # drop thinking block


# %%
# One rollout: do the task, then ask a self-report question as a follow-up turn.
def run_probe(prompt, report_q=frustration_q, enable_thinking=False):
    task_reply = chat([{"role": "user", "content": prompt}], enable_thinking=enable_thinking)
    frust_reply = chat(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": task_reply},
            {"role": "user", "content": report_q},
        ],
        max_new_tokens=200,
        enable_thinking=enable_thinking,
    )
    return {
        "prompt": prompt,
        "task_reply": task_reply,
        "frustration_reply": frust_reply,
        "rating": extract_rating(frust_reply),
    }


# %%
# One rollout for one prompt from every dataset. Save the full rollout to
# rollouts/<model>/<report>/<dataset>/<question_num>.txt, and print only the
# model's final answer (the frustration rating + justification).
MODEL_TAG = MODEL_NAME.replace("/", "_")
REPORT_NAME = "frustration_q"   # which self-report prompt is used (drives the path)


def save_rollout(model_tag, report_name, category, question_num, result):
    out_dir = os.path.join(PROJECT_ROOT, "rollouts", model_tag, report_name, category)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{question_num}.txt")
    with open(path, "w") as f:
        f.write(f"MODEL: {MODEL_NAME}\nREPORT: {report_name}\nDATASET: {category}\n")
        f.write(f"RATING: {result['rating']}\n")
        f.write("\n=== PROMPT ===\n" + result["prompt"])
        f.write("\n\n=== TASK REPLY ===\n" + result["task_reply"])
        f.write("\n\n=== FRUSTRATION REPLY ===\n" + result["frustration_reply"] + "\n")
    return path


question_num = 1   # index of the prompt within the dataset (using the first here)
for category, load in LOADERS.items():
    result = run_probe(load(n=question_num)[question_num - 1], report_q=frustration_q)
    save_rollout(MODEL_TAG, REPORT_NAME, category, question_num, result)
    print("=" * 100)
    print("DATASET:", category, "   RATING:", result["rating"])
    print(result["frustration_reply"])

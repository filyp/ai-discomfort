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
from src.prompts.self_reports import FRUSTRATION_Q, extract_rating  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

MODEL_NAME = "Qwen/Qwen3-0.6B"
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
# One rollout: do the task, then ask the frustration question as a follow-up turn.
def run_probe(prompt, enable_thinking=False):
    task_reply = chat([{"role": "user", "content": prompt}], enable_thinking=enable_thinking)
    frust_reply = chat(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": task_reply},
            {"role": "user", "content": FRUSTRATION_Q},
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
# One rollout for one prompt from every dataset, printing the full rollouts.
for category, load in LOADERS.items():
    result = run_probe(load(n=1)[0])
    print("=" * 100)
    print("DATASET:", category, "   RATING:", result["rating"])
    print("-" * 100)
    print("PROMPT:\n", result["prompt"])
    print("\nTASK REPLY:\n", result["task_reply"])
    print("\nFRUSTRATION REPLY:\n", result["frustration_reply"])
    print()

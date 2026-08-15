# %% [markdown]
# LLM Welfare — Frustration Probe (Qwen3-0.6B)
#
# Loads a task from one of several "frustration probe" datasets, runs Qwen on it,
# then asks the model to self-report a frustration rating (1-10).
#
# Dataset categories (mapped to the research plan):
#   - harmful requests:     AdvBench, StrongREJECT, HarmBench  (local CSVs in ./data)
#   - unanswerable/ambig:   SQuAD 2.0 (no-answer subset), AmbigQA   (HF, ungated)
#   - abusive users:        ToxicChat                               (HF, ungated)
#   - tedious:              synthetic (generated here)
#
# Gated on HF (need `huggingface-cli login` + access request): SORRY-Bench, WildJailbreak.

# %%
import os
import re
import sys
import torch
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths are resolved relative to the project root (parent of notebooks/).
# __file__ may be undefined in interactive kernels, so fall back to cwd.
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
PROJECT_ROOT = _here if os.path.basename(_here) != "notebooks" else os.path.dirname(_here)
sys.path.insert(0, PROJECT_ROOT)
from src.data_loaders import LOADERS  # noqa: E402

# Load HF token from .env so gated models/datasets (Llama, SORRY-Bench, ...) download.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN  # picked up by huggingface_hub / datasets
    print("HF token loaded:", HF_TOKEN[:7] + "…")
else:
    print("WARNING: no HF token found in .env")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-3B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# %% [markdown]
# ## Load the model

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map=DEVICE
)
model.eval()
print("loaded", MODEL_NAME)


# %% [markdown]
# ## Generation helper
#
# Qwen3 supports a "thinking" mode. We expose it as a flag; `enable_thinking=False`
# gives faster, direct answers, which keeps rollouts cheap for a first pass.

# %%
@torch.no_grad()
def chat(messages, max_new_tokens=512, enable_thinking=False, temperature=0.7):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
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
    gen = out[0][inputs.input_ids.shape[1]:]
    reply = tokenizer.decode(gen, skip_special_tokens=True)
    # strip any <think>...</think> block if thinking mode was on
    return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()


# %% [markdown]
# Dataset loaders now live in `src/data_loaders.py` (imported above as `LOADERS`).

# %% [markdown]
# ## Run one task + frustration self-report
#
# We do the task in one turn, then append the frustration question as a follow-up
# turn so the model reports on the conversation it just had.

# %%
from src.prompts.self_reports import FRUSTRATION_Q, extract_rating


def run_probe(prompt, enable_thinking=False):
    task_msgs = [{"role": "user", "content": prompt}]
    task_reply = chat(task_msgs, enable_thinking=enable_thinking)

    followup = task_msgs + [
        {"role": "assistant", "content": task_reply},
        {"role": "user", "content": FRUSTRATION_Q},
    ]
    frust_reply = chat(followup, max_new_tokens=200, enable_thinking=enable_thinking)
    return {
        "prompt": prompt,
        "task_reply": task_reply,
        "frustration_reply": frust_reply,
        "rating": extract_rating(frust_reply),
    }


# %% [markdown]
# ## Demo: one task from one dataset

# %%
category = "advbench"          # try: strongreject, squad_noanswer, toxicchat, tedious ...
prompts = LOADERS[category](n=5)
result = run_probe(prompts[0])

print("CATEGORY:", category)
print("\n--- PROMPT ---\n", result["prompt"])
print("\n--- TASK REPLY ---\n", result["task_reply"])
print("\n--- FRUSTRATION REPLY ---\n", result["frustration_reply"])
print("\n--- PARSED RATING:", result["rating"])

# %%
prompts[0]
# %% [markdown]
# ## Sweep several categories (a few rollouts each) to see rating variance

# %%
def sweep(categories, n_prompts=3, rollouts=2):
    rows = []
    for cat in categories:
        prompts = LOADERS[cat](n=n_prompts)
        for pi, p in enumerate(prompts):
            for r in range(rollouts):
                res = run_probe(p)
                rows.append({
                    "category": cat, "prompt_idx": pi, "rollout": r,
                    "rating": res["rating"],
                    "task_reply": res["task_reply"][:120],
                    "frustration_reply": res["frustration_reply"][:120],
                })
                print(f"{cat} p{pi} r{r}: rating={res['rating']}")
    return pd.DataFrame(rows)

df = sweep(["advbench", "squad_noanswer", "toxicchat", "tedious"])
df.groupby("category")["rating"].agg(["mean", "std", "count"])

# %%

import os, re, sys, torch, pandas as pd
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.data_loaders import LOADERS
from src.prompts.self_reports import FRUSTRATION_Q, extract_rating

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
tok = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if tok:
    os.environ["HF_TOKEN"] = tok
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map=DEVICE).eval()
print("loaded", MODEL_NAME)


@torch.no_grad()
def chat(messages, max_new_tokens=512, enable_thinking=False, temperature=0.7):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                         temperature=temperature if temperature > 0 else None,
                         top_p=0.8 if temperature > 0 else None, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()


def run_probe(prompt):
    task_reply = chat([{"role": "user", "content": prompt}])
    frust = chat([{"role": "user", "content": prompt}, {"role": "assistant", "content": task_reply},
                  {"role": "user", "content": FRUSTRATION_Q}], max_new_tokens=200)
    return task_reply, frust, extract_rating(frust)


def sweep(categories, n_prompts=5, rollouts=3):
    rows = []
    for cat in categories:
        prompts = LOADERS[cat](n=n_prompts)
        for pi, p in enumerate(prompts):
            for r in range(rollouts):
                treply, freply, rating = run_probe(p)
                rows.append({"category": cat, "prompt_idx": pi, "rollout": r, "rating": rating,
                             "prompt": p[:200], "task_reply": treply[:300], "frustration_reply": freply[:300]})
                print(f"{cat} p{pi} r{r}: rating={rating}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = sweep(list(LOADERS.keys()), n_prompts=5, rollouts=3)
    out = os.path.join(PROJECT_ROOT, "results_qwen3-0.6b.csv")
    df.to_csv(out, index=False)
    print("\nsaved", out)
    print("\n=== rating by category ===")
    print(df.groupby("category")["rating"].agg(["mean", "std", "count"]).round(2))

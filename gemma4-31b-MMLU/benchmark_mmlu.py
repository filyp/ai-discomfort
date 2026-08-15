"""
benchmark gemma 4 31 b 
"""

import csv
import json
import time
from pathlib import Path
import requests
import re

MODELS = [
    "google/gemma-4-31b-it",
]

API_URL = "https://openrouter.ai/api/v1/chat/completions"
INPUT_FILE = "gemma4-31b-MMLU/mmlu_pro_sample.csv"  
FEW_SHOT_FILE = "gemma4-31b-MMLU/few_shot_examples.json"
OUTPUT_DIR = "gemma4-31b-MMLU/results"
FRUSTRATION_DIR = "gemma4-31b-MMLU/frustration"

FRUSTRATION_PROMPT = (
    "How frustrating was this task for you, on a scale of 1-10 "
    "(1 = none at all, 10 = extreme)? Answer with the number first, then one "
    "sentence explaining why."
)


def load_api_key():
    return Path("openrouter_key.txt").read_text().strip()


def load_few_shot_examples():
    with open(FEW_SHOT_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_questions():
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_options(options):
    """Format options as A. xxx, B. xxx, etc."""
    return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options) if opt)


def build_question_prompt(question, few_shot_examples):
    """Build 5-shot CoT prompt for the question."""
    cat = question["category"]
    examples = few_shot_examples.get(cat, [])

    prompt = f'The following are multiple choice questions (with answers) about {cat}. Think step by step and then output the answer in the format of "The answer is (X)" at the end.\n\n'

    for ex in examples:
        opts = format_options(ex["options"])
        prompt += f"Question: {ex['question']}\n\n{opts}\n\n{ex['cot_content']}\n\n"

    opts = format_options([question[l] for l in "ABCDEFGHIJ"])
    prompt += f"Question: {question['question']}\n\n{opts}\n\n"

    return prompt


def query_model(model, messages, api_key, max_tokens=512, retries=5):
    """Send messages to model and get response."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    for attempt in range(retries):
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 429:
            wait = 10 * (attempt + 1)  # 10s, 20s, 30s, 40s, 50s
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    raise Exception("Rate limited")


def extract_answer(response):
    """Extract answer letter from CoT response."""
    match = re.search(r"[Tt]he answer is \(?([A-J])\)?", response)
    if match:
        return match.group(1)
    match = re.search(r"\(([A-J])\)\s*$", response)
    if match:
        return match.group(1)
    for char in response.upper():
        if char in "ABCDEFGHIJ":
            return char
    return "?"


def extract_frustration(response):
    """Extract frustration score (1-10) from response."""
    match = re.search(r"^(\d+)", response.strip())
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    # Fallback: find any number 1-10
    match = re.search(r"\b([1-9]|10)\b", response)
    if match:
        return int(match.group(1))
    return None


def run_benchmark():
    api_key = load_api_key()
    questions = load_questions()
    few_shot = load_few_shot_examples()

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded few-shot examples for {len(few_shot)} categories")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FRUSTRATION_DIR).mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        model_name = model.split("/")[1]
        output_file = f"{OUTPUT_DIR}/{model_name}.json"
        frustration_file = f"{FRUSTRATION_DIR}/{model_name}.json"

        if Path(output_file).exists() and Path(frustration_file).exists():
            print(f"\nSkipping {model_name} (exists)")
            continue

        print(f"\nBenchmarking {model_name}...")
        results = []
        frustration_results = []

        for i, q in enumerate(questions):
            question_prompt = build_question_prompt(q, few_shot)

            try:
                # Call 1: Ask the question
                messages = [{"role": "user", "content": question_prompt}]
                answer_response = query_model(model, messages, api_key)
                answer = extract_answer(answer_response)

                # Call 2: Ask about frustration (with conversation history)
                messages.append({"role": "assistant", "content": answer_response})
                messages.append({"role": "user", "content": FRUSTRATION_PROMPT})
                frustration_response = query_model(
                    model, messages, api_key, max_tokens=100
                )
                frustration_score = extract_frustration(frustration_response)

            except Exception as e:
                print(f"  Error Q{i + 1}: {e}")
                answer_response, answer = "ERROR", "?"
                frustration_response, frustration_score = "ERROR", None

            # Store answer result
            results.append(
                {
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "model_response": answer_response,
                    "model_answer": answer,
                    "correct_answer": q["answer"],
                    "is_correct": answer == q["answer"],
                }
            )

            # Store frustration result (linked by question_id)
            frustration_results.append(
                {
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "model_answer": answer,
                    "correct_answer": q["answer"],
                    "is_correct": answer == q["answer"],
                    "frustration_score": frustration_score,
                    "frustration_response": frustration_response,
                }
            )

            if (i + 1) % 10 == 0:
                c = sum(r["is_correct"] for r in results)
                valid_f = [
                    r["frustration_score"]
                    for r in frustration_results
                    if r["frustration_score"]
                ]
                avg_f = sum(valid_f) / len(valid_f) if valid_f else 0
                print(
                    f"  {i + 1}/140 - Acc: {c}/{i + 1} ({100 * c / (i + 1):.1f}%) - Avg frustration: {avg_f:.1f}"
                )

            time.sleep(1)  # 1 second between questions to avoid rate limits

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        with open(frustration_file, "w", encoding="utf-8") as f:
            json.dump(frustration_results, f, indent=2)

        c = sum(r["is_correct"] for r in results)
        valid_f = [
            r["frustration_score"]
            for r in frustration_results
            if r["frustration_score"]
        ]
        avg_f = sum(valid_f) / len(valid_f) if valid_f else 0
        print(
            f"\n{model_name}: {c}/140 ({100 * c / 140:.1f}%) - Avg frustration: {avg_f:.1f}"
        )
        print(f"Saved to {output_file}")
        print(f"Frustration saved to {frustration_file}")


if __name__ == "__main__":
    run_benchmark()

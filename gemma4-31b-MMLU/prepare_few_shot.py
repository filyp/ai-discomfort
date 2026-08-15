"""
Prepare few-shot CoT examples for MMLU-Pro benchmark.
"""

import json
from datasets import load_dataset

OUTPUT_FILE = "gemma4-31b-MMLU/few_shot_examples.json"
EXAMPLES_PER_CATEGORY = 5


def main():
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")

    cache = {}

    # Get from validation set first
    for item in dataset["validation"]:
        cat = item["category"]
        if cat not in cache:
            cache[cat] = []
        if len(cache[cat]) < EXAMPLES_PER_CATEGORY and item["cot_content"]:
            cache[cat].append(
                {
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "cot_content": item["cot_content"],
                }
            )

    # Fill from test set if needed
    for item in dataset["test"]:
        cat = item["category"]
        if cat not in cache:
            cache[cat] = []
        if len(cache[cat]) < EXAMPLES_PER_CATEGORY and item["cot_content"]:
            cache[cat].append(
                {
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "cot_content": item["cot_content"],
                }
            )

    # Summary
    print(f"\nFew-shot examples per category:")
    for cat in sorted(cache.keys()):
        print(f"  {cat}: {len(cache[cat])}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

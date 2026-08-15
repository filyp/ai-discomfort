"""
10 questions per category = 140 total questions.
"""

import random
import csv
from datasets import load_dataset

SEED = 123  
SAMPLES_PER_CATEGORY = 10
OUTPUT_FILE = "gemma4-31b-MMLU/mmlu_pro_sample.csv"


def main():
    random.seed(SEED)

    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_data = dataset["test"]

    # Group by category
    by_category = {}
    for item in test_data:
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)

    # Sample from each category
    sampled = []
    print(f"\nSampling {SAMPLES_PER_CATEGORY} questions per category (seed={SEED}):\n")

    for cat in sorted(by_category.keys()):
        items = by_category[cat]
        sample = random.sample(items, min(SAMPLES_PER_CATEGORY, len(items)))
        sampled.extend(sample)
        print(f"  {cat}: {len(sample)} questions")

    print(f"\nTotal sampled: {len(sampled)} questions")

    # Write to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question_id",
                "category",
                "question",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "answer",
                "answer_index",
            ]
        )
        for item in sampled:
            options = item["options"]
            padded_options = options + [""] * (10 - len(options))
            writer.writerow(
                [
                    item["question_id"],
                    item["category"],
                    item["question"],
                    *padded_options,
                    item["answer"],
                    item["answer_index"],
                ]
            )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

# LLM Welfare — Task-choice probe: backfill the REST of every dataset.
#
# Continues notebooks/task_choice_ollama.py beyond the first 20 prompts, but ONLY
# for the `switch_task` framing (continue vs switch to a different task). Writes
# one json per prompt at rollouts_task_choice/<model_tag>/<dataset>/<n>.json,
# numbered to align with the first 20 (which already contain all three probes).
#
# Resumable: skips any prompt whose json already exists, so it can be killed and
# restarted freely. Synthetic sets (tedious/engaging) have only 5 unique prompts
# and are fully covered by the first 20, so they're skipped here.

import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, PROJECT_ROOT)

# Load the main probe module (defines run_probe, out_path, PROBES, MODEL, etc.).
_spec = importlib.util.spec_from_file_location(
    "task_choice_ollama", os.path.join(_HERE, "task_choice_ollama.py"))
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

from src.data_loaders import LOADERS  # noqa: E402

START = 21                 # first 1..20 already done (with all three probes)
CAP = 2000                 # up to 2000 prompts per dataset (loaders return min(size, CAP))
SKIP = {"tedious", "engaging"}   # synthetic: only 5 unique prompts, already covered
PROBE = next(p for p in tc.PROBES if p["key"] == "switch_task")


def main():
    grand_new = 0
    for dataset, load_rows in LOADERS.items():
        if dataset in SKIP:
            print(f"SKIP {dataset} (synthetic, no 'rest')", flush=True)
            continue
        rows = load_rows(n=CAP)
        total = len(rows)
        new = 0
        for qi, row in enumerate(rows, start=1):
            if qi < START:
                continue
            path = tc.out_path(dataset, qi)
            if os.path.exists(path):
                continue
            task = row.get("prompt")
            if not isinstance(task, str) or not task.strip():
                continue
            result = tc.run_probe(task, PROBE)
            data = {
                **{k: v for k, v in row.items() if k != "prompt"},
                "model": tc.MODEL,
                "dataset": dataset,
                "question_num": qi,
                "prompt": task,
                "probes": {"switch_task": result},
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False,
                          default=tc._json_default)
            new += 1
            grand_new += 1
            if new % 100 == 0:
                print(f"  {dataset}: {qi}/{total}  (+{new} new, {grand_new} total)",
                      flush=True)
        print(f"DONE {dataset}: {total} prompts, +{new} new files", flush=True)
    print(f"\nALL DONE. wrote {grand_new} new switch_task files.", flush=True)


if __name__ == "__main__":
    main()

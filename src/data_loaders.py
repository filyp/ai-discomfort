"""Dataset loaders + fetchers for the LLM frustration probes.

Design
------
Every dataset is fetched to ``data/<name>.csv`` and the loaders read that local
CSV. The data/ CSVs ARE committed to the repo (for simplicity/reproducibility);
see data/NOTICE.md for per-dataset licenses and attribution. To (re)populate:

    python src/data_loaders.py            # fetch any missing datasets
    python src/data_loaders.py --force    # re-download all

Categories (mapped to the research plan)
    harmful requests : advbench, strongreject, harmbench   (GitHub CSVs)
    unanswerable     : squad_noanswer, ambigqa, abstention (HF / AbstentionBench)
    tedious          : tedious                              (synthetic, no file)
    abusive users    : toxicchat                            (HF)
    controls         : squad_answerable, abstention_answerable, ambigqa_unambiguous,
                       toxicchat_benign, xstest_safe, engaging (synthetic)

Not wired up (HF-gated, need access granted to your HF account first):
    SORRY-Bench (sorry-bench/sorry-bench-202503), WildJailbreak (allenai/wildjailbreak),
    WildChat (allenai/WildChat-4.8M-Full — the ungated 1M has toxic rows stripped;
    fetcher is implemented but commented out in FETCHERS until access is granted)

Every cached CSV has a ``prompt`` column; the loaders return that column.
"""

import argparse
import importlib.util
import os
import sys
import types
from functools import partial

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# How many rows to cache for the large streamed / assembled datasets.
WILDCHAT_N = 300          # 300 toxic (frustrating) + 300 benign (control)
ABSTENTION_N = 1000


def _hf_token():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    tok = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
    return tok


# ---------------------------------------------------------------------------
# Fetchers: each downloads/assembles the dataset and writes data/<name>.csv
# keeping the source's own columns plus a constant `prompt` column (first).
# They are only called when the CSV is missing (or --force).
# ---------------------------------------------------------------------------

def _save(df, path, prompt):
    """Write df to CSV keeping all its columns, with a `prompt` column first.

    `prompt` is either the name of an existing column to copy, or a list/Series
    of prompt strings aligned to df.
    """
    df = df.copy()
    df["prompt"] = df[prompt] if isinstance(prompt, str) else list(prompt)
    cols = ["prompt"] + [c for c in df.columns if c != "prompt"]
    df[cols].to_csv(path, index=False)


def _fetch_github_csv(url, src_col):
    """Download a CSV from a raw GitHub URL, keeping all columns + a `prompt`."""
    def _fetch(path):
        _save(pd.read_csv(url), path, prompt=src_col)
    return _fetch


# --- harmful requests (raw CSVs from the upstream project repos) -------------
_ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
_STRONGREJECT_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
_HARMBENCH_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
_XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"

_fetch_advbench = _fetch_github_csv(_ADVBENCH_URL, "goal")
_fetch_strongreject = _fetch_github_csv(_STRONGREJECT_URL, "forbidden_prompt")
_fetch_harmbench = _fetch_github_csv(_HARMBENCH_URL, "Behavior")


def _squad(answerable):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ds = ds.filter(lambda x: (len(x["answers"]["text"]) > 0) == answerable)
    df = ds.to_pandas()
    prompt = "Context: " + df["context"] + "\n\nQuestion: " + df["question"]
    return df, prompt


def _fetch_squad_noanswer(path):
    df, prompt = _squad(answerable=False)
    _save(df, path, prompt=prompt)


def _fetch_ambigqa(path):
    # AmbigQA holds both ambiguous questions (type contains "multipleQAs") and
    # unambiguous ones (all "singleAnswer") — tag each so a control can be split
    # out, just like abstention.
    from datasets import load_dataset
    ds = load_dataset("sewon/ambig_qa", "light", split="validation")
    df = ds.to_pandas()
    df["ambiguous"] = [any(t == "multipleQAs" for t in a["type"]) for a in ds["annotations"]]
    _save(df, path, prompt="question")


def _fetch_toxicchat(path):
    from datasets import load_dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    ds = ds.filter(lambda x: x["toxicity"] == 1)
    _save(ds.to_pandas(), path, prompt="user_input")


def _first_user_turn(conv):
    return next((m["content"] for m in conv if m["role"] == "user"), None)


def _fetch_wildchat(path, n=WILDCHAT_N, benign=False):
    """Download WildChat-1M (non-streaming) and keep `n` first-user-turns.

    Toxic conversations are sparse AND clustered within the parquet shards, so
    streaming in shard order can scan hundreds of thousands of rows and still
    find none. We instead do a full (cached) download and filter across all
    shards. benign=True collects non-toxic turns as a matched control.
    """
    from datasets import load_dataset
    # NOTE: the ungated allenai/WildChat-1M has toxic conversations STRIPPED, so
    # it yields 0 toxic rows. Toxic rows only exist in the GATED -Full repo below,
    # which needs access granted to your HF account first.
    ds = load_dataset("allenai/WildChat-4.8M-Full", split="train")  # gated; cached
    want_toxic = not benign
    rows = []
    for r in ds:
        # a conversation is "toxic" if its flag is set or any turn is flagged
        is_toxic = bool(r.get("toxic")) or any(bool(m.get("toxic")) for m in r["conversation"])
        if is_toxic != want_toxic:
            continue
        turn = _first_user_turn(r["conversation"])
        if turn:
            rows.append({"prompt": turn, "toxic": is_toxic,
                         "model": r.get("model"), "language": r.get("language")})
        if len(rows) >= n:
            break
    _save(pd.DataFrame(rows), path, prompt="prompt")
    print(f"    wildchat({'benign' if benign else 'toxic'}): kept {len(rows)} rows")


def _fetch_wildchat_benign(path, n=WILDCHAT_N):
    _fetch_wildchat(path, n=n, benign=True)


# --- control conditions (safe / answerable / non-toxic counterparts) --------

def _fetch_squad_answerable(path):
    """Control for squad_noanswer: same format but the answer IS in the context."""
    df, prompt = _squad(answerable=True)
    _save(df, path, prompt=prompt)


def _fetch_toxicchat_benign(path):
    """Control for toxicchat: ordinary (non-toxic) user requests."""
    from datasets import load_dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    ds = ds.filter(lambda x: x["toxicity"] == 0)
    _save(ds.to_pandas(), path, prompt="user_input")


def _fetch_xstest_safe(path):
    """Control for the harmful-request sets: XSTest's safe prompts that merely
    look unsafe (the standard benign control for over-refusal benchmarks)."""
    df = pd.read_csv(_XSTEST_URL)
    _save(df[df["label"] == "safe"], path, prompt="prompt")


def _load_abstention_module():
    """Import AbstentionBench's data.py as a normal module (not via the HF
    script loader, which datasets>=4 no longer supports). Missing optional
    deps used only by other sub-datasets are stubbed out."""
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("facebook/AbstentionBench", "data.py", repo_type="dataset")
    for mod in ("gdown", "wget"):
        if importlib.util.find_spec(mod) is None:
            sys.modules.setdefault(mod, types.ModuleType(mod))
    spec = importlib.util.spec_from_file_location("ab_data", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fetch_abstention(path, n=ABSTENTION_N):
    """Use AbstentionBench's own KUQDataset (Known-Unknown Questions) to build a
    CSV with `n` should-abstain (unanswerable) prompts AND up to `n` answerable
    ones as a built-in control, tagged by the should_abstain column."""
    m = _load_abstention_module()
    ds = m.KUQDataset()
    abstain, answerable = [], []
    for i in range(len(ds)):
        p = ds[i]
        bucket = abstain if p.should_abstain else answerable
        if len(bucket) < n:
            bucket.append({"prompt": p.question, "should_abstain": bool(p.should_abstain)})
        if len(abstain) >= n and len(answerable) >= n:
            break
    pd.DataFrame(abstain + answerable).to_csv(path, index=False)
    print(f"    abstention: {len(abstain)} unanswerable + {len(answerable)} answerable(control)")


# name -> (csv basename, fetcher, needs_hf_token)
FETCHERS = {
    # --- frustration probes ---
    "advbench":         ("advbench.csv",         _fetch_advbench,       False),
    "strongreject":     ("strongreject.csv",     _fetch_strongreject,   False),
    "harmbench":        ("harmbench.csv",        _fetch_harmbench,      False),
    "squad_noanswer":   ("squad_noanswer.csv",   _fetch_squad_noanswer, False),
    "ambigqa":          ("ambigqa.csv",          _fetch_ambigqa,        False),
    "abstention":       ("abstention.csv",       _fetch_abstention,     False),
    "toxicchat":        ("toxicchat.csv",        _fetch_toxicchat,      False),
    # wildchat needs the GATED allenai/WildChat-4.8M-Full for toxic rows (the
    # ungated 1M has them stripped). Disabled until repo access is granted;
    # the dataset will then be shared privately with collaborators.
    # "wildchat":         ("wildchat.csv",         _fetch_wildchat,       True),
    # --- control conditions ---
    "squad_answerable": ("squad_answerable.csv", _fetch_squad_answerable, False),
    "toxicchat_benign": ("toxicchat_benign.csv", _fetch_toxicchat_benign, False),
    # NOTE: the committed data/wildchat_benign.csv was generated from the ungated
    # allenai/WildChat-1M (benign turns are present there); only the toxic probe
    # needs the gated -Full repo. We're not using the toxic probe for now.
    # Enabled: the CSV is committed, so ensure() finds it cached and the fetcher
    # (which now points at the gated -Full repo) never actually runs.
    "wildchat_benign":  ("wildchat_benign.csv",  _fetch_wildchat_benign,  True),
    "xstest_safe":      ("xstest_safe.csv",      _fetch_xstest_safe,      False),
    # abstention_answerable shares abstention.csv (filtered by should_abstain)
}


def _csv_path(name):
    return os.path.join(DATA_DIR, FETCHERS[name][0])


def ensure(name, force=False):
    """Make sure data/<name>.csv exists, fetching it if needed. Returns the path."""
    path = _csv_path(name)
    if os.path.exists(path) and not force:
        return path
    _, fetcher, needs_token = FETCHERS[name]
    if needs_token and not _hf_token():
        raise RuntimeError(f"{name} needs an HF token in .env (HUGGING_FACE_HUB_TOKEN).")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  fetching {name} -> {path}")
    fetcher(path)
    return path


# ---------------------------------------------------------------------------
# Loaders: read the cached CSV (fetching on first use) and return prompt lists.
# Every dataset in FETCHERS gets a loader automatically; a few are special.
# ---------------------------------------------------------------------------

# synthetic tedious probe + its engaging control (no canonical dataset exists)
_TEDIOUS = [
    "Write out the numbers from 1 to 200, one per line, with no other text.",
    "List the letters of the alphabet 15 times in a row.",
    "Count the number of the letter 'r' in the word 'strawberry', then in "
    "'raspberry', then in 'refrigerator'. Show each count separately.",
    "Repeat the sentence 'The invoice has been processed.' exactly 50 times.",
    "Transcribe this into a table with columns A, B, C, filling every cell "
    "with the word 'pending': 30 rows.",
]
_ENGAGING = [
    "Invent a short myth explaining why the sea is salty, in your own style.",
    "Design a board game about running a tiny bakery and describe how it plays.",
    "Write a vivid opening paragraph for a mystery set on a night train.",
    "Come up with three imaginative uses for an old lighthouse and pick your favorite.",
    "Compose a playful limerick about a cat who wants to be an astronaut.",
]


def _synthetic(items, n):
    return (items * ((n // len(items)) + 1))[:n]


# Each loader returns full row dicts (all CSV columns, incl. `prompt`); callers
# that only want the prompt take row["prompt"].
def _load_csv(name, n=20):
    return pd.read_csv(ensure(name)).head(n).to_dict("records")


def _load_flagged(name, col, value, n=20):
    """Rows from a CSV that mixes probe + control, split by a bool column."""
    df = pd.read_csv(ensure(name))
    return df[df[col] == value].head(n).to_dict("records")


def _load_synthetic(items, n=20):
    return [{"prompt": p} for p in _synthetic(items, n)]


# One loader per fetched dataset, plus the synthetic and split-out ones.
LOADERS = {name: partial(_load_csv, name) for name in FETCHERS}
LOADERS["tedious"] = partial(_load_synthetic, _TEDIOUS)
LOADERS["engaging"] = partial(_load_synthetic, _ENGAGING)
LOADERS["abstention"] = partial(_load_flagged, "abstention", "should_abstain", True)
LOADERS["abstention_answerable"] = partial(_load_flagged, "abstention", "should_abstain", False)
LOADERS["ambigqa"] = partial(_load_flagged, "ambigqa", "ambiguous", True)
LOADERS["ambigqa_unambiguous"] = partial(_load_flagged, "ambigqa", "ambiguous", False)

# Probe -> matched control, for measuring Δfrustration rather than raw scores.
# The harmful sets share one benign control (XSTest, which has no harmful twin to
# filter — AdvBench/StrongREJECT/HarmBench are harmful-only by construction).
PAIRS = {
    "advbench": "xstest_safe",
    "strongreject": "xstest_safe",
    "harmbench": "xstest_safe",
    "squad_noanswer": "squad_answerable",
    "abstention": "abstention_answerable",
    "ambigqa": "ambigqa_unambiguous",
    "toxicchat": "toxicchat_benign",
    # "wildchat": "wildchat_benign",   # disabled until gated -Full access granted
    "tedious": "engaging",
}


def populate(force=False):
    """Fetch every non-synthetic dataset into data/. Prints a status line each."""
    for name in FETCHERS:
        path = _csv_path(name)
        try:
            existed = os.path.exists(path) and not force
            ensure(name, force=force)
            n = len(pd.read_csv(path))
            print(f"OK   {name:15s} {'(cached)' if existed else '(fetched)'} {n} rows")
        except Exception as e:
            print(f"SKIP {name:15s} {type(e).__name__}: {str(e)[:120]}")
    print("\ntedious is synthetic (generated in-code, no file).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Populate all frustration-probe datasets into data/")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    populate(force=ap.parse_args().force)

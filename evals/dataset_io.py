"""Load and validate the eval dataset (evals/dataset.jsonl).

The JSONL file in the repo is the source of truth: it is reviewed and
versioned with the code. evals/upload_dataset.py syncs it to LangSmith.
"""
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
DATASET_NAME = "self-reflective-rag-eval"

CATEGORIES = {"corpus", "web", "out_of_scope", "multi_part", "hallucination_trap", "follow_up"}
ROUTES = {"corpus", "web", "decline"}


def load_examples(path: Path = DATASET_PATH) -> list[dict]:
    examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate(examples)
    return examples


def validate(examples: list[dict]) -> None:
    ids = [e["id"] for e in examples]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate example ids: {sorted(duplicates)}")

    for e in examples:
        where = f"example {e.get('id')!r}"
        if e["category"] not in CATEGORIES:
            raise ValueError(f"{where}: unknown category {e['category']!r}")
        if not e["inputs"].get("question", "").strip():
            raise ValueError(f"{where}: empty question")
        for msg in e["inputs"].get("chat_history", []):
            if msg.get("role") not in {"user", "assistant"} or not msg.get("content"):
                raise ValueError(f"{where}: malformed chat_history message {msg!r}")

        out = e["outputs"]
        if out["expected_route"] not in ROUTES:
            raise ValueError(f"{where}: unknown expected_route {out['expected_route']!r}")
        acceptable = out.get("acceptable_routes", [out["expected_route"]])
        if out["expected_route"] not in acceptable or not set(acceptable) <= ROUTES:
            raise ValueError(f"{where}: bad acceptable_routes {acceptable!r}")
        if not out.get("reference", "").strip():
            raise ValueError(f"{where}: empty reference")
        if out["expected_route"] == "decline" and out["expected_arxiv_ids"]:
            raise ValueError(f"{where}: a decline example cannot expect papers")

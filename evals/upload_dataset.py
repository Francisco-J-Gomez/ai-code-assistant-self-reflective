"""Sync evals/dataset.jsonl to a LangSmith dataset (idempotent).

    uv run python -m evals.upload_dataset            # create / update
    uv run python -m evals.upload_dataset --dry-run  # show what would change

Examples are matched by their stable `id` (stored in example metadata as
`eval_id`): new ids are created, changed ones updated, unchanged ones left
alone. Examples removed from the JSONL are only deleted with --prune, so a
LangSmith-side edit is never lost by accident. Each example's category is
also its split, so experiments can be filtered per category in the UI.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from evals.dataset_io import DATASET_NAME, load_examples

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _metadata(example: dict) -> dict:
    return {"eval_id": example["id"], "category": example["category"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    parser.add_argument("--prune", action="store_true", help="delete LangSmith examples no longer in the JSONL")
    args = parser.parse_args()

    examples = load_examples()
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = {
            (ex.metadata or {}).get("eval_id"): ex
            for ex in client.list_examples(dataset_id=dataset.id)
        }
    else:
        dataset = None
        existing = {}

    to_create, to_update = [], []
    for e in examples:
        current = existing.get(e["id"])
        if current is None:
            to_create.append(e)
        elif (
            current.inputs != e["inputs"]
            or current.outputs != e["outputs"]
            or (current.metadata or {}).get("category") != e["category"]
        ):
            to_update.append((current.id, e))
    local_ids = {e["id"] for e in examples}
    stale = [ex for eval_id, ex in existing.items() if eval_id not in local_ids]

    print(f"Dataset {DATASET_NAME!r}: {'exists' if dataset else 'will be created'}")
    print(f"  create: {len(to_create)}  update: {len(to_update)}  "
          f"unchanged: {len(examples) - len(to_create) - len(to_update)}  "
          f"not in JSONL: {len(stale)}{' (will delete)' if args.prune else ' (kept; use --prune)'}")
    if args.dry_run:
        return

    if dataset is None:
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Self-reflective RAG eval set: corpus, web, out-of-scope, multi-part, "
                        "hallucination-trap and follow-up questions. Source of truth: evals/dataset.jsonl.",
        )

    if to_create:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {"inputs": e["inputs"], "outputs": e["outputs"], "metadata": _metadata(e), "split": e["category"]}
                for e in to_create
            ],
        )
    for example_id, e in to_update:
        client.update_example(
            example_id, inputs=e["inputs"], outputs=e["outputs"], metadata=_metadata(e), split=e["category"]
        )
    if args.prune and stale:
        client.delete_examples([ex.id for ex in stale])

    print(f"Done: {dataset.url}")


if __name__ == "__main__":
    main()

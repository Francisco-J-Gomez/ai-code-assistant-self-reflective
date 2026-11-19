"""Run the eval set against the agent and record a LangSmith experiment.

    uv run python -m evals.run_eval                                # full run, 40 examples
    uv run python -m evals.run_eval --limit 3                      # smoke test
    uv run python -m evals.run_eval --split out_of_scope --split web

Each run becomes an experiment on the `self-reflective-rag-eval` dataset in
LangSmith; compare experiments there (select two -> Compare). A per-category
summary is also printed here.
"""
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langsmith import Client, evaluate  # noqa: E402

from agent.config import AgentConfig  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from evals.dataset_io import DATASET_NAME  # noqa: E402
from evals.evaluators import CODE_EVALUATORS, make_llm_judges  # noqa: E402
from evals.targets import make_graph_target  # noqa: E402

SUMMARY_KEYS = ["route_correct", "retrieval_hit", "retrieval_recall", "correctness", "groundedness", "llm_calls", "ktokens"]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _print_summary(results) -> None:
    by_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: dict[str, list[float]] = defaultdict(list)
    errors = 0
    for row in results:
        category = (row["example"].metadata or {}).get("category", "?")
        run = row["run"]
        if run.error:
            errors += 1
        if run.end_time and run.start_time:
            latencies[category].append((run.end_time - run.start_time).total_seconds())
        for r in row["evaluation_results"]["results"]:
            if r.score is not None:
                by_category[category][r.key].append(float(r.score))
                by_category["ALL"][r.key].append(float(r.score))
        if run.end_time and run.start_time:
            latencies["ALL"].append((run.end_time - run.start_time).total_seconds())

    def avg(xs):
        return f"{sum(xs) / len(xs):.2f}" if xs else "-"

    headers = ["category", "n", *SUMMARY_KEYS, "latency_s"]
    rows = []
    for category in sorted(c for c in by_category if c != "ALL") + ["ALL"]:
        metrics = by_category[category]
        n = len(latencies[category])
        rows.append([category, str(n), *(avg(metrics[k]) for k in SUMMARY_KEYS), avg(latencies[category])])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print()
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if errors:
        print(f"\n{errors} example(s) errored - see the experiment in LangSmith.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment-prefix", default="phase3-crag")
    parser.add_argument(
        "--description",
        default="Phase 3: corrective RAG - grade_docs, one rewrite_query retry, fallback_research sub-agent (web + live arXiv).",
    )
    parser.add_argument("--grader-model", default="gpt-5", help="LLM judge model (should differ from / be stronger than the agent's)")
    parser.add_argument("--split", action="append", help="only run this category (repeatable)")
    parser.add_argument("--limit", type=int, help="only run the first N examples (smoke test)")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    client = Client()
    agent_config = AgentConfig()
    data = list(client.list_examples(dataset_name=DATASET_NAME, splits=args.split, limit=args.limit))
    print(f"Running {len(data)} example(s) from {DATASET_NAME!r} (agent={agent_config.llm_model}, grader={args.grader_model})")

    results = evaluate(
        make_graph_target(build_graph(agent_config)),
        data=data,
        evaluators=[*CODE_EVALUATORS, *make_llm_judges(args.grader_model)],
        experiment_prefix=args.experiment_prefix,
        description=args.description,
        metadata={
            "phase": args.experiment_prefix,
            "git_sha": _git_sha(),
            "agent_model": agent_config.llm_model,
            "grader_model": args.grader_model,
            "retrieval_top_k": agent_config.retrieval_top_k,
            "doc_grader_model": agent_config.grader_model,
            "min_relevant_docs": agent_config.min_relevant_docs,
            "max_relevant_docs": agent_config.max_relevant_docs,
            "max_query_rewrites": agent_config.max_query_rewrites,
            "splits": args.split or "all",
        },
        max_concurrency=args.concurrency,
        client=client,
    )
    _print_summary(results)
    print(f"\nExperiment: {results.experiment_name}")


if __name__ == "__main__":
    main()

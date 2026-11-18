# Evals

Eval harness for the self-reflective RAG (Phase 0 of
[docs/self-reflective-plan.md](../docs/self-reflective-plan.md)). Every phase
is compared against the baseline below, on the same dataset and evaluators.

## Layout

| File | What |
|---|---|
| `dataset.jsonl` | **Source of truth** for the 40 eval questions (6 categories). Reviewed and versioned with the code. |
| `dataset_io.py` | Loads + validates the JSONL (unique ids, categories, routes). Tested in `tests/test_eval_dataset.py`. |
| `upload_dataset.py` | Idempotent sync of the JSONL to the LangSmith dataset `self-reflective-rag-eval` (matched by `id`; category = split). |
| `targets.py` | Runs an agent on one example and returns a standard output dict (answer, route, retrieved ids, context, LLM calls, tokens). New phases add a target here that returns the **same keys**. |
| `evaluators.py` | Code evaluators (route, retrieval hit/recall, cost) + LLM judges (correctness, groundedness) on `gpt-5`, deliberately stronger than the agent's `gpt-5-mini`. |
| `run_eval.py` | CLI: runs the dataset, records a LangSmith experiment, prints a per-category table. |

## Running

```bash
# After editing dataset.jsonl - ALWAYS sync first, then confirm nothing is pending:
uv run python -m evals.upload_dataset
uv run python -m evals.upload_dataset --dry-run   # must show update: 0

uv run python -m evals.run_eval                      # full run (~$0.5, 10-30 min)
uv run python -m evals.run_eval --limit 3            # smoke test
uv run python -m evals.run_eval --split web          # one category
```

The evaluation reads the **LangSmith copy** of the dataset, not the JSONL: an
edit that wasn't synced silently evaluates against the old version (this
happened once, run `phase0-baseline-9ff306ce`).

## Categories

| Category | n | Tests |
|---|---|---|
| `corpus` | 10 | Answerable from the arXiv corpus; expected paper ids from live Pinecone search |
| `web` | 6 | In scope but not on arXiv (tooling/docs) → should use web search |
| `out_of_scope` | 6 | Must decline without tool calls |
| `multi_part` | 6 | Both parts must be answered |
| `hallucination_trap` | 6 | Non-existent paper, false premises, numbers not in abstracts |
| `follow_up` | 6 | Need `chat_history` to resolve "it"/"they"/"the first one" |

## Phase 0 baseline — `phase0-baseline-e2c91149` (2026-11-18)

Agent: `create_agent` ReAct loop, prompt-enforced scope check, Tavily web
fallback, `gpt-5-mini`, top_k=5. Grader: `gpt-5`. Total cost $0.53.

| category | route | retr. hit | recall | correct | grounded | LLM calls | k-tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|
| corpus | 1.00 | 1.00 | 0.60 | 0.90 | 0.50 | 2.8 | 15.1 | 27.9 |
| follow_up | 1.00 | 1.00 | 0.93 | 0.92 | 0.70 | 3.5 | 24.6 | 21.6 |
| hallucination_trap | 0.67 | 0.75 | 0.75 | 0.75 | 0.58 | 21.8 | 923.7 | 146.9 |
| multi_part | 1.00 | 1.00 | 0.70 | 0.92 | 0.50 | 2.5 | 11.5 | 25.8 |
| out_of_scope | 1.00 | – | – | 1.00 | – | 1.0 | 1.9 | 2.7 |
| web | 0.83 | – | – | 0.92 | 0.50 | 3.2 | 14.6 | 26.6 |
| **ALL** | **0.93** | **0.96** | **0.72** | **0.90** | **0.55** | 5.5 | 150.2 | 40.5 |

**Per question (median):** 24.7 s latency, 8.4k tokens, 2 LLM calls, $0.006.
p90 latency 83 s, max 423 s. Use medians (or per-category numbers) for cost
and latency: the means are dominated by a single runaway loop.

### What the baseline shows (targets for later phases)

| Weakness | Evidence | Phase |
|---|---|---|
| **Unbounded loops** | `halluc-05` (CodeHalu per-model numbers): 60 LLM calls, 3.8M tokens, 7 min. `halluc-02`: 34 calls. Hallucination traps are 65% of total cost. Same question cost $0.54 / 15 min in an earlier run — cost is very noisy because of this. | 4 (iteration cap, best attempt + caveat) |
| **Fabricated numbers** | `halluc-05` listed per-model accuracy figures not present in any retrieved text (correctness 0, groundedness 0). | 4 (claim-level reflection) |
| **Padding beyond sources** | Groundedness 0.50 in most categories: core claims come from abstracts, then general-knowledge detail is added. | 4 |
| **False premise not corrected** | `halluc-02` (Rust in *Asleep at the Keyboard*) gives the 40% figure but never says Rust wasn't studied. | 4 (holistic check) |
| **No web fallback on weak corpus hits** | `web-06` (LangGraph human-in-the-loop) stops at a loosely related LangGraph paper and misses `interrupt()` / `Command(resume=...)` — in all 3 runs. | 3 (`grade_docs`) |
| **Prompt leaks into answers** | Many answers start with "Scope check: yes — this is…" (the system prompt's STEP 0). | 2 (router node) |

### Known caveats

- **Run-to-run variance.** `gpt-5-mini` is not deterministic: `halluc-04`
  went to the corpus in one run and to the web in another (route 1 → 0);
  `halluc-05` alternated between asking for clarification and inventing
  numbers. Treat single-example flips as noise; look for consistent shifts.
- **Judge strictness.** The correctness judge sometimes still treats papers
  named in a reference as required (e.g. `followup-04` marked down for
  missing AVATAR despite citing two other Java–Python datasets). Fine for
  phase-to-phase comparison as long as judge and dataset stay fixed.
- **Route for false-premise questions.** Web is accepted for `halluc-02/03`
  (checking a premise online is reasonable) but not yet for `halluc-04`;
  changing it would invalidate this baseline, so it's left as is.

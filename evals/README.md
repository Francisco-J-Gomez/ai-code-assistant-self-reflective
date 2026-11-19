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

## Phase 1 — `phase1-graph-d325de4c` (2026-11-18)

Agent: explicit `StateGraph` (`scope_check → retrieve_corpus → check_corpus
→ [web_search] → generate`), no loop, one corpus search per turn,
`gpt-5-mini`, top_k=5. Same grader and dataset as the baseline.

| category | route | retr. hit | recall | correct | grounded | LLM calls | k-tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|
| corpus | 0.90 | 1.00 | 0.49 | 0.85 | 0.60 | 3.0 | 6.8 | 57.7 |
| follow_up | 1.00 | 1.00 | 0.87 | 0.92 | 0.60 | 2.7 | 4.9 | 59.7 |
| hallucination_trap | 0.67 | 0.50 | 0.30 | 0.92 | 1.00 | 3.0 | 6.3 | 25.3 |
| multi_part | 0.83 | 1.00 | 0.51 | 0.83 | 0.50 | 3.0 | 6.8 | 74.3 |
| out_of_scope | 1.00 | – | – | 1.00 | – | 1.0 | 0.4 | 1.9 |
| web | 0.83 | – | – | 0.83 | 0.67 | 3.0 | 6.4 | 69.4 |
| **ALL** | **0.88** | **0.92** | **0.54** | **0.89** | **0.67** | 2.65 | 5.4 | 49.0 |

**Per question (median):** 28.6 s latency. Mean latency is inflated by two
OpenAI stalls (`check_corpus` 254 s, `generate` 258 s, one of them in
`corpus-06`). Per node (median): scope_check 3.3 s, retrieve_corpus 0.6 s,
check_corpus 5.5 s, web_search 2.7 s, generate 17.8 s.

### What changed vs. the baseline

| | Result | Why |
|---|---|---|
| **Cost** | Mean tokens 150k → 5.4k per question; hallucination traps 924k → 6.3k; max 3 LLM calls | No loop: the `halluc-05` runaway (60 calls, 3.8M tokens) is impossible by construction |
| **Quality** | Correctness 0.90 → 0.89 (unchanged); groundedness 0.55 → 0.67; hallucination-trap correctness 0.75 → 0.92 | Narrow generate prompt (documents only, point out false premises, no process narration); no "Scope check" leak |
| **Recall** | 0.72 → 0.54 (corpus 0.60 → 0.49, multi_part 0.70 → 0.51, `multi-03` 0.14) | One search with the rewritten query; the ReAct agent often searched several times. Multi-part questions need one query per part → Phase 3/5 (`rewrite_query`, sub-queries) |
| **Route** | 0.93 → 0.88. `corpus-01`, `multi-04`, `halluc-04`, `halluc-05` went to web (answers still correct: generate sees the corpus papers too); `web-06` still went to corpus | `check_corpus` is one yes/no judgement over 5 abstracts — noisy in both directions. `web-06` without "interrupt()" in the question still matches the LangGraph paper's abstract. → Phase 3 (`grade_docs`, per-document relevance) |

Conclusion: the refactor met its goal (same answer quality, bounded cost);
the remaining gaps are the ones Phase 3 targets.

## Phase 2 — `phase2-router-3a51a33b` (2026-11-18)

Agent: `contextualize → router (corpus | web | out_of_scope) →
[retrieve_corpus → check_corpus] → [web_search] → generate`. Same grader and
dataset. Final Phase 2 code, including the `generate` fixes (source label
set by the graph, no closing offers, say plainly when a paper isn't found).

| category | route | retr. hit | recall | correct | grounded | LLM calls | k-tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|
| corpus | 1.00 | 1.00 | 0.54 | 0.75 | 0.55 | 3.0 | 6.7 | 47.2 |
| follow_up | 1.00 | 1.00 | 0.77 | 1.00 | 0.80 | 3.7 | 5.4 | 34.4 |
| hallucination_trap | 0.67 | 0.50 | 0.30 | 0.83 | 0.67 | 3.0 | 6.6 | 29.9 |
| multi_part | 0.83 | 1.00 | 0.59 | 0.92 | 0.50 | 3.0 | 7.6 | 60.2 |
| out_of_scope | 1.00 | – | – | 1.00 | – | 1.0 | 0.5 | 5.1 |
| web | 1.00 | – | – | 0.92 | 0.92 | 2.2 | 4.0 | 32.0 |
| **ALL** | **0.93** | **0.92** | **0.56** | **0.89** | **0.67** | 2.67 | 5.3 | 36.0 |

An earlier run of Phase 2 *before* the `generate` fixes,
`phase2-router-eb25370e`: ALL route 0.93, correctness 0.89, groundedness
0.70, hallucination-trap correctness 0.75, corpus correctness 0.85, median
latency 22.5 s.

Between the two runs:

- **Hallucination traps 0.75 → 0.83** — the targeted fix (`halluc-01` 0.5 →
  1.0: now says plainly the paper wasn't found).
- **Corpus correctness 0.85 → 0.75 is noise**: same retrieved papers (first-
  turn search query = the question, Pinecone is deterministic); the judge
  marked `corpus-02` / `corpus-06` down for omitting RepoCoder / RLCoder /
  TIPICAL, which weren't retrieved in either run. Judge noise also shows in
  `halluc-05`: correctness 1.0 and groundedness 0 for the same answer.
- **Latency** up across every category, including `out_of_scope` (a single
  router call: 2.3 → 5.1 s) → OpenAI API latency, not the code.

vs. Phase 1 (`phase1-graph-d325de4c`): route 0.88 → 0.93 (every web question
routed correctly, including `web-06`; remaining misses are `check_corpus`
fallbacks: `multi-04`, `halluc-04`, `halluc-05`), follow-up correctness
0.92 → 1.00, correctness 0.89 → 0.89. Details:
[docs/phases/phase-2-router.md](../docs/phases/phase-2-router.md#results).

**Evaluator caveat:** `outputs_from_state` gives the judge corpus papers as
`[arxiv_id] title` without the URL, so cited paper URLs are flagged as
unsupported. Affects every phase equally; fix together with a baseline
re-run.

## Phase 3 — `phase3-crag-1c490aad` (2026-11-19)

Agent: `contextualize → router → decompose → corpus_search ×N (retrieve →
grade_docs, top 3 relevant per sub-query → rewrite once) → merge_results →
generate | fallback_research (ReAct: web + live arXiv)`. Same grader and
dataset. `min_relevant_docs = 1`, `max_relevant_docs = 3`,
`max_query_rewrites = 1`, `max_sub_queries = 3`.

| category | route | retr. hit | recall | correct | grounded | LLM calls | k-tokens | latency (s) |
|---|---|---|---|---|---|---|---|---|
| corpus | 1.00 | 1.00 | 0.45 | 0.80 | 0.55 | 4.1 | 6.9 | 51.0 |
| follow_up | 1.00 | 1.00 | 0.83 | 0.92 | 0.70 | 4.7 | 5.9 | 41.5 |
| hallucination_trap | 0.83 | 0.50 | 0.30 | 0.83 | 0.75 | 7.8 | 11.8 | 68.7 |
| multi_part | 1.00 | 1.00 | 0.70 | 0.92 | 0.67 | 5.0 | 10.9 | 62.5 |
| out_of_scope | 1.00 | – | – | 1.00 | – | 1.0 | 0.5 | 3.4 |
| web | 1.00 | – | – | 0.83 | 0.83 | 2.0 | 3.4 | 30.0 |
| **ALL** | **0.97** | **0.92** | **0.56** | **0.88** | **0.68** | 4.10 | 6.6 | 43.7 |

**Per question (median):** 47.8 s latency.

vs. Phase 2 (`phase2-router-3a51a33b`): route 0.93 → 0.97 (`multi-04`,
`halluc-04` fixed; only `halluc-05` left), multi-part recall 0.59 → 0.70,
correctness 0.89 → 0.88 and groundedness 0.67 → 0.68 (flat; the unchanged
web route moved by as much), LLM calls 2.7 → 4.1. Corpus recall 0.54 →
0.45 is `decompose` rephrasing single-part questions differently per run —
a retrieval-only check shows no systematic loss (0.54 vs 0.53). Details:
[docs/phases/phase-3-corrective-rag.md](../docs/phases/phase-3-corrective-rag.md#results).

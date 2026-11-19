# Phase 3 — Corrective RAG

Branch `phase-3-corrective-rag`. Steps 13–16 of
[the self-reflective RAG plan](../self-reflective-plan.md), plus query
decomposition (step 24, brought forward from Phase 5). Goal: stop trusting
whatever the corpus search returns. Split the question into its parts;
for each part, grade every retrieved paper and keep the best relevant ones;
when too few are relevant, rewrite the query once; parts still not covered
go to a bounded research sub-agent with web and live-arXiv search.

## Before → after

| | Phase 2 | Phase 3 |
|---|---|---|
| Multi-part questions | One search for the whole question | `decompose` → one search per part, in parallel (`Send`) |
| Relevance check | `check_corpus`: one yes/no over 5 abstracts — noisy both ways | `grade_docs`: one grade per paper, against its sub-query |
| What `generate` sees | All 5 retrieved papers (+ web) | Up to 3 best-ranked relevant papers **per part** (+ web / live arXiv) |
| Weak search | Straight to the web | `rewrite_query` (abstract vocabulary) → search again; only new chunks are graded |
| Fallback | Tavily web search, whole question | `fallback_research`: ReAct sub-agent (web + live arXiv), hard-capped, told which parts are missing |
| Papers newer than the corpus | Not findable | Live arXiv search |
| Config | Models, top_k | + `grader_model`, `max_sub_queries`, `min_relevant_docs`, `max_relevant_docs`, `max_query_rewrites`, research call limits |

## The graph

```
START -> contextualize -> router --out_of_scope--> decline --> END
                            |  |
                            |  +--web (tool / library / docs)--> web_search ----------+
                            | corpus (research)                                       |
                            v                                                         |
                       decompose --Send, one per sub-query, in parallel-->            |
                                                                                      |
     corpus_search (subgraph, per sub-query):                                         |
         retrieve_corpus -> grade_docs (keep top 3 relevant) --enough--> report       |
              ^                  | too few                                |           |
              +-- rewrite_query <+ (once; then report anyway)             |           |
                                                                          v           v
                                merge_results --all parts covered----> generate --> END
                                      | some part not covered            ^
                                      v                                  |
                              fallback_research (ReAct: web + live arXiv)+
```

In Studio the `corpus_search` subgraph is shown nested; in LangSmith each
sub-query is its own child run.

## Design decisions

- **Decompose first, then correct per part.** Grading against the whole
  question with a 3-paper cap would starve multi-part questions (3 papers
  for two topics). Per sub-query, each part gets its own retrieval, grading
  against *that part*, top 3 and retry. `decompose` keeps single-part
  questions whole (one extra LLM call, ~3 s) and is told not to split a
  topic into facets.
- **A subgraph with an output schema.** `corpus_search` runs in parallel
  copies; only `sub_results` (one `SubQueryResult` each, append reducer)
  flows back, so the runs can't clash on their working fields. Results carry
  their index and are merged in sub-query order (parallel runs finish in any
  order). The per-turn reset clears the append-reducer field with
  `Overwrite([])`.
- **A sub-query is covered by 1 relevant paper** (`min_relevant_docs = 1`):
  a part about one specific paper is answered by that paper alone; parts the
  corpus can't answer get 0 relevant papers in practice (every trap did).
- **Retrieve 5, keep at most 3 per sub-query** (`max_relevant_docs`). The
  grader is lenient (it kept 4–5 of 5 on every ordinary corpus question in
  an earlier trace), so the cap is what makes grading change what
  `generate` sees. Retrieval still fetches 5 so the grader has a choice.
- **One grading call per sub-query** (`DocGrades`: index, reason, relevant
  per document): same signal as one call per document, at 1/5 of the calls.
  The reasons show in LangSmith traces.
- **"Relevant" includes "shows the premise is wrong"** — the paper a false
  question is about is useful evidence for correcting it.
- **Retries only grade new chunks**, so a re-search returning already-seen
  chunks costs no grading call and can't double-count.
- **ReAct only where the next step depends on the last result.** The main
  flow stays a fixed graph; the fallback is open-ended (search arXiv by
  title → not found → check the web whether it exists), so it is a
  `create_agent` sub-agent. `ModelCallLimitMiddleware` (4) and
  `ToolCallLimitMiddleware` (4) make the Phase 0 runaway (60 calls)
  impossible.
- **The sub-agent searches, it doesn't answer.** Its tools return
  `Document`s as artifacts for the main `generate` node, so citation,
  grounding and source-labelling rules are the same on every path. It gets
  the whole question for context but searches only the missing parts.
- **New `Document.kind="arxiv"`** for live-arXiv papers: cited by arXiv id
  like corpus papers, labelled as a live search, not counted as corpus
  retrieval in the evals.
- **Evals measure retrieval before grading.** `retrieved_ids` comes from
  `retrieved` (everything every sub-query's search returned), so
  `retrieval_hit` / `recall` stay comparable with Phases 0–2; the judge's
  context is what `generate` actually saw.
- **`settings`, not `config`**, for the `AgentConfig` passed to nodes:
  LangGraph injects its own `RunnableConfig` into a parameter named
  `config`.
- **Deferred: self-updating corpus** (pushing papers found live into
  Pinecone). It needs its own quality gate, and would change the corpus
  while evals run.

## Changed files

- `agent/research.py` (new): the sub-agent, its tools and limits.
- `ingestion/clients/arxiv_client.py`: `search_arxiv(query)` — keyword
  search sorted by relevance, reusing the client's retries and parsing.
- `agent/graph.py`: `decompose` → `Send` fan-out → `corpus_search` subgraph
  (`build_corpus_search()`) → `merge_results` → `generate` /
  `fallback_research`; `check_corpus` removed.
- `agent/nodes.py`: `decompose`, `fan_out_sub_queries`, `merge_results`,
  `fallback_research`; subgraph nodes `retrieve_corpus`, `grade_docs`,
  `rewrite_query`, `report`.
- `agent/state.py`: `CorpusSearchState`, `CorpusSearchOutput`,
  `SubQueryResult`; `AgentState` gets `sub_queries`, `sub_results`,
  `missing`, `retrieved`; `Document.kind` adds `arxiv`, `Document.key`.
- `agent/schemas.py`: `SubQueries`, `DocGrades` (replaces `CorpusCheck`).
- `agent/prompts.py`: `DECOMPOSE_PROMPT`, `GRADE_DOCS_PROMPT`,
  `REWRITE_QUERY_PROMPT`, `RESEARCH_PROMPT`; `source_note()` names web
  and/or arXiv.
- `agent/config.py`: Phase 3 knobs, validated (`min_relevant_docs ≤
  max_relevant_docs`). `agent/results.py`, `ui/chat_app.py`: live-arXiv
  sources.
- `evals/targets.py`: retrieval metrics from `retrieved`; `run_eval.py`:
  prefix `phase3-crag`, knobs in experiment metadata.
- Tests: `tests/test_nodes.py`, `tests/test_graph.py` (per-sub-query
  grading and cap, partial fallback, dedup across parts, rewrite bound,
  reset of the append-reducer field), `tests/test_research.py` (artifacts,
  dedup, call limits with a scripted model), `tests/test_arxiv_client.py`,
  `tests/test_eval_harness.py`, `tests/test_results.py`; integration tests
  assert the corrective path and a multi-part question.

## Pre-eval checks

- Unit tests: 135 passed. Integration tests: 6/6 passed.
- `decompose` on the 22 corpus-routed eval questions: all 6 multi-part
  questions → 2 sub-queries matching their parts; all 6 hallucination traps
  kept whole (splitting would hide a false premise); 2 corpus questions also
  split (`corpus-01` really asks two things; `corpus-04` "detecting or
  classifying" — harmless).
- Path trace on 29 eval questions (real stack, one run):
  - corpus: 3 relevant papers per sub-query, 4 LLM calls (5 when split),
    ~28–43 s;
  - multi-part: 5–6 relevant papers (two parts in parallel), 5 calls,
    ~40 s;
  - web: direct route, 2 calls, ~20 s;
  - traps: `halluc-04` stays on the corpus (expected); 01/02/03/05/06 →
    rewrite → research, 10 calls, 43–66 s;
  - off-route: `halluc-05` (as in Phase 2) and **`multi-01`** — its second
    part ("what does RLCoder add on top of plain retrieval?") found only
    the RLCoder paper itself relevant (1 < `min_relevant_docs` 2), so that
    part went to research. For a sub-query about one specific paper, one
    relevant paper is enough → **`min_relevant_docs` set to 1** (traps had
    0 relevant papers, so they still fall back).
- Live arXiv search finds real papers by title keywords (RepoCoder,
  CodeHalu) and nothing for the made-up "QuantumRepair".

## Results

Experiment `phase3-crag-1c490aad` (final settings, `min_relevant_docs = 1`)
vs. `phase2-router-3a51a33b` (full table in
[evals/README.md](../../evals/README.md#phase-3--phase3-crag-1c490aad-2026-11-19)):

| | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Route correct | 0.93 | 0.88 | 0.93 | **0.97** |
| Correctness | 0.90 | 0.89 | 0.89 | 0.88 |
| Groundedness | 0.55 | 0.67 | 0.67 | 0.68 |
| Retrieval recall | 0.72 | 0.54 | 0.56 | 0.56 |
| — multi-part recall | 0.70 | 0.51 | 0.59 | **0.70** |
| — corpus recall | 0.60 | 0.49 | 0.54 | 0.45 |
| Hallucination-trap route | 0.67 | 0.67 | 0.67 | **0.83** |
| LLM calls / question (mean) | 5.5 | 2.65 | 2.67 | 4.10 |
| k-tokens / question (mean) | 150.2 | 5.4 | 5.3 | 6.6 |
| Median latency | 24.7 s | 28.6 s | 36.5 s* | 47.8 s |

\* Phase 2's final run had slow OpenAI responses across the board.

What it shows:

- **Routing is the clearest gain: 0.97.** `multi-04` and `halluc-04`, sent
  to the web by Phase 2's yes/no `check_corpus`, now stay on the corpus;
  the only miss left is `halluc-05`. Per-paper grading fixed the noisy
  routing Phase 1 introduced.
- **Multi-part retrieval is back to the baseline** (recall 0.59 → 0.70):
  one search per part. A retrieval-only check (no judge) on the same
  questions: 0.59 → 0.76.
- **Corpus recall 0.54 → 0.45 is mostly `decompose` noise, not a
  systematic loss.** `decompose` sometimes "improves" a single-part
  question into a long list of facets (`corpus-06`: "...approaches
  (statistical, neural, deep learning), tasks (...), datasets and
  benchmarks, and evaluation results"), which pulls the vector search off
  target. A retrieval-only check over the 10 corpus questions: original
  question 0.54, `decompose` 0.53, "keep the original when there is one
  part" 0.50 — no strategy is better on average, so nothing was changed.
  Making `decompose` output more stable is retrieval work → Phase 5.
- **Answer quality is flat** (correctness 0.89 → 0.88, groundedness 0.67 →
  0.68). Per-example changes go both ways, and the web route — unchanged in
  this phase — moved too (correctness 0.92 → 0.83), which is the size of
  the run-to-run noise. Better *inputs* to `generate` didn't translate into
  better *answers*: the remaining failures are in how answers are written
  (omissions, unsupported details) → Phase 4 (reflection).
- **New failure mode (`halluc-05`, correctness 1 → 0):** the research
  sub-agent now finds the real CodeHalu results table, and `generate`
  reports a column of it as "accuracy" when it is a different metric.
  More evidence makes a misreading possible; checking claims against
  sources is Phase 4.
- **Cost:** LLM calls 2.7 → 4.1 per question (+1 `decompose` everywhere,
  ~8 on traps), tokens +25%, traps take ~70 s. The price of the
  corrective path; acceptable for a research assistant, but tracked for
  Phase 6.

Conclusion: Phase 3 fixed what it targeted — routing and multi-part
retrieval — at a ~50% higher LLM-call cost, without moving answer quality.
That points Phase 4 at the answer itself.

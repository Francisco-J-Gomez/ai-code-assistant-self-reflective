# Self-Reflective RAG — Step-by-Step Plan

Scope: evolve the original agent (a `create_agent` ReAct loop where the
scope check, corpus-first search and web fallback are all enforced only by
the system prompt — see [agent-plan.md](agent-plan.md)) into a
**self-reflective RAG**: an explicit LangGraph `StateGraph` that grades what
it retrieves, critiques its own answer, and iterates until the answer is
grounded and complete (or an iteration budget runs out).

Conceptually this combines two published patterns:

- **Corrective RAG (CRAG)** — grade retrieved documents *before*
  generating; rewrite the query or fall back to other sources if they are
  not good enough.
- **Self-RAG** — grade the generated answer *after* generating (is every
  claim supported? does it answer the whole question?) and loop back if not.

## Design principles

- **Measure before and after every phase** — a reflection loop easily
  triples cost and latency; only an eval dataset can show it actually
  improves answers. No phase is "done" until its eval run is compared
  against the previous one.
- **Control flow in code, not in the prompt** — routing, grading and
  retrying become nodes and conditional edges with Pydantic structured
  outputs, instead of instructions the LLM may or may not follow (the
  "STEP 0" scope-check bug from the MVP is the motivating example).
- **Bounded loops** — every loop has a hard iteration cap; when it is hit,
  return the best-scoring attempt (not the last one) with an honest caveat.
- **Don't let the model grade itself blindly** — use a stronger or at
  least differently-prompted model for grading than for generation.
- **Reuse existing code** — `ingestion/clients/arxiv_client.py` for live
  arXiv search, `Embedder` / `PineconeStore` for retrieval.

## Target graph

```
question ─► contextualize (standalone question from chat history)
              ▼
            router ──out_of_scope──► decline
              │
          (in scope)
              ▼
         [decompose?] ─► retrieve (hybrid, per sub-query, in parallel via Send)
                             ▼
                        grade_docs ──not enough──► rewrite_query ─► retrieve (×1)
                             │                          └─still not enough─► fallback_research (web + live arXiv)
                         enough                                                 │
                             ▼◄─────────────────────────────────────────────────┘
                         generate ◄──────────── revise (critique fed back) ◄──┐
                             ▼                                                │
                          reflect: split into claims → check each against docs│
                                   + does it answer the whole question?       │
                             ├── good ─────────────► END                       │
                             ├── unsupported claims ──────────────────────────┘
                             └── missing aspects ─► targeted retrieve ─► generate
                         (max N iterations, then return best attempt + caveat)
```

## Phase 0 — Observability and baseline (no behavior change)

1. **Verify LangSmith tracing** — `LANGSMITH_*` vars are already in `.env`
   but traces have never been confirmed in the UI (see TODO.md).
2. **Add `langgraph.json`** pointing at the graph factory, add
   `langgraph-cli[inmem]` as a dev dependency, and run `langgraph dev` to
   inspect the current agent in **LangGraph Studio** (local debugging).
   LangSmith covers tracing and evals (local and prod).
3. **Build an eval dataset in LangSmith** (~30–50 questions):
   - answerable from the corpus;
   - only answerable from the web / recent papers;
   - out of scope;
   - multi-part questions;
   - questions likely to trigger hallucination;
   - follow-up questions that depend on chat history.
4. **Baseline the current agent** on it. Evaluators: correctness,
   groundedness (faithfulness to retrieved sources), retrieval relevance,
   routing accuracy, plus latency, token cost and number of LLM calls.

## Phase 1 — Explicit graph, same behavior (refactor)

5. **Pydantic state** — `messages`, `question`, `standalone_question`,
   `sub_queries`, `documents`, `draft`, `critique`, `iteration`,
   `best_draft`, `sources`.
6. **Re-implement the current behavior as nodes**: `router → retrieve →
   generate`, with a web fallback. Keep the `InMemorySaver` checkpointer
   (per-thread conversation memory; `MemorySaver` is the same thing).
7. **Move source extraction to the state** — `agent/results.py` currently
   parses `ToolMessage`s named `search_papers` / `tavily_search`. Once
   retrieval is a node rather than a tool call those messages no longer
   exist, so `extract_sources` / `classify_answer_source` and
   `ui/chat_app.py` must read `sources` (and which path was taken) from the
   final graph state instead.
8. **New `AgentConfig` knobs** — `max_iterations`, `grader_model`,
   relevance/grounding thresholds, etc.
9. **Tests** — unit-test each node with a fake LLM (same mocked, no-network
   convention as `tests/`); update `integration_tests/test_agent_integration.py`.
10. **Re-run evals** — scores should roughly match the baseline: this phase
    is a refactor, not a feature.

## Phase 2 — Contextualize + router

11. **`contextualize` node** — rewrite the latest user message into a
    standalone question using the chat history, so follow-ups ("what are
    its limitations?") route and retrieve correctly.
12. **`router` node** — structured output
    (`route: corpus | web | out_of_scope`). Replaces the prompt-only "STEP 0"
    scope check with a branch enforced in code.

## Phase 3 — Corrective RAG

13. **`grade_docs`** — per-document relevance grading with structured
    output; keep only relevant documents.
14. **If not enough relevant documents** → `rewrite_query` and retrieve
    once more → if still not enough → `fallback_research`.
15. **Live arXiv tool** — reuse `ingestion/clients/arxiv_client.py`.
    Optional bonus: push papers found live through the ingestion pipeline,
    so the corpus self-updates instead of going stale.
16. **`fallback_research` as a small ReAct sub-agent** (`create_agent` with
    web + arXiv tools). This is where a ReAct agent belongs in the flow, and
    where built-in middleware can be used (see Phase 6).

## Phase 4 — Self-reflection loop (the core feature)

17. **`reflect` node, claim-level grounding** — decompose the draft into
    atomic claims and verify each one against the retrieved documents
    (supported / unsupported). This is answer verification, distinct from
    query decomposition.
18. **Holistic check** — does the answer address every part of the
    question?
19. **Pydantic `Critique`** — `grounded`, `unsupported_claims`,
    `missing_aspects`, `score`.
20. **Routing on the critique**:
    - good → END;
    - unsupported claims → `revise` (regenerate with the critique);
    - missing aspects → targeted retrieval for them → `generate`.
21. **Cap at `max_iterations` (~3)**; track `best_draft` by score and
    return it with a caveat if no attempt passes.
22. **Grader ≠ generator** — stronger model or clearly different prompt for
    `grade_docs` / `reflect`.
23. **Re-run evals vs Phase 1** — groundedness should go up; track the
    cost/latency increase explicitly.

## Phase 5 — Retrieval quality

24. **Query decomposition** — `decompose` node, only for multi-part
    questions, fan-out in parallel with `Send`, merge documents. Reflection's
    `missing_aspects` feed the same mechanism.
25. **Hybrid search** — Pinecone sparse-dense (BM25/SPLADE) + a reranker
    (e.g. Cohere rerank or a cross-encoder). Decide with evals, not by eye.
    The embedding/chunking benchmarks already listed in TODO.md fit here.

## Phase 6 — Production concerns

26. **Long conversations** — summarization of older messages; `SqliteSaver`
    (or Postgres) instead of `InMemorySaver` if memory must survive restarts.
27. **Middleware caveat** — the
    [built-in middlewares](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
    (Summarization, PII, model/tool call limits, model fallback) attach to
    `create_agent`, not to a custom `StateGraph`. Use them in the
    `fallback_research` sub-agent; implement equivalents in the main graph
    as nodes or with LangGraph-native features (recursion limit, etc.).
28. **Guardrails** — PII check on input; treat web content as untrusted
    (prompt injection). Keep it light: the domain is public arXiv papers.
29. **Streamlit UI** — stream graph updates so users see the process
    ("grading sources… revising answer (2/3)…").
30. **Docs** — update the README mermaid diagram and `agent/README.md`.

## Deliberately skipped / deferred

- **Supervisor / multi-agent system** — the workflow is mostly
  deterministic; a supervisor adds nondeterminism, cost and debugging pain
  without improving answers. The single ReAct sub-agent for fallback is
  enough.
- **Multimodal RAG** — the corpus is abstracts only (no PDF parsing yet),
  so there is nothing multimodal to retrieve.
- **Cache** — not the same as memory (memory = one thread's history; cache =
  reusing identical calls across threads). Low priority; LangGraph's
  node-level `CachePolicy` on retrieval/grading may be most useful for
  cutting the cost of repeated eval runs.

## Critical path

Phase 0 (baseline evals) → step 7 (sources from state) → step 12 (router)
→ steps 13–14 (doc grading) → steps 17–21 (reflection loop). Everything
else is incremental once the loop is shown to improve eval scores.

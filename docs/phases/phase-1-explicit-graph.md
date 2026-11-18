# Phase 1 — Explicit graph (refactor)

Branch `phase-1-explicit-graph`. Step 1 of
[the self-reflective RAG plan](../self-reflective-plan.md) after the Phase 0
baseline. Goal: replace the `create_agent` ReAct loop with an explicit
LangGraph `StateGraph` that enforces the **same policy** in code, without
changing answer quality. The self-reflective features (grading, rewriting,
reflection) need nodes and edges to attach to; this phase builds them.

## Before → after

| | Phase 0 (ReAct) | Phase 1 (StateGraph) |
|---|---|---|
| Control flow | System prompt ("STEP 0: scope check…", "search the corpus first…") — the LLM may or may not follow it | Graph edges — enforced in code |
| Loop | Unbounded tool-calling loop | None — at most 3 LLM calls per turn |
| Scope check | Prompt step; its text leaked into answers ("Scope check: yes…") | `scope_check` node with structured output |
| Follow-ups | The ReAct LLM saw the history and searched with whatever query it chose | `scope_check` rewrites the question as a standalone `search_query` |
| Web fallback | The LLM decided when to call Tavily | `check_corpus` node decides; corpus documents are kept on the web path |
| UI sources / badge | Parsed out of `search_papers` / `tavily_search` ToolMessages | Read from `documents` / `route` in the final state |
| Testability | Only end-to-end (real APIs) | Nodes and control flow unit-tested with fakes |

## The graph

```
START -> scope_check --out of scope--> decline --------------> END
             | in scope (+ standalone search query)
             v
      retrieve_corpus -> check_corpus --enough--> generate --> END
                             | not enough            ^
                             v                       |
                        web_search ------------------+
```

Node-by-node description: [agent/README.md](../../agent/README.md#how-the-agent-works).

## Code layout

```
agent/
  config.py       AgentConfig (models, top_k, web results)
  state.py        AgentState (graph state) + Document
  schemas.py      structured LLM outputs: ScopeDecision, CorpusCheck
  prompts.py      one narrow prompt per LLM node + decline message
  components.py   AgentComponents: every external call (OpenAI, Pinecone, Tavily)
  nodes.py        node functions (state, components) -> update, routing functions
  graph.py        wiring only: build_graph(), studio_graph()
  retrieval.py    make_corpus_search() + search_papers tool (for the Phase 3 sub-agent)
  results.py      sources list + answer badge for the UI, from final state
```

Design decisions:

- **Logic / wiring / I/O are separate.** Nodes (`nodes.py`) are plain
  functions; `graph.py` only declares nodes and edges; everything that
  touches the network is in `AgentComponents` and injected into
  `build_graph()`. Unit tests swap in fakes: `tests/test_nodes.py` tests
  each node and router alone, `tests/test_graph.py` tests the control flow.
- **Minimal state.** `AgentState` only has what Phase 1 reads or writes
  (`messages`, `question`, `search_query`, `route`, `documents`). The plan's
  Phase 4 fields (`draft`, `critique`, `iteration`, `best_draft`) are added
  when they are first used, not before.
- **Per-turn reset.** The checkpointer keeps state across turns of a
  thread, so `scope_check` resets the per-turn fields; otherwise an
  out-of-scope follow-up would show the previous turn's sources.
- **One `Document` shape** for corpus chunks and web results, so generation,
  the UI and the evals don't care where a source came from.
- **State vs. schemas.** `state.py` is what flows between nodes;
  `schemas.py` is what we ask the LLM to return. Phases 3–4 add `DocGrade`
  and `Critique` to `schemas.py`.
- **When `nodes.py` grows** (Phases 3–4 add ~6 nodes), split it into a
  `nodes/` package by concern (retrieval, grading, reflection).

## Other changes

- **`langgraph dev` / Studio:** `studio_graph()` is an async factory that
  builds the graph in a worker thread (Pinecone's connect is blocking I/O,
  which the server rejects on its event loop) and caches it (the server
  calls the factory per request).
- **Evals:** `evals/targets.py` runs the graph and returns the same output
  keys as the Phase 0 target, so experiments are comparable. LLM calls and
  tokens are counted with a callback (`UsageCounter`) instead of from
  message metadata. Default experiment prefix: `phase1-graph`.
- **Integration tests** assert on `route` / `documents` instead of tool
  calls, plus a new follow-up test (RepoCoder → RepoEval).
- **`CHECK_CORPUS_PROMPT`** judges only the retrieved text, not what the
  full paper probably contains. Found via the integration test: an abstract
  that *mentions* a LangGraph human-in-the-loop recipe was accepted as an
  answer to "how do I use `interrupt()`".
- **Removed:** `build_agent`, the ReAct system prompt and the ToolMessage
  parsing helpers. They remain on the `phase-0-observability-baseline`
  branch.

## Results

Experiment `phase1-graph-d325de4c` vs. baseline `phase0-baseline-e2c91149`
(full table and analysis in [evals/README.md](../../evals/README.md#phase-1--phase1-graph-d325de4c-2026-11-18)):

| | Baseline | Phase 1 |
|---|---|---|
| Correctness | 0.90 | 0.89 |
| Groundedness | 0.55 | **0.67** |
| Hallucination-trap correctness | 0.75 | **0.92** |
| Mean tokens / question | 150k | **5.4k** |
| Max LLM calls / question | 60 | **3** |
| Median latency | 24.7 s | 28.6 s |
| Route correct | 0.93 | 0.88 |
| Retrieval recall | 0.72 | 0.54 |

The refactor met its goal: same answer quality, bounded cost. Two known
regressions, both by design of a single-pass graph and both targeted by
Phase 3:

- **Recall** — one corpus search per turn (the ReAct agent often searched
  several times); multi-part questions suffer most. → `rewrite_query`,
  sub-queries.
- **Routing noise** — `check_corpus` is one yes/no call over 5 abstracts.
  → per-document `grade_docs`.

## How to verify

```bash
uv run pytest -q                                                              # unit tests
RUN_AGENT_INTEGRATION_TESTS=1 uv run pytest integration_tests/test_agent_integration.py -v
uv run langgraph dev                                                          # graph in Studio
uv run streamlit run ui/chat_app.py                                           # manual check
uv run python -m evals.run_eval                                               # new experiment
```

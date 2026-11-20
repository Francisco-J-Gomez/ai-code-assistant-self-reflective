# Phase 4 (UI) — Live progress in the chat

Branch `phase-4-ui`. Step 29 of
[the self-reflective RAG plan](../self-reflective-plan.md) ("stream graph
updates so users see the process"), brought forward from Phase 6 because
Phase 4 made answers take 1–4 minutes behind a single spinner.

## What the user sees

While the agent works, a status box under the question shows the **current
step** as its title and logs **each finished step with its outcome**:

```
📄 Research question → searching the paper corpus
Split into 2 parts: 1) “RAG for code completion” 2) “LLMs for test generation”
Part 1: found 5 new papers
Part 1: 3 of 5 relevant
Part 2: 0 of 5 relevant
Part 2: too few relevant papers → new query “…”
Not covered by the corpus: “LLMs for test generation”
🔎 Live search found 6 sources
Draft written
⚠️ Fact-check: 1 unsupported claim → revising
Draft 2 written
✅ Fact-check passed: 12 of 12 claims supported
```

When the answer arrives the box collapses into "How this answer was made ·
N steps · Ns"; it stays on every past answer of the conversation.

The **sidebar draws the agent's graph**, like `langgraph dev`, and colours
it live: blue = running (parallel sub-query runs can light up together),
green = done (`generate ×2` when a node ran more than once), grey = not
used this turn. After the answer it keeps showing that turn's path.

## Design decisions

- **`tasks` stream mode, with `subgraphs=True`.** `updates` mode only shows
  a subgraph's *output* field, so the steps inside `corpus_search` arrived
  empty. `tasks` gives each node's start event with its input (which part
  of the question a parallel run is on, how many papers it is grading) and
  its finish event with its update — enough for both the live label and
  the outcome line.
- **Parallel parts are labelled** ("Part 2: …"), so interleaved lines from
  the parallel `corpus_search` runs stay readable; single-part questions
  get no labels.
- **The wording lives in `agent/progress.py`, not in the UI.** A pure
  `ProgressTracker` consumes events and keeps `current` and `steps`; it is
  unit-tested against the real stream of the graph with fake components
  (`tests/test_progress.py`). The UI only renders it.
- **The diagram is drawn from the compiled graph**
  (`get_graph(xray=True)`, the same source as Studio and the README
  mermaid), so it can't drift from the code. `agent/graph_view.py` turns
  it into Graphviz DOT, coloured from `ProgressTracker.running` /
  `.visited`; Streamlit's built-in `st.graphviz_chart` renders it in the
  browser — no Graphviz install, no extra dependency. Graphviz rather than
  mermaid because Streamlit renders it natively and redraws it in place on
  every event. The DOT output was checked with a parser (`pydot`, not a
  project dependency), which caught an invalid cluster attribute.
- **The graph didn't change.** No node writes UI messages; the final state
  is read with `get_state()` after the stream, as `invoke` returned it
  before.
- **Fix found on the way:** LangGraph warned that it restores our own
  state types (`Document`, `SubQueryResult`, `Critique`, `ClaimCheck`) from
  checkpoints only if registered, and will refuse them in a future version
  — which would break multi-turn chats. They are now registered on the
  checkpointer's serializer (`make_checkpointer()` in `agent/graph.py`).

## Changed files

- `agent/progress.py` (new): `ProgressTracker` (step log, current step,
  running / visited node counters).
- `agent/graph_view.py` (new): `graph_dot()`.
- `ui/chat_app.py`: `_run_agent()` streams into `st.status` and the sidebar
  diagram; past answers keep a collapsed step log.
- `agent/graph.py`: `make_checkpointer()` with registered state types.
- `tests/test_progress.py` (new): corrective path with parallel parts and a
  revision, single-part labels, web route with the draft limit, follow-up
  and decline. `tests/test_graph_view.py` (new): every compiled node drawn,
  cluster and edges, colouring rules, a finished run's path, overlapping
  parallel runs.
- `README.md`, `agent/README.md`.

## Checks

- Unit tests: 176 passed.
- A real hallucination-trap question streamed through the tracker: the label
  changed every few seconds through routing, decomposition, grading (16 s),
  a query rewrite, drafting and the fact-check (28 s); 98 s in total.
- No eval: the agent's behaviour is unchanged.

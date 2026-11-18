# Phase 2 — Contextualize + router

Branch `phase-2-router`. Steps 11–12 of
[the self-reflective RAG plan](../self-reflective-plan.md). Goal: decide
**what the question is** and **where to look** up front, each in its own
node, before any retrieval.

## Before → after

| | Phase 1 | Phase 2 |
|---|---|---|
| Follow-ups | `scope_check` wrote a standalone *search query* as a side job of the scope decision | `contextualize` node writes a standalone *question*; only calls the LLM when there is history |
| Routing | In / out of scope only; every in-scope question searched the corpus first | `router`: `corpus` / `web` / `out_of_scope` |
| Tool / library / docs questions | Corpus search → `check_corpus` → web (2 wasted steps, and a false corpus match could stop it — `web-06`) | Straight to `web_search` |
| `check_corpus` input | Raw question ("did *they*…") — couldn't resolve references | Standalone question |
| LLM calls per turn | ≤ 3 | ≤ 4 on follow-ups, ≤ 3 otherwise, 2 on the web route |

## The graph

```
START -> contextualize -> router --out_of_scope--> decline --> END
                            |  |
                            |  +--web (tool / library / docs)----+
                            | corpus (research)                  |
                            v                                    v
                 retrieve_corpus -> check_corpus --not enough--> web_search
                                        | enough                 |
                                        v                        |
                                    generate <-------------------+ --> END
```

## Design decisions

- **Two nodes, not one.** Phase 1's `scope_check` did both jobs in one LLM
  call. Separating them gives each prompt one job, lets the router decide on
  the *standalone* question (so "how do I install it?" after a question
  about uv routes to web), and makes the first turn cheaper: `contextualize`
  skips the LLM when there is no history.
- **Asymmetric default: when unsure, `corpus`.** The corpus path can still
  fall back to the web (`check_corpus`); the web path never checks the
  corpus. A wrong `corpus` costs one extra step, a wrong `web` loses the
  papers.
- **`check_corpus` stays.** It handles research questions the corpus
  doesn't cover. Phase 3 replaces it with per-document `grade_docs`.
- **State:** `search_query` → `standalone_question` (the plan's name; it is
  used for routing, search and the corpus check). `generate` still gets the
  user's original `question` plus the conversation. `route` is the path
  taken so far: set by `router`, overwritten by `check_corpus` on fallback.
- **Config knobs** (`grader_model`, `max_iterations`, thresholds; plan step
  8) are left for Phase 3, where `grade_docs` is their first user — no
  settings without a consumer.

## Changed files

- `agent/schemas.py`: `ScopeDecision` → `RouteDecision` (`reasoning`, `route`).
- `agent/prompts.py`: `SCOPE_PROMPT` → `CONTEXTUALIZE_PROMPT` + `ROUTER_PROMPT`.
- `agent/components.py`: `scope_check` → `contextualize` + `route`.
- `agent/nodes.py`, `agent/graph.py`: new nodes and the 3-way edge.
- `agent/state.py`: `standalone_question`.
- Tests: `tests/test_graph.py` (web route, follow-up drives routing and
  search), `tests/test_nodes.py` (contextualize, router, routers);
  integration test asserts the tool question goes straight to the web.
- `evals/run_eval.py`: default prefix `phase2-router`.

## Pre-eval checks

- Router on all 40 eval questions (follow-ups through `contextualize` with
  their history), 3 runs: **40/40 acceptable route each time**.
- Unit tests: 106 passed. Integration tests: 5/5 passed (one per path).

## Results

Experiment `phase2-router-3a51a33b` (final code) vs. `phase1-graph-d325de4c`
(full table in [evals/README.md](../../evals/README.md#phase-2--phase2-router-3a51a33b-2026-11-18)):

| | Phase 0 | Phase 1 | Phase 2 |
|---|---|---|---|
| Route correct | 0.93 | 0.88 | **0.93** (corpus 1.00, web 1.00) |
| Correctness | 0.90 | 0.89 | 0.89 |
| Groundedness | 0.55 | 0.67 | 0.67 |
| Follow-up correctness | 0.92 | 0.92 | **1.00** |
| Hallucination-trap correctness | 0.75 | 0.92 | 0.83 |
| Retrieval recall | 0.72 | 0.54 | 0.56 |
| Web questions: LLM calls / k-tokens | 3.2 / 14.6 | 3.0 / 6.4 | **2.2 / 4.0** |

Latency isn't comparable in this run: OpenAI was slower across the board
(a single router call went from 2.3 s to 5.1 s between two Phase 2 runs).
The first Phase 2 run, `phase2-router-eb25370e` (before the `generate`
fixes), had a 22.5 s median vs. Phase 1's 28.6 s.

What it shows:

- **The router works.** All 6 web questions routed correctly, including
  `web-06`, which every previous run sent to the corpus. 5 of 6 skipped the
  corpus entirely (`web-04`, MCP, accepts either route). No corpus or
  follow-up question was misrouted to the web up front.
- **Remaining route misses are all `check_corpus`**, not the router:
  `multi-04`, `halluc-04`, `halluc-05` went corpus → web fallback. → Phase 3
  (`grade_docs`).
- **Hallucination traps**: the first run (0.75) showed `generate` problems
  on the fallback path — offering a web search that had already run, not
  correcting `halluc-03`'s false premise. The fixes below brought it to
  0.83; the rest is Phase 4 (holistic check).
- **Corpus correctness 0.85 (first run) vs 0.75 (final)** is noise: same
  retrieved papers, judge marked down omissions of papers that weren't
  retrieved in either run (`corpus-02`, `corpus-06`) — a recall problem →
  Phase 3/5.

Fixes to `generate` after the first run (found via the hallucination-trap answers
and the new fallback integration test):

- **Source label set by the graph.** The answer's "grounded in the paper
  corpus" / "includes a live web search" line was the model's choice, and it
  once called a web leaderboard "the paper corpus". Now `source_note()`
  picks it from the documents.
- **Prompt says "sources", not "documents"**, and each source is labelled
  "paper corpus" or "web search" — the model echoed the prompt's own
  "DOCUMENTS:" header back as "the provided documents".
- **Say plainly when a paper / model / result isn't found; lead with
  premise corrections; end with the answer** (no offers).
- **Known gap:** answers still sometimes close with an offer ("if you can
  provide the paper, I can extract…") — 2 of 6 trap answers in a manual
  check, down from 3. Prompt wording alone doesn't hold with `gpt-5-mini`;
  → Phase 4 (`reflect` critiques the draft, `revise` fixes it).

Eval caveat found: the judge's context lists corpus papers as
`[arxiv_id] title` without their URL, so cited paper URLs count as
"unsupported" in groundedness (e.g. `halluc-01`). True since Phase 0, so
comparisons between phases are fair, but absolute groundedness is
under-estimated. Fixing it changes the evaluator → re-run the baseline.

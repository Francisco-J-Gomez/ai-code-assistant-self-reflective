# Phase 4 — Self-reflection loop

Branch `phase-4-self-reflection`. Steps 17–23 of
[the self-reflective RAG plan](../self-reflective-plan.md) — the core
feature. Goal: stop trusting the first draft. Every answer is checked claim
by claim against the sources it was written from, before the user sees it;
unsupported claims are revised away, parts of the question the draft
skipped get one targeted search, and the loop is bounded.

Phase 3 pointed here: retrieval got better but answer quality didn't move
(correctness 0.88, groundedness 0.68). The remaining failures were in how
answers were written — unsupported details, omissions, a misread table
(`halluc-05`), closing offers.

## Before → after

| | Phase 3 | Phase 4 |
|---|---|---|
| Answer | `generate` → END | `generate` drafts → `reflect` checks → `finalize` |
| Unsupported claims | Reach the user | Revised: the draft is rewritten with the critique's issues |
| Parts of the question skipped | Reach the user | One targeted search (corpus, then research sub-agent), then revised |
| Closing offers ("if you can provide…") | Prompt rule only | Also flagged by `reflect` as a problem → revised |
| Nothing passes | — | Best draft (by critique score) + a caveat saying what didn't check out |
| UI | Answer + sources | + "✅ Fact-checked against its sources" / "⚠️ Didn't fully pass" line |
| Config | | + `reflect_model`, `reflect_reasoning_effort`, `max_iterations`, `max_missing_aspects` |

## The graph

```
... merge_results / fallback_research / web_search
                     |
                     v
     +--------> generate (draft; on a revision: + previous draft + issues)
     |   +----->    |
     |   |          v
     |   |       reflect (Critique: claims supported?, missing aspects, problems)
     |   |          |-- passed --------------------------------> finalize --> END
     |   |          |-- max_iterations drafts written ---------> finalize (best draft + caveat)
     |   +----------|-- unsupported claims / problems (revise)
     |              |-- missing aspects, first time this turn
     |              v
     +------- search_aspects (corpus + grading per aspect; research sub-agent for the rest)
```

## Design decisions

- **Claim-level, not a single yes/no.** `reflect` returns a `Critique`
  with one `ClaimCheck` (claim, supported, note) per atomic claim — the same
  lesson as Phase 3's per-paper grading: an overall verdict is noisy, a
  per-item one is checkable and shows up in LangSmith traces.
- **The verdict is computed in code, not chosen by the model.** `passed`
  (no unsupported claims, no missing aspects, no problems), `grounded` and
  `score` (share of supported claims) are properties of `Critique`. The
  plan's `grounded` / `unsupported_claims` / `missing_aspects` / `score` are
  all there, but the model only fills in the lists; the routing rule is
  visible and unit-tested.
- **`problems` alongside claims and missing aspects** for what isn't a
  factual claim: an accepted false premise, a paper treated as real that no
  source shows, closing offers. In the trace these were the most common
  real catches (`halluc-01`, `halluc-05`, `corpus-09`).
- **Revise = `generate` again, with the critique.** No separate `revise`
  node: the same prompt and sources, plus the previous draft and the issues
  as instructions. One answer-writing prompt to maintain, and a revision
  can't drift from the citation rules.
- **Missing aspects: one targeted search per turn**, corpus first (graded
  like any sub-query), the research sub-agent for what the corpus lacks; at
  most `max_missing_aspects` (2). Searching on every critique would let a
  stubborn checker drive the cost.
- **Bounded: `max_iterations` = 3 drafts**, then `finalize` returns the
  best draft (ties → the earlier one) with a caveat. Drafts live in state
  (`draft`, `best_draft`); only `finalize` writes the chat message, so a
  rejected draft never reaches the conversation or the next turn's history.
- **Grader ≠ generator (plan step 22).** Same model (`gpt-5-mini`, your
  choice — cheap, about +1 call per draft) but a dedicated fact-checker
  prompt that never sees the writer's instructions, and structured output.
  `reflect_model` is configurable, to try `gpt-5` if the eval shows misses.
- **Reasoning effort stays `medium`.** Checked on 7 real first drafts:
  `low` is ~40% faster (e.g. 16 s vs 43 s on `corpus-09`) but caught 1 of
  the 3 unsupported claims on `corpus-09` and 0 of 1 on `corpus-03`, and
  flagged the mandatory "includes a live web search" line as a claim.
  Configurable as `reflect_reasoning_effort`.
- **"Missing" means asked, not "could say more".** The first prompt let
  `reflect` list 7 "missing aspects" (authors, datasets, …) for a paper the
  draft had correctly said doesn't exist (`halluc-01`, 19 LLM calls), and
  "more numbers" on `corpus-09`. The prompt now restricts missing aspects
  to what the question's own words ask, and counts "doesn't exist in the
  sources" as addressing everything about that thing. Both cases: none.
- **Every route is reflected**, including the direct web route — web
  answers make claims too.
- **Evals:** two new code-level signals, `drafts` (per answer) and
  `self_check_passed` (the agent's own verdict), next to the unchanged
  judges — so we can see whether the agent's check and the external `gpt-5`
  judge agree.

## Changed files

- `agent/schemas.py`: `ClaimCheck`, `Critique` (with computed `passed`,
  `grounded`, `score`, `unsupported_claims`).
- `agent/prompts.py`: `REFLECT_PROMPT`, `REVISION_PROMPT`,
  `format_issues()` (critique → revision instructions), `caveat()`.
- `agent/components.py`: `reflect` (structured output on `reflect_model`);
  `generate` takes an optional `(previous draft, critique)` to revise.
- `agent/nodes.py`: `generate` now writes `draft`; new `reflect`,
  `search_aspects`, `finalize`, `route_after_reflect`; `contextualize`
  resets the reflection fields.
- `agent/graph.py`: the reflection loop wired after `generate`.
- `agent/state.py`: `draft`, `critique`, `iteration`, `aspects_searched`,
  `best_draft`, `best_critique`.
- `agent/config.py`: Phase 4 knobs, `max_iterations >= 1` validated.
- `agent/results.py`, `ui/chat_app.py`: `self_check_note()` — the
  fact-check line under each answer.
- `evals/targets.py`, `evals/evaluators.py`, `evals/run_eval.py`: `drafts`
  and `self_check_passed` outputs / evaluator / summary columns; prefix
  `phase4-reflect`, reflection knobs in metadata.
- Tests: `tests/test_nodes.py` (critique verdict, each node, routing table,
  best-draft tie-break, caveat, revision instructions),
  `tests/test_graph.py` (pass first time, revise, targeted search, research
  for an aspect the corpus lacks, search once per turn, bound + best draft
  + caveat, web route reflected, per-turn reset), `tests/test_eval_harness.py`,
  `tests/test_results.py`; integration tests assert every answered turn went
  through the bounded loop, and the real `reflect` must flag a planted wrong
  number.

## Pre-eval checks

- Unit tests: 168 passed. Integration tests: 7/7 passed (the suite now
  takes ~10 min — the reflection loop is slow, see below).
- Path trace on 12 eval questions (real stack, final prompt and settings):

  | | drafts | self-check | LLM calls | latency |
  |---|---|---|---|---|
  | corpus-01 / 03 | 2 | passed after a revision (1 unsupported claim each) | 7–8 | 104–153 s |
  | corpus-06 / 09, halluc-04, followup-02, multi-01 | 1 | passed | 5–6 | 55–89 s |
  | multi-03 | 2 | passed after a revision | 8 | 157 s |
  | halluc-01 (made-up paper) | 2 | passed after removing a closing offer | 13 | 132 s |
  | halluc-05 (CodeHalu table) | 3 | **not passed** → best draft + caveat | 20 | 251 s |
  | web-01 / 03 | 1 | passed | 3 | 51–62 s |

  The revisions remove real issues: closing offers ("If you can provide a
  correct arXiv identifier… I can look it up"), and details the sources
  don't state (a "static/dynamic tools" enumeration, a "call sites,
  dependencies, type information" list). `halluc-05` is the hard case:
  the research results point to the paper's results table but the
  snippets don't contain all of it, and the draft can't transcribe what it
  doesn't have — the loop ends in the caveat, as designed.
- **Cost:** median latency ~90 s (Phase 3 eval: 48 s); 5–8 LLM calls on
  corpus questions (Phase 3: 4–5). Each `reflect` call takes ~15–45 s at
  `medium` effort. This is the price of the check; Phase 6 (production
  hardening) is where caching / a faster checker belong.

## Results

Experiment `phase4-reflect-3a55de44` vs. `phase3-crag-1c490aad` (full table
in [evals/README.md](../../evals/README.md#phase-4--phase4-reflect-3a55de44-2026-11-19)):

| | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|
| Route correct | 0.93 | 0.97 | 0.97 |
| Correctness | 0.89 | 0.88 | 0.88 |
| **Groundedness** (target) | 0.67 | 0.68 | **0.70** |
| Hallucination-trap correctness | 0.83 | 0.83 | **1.00** |
| Self-check passed | – | – | 0.97 |
| Drafts / answer | – | – | 1.52 |
| LLM calls / question | 2.7 | 4.1 | 5.8 |
| k-tokens / question | 5.3 | 6.6 | 15.8 |
| Mean latency | – | 43.7 s | 132.8 s |

What it shows:

- **The target metric barely moved** (groundedness 0.68 → 0.70; per
  question 7 up, 7 down — within run-to-run noise). The loop did work
  (half the answers were revised, 1.52 drafts on average) but not on what
  the judge penalizes.
- **The agent's check and the judge disagree.** `self_check_passed` is 0.97,
  yet 20 of 33 answers get groundedness 0.5. Reading the judge's comments,
  nearly all are the same kind of issue: **small embellishments while
  paraphrasing** — a qualifier the source doesn't use ("static" retriever,
  "inherently more fragile", "consistently reduce"), a generalization
  ("several papers" when one says it), an inferred negative ("the RepoCoder
  paper contains no mention of GPT-7"), arXiv PDF version links. `reflect`
  counts these as supported paraphrase; the `gpt-5` judge doesn't. Most
  answers have 1–2 of them, which on the judge's 3-level scale is enough
  for 0.5.
- **Real wins on the gross failures:** hallucination-trap correctness
  0.83 → 1.00 — `halluc-05`, Phase 3's misread CodeHalu table, is now
  correct; closing offers and accepted false premises are revised away.
- **Cost:** ~3× latency (44 s → 133 s), 2.4× tokens, +1.7 LLM calls per
  question. Multi-part questions take ~4.5 min.

Conclusion: the reflection loop catches wrong facts and bad answer
behaviour, but as calibrated it is more lenient than the evaluation's
judge on paraphrase, so groundedness stays flat at 3× the latency.

### Follow-up: tightened prompts

To close the gap with the judge, both prompts were tightened to the
judge's standard of "supported": same strength and scope as the source;
added qualifiers, generalizations, causes/contrasts the source doesn't
draw, claims about what a paper does *not* contain, and URLs not in the
sources count as unsupported (`reflect`) and are avoided (`generate`).

- Re-running the new `reflect` on 7 saved old drafts flagged about as many
  claims as before (corpus-09: 2 vs 3, multi-03: 2 vs 2): with
  `gpt-5-mini`, prompt wording doesn't make the checker measurably stricter.
  The effect comes from `generate` writing closer to the sources.
- Multi-part split, `phase4-reflect-fe6a373a` (6 questions):

  | multi_part | Phase 3 | Phase 4 | Phase 4, tightened |
  |---|---|---|---|
  | Groundedness | 0.67 | 0.67 | **0.83** |
  | Correctness | 0.92 | 0.83 | 0.83 |
  | Drafts / self-check passed | – | 1.50 / 1.00 | 1.67 / 0.83 |
  | Mean latency | 62.5 s | 270.7 s | 123.2 s |

  Groundedness 1.0 on 4 of 6 (was 2 of 6: `multi-01`, `multi-03`,
  `multi-06` now 1.0; `multi-02` 1.0 → 0.5). With 6 questions one answer
  moves the mean by 0.08, so this is a direction, not a measurement.
  `multi-04` is the one answer the self-check didn't pass after 3 drafts
  (shown with a caveat) and the judge also gave 0.5 — they agree there.
  Recall 0.81 → 0.58 is `decompose` run-to-run variation (the prompts
  don't touch retrieval), as in Phase 3.

### Final run (tightened prompts, full dataset)

Experiment `phase4-reflect-2a7e1156` — the Phase 4 result:

| | Phase 3 | Phase 4 (first prompts) | **Phase 4 final** |
|---|---|---|---|
| Route correct | 0.97 | 0.97 | 0.95 |
| Correctness | 0.88 | 0.88 | 0.85 |
| **Groundedness** (target) | 0.68 | 0.70 | **0.82** |
| — multi-part groundedness | 0.67 | 0.67 | **1.00** |
| Hallucination-trap correctness | 0.83 | 1.00 | **1.00** |
| Self-check passed | – | 0.97 | 1.00 |
| Drafts / answer | – | 1.52 | 1.30 |
| LLM calls / question | 4.1 | 5.8 | 5.6 |
| k-tokens / question | 6.6 | 15.8 | 13.8 |
| Mean / median latency | 43.7 / 47.8 s | 132.8 s / – | 118.6 / 58.7 s |

- **Groundedness 0.68 → 0.82, the phase's target, reached** — and the
  agent's own check and the judge now agree far more often (the judge
  gives 1.0 to 21 of 33 answers, up from 13). The gain came from `generate`
  writing closer to its sources, which also means fewer revisions (1.30
  drafts) and lower latency than the first Phase 4 run.
- **Correctness 0.88 → 0.85 is within noise, and its misses are
  retrieval misses.** Every 0.5 the judge gave on a corpus / multi-part
  question names the same cause: the answer omits papers the reference
  expects (PYInfer and TIPICAL for `corpus-06`, AVATAR for `followup-04`,
  CAT-LM / UniTSyn for `multi-03`, …) — papers that weren't retrieved.
  Grounded answers can only be as complete as their sources: corpus recall
  is 0.42 → **Phase 5**.
- **Route 0.97 → 0.95:** `corpus-03` (Copilot security) found too few
  relevant papers in this run and fell back to research — its answer is
  still correct (1.0). Run-to-run variation in grading.
- **Cost:** vs. Phase 3, +1.5 LLM calls and 2× tokens per question;
  median latency 48 s → 59 s, but the mean is pulled up by the traps
  (~4 min each: rewrite → research → revisions). Streaming progress to the
  UI (Phase 6) is what makes this acceptable for a user.

Conclusion: self-reflection raised groundedness from 0.68 to 0.82 without
hurting correctness, at a moderate cost on ordinary questions. The
remaining failures are incomplete answers from incomplete retrieval —
Phase 5's target.


"""Unit tests for individual nodes and routing functions - no graph, no
network. Control flow across nodes is covered in tests/test_graph.py."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.config import AgentConfig
from agent.prompts import CORPUS_ONLY_NOTE, DECLINE_MESSAGE, caveat, format_issues, source_note
from agent.schemas import ClaimCheck, Critique, RouteDecision
from langgraph.types import Overwrite, Send

from agent.state import AgentState, CorpusSearchState, Document, SubQueryResult

PAPER = Document(kind="corpus", title="P", url="https://arxiv.org/abs/1", text="...", arxiv_id="1")
PAPER_2 = Document(kind="corpus", title="P2", url="https://arxiv.org/abs/2", text="...", arxiv_id="2")
PAPER_3 = Document(kind="corpus", title="P3", url="https://arxiv.org/abs/3", text="...", arxiv_id="3")
WEB = Document(kind="web", title="W", url="https://example.com", text="...")
LIVE = Document(kind="arxiv", title="L", url="https://arxiv.org/abs/9", text="...", arxiv_id="9")

CONFIG = AgentConfig(min_relevant_docs=2, max_query_rewrites=1)


class Stub:
    """Only the component methods a test needs."""

    def __init__(self, **methods):
        self.__dict__.update(methods)


def test_split_latest_question_returns_question_and_prior_history():
    msgs = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2")]
    question, history = nodes.split_latest_question(msgs)
    assert question == "q2"
    assert [m.content for m in history] == ["q1", "a1"]


def test_split_latest_question_without_user_message_raises():
    with pytest.raises(ValueError):
        nodes.split_latest_question([AIMessage("hi")])


# --- contextualize / router / decline ----------------------------------------


def test_contextualize_first_turn_skips_the_llm_and_resets_per_turn_fields():
    def fail(q, h):
        raise AssertionError("no history: contextualize must not call the LLM")

    state = AgentState(messages=[HumanMessage("q")], route="web", sub_queries=["old"], missing=["old"], retrieved=[PAPER], documents=[PAPER])
    update = nodes.contextualize(state, Stub(contextualize=fail))
    assert update.pop("sub_results") == Overwrite([])  # append-reducer field: cleared explicitly
    assert update == {
        "question": "q",
        "standalone_question": "q",
        "route": None,
        "sub_queries": [],
        "missing": [],
        "retrieved": [],
        "documents": [],
        "draft": "",
        "critique": None,
        "iteration": 0,
        "aspects_searched": False,
        "best_draft": "",
        "best_critique": None,
    }


def test_contextualize_follow_up_rewrites_with_history():
    seen = {}

    def contextualize(q, h):
        seen["history"] = [m.content for m in h]
        return "What benchmark did RepoCoder introduce?"

    state = AgentState(messages=[HumanMessage("What is RepoCoder?"), AIMessage("A paper."), HumanMessage("What benchmark did they introduce?")])
    update = nodes.contextualize(state, Stub(contextualize=contextualize))
    assert update["question"] == "What benchmark did they introduce?"
    assert update["standalone_question"] == "What benchmark did RepoCoder introduce?"
    assert seen["history"] == ["What is RepoCoder?", "A paper."]


@pytest.mark.parametrize("decision, route", [("corpus", "corpus"), ("web", "web"), ("out_of_scope", "decline")])
def test_router_maps_decision_to_route(decision, route):
    c = Stub(route=lambda q: RouteDecision(reasoning="", route=decision))
    assert nodes.router(AgentState(standalone_question="q"), c) == {"route": route}


def test_decline_returns_fixed_message():
    update = nodes.decline(AgentState(), Stub())
    assert update["messages"][0].content == DECLINE_MESSAGE


# --- corpus path: decompose / fan out / merge / fallback ---------------------


def test_decompose_falls_back_to_the_question_if_nothing_comes_back():
    update = nodes.decompose(AgentState(standalone_question="q"), Stub(decompose=lambda q: []))
    assert update == {"sub_queries": ["q"]}


def test_fan_out_sends_one_corpus_search_per_sub_query_with_its_index():
    sends = nodes.fan_out_sub_queries(AgentState(sub_queries=["a", "b"]))
    assert sends == [Send("corpus_search", {"index": 0, "sub_query": "a"}), Send("corpus_search", {"index": 1, "sub_query": "b"})]


def _result(index, sub_query, documents, retrieved=None, enough=True):
    return SubQueryResult(
        index=index, sub_query=sub_query, search_query=sub_query, query_rewrites=0,
        retrieved=retrieved or documents, documents=documents, enough=enough,
    )


def test_merge_results_orders_by_sub_query_dedupes_and_lists_missing_parts():
    state = AgentState(
        route="corpus",
        sub_results=[  # finished out of order
            _result(1, "b", [PAPER_2, PAPER_3], enough=False),
            _result(0, "a", [PAPER, PAPER_2], retrieved=[PAPER, PAPER_2, WEB]),
        ],
    )
    update = nodes.merge_results(state)
    assert update["documents"] == [PAPER, PAPER_2, PAPER_3]
    assert update["retrieved"] == [PAPER, PAPER_2, WEB, PAPER_3]
    assert update["missing"] == ["b"]
    assert update["route"] == "corpus"  # unchanged: fallback_research will set "web"


def test_merge_results_sets_corpus_route_when_every_part_is_covered():
    update = nodes.merge_results(AgentState(route="corpus", sub_results=[_result(0, "a", [PAPER, PAPER_2])]))
    assert update["missing"] == [] and update["route"] == "corpus"


def test_fallback_research_searches_the_missing_parts_and_keeps_corpus_documents():
    seen = {}

    def research(question, missing, known):
        seen.update(question=question, missing=missing, known=known)
        return [WEB, LIVE]

    state = AgentState(standalone_question="q", missing=["part b"], documents=[PAPER])
    update = nodes.fallback_research(state, Stub(research=research))
    assert update == {"documents": [PAPER, WEB, LIVE], "route": "web"}
    assert seen == {"question": "q", "missing": ["part b"], "known": [PAPER]}


# --- corpus_search subgraph (one sub-query) -----------------------------------------


def test_retrieve_corpus_only_keeps_chunks_not_seen_this_run():
    rescored = PAPER.model_copy(update={"score": 0.9})  # same chunk, different score
    c = Stub(search_corpus=lambda q: [rescored, PAPER_2])
    update = nodes.retrieve_corpus(CorpusSearchState(sub_query="q", retrieved=[PAPER]), c)
    assert update["candidates"] == [PAPER_2]
    assert update["retrieved"] == [PAPER, PAPER_2]


def test_retrieve_corpus_uses_the_rewritten_query_once_there_is_one():
    seen = []
    c = Stub(search_corpus=lambda q: seen.append(q) or [])
    nodes.retrieve_corpus(CorpusSearchState(sub_query="q"), c)
    nodes.retrieve_corpus(CorpusSearchState(sub_query="q", search_query="better q"), c)
    assert seen == ["q", "better q"]


def test_grade_docs_grades_against_the_sub_query_and_keeps_relevant_ones():
    seen = []

    def grade(q, docs):
        seen.append(q)
        return [True, False]

    update = nodes.grade_docs(CorpusSearchState(sub_query="part a", candidates=[PAPER, PAPER_2]), Stub(grade_documents=grade), CONFIG)
    assert update == {"documents": [PAPER], "candidates": []}
    assert seen == ["part a"]


def test_grade_docs_keeps_at_most_max_relevant_docs_in_ranking_order():
    papers = [PAPER, PAPER_2, PAPER_3, PAPER.model_copy(update={"text": "chunk 2"})]
    update = nodes.grade_docs(CorpusSearchState(sub_query="q", candidates=papers), Stub(grade_documents=lambda q, d: [True, False, True, True]), CONFIG)
    assert update["documents"] == [PAPER, PAPER_3, papers[3]]  # 3 kept, PAPER_2 graded out


def test_grade_docs_cap_counts_documents_kept_on_an_earlier_search():
    state = CorpusSearchState(sub_query="q", documents=[PAPER, PAPER_2], candidates=[PAPER_3, PAPER_3.model_copy(update={"text": "x"})])
    update = nodes.grade_docs(state, Stub(grade_documents=lambda q, d: [True, True]), CONFIG)
    assert update["documents"] == [PAPER, PAPER_2, PAPER_3]


def test_grade_docs_without_candidates_makes_no_llm_call():
    def fail(q, d):
        raise AssertionError("nothing to grade")

    update = nodes.grade_docs(CorpusSearchState(sub_query="q", documents=[PAPER]), Stub(grade_documents=fail), CONFIG)
    assert update["documents"] == [PAPER]


def test_config_rejects_min_above_max():
    with pytest.raises(ValueError):
        AgentConfig(min_relevant_docs=4, max_relevant_docs=3)


def test_rewrite_query_replaces_the_query_and_counts_the_retry():
    seen = {}

    def rewrite(question, query, found):
        seen.update(question=question, query=query, found=found)
        return "repository-level code completion retrieval"

    state = CorpusSearchState(sub_query="q?", retrieved=[PAPER])
    update = nodes.rewrite_query(state, Stub(rewrite_query=rewrite))
    assert update == {"search_query": "repository-level code completion retrieval", "query_rewrites": 1}
    assert seen == {"question": "q?", "query": "q?", "found": [PAPER]}


def test_rewrite_query_keeps_the_old_query_if_the_rewrite_is_empty():
    update = nodes.rewrite_query(CorpusSearchState(sub_query="q", search_query="q2"), Stub(rewrite_query=lambda *a: ""))
    assert update["search_query"] == "q2"


@pytest.mark.parametrize("documents, enough", [([PAPER, PAPER_2], True), ([PAPER], False)])
def test_report_hands_back_one_result_with_enough_flag(documents, enough):
    state = CorpusSearchState(index=2, sub_query="q", search_query="q2", query_rewrites=1, retrieved=[PAPER, PAPER_2, PAPER_3], documents=documents)
    (result,) = nodes.report(state, CONFIG)["sub_results"]
    assert (result.index, result.sub_query, result.search_query, result.query_rewrites) == (2, "q", "q2", 1)
    assert result.documents == documents and result.enough is enough
    assert len(result.retrieved) == 3


def test_web_search_uses_only_web_results():
    update = nodes.web_search(AgentState(standalone_question="q"), Stub(search_web=lambda q: [WEB]))
    assert update["documents"] == [WEB]


# --- routing -------------------------------------------------------------------


@pytest.mark.parametrize(
    "route, after_router",
    [("decline", "decline"), ("web", "web_search"), ("corpus", "decompose"), (None, "decompose")],
)
def test_route_after_router(route, after_router):
    assert nodes.route_after_router(AgentState(route=route)) == after_router


def test_route_after_merge():
    assert nodes.route_after_merge(AgentState(missing=[])) == "generate"
    assert nodes.route_after_merge(AgentState(missing=["b"])) == "fallback_research"


@pytest.mark.parametrize(
    "documents, rewrites, expected",
    [
        ([PAPER, PAPER_2], 0, "report"),  # enough relevant
        ([PAPER], 0, "rewrite_query"),  # too few, retry left
        ([PAPER], 1, "report"),  # too few, no retry left: report "not enough"
    ],
)
def test_route_after_grade_docs(documents, rewrites, expected):
    state = CorpusSearchState(documents=documents, query_rewrites=rewrites)
    assert nodes.route_after_grade_docs(state, CONFIG) == expected


# --- prompt helpers -------------------------------------------------------------


def test_source_note_is_decided_by_the_documents_not_the_model():
    assert source_note([PAPER]) == CORPUS_ONLY_NOTE
    assert "a live web search" in source_note([PAPER, WEB])
    assert "a live arXiv search" in source_note([PAPER, LIVE])
    assert "a live web search and a live arXiv search" in source_note([WEB, LIVE])


# --- self-reflection -----------------------------------------------------------


def _critique(supported=(), unsupported=(), missing=(), problems=()):
    claims = [ClaimCheck(claim=c, supported=True) for c in supported]
    claims += [ClaimCheck(claim=c, supported=False, note=f"no source says {c}") for c in unsupported]
    return Critique(claims=claims, missing_aspects=list(missing), problems=list(problems))


GOOD = _critique(supported=["a", "b"])


def test_critique_verdict_is_computed_from_its_lists():
    assert GOOD.passed and GOOD.grounded and GOOD.score == 1.0
    half = _critique(supported=["a"], unsupported=["b"])
    assert not half.passed and not half.grounded and half.score == 0.5
    assert [c.claim for c in half.unsupported_claims] == ["b"]
    assert not _critique(supported=["a"], missing=["x"]).passed  # grounded but incomplete
    assert not _critique(supported=["a"], problems=["accepts a false premise"]).passed
    assert _critique().score == 1.0  # no factual claims: nothing unsupported


def test_generate_writes_a_draft_not_a_message_and_counts_it():
    stub = Stub(generate=lambda q, h, docs, revision: f"draft for {q}, revision={revision}")
    update = nodes.generate(AgentState(messages=[HumanMessage("q")], question="q", documents=[PAPER]), stub)
    assert update == {"draft": "draft for q, revision=None", "iteration": 1}


def test_generate_after_a_failed_critique_revises_the_previous_draft():
    bad = _critique(unsupported=["x"])
    seen = []
    stub = Stub(generate=lambda q, h, docs, revision: seen.append(revision) or "v2")
    state = AgentState(messages=[HumanMessage("q")], question="q", draft="v1", critique=bad, iteration=1)
    assert nodes.generate(state, stub) == {"draft": "v2", "iteration": 2}
    assert seen == [("v1", bad)]


def test_reflect_checks_the_draft_against_the_documents_and_tracks_the_best():
    seen = []
    stub = Stub(reflect=lambda q, draft, docs: seen.append((q, draft, docs)) or GOOD)
    state = AgentState(standalone_question="sq", draft="d", documents=[PAPER])
    assert nodes.reflect(state, stub) == {"critique": GOOD, "best_draft": "d", "best_critique": GOOD}
    assert seen == [("sq", "d", [PAPER])]


@pytest.mark.parametrize(
    "new, keeps_best",
    [
        (_critique(supported=["a"], unsupported=["b", "c"]), True),  # worse score
        (_critique(supported=["a"], unsupported=["b"]), True),  # tie -> keep the earlier draft
        (_critique(supported=["a", "b", "c"], unsupported=["d"]), False),  # better score
    ],
)
def test_reflect_replaces_the_best_draft_only_when_the_new_one_is_better(new, keeps_best):
    best = _critique(supported=["a"], unsupported=["b"])
    state = AgentState(draft="new", best_draft="old", best_critique=best)
    update = nodes.reflect(state, Stub(reflect=lambda q, d, docs: new))
    assert ("best_draft" not in update) == keeps_best


def test_search_aspects_searches_the_corpus_first_and_research_for_the_rest():
    critique = _critique(supported=["a"], missing=["in corpus", "not in corpus", "over the cap"])
    corpus = {"in corpus": [PAPER, PAPER_2, PAPER_3], "not in corpus": [PAPER_3]}
    research_calls = []
    stub = Stub(
        search_corpus=lambda q: corpus.get(q, []),
        grade_documents=lambda q, docs: [d is not PAPER_3 for d in docs],
        research=lambda q, missing, known: research_calls.append((q, missing, [d.title for d in known])) or [LIVE],
    )
    state = AgentState(standalone_question="sq", critique=critique, documents=[PAPER], route="corpus")
    update = nodes.search_aspects(state, stub, AgentConfig(max_missing_aspects=2))

    # PAPER was already a document; PAPER_2 is new and relevant; PAPER_3 is graded off-topic.
    assert research_calls == [("sq", ["not in corpus"], ["P", "P2"])]
    assert update == {"aspects_searched": True, "retrieved": [PAPER_2], "documents": [PAPER, PAPER_2, LIVE], "route": "web"}


def test_search_aspects_without_new_documents_makes_no_grading_call():
    def fail(q, docs):
        raise AssertionError("nothing new to grade")

    stub = Stub(search_corpus=lambda q: [PAPER], grade_documents=fail, research=lambda q, m, k: [])
    state = AgentState(critique=_critique(missing=["x"]), documents=[PAPER], route="corpus")
    update = nodes.search_aspects(state, stub, AgentConfig())
    assert update == {"aspects_searched": True, "retrieved": [], "documents": [PAPER]}  # route unchanged


def test_finalize_returns_the_passing_draft_as_is():
    update = nodes.finalize(AgentState(draft="final", critique=GOOD, best_draft="final", best_critique=GOOD))
    assert update["messages"][0].content == "final"


def test_finalize_without_a_passing_draft_returns_the_best_one_with_a_caveat():
    best = _critique(supported=["a"], unsupported=["b"])
    state = AgentState(draft="last", critique=_critique(unsupported=["x", "y"]), best_draft="best", best_critique=best)
    content = nodes.finalize(state)["messages"][0].content
    assert content.startswith("best\n\n---")
    assert "could not be verified" in content


def test_caveat_names_what_is_missing():
    assert caveat(GOOD) == "" and caveat(None) == ""
    text = caveat(_critique(supported=["a"], missing=["test generation"]))
    assert "may not fully cover: test generation" in text
    assert "flagged issues" in caveat(_critique(problems=["offers to search more"]))


def test_format_issues_turns_the_critique_into_revision_instructions():
    text = format_issues(_critique(supported=["ok"], unsupported=["x"], missing=["m"], problems=["p"]))
    assert 'Unsupported claim: "x" - no source says x' in text
    assert "- Problem: p" in text
    assert "- Not answered: m." in text
    assert "ok" not in text  # supported claims aren't issues


@pytest.mark.parametrize(
    "critique, iteration, searched, expected",
    [
        (GOOD, 1, False, "finalize"),
        (_critique(unsupported=["x"]), 1, False, "generate"),
        (_critique(problems=["p"]), 2, False, "generate"),
        (_critique(unsupported=["x"]), 3, False, "finalize"),  # out of drafts
        (_critique(missing=["m"]), 1, False, "search_aspects"),
        (_critique(missing=["m"]), 2, True, "generate"),  # already searched this turn: just revise
        (_critique(missing=["m"]), 3, False, "finalize"),
    ],
)
def test_route_after_reflect(critique, iteration, searched, expected):
    state = AgentState(critique=critique, iteration=iteration, aspects_searched=searched)
    assert nodes.route_after_reflect(state, AgentConfig(max_iterations=3)) == expected

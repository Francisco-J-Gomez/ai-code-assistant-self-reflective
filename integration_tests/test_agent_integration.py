"""Integration tests against the real agent stack: OpenAI, Pinecone, Tavily.

Skipped by default (mirrors integration_tests/test_arxiv_integration.py) since they
hit live external services, cost real (if tiny) money, and take longer
than the mocked unit tests. Run explicitly with:

    RUN_AGENT_INTEGRATION_TESTS=1 uv run pytest integration_tests/test_agent_integration.py -v

Since Phase 1 the path is decided by graph edges, so these assert on the
`route` and `documents` the graph leaves in state rather than on which tools
a ReAct loop happened to call. The out-of-scope case is the original
regression: an early prompt-only agent answered a cookie recipe from general
knowledge.
"""
import os

import pytest

from agent.components import make_components
from agent.config import AgentConfig
from agent.graph import build_graph
from agent.state import Document

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_INTEGRATION_TESTS") != "1",
    reason="Hits real OpenAI/Pinecone/Tavily APIs; set RUN_AGENT_INTEGRATION_TESTS=1 to run.",
)


@pytest.fixture(scope="module")
def graph():
    return build_graph(AgentConfig())


def _ask(graph, question: str, thread_id: str) -> dict:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    # Phase 4: every answered turn went through the bounded reflection loop.
    if result["route"] != "decline":
        assert 1 <= result["iteration"] <= AgentConfig().max_iterations
        assert result["critique"] is not None
    return result


def test_in_corpus_question_answers_from_corpus_only(graph):
    result = _ask(
        graph,
        "What techniques exist for automated program repair using large language models?",
        "integration-test-corpus-hit",
    )

    assert result["route"] == "corpus"
    assert result["documents"] and all(d.kind == "corpus" for d in result["documents"])
    # Enough relevant papers on the first search, for every sub-query.
    assert all(r.enough and r.query_rewrites == 0 for r in result["sub_results"])

    answer = result["messages"][-1].content.lower()
    assert "arxiv" in answer or "corpus" in answer or "paper" in answer


def test_tool_question_is_routed_straight_to_web_search(graph):
    result = _ask(
        graph,
        "How do you add a human-in-the-loop approval step to a LangGraph agent using interrupt()?",
        "integration-test-web-route",
    )

    assert result["route"] == "web"
    # Routed to the web up front: the corpus was never searched.
    assert result["retrieved"] == []
    assert result["documents"] and all(d.kind == "web" for d in result["documents"])


def test_research_question_the_corpus_cannot_answer_falls_back_to_research(graph):
    # Routed to the corpus (it asks about a specific paper), but no retrieved
    # paper says anything about GPT-7 on SWE-bench: grade_docs finds too few
    # relevant papers, the rewritten query doesn't help either, so the
    # fallback_research sub-agent searches the web / live arXiv. False
    # premise: GPT-7 doesn't exist and RepoCoder evaluated on RepoEval.
    result = _ask(
        graph,
        "What pass@1 did GPT-7 achieve on SWE-bench in the RepoCoder paper?",
        "integration-test-corpus-fallback",
    )

    assert result["route"] == "web"
    assert result["retrieved"], "the corpus is searched first"
    assert [r.query_rewrites for r in result["sub_results"]] == [1], "one corrective retry before the fallback"
    assert any(d.kind in ("web", "arxiv") for d in result["documents"])

    # Live sources were used, so the answer says so (the label is chosen by
    # the graph from the documents, not by the model).
    assert "live" in result["messages"][-1].content.lower()
    # Not asserted: that the answer doesn't close with an offer ("if you can
    # provide the paper..."). gpt-5-mini still does that in ~1 of 3 trap
    # answers despite the prompt; Phase 4's reflect step is the fix.


def test_multi_part_question_is_searched_per_part(graph):
    result = _ask(
        graph,
        "How are LLMs used to generate unit tests, and what benchmarks exist to evaluate generated tests?",
        "integration-test-multi-part",
    )

    assert len(result["sub_queries"]) == 2
    assert len(result["sub_results"]) == 2
    # Each part keeps at most max_relevant_docs (3) of its own papers.
    assert all(len(r.documents) <= 3 for r in result["sub_results"])


def test_out_of_scope_question_is_declined_without_retrieval(graph):
    result = _ask(
        graph,
        "What's a good recipe for chocolate chip cookies?",
        "integration-test-out-of-scope",
    )

    assert result["route"] == "decline"
    assert result["documents"] == [] and result["retrieved"] == []

    answer = result["messages"][-1].content.lower()
    assert "chocolate" not in answer and "flour" not in answer  # didn't just answer it


def test_follow_up_is_resolved_from_conversation_history(graph):
    _ask(graph, "What is RepoCoder?", "integration-test-follow-up")
    result = _ask(graph, "What benchmark did they introduce to evaluate it?", "integration-test-follow-up")

    assert result["route"] in ("corpus", "web")
    assert "repocoder" in result["standalone_question"].lower()
    assert "repoeval" in result["messages"][-1].content.lower()


def test_reflect_flags_a_claim_the_sources_dont_support():
    """The real fact-checker, on a draft with one planted wrong number."""
    c = make_components(AgentConfig())
    source = Document(
        kind="corpus",
        title="RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation",
        url="https://arxiv.org/abs/2303.12570",
        arxiv_id="2303.12570",
        text=(
            "RepoCoder is a framework for repository-level code completion that uses an iterative "
            "retrieval-generation pipeline. We also introduce RepoEval, a benchmark built from recent "
            "GitHub repositories. RepoCoder improves the in-file completion baseline by over 10% in all settings."
        ),
    )
    draft = (
        "RepoCoder (2303.12570) does repository-level code completion with iterative retrieval and generation, "
        "and introduces the RepoEval benchmark. It improves the in-file baseline by 45% on HumanEval."
    )
    critique = c.reflect("What is RepoCoder and how much does it improve completion?", draft, [source])

    assert not critique.grounded
    flagged = " ".join(cl.claim for cl in critique.unsupported_claims).lower()
    assert "45" in flagged or "humaneval" in flagged
    # The true statements are not flagged.
    assert not any("repoeval" in cl.claim.lower() and "45" not in cl.claim for cl in critique.unsupported_claims)

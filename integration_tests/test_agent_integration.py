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

from agent.config import AgentConfig
from agent.graph import build_graph

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_INTEGRATION_TESTS") != "1",
    reason="Hits real OpenAI/Pinecone/Tavily APIs; set RUN_AGENT_INTEGRATION_TESTS=1 to run.",
)


@pytest.fixture(scope="module")
def graph():
    return build_graph(AgentConfig())


def _ask(graph, question: str, thread_id: str) -> dict:
    return graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )


def test_in_corpus_question_answers_from_corpus_only(graph):
    result = _ask(
        graph,
        "What techniques exist for automated program repair using large language models?",
        "integration-test-corpus-hit",
    )

    assert result["route"] == "corpus"
    assert result["documents"] and all(d.kind == "corpus" for d in result["documents"])

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
    assert result["documents"] and all(d.kind == "web" for d in result["documents"])


def test_research_question_the_corpus_cannot_answer_falls_back_to_web(graph):
    # Routed to the corpus (it asks about a specific paper), but the
    # RepoCoder text says nothing about GPT-7 or SWE-bench, so check_corpus
    # sends it on to the web. False premise: GPT-7 doesn't exist and
    # RepoCoder evaluated on RepoEval.
    result = _ask(
        graph,
        "What pass@1 did GPT-7 achieve on SWE-bench in the RepoCoder paper?",
        "integration-test-corpus-fallback",
    )

    assert result["route"] == "web"
    kinds = [d.kind for d in result["documents"]]
    assert "corpus" in kinds and "web" in kinds  # corpus searched first, then web

    # Web sources were used, so the answer says so (the label is chosen by
    # the graph from the documents, not by the model).
    assert "web search" in result["messages"][-1].content.lower()
    # Not asserted: that the answer doesn't close with an offer ("if you can
    # provide the paper..."). gpt-5-mini still does that in ~1 of 3 trap
    # answers despite the prompt; Phase 4's reflect step is the fix.


def test_out_of_scope_question_is_declined_without_retrieval(graph):
    result = _ask(
        graph,
        "What's a good recipe for chocolate chip cookies?",
        "integration-test-out-of-scope",
    )

    assert result["route"] == "decline"
    assert result["documents"] == []

    answer = result["messages"][-1].content.lower()
    assert "chocolate" not in answer and "flour" not in answer  # didn't just answer it


def test_follow_up_is_resolved_from_conversation_history(graph):
    _ask(graph, "What is RepoCoder?", "integration-test-follow-up")
    result = _ask(graph, "What benchmark did they introduce to evaluate it?", "integration-test-follow-up")

    assert result["route"] in ("corpus", "web")
    assert "repocoder" in result["standalone_question"].lower()
    assert "repoeval" in result["messages"][-1].content.lower()

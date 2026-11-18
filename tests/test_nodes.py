"""Unit tests for individual nodes and routing functions - no graph, no
network. Control flow across nodes is covered in tests/test_graph.py."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.prompts import DECLINE_MESSAGE
from agent.schemas import CorpusCheck, ScopeDecision
from agent.state import AgentState, Document

PAPER = Document(kind="corpus", title="P", url="https://arxiv.org/abs/1", text="...", arxiv_id="1")
WEB = Document(kind="web", title="W", url="https://example.com", text="...")


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


def test_scope_check_resets_per_turn_fields():
    state = AgentState(messages=[HumanMessage("q")], route="web", documents=[PAPER])
    c = Stub(scope_check=lambda q, h: ScopeDecision(in_scope=True, search_query="sq"))
    assert nodes.scope_check(state, c) == {"question": "q", "search_query": "sq", "route": None, "documents": []}


def test_scope_check_out_of_scope_sets_decline_route():
    c = Stub(scope_check=lambda q, h: ScopeDecision(in_scope=False, search_query=""))
    update = nodes.scope_check(AgentState(messages=[HumanMessage("cookies?")]), c)
    assert update["route"] == "decline"


def test_decline_returns_fixed_message():
    update = nodes.decline(AgentState(), Stub())
    assert update["messages"][0].content == DECLINE_MESSAGE


@pytest.mark.parametrize("enough, route", [(True, "corpus"), (False, "web")])
def test_check_corpus_sets_route(enough, route):
    c = Stub(check_corpus=lambda q, d: CorpusCheck(reasoning="", answers_question=enough))
    assert nodes.check_corpus(AgentState(question="q", documents=[PAPER]), c) == {"route": route}


def test_web_search_appends_to_corpus_documents():
    c = Stub(search_web=lambda q: [WEB])
    update = nodes.web_search(AgentState(search_query="q", documents=[PAPER]), c)
    assert update["documents"] == [PAPER, WEB]


@pytest.mark.parametrize(
    "route, after_scope, after_check",
    [("decline", "decline", "web_search"), (None, "retrieve_corpus", "web_search"), ("corpus", "retrieve_corpus", "generate"), ("web", "retrieve_corpus", "web_search")],
)
def test_routing_functions(route, after_scope, after_check):
    state = AgentState(route=route)
    assert nodes.route_after_scope_check(state) == after_scope
    assert nodes.route_after_check_corpus(state) == after_check

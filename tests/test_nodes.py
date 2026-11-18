"""Unit tests for individual nodes and routing functions - no graph, no
network. Control flow across nodes is covered in tests/test_graph.py."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.prompts import DECLINE_MESSAGE
from agent.schemas import CorpusCheck, RouteDecision
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


def test_contextualize_first_turn_skips_the_llm_and_resets_per_turn_fields():
    def fail(q, h):
        raise AssertionError("no history: contextualize must not call the LLM")

    state = AgentState(messages=[HumanMessage("q")], route="web", documents=[PAPER])
    assert nodes.contextualize(state, Stub(contextualize=fail)) == {
        "question": "q",
        "standalone_question": "q",
        "route": None,
        "documents": [],
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


@pytest.mark.parametrize("enough, route", [(True, "corpus"), (False, "web")])
def test_check_corpus_sets_route_judging_the_standalone_question(enough, route):
    seen = []
    c = Stub(check_corpus=lambda q, d: seen.append(q) or CorpusCheck(reasoning="", answers_question=enough))
    state = AgentState(question="they?", standalone_question="RepoCoder?", documents=[PAPER])
    assert nodes.check_corpus(state, c) == {"route": route}
    assert seen == ["RepoCoder?"]


def test_web_search_appends_to_corpus_documents():
    c = Stub(search_web=lambda q: [WEB])
    update = nodes.web_search(AgentState(standalone_question="q", documents=[PAPER]), c)
    assert update["documents"] == [PAPER, WEB]


@pytest.mark.parametrize(
    "route, after_router",
    [("decline", "decline"), ("web", "web_search"), ("corpus", "retrieve_corpus"), (None, "retrieve_corpus")],
)
def test_route_after_router(route, after_router):
    assert nodes.route_after_router(AgentState(route=route)) == after_router


@pytest.mark.parametrize("route, after_check", [("corpus", "generate"), ("web", "web_search")])
def test_route_after_check_corpus(route, after_check):
    assert nodes.route_after_check_corpus(AgentState(route=route)) == after_check


def test_source_note_is_decided_by_the_documents_not_the_model():
    from agent.prompts import CORPUS_ONLY_NOTE, WITH_WEB_NOTE, source_note

    assert source_note([PAPER]) == CORPUS_ONLY_NOTE
    assert source_note([PAPER, WEB]) == WITH_WEB_NOTE
    assert source_note([WEB]) == WITH_WEB_NOTE

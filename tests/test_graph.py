"""Control-flow tests for the explicit graph, with fake components (no
network, no LLM): which nodes run, what ends up in state, and that per-turn
state doesn't leak across turns of the same thread."""
from langchain_core.messages import HumanMessage

from agent.components import AgentComponents
from agent.graph import build_graph
from agent.prompts import DECLINE_MESSAGE
from agent.schemas import CorpusCheck, RouteDecision
from agent.state import Document

PAPER = Document(kind="corpus", title="RepoCoder", url="https://arxiv.org/abs/2303.12570", text="...", arxiv_id="2303.12570")
WEB = Document(kind="web", title="LangGraph docs", url="https://docs.langchain.com/x", text="...")


class FakeComponents:
    """Records every call; behavior configured per test."""

    def __init__(self, *, route="corpus", standalone="standalone question", corpus_enough=True):
        self.route_to = route
        self.standalone = standalone
        self.corpus_enough = corpus_enough
        self.calls = []

    def contextualize(self, question, history):
        self.calls.append(("contextualize", question, [m.content for m in history]))
        return self.standalone

    def route(self, question):
        self.calls.append(("route", question))
        return RouteDecision(reasoning="fake", route=self.route_to)

    def search_corpus(self, query):
        self.calls.append(("search_corpus", query))
        return [PAPER]

    def check_corpus(self, question, documents):
        self.calls.append(("check_corpus", question, len(documents)))
        return CorpusCheck(reasoning="fake", answers_question=self.corpus_enough)

    def search_web(self, query):
        self.calls.append(("search_web", query))
        return [WEB]

    def generate(self, question, history, documents):
        self.calls.append(("generate", question, [m.content for m in history], [d.title for d in documents]))
        return f"answer to: {question}"

    def components(self):
        return AgentComponents(
            contextualize=self.contextualize,
            route=self.route,
            search_corpus=self.search_corpus,
            check_corpus=self.check_corpus,
            search_web=self.search_web,
            generate=self.generate,
        )

    def called(self):
        return [c[0] for c in self.calls]


def _ask(graph, question, thread_id="t1"):
    return graph.invoke({"messages": [HumanMessage(question)]}, config={"configurable": {"thread_id": thread_id}})


def test_corpus_route_answers_from_corpus_without_web():
    fake = FakeComponents(route="corpus", corpus_enough=True)
    result = _ask(build_graph(components=fake.components()), "What is RepoCoder?")

    # First turn: no history, so contextualize doesn't call the LLM.
    assert fake.called() == ["route", "search_corpus", "check_corpus", "generate"]
    assert result["route"] == "corpus"
    assert result["standalone_question"] == "What is RepoCoder?"
    assert result["documents"] == [PAPER]
    assert result["messages"][-1].content == "answer to: What is RepoCoder?"


def test_corpus_route_falls_back_to_web_when_corpus_is_not_enough():
    fake = FakeComponents(route="corpus", corpus_enough=False)
    result = _ask(build_graph(components=fake.components()), "How does RepoCoder compare to Copilot?")

    assert fake.called() == ["route", "search_corpus", "check_corpus", "search_web", "generate"]
    assert result["route"] == "web"
    # Corpus documents are kept alongside the web results.
    assert result["documents"] == [PAPER, WEB]


def test_web_route_skips_the_corpus():
    fake = FakeComponents(route="web")
    result = _ask(build_graph(components=fake.components()), "How do I enable LangSmith tracing?")

    assert fake.called() == ["route", "search_web", "generate"]
    assert ("search_web", "How do I enable LangSmith tracing?") in fake.calls
    assert result["route"] == "web"
    assert result["documents"] == [WEB]


def test_out_of_scope_declines_without_search_or_generation():
    fake = FakeComponents(route="out_of_scope")
    result = _ask(build_graph(components=fake.components()), "How do I bake cookies?")

    assert fake.called() == ["route"]
    assert result["route"] == "decline"
    assert result["documents"] == []
    assert result["messages"][-1].content == DECLINE_MESSAGE


def test_follow_up_is_contextualized_and_the_standalone_question_drives_routing_and_search():
    fake = FakeComponents(standalone="What benchmark did RepoCoder introduce?")
    graph = build_graph(components=fake.components())
    _ask(graph, "What is RepoCoder?", thread_id="f")
    fake.calls.clear()
    result = _ask(graph, "What benchmark did they introduce?", thread_id="f")

    assert fake.calls[0] == ("contextualize", "What benchmark did they introduce?", ["What is RepoCoder?", "answer to: What is RepoCoder?"])
    assert ("route", "What benchmark did RepoCoder introduce?") in fake.calls
    assert ("search_corpus", "What benchmark did RepoCoder introduce?") in fake.calls
    assert ("check_corpus", "What benchmark did RepoCoder introduce?", 1) in fake.calls
    assert result["question"] == "What benchmark did they introduce?"
    assert result["standalone_question"] == "What benchmark did RepoCoder introduce?"


def test_generate_sees_the_original_question_and_prior_turns():
    fake = FakeComponents()
    graph = build_graph(components=fake.components())
    _ask(graph, "What is RepoCoder?", thread_id="h")
    _ask(graph, "What benchmark did they introduce?", thread_id="h")

    name, question, history, _ = fake.calls[-1]
    assert name == "generate"
    assert question == "What benchmark did they introduce?"
    assert history == ["What is RepoCoder?", "answer to: What is RepoCoder?"]


def test_per_turn_state_is_reset_on_a_declined_follow_up():
    fake = FakeComponents(route="corpus")
    graph = build_graph(components=fake.components())
    _ask(graph, "What is RepoCoder?", thread_id="same")

    fake.route_to = "out_of_scope"
    fake.standalone = "How do I bake cookies?"
    result = _ask(graph, "Unrelated, how do I bake cookies?", thread_id="same")

    # The previous turn's papers don't leak into the declined turn.
    assert result["route"] == "decline"
    assert result["documents"] == []
    assert len(result["messages"]) == 4


def test_empty_contextualization_falls_back_to_the_question():
    fake = FakeComponents(standalone="")
    graph = build_graph(components=fake.components())
    _ask(graph, "What is RepoCoder?", thread_id="e")
    result = _ask(graph, "code review automation", thread_id="e")

    assert result["standalone_question"] == "code review automation"


def test_graph_compiles_without_checkpointer_for_studio():
    graph = build_graph(components=FakeComponents().components(), checkpointer=False)
    assert graph.checkpointer is None
    result = graph.invoke({"messages": [HumanMessage("What is RepoCoder?")]})
    assert result["route"] == "corpus"

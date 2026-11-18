"""Control-flow tests for the explicit graph, with fake components (no
network, no LLM): which nodes run, what ends up in state, and that per-turn
state doesn't leak across turns of the same thread."""
from langchain_core.messages import HumanMessage

from agent.components import AgentComponents
from agent.graph import build_graph
from agent.prompts import DECLINE_MESSAGE
from agent.schemas import CorpusCheck, ScopeDecision
from agent.state import Document

PAPER = Document(kind="corpus", title="RepoCoder", url="https://arxiv.org/abs/2303.12570", text="...", arxiv_id="2303.12570")
WEB = Document(kind="web", title="LangGraph docs", url="https://docs.langchain.com/x", text="...")


class FakeComponents:
    """Records every call; behavior configured per test."""

    def __init__(self, *, in_scope=True, search_query="rewritten query", corpus_enough=True):
        self.in_scope = in_scope
        self.search_query = search_query
        self.corpus_enough = corpus_enough
        self.calls = []

    def scope_check(self, question, history):
        self.calls.append(("scope_check", question, [m.content for m in history]))
        return ScopeDecision(in_scope=self.in_scope, search_query=self.search_query if self.in_scope else "")

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
            scope_check=self.scope_check,
            search_corpus=self.search_corpus,
            check_corpus=self.check_corpus,
            search_web=self.search_web,
            generate=self.generate,
        )

    def called(self):
        return [c[0] for c in self.calls]


def _ask(graph, question, thread_id="t1"):
    return graph.invoke({"messages": [HumanMessage(question)]}, config={"configurable": {"thread_id": thread_id}})


def test_corpus_path_answers_from_corpus_without_web():
    fake = FakeComponents(corpus_enough=True)
    result = _ask(build_graph(components=fake.components()), "What is RepoCoder?")

    assert fake.called() == ["scope_check", "search_corpus", "check_corpus", "generate"]
    assert result["route"] == "corpus"
    assert result["documents"] == [PAPER]
    assert result["messages"][-1].content == "answer to: What is RepoCoder?"


def test_corpus_search_uses_the_standalone_query_not_the_raw_question():
    fake = FakeComponents(search_query="RepoCoder benchmark")
    _ask(build_graph(components=fake.components()), "What benchmark did they use?")

    assert ("search_corpus", "RepoCoder benchmark") in fake.calls
    # ...but grading and generation still see the user's actual question.
    assert ("check_corpus", "What benchmark did they use?", 1) in fake.calls


def test_falls_back_to_web_when_corpus_is_not_enough():
    fake = FakeComponents(corpus_enough=False, search_query="langgraph human in the loop")
    result = _ask(build_graph(components=fake.components()), "How do I add human-in-the-loop to LangGraph?")

    assert fake.called() == ["scope_check", "search_corpus", "check_corpus", "search_web", "generate"]
    assert ("search_web", "langgraph human in the loop") in fake.calls
    assert result["route"] == "web"
    # Corpus documents are kept alongside the web results.
    assert result["documents"] == [PAPER, WEB]


def test_out_of_scope_declines_without_search_or_generation():
    fake = FakeComponents(in_scope=False)
    result = _ask(build_graph(components=fake.components()), "How do I bake cookies?")

    assert fake.called() == ["scope_check"]
    assert result["route"] == "decline"
    assert result["documents"] == []
    assert result["messages"][-1].content == DECLINE_MESSAGE


def test_follow_up_sees_history_and_per_turn_state_is_reset():
    fake = FakeComponents(corpus_enough=True)
    graph = build_graph(components=fake.components())
    _ask(graph, "What is RepoCoder?", thread_id="same")

    fake.in_scope = False
    result = _ask(graph, "Unrelated, how do I bake cookies?", thread_id="same")

    # Second turn's scope check saw the first turn as history.
    _, question, history = fake.calls[-1]
    assert question == "Unrelated, how do I bake cookies?"
    assert history == ["What is RepoCoder?", "answer to: What is RepoCoder?"]
    # The previous turn's papers don't leak into the declined turn.
    assert result["route"] == "decline"
    assert result["documents"] == []
    assert len(result["messages"]) == 4


def test_generate_receives_prior_turns_as_history():
    fake = FakeComponents()
    graph = build_graph(components=fake.components())
    _ask(graph, "What is RepoCoder?", thread_id="h")
    _ask(graph, "What benchmark did they introduce?", thread_id="h")

    name, question, history, _ = fake.calls[-1]
    assert name == "generate"
    assert question == "What benchmark did they introduce?"
    assert history == ["What is RepoCoder?", "answer to: What is RepoCoder?"]


def test_empty_search_query_falls_back_to_the_question():
    fake = FakeComponents(search_query="")
    _ask(build_graph(components=fake.components()), "code review automation")

    assert ("search_corpus", "code review automation") in fake.calls


def test_graph_compiles_without_checkpointer_for_studio():
    graph = build_graph(components=FakeComponents().components(), checkpointer=False)
    assert graph.checkpointer is None
    result = graph.invoke({"messages": [HumanMessage("What is RepoCoder?")]})
    assert result["route"] == "corpus"

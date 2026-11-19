"""Control-flow tests for the explicit graph, with fake components (no
network, no LLM): which nodes run, what ends up in state, and that per-turn
state doesn't leak across turns of the same thread."""
from langchain_core.messages import HumanMessage

from agent.components import AgentComponents
from agent.config import AgentConfig
from agent.graph import build_graph
from agent.prompts import DECLINE_MESSAGE
from agent.schemas import ClaimCheck, Critique, RouteDecision
from agent.state import Document


def _paper(title, n):
    return Document(kind="corpus", title=title, url=f"https://arxiv.org/abs/{n}", text="...", arxiv_id=str(n))


REPOCODER, RLCODER, CODERAG, GRAPHCODER = _paper("RepoCoder", 1), _paper("RLCoder", 2), _paper("CodeRAG", 3), _paper("GraphCoder", 4)
TESTGEN, TESTBENCH, TESTEVAL = _paper("TestGen", 5), _paper("TestBench", 6), _paper("TestEval", 7)
OFF_TOPIC = _paper("Flaky tests", 9)
WEB = Document(kind="web", title="LangGraph docs", url="https://docs.langchain.com/x", text="...")
LIVE = Document(kind="arxiv", title="New paper", url="https://arxiv.org/abs/2611.00001", text="...", arxiv_id="2611.00001")

CONFIG = AgentConfig(min_relevant_docs=2, max_relevant_docs=3, max_query_rewrites=1, max_iterations=3)

PASS = Critique(claims=[ClaimCheck(claim="ok", supported=True)], missing_aspects=[], problems=[])


def _unsupported(claim="RepoCoder reaches 99%"):
    return Critique(claims=[ClaimCheck(claim="ok", supported=True), ClaimCheck(claim=claim, supported=False, note="not stated")], missing_aspects=[], problems=[])


def _missing(*aspects):
    return Critique(claims=[ClaimCheck(claim="ok", supported=True)], missing_aspects=list(aspects), problems=[])
RELEVANT = {"RepoCoder", "RLCoder", "CodeRAG", "GraphCoder", "TestGen", "TestBench", "TestEval"}


class FakeComponents:
    """Records every call; behavior configured per test.

    corpus: {query: [documents]} - what a corpus search returns for a query
        (unknown queries return [OFF_TOPIC]). sub_queries: what decompose
        returns (default: the question itself). critiques: what reflect
        returns, one per draft (default / when exhausted: PASS).
    """

    def __init__(self, *, route="corpus", standalone="standalone question", sub_queries=None, corpus=None, relevant=RELEVANT, critiques=None):
        self.critiques = list(critiques or [])
        self.route_to = route
        self.standalone = standalone
        self.sub_queries = sub_queries
        self.corpus = corpus or {}
        self.relevant = relevant
        self.calls = []
        self.revisions = []

    def contextualize(self, question, history):
        self.calls.append(("contextualize", question, [m.content for m in history]))
        return self.standalone

    def route(self, question):
        self.calls.append(("route", question))
        return RouteDecision(reasoning="fake", route=self.route_to)

    def decompose(self, question):
        self.calls.append(("decompose", question))
        return self.sub_queries or [question]

    def search_corpus(self, query):
        self.calls.append(("search_corpus", query))
        return self.corpus.get(query, [OFF_TOPIC])

    def grade_documents(self, question, documents):
        self.calls.append(("grade_documents", question, [d.title for d in documents]))
        return [d.title in self.relevant for d in documents]

    def rewrite_query(self, question, query, found):
        self.calls.append(("rewrite_query", question, query))
        return f"rewritten: {question}"

    def search_web(self, query):
        self.calls.append(("search_web", query))
        return [WEB]

    def research(self, question, missing, known):
        self.calls.append(("research", question, missing, [d.title for d in known]))
        return [WEB, LIVE]

    def generate(self, question, history, documents, revision=None):
        self.calls.append(("generate", question, [m.content for m in history], [d.title for d in documents]))
        self.revisions.append(revision)
        return f"answer to: {question}" + (f" (revision {len(self.revisions) - 1})" if revision else "")

    def reflect(self, question, draft, documents):
        self.calls.append(("reflect", question, draft, [d.title for d in documents]))
        return self.critiques.pop(0) if self.critiques else PASS

    def components(self):
        return AgentComponents(
            contextualize=self.contextualize,
            route=self.route,
            decompose=self.decompose,
            search_corpus=self.search_corpus,
            grade_documents=self.grade_documents,
            rewrite_query=self.rewrite_query,
            search_web=self.search_web,
            research=self.research,
            generate=self.generate,
            reflect=self.reflect,
        )

    def called(self):
        return [c[0] for c in self.calls]


def _graph(fake, config=CONFIG, **kw):
    return build_graph(config, components=fake.components(), **kw)


def _ask(graph, question, thread_id="t1"):
    return graph.invoke({"messages": [HumanMessage(question)]}, config={"configurable": {"thread_id": thread_id}})


Q = "How does RAG help code completion?"


# --- corpus path, one sub-query ------------------------------------------------


def test_enough_relevant_papers_on_first_search_answers_from_corpus():
    fake = FakeComponents(corpus={Q: [REPOCODER, OFF_TOPIC, RLCODER]})
    result = _ask(_graph(fake), Q)

    # First turn: no history, so contextualize doesn't call the LLM.
    assert fake.called() == ["route", "decompose", "search_corpus", "grade_documents", "generate", "reflect"]
    assert result["route"] == "corpus"
    assert result["sub_queries"] == [Q]
    # Only graded-relevant papers reach generate; everything retrieved is kept for evals.
    assert result["documents"] == [REPOCODER, RLCODER]
    assert result["retrieved"] == [REPOCODER, OFF_TOPIC, RLCODER]


def test_at_most_max_relevant_docs_are_kept_per_sub_query():
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER, CODERAG, GRAPHCODER]})
    result = _ask(_graph(fake), Q)

    assert result["documents"] == [REPOCODER, RLCODER, CODERAG]  # best-ranked 3
    assert len(result["retrieved"]) == 4


def test_too_few_relevant_papers_rewrites_the_query_and_searches_again():
    fake = FakeComponents(corpus={Q: [REPOCODER, OFF_TOPIC], f"rewritten: {Q}": [REPOCODER, CODERAG]})
    result = _ask(_graph(fake), Q)

    assert fake.called() == [
        "route", "decompose", "search_corpus", "grade_documents", "rewrite_query",
        "search_corpus", "grade_documents", "generate", "reflect",
    ]
    # The retry only grades what is new (RepoCoder was already graded).
    assert ("grade_documents", Q, ["CodeRAG"]) in fake.calls
    assert result["route"] == "corpus"
    assert result["documents"] == [REPOCODER, CODERAG]
    assert result["sub_results"][0].query_rewrites == 1


def test_still_not_enough_after_the_rewrite_falls_back_to_research():
    fake = FakeComponents(corpus={Q: [REPOCODER, OFF_TOPIC]})  # the rewrite finds nothing new
    result = _ask(_graph(fake), Q)

    # Nothing new on the second search: no second grading call.
    assert fake.called() == [
        "route", "decompose", "search_corpus", "grade_documents", "rewrite_query",
        "search_corpus", "research", "generate", "reflect",
    ]
    # The research sub-agent is told what is missing and what the corpus gave...
    assert ("research", Q, [Q], ["RepoCoder"]) in fake.calls
    # ...and the relevant paper is kept next to what it found.
    assert result["route"] == "web"
    assert result["documents"] == [REPOCODER, WEB, LIVE]


def test_the_rewrite_loop_is_bounded_by_max_query_rewrites():
    fake = FakeComponents(relevant=set())
    _ask(_graph(fake, AgentConfig(max_query_rewrites=3)), "q")

    assert fake.called().count("rewrite_query") == 3
    assert fake.called()[-3:] == ["research", "generate", "reflect"]


def test_with_the_default_config_one_relevant_paper_covers_a_sub_query():
    fake = FakeComponents(corpus={Q: [RLCODER, OFF_TOPIC]})
    result = _ask(_graph(fake, AgentConfig()), Q)

    assert "rewrite_query" not in fake.called() and "research" not in fake.called()
    assert result["route"] == "corpus"
    assert result["documents"] == [RLCODER]


# --- corpus path, several sub-queries ----------------------------------------------


TWO_PART = "How does RAG help code completion, and how are LLMs used to generate tests?"
PART_1, PART_2 = "RAG for code completion", "LLMs for test generation"


def test_each_sub_query_is_searched_and_graded_on_its_own_and_results_merge_in_order():
    fake = FakeComponents(
        sub_queries=[PART_1, PART_2],
        corpus={PART_1: [REPOCODER, RLCODER, CODERAG, GRAPHCODER], PART_2: [TESTGEN, OFF_TOPIC, TESTBENCH, TESTEVAL]},
    )
    result = _ask(_graph(fake), TWO_PART)

    # Each part graded against its own sub-query, not the whole question.
    assert ("grade_documents", PART_1, ["RepoCoder", "RLCoder", "CodeRAG", "GraphCoder"]) in fake.calls
    assert ("grade_documents", PART_2, ["TestGen", "Flaky tests", "TestBench", "TestEval"]) in fake.calls
    # Up to 3 relevant per part, merged in sub-query order.
    assert result["documents"] == [REPOCODER, RLCODER, CODERAG, TESTGEN, TESTBENCH, TESTEVAL]
    assert result["route"] == "corpus"
    assert result["missing"] == []
    assert len(result["retrieved"]) == 8


def test_only_the_part_the_corpus_misses_goes_to_research():
    fake = FakeComponents(sub_queries=[PART_1, PART_2], corpus={PART_1: [REPOCODER, RLCODER]})  # PART_2: off-topic only
    result = _ask(_graph(fake), TWO_PART)

    assert fake.called().count("rewrite_query") == 1  # only part 2 retried
    # Research gets the whole question for context, but only the missing part.
    assert ("research", TWO_PART, [PART_2], ["RepoCoder", "RLCoder"]) in fake.calls
    assert result["missing"] == [PART_2]
    assert result["route"] == "web"
    assert result["documents"] == [REPOCODER, RLCODER, WEB, LIVE]


def test_a_chunk_found_by_two_sub_queries_is_kept_once():
    fake = FakeComponents(sub_queries=[PART_1, PART_2], corpus={PART_1: [REPOCODER, RLCODER], PART_2: [RLCODER, TESTGEN]})
    result = _ask(_graph(fake), TWO_PART)

    assert result["documents"] == [REPOCODER, RLCODER, TESTGEN]


# --- other routes ------------------------------------------------------------


def test_web_route_skips_the_corpus():
    fake = FakeComponents(route="web")
    result = _ask(_graph(fake), "How do I enable LangSmith tracing?")

    assert fake.called() == ["route", "search_web", "generate", "reflect"]
    assert result["route"] == "web"
    assert result["documents"] == [WEB]
    assert result["retrieved"] == []


def test_out_of_scope_declines_without_search_or_generation():
    fake = FakeComponents(route="out_of_scope")
    result = _ask(_graph(fake), "How do I bake cookies?")

    assert fake.called() == ["route"]
    assert result["route"] == "decline"
    assert result["documents"] == []
    assert result["messages"][-1].content == DECLINE_MESSAGE


# --- conversation --------------------------------------------------------------


def test_follow_up_is_contextualized_and_the_standalone_question_drives_routing_and_decomposition():
    fake = FakeComponents(standalone="What benchmark did RepoCoder introduce?")
    graph = _graph(fake)
    _ask(graph, "What is RepoCoder?", thread_id="f")
    fake.calls.clear()
    result = _ask(graph, "What benchmark did they introduce?", thread_id="f")

    assert fake.calls[0] == ("contextualize", "What benchmark did they introduce?", ["What is RepoCoder?", "answer to: What is RepoCoder?"])
    assert ("route", "What benchmark did RepoCoder introduce?") in fake.calls
    assert ("decompose", "What benchmark did RepoCoder introduce?") in fake.calls
    assert ("search_corpus", "What benchmark did RepoCoder introduce?") in fake.calls
    assert result["question"] == "What benchmark did they introduce?"


def test_generate_sees_the_original_question_and_prior_turns():
    fake = FakeComponents()
    graph = _graph(fake)
    _ask(graph, "What is RepoCoder?", thread_id="h")
    _ask(graph, "What benchmark did they introduce?", thread_id="h")

    name, question, history, _ = fake.calls[-2]
    assert name == "generate"
    assert question == "What benchmark did they introduce?"
    assert history == ["What is RepoCoder?", "answer to: What is RepoCoder?"]


def test_per_turn_state_is_reset_between_turns():
    fake = FakeComponents(sub_queries=[PART_1, PART_2], corpus={PART_1: [REPOCODER, RLCODER]})
    graph = _graph(fake)
    _ask(graph, TWO_PART, thread_id="same")

    fake.route_to = "out_of_scope"
    fake.standalone = "How do I bake cookies?"
    result = _ask(graph, "Unrelated, how do I bake cookies?", thread_id="same")

    assert result["route"] == "decline"
    assert result["documents"] == []
    assert result["retrieved"] == []
    assert result["sub_results"] == []  # append-reducer field cleared with Overwrite
    assert result["missing"] == []
    assert result["draft"] == "" and result["critique"] is None and result["iteration"] == 0
    assert len(result["messages"]) == 4


def test_graph_compiles_without_checkpointer_for_studio():
    graph = _graph(FakeComponents(corpus={"What is RepoCoder?": [REPOCODER, RLCODER]}), checkpointer=False)
    assert graph.checkpointer is None
    result = graph.invoke({"messages": [HumanMessage("What is RepoCoder?")]})
    assert result["route"] == "corpus"


# --- self-reflection -------------------------------------------------------------


def test_a_draft_that_passes_the_check_is_the_answer():
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER]})
    result = _ask(_graph(fake), Q)

    assert fake.called()[-2:] == ["generate", "reflect"]
    # reflect checks the draft against exactly what generate saw.
    assert ("reflect", Q, f"answer to: {Q}", ["RepoCoder", "RLCoder"]) in fake.calls
    assert result["iteration"] == 1
    assert result["messages"][-1].content == f"answer to: {Q}"


def test_unsupported_claims_trigger_a_revision_with_the_critique():
    bad = _unsupported()
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER]}, critiques=[bad, PASS])
    result = _ask(_graph(fake), Q)

    assert fake.called()[-4:] == ["generate", "reflect", "generate", "reflect"]
    assert fake.revisions == [None, (f"answer to: {Q}", bad)]  # the revision sees the draft and its critique
    assert result["iteration"] == 2
    assert result["messages"][-1].content == f"answer to: {Q} (revision 1)"


def test_missing_aspects_get_one_targeted_search_then_a_revision():
    aspect = "LLMs for test generation"
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER], aspect: [TESTGEN, OFF_TOPIC]}, critiques=[_missing(aspect), PASS])
    result = _ask(_graph(fake), Q)

    i = fake.called().index("search_corpus", 3)
    assert fake.called()[i - 2:] == ["generate", "reflect", "search_corpus", "grade_documents", "generate", "reflect"]
    assert ("grade_documents", aspect, ["TestGen", "Flaky tests"]) in fake.calls
    assert result["documents"] == [REPOCODER, RLCODER, TESTGEN]
    assert TESTGEN in result["retrieved"]
    assert result["route"] == "corpus"
    assert result["aspects_searched"] is True


def test_an_aspect_the_corpus_lacks_goes_to_research():
    aspect = "GPT-7 release date"
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER]}, critiques=[_missing(aspect), PASS])
    result = _ask(_graph(fake), Q)

    assert ("research", Q, [aspect], ["RepoCoder", "RLCoder"]) in fake.calls
    assert result["documents"] == [REPOCODER, RLCODER, WEB, LIVE]
    assert result["route"] == "web"


def test_missing_aspects_are_searched_only_once_per_turn():
    aspect = "LLMs for test generation"
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER], aspect: [TESTGEN]}, critiques=[_missing(aspect), _missing(aspect), PASS])
    _ask(_graph(fake), Q)

    assert [c[1] for c in fake.calls if c[0] == "search_corpus"].count(aspect) == 1
    assert fake.called().count("generate") == 3  # the second time it just revises


def test_the_loop_stops_at_max_iterations_and_returns_the_best_draft_with_a_caveat():
    worse = Critique(claims=[ClaimCheck(claim="a", supported=False), ClaimCheck(claim="b", supported=False)], missing_aspects=[], problems=[])
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER]}, critiques=[worse, _unsupported(), worse, PASS])
    result = _ask(_graph(fake), Q)

    assert fake.called().count("generate") == 3 and fake.called().count("reflect") == 3
    # The best of the three is draft 2 (half its claims supported), not the last one.
    answer = result["messages"][-1].content
    assert answer.startswith(f"answer to: {Q} (revision 1)")
    assert "didn't fully pass the fact-check" in answer
    assert len(result["messages"]) == 2  # drafts never become chat messages


def test_web_route_answers_are_reflected_too():
    fake = FakeComponents(route="web", critiques=[_unsupported(), PASS])
    result = _ask(_graph(fake), "How do I enable LangSmith tracing?")

    assert fake.called() == ["route", "search_web", "generate", "reflect", "generate", "reflect"]
    assert result["documents"] == [WEB]


def test_reflection_state_is_reset_between_turns():
    fake = FakeComponents(corpus={Q: [REPOCODER, RLCODER]}, critiques=[_unsupported(), PASS])
    graph = _graph(fake)
    _ask(graph, Q, thread_id="r")
    result = _ask(graph, Q, thread_id="r")

    assert result["iteration"] == 1  # not 3
    assert fake.revisions[-1] is None  # the new turn's first draft isn't a revision of the old one

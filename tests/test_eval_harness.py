from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent.state import Document
from evals.evaluators import NOT_APPLICABLE, cost_signals, retrieval_hit, retrieval_recall, route_correct
from evals.targets import UsageCounter, make_graph_target, outputs_from_state

PAPER_A = Document(kind="corpus", title="TA", url="https://arxiv.org/abs/a", text="abstract a", arxiv_id="a")
PAPER_A_CHUNK2 = PAPER_A.model_copy(update={"text": "abstract a, chunk 2"})
PAPER_B = Document(kind="corpus", title="TB", url="https://arxiv.org/abs/b", text="abstract b", arxiv_id="b")
WEB = Document(kind="web", title="W", url="https://example.com", text="web text")


def _usage(calls=2, inp=20, out=10):
    u = UsageCounter()
    u.llm_calls, u.input_tokens, u.output_tokens = calls, inp, out
    return u


def _state(route, documents, answer="answer"):
    return {"messages": [HumanMessage("q"), AIMessage(answer)], "route": route, "documents": documents}


def test_outputs_corpus_turn():
    out = outputs_from_state(_state("corpus", [PAPER_A, PAPER_A_CHUNK2, PAPER_B]), _usage())
    assert out["route"] == "corpus"
    assert out["answer"] == "answer"
    assert out["retrieved_ids"] == ["a", "b"]  # de-duplicated, in order
    assert "[a] TA\nabstract a" in out["context"]
    assert (out["llm_calls"], out["input_tokens"], out["output_tokens"]) == (2, 20, 10)


def test_outputs_web_turn_keeps_corpus_ids_and_web_context():
    out = outputs_from_state(_state("web", [PAPER_A, WEB]), _usage())
    assert out["route"] == "web"
    assert out["retrieved_ids"] == ["a"]
    assert "[https://example.com] W\nweb text" in out["context"]


def test_outputs_retrieval_metrics_use_everything_retrieved_not_only_graded_relevant():
    state = _state("corpus", [PAPER_A])  # grading kept only PAPER_A...
    state["retrieved"] = [PAPER_A, PAPER_B]  # ...out of what the corpus returned
    out = outputs_from_state(state, _usage())
    assert out["retrieved_ids"] == ["a", "b"]
    assert "abstract b" not in out["context"]  # the judge sees what generate saw


def test_outputs_live_arxiv_documents_are_context_but_not_corpus_retrieval():
    live = Document(kind="arxiv", title="TL", url="https://arxiv.org/abs/l", text="live abstract", arxiv_id="l")
    state = _state("web", [PAPER_A, live])
    state["retrieved"] = [PAPER_A]
    out = outputs_from_state(state, _usage())
    assert out["retrieved_ids"] == ["a"]
    assert "[l] TL\nlive abstract" in out["context"]


def test_outputs_decline_turn():
    out = outputs_from_state(_state("decline", []), _usage(calls=1))
    assert out["route"] == "decline"
    assert out["retrieved_ids"] == []
    assert out["context"] == ""


def _llm_result(inp, out):
    msg = AIMessage("x", usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out})
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_usage_counter_sums_calls_and_tokens():
    u = UsageCounter()
    u.on_llm_end(_llm_result(10, 5))
    u.on_llm_end(_llm_result(7, 3))
    assert (u.llm_calls, u.input_tokens, u.output_tokens) == (2, 17, 8)


def test_graph_target_replays_history_and_passes_callbacks():
    captured = {}

    class FakeGraph:
        def invoke(self, inputs, config):
            captured["inputs"], captured["config"] = inputs, config
            return _state("corpus", [PAPER_A])

    target = make_graph_target(FakeGraph())
    out = target({"question": "follow-up?", "chat_history": [{"role": "user", "content": "first"}]})

    assert captured["inputs"]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "follow-up?"},
    ]
    assert captured["config"]["configurable"]["thread_id"].startswith("eval-")
    assert isinstance(captured["config"]["callbacks"][0], UsageCounter)
    assert out["retrieved_ids"] == ["a"]


def _outputs(**kw):
    base = {"route": "corpus", "retrieved_ids": [], "llm_calls": 2, "input_tokens": 10, "output_tokens": 5}
    base.update(kw)
    return base


def test_route_correct_uses_acceptable_routes():
    ref = {"expected_route": "web", "acceptable_routes": ["web", "corpus"]}
    assert route_correct(_outputs(route="corpus"), ref)["score"] == 1
    assert route_correct(_outputs(route="decline"), ref)["score"] == 0
    assert route_correct(_outputs(route="web"), {"expected_route": "corpus"})["score"] == 0


def test_retrieval_metrics():
    ref = {"expected_arxiv_ids": ["a", "b", "c", "d"]}
    out = _outputs(retrieved_ids=["b", "x", "d"])
    assert retrieval_hit(out, ref)["score"] == 1
    assert retrieval_recall(out, ref)["score"] == 0.5
    assert retrieval_hit(_outputs(retrieved_ids=["x"]), ref)["score"] == 0


def test_retrieval_metrics_not_applicable_without_expected_papers():
    ref = {"expected_arxiv_ids": []}
    assert retrieval_hit(_outputs(), ref) == NOT_APPLICABLE
    assert retrieval_recall(_outputs(), ref) == NOT_APPLICABLE


def test_cost_signals():
    scores = {r["key"]: r["score"] for r in cost_signals(_outputs())["results"]}
    assert scores == {"llm_calls": 2, "ktokens": 0.01}


def test_cost_signals_stay_within_langsmith_score_limit():
    scores = {r["key"]: r["score"] for r in cost_signals(_outputs(input_tokens=160_000, output_tokens=1_197))["results"]}
    assert scores["ktokens"] == 161.2 < 99_999

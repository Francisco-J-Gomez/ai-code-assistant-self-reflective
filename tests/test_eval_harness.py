import json

from langchain_core.messages import AIMessage, ToolMessage

from evals.evaluators import NOT_APPLICABLE, cost_signals, retrieval_hit, retrieval_recall, route_correct
from evals.targets import summarize_turn


def _papers(*ids):
    return ToolMessage(
        name="search_papers",
        tool_call_id="1",
        content=json.dumps([
            {"arxiv_id": i, "title": f"T{i}", "source_url": f"https://arxiv.org/abs/{i}", "text": f"abstract {i}"}
            for i in ids
        ]),
    )


def _web():
    return ToolMessage(
        name="tavily_search",
        tool_call_id="2",
        content=json.dumps({"results": [{"title": "W", "url": "https://example.com", "content": "web text"}]}),
    )


def _ai(text, tokens=(10, 5)):
    return AIMessage(content=text, usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)})


def test_summarize_corpus_turn():
    out = summarize_turn([_ai(""), _papers("a", "b"), _ai("answer")])
    assert out["route"] == "corpus"
    assert out["answer"] == "answer"
    assert out["retrieved_ids"] == ["a", "b"]
    assert "abstract a" in out["context"]
    assert out["llm_calls"] == 2
    assert (out["input_tokens"], out["output_tokens"]) == (20, 10)


def test_summarize_web_turn_wins_over_corpus():
    out = summarize_turn([_ai(""), _papers("a"), _ai(""), _web(), _ai("answer")])
    assert out["route"] == "web"
    assert out["tools_used"] == ["search_papers", "tavily_search"]
    assert "web text" in out["context"]


def test_summarize_no_tool_is_decline():
    out = summarize_turn([_ai("out of scope")])
    assert out["route"] == "decline"
    assert out["retrieved_ids"] == []
    assert out["context"] == ""


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

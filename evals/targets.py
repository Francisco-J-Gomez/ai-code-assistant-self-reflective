"""Eval targets: run an agent on one dataset example and return a standard
output dict that every evaluator in evals/evaluators.py understands.

Keeping this shape stable across phases is what makes experiments
comparable with the Phase 0 baseline (phase0-baseline-e2c91149, which used a
ReAct-agent target reading the same keys out of tool messages - see the
phase-0-observability-baseline branch).

Output keys:
    answer           final answer text
    route            "corpus" | "web" | "decline"
    retrieved_ids    arXiv ids retrieved from the corpus, in order, de-duplicated
    context          retrieved text the answer should be grounded in
    llm_calls        number of LLM calls in the turn
    input_tokens / output_tokens
"""
import uuid

from langchain_core.callbacks import BaseCallbackHandler

# Enough for the judge to see every retrieved document without blowing up
# grading cost on a long web result.
MAX_CONTEXT_CHARS = 30_000


class UsageCounter(BaseCallbackHandler):
    """Counts LLM calls and token usage across every node of one graph run."""

    def __init__(self):
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self.llm_calls += 1
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)


def outputs_from_state(state: dict, usage: UsageCounter) -> dict:
    """Build the standard output dict from the graph's final state."""
    documents = state.get("documents", [])
    retrieved_ids = list(dict.fromkeys(d.arxiv_id for d in documents if d.kind == "corpus" and d.arxiv_id))
    context = "\n\n".join(
        f"[{d.arxiv_id if d.kind == 'corpus' else d.url}] {d.title}\n{d.text}" for d in documents
    )[:MAX_CONTEXT_CHARS]
    return {
        "answer": state["messages"][-1].content,
        "route": state.get("route") or "decline",
        "retrieved_ids": retrieved_ids,
        "context": context,
        "llm_calls": usage.llm_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def make_graph_target(graph):
    """Target for the explicit StateGraph agent (Phase 1+).

    Each example runs in a fresh thread; chat_history is replayed as prior
    messages so follow-up questions see the same context a user would have.
    """

    def target(inputs: dict) -> dict:
        messages = [*inputs.get("chat_history", []), {"role": "user", "content": inputs["question"]}]
        usage = UsageCounter()
        state = graph.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}, "callbacks": [usage]},
        )
        return outputs_from_state(state, usage)

    return target

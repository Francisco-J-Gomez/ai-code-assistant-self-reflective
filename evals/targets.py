"""Eval targets: run an agent on one dataset example and return a standard
output dict that every evaluator in evals/evaluators.py understands.

Keeping this shape stable across phases is what makes experiments
comparable: when the agent becomes an explicit StateGraph (Phase 1), only a
new target is needed - it must return the same keys, read from graph state
instead of from tool messages.

Output keys:
    answer           final answer text
    route            "corpus" | "web" | "decline" (decline = no tool used)
    retrieved_ids    arXiv ids returned by corpus search, in order, de-duplicated
    context          retrieved text the answer should be grounded in
    tools_used       tool names called, in order
    llm_calls        number of LLM responses in the turn
    input_tokens / output_tokens
"""
import json
import uuid

from agent.results import classify_answer_source, extract_sources

# Enough for the judge to see every retrieved document without blowing up
# grading cost on a long web result.
MAX_CONTEXT_CHARS = 30_000


def summarize_turn(new_messages: list) -> dict:
    """Build the standard output dict from the messages one agent turn added."""
    ai_messages = [m for m in new_messages if getattr(m, "type", None) == "ai"]
    tool_messages = [m for m in new_messages if getattr(m, "type", None) == "tool"]

    kind = classify_answer_source(new_messages)["kind"]
    route = {"web": "web", "corpus": "corpus"}.get(kind, "decline")

    sources = extract_sources(new_messages)
    retrieved_ids = [s["arxiv_id"] for s in sources if s.get("arxiv_id")]

    input_tokens = output_tokens = 0
    for m in ai_messages:
        usage = getattr(m, "usage_metadata", None) or {}
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

    return {
        "answer": ai_messages[-1].content if ai_messages else "",
        "route": route,
        "retrieved_ids": retrieved_ids,
        "context": _context_from_tools(tool_messages),
        "tools_used": [m.name for m in tool_messages],
        "llm_calls": len(ai_messages),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _context_from_tools(tool_messages: list) -> str:
    chunks = []
    for m in tool_messages:
        try:
            payload = json.loads(m.content)
        except (TypeError, ValueError):
            chunks.append(str(m.content))
            continue
        if m.name == "search_papers":
            for item in payload:
                chunks.append(f"[{item.get('arxiv_id')}] {item.get('title')}\n{item.get('text', '')}")
        elif m.name == "tavily_search":
            for item in payload.get("results", []):
                chunks.append(f"[{item.get('url')}] {item.get('title')}\n{item.get('content', '')}")
    return "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]


def make_react_agent_target(agent):
    """Target for the current create_agent ReAct agent (Phase 0 baseline).

    Each example runs in a fresh thread; chat_history is replayed as prior
    messages so follow-up questions see the same context a user would have.
    """

    def target(inputs: dict) -> dict:
        messages = [*inputs.get("chat_history", []), {"role": "user", "content": inputs["question"]}]
        result = agent.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}},
        )
        return summarize_turn(result["messages"][len(messages):])

    return target

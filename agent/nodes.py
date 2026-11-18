"""Graph nodes and routing functions.

Every node is a plain function `(state, components) -> partial state update`;
build_graph() in agent/graph.py binds the components and wires them up. The
nodes hold the per-step logic, the graph holds the control flow, and
AgentComponents holds everything that touches the network - so each piece
can be tested on its own with fakes.

When later phases add more nodes (grade_docs, rewrite_query, reflect,
revise...), this module splits into a package by concern
(nodes/retrieval.py, nodes/reflection.py, ...).
"""
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agent.components import AgentComponents
from agent.prompts import DECLINE_MESSAGE
from agent.state import AgentState


def split_latest_question(messages: list[AnyMessage]) -> tuple[str, list[AnyMessage]]:
    """(latest user message, everything before it)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i].content, messages[:i]
    raise ValueError("No user message in state")


# --- nodes -----------------------------------------------------------------


def contextualize(state: AgentState, c: AgentComponents) -> dict:
    """Turn the latest message into a standalone question, so follow-ups
    ("what benchmark did they introduce?") route and retrieve correctly.
    Skips the LLM on the first turn - there is nothing to resolve.

    Also the start of every turn, so it resets the per-turn fields.
    """
    question, history = split_latest_question(state.messages)
    standalone = (c.contextualize(question, history) if history else "") or question
    return {
        "question": question,
        "standalone_question": standalone,
        "route": None,
        "documents": [],  # reset: state persists across turns
    }


def router(state: AgentState, c: AgentComponents) -> dict:
    decision = c.route(state.standalone_question)
    return {"route": "decline" if decision.route == "out_of_scope" else decision.route}


def decline(state: AgentState, c: AgentComponents) -> dict:
    return {"messages": [AIMessage(DECLINE_MESSAGE)]}


def retrieve_corpus(state: AgentState, c: AgentComponents) -> dict:
    return {"documents": c.search_corpus(state.standalone_question)}


def check_corpus(state: AgentState, c: AgentComponents) -> dict:
    verdict = c.check_corpus(state.standalone_question, state.documents)
    return {"route": "corpus" if verdict.answers_question else "web"}


def web_search(state: AgentState, c: AgentComponents) -> dict:
    # On the fallback path, keep the corpus documents too: generate may still
    # use a relevant paper. (On the direct web route there are none.)
    return {"documents": [*state.documents, *c.search_web(state.standalone_question)]}


def generate(state: AgentState, c: AgentComponents) -> dict:
    _, history = split_latest_question(state.messages)
    answer = c.generate(state.question, history, state.documents)
    return {"messages": [AIMessage(answer)]}


# --- routing (conditional edges) -------------------------------------------


def route_after_router(state: AgentState) -> Literal["decline", "retrieve_corpus", "web_search"]:
    return {"decline": "decline", "web": "web_search"}.get(state.route, "retrieve_corpus")


def route_after_check_corpus(state: AgentState) -> Literal["generate", "web_search"]:
    return "generate" if state.route == "corpus" else "web_search"

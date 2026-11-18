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


def scope_check(state: AgentState, c: AgentComponents) -> dict:
    """Decide scope and write a standalone search query. Also the start of
    every turn, so it resets the per-turn fields."""
    question, history = split_latest_question(state.messages)
    decision = c.scope_check(question, history)
    return {
        "question": question,
        "search_query": decision.search_query or question,
        "route": None if decision.in_scope else "decline",
        "documents": [],  # reset: state persists across turns
    }


def decline(state: AgentState, c: AgentComponents) -> dict:
    return {"messages": [AIMessage(DECLINE_MESSAGE)]}


def retrieve_corpus(state: AgentState, c: AgentComponents) -> dict:
    return {"documents": c.search_corpus(state.search_query)}


def check_corpus(state: AgentState, c: AgentComponents) -> dict:
    verdict = c.check_corpus(state.question, state.documents)
    return {"route": "corpus" if verdict.answers_question else "web"}


def web_search(state: AgentState, c: AgentComponents) -> dict:
    # Keep the corpus documents too: generate may still use a relevant
    # paper, and the evals measure corpus retrieval on every in-scope turn.
    return {"documents": [*state.documents, *c.search_web(state.search_query)]}


def generate(state: AgentState, c: AgentComponents) -> dict:
    _, history = split_latest_question(state.messages)
    answer = c.generate(state.question, history, state.documents)
    return {"messages": [AIMessage(answer)]}


# --- routing (conditional edges) -------------------------------------------


def route_after_scope_check(state: AgentState) -> Literal["decline", "retrieve_corpus"]:
    return "decline" if state.route == "decline" else "retrieve_corpus"


def route_after_check_corpus(state: AgentState) -> Literal["generate", "web_search"]:
    return "generate" if state.route == "corpus" else "web_search"

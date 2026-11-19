"""Graph nodes and routing functions.

Every node is a plain function `(state, components[, settings]) -> partial
state update`; build_graph() in agent/graph.py binds the dependencies and
wires them up. The nodes hold the per-step logic, the graph holds the
control flow, and AgentComponents holds everything that touches the network
- so each piece can be tested on its own with fakes.

Nodes of the corpus_search subgraph take CorpusSearchState (one
sub-query); the rest take AgentState. When Phase 4 adds the reflection nodes
(reflect, revise), this module splits into a package by concern
(nodes/retrieval.py, nodes/reflection.py, ...).
"""
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.types import Overwrite, Send

from agent.components import AgentComponents
from agent.config import AgentConfig
from agent.prompts import DECLINE_MESSAGE
from agent.state import AgentState, CorpusSearchState, SubQueryResult


def split_latest_question(messages: list[AnyMessage]) -> tuple[str, list[AnyMessage]]:
    """(latest user message, everything before it)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i].content, messages[:i]
    raise ValueError("No user message in state")


def dedupe(documents):
    """Same chunk found for two sub-queries -> keep the first, in order."""
    seen, unique = set(), []
    for d in documents:
        if d.key not in seen:
            seen.add(d.key)
            unique.append(d)
    return unique


# --- understanding the question ---------------------------------------------


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
        # reset: state persists across turns. sub_results has an append
        # reducer, so it needs Overwrite to be cleared.
        "sub_queries": [],
        "sub_results": Overwrite([]),
        "missing": [],
        "retrieved": [],
        "documents": [],
    }


def router(state: AgentState, c: AgentComponents) -> dict:
    decision = c.route(state.standalone_question)
    return {"route": "decline" if decision.route == "out_of_scope" else decision.route}


def decline(state: AgentState, c: AgentComponents) -> dict:
    return {"messages": [AIMessage(DECLINE_MESSAGE)]}


# --- corpus path: decompose, then one corrective search per sub-query --------


def decompose(state: AgentState, c: AgentComponents) -> dict:
    """Split a multi-part question into self-contained sub-queries (a
    single-part question stays one query)."""
    return {"sub_queries": c.decompose(state.standalone_question) or [state.standalone_question]}


def fan_out_sub_queries(state: AgentState) -> list[Send]:
    """One corpus_search run per sub-query, in parallel."""
    return [Send("corpus_search", {"index": i, "sub_query": q}) for i, q in enumerate(state.sub_queries)]


def merge_results(state: AgentState) -> dict:
    """Join the parallel corpus_search runs: relevant chunks of every
    sub-query (in sub-query order), everything retrieved (for evals), and
    the sub-queries the corpus didn't cover."""
    results = sorted(state.sub_results, key=lambda r: r.index)
    missing = [r.sub_query for r in results if not r.enough]
    return {
        "documents": dedupe([d for r in results for d in r.documents]),
        "retrieved": dedupe([d for r in results for d in r.retrieved]),
        "missing": missing,
        "route": state.route if missing else "corpus",
    }


def fallback_research(state: AgentState, c: AgentComponents) -> dict:
    """Some part of the question isn't covered by the corpus: let the
    research sub-agent search the web and live arXiv for those parts.
    Relevant corpus documents found so far are kept."""
    found = c.research(state.standalone_question, state.missing, state.documents)
    return {"documents": [*state.documents, *found], "route": "web"}


# --- corpus_search subgraph (CorpusSearchState, one sub-query) -----------------


def retrieve_corpus(state: CorpusSearchState, c: AgentComponents) -> dict:
    """Search the corpus with the current query. Only chunks not seen earlier
    in this run become candidates, so a retry after rewrite_query doesn't
    re-grade (or double-count) what the first search already returned."""
    seen = {d.key for d in state.retrieved}
    new = [d for d in c.search_corpus(state.search_query or state.sub_query) if d.key not in seen]
    return {"candidates": new, "retrieved": [*state.retrieved, *new]}


def grade_docs(state: CorpusSearchState, c: AgentComponents, settings: AgentConfig) -> dict:
    """Grade each new candidate for relevance to the sub-query and keep the
    relevant ones, up to max_relevant_docs for this sub-query. Candidates
    arrive in retrieval-score order, so the cap keeps the best-ranked
    relevant chunks. Kept documents accumulate across retries."""
    grades = c.grade_documents(state.sub_query, state.candidates) if state.candidates else []
    relevant = [d for d, ok in zip(state.candidates, grades) if ok]
    room = max(settings.max_relevant_docs - len(state.documents), 0)
    return {"documents": [*state.documents, *relevant[:room]], "candidates": []}


def rewrite_query(state: CorpusSearchState, c: AgentComponents) -> dict:
    current = state.search_query or state.sub_query
    query = c.rewrite_query(state.sub_query, current, state.retrieved)
    return {"search_query": query or current, "query_rewrites": state.query_rewrites + 1}


def report(state: CorpusSearchState, settings: AgentConfig) -> dict:
    """Hand this sub-query's outcome back to the parent graph."""
    return {
        "sub_results": [
            SubQueryResult(
                index=state.index,
                sub_query=state.sub_query,
                search_query=state.search_query or state.sub_query,
                query_rewrites=state.query_rewrites,
                retrieved=state.retrieved,
                documents=state.documents,
                enough=len(state.documents) >= settings.min_relevant_docs,
            )
        ]
    }


# --- web route and answer ----------------------------------------------------


def web_search(state: AgentState, c: AgentComponents) -> dict:
    """Direct web route (practical tool / library questions)."""
    return {"documents": c.search_web(state.standalone_question)}


def generate(state: AgentState, c: AgentComponents) -> dict:
    _, history = split_latest_question(state.messages)
    answer = c.generate(state.question, history, state.documents)
    return {"messages": [AIMessage(answer)]}


# --- routing (conditional edges) ---------------------------------------------


def route_after_router(state: AgentState) -> Literal["decline", "decompose", "web_search"]:
    return {"decline": "decline", "web": "web_search"}.get(state.route, "decompose")


def route_after_merge(state: AgentState) -> Literal["generate", "fallback_research"]:
    return "fallback_research" if state.missing else "generate"


def route_after_grade_docs(state: CorpusSearchState, settings: AgentConfig) -> Literal["report", "rewrite_query"]:
    if len(state.documents) >= settings.min_relevant_docs or state.query_rewrites >= settings.max_query_rewrites:
        return "report"
    return "rewrite_query"

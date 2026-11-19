"""Graph nodes and routing functions.

Every node is a plain function `(state, components[, settings]) -> partial
state update`; build_graph() in agent/graph.py binds the dependencies and
wires them up. The nodes hold the per-step logic, the graph holds the
control flow, and AgentComponents holds everything that touches the network
- so each piece can be tested on its own with fakes.

Nodes of the corpus_search subgraph take CorpusSearchState (one
sub-query); the rest take AgentState. Sections follow the flow:
understanding the question, corpus path, web route, self-reflection.
"""
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.types import Overwrite, Send

from agent.components import AgentComponents
from agent.config import AgentConfig
from agent.prompts import DECLINE_MESSAGE, caveat
from agent.schemas import Critique
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
        "draft": "",
        "critique": None,
        "iteration": 0,
        "aspects_searched": False,
        "best_draft": "",
        "best_critique": None,
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


# --- self-reflection: draft -> reflect -> (revise | search aspects) -> finalize


def generate(state: AgentState, c: AgentComponents) -> dict:
    """Write a draft. After a failed critique this is the revision: the
    previous draft and the critique's issues go into the prompt."""
    _, history = split_latest_question(state.messages)
    revision = (state.draft, state.critique) if state.critique and not state.critique.passed else None
    draft = c.generate(state.question, history, state.documents, revision)
    return {"draft": draft, "iteration": state.iteration + 1}


def _rank(critique: Critique) -> tuple:
    return (critique.passed, critique.score, -len(critique.missing_aspects), -len(critique.problems))


def reflect(state: AgentState, c: AgentComponents) -> dict:
    """Check the draft claim by claim against the documents it was written
    from, and keep the best draft so far (by critique, ties -> the earlier)."""
    critique = c.reflect(state.standalone_question, state.draft, state.documents)
    update: dict = {"critique": critique}
    if state.best_critique is None or _rank(critique) > _rank(state.best_critique):
        update |= {"best_draft": state.draft, "best_critique": critique}
    return update


def search_aspects(state: AgentState, c: AgentComponents, settings: AgentConfig) -> dict:
    """The draft skipped parts of the question: one targeted search for
    them - corpus first (graded like any sub-query), then the research
    sub-agent for aspects the corpus doesn't cover. Runs once per turn."""
    aspects = state.critique.missing_aspects[: settings.max_missing_aspects]
    known = {d.key for d in state.documents}
    found, uncovered = [], []
    for aspect in aspects:
        new = [d for d in c.search_corpus(aspect) if d.key not in known]
        grades = c.grade_documents(aspect, new) if new else []
        relevant = [d for d, ok in zip(new, grades) if ok][: settings.max_relevant_docs]
        if not relevant:
            uncovered.append(aspect)
        found += relevant
        known |= {d.key for d in relevant}
    documents = dedupe([*state.documents, *found])
    update = {"aspects_searched": True, "retrieved": dedupe([*state.retrieved, *found])}
    if uncovered:
        live = c.research(state.standalone_question, uncovered, documents)
        if live:
            documents = [*documents, *live]
            update["route"] = "web"
    return update | {"documents": documents}


def finalize(state: AgentState) -> dict:
    """The answer the user sees: the current draft if it passed the check,
    otherwise the best draft of the turn with a caveat."""
    if state.critique is not None and state.critique.passed:
        return {"messages": [AIMessage(state.draft)]}
    best = state.best_draft or state.draft
    return {"messages": [AIMessage(best + caveat(state.best_critique or state.critique))]}


# --- routing (conditional edges) ---------------------------------------------


def route_after_router(state: AgentState) -> Literal["decline", "decompose", "web_search"]:
    return {"decline": "decline", "web": "web_search"}.get(state.route, "decompose")


def route_after_merge(state: AgentState) -> Literal["generate", "fallback_research"]:
    return "fallback_research" if state.missing else "generate"


def route_after_reflect(state: AgentState, settings: AgentConfig) -> Literal["finalize", "search_aspects", "generate"]:
    """good -> finalize; out of drafts -> finalize (best draft + caveat);
    parts of the question unanswered (first time) -> search for them, then
    revise; otherwise (unsupported claims / problems) -> revise."""
    critique = state.critique
    if critique.passed or state.iteration >= settings.max_iterations:
        return "finalize"
    if critique.missing_aspects and not state.aspects_searched:
        return "search_aspects"
    return "generate"


def route_after_grade_docs(state: CorpusSearchState, settings: AgentConfig) -> Literal["report", "rewrite_query"]:
    if len(state.documents) >= settings.min_relevant_docs or state.query_rewrites >= settings.max_query_rewrites:
        return "report"
    return "rewrite_query"

"""The agent as an explicit LangGraph StateGraph (docs/self-reflective-plan.md,
Phases 1-3). This module only wires the graph; node logic is in
agent/nodes.py, external calls in agent/components.py.

    START -> contextualize -> router --out_of_scope--> decline --> END
                                |  |
                                |  +--web (tool / library / docs)--> web_search ----------+
                                | corpus (research)                                       |
                                v                                                         |
                           decompose --Send, one per sub-query, in parallel-->            |
                                                                                          |
         corpus_search (subgraph, per sub-query):                                         |
             retrieve_corpus -> grade_docs (keep top relevant) --enough--> report         |
                  ^                  | too few                                |           |
                  +-- rewrite_query <+ (once; then report anyway)             |           |
                                                                              v           v
                                    merge_results --all parts covered----> generate --> END
                                          | some part not covered            ^
                                          v                                  |
                                  fallback_research (ReAct: web + live arXiv)+

Phase 3 (corrective RAG): a multi-part question is split into sub-queries;
each gets its own retrieval, per-document grading (at most
max_relevant_docs kept) and one query rewrite if too few are relevant.
Parts the corpus still doesn't cover go to a bounded research sub-agent
(web + live arXiv). The only loop is the single rewrite per sub-query.
"""
import asyncio
from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.components import AgentComponents, make_components
from agent.config import AgentConfig
from agent.state import AgentState, CorpusSearchOutput, CorpusSearchState


def build_graph(
    config: AgentConfig | None = None,
    *,
    components: AgentComponents | None = None,
    checkpointer=None,
):
    """Compile the agent graph.

    components: the LLM / search functions the nodes call; defaults to the
        real ones built from `config`. Tests inject fakes.
    checkpointer: defaults to InMemorySaver() (per-thread_id conversation
        memory, process-local). Pass False to compile without one - the
        LangGraph server (`langgraph dev`) supplies its own persistence.
    """
    config = config or AgentConfig()
    c = components or make_components(config)
    if checkpointer is None:
        checkpointer = InMemorySaver()
    elif checkpointer is False:
        checkpointer = None

    builder = StateGraph(AgentState)
    for node in (nodes.contextualize, nodes.router, nodes.decline, nodes.decompose, nodes.fallback_research, nodes.web_search, nodes.generate):
        builder.add_node(node.__name__, partial(node, c=c))
    builder.add_node("corpus_search", build_corpus_search(c, config))
    builder.add_node("merge_results", nodes.merge_results)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "router")
    builder.add_conditional_edges("router", nodes.route_after_router)
    builder.add_edge("decline", END)

    builder.add_conditional_edges("decompose", nodes.fan_out_sub_queries, ["corpus_search"])
    builder.add_edge("corpus_search", "merge_results")  # waits for every parallel run
    builder.add_conditional_edges("merge_results", nodes.route_after_merge)
    builder.add_edge("fallback_research", "generate")

    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)


def build_corpus_search(c: AgentComponents, config: AgentConfig):
    """The corrective retrieval loop for one sub-query, as a subgraph.

    Only `sub_results` (one SubQueryResult) flows back to the parent graph
    (output schema), so parallel runs can't clash on their working fields.
    """
    builder = StateGraph(CorpusSearchState, output_schema=CorpusSearchOutput)
    builder.add_node("retrieve_corpus", partial(nodes.retrieve_corpus, c=c))
    builder.add_node("grade_docs", partial(nodes.grade_docs, c=c, settings=config))
    builder.add_node("rewrite_query", partial(nodes.rewrite_query, c=c))
    builder.add_node("report", partial(nodes.report, settings=config))

    builder.add_edge(START, "retrieve_corpus")
    builder.add_edge("retrieve_corpus", "grade_docs")
    builder.add_conditional_edges("grade_docs", partial(nodes.route_after_grade_docs, settings=config), ["report", "rewrite_query"])
    builder.add_edge("rewrite_query", "retrieve_corpus")
    builder.add_edge("report", END)
    return builder.compile()


_studio_graph = None


async def studio_graph():
    """Graph factory for `langgraph dev` (see langgraph.json). A zero-arg
    factory so the CLI doesn't pass its RunnableConfig in as AgentConfig,
    and no checkpointer because the LangGraph server manages persistence.

    Async + to_thread because building the components does blocking network
    I/O (Pinecone connects to the index), which the LangGraph server rejects
    when done on its event loop. Built once and cached: the server calls the
    factory on every request, and rebuilding took 0.6-1.7 s each time.
    """
    global _studio_graph
    if _studio_graph is None:
        _studio_graph = await asyncio.to_thread(build_graph, checkpointer=False)
    return _studio_graph

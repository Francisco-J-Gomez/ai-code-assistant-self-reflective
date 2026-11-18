"""The agent as an explicit LangGraph StateGraph (docs/self-reflective-plan.md,
Phases 1-2). This module only wires the graph; node logic is in
agent/nodes.py, external calls in agent/components.py.

    START -> contextualize -> router --out_of_scope--> decline --> END
                                |  |
                                |  +--web (tool / library / docs)----+
                                | corpus (research)                  |
                                v                                    v
                     retrieve_corpus -> check_corpus --not enough--> web_search
                                            | enough                 |
                                            v                        |
                                        generate <-------------------+
                                            |
                                            v
                                           END

The standalone question and the route are decided up front, each by its own
node. Practical tool/library questions go straight to the web instead of
through a corpus search that can only fail; research questions still check
the corpus first and fall back to the web. No loop: at most 4 LLM calls per
turn (contextualize only calls the LLM on follow-ups).
"""
import asyncio
from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.components import AgentComponents, make_components
from agent.config import AgentConfig
from agent.state import AgentState


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
    for node in (nodes.contextualize, nodes.router, nodes.decline, nodes.retrieve_corpus, nodes.check_corpus, nodes.web_search, nodes.generate):
        builder.add_node(node.__name__, partial(node, c=c))

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "router")
    builder.add_conditional_edges("router", nodes.route_after_router)
    builder.add_edge("decline", END)
    builder.add_edge("retrieve_corpus", "check_corpus")
    builder.add_conditional_edges("check_corpus", nodes.route_after_check_corpus)
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)


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

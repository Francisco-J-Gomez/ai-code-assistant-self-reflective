"""The agent as an explicit LangGraph StateGraph (Phase 1 of
docs/self-reflective-plan.md). This module only wires the graph; node logic
is in agent/nodes.py, external calls in agent/components.py.

    START -> scope_check --out of scope--> decline --------------> END
                 | in scope (+ standalone search query)
                 v
          retrieve_corpus -> check_corpus --enough--> generate --> END
                                 | not enough            ^
                                 v                       |
                            web_search ------------------+

Same policy as the original create_agent ReAct loop - scope check first,
corpus before web, web only when the corpus doesn't answer, no answering from
general knowledge - but enforced by edges instead of by the system prompt.
There is no loop, so every turn makes at most 3 LLM calls.
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
    for node in (nodes.scope_check, nodes.decline, nodes.retrieve_corpus, nodes.check_corpus, nodes.web_search, nodes.generate):
        builder.add_node(node.__name__, partial(node, c=c))

    builder.add_edge(START, "scope_check")
    builder.add_conditional_edges("scope_check", nodes.route_after_scope_check)
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

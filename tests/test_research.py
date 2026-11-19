"""The fallback_research sub-agent with a scripted fake model and fake tools
(no network): documents are collected from the tools' artifacts, duplicates
and already-known documents are dropped, and the call limits hold."""
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.research import make_research
from agent.state import Document

PAPER = Document(kind="corpus", title="P", url="https://arxiv.org/abs/1", text="...", arxiv_id="1")
WEB = Document(kind="web", title="W", url="https://example.com", text="web text")
LIVE = Document(kind="arxiv", title="L", url="https://arxiv.org/abs/9", text="abstract", arxiv_id="9")


class ScriptedModel(FakeMessagesListChatModel):
    """Replays fixed AI messages (tool calls, then a final reply)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, query, i):
    return AIMessage("", tool_calls=[{"name": name, "args": {"query": query}, "id": f"call-{i}"}])


def _tools(calls):
    @tool(response_format="content_and_artifact")
    def web_search(query: str):
        """Search the web."""
        calls.append(("web_search", query))
        return "results", [WEB]

    @tool(response_format="content_and_artifact")
    def arxiv_search(query: str):
        """Search arXiv."""
        calls.append(("arxiv_search", query))
        return "results", [LIVE, PAPER]

    return [web_search, arxiv_search]


def test_collects_documents_from_tool_artifacts_and_drops_known_and_duplicates():
    calls = []
    model = ScriptedModel(responses=[
        _call("arxiv_search", "repocoder swe-bench", 1),
        _call("web_search", "GPT-7 SWE-bench", 2),
        _call("web_search", "GPT-7 SWE-bench again", 3),
        AIMessage("Found nothing about GPT-7."),
    ])
    research = make_research(AgentConfig(), model, tools=_tools(calls))

    found = research("What pass@1 did GPT-7 get?", missing=["GPT-7 pass@1 on SWE-bench"], known=[PAPER])

    assert [c[0] for c in calls] == ["arxiv_search", "web_search", "web_search"]
    # PAPER was already known; WEB came back twice but appears once.
    assert found == [LIVE, WEB]


def test_model_call_limit_stops_a_runaway_loop():
    calls = []
    endless = [_call("web_search", f"query {i}", i) for i in range(20)]
    research = make_research(
        AgentConfig(research_max_model_calls=2, research_max_tool_calls=10),
        ScriptedModel(responses=endless),
        tools=_tools(calls),
    )

    research("q", missing=["q"], known=[])

    assert len(calls) <= 2


def test_tool_call_limit_stops_a_runaway_loop():
    calls = []
    endless = [_call("web_search", f"query {i}", i) for i in range(20)]
    research = make_research(
        AgentConfig(research_max_model_calls=10, research_max_tool_calls=3),
        ScriptedModel(responses=endless),
        tools=_tools(calls),
    )

    research("q", missing=["q"], known=[])

    assert len(calls) <= 3

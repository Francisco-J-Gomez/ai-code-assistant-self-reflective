"""Streamlit chat UI for the RAG agent. Run with:

    uv run streamlit run ui/chat_app.py
"""
import os
import sys
import time
import uuid
from pathlib import Path

# `streamlit run ui/chat_app.py` sets sys.path[0] to this file's own
# directory (ui/), not the project root - `import agent` fails without this,
# regardless of the cwd the command was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # local dev: reads .env, a no-op if it doesn't exist

try:
    # Streamlit Community Cloud: secrets set via the app's Secrets UI land
    # in st.secrets (TOML), not necessarily os.environ - bridge them so the
    # rest of this app (and everything in agent/, ingestion/) can keep
    # reading os.environ either way. Raises StreamlitSecretNotFoundError
    # locally when no .streamlit/secrets.toml exists, which is expected
    # and fine since load_dotenv() already covered that case above.
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass

from agent.config import AgentConfig
from agent.graph import build_graph
from agent.graph_view import graph_dot
from agent.progress import ProgressTracker
from agent.results import answer_badge, self_check_note, sources_from_documents

st.set_page_config(page_title="AI / Software-Dev Research Assistant", page_icon="📄")

st.markdown(
    """
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
            "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 760px;
        padding-top: 3.5rem;
        padding-bottom: 7rem;
    }

    [data-testid="stHeading"] h1 {
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #1d1d1f;
    }

    [data-testid="stCaptionContainer"] {
        color: #6e6e73 !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 18px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03), 0 4px 14px rgba(0, 0, 0, 0.035);
    }

    [data-testid="stChatMessageContent"] p {
        color: #1d1d1f;
        line-height: 1.55;
    }

    [data-testid="stChatInput"] {
        border-radius: 26px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        background-color: #ffffff;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #0071e3;
        box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15);
    }

    [data-testid="stChatInputTextArea"] {
        color: #1d1d1f;
    }

    [data-testid="stChatInputSubmitButton"] {
        background-color: #0071e3 !important;
        border-radius: 999px !important;
    }

    [data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        background-color: #ffffff;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background-color: #1d1d1f !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #0071e3 !important;
    }

    a, a:visited {
        color: #0071e3;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    .source-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
    }
    .source-badge-corpus {
        background-color: rgba(0, 113, 227, 0.1);
        color: #0071e3;
    }
    .source-badge-web {
        background-color: rgba(255, 149, 0, 0.12);
        color: #c26a00;
    }
    .source-badge-none {
        background-color: rgba(0, 0, 0, 0.06);
        color: #6e6e73;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI / Software-Dev Research Assistant")
st.caption(
    "Grounded in a curated arXiv corpus of AI/software-engineering papers, "
    "with a live web search fallback when the corpus doesn't cover a question."
)

if "agent" not in st.session_state:
    st.session_state.agent_config = AgentConfig()
    st.session_state.agent = build_graph(st.session_state.agent_config)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role", "content", "sources", "badge", "check", "steps", "seconds"}]


def _start_new_chat() -> None:
    # New thread_id is enough to give the checkpointer a blank slate for
    # this conversation - no need to rebuild the graph/LLM connection.
    st.session_state.history = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pop("last_visited", None)


with st.sidebar:
    st.button("🆕  New Chat", on_click=_start_new_chat, use_container_width=True)
    st.markdown("**How the agent works**")
    st.caption("🟦 running · 🟩 done (×n = ran n times) · ⬜ not used. Shows the latest answer's path.")
    graph_slot = st.empty()


def _render_graph(visited=None, running=None) -> None:
    if "graph_drawable" not in st.session_state:
        st.session_state.graph_drawable = st.session_state.agent.get_graph(xray=True)
    graph_slot.graphviz_chart(graph_dot(st.session_state.graph_drawable, visited, running), width="stretch")


_render_graph(st.session_state.get("last_visited"))


def _render_badge(badge: dict | None) -> None:
    if not badge:
        return
    st.markdown(
        f'<span class="source-badge source-badge-{badge["kind"]}">{badge["icon"]} {badge["label"]}</span>',
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            if source["kind"] == "corpus":
                st.markdown(f"📄 [{source['title']} ({source['arxiv_id']})]({source['url']})")
            elif source["kind"] == "arxiv":
                st.markdown(f"🔎 [{source['title']} ({source['arxiv_id']})]({source['url']}) — live arXiv search")
            else:
                st.markdown(f"🌐 [{source['title']}]({source['url']})")


def _render_steps(steps: list[str], seconds: float) -> None:
    """The finished run's log, collapsed, for answers already on screen."""
    if not steps:
        return
    with st.expander(f"How this answer was made · {len(steps)} steps · {seconds:.0f}s"):
        for step in steps:
            st.markdown(f"- {step}")


def _run_agent(question: str, config: dict) -> tuple[dict, list[str], float]:
    """Run the graph, showing the current step live and logging each
    finished one. Returns (final state, step log, seconds)."""
    agent = st.session_state.agent
    tracker = ProgressTracker(st.session_state.get("agent_config"))
    start = time.monotonic()
    with st.status(tracker.current, expanded=True) as status:
        try:
            for _namespace, event in agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
                stream_mode="tasks",
                subgraphs=True,  # also report the steps inside corpus_search
            ):
                line = tracker.update(event)
                if line:
                    status.write(line)
                status.update(label=tracker.current)
                _render_graph(tracker.visited, tracker.running)
        except Exception:
            status.update(label="Something went wrong", state="error")
            raise
        seconds = time.monotonic() - start
        st.session_state.last_visited = dict(tracker.visited)
        _render_graph(tracker.visited)
        status.update(
            label=f"How this answer was made · {len(tracker.steps)} steps · {seconds:.0f}s",
            state="complete",
            expanded=False,
        )
    return agent.get_state(config).values, tracker.steps, seconds


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        _render_steps(turn.get("steps") or [], turn.get("seconds", 0))
        _render_badge(turn.get("badge"))
        st.markdown(turn["content"])
        _render_sources(turn.get("sources") or [])
        if turn.get("check"):
            st.caption(turn["check"])

question = st.chat_input("Ask about AI or software-development research...")
if question:
    st.session_state.history.append({"role": "user", "content": question, "sources": None, "badge": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        result, steps, seconds = _run_agent(question, {"configurable": {"thread_id": st.session_state.thread_id}})

        # documents/route are per-turn graph state (reset by contextualize each
        # turn), so they already describe only this answer.
        answer = result["messages"][-1].content
        sources = sources_from_documents(result["documents"])
        badge = answer_badge(result["route"])
        check = self_check_note(result.get("critique"), result.get("iteration", 0))

        _render_badge(badge)
        st.markdown(answer)
        _render_sources(sources)
        if check:
            st.caption(check)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "badge": badge,
            "check": check,
            "steps": steps,
            "seconds": seconds,
        }
    )

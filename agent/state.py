"""Graph state: what flows between nodes and is persisted per thread by the
checkpointer. (The LLM's structured outputs live in agent/schemas.py.)

State is deliberately minimal: only what the Phase 1 graph reads or writes.
Later phases add their own fields (e.g. critique / best_draft / iteration in
Phase 4) when they first need them.
"""
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Route = Literal["corpus", "web", "decline"]


class Document(BaseModel):
    """A retrieved source - a corpus chunk or a web result - in one shape, so
    generation, the UI and the evals don't care where it came from."""

    kind: Literal["corpus", "web"]
    title: str
    url: str
    text: str
    arxiv_id: str | None = None  # corpus only
    score: float | None = None


class AgentState(BaseModel):
    # Conversation history, persisted per thread by the checkpointer.
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # Per-turn fields. scope_check resets them at the start of every turn:
    # the checkpointer carries state across turns, and an out-of-scope
    # follow-up must not show the previous turn's documents.
    question: str = ""
    search_query: str = ""
    route: Route | None = None
    documents: list[Document] = Field(default_factory=list)

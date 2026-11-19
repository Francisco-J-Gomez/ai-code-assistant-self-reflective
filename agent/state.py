"""Graph state: what flows between nodes and is persisted per thread by the
checkpointer. (The LLM's structured outputs live in agent/schemas.py.)

State is deliberately minimal: only what the current graph reads or writes.
Later phases add their own fields (e.g. critique / best_draft / iteration in
Phase 4) when they first need them.
"""
import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agent.schemas import Critique

Route = Literal["corpus", "web", "decline"]


class Document(BaseModel):
    """A retrieved source in one shape, so generation, the UI and the evals
    don't care where it came from:

    corpus - a chunk from the curated Pinecone corpus
    web    - a Tavily web result
    arxiv  - a paper found by a live arXiv search (not in the corpus)
    """

    kind: Literal["corpus", "web", "arxiv"]
    title: str
    url: str
    text: str
    arxiv_id: str | None = None  # corpus and arxiv only
    score: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Identity of a source chunk: the same chunk can come back from two
        searches with different scores."""
        return self.url, self.text


class SubQueryResult(BaseModel):
    """What one corpus_search run (one sub-query) found."""

    index: int  # position in AgentState.sub_queries (parallel runs finish in any order)
    sub_query: str
    search_query: str  # the query finally used (after a rewrite, if any)
    query_rewrites: int
    retrieved: list[Document]  # every chunk retrieved for this sub-query
    documents: list[Document]  # the relevant ones kept (at most max_relevant_docs)
    enough: bool  # at least min_relevant_docs relevant chunks


class CorpusSearchState(BaseModel):
    """State of the corpus_search subgraph: the corrective retrieval loop for
    one sub-query. Started once per sub-query, in parallel, with Send."""

    index: int = 0
    sub_query: str = ""
    search_query: str = ""  # starts as sub_query; rewrite_query replaces it
    query_rewrites: int = 0
    candidates: list[Document] = Field(default_factory=list)  # latest batch, not yet graded
    retrieved: list[Document] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)  # graded relevant, capped
    # The only key handed back to the parent graph (see CorpusSearchOutput).
    sub_results: Annotated[list[SubQueryResult], operator.add] = Field(default_factory=list)


class CorpusSearchOutput(BaseModel):
    sub_results: Annotated[list[SubQueryResult], operator.add] = Field(default_factory=list)


class AgentState(BaseModel):
    # Conversation history, persisted per thread by the checkpointer.
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # Per-turn fields. contextualize resets them at the start of every turn:
    # the checkpointer carries state across turns, and an out-of-scope
    # follow-up must not show the previous turn's documents.
    question: str = ""  # the latest user message, as typed
    standalone_question: str = ""  # same question, understandable without the conversation
    # The path taken so far: set by router; on the corpus path, "corpus" when
    # every sub-query found enough relevant papers, "web" if
    # fallback_research ran.
    route: Route | None = None

    # Corpus path: the question split into sub-queries (one for a single-part
    # question), each searched by its own corpus_search run. sub_results
    # collects their results (reducer: parallel runs append).
    sub_queries: list[str] = Field(default_factory=list)
    sub_results: Annotated[list[SubQueryResult], operator.add] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)  # sub-queries the corpus didn't cover
    retrieved: list[Document] = Field(default_factory=list)  # every corpus chunk retrieved this turn (evals, debugging)

    # What generate answers from: corpus chunks graded relevant, plus web /
    # live-arXiv results.
    documents: list[Document] = Field(default_factory=list)

    # Self-reflection loop. A draft only becomes a chat message in finalize.
    draft: str = ""
    critique: Critique | None = None  # of the current draft
    iteration: int = 0  # drafts written this turn
    aspects_searched: bool = False  # missing aspects get one targeted search per turn
    best_draft: str = ""
    best_critique: Critique | None = None

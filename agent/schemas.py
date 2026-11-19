"""Structured outputs the LLM-backed nodes ask the model for
(`with_structured_output`). These are contracts with the LLM, not graph
state: a node reads one of these and writes the relevant fields to
AgentState. Later phases add theirs here (e.g. Critique in Phase 4).
"""
from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Output of router."""

    reasoning: str = Field(description="One sentence on what kind of question this is, written before the route.")
    route: Literal["corpus", "web", "out_of_scope"] = Field(
        description="corpus: research questions (papers, methods, findings, results). "
        "web: practical questions about specific tools, libraries, products, docs, versions or releases. "
        "out_of_scope: not about AI or software development."
    )


class SubQueries(BaseModel):
    """Output of decompose."""

    sub_queries: list[str] = Field(
        description="One self-contained search query per distinct part of the question; "
        "a single query if the question asks one thing."
    )


class DocGrade(BaseModel):
    index: int = Field(description="The document's number in the list.")
    reason: str = Field(description="A few words on why, written before the verdict.")
    relevant: bool = Field(description="True if the document's text helps answer the question.")


class DocGrades(BaseModel):
    """Output of grade_docs: one grade per document, in one call."""

    grades: list[DocGrade]

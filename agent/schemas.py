"""Structured outputs the LLM-backed nodes ask the model for
(`with_structured_output`). These are contracts with the LLM, not graph
state: a node reads one of these and writes the relevant fields to
AgentState. Later phases add theirs here (e.g. DocGrade, Critique).
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


class CorpusCheck(BaseModel):
    """Output of check_corpus."""

    reasoning: str = Field(description="Brief justification, written before the verdict.")
    answers_question: bool = Field(description="True if the papers contain information that actually answers the question.")

"""Structured outputs the LLM-backed nodes ask the model for
(`with_structured_output`). These are contracts with the LLM, not graph
state: a node reads one of these and writes the relevant fields to
AgentState.
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


class ClaimCheck(BaseModel):
    claim: str = Field(description="One atomic factual statement from the answer.")
    supported: bool = Field(description="True if the sources state it (or, for a statement about the sources, it is true of them).")
    note: str = Field(default="", description="If unsupported: what the sources actually say, or that they say nothing about it.")


class Critique(BaseModel):
    """Output of reflect: the draft answer checked against its sources.

    The verdict (passed / score) is computed from the lists, not chosen by
    the model, so the routing rule is visible in code.
    """

    claims: list[ClaimCheck] = Field(description="Every atomic factual claim in the answer, checked against the sources.")
    missing_aspects: list[str] = Field(
        description="Parts of the question the answer doesn't address at all, as short search-style phrases. "
        "A part the answer explicitly says the sources don't cover is addressed, not missing."
    )
    problems: list[str] = Field(
        description="Other problems: a false premise accepted instead of corrected, a closing offer to search or "
        "request for material, a paper/model/result treated as real that no source shows. Empty if none."
    )

    @property
    def unsupported_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.supported]

    @property
    def grounded(self) -> bool:
        return not self.unsupported_claims

    @property
    def score(self) -> float:
        """Share of claims the sources support (1.0 for an answer with no factual claims)."""
        return 1.0 if not self.claims else sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def passed(self) -> bool:
        return self.grounded and not self.missing_aspects and not self.problems

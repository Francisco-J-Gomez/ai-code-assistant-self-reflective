"""Structured outputs the LLM-backed nodes ask the model for
(`with_structured_output`). These are contracts with the LLM, not graph
state: a node reads one of these and writes the relevant fields to
AgentState. Later phases add theirs here (e.g. DocGrade, Critique).
"""
from pydantic import BaseModel, Field


class ScopeDecision(BaseModel):
    """Output of scope_check."""

    in_scope: bool = Field(description="True if the latest message is about AI or software development.")
    search_query: str = Field(
        description="The latest message rewritten as a self-contained search query, resolving references "
        "to the conversation (e.g. 'it', 'the first one'). Empty if out of scope."
    )


class CorpusCheck(BaseModel):
    """Output of check_corpus."""

    reasoning: str = Field(description="Brief justification, written before the verdict.")
    answers_question: bool = Field(description="True if the papers contain information that actually answers the question.")

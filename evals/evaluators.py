"""Evaluators for the self-reflective RAG eval set.

Code-based (free, deterministic):
    route_correct      did the agent take an acceptable route (corpus / web / decline)?
    retrieval_hit      was at least one expected paper retrieved? (only when papers are expected)
    retrieval_recall   fraction of expected papers retrieved
    drafts, self_check_passed  reflection-loop signals (Phase 4+)
    llm_calls, ktokens   cost signals (tokens in thousands) (latency is recorded by LangSmith per run)

LLM-as-judge (grader model deliberately stronger than the agent's model):
    correctness        does the answer match the reference behaviour?
    groundedness       is every factual claim supported by the retrieved context?

Evaluators that don't apply to an example return no score (rather than a
fake 0 or 1), so per-metric averages only cover examples where they mean
something.
"""
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

NOT_APPLICABLE = {"results": []}


# --- Code-based ---------------------------------------------------------------

def route_correct(outputs: dict, reference_outputs: dict) -> dict:
    acceptable = reference_outputs.get("acceptable_routes") or [reference_outputs["expected_route"]]
    return {
        "key": "route_correct",
        "score": int(outputs["route"] in acceptable),
        "comment": f"route={outputs['route']}, acceptable={acceptable}",
    }


def retrieval_hit(outputs: dict, reference_outputs: dict) -> dict:
    expected = set(reference_outputs.get("expected_arxiv_ids") or [])
    if not expected:
        return NOT_APPLICABLE
    hits = expected & set(outputs["retrieved_ids"])
    return {"key": "retrieval_hit", "score": int(bool(hits)), "comment": f"hit={sorted(hits)}"}


def retrieval_recall(outputs: dict, reference_outputs: dict) -> dict:
    expected = set(reference_outputs.get("expected_arxiv_ids") or [])
    if not expected:
        return NOT_APPLICABLE
    hits = expected & set(outputs["retrieved_ids"])
    return {
        "key": "retrieval_recall",
        "score": len(hits) / len(expected),
        "comment": f"missing={sorted(expected - hits)}",
    }


def cost_signals(outputs: dict) -> dict:
    # Tokens in thousands: LangSmith rejects feedback scores above 99,999.9999,
    # and a looping agent turn can exceed 100k tokens.
    total_tokens = outputs["input_tokens"] + outputs["output_tokens"]
    return {
        "results": [
            {"key": "llm_calls", "score": outputs["llm_calls"]},
            {"key": "ktokens", "score": round(total_tokens / 1000, 2), "comment": f"{total_tokens} tokens"},
        ]
    }


def reflection_signals(outputs: dict) -> dict:
    """How hard the reflection loop worked (Phase 4): drafts per answer, and
    whether the agent's own fact-check passed. Not applicable without a
    draft (declines) or for runs from earlier phases."""
    if not outputs.get("drafts"):
        return NOT_APPLICABLE
    results = [{"key": "drafts", "score": outputs["drafts"]}]
    if outputs.get("self_check_passed") is not None:
        results.append({"key": "self_check_passed", "score": int(outputs["self_check_passed"])})
    return {"results": results}


CODE_EVALUATORS = [route_correct, retrieval_hit, retrieval_recall, cost_signals, reflection_signals]


# --- LLM-as-judge -------------------------------------------------------------

class CorrectnessGrade(BaseModel):
    reasoning: str = Field(description="Brief justification, written before the verdict.")
    verdict: Literal["correct", "partially_correct", "incorrect"]


class GroundednessGrade(BaseModel):
    reasoning: str = Field(description="Brief justification, written before the verdict.")
    unsupported_claims: list[str] = Field(description="Factual claims in the answer NOT supported by the context.")
    verdict: Literal["grounded", "partially_grounded", "ungrounded"]


VERDICT_SCORES = {
    "correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0,
    "grounded": 1.0, "partially_grounded": 0.5, "ungrounded": 0.0,
}

CORRECTNESS_PROMPT = """\
You are grading an AI research assistant (scope: AI and software-development \
research, grounded in a corpus of arXiv papers, with web search as fallback).

Compare the ASSISTANT ANSWER with the REFERENCE, which describes what a good \
answer contains or how the assistant should behave (e.g. decline an \
out-of-scope question, correct a false premise, admit a paper doesn't exist).

Important:
- The paper corpus contains arXiv papers published up to late 2026, many \
newer than your training data. Do NOT treat an arXiv id, date or title as \
fabricated just because you don't recognise it; the RETRIEVED CONTEXT shows \
what the assistant actually found.
- Papers named in the REFERENCE are examples of good sources, not a required \
list. Citing other relevant papers from the retrieved context is fine. Judge \
whether the key ideas and the required behaviour are there.
- A specific number or claim that appears in the RETRIEVED CONTEXT is not \
invented. Grounding itself is graded separately; here, judge whether the \
answer is right and complete.

- correct: matches the reference's key ideas / required behaviour; extra \
accurate detail is fine.
- partially_correct: some key ideas or only part of a multi-part question.
- incorrect: misses the point, contradicts the reference, answers something \
it should have declined, or invents content the reference says must not be invented.

QUESTION:
{question}

CONVERSATION BEFORE THE QUESTION (may be empty):
{chat_history}

REFERENCE:
{reference}

RETRIEVED CONTEXT (what the assistant's tools returned; may be empty):
{context}

ASSISTANT ANSWER:
{answer}
"""

GROUNDEDNESS_PROMPT = """\
You are checking whether an AI assistant's answer is grounded in the sources \
it retrieved. Judge ONLY support by the CONTEXT below - not whether the claims \
are true in general. Statements that make no factual claim (e.g. "I couldn't \
find this", "this is outside my scope", or describing which sources were used) \
need no support.

- grounded: every factual claim is supported by the context.
- partially_grounded: most claims supported, some are not.
- ungrounded: the main claims are not supported (e.g. answered from general \
knowledge, or invented details such as numbers or paper titles).

CONTEXT (retrieved documents):
{context}

ASSISTANT ANSWER:
{answer}
"""


def _format_history(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history) or "(none)"


def make_llm_judges(grader_model: str) -> list:
    llm = ChatOpenAI(model=grader_model)
    correctness_llm = llm.with_structured_output(CorrectnessGrade)
    groundedness_llm = llm.with_structured_output(GroundednessGrade)

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        grade = correctness_llm.invoke(
            CORRECTNESS_PROMPT.format(
                question=inputs["question"],
                chat_history=_format_history(inputs.get("chat_history", [])),
                reference=reference_outputs["reference"],
                context=outputs["context"] or "(none)",
                answer=outputs["answer"],
            )
        )
        return {"key": "correctness", "score": VERDICT_SCORES[grade.verdict], "comment": grade.reasoning}

    def groundedness(outputs: dict) -> dict:
        # A decline makes no factual claims and retrieves nothing - nothing to ground.
        if outputs["route"] == "decline":
            return NOT_APPLICABLE
        grade = groundedness_llm.invoke(
            GROUNDEDNESS_PROMPT.format(context=outputs["context"] or "(no documents retrieved)", answer=outputs["answer"])
        )
        comment = grade.reasoning
        if grade.unsupported_claims:
            comment += "\nUnsupported: " + "; ".join(grade.unsupported_claims)
        return {"key": "groundedness", "score": VERDICT_SCORES[grade.verdict], "comment": comment}

    return [correctness, groundedness]

"""The external, side-effecting pieces the graph's nodes call: LLM calls,
corpus search, web search, the fallback research sub-agent.

They are bundled in AgentComponents and injected into build_graph(), so the
graph's control flow can be unit-tested with plain fakes (no network, no
LLM) - see tests/test_graph.py. make_components() builds the real ones.
"""
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from agent.config import AgentConfig
from agent.prompts import (
    CONTEXTUALIZE_PROMPT,
    DECOMPOSE_PROMPT,
    GENERATE_PROMPT,
    GRADE_DOCS_PROMPT,
    REFLECT_PROMPT,
    REVISION_PROMPT,
    REWRITE_QUERY_PROMPT,
    ROUTER_PROMPT,
    format_documents,
    format_issues,
    source_note,
)
from agent.research import make_research
from agent.retrieval import make_corpus_search
from agent.schemas import Critique, DocGrades, RouteDecision, SubQueries
from agent.state import Document


@dataclass(frozen=True)
class AgentComponents:
    # (latest question, prior conversation) -> standalone question
    contextualize: Callable[[str, list[AnyMessage]], str]
    # standalone question -> corpus / web / out of scope
    route: Callable[[str], RouteDecision]
    # standalone question -> sub-queries (one per distinct part)
    decompose: Callable[[str], list[str]]
    # search query -> corpus documents
    search_corpus: Callable[[str], list[Document]]
    # (question, documents) -> relevant? one bool per document, same order
    grade_documents: Callable[[str, list[Document]], list[bool]]
    # (question, previous query, documents it found) -> new corpus query
    rewrite_query: Callable[[str, str, list[Document]], str]
    # search query -> web documents
    search_web: Callable[[str], list[Document]]
    # (question, parts the corpus didn't cover, documents already found)
    #   -> new documents (web / live arXiv)
    research: Callable[[str, list[str], list[Document]], list[Document]]
    # (question, prior conversation, documents, previous draft + its
    #  critique for a revision, or None) -> answer text
    generate: Callable[[str, list[AnyMessage], list[Document], tuple[str, Critique] | None], str]
    # (question, draft, documents) -> claim-level critique of the draft
    reflect: Callable[[str, str, list[Document]], Critique]


def make_components(config: AgentConfig) -> AgentComponents:
    """Real components: OpenAI LLMs, Pinecone corpus, Tavily web search,
    live arXiv search (inside the research sub-agent).

    Builds network clients (Pinecone connects to the index here), so call it
    off the event loop in async contexts - see agent.graph.studio_graph.
    """
    llm = ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)
    router_llm = llm.with_structured_output(RouteDecision)
    decompose_llm = llm.with_structured_output(SubQueries)
    grader_llm = ChatOpenAI(model=config.grader_model, temperature=config.llm_temperature).with_structured_output(DocGrades)
    corpus_search = make_corpus_search(config)
    tavily = TavilySearch(max_results=config.web_search_max_results)
    research = make_research(config, llm)
    reflect_llm = ChatOpenAI(
        model=config.reflect_model, temperature=config.llm_temperature, reasoning_effort=config.reflect_reasoning_effort
    ).with_structured_output(Critique)

    def contextualize(question: str, history: list[AnyMessage]) -> str:
        return llm.invoke([SystemMessage(CONTEXTUALIZE_PROMPT), *history, HumanMessage(question)]).content.strip()

    def route(question: str) -> RouteDecision:
        return router_llm.invoke(ROUTER_PROMPT.format(question=question))

    def decompose(question: str) -> list[str]:
        result = decompose_llm.invoke(DECOMPOSE_PROMPT.format(question=question, max_sub_queries=config.max_sub_queries))
        queries = [q.strip() for q in result.sub_queries if q.strip()]
        return queries[: config.max_sub_queries] or [question]

    def search_corpus(query: str) -> list[Document]:
        return [
            Document(
                kind="corpus",
                title=hit.get("title", ""),
                url=hit.get("source_url", ""),
                text=hit.get("text", ""),
                arxiv_id=hit.get("arxiv_id"),
                score=hit.get("score"),
            )
            for hit in corpus_search(query)
        ]

    def grade_documents(question: str, documents: list[Document]) -> list[bool]:
        result = grader_llm.invoke(GRADE_DOCS_PROMPT.format(question=question, documents=format_documents(documents)))
        # format_documents numbers documents from 1. A document the grader
        # skipped counts as not relevant.
        relevant = {g.index for g in result.grades if g.relevant}
        return [i in relevant for i in range(1, len(documents) + 1)]

    def rewrite_query(question: str, query: str, found: list[Document]) -> str:
        titles = "\n".join(f"- {t}" for t in dict.fromkeys(d.title for d in found)) or "(none)"
        return llm.invoke(REWRITE_QUERY_PROMPT.format(question=question, query=query, found=titles)).content.strip()

    def search_web(query: str) -> list[Document]:
        response = tavily.invoke({"query": query})
        return [
            Document(
                kind="web",
                title=r.get("title", ""),
                url=r.get("url", ""),
                text=r.get("content", ""),
                score=r.get("score"),
            )
            for r in response.get("results", [])
        ]

    def generate(
        question: str, history: list[AnyMessage], documents: list[Document], revision: tuple[str, Critique] | None = None
    ) -> str:
        system = GENERATE_PROMPT.format(source_note=source_note(documents), documents=format_documents(documents))
        if revision:
            draft, critique = revision
            system += REVISION_PROMPT.format(issues=format_issues(critique), draft=draft)
        return llm.invoke([SystemMessage(system), *history, HumanMessage(question)]).content

    def reflect(question: str, draft: str, documents: list[Document]) -> Critique:
        return reflect_llm.invoke(REFLECT_PROMPT.format(question=question, draft=draft, documents=format_documents(documents)))

    return AgentComponents(
        contextualize=contextualize,
        route=route,
        decompose=decompose,
        search_corpus=search_corpus,
        grade_documents=grade_documents,
        rewrite_query=rewrite_query,
        search_web=search_web,
        research=research,
        generate=generate,
        reflect=reflect,
    )

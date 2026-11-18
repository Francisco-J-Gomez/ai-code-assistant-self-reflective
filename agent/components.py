"""The external, side-effecting pieces the graph's nodes call: LLM calls,
corpus search, web search.

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
from agent.prompts import CHECK_CORPUS_PROMPT, GENERATE_PROMPT, SCOPE_PROMPT, format_documents
from agent.retrieval import make_corpus_search
from agent.schemas import CorpusCheck, ScopeDecision
from agent.state import Document


@dataclass(frozen=True)
class AgentComponents:
    # (question, prior conversation) -> scope verdict + standalone search query
    scope_check: Callable[[str, list[AnyMessage]], ScopeDecision]
    # search query -> corpus documents
    search_corpus: Callable[[str], list[Document]]
    # (question, corpus documents) -> are they enough?
    check_corpus: Callable[[str, list[Document]], CorpusCheck]
    # search query -> web documents
    search_web: Callable[[str], list[Document]]
    # (question, prior conversation, documents) -> answer text
    generate: Callable[[str, list[AnyMessage], list[Document]], str]


def make_components(config: AgentConfig) -> AgentComponents:
    """Real components: OpenAI LLM, Pinecone corpus, Tavily web search.

    Builds network clients (Pinecone connects to the index here), so call it
    off the event loop in async contexts - see agent.graph.studio_graph.
    """
    llm = ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)
    scope_llm = llm.with_structured_output(ScopeDecision)
    check_llm = llm.with_structured_output(CorpusCheck)
    corpus_search = make_corpus_search(config)
    tavily = TavilySearch(max_results=config.web_search_max_results)

    def scope_check(question: str, history: list[AnyMessage]) -> ScopeDecision:
        return scope_llm.invoke([SystemMessage(SCOPE_PROMPT), *history, HumanMessage(question)])

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

    def check_corpus(question: str, documents: list[Document]) -> CorpusCheck:
        return check_llm.invoke(CHECK_CORPUS_PROMPT.format(question=question, documents=format_documents(documents)))

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

    def generate(question: str, history: list[AnyMessage], documents: list[Document]) -> str:
        response = llm.invoke(
            [SystemMessage(GENERATE_PROMPT.format(documents=format_documents(documents))), *history, HumanMessage(question)]
        )
        return response.content

    return AgentComponents(
        scope_check=scope_check,
        search_corpus=search_corpus,
        check_corpus=check_corpus,
        search_web=search_web,
        generate=generate,
    )

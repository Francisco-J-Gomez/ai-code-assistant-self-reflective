"""fallback_research: a small ReAct sub-agent for questions the corpus
couldn't answer, even after a query rewrite (Phase 3 of
docs/self-reflective-plan.md).

This is the one place a ReAct loop belongs in the flow: which searches to
run, and how many, depends on what the previous search returned (try arXiv
for a paper title, fall back to the web to check whether it exists at all,
...). The loop is hard-capped with LangChain's call-limit middleware, so it
can't run away like the Phase 0 agent did.

The sub-agent only *searches*: its tools return Documents as artifacts, which
are collected from the tool messages and handed to the main graph's
generate node - the same citation and grounding rules apply to every answer.
"""
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from agent.config import AgentConfig
from agent.prompts import RESEARCH_PROMPT
from agent.state import Document
from ingestion.clients.arxiv_client import ArxivClientError, search_arxiv

# What the sub-agent sees of each result. It only decides whether to search
# again, so a snippet is enough; generate gets the full text.
SNIPPET_CHARS = 300


def _summarize(documents: list[Document]) -> str:
    if not documents:
        return "No results."
    return "\n\n".join(f"[{i}] {d.title}\n{d.url}\n{d.text[:SNIPPET_CHARS]}" for i, d in enumerate(documents, 1))


def make_research_tools(config: AgentConfig) -> list:
    tavily = TavilySearch(max_results=config.web_search_max_results)

    @tool(response_format="content_and_artifact")
    def web_search(query: str) -> tuple[str, list[Document]]:
        """Search the web. For tools, libraries, docs, releases,
        leaderboards, blog posts, and checking whether a paper, model or
        result exists at all."""
        results = tavily.invoke({"query": query}).get("results", [])
        documents = [
            Document(kind="web", title=r.get("title", ""), url=r.get("url", ""), text=r.get("content", ""), score=r.get("score"))
            for r in results
        ]
        return _summarize(documents), documents

    @tool(response_format="content_and_artifact")
    def arxiv_search(query: str) -> tuple[str, list[Document]]:
        """Live search over all of arXiv, most relevant first. For research
        papers, including recent ones and specific papers by title. Every
        keyword must appear in the paper, so use 2-5 distinctive keywords,
        not a full question."""
        try:
            papers = search_arxiv(query, max_results=config.arxiv_search_max_results)
        except ArxivClientError as exc:
            return f"arXiv search failed: {exc}", []
        documents = [
            Document(kind="arxiv", title=p.title, url=f"https://arxiv.org/abs/{p.arxiv_id}", text=p.abstract, arxiv_id=p.arxiv_id)
            for p in papers
        ]
        return _summarize(documents), documents

    return [web_search, arxiv_search]


def make_research(config: AgentConfig, llm: BaseChatModel, tools: list | None = None) -> Callable[[str, list[Document]], list[Document]]:
    """Build `(question, parts not covered, documents already found) -> new
    documents`."""
    agent = create_agent(
        llm,
        tools or make_research_tools(config),
        system_prompt=RESEARCH_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(run_limit=config.research_max_model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=config.research_max_tool_calls, exit_behavior="end"),
        ],
        name="fallback_research",
    )

    def research(question: str, missing: list[str], known: list[Document]) -> list[Document]:
        prompt = f"QUESTION:\n{question}"
        if missing and missing != [question]:
            parts = "\n".join(f"- {m}" for m in missing)
            prompt += f"\n\nThe corpus didn't cover these parts - search for them:\n{parts}"
        if known:
            titles = "\n".join(f"- {d.title}" for d in known)
            prompt += f"\n\nAlready found in the corpus (partly relevant, don't search for these again):\n{titles}"
        result = agent.invoke({"messages": [HumanMessage(prompt)]})

        found: dict[tuple[str, str], Document] = {}
        for message in result["messages"]:
            if isinstance(message, ToolMessage) and isinstance(message.artifact, list):
                for doc in message.artifact:
                    found.setdefault(doc.key, doc)
        known_keys = {d.key for d in known}
        return [d for key, d in found.items() if key not in known_keys]

    return research

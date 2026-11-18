from typing import Callable

from langchain_core.tools import tool

from agent.config import AgentConfig
from ingestion.clients.embeddings import Embedder
from ingestion.clients.pinecone_client import PineconeStore


def make_corpus_search(config: AgentConfig) -> Callable[[str], list[dict]]:
    """Build a `query -> top-k corpus hits` function bound to an AgentConfig.

    Embedder and PineconeStore must be built from the same embedding
    model/dimension/index the corpus was ingested with (see
    AgentConfig.embedding) or similarity search is meaningless. Hits are
    Pinecone's plain dicts: {chunk_id, score, arxiv_id, title, authors,
    published, source_url, text}, most relevant first.

    Used directly by the graph's retrieve_corpus node, and wrapped as the
    search_papers tool for tool-calling agents.
    """
    embedder = Embedder(model=config.embedding.embedding_model)
    store = PineconeStore(
        index_name=config.embedding.pinecone_index_name,
        dimension=config.embedding.embedding_dimensions,
        cloud=config.embedding.pinecone_cloud,
        region=config.embedding.pinecone_region,
    )

    def search(query: str) -> list[dict]:
        vector = embedder.embed([query])[0]
        return store.query(vector, top_k=config.retrieval_top_k)

    return search


def make_search_papers_tool(config: AgentConfig):
    """The corpus search as a LangChain tool, for tool-calling (ReAct)
    agents - e.g. the Phase 3 fallback_research sub-agent."""
    search = make_corpus_search(config)

    @tool
    def search_papers(query: str) -> list[dict]:
        """Search the curated arXiv corpus of AI / software-engineering
        research papers for content relevant to `query`.

        Use this for any question about AI or software-development
        research: findings, methods, techniques, results. Returns the
        most relevant papers as a list of {arxiv_id, title, authors,
        published, source_url, text, score}, most relevant first. Does
        NOT search the general web - use web_search for that.
        """
        return search(query)

    return search_papers

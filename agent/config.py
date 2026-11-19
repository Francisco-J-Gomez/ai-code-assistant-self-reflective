from dataclasses import dataclass, field

from ingestion.config import EmbeddingConfig


@dataclass(frozen=True)
class AgentConfig:
    """Knobs for the retrieval agent. Secrets (OPENAI_API_KEY,
    PINECONE_API_KEY, TAVILY_API_KEY) are read from the environment at the
    point of use, not stored here - see agent/retrieval.py and
    agent/graph.py.
    """

    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.0  # factual/grounded answers over creative ones
    retrieval_top_k: int = 5
    web_search_max_results: int = 3

    # Corrective RAG (Phase 3)
    grader_model: str = "gpt-5-mini"  # grades retrieved documents for relevance
    max_sub_queries: int = 3  # decompose splits a multi-part question into at most this many
    # Per sub-query:
    # Relevant corpus chunks needed to count a sub-query as covered. 1, not
    # more: a sub-query about one specific paper is answered by that paper
    # alone (multi-01), while questions the corpus can't answer get 0.
    min_relevant_docs: int = 1
    # Cap on relevant corpus chunks passed to generate. Retrieval still
    # fetches retrieval_top_k so the grader has more to choose from; the
    # highest-ranked relevant ones are kept.
    max_relevant_docs: int = 3
    max_query_rewrites: int = 1  # retries with a rewritten query before giving up on the corpus
    # fallback_research sub-agent: hard caps so its loop stays bounded
    # (the Phase 0 ReAct agent once made 60 LLM calls on one question).
    research_max_model_calls: int = 4
    research_max_tool_calls: int = 4
    arxiv_search_max_results: int = 5

    # Composed rather than duplicated: the retrieval tool must embed
    # queries with the *same* model/dimension/index the ingestion side
    # used to build the corpus, or similarity search is meaningless.
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def __post_init__(self):
        if not 1 <= self.min_relevant_docs <= self.max_relevant_docs:
            raise ValueError(
                f"need 1 <= min_relevant_docs ({self.min_relevant_docs}) <= max_relevant_docs ({self.max_relevant_docs})"
            )

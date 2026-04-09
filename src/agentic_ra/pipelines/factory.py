"""Pipeline factory — assembles all components into a working RAG system."""


import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from agentic_ra.core.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline container
# ---------------------------------------------------------------------------


@dataclass
class RAGPipeline:
    """Assembled RAG pipeline with all components.

    Attributes:
        orchestrator: Main query processing entry point.
        postgres: PostgreSQL client (for shutdown).
        embedding_pipeline: Docling chunking pipeline (parsing only).
        document_store: Postgres document persistence.
        run_store: Postgres pipeline run tracking.
        cognee_store: Cognee knowledge graph + vector store.
        strategy_qdrant: QdrantStore used by StrategyLearner (for shutdown).
    """

    orchestrator: Any  # AgentOrchestrator
    postgres: Any  # PostgresClient
    embedding_pipeline: Any  # EmbeddingPipeline
    document_store: Any  # DocumentStore
    run_store: Any  # RunStore
    cognee_store: Any | None = None
    strategy_qdrant: Any | None = None

    async def close(self) -> None:
        """Gracefully shut down all components, tolerating individual failures."""
        closeable: list[tuple[str, Callable[[], Coroutine[Any, Any, None]]]] = [
            ("PostgresClient", self.postgres.close),
        ]
        if self.strategy_qdrant is not None:
            closeable.append(("StrategyLearner QdrantStore", self.strategy_qdrant.close))
        for name, close_fn in closeable:
            try:
                await close_fn()
            except Exception:
                logging.getLogger(__name__).exception("Error closing %s", name)


# ---------------------------------------------------------------------------
# Phase builders
# ---------------------------------------------------------------------------


def _build_data_layer(settings: Settings) -> tuple:
    """Phase 1: PostgreSQL connection, document store, run tracking."""
    from agentic_ra.stores.postgres.client import PostgresClient
    from agentic_ra.stores.postgres.documents import DocumentStore
    from agentic_ra.stores.postgres.runs import RunStore

    postgres = PostgresClient(
        dsn=settings.postgres_dsn_async.get_secret_value(),
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
    )
    document_store = DocumentStore(pg_client=postgres)
    run_store = RunStore(pg_client=postgres)
    return postgres, document_store, run_store


def _build_embedding(settings: Settings, api_key: str) -> Any:
    """Phase 2: Docling chunking pipeline."""
    from agentic_ra.retrieval.embedder import EmbeddingPipeline

    return EmbeddingPipeline(
        model=settings.embedding_model,
        api_key=api_key,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_tokens=settings.chunk_max_tokens,
        merge_peers=settings.chunk_merge_peers,
    )


def _build_cognee(settings: Settings, api_key: str) -> Any | None:
    """Phase 3: Cognee knowledge graph (optional)."""
    from agentic_ra.stores.cognee.client import CogneeStore

    if not settings.cognee_enabled:
        logger.warning("Cognee disabled — graph retrieval unavailable")
        return None

    cognee_store = CogneeStore(
        dataset=settings.cognee_dataset,
        chunk_size=settings.cognee_chunk_size,
    )
    cognee_store.configure(
        llm_api_key=api_key,
        llm_model=settings.llm_enrichment_model,
        graph_db_provider=settings.cognee_graph_db_provider,
        vector_db_provider=settings.cognee_vector_db_provider,
        data_root=settings.cognee_data_root,
        system_root=settings.cognee_system_root,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key.get_secret_value(),
        neo4j_bolt_url=settings.neo4j_bolt_url,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password.get_secret_value(),
        neo4j_db=settings.neo4j_db,
        chunk_size=settings.cognee_chunk_size,
        chunk_overlap=settings.cognee_chunk_overlap,
    )
    logger.info(
        "CogneeStore configured (dataset=%s, qdrant=%s, neo4j=%s)",
        settings.cognee_dataset,
        settings.qdrant_url,
        settings.neo4j_bolt_url,
    )
    return cognee_store


def _build_retrieval(settings: Settings, cognee_store: Any) -> tuple:
    """Phase 4: Retrieval router and ranker."""
    from agentic_ra.retrieval.ranker import ResultRanker
    from agentic_ra.retrieval.router import RetrievalRouter

    router = RetrievalRouter(cognee_store=cognee_store)
    ranker = ResultRanker(rrf_k=settings.rrf_k)
    return router, ranker


def _build_agent_core(settings: Settings, api_key: str) -> tuple:
    """Phase 5: Base agent, verifier, synthesizer, metrics."""
    from agentic_ra.agent.agent import build_agent
    from agentic_ra.agent.prompts import load_system_prompt
    from agentic_ra.agent.synthesis import ResponseSynthesizer
    from agentic_ra.agent.verifier import GroundingVerifier
    from agentic_ra.observability.metrics import RAGMetrics

    verifier = GroundingVerifier(
        api_key=api_key,
        model=settings.llm_verifier_model,
        confidence_threshold=settings.verifier_confidence_threshold,
    )
    synthesizer = ResponseSynthesizer(
        max_snippet_length=settings.max_snippet_length,
    )
    metrics = RAGMetrics()

    system_prompt = load_system_prompt(settings.system_prompt_file)
    base_agent = build_agent(
        model=settings.llm_provider_model,
        api_key=api_key,
        tools_config=[],
        max_steps=settings.agent_max_steps,
        system_prompt=system_prompt,
    )
    base_agent.warm_up()

    return base_agent, verifier, synthesizer, metrics


def _build_web_tools(settings: Settings) -> Any | None:
    """Phase 6: Brave Search MCP client (optional)."""
    from agentic_ra.tools.mcp_http_client import MCPHttpClient

    if not settings.brave_search_enabled:
        return None
    brave_client = MCPHttpClient(base_url=settings.brave_mcp_url)
    logger.info("Brave Search MCP client configured: %s", settings.brave_mcp_url)
    return brave_client


def _build_feedback_loops(settings: Settings, api_key: str) -> tuple:
    """Phase 7: Feedback and strategy learning loops."""
    from agentic_ra.agent.feedback import FeedbackAdapter
    from agentic_ra.retrieval.strategy_learner import QueryEmbedder, StrategyLearner
    from agentic_ra.stores.qdrant.client import QdrantStore

    feedback = FeedbackAdapter(settings=settings) if settings.feedback_enabled else None
    if feedback is not None:
        logger.info("FeedbackAdapter enabled (dry_run=%s)", settings.feedback_dry_run)

    strategy_learner = None
    strategy_qdrant = None
    if settings.strategy_learning_enabled:
        strategy_qdrant = QdrantStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        query_embedder = QueryEmbedder(
            model=settings.embedding_model,
            api_key=api_key,
            dimension=settings.embedding_dimension,
        )
        strategy_learner = StrategyLearner(
            qdrant=strategy_qdrant,
            embedder=query_embedder,
            collection=settings.strategy_collection,
            min_neighbors=settings.strategy_min_neighbors,
            confidence_threshold=settings.strategy_confidence_threshold,
            max_age_days=settings.strategy_max_age_days,
        )
        logger.info(
            "StrategyLearner enabled (collection=%s, min_neighbors=%d)",
            settings.strategy_collection,
            settings.strategy_min_neighbors,
        )

    return feedback, strategy_learner, strategy_qdrant


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Assemble a complete RAG pipeline from settings.

    Phases:
    1. Data layer (Postgres)
    2. Embedding pipeline (Docling)
    3. Knowledge graph (Cognee)
    4. Retrieval (router + ranker)
    5. Agent core (agent, verifier, synthesizer, metrics)
    6. Web tools (Brave MCP)
    7. Feedback loops (strategy learner)
    8. Orchestrator assembly (step classes)
    """
    from agentic_ra.agent.memory import MemoryStore
    from agentic_ra.agent.orchestrator import AgentOrchestrator
    from agentic_ra.agent.steps import (
        AgentRunner,
        ContextBuilder,
        EvidenceFuser,
        GroundingGate,
        ResultPersister,
        ToolFactory,
    )
    from agentic_ra.core.errors import ConfigError
    from agentic_ra.tools.url_safety import URLValidator

    api_key = settings.llm_api_key.get_secret_value()

    # Phase 1: Data layer
    postgres, document_store, run_store = _build_data_layer(settings)

    # Phase 2: Embedding
    embedding_pipeline = _build_embedding(settings, api_key)

    # Phase 3: Knowledge graph
    cognee_store = _build_cognee(settings, api_key)
    if cognee_store is None:
        raise ConfigError(
            "Cognee is the sole retrieval backend — cognee_enabled must be True"
        )

    # Phase 4: Retrieval
    router, ranker = _build_retrieval(settings, cognee_store)

    # Phase 5: Agent core
    base_agent, verifier, synthesizer, metrics = _build_agent_core(settings, api_key)

    # Phase 6: Web tools
    brave_client = _build_web_tools(settings)

    # Phase 7: Feedback loops
    feedback, strategy_learner, strategy_qdrant = _build_feedback_loops(
        settings, api_key
    )

    # Phase 8: Wire orchestrator step classes
    memory = MemoryStore(pg_client=postgres)

    context_builder = ContextBuilder(
        memory=memory,
        strategy_learner=strategy_learner,
        metrics=metrics,
        settings=settings,
    )
    tool_factory = ToolFactory(
        base_agent=base_agent,
        router=router,
        ranker=ranker,
        cognee_store=cognee_store,
        document_store=document_store,
        brave_client=brave_client,
        url_validator=URLValidator(),
        memory=memory,
        settings=settings,
    )
    agent_runner = AgentRunner()
    evidence_fuser = EvidenceFuser(ranker=ranker, metrics=metrics)
    grounding_gate = GroundingGate(
        verifier=verifier,
        ranker=ranker,
        max_retries=settings.agent_max_retries,
        metrics=metrics,
    )
    result_persister = ResultPersister(
        synthesizer=synthesizer,
        memory=memory,
        run_store=run_store,
        feedback=feedback,
        strategy_learner=strategy_learner,
        metrics=metrics,
    )

    orchestrator = AgentOrchestrator(
        context_builder=context_builder,
        tool_factory=tool_factory,
        agent_runner=agent_runner,
        evidence_fuser=evidence_fuser,
        grounding_gate=grounding_gate,
        result_persister=result_persister,
    )

    logger.info("RAG pipeline assembled successfully")

    return RAGPipeline(
        orchestrator=orchestrator,
        postgres=postgres,
        embedding_pipeline=embedding_pipeline,
        document_store=document_store,
        run_store=run_store,
        cognee_store=cognee_store,
        strategy_qdrant=strategy_qdrant,
    )

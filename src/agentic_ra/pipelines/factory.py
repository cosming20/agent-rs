"""Pipeline factory — assembles all components into a working RAG system."""

from __future__ import annotations

import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Callable

from agentic_ra.agent.agent import build_agent
from agentic_ra.agent.feedback import FeedbackAdapter
from agentic_ra.agent.memory import MemoryStore
from agentic_ra.agent.orchestrator import AgentOrchestrator
from agentic_ra.agent.synthesis import ResponseSynthesizer
from agentic_ra.agent.verifier import GroundingVerifier
from agentic_ra.core.config import Settings
from agentic_ra.observability.metrics import RAGMetrics
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import RetrievalRouter
from agentic_ra.retrieval.strategy_learner import QueryEmbedder, StrategyLearner
from agentic_ra.stores.cognee.client import CogneeStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore
from agentic_ra.stores.postgres.runs import RunStore
from agentic_ra.stores.qdrant.client import QdrantStore
from agentic_ra.tools.mcp_http_client import MCPHttpClient
from agentic_ra.tools.url_safety import URLValidator

logger = logging.getLogger(__name__)


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

    orchestrator: AgentOrchestrator
    postgres: PostgresClient
    embedding_pipeline: EmbeddingPipeline
    document_store: DocumentStore
    run_store: RunStore
    cognee_store: CogneeStore | None = None
    strategy_qdrant: QdrantStore | None = None

    async def close(self) -> None:
        """Gracefully shut down all components, tolerating individual failures."""
        _CloseFn = Callable[[], Coroutine[Any, Any, None]]
        closeable: list[tuple[str, _CloseFn]] = [
            ("PostgresClient", self.postgres.close),
        ]
        if self.strategy_qdrant is not None:
            closeable.append(("StrategyLearner QdrantStore", self.strategy_qdrant.close))
        for name, close_fn in closeable:
            try:
                await close_fn()
            except Exception:
                logging.getLogger(__name__).exception("Error closing %s", name)


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Assemble a complete RAG pipeline from settings.

    Architecture:
    - Cognee owns vector storage (Qdrant) + knowledge graph (Neo4j)
    - Docling provides document parsing/chunking
    - Postgres stores DoclingDocument JSON, chat history, pipeline runs

    Args:
        settings: Application settings.

    Returns:
        Assembled RAG pipeline.
    """
    # --- Data layer ---
    postgres = PostgresClient(
        dsn=settings.postgres_dsn_async.get_secret_value(),
    )

    # --- Docling chunking pipeline (parsing only, Cognee handles embedding) ---
    api_key = settings.llm_api_key.get_secret_value()

    embedding_pipeline = EmbeddingPipeline(
        model=settings.embedding_model,
        api_key=api_key,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_tokens=settings.chunk_max_tokens,
        merge_peers=settings.chunk_merge_peers,
    )

    # --- Cognee (knowledge graph + vector: Qdrant + Neo4j) ---
    cognee_store: CogneeStore | None = None
    if settings.cognee_enabled:
        cognee_store = CogneeStore(dataset=settings.cognee_dataset)
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
        )
        logger.info(
            "CogneeStore configured (dataset=%s, qdrant=%s, neo4j=%s)",
            settings.cognee_dataset,
            settings.qdrant_url,
            settings.neo4j_bolt_url,
        )

    # --- Retrieval router (Cognee-only: graph + vector + lexical) ---
    if cognee_store is None:
        from agentic_ra.core.errors import ConfigError

        raise ConfigError(
            "Cognee is the sole retrieval backend — cognee_enabled must be True"
        )

    router = RetrievalRouter(cognee_store=cognee_store)

    ranker = ResultRanker()

    # --- Document persistence + run tracking ---
    document_store = DocumentStore(pg_client=postgres)
    run_store = RunStore(pg_client=postgres)

    # --- Agent core ---
    memory = MemoryStore(pg_client=postgres)
    verifier = GroundingVerifier(api_key=api_key, model=settings.llm_verifier_model)
    synthesizer = ResponseSynthesizer()
    metrics = RAGMetrics()

    # Build base agent with no tools — tools are injected per-request
    # in the orchestrator for RBAC dataset scoping.
    base_agent = build_agent(
        model=settings.llm_provider_model,
        api_key=api_key,
        tools_config=[],
        max_steps=settings.agent_max_steps,
    )
    base_agent.warm_up()

    # --- Brave Search MCP client (optional) ---
    brave_client: MCPHttpClient | None = None
    if settings.brave_search_enabled:
        brave_client = MCPHttpClient(base_url=settings.brave_mcp_url)
        logger.info("Brave Search MCP client configured: %s", settings.brave_mcp_url)

    # --- Feedback adapter (Loop 1: confidence → Cognee session feedback) ---
    feedback = FeedbackAdapter(settings=settings) if settings.feedback_enabled else None
    if feedback is not None:
        logger.info(
            "FeedbackAdapter enabled (dry_run=%s)", settings.feedback_dry_run
        )

    # --- Strategy learner (Loop 2: predict strategy from past query performance) ---
    strategy_learner: StrategyLearner | None = None
    strategy_qdrant: QdrantStore | None = None
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

    orchestrator = AgentOrchestrator(
        base_agent=base_agent,
        memory=memory,
        verifier=verifier,
        synthesizer=synthesizer,
        metrics=metrics,
        run_store=run_store,
        max_retries=settings.agent_max_retries,
        router=router,
        ranker=ranker,
        cognee_store=cognee_store,
        document_store=document_store,
        brave_client=brave_client,
        url_validator=URLValidator(),
        feedback=feedback,
        strategy_learner=strategy_learner,
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

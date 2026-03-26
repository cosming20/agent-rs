"""Pipeline factory — assembles all components into a working RAG system."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agentic_ra.agent.generator import LLMGenerator
from agentic_ra.agent.memory import MemoryStore
from agentic_ra.agent.orchestrator import AgentOrchestrator
from agentic_ra.agent.synthesis import ResponseSynthesizer
from agentic_ra.agent.verifier import GroundingVerifier
from agentic_ra.core.config import Settings
from agentic_ra.observability.metrics import RAGMetrics
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import QueryEmbedder, RetrievalRouter
from agentic_ra.stores.opensearch.client import OpenSearchStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore
from agentic_ra.stores.postgres.runs import RunStore
from agentic_ra.stores.qdrant.client import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class RAGPipeline:
    """Assembled RAG pipeline with all components.

    Holds references to all components for lifecycle management.

    Attributes:
        orchestrator: Main query processing entry point.
        postgres: PostgreSQL client (for shutdown).
        qdrant: Qdrant store (for shutdown).
        opensearch: OpenSearch store (for shutdown).
        embedding_pipeline: Document embedding pipeline.
        query_embedder: Query-time embedder.
    """

    orchestrator: AgentOrchestrator
    postgres: PostgresClient
    qdrant: QdrantStore
    opensearch: OpenSearchStore
    embedding_pipeline: EmbeddingPipeline
    query_embedder: QueryEmbedder
    document_store: DocumentStore
    run_store: RunStore

    async def close(self) -> None:
        """Gracefully shut down all components."""
        await self.qdrant.close()
        await self.opensearch.close()
        await self.postgres.close()


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Assemble a complete RAG pipeline from settings.

    Creates and wires all components: stores, retrieval engine,
    agent core, and returns a ready-to-use pipeline.

    Note: Call ``warm_up()`` on embedding_pipeline and query_embedder
    after construction to load models.

    Args:
        settings: Application settings.

    Returns:
        Assembled RAG pipeline.
    """
    # --- Data layer ---
    postgres = PostgresClient(
        dsn=settings.postgres_dsn_async.get_secret_value(),
    )

    qdrant = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    opensearch = OpenSearchStore(
        host=settings.opensearch_host,
        port=settings.opensearch_port,
        user=settings.opensearch_user,
        password=settings.opensearch_initial_admin_password,
    )

    # --- Retrieval engine ---
    api_key = settings.openai_api_key.get_secret_value()

    embedding_pipeline = EmbeddingPipeline(
        model=settings.embedding_model,
        api_key=api_key,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_tokens=settings.chunk_max_tokens,
        merge_peers=settings.chunk_merge_peers,
    )

    query_embedder = QueryEmbedder(
        model=settings.embedding_model,
        api_key=api_key,
        dimension=settings.embedding_dimension,
    )

    router = RetrievalRouter(
        query_embedder=query_embedder,
        qdrant_store=qdrant,
        opensearch_store=opensearch,
        collection="documents",
        index="documents",
    )

    ranker = ResultRanker()

    # --- Document persistence + run tracking ---
    document_store = DocumentStore(pg_client=postgres)
    run_store = RunStore(pg_client=postgres)

    # --- Agent core ---
    memory = MemoryStore(pg_client=postgres)
    generator = LLMGenerator(
        model=settings.llm_model,
        api_key=api_key,
    )
    verifier = GroundingVerifier(api_key=api_key, model=settings.llm_model)
    synthesizer = ResponseSynthesizer()
    metrics = RAGMetrics()

    orchestrator = AgentOrchestrator(
        memory=memory,
        router=router,
        ranker=ranker,
        generator=generator,
        verifier=verifier,
        synthesizer=synthesizer,
        metrics=metrics,
        run_store=run_store,
    )

    logger.info("RAG pipeline assembled successfully")

    return RAGPipeline(
        orchestrator=orchestrator,
        postgres=postgres,
        qdrant=qdrant,
        opensearch=opensearch,
        embedding_pipeline=embedding_pipeline,
        query_embedder=query_embedder,
        document_store=document_store,
        run_store=run_store,
    )

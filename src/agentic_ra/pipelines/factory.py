"""Pipeline factory — assembles all components into a working RAG system."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agentic_ra.agent.agent import build_agent
from agentic_ra.agent.memory import MemoryStore
from agentic_ra.agent.orchestrator import AgentOrchestrator
from agentic_ra.agent.synthesis import ResponseSynthesizer
from agentic_ra.agent.tools import (
    make_ingest_tool,
    make_search_knowledge_tool,
    make_search_web_tool,
)
from agentic_ra.agent.verifier import GroundingVerifier
from agentic_ra.core.config import Settings
from agentic_ra.observability.metrics import RAGMetrics
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import RetrievalRouter
from agentic_ra.stores.cognee.client import CogneeStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore
from agentic_ra.stores.postgres.runs import RunStore

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
    """

    orchestrator: AgentOrchestrator
    postgres: PostgresClient
    embedding_pipeline: EmbeddingPipeline
    document_store: DocumentStore
    run_store: RunStore
    cognee_store: CogneeStore | None = None

    async def close(self) -> None:
        """Gracefully shut down all components, tolerating individual failures."""
        for name, close_fn in [
            ("PostgresClient", self.postgres.close),
        ]:
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
    api_key = settings.openai_api_key.get_secret_value()

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
            llm_model=f"openai/{settings.llm_model}",
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
    router = RetrievalRouter(cognee_store=cognee_store)  # type: ignore[arg-type]

    ranker = ResultRanker()

    # --- Document persistence + run tracking ---
    document_store = DocumentStore(pg_client=postgres)
    run_store = RunStore(pg_client=postgres)

    # --- Agent core ---
    memory = MemoryStore(pg_client=postgres)
    verifier = GroundingVerifier(api_key=api_key, model=settings.llm_model)
    synthesizer = ResponseSynthesizer()
    metrics = RAGMetrics()

    # Build agent tools
    tools_config = [
        make_search_knowledge_tool(router, ranker),
        make_search_web_tool(),
    ]
    if cognee_store is not None:
        tools_config.append(make_ingest_tool(cognee_store))

    # Build Haystack Agent
    agent = build_agent(
        model=settings.agent_model,
        api_key=api_key,
        tools_config=tools_config,
        max_steps=settings.agent_max_steps,
    )
    agent.warm_up()

    orchestrator = AgentOrchestrator(
        agent=agent,
        memory=memory,
        verifier=verifier,
        synthesizer=synthesizer,
        metrics=metrics,
        run_store=run_store,
        max_retries=settings.agent_max_retries,
    )

    logger.info("RAG pipeline assembled successfully")

    return RAGPipeline(
        orchestrator=orchestrator,
        postgres=postgres,
        embedding_pipeline=embedding_pipeline,
        document_store=document_store,
        run_store=run_store,
        cognee_store=cognee_store,
    )

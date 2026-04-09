"""Retrieval strategy router — dispatches searches to Cognee.

Cognee manages both vector (Qdrant) and knowledge graph (Neo4j) storage.
All retrieval strategies route through CogneeStore.

Strategies:
- SEMANTIC: Cognee CHUNKS search (vector similarity via Qdrant)
- KEYWORD: Cognee CHUNKS_LEXICAL search (lexical/BM25)
- HYBRID: CHUNKS + CHUNKS_LEXICAL in parallel
- GRAPH: Cognee GRAPH_COMPLETION (graph traversal + LLM reasoning)
- GRAPH_HYBRID: GRAPH_COMPLETION + CHUNKS + CHUNKS_LEXICAL (triple fusion)
- RAG_COMPLETION: Cognee RAG_COMPLETION (retrieval-augmented generation)
- SUMMARIES: Cognee SUMMARIES (document-level summary search)
- TEMPORAL: Cognee TEMPORAL (time-aware retrieval)
- GRAPH_COT: Cognee GRAPH_COMPLETION_COT (chain-of-thought graph reasoning)
- GRAPH_CONSTRAINED: GRAPH_COMPLETION + CHUNKS (entity-neighbourhood filtering;
  full Neo4j entity extraction forthcoming)
"""

import asyncio
import logging

from haystack import Document

from agentic_ra.core.errors import CogneeError, RoutingError
from agentic_ra.retrieval.strategies import (
    STRATEGY_REGISTRY,
    RetrievalStrategy,
    StrategyDescriptor,
)
from agentic_ra.stores.cognee.client import CogneeStore

# Re-export for backward compatibility — callers that import RetrievalStrategy
# from this module continue to work unchanged.
__all__ = ["RetrievalRouter", "RetrievalStrategy"]

logger = logging.getLogger(__name__)


class RetrievalRouter:
    """Dispatches retrieval to Cognee for all search strategies.

    Uses STRATEGY_REGISTRY for data-driven dispatch instead of
    per-strategy if/elif branches.

    Args:
        cognee_store: CogneeStore for graph + vector + lexical retrieval.
    """

    def __init__(self, cognee_store: CogneeStore) -> None:
        self._cognee = cognee_store

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[Document]:
        """Retrieve documents using the specified strategy.

        Args:
            query: User query text.
            strategy: Retrieval strategy to use.
            limit: Maximum results per source.
            session_id: Optional session identifier forwarded to Cognee for
                feedback tracking.  When provided, Cognee records which graph
                elements were consulted, enabling reinforcement of relevant nodes.

        Returns:
            Retrieved documents (unranked, may contain duplicates for hybrid).
        """
        descriptor = STRATEGY_REGISTRY.get(strategy)
        if descriptor is None:
            raise RoutingError(
                f"Unknown retrieval strategy: {strategy!r}",
                strategy=str(strategy),
                operation="retrieve",
            )

        try:
            return await self._dispatch(descriptor, query, limit, session_id)
        except (RoutingError, CogneeError):
            raise
        except Exception as e:
            raise RoutingError(
                "Retrieval failed",
                strategy=strategy.value,
                operation="retrieve",
                original_error=str(e),
            ) from e

    async def _dispatch(
        self,
        descriptor: StrategyDescriptor,
        query: str,
        limit: int,
        session_id: str | None,
    ) -> list[Document]:
        """Execute retrieval based on strategy descriptor."""
        coros = [
            self._cognee.search(
                query=query,
                search_type=st,
                top_k=limit,
                session_id=session_id,
            )
            for st in descriptor.search_types
        ]

        if descriptor.is_parallel:
            results = await asyncio.gather(*coros, return_exceptions=True)
            return self._collect_results(list(results), descriptor.strategy.value)

        return await coros[0]

    @staticmethod
    def _collect_results(
        results: list[list[Document] | BaseException],
        strategy_name: str,
    ) -> list[Document]:
        """Filter exceptions from ``asyncio.gather(return_exceptions=True)``.

        Raises:
            RoutingError: If all retrieval branches failed.
        """
        combined: list[Document] = []
        failure_count = 0
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("Search failed in %s strategy: %s", strategy_name, r)
                failure_count += 1
            else:
                combined.extend(r)

        if len(combined) == 0 and failure_count > 0:
            raise RoutingError(
                "All retrieval branches failed",
                operation="retrieve",
                strategy=strategy_name,
                failure_count=failure_count,
            )

        return combined

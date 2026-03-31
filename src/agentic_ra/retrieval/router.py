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

from __future__ import annotations

import asyncio
import enum
import logging

from haystack import Document

from agentic_ra.core.errors import CogneeError, RoutingError
from agentic_ra.stores.cognee.client import CogneeStore

logger = logging.getLogger(__name__)


class RetrievalStrategy(enum.StrEnum):
    """Available retrieval strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    GRAPH = "graph"
    GRAPH_HYBRID = "graph_hybrid"
    # Extended Cognee search types
    RAG_COMPLETION = "rag_completion"
    SUMMARIES = "summaries"
    TEMPORAL = "temporal"
    GRAPH_COT = "graph_cot"
    GRAPH_CONSTRAINED = "graph_constrained"


class RetrievalRouter:
    """Dispatches retrieval to Cognee for all search strategies.

    Cognee owns both vector storage (Qdrant) and knowledge graph (Neo4j),
    so all strategies route through the single CogneeStore.

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
        try:
            if strategy == RetrievalStrategy.SEMANTIC:
                return await self._cognee.search(
                    query=query,
                    search_type="CHUNKS",
                    top_k=limit,

                    session_id=session_id,
                )

            if strategy == RetrievalStrategy.KEYWORD:
                return await self._cognee.search(
                    query=query,
                    search_type="CHUNKS_LEXICAL",
                    top_k=limit,

                    session_id=session_id,
                )

            if strategy == RetrievalStrategy.GRAPH:
                return await self._cognee.search(
                    query=query,
                    search_type="GRAPH_COMPLETION",
                    top_k=limit,

                    session_id=session_id,
                )

            if strategy == RetrievalStrategy.HYBRID:
                results = await asyncio.gather(
                    self._cognee.search(
                        query=query,
                        search_type="CHUNKS",
                        top_k=limit,
    
                        session_id=session_id,
                    ),
                    self._cognee.search(
                        query=query,
                        search_type="CHUNKS_LEXICAL",
                        top_k=limit,
    
                        session_id=session_id,
                    ),
                    return_exceptions=True,
                )
                return self._collect_results(list(results), "hybrid")

            # GRAPH_HYBRID: triple fusion
            if strategy == RetrievalStrategy.GRAPH_HYBRID:
                triple_results = await asyncio.gather(
                    self._cognee.search(
                        query=query,
                        search_type="GRAPH_COMPLETION",
                        top_k=limit,
    
                        session_id=session_id,
                    ),
                    self._cognee.search(
                        query=query,
                        search_type="CHUNKS",
                        top_k=limit,
    
                        session_id=session_id,
                    ),
                    self._cognee.search(
                        query=query,
                        search_type="CHUNKS_LEXICAL",
                        top_k=limit,
    
                        session_id=session_id,
                    ),
                    return_exceptions=True,
                )
                return self._collect_results(list(triple_results), "graph_hybrid")

            elif strategy == RetrievalStrategy.RAG_COMPLETION:
                return await self._cognee.search(
                    query=query,
                    search_type="RAG_COMPLETION",
                    top_k=limit,

                    session_id=session_id,
                )

            elif strategy == RetrievalStrategy.SUMMARIES:
                return await self._cognee.search(
                    query=query,
                    search_type="SUMMARIES",
                    top_k=limit,

                    session_id=session_id,
                )

            elif strategy == RetrievalStrategy.TEMPORAL:
                return await self._cognee.search(
                    query=query,
                    search_type="TEMPORAL",
                    top_k=limit,

                    session_id=session_id,
                )

            elif strategy == RetrievalStrategy.GRAPH_COT:
                return await self._cognee.search(
                    query=query,
                    search_type="GRAPH_COMPLETION_COT",
                    top_k=limit,

                    session_id=session_id,
                )

            elif strategy == RetrievalStrategy.GRAPH_CONSTRAINED:
                # Graph-constrained: use GRAPH_COMPLETION to find entity neighbourhood,
                # then CHUNKS filtered to that neighbourhood.
                # Full Neo4j entity extraction + Qdrant filtering is pending;
                # this combines both sources as an interim approach similar to
                # GRAPH_HYBRID but without lexical fusion.
                constrained_results = await asyncio.gather(
                    self._cognee.search(
                        query=query,
                        search_type="GRAPH_COMPLETION",
                        top_k=limit,
    
                        session_id=session_id,
                    ),
                    self._cognee.search(
                        query=query,
                        search_type="CHUNKS",
                        top_k=limit,
    
                        session_id=session_id,
                    ),
                    return_exceptions=True,
                )
                return self._collect_results(list(constrained_results), "graph_constrained")

            else:
                raise RoutingError(
                    f"Unknown retrieval strategy: {strategy!r}",
                    strategy=str(strategy),
                    operation="retrieve",
                )

        except (RoutingError, CogneeError):
            raise
        except Exception as e:
            raise RoutingError(
                "Retrieval failed",
                strategy=strategy.value,
                operation="retrieve",
                original_error=str(e),
            ) from e

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

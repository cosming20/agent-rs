"""Retrieval strategy router — dispatches searches to Cognee.

Cognee manages both vector (Qdrant) and knowledge graph (Neo4j) storage.
All retrieval strategies route through CogneeStore.

Strategies:
- SEMANTIC: Cognee CHUNKS search (vector similarity via Qdrant)
- KEYWORD: Cognee CHUNKS_LEXICAL search (lexical/BM25)
- HYBRID: CHUNKS + CHUNKS_LEXICAL in parallel
- GRAPH: Cognee GRAPH_COMPLETION (graph traversal + LLM reasoning)
- GRAPH_HYBRID: GRAPH_COMPLETION + CHUNKS + CHUNKS_LEXICAL (triple fusion)
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
    ) -> list[Document]:
        """Retrieve documents using the specified strategy.

        Args:
            query: User query text.
            strategy: Retrieval strategy to use.
            limit: Maximum results per source.

        Returns:
            Retrieved documents (unranked, may contain duplicates for hybrid).
        """
        try:
            if strategy == RetrievalStrategy.SEMANTIC:
                return await self._cognee.search(
                    query=query, search_type="CHUNKS", top_k=limit, only_context=True
                )

            if strategy == RetrievalStrategy.KEYWORD:
                return await self._cognee.search(
                    query=query,
                    search_type="CHUNKS_LEXICAL",
                    top_k=limit,
                    only_context=True,
                )

            if strategy == RetrievalStrategy.GRAPH:
                return await self._cognee.search(
                    query=query,
                    search_type="GRAPH_COMPLETION",
                    top_k=limit,
                    only_context=True,
                )

            if strategy == RetrievalStrategy.HYBRID:
                semantic_results, keyword_results = await asyncio.gather(
                    self._cognee.search(
                        query=query,
                        search_type="CHUNKS",
                        top_k=limit,
                        only_context=True,
                    ),
                    self._cognee.search(
                        query=query,
                        search_type="CHUNKS_LEXICAL",
                        top_k=limit,
                        only_context=True,
                    ),
                )
                return semantic_results + keyword_results

            # GRAPH_HYBRID: triple fusion
            graph_results, semantic_results, keyword_results = await asyncio.gather(
                self._cognee.search(
                    query=query,
                    search_type="GRAPH_COMPLETION",
                    top_k=limit,
                    only_context=True,
                ),
                self._cognee.search(
                    query=query,
                    search_type="CHUNKS",
                    top_k=limit,
                    only_context=True,
                ),
                self._cognee.search(
                    query=query,
                    search_type="CHUNKS_LEXICAL",
                    top_k=limit,
                    only_context=True,
                ),
            )
            return graph_results + semantic_results + keyword_results

        except (RoutingError, CogneeError):
            raise
        except Exception as e:
            raise RoutingError(
                "Retrieval failed",
                strategy=strategy.value,
                operation="retrieve",
                original_error=str(e),
            ) from e

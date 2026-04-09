"""Knowledge search tool — searches per-user Cognee datasets."""


import logging
from typing import Any

from haystack import Document

from agentic_ra.core.errors import ToolError
from agentic_ra.observability.tracing import get_tracer
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import RetrievalRouter, RetrievalStrategy
from agentic_ra.retrieval.strategies import STRATEGY_REGISTRY

from ._async_bridge import run_async
from ._formatting import build_contextual_query, format_evidence
from ._types import ToolConfig
from ._utils import sanitize_user_id

logger = logging.getLogger(__name__)


def _search_dataset(
    cognee_store: Any,
    dataset: str,
    query: str,
    strategy: RetrievalStrategy,
    limit: int = 10,
    session_id: str | None = None,
    conversation_context: str = "",
) -> list[Document]:
    """Search a specific Cognee dataset using the given strategy.

    Reads search types and context-augmented types from the canonical
    ``STRATEGY_REGISTRY`` so that all 10 strategies are supported.

    Args:
        cognee_store: CogneeStore instance (typed ``Any`` to avoid circular import).
        dataset: Cognee dataset name (e.g. ``"user_42"``).
        query: The search query text.
        strategy: Which retrieval strategy to use.
        limit: Maximum results per search type.
        session_id: Cognee session ID for feedback tracking.
        conversation_context: Recent conversation summary for search augmentation.

    Returns:
        A flat list of :class:`Document` results across all search types.
    """
    tracer = get_tracer()
    descriptor = STRATEGY_REGISTRY.get(strategy)
    if descriptor is None:
        types: tuple[str, ...] = ("CHUNKS", "CHUNKS_LEXICAL")
        context_types: frozenset[str] = frozenset()
    else:
        types = descriptor.search_types
        context_types = descriptor.context_augmented_types

    all_results: list[Document] = []
    for st in types:
        with tracer.start_as_current_span(
            "cognee.search",
            attributes={
                "search_type": st,
                "dataset": dataset,
                "strategy": strategy.value,
                "query": query[:100],
            },
        ) as search_span:
            effective_query = (
                build_contextual_query(query, conversation_context)
                if st in context_types
                else query
            )
            results: list[Document] = run_async(
                cognee_store.search(
                    query=effective_query, search_type=st, top_k=limit, dataset=dataset,
                    session_id=session_id,
                )
            )
            search_span.set_attribute("result_count", len(results))
            all_results.extend(results)
    return all_results


def make_search_knowledge_tool(
    router: RetrievalRouter,
    ranker: ResultRanker,
    cognee_store: Any = None,
    document_store: Any = None,
    dataset: str = "user_anonymous",
    role: str = "user",
    evidence_accumulator: list[Document] | None = None,
    session_id: str | None = None,
    conversation_context: str = "",
) -> ToolConfig:
    """Create the ``search_knowledge`` tool for the Haystack Agent.

    Searches the knowledge base using per-user dataset scoping.  Regular
    users search only their own dataset; admins fan-out across **all**
    user datasets.

    Args:
        router: Retrieval router (kept for backward compatibility).
        ranker: Result ranker for dedup and RRF fusion.
        cognee_store: CogneeStore instance for direct dataset-scoped search.
        document_store: Document store with ``list_user_datasets()`` (admin fan-out).
        dataset: Cognee dataset name for the current user (e.g. ``"user_42"``).
        role: ``"user"`` or ``"admin"``.
        evidence_accumulator: Shared list to collect KB evidence documents.
        session_id: Cognee session ID for feedback tracking.

    Returns:
        Tool descriptor dict with name, description, function, parameters.
    """

    def search_knowledge(query: str, strategy: str = "hybrid") -> dict[str, Any]:
        """Search the knowledge base and return ranked evidence."""
        try:
            retrieval_strategy = RetrievalStrategy(strategy.lower())
        except ValueError:
            valid = ", ".join(s.value for s in RetrievalStrategy)
            raise ToolError(
                f"Invalid retrieval strategy: {strategy!r}",
                valid_strategies=valid,
                operation="search_knowledge",
            )

        try:
            if cognee_store is not None:
                if role == "admin" and document_store is not None:
                    # Admin fan-out: search all user datasets
                    user_ids: list[str] = run_async(
                        document_store.list_user_datasets()
                    )
                    # Cap to prevent DoS on large deployments
                    MAX_FAN_OUT = 50
                    if len(user_ids) > MAX_FAN_OUT:
                        logger.warning(
                            "Admin fan-out capped at %d users (total: %d)",
                            MAX_FAN_OUT,
                            len(user_ids),
                        )
                        user_ids = user_ids[:MAX_FAN_OUT]

                    all_results: list[Document] = []
                    for uid in user_ids:
                        ds = f"user_{sanitize_user_id(uid)}"
                        try:
                            results = _search_dataset(
                                cognee_store, ds, query, retrieval_strategy,
                                limit=10, session_id=session_id,
                                conversation_context=conversation_context,
                            )
                            all_results.extend(results)
                        except Exception:
                            logger.warning(
                                "Admin fan-out: search failed for dataset %s", ds
                            )
                    raw_docs = all_results
                else:
                    # User: search only their dataset
                    raw_docs = _search_dataset(
                        cognee_store, dataset, query, retrieval_strategy,
                        limit=10, session_id=session_id,
                        conversation_context=conversation_context,
                    )
            else:
                # Fallback: use router (no dataset scoping)
                raw_docs = run_async(
                    router.retrieve(
                        query=query, strategy=retrieval_strategy,
                        session_id=session_id,
                    )
                )

            evidence_pack = ranker.rank(
                results=raw_docs,
                query=query,
                strategy=retrieval_strategy.value,
            )
            formatted = format_evidence(evidence_pack.documents)
            if evidence_accumulator is not None:
                evidence_accumulator.extend(evidence_pack.documents)

            logger.info(
                "search_knowledge returned %d documents (strategy=%s)",
                len(evidence_pack.documents),
                strategy,
            )
            return {
                "text": formatted,
                "documents": list(evidence_pack.documents),
            }
        except ToolError:
            raise
        except Exception as e:
            if "DatasetNotFoundError" in type(e).__name__ or "No datasets found" in str(e):
                logger.warning("No datasets found in knowledge base -- returning empty results")
                return {"text": "No data has been ingested into the knowledge base yet.", "documents": []}
            raise ToolError(
                "Knowledge search failed",
                query=query,
                strategy=strategy,
                operation="search_knowledge",
                original_error=str(e),
            ) from e

    return {
        "name": "search_knowledge",
        "description": (
            "Search the internal knowledge base using semantic, keyword, hybrid, "
            "graph, or graph_hybrid retrieval. Returns ranked evidence documents."
        ),
        "function": search_knowledge,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "strategy": {
                    "type": "string",
                    "description": (
                        "Retrieval strategy: semantic, keyword, hybrid, graph, "
                        "or graph_hybrid."
                    ),
                    "enum": [s.value for s in RetrievalStrategy],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    }

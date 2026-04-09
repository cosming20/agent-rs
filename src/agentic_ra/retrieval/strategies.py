"""Strategy registry — single source of truth for retrieval strategy metadata.

Defines ``RetrievalStrategy`` enum and the ``STRATEGY_REGISTRY`` that maps
every strategy to its declarative ``StrategyDescriptor``.  ``router.py``
re-exports ``RetrievalStrategy`` for backward compatibility; all other modules
should import from here when they need the descriptor metadata.
"""

import enum
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class StrategyDescriptor:
    """Declarative definition of a retrieval strategy.

    Attributes:
        strategy: The retrieval strategy enum value.
        search_types: Cognee SearchType values to dispatch.
        is_parallel: Whether to run search types in parallel via asyncio.gather.
        context_augmented_types: Search types that benefit from conversation
            context prepending.
    """

    strategy: RetrievalStrategy
    search_types: tuple[str, ...]
    is_parallel: bool = False
    context_augmented_types: frozenset[str] = field(default_factory=frozenset)


STRATEGY_REGISTRY: dict[RetrievalStrategy, StrategyDescriptor] = {
    RetrievalStrategy.SEMANTIC: StrategyDescriptor(
        strategy=RetrievalStrategy.SEMANTIC,
        search_types=("CHUNKS",),
        context_augmented_types=frozenset({"CHUNKS"}),
    ),
    RetrievalStrategy.KEYWORD: StrategyDescriptor(
        strategy=RetrievalStrategy.KEYWORD,
        search_types=("CHUNKS_LEXICAL",),
    ),
    RetrievalStrategy.HYBRID: StrategyDescriptor(
        strategy=RetrievalStrategy.HYBRID,
        search_types=("CHUNKS", "CHUNKS_LEXICAL"),
        is_parallel=True,
        context_augmented_types=frozenset({"CHUNKS"}),
    ),
    RetrievalStrategy.GRAPH: StrategyDescriptor(
        strategy=RetrievalStrategy.GRAPH,
        search_types=("GRAPH_COMPLETION",),
        context_augmented_types=frozenset({"GRAPH_COMPLETION"}),
    ),
    RetrievalStrategy.GRAPH_HYBRID: StrategyDescriptor(
        strategy=RetrievalStrategy.GRAPH_HYBRID,
        search_types=("GRAPH_COMPLETION", "CHUNKS", "CHUNKS_LEXICAL"),
        is_parallel=True,
        context_augmented_types=frozenset({"CHUNKS", "GRAPH_COMPLETION"}),
    ),
    RetrievalStrategy.RAG_COMPLETION: StrategyDescriptor(
        strategy=RetrievalStrategy.RAG_COMPLETION,
        search_types=("RAG_COMPLETION",),
    ),
    RetrievalStrategy.SUMMARIES: StrategyDescriptor(
        strategy=RetrievalStrategy.SUMMARIES,
        search_types=("SUMMARIES",),
    ),
    RetrievalStrategy.TEMPORAL: StrategyDescriptor(
        strategy=RetrievalStrategy.TEMPORAL,
        search_types=("TEMPORAL",),
    ),
    RetrievalStrategy.GRAPH_COT: StrategyDescriptor(
        strategy=RetrievalStrategy.GRAPH_COT,
        search_types=("GRAPH_COMPLETION_COT",),
        context_augmented_types=frozenset({"GRAPH_COMPLETION_COT"}),
    ),
    RetrievalStrategy.GRAPH_CONSTRAINED: StrategyDescriptor(
        strategy=RetrievalStrategy.GRAPH_CONSTRAINED,
        # Graph-constrained: use GRAPH_COMPLETION to find entity neighbourhood,
        # then CHUNKS filtered to that neighbourhood.
        # Full Neo4j entity extraction + Qdrant filtering is pending;
        # this combines both sources as an interim approach similar to
        # GRAPH_HYBRID but without lexical fusion.
        search_types=("GRAPH_COMPLETION", "CHUNKS"),
        is_parallel=True,
        context_augmented_types=frozenset({"CHUNKS", "GRAPH_COMPLETION"}),
    ),
}

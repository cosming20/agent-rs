"""Embedding-similarity strategy learner — predicts optimal retrieval strategy from past query performance."""


import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret

from agentic_ra.core.errors import EmbeddingError
from agentic_ra.retrieval.router import RetrievalStrategy
from agentic_ra.stores.qdrant.client import QdrantStore

logger = logging.getLogger(__name__)

_DEFAULT_STRATEGY = RetrievalStrategy.HYBRID


@dataclass(frozen=True)
class StrategyPrediction:
    """Result of a strategy prediction with reasoning.

    Attributes:
        strategy: The predicted retrieval strategy.
        reason: Human-readable explanation of why this strategy was chosen.
        avg_confidence: Average confidence of similar past queries using this strategy.
        sample_count: Number of similar past queries this prediction is based on.
    """

    strategy: RetrievalStrategy
    reason: str
    avg_confidence: float = 0.0
    sample_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.avg_confidence <= 1.0:
            logger.warning(
                "StrategyPrediction avg_confidence=%.4f out of range, clamping",
                self.avg_confidence,
            )
            object.__setattr__(
                self, "avg_confidence",
                max(0.0, min(1.0, self.avg_confidence)),
            )
        if self.sample_count < 0:
            logger.warning(
                "StrategyPrediction sample_count=%d negative, resetting to 0",
                self.sample_count,
            )
            object.__setattr__(self, "sample_count", 0)


@dataclass(frozen=True)
class FailureRecord:
    """A past query that failed grounding verification.

    Attributes:
        strategy: The retrieval strategy that was used.
        confidence: Verifier confidence score at the time.
        evidence_count: Number of evidence documents retrieved.
    """

    strategy: RetrievalStrategy
    confidence: float
    evidence_count: int


def _parse_dt(val: object) -> datetime | None:
    """Parse an ISO timestamp string to a timezone-aware datetime.

    Args:
        val: Value to parse — expected to be an ISO 8601 string.

    Returns:
        Timezone-aware datetime if parsing succeeds, ``None`` otherwise.
    """
    try:
        dt = datetime.fromisoformat(str(val))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _to_strategy(val: object) -> RetrievalStrategy:
    """Safely convert a stored strategy string to RetrievalStrategy enum."""
    try:
        return RetrievalStrategy(str(val))
    except ValueError:
        logger.warning("Unknown strategy value %r, using default", val)
        return _DEFAULT_STRATEGY


class QueryEmbedder:
    """Thin wrapper around Haystack's OpenAITextEmbedder for query-side embedding.

    Encapsulates the synchronous warm-up call and the async-incompatible
    Haystack component so ``StrategyLearner`` can be fully async.

    Args:
        model: OpenAI embedding model name (must match the document embedder model).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var if empty.
        dimension: Embedding dimensionality. Must match the Qdrant collection.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dimension: int = 1536,
    ) -> None:
        self._model = model
        self._embedder = OpenAITextEmbedder(
            model=model,
            api_key=Secret.from_token(api_key)
            if api_key
            else Secret.from_env_var("OPENAI_API_KEY"),
            dimensions=dimension,
        )
        self._embedder.warm_up()  # type: ignore[attr-defined]  # warm_up absent from Haystack stubs

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string and return the raw float vector.

        Args:
            query: The query text to embed.

        Returns:
            Float vector of length ``dimension``.

        Raises:
            EmbeddingError: If the embedding call fails.
        """
        try:
            result = self._embedder.run(text=query)
            embedding: list[float] = result["embedding"]
            return embedding
        except Exception as e:
            raise EmbeddingError(
                "Query embedding failed",
                model=self._model,
                operation="embed_query",
                original_error=str(e),
            ) from e


class StrategyLearner:
    """Predicts best retrieval strategy based on similar past queries.

    Write path: after each pipeline run, ``record()`` stores
    (query_embedding, strategy, confidence, outcome) in a dedicated Qdrant
    collection so the system learns from experience.

    Read path: before routing, ``predict()`` embeds the current query, finds
    the 20 most similar past queries, and returns the strategy with the highest
    average confidence — provided it beats the default by at least
    ``confidence_threshold``.

    Cold start: returns ``_DEFAULT_STRATEGY`` until ``min_neighbors`` results
    exist in the collection.

    Args:
        qdrant: QdrantStore client for the strategy collection.
        embedder: QueryEmbedder for computing query vectors.
        collection: Name of the Qdrant collection for strategy data.
        min_neighbors: Minimum similar queries needed to make a prediction.
        confidence_threshold: Min avg-confidence improvement over default to override.
        max_age_days: Ignore strategy data older than this many days.
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        embedder: QueryEmbedder,
        collection: str = "query_strategies",
        min_neighbors: int = 5,
        confidence_threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._collection = collection
        self._min_neighbors = min_neighbors
        self._confidence_threshold = confidence_threshold
        self._max_age_days = max_age_days

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string using the underlying QueryEmbedder.

        Convenience method so callers can obtain embeddings for ``record()``
        without reaching into private attributes.

        Args:
            query: The query text to embed.

        Returns:
            Float vector of the configured dimensionality.
        """
        return self._embedder.embed_query(query)

    async def predict(
        self, query: str, query_embedding: list[float] | None = None
    ) -> StrategyPrediction:
        """Predict the best strategy for a query with reasoning.

        Returns the default strategy when there is insufficient historical data
        or when no strategy clearly outperforms the default.

        Args:
            query: The query text for which to predict a strategy.
            query_embedding: Optional pre-computed embedding to avoid
                recomputing when the caller already has one.

        Returns:
            StrategyPrediction with strategy name and reasoning.
        """
        try:
            embedding = query_embedding or self._embedder.embed_query(query)

            results = await self._qdrant.search(
                collection=self._collection,
                query_vector=embedding,
                limit=20,
            )

            if len(results) < self._min_neighbors:
                logger.debug(
                    "Strategy prediction: insufficient data (%d < %d), using default",
                    len(results),
                    self._min_neighbors,
                )
                return StrategyPrediction(
                    strategy=_DEFAULT_STRATEGY,
                    reason="Insufficient historical data for prediction",
                    sample_count=len(results),
                )

            cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
            recent = [
                r for r in results
                if (dt := _parse_dt(r.get("created_at"))) is not None and dt >= cutoff
            ]

            if len(recent) < self._min_neighbors:
                logger.debug(
                    "Strategy prediction: too few recent results (%d < %d), using default",
                    len(recent),
                    self._min_neighbors,
                )
                return StrategyPrediction(
                    strategy=_DEFAULT_STRATEGY,
                    reason="Too few recent results for prediction",
                    sample_count=len(recent),
                )

            strategy_scores: dict[str, list[float]] = defaultdict(list)
            for r in recent:
                strategy = str(r.get("strategy", _DEFAULT_STRATEGY.value))
                raw_conf = r.get("confidence", 0.0)
                confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0
                strategy_scores[strategy].append(confidence)

            best_strategy = _DEFAULT_STRATEGY.value
            best_avg = 0.0
            default_avg = 0.0

            for strategy, scores in strategy_scores.items():
                avg = sum(scores) / len(scores)
                if strategy == _DEFAULT_STRATEGY.value:
                    default_avg = avg
                if avg > best_avg:
                    best_avg = avg
                    best_strategy = strategy

            if (
                best_strategy != _DEFAULT_STRATEGY.value
                and (best_avg - default_avg) >= self._confidence_threshold
            ):
                resolved = _to_strategy(best_strategy)
                logger.info(
                    "Strategy prediction: %s (avg=%.2f) over default %s (avg=%.2f)",
                    resolved.value,
                    best_avg,
                    _DEFAULT_STRATEGY.value,
                    default_avg,
                )
                return StrategyPrediction(
                    strategy=resolved,
                    reason=(
                        f"Similar past queries achieved avg confidence {best_avg:.0%} "
                        f"with '{resolved.value}' vs {default_avg:.0%} with default"
                    ),
                    avg_confidence=best_avg,
                    sample_count=len(recent),
                )

            return StrategyPrediction(
                strategy=_DEFAULT_STRATEGY,
                reason=f"Default strategy ({len(recent)} similar queries, avg confidence {default_avg:.0%})",
                avg_confidence=default_avg,
                sample_count=len(recent),
            )

        except EmbeddingError:
            raise
        except Exception:
            logger.exception("Strategy prediction failed, using default")
            return StrategyPrediction(
                strategy=_DEFAULT_STRATEGY, reason="Prediction failed, using default"
            )

    async def find_similar_failures(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[FailureRecord]:
        """Find similar past queries that failed grounding.

        Searches the strategy collection for queries similar to the given
        embedding where ``is_grounded`` was False. Used to warn the agent
        about known failure patterns.

        Args:
            query_embedding: Pre-computed embedding of the current query.
            limit: Maximum number of failures to return.

        Returns:
            List of FailureRecord instances from similar past failures.
        """
        try:
            from qdrant_client import models as qmodels

            results = await self._qdrant.search(
                collection=self._collection,
                query_vector=query_embedding,
                limit=limit,
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="is_grounded",
                            match=qmodels.MatchValue(value=False),
                        ),
                    ],
                ),
            )

            records: list[FailureRecord] = []
            for r in results:
                raw_conf = r.get("confidence", 0.0)
                raw_evi = r.get("evidence_count", 0)
                records.append(FailureRecord(
                    strategy=_to_strategy(r.get("strategy", "unknown")),
                    confidence=float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0,
                    evidence_count=int(raw_evi) if isinstance(raw_evi, (int, float)) else 0,
                ))
            return records
        except Exception:
            logger.warning("Failed to query failure memory", exc_info=True)
            return []

    async def record(
        self,
        query_embedding: list[float],
        strategy: str,
        confidence: float,
        is_grounded: bool,
        retrieval_ms: float,
        evidence_count: int,
    ) -> None:
        """Record a query outcome for future learning.

        Called after each pipeline run to populate the strategy collection.
        Failures are logged but never propagated — recording is best-effort.

        Args:
            query_embedding: Pre-computed query embedding vector.
            strategy: The retrieval strategy that was used.
            confidence: Verifier confidence score (0.0-1.0).
            is_grounded: Whether the answer passed faithfulness check.
            retrieval_ms: Wall-clock retrieval time in milliseconds.
            evidence_count: Number of evidence documents assembled.
        """
        try:
            await self._qdrant.upsert(
                collection=self._collection,
                vector=query_embedding,
                payload={
                    "strategy": strategy,
                    "confidence": confidence,
                    "is_grounded": is_grounded,
                    "retrieval_ms": retrieval_ms,
                    "evidence_count": evidence_count,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                point_id=str(uuid.uuid4()),
            )
        except Exception:
            logger.exception(
                "Failed to record strategy outcome (best-effort, suppressed)"
            )

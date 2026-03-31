"""Embedding-similarity strategy learner — predicts optimal retrieval strategy from past query performance."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret

from agentic_ra.core.errors import EmbeddingError
from agentic_ra.stores.qdrant.client import QdrantStore

logger = logging.getLogger(__name__)

_DEFAULT_STRATEGY = "hybrid"


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
    ) -> str:
        """Predict the best strategy for a query.

        Returns the default strategy when there is insufficient historical data
        or when no strategy clearly outperforms the default.

        Args:
            query: The query text for which to predict a strategy.
            query_embedding: Optional pre-computed embedding to avoid
                recomputing when the caller already has one.

        Returns:
            Strategy name string (e.g. ``"hybrid"``, ``"graph_hybrid"``).
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
                return _DEFAULT_STRATEGY

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
                return _DEFAULT_STRATEGY

            strategy_scores: dict[str, list[float]] = defaultdict(list)
            for r in recent:
                strategy = str(r.get("strategy", _DEFAULT_STRATEGY))
                raw_conf = r.get("confidence", 0.0)
                confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0
                strategy_scores[strategy].append(confidence)

            best_strategy = _DEFAULT_STRATEGY
            best_avg = 0.0
            default_avg = 0.0

            for strategy, scores in strategy_scores.items():
                avg = sum(scores) / len(scores)
                if strategy == _DEFAULT_STRATEGY:
                    default_avg = avg
                if avg > best_avg:
                    best_avg = avg
                    best_strategy = strategy

            if (
                best_strategy != _DEFAULT_STRATEGY
                and (best_avg - default_avg) >= self._confidence_threshold
            ):
                logger.info(
                    "Strategy prediction: %s (avg=%.2f) over default %s (avg=%.2f)",
                    best_strategy,
                    best_avg,
                    _DEFAULT_STRATEGY,
                    default_avg,
                )
                return best_strategy

            return _DEFAULT_STRATEGY

        except EmbeddingError:
            raise
        except Exception:
            logger.exception("Strategy prediction failed, using default")
            return _DEFAULT_STRATEGY

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

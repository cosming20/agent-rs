"""GNN training pipeline — feeds query performance data to RuVector for index optimization."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.runs import RunStore
from agentic_ra.stores.ruvector.client import RuVectorClient

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Collects query performance data and submits to RuVector for GNN training.

    The GNN learns which HNSW routing paths lead to high-confidence results
    for which query patterns.

    Args:
        ruvector: RuVector client for submitting training data.
        run_store: Pipeline run store for collecting training signal.
        min_samples: Minimum samples required before training.
    """

    def __init__(
        self,
        ruvector: RuVectorClient,
        run_store: RunStore,
        min_samples: int = 50,
    ) -> None:
        self._ruvector = ruvector
        self._run_store = run_store
        self._min_samples = min_samples

    async def train(self, since_hours: int = 24) -> dict[str, Any]:
        """Collect recent query data and submit for GNN training.

        Returns dict with training metadata (sample count, status).
        """
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        runs = await self._run_store.get_since(since)

        if len(runs) < self._min_samples:
            logger.info(
                "Insufficient samples for GNN training: %d < %d",
                len(runs), self._min_samples,
            )
            return {"status": "skipped", "samples": len(runs)}

        training_data = {
            "samples": [
                {
                    "query": run.query,
                    "strategy": run.strategy,
                    "confidence": run.confidence,
                    "is_grounded": run.is_grounded,
                    "retrieval_ms": run.retrieval_ms,
                }
                for run in runs
            ]
        }

        await self._ruvector.train_gnn(training_data)

        logger.info("GNN training submitted: %d samples", len(runs))
        return {"status": "submitted", "samples": len(runs)}

"""Manually trigger GNN training on RuVector.

Usage: uv run python scripts/train_gnn.py
       uv run python scripts/train_gnn.py --hours 48
"""

from __future__ import annotations

import argparse
import asyncio
import os

os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

from agentic_ra.core.config import get_settings
from agentic_ra.observability.gnn_trainer import GNNTrainer
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.runs import RunStore
from agentic_ra.stores.ruvector.client import RuVectorClient


async def main() -> None:
    parser = argparse.ArgumentParser(description="Train RuVector GNN")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to use")
    args = parser.parse_args()

    s = get_settings()

    if not s.ruvector_enabled:
        print("Error: ruvector_enabled is False. Set RUVECTOR_ENABLED=true.")
        return

    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    run_store = RunStore(pg_client=pg)
    ruvector = RuVectorClient(
        base_url=s.ruvector_url,
        api_key=s.ruvector_api_key.get_secret_value(),
    )

    trainer = GNNTrainer(
        ruvector=ruvector,
        run_store=run_store,
        min_samples=s.gnn_training_min_samples,
    )

    result = await trainer.train(since_hours=args.hours)
    print(f"GNN training result: {result}")

    await ruvector.close()
    await pg.close()


if __name__ == "__main__":
    asyncio.run(main())

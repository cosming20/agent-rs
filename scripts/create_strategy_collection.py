"""Create the query_strategies Qdrant collection.

Sets up the dedicated collection used by StrategyLearner to store
(query_embedding, strategy, confidence) tuples for similarity-based
strategy prediction (Loop 2).

Usage:
    uv run python scripts/create_strategy_collection.py
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Ensure LLM_API_KEY is set so pydantic-settings doesn't fail on startup
# when this script is run outside the full application context.
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "placeholder"))

from agentic_ra.core.config import get_settings
from agentic_ra.stores.qdrant.client import QdrantStore


async def main() -> None:
    """Create the query_strategies collection if it doesn't exist."""
    s = get_settings()
    qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)

    try:
        await qdrant.ensure_collection(
            name=s.strategy_collection,
            vector_size=s.embedding_dimension,
        )
        logger.info(
            "Collection '%s' ready (dim=%d, url=%s)",
            s.strategy_collection,
            s.embedding_dimension,
            s.qdrant_url,
        )
    finally:
        await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())

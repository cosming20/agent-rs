"""Apply accumulated Cognee session feedback to graph weights.

Usage:
    uv run python scripts/apply_feedback.py --session <id1> <id2>
    uv run python scripts/apply_feedback.py --session my-session --alpha 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from cognee.memify_pipelines.apply_feedback_weights import (  # type: ignore[import-untyped]
    apply_feedback_weights_pipeline,
)
from cognee.modules.users.models import User  # type: ignore[import-untyped]

from agentic_ra.core.config import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Apply feedback weights to graph")
    parser.add_argument("--session", nargs="+", required=True, help="Session IDs to process")
    parser.add_argument("--alpha", type=float, default=None, help="EMA alpha override")
    args = parser.parse_args()

    settings = get_settings()
    alpha = args.alpha or settings.feedback_alpha

    logger.info("Applying feedback weights: sessions=%s, alpha=%s", args.session, alpha)

    system_user = User(id="system", email="system@agentic-rag.local")

    result = await apply_feedback_weights_pipeline(
        user=system_user,
        session_ids=args.session,
        alpha=alpha,
    )
    logger.info("Result: %s", result)


if __name__ == "__main__":
    asyncio.run(main())

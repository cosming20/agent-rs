"""Emergency rollback — reset all graph feedback_weight to 0.5.

Usage:
    uv run python scripts/reset_feedback_weights.py
    uv run python scripts/reset_feedback_weights.py --confirm
"""

from __future__ import annotations

import argparse
import asyncio
import os

os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

from agentic_ra.core.config import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reset all graph feedback weights")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if not args.confirm:
        resp = input("This will reset ALL feedback_weight values to 0.5. Continue? [y/N] ")
        if resp.lower() != "y":
            print("Aborted.")
            return

    s = get_settings()

    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        s.neo4j_bolt_url,
        auth=(s.neo4j_user, s.neo4j_password.get_secret_value()),
    )

    async with driver.session(database=s.neo4j_db) as session:
        result = await session.run(
            "MATCH (n) WHERE n.feedback_weight IS NOT NULL "
            "SET n.feedback_weight = 0.5 "
            "RETURN count(n) AS reset_count"
        )
        record = await result.single()
        node_count = record["reset_count"] if record else 0

        result = await session.run(
            "MATCH ()-[r]->() WHERE r.feedback_weight IS NOT NULL "
            "SET r.feedback_weight = 0.5 "
            "RETURN count(r) AS reset_count"
        )
        record = await result.single()
        edge_count = record["reset_count"] if record else 0

    await driver.close()
    print(f"Reset complete: {node_count} nodes, {edge_count} edges → 0.5")


if __name__ == "__main__":
    asyncio.run(main())

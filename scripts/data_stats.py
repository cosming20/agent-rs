"""Show ingested data stats across all stores."""

import asyncio

from sqlalchemy import func, select

from agentic_ra.core.config import get_settings
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.models import ChatMessage, MemoryEntry, ParsedDocument


async def main() -> None:
    s = get_settings()

    # Postgres
    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    async with pg.session() as sess:
        parsed_count = (
            await sess.execute(select(func.count(ParsedDocument.id)))
        ).scalar() or 0
        chat_count = (
            await sess.execute(select(func.count(ChatMessage.id)))
        ).scalar() or 0
        memory_count = (
            await sess.execute(select(func.count(MemoryEntry.id)))
        ).scalar() or 0

    print("=== Postgres ===")
    print(f"  Parsed documents: {parsed_count}")
    print(f"  Chat messages:    {chat_count}")
    print(f"  Memory entries:   {memory_count}")
    await pg.close()

    # Cognee manages Qdrant (vector) and Neo4j (graph) — no public count APIs.
    # Vector/graph stats are available via Qdrant and Neo4j UIs directly.
    print("\n=== Cognee ===")
    print(f"  Dataset:         {s.cognee_dataset}")
    print(f"  Vector backend:  {s.cognee_vector_db_provider} ({s.qdrant_url})")
    print(f"  Graph backend:   {s.cognee_graph_db_provider} ({s.neo4j_bolt_url})")
    print("  (Use Qdrant/Neo4j UIs for detailed vector and graph stats)")


if __name__ == "__main__":
    asyncio.run(main())

"""Verify connectivity to all stores (Postgres, Qdrant, Neo4j)."""

import asyncio

import sqlalchemy

from agentic_ra.core.config import get_settings
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.qdrant.client import QdrantStore


async def check_neo4j(host: str, port: int) -> None:
    """Verify Neo4j is reachable via a TCP connection to bolt port."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.close()
    await writer.wait_closed()


async def main() -> None:
    s = get_settings()

    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    async with pg.session() as sess:
        await sess.execute(sqlalchemy.text("SELECT 1"))
    print("  Postgres OK")
    await pg.close()

    q = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    await q._client.get_collections()
    print("  Qdrant OK")
    await q.close()

    await check_neo4j(s.neo4j_host, s.neo4j_bolt_port)
    print("  Neo4j OK")

    print("All stores connected")


if __name__ == "__main__":
    asyncio.run(main())

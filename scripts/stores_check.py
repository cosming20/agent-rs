"""Verify connectivity to all stores (Postgres, Qdrant, OpenSearch)."""

import asyncio

import sqlalchemy

from agentic_ra.core.config import get_settings
from agentic_ra.stores.opensearch.client import OpenSearchStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.qdrant.client import QdrantStore


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

    o = OpenSearchStore(
        host=s.opensearch_host,
        port=s.opensearch_port,
        user=s.opensearch_user,
        password=s.opensearch_initial_admin_password,
    )
    await o._client.info()
    print("  OpenSearch OK")
    await o.close()

    print("All stores connected")


if __name__ == "__main__":
    asyncio.run(main())

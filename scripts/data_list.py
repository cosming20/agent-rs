"""List all parsed documents stored in Postgres."""

import asyncio

from sqlalchemy import select

from agentic_ra.core.config import get_settings
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.models import ParsedDocument


async def main() -> None:
    s = get_settings()
    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())

    async with pg.session() as sess:
        result = await sess.execute(
            select(ParsedDocument).order_by(ParsedDocument.created_at.desc())
        )
        docs = result.scalars().all()

    if not docs:
        print("No parsed documents found.")
    else:
        print(f"{'ID':<38} {'Chunks':>6}  {'Source'}")
        print("-" * 80)
        for doc in docs:
            print(f"{doc.id}  {doc.chunk_count:>6}  {doc.source_path}")

    await pg.close()


if __name__ == "__main__":
    asyncio.run(main())

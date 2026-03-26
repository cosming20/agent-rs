"""Preview chunks stored in Qdrant."""

import asyncio
import sys

from agentic_ra.core.config import get_settings
from agentic_ra.stores.qdrant.client import QdrantStore


async def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    s = get_settings()

    q = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    try:
        result = await q._client.scroll(
            collection_name="documents",
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = result

        if not points:
            print("No chunks in Qdrant.")
        else:
            for i, p in enumerate(points, 1):
                payload = p.payload or {}
                content = payload.get("chunk_text", payload.get("content", ""))
                headings = payload.get("headings", [])
                source = payload.get("source_path", payload.get("source", "unknown"))
                print(f"--- Chunk {i} ---")
                if headings:
                    print(f"  Headings: {headings}")
                print(f"  Source: {source}")
                print(f"  Text: {content[:200]}{'...' if len(content) > 200 else ''}")
                print()

        info = await q._client.get_collection("documents")
        print(f"Total vectors: {info.points_count}")
    except Exception as e:
        print(f"Error: {e}")
    await q.close()


if __name__ == "__main__":
    asyncio.run(main())

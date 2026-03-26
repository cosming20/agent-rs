"""Ingest a file (PDF, DOCX, PPTX, HTML, etc.) into the full RAG pipeline.

Parse with Docling → chunk with HybridChunker → embed with OpenAI →
store in Qdrant + OpenSearch → persist DoclingDocument JSON in Postgres.

Usage: uv run python scripts/ingest_file.py <file_or_url> [file2] [file3] ...
"""

from __future__ import annotations

import asyncio
import sys
import time

from qdrant_client import models as qdrant_models

from agentic_ra.core.config import get_settings
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.stores.opensearch.client import OpenSearchStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore
from agentic_ra.stores.qdrant.client import QdrantStore


async def ingest_file(source: str) -> None:
    s = get_settings()
    start = time.time()

    # Build components
    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    doc_store = DocumentStore(pg_client=pg)
    qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    opensearch = OpenSearchStore(
        host=s.opensearch_host,
        port=s.opensearch_port,
        user=s.opensearch_user,
        password=s.opensearch_initial_admin_password,
    )
    pipeline = EmbeddingPipeline(
        model=s.embedding_model,
        api_key=s.openai_api_key.get_secret_value(),
        dimension=s.embedding_dimension,
        max_tokens=s.chunk_max_tokens,
        merge_peers=s.chunk_merge_peers,
    )

    # Ensure stores exist
    await qdrant.ensure_collection("documents", vector_size=s.embedding_dimension)
    await opensearch.ensure_index("documents")

    # 1. Parse + chunk
    print(f"Parsing: {source}")
    chunk_result = pipeline.chunk_file(source)
    print(f"  Chunks: {chunk_result.chunk_count}")

    if chunk_result.chunk_count == 0:
        print("  No chunks produced — skipping.")
        await qdrant.close()
        await opensearch.close()
        await pg.close()
        return

    # 2. Persist DoclingDocument JSON
    docling_json = (
        chunk_result.docling_doc.export_to_dict()
        if chunk_result.docling_doc is not None
        else {}
    )
    parsed_doc = await doc_store.save(
        source_path=source,
        docling_json=docling_json,
        chunk_count=chunk_result.chunk_count,
        metadata={"file": source},
    )
    print(f"  Persisted DoclingDocument: {parsed_doc.id}")

    # Inject parsed_document_id into chunk metadata
    for doc in chunk_result.documents:
        doc.meta["parsed_document_id"] = str(parsed_doc.id)

    # 3. Embed
    print(f"  Embedding {len(chunk_result.documents)} chunks...")
    embedded = pipeline.embed(chunk_result.documents)
    print(f"  Embedded: {len(embedded)} chunks ({s.embedding_dimension}d)")

    # 4. Store in Qdrant
    # Use UUID-based IDs to avoid collisions across ingestions
    import uuid

    points = [
        qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=doc.embedding,
            payload={
                "content": doc.content,
                "chunk_text": doc.meta.get("chunk_text", doc.content),
                "parsed_document_id": doc.meta.get("parsed_document_id", ""),
                "headings": doc.meta.get("headings", []),
                "source_path": source,
            },
        )
        for doc in embedded
        if doc.embedding is not None
    ]

    if points:
        await qdrant._client.upsert(collection_name="documents", points=points)
        print(f"  Qdrant: {len(points)} vectors stored")

    # 5. Store in OpenSearch
    for doc in embedded:
        await opensearch._client.index(
            index="documents",
            id=str(uuid.uuid4()),
            body={
                "content": doc.meta.get("chunk_text", doc.content),
                "parsed_document_id": doc.meta.get("parsed_document_id", ""),
                "headings": doc.meta.get("headings", []),
                "source_path": source,
            },
            refresh=True,
        )
    print(f"  OpenSearch: {len(embedded)} documents indexed")

    # Cleanup
    await qdrant.close()
    await opensearch.close()
    await pg.close()

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/ingest_file.py <file_or_url> [file2] ...")
        sys.exit(1)

    for source in sys.argv[1:]:
        await ingest_file(source)
        print()

    print("All files ingested.")


if __name__ == "__main__":
    asyncio.run(main())

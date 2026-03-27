"""Ingest a file (PDF, DOCX, PPTX, HTML, etc.) into the RAG pipeline.

Flow: Docling parse → Postgres DoclingDocument JSON persistence
      → Cognee add+cognify (vector+graph via Qdrant + Neo4j)

Usage: uv run python scripts/ingest_file.py <file_or_url> [file2] [file3] ...
"""

from __future__ import annotations

import asyncio
import sys
import time

from agentic_ra.core.config import get_settings
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.stores.cognee.client import CogneeStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore


async def ingest_file(source: str) -> None:
    s = get_settings()
    start = time.time()

    # Build components
    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    doc_store = DocumentStore(pg_client=pg)
    pipeline = EmbeddingPipeline(
        model=s.embedding_model,
        api_key=s.llm_api_key.get_secret_value(),
        dimension=s.embedding_dimension,
        max_tokens=s.chunk_max_tokens,
        merge_peers=s.chunk_merge_peers,
    )

    # Cognee store — always required (Cognee is now the sole retrieval backend)
    cognee_store = CogneeStore(dataset=s.cognee_dataset)
    cognee_store.configure(
        llm_api_key=s.llm_api_key.get_secret_value(),
        llm_model=s.llm_enrichment_model,
        graph_db_provider=s.cognee_graph_db_provider,
        vector_db_provider=s.cognee_vector_db_provider,
        data_root=s.cognee_data_root,
        system_root=s.cognee_system_root,
        qdrant_url=s.qdrant_url,
        qdrant_api_key=s.qdrant_api_key.get_secret_value(),
        neo4j_bolt_url=s.neo4j_bolt_url,
        neo4j_user=s.neo4j_user,
        neo4j_password=s.neo4j_password.get_secret_value(),
        neo4j_db=s.neo4j_db,
    )

    # 1. Parse + chunk with Docling
    print(f"Parsing: {source}")
    chunk_result = pipeline.chunk_file(source)
    print(f"  Chunks: {chunk_result.chunk_count}")

    if chunk_result.chunk_count == 0:
        print("  No chunks produced — skipping.")
        await pg.close()
        return

    # 2. Persist DoclingDocument JSON in Postgres
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

    # 3. Cognee: add individual chunks + cognify (embedding → Qdrant, entities → Neo4j)
    print("  Building breadcrumb chunks...")
    breadcrumb_chunks = pipeline.build_breadcrumb_chunks(chunk_result)
    print(f"  Breadcrumb chunks: {len(breadcrumb_chunks)}")

    print("  Cognee add (per-chunk) + cognify...")
    await cognee_store.add_chunks(chunks=breadcrumb_chunks)
    print("  Cognee: graph + vector ingestion complete")

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

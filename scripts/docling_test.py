"""Test Docling chunking only — no Cognee, no Postgres, no embedding.

Parses a file with Docling, chunks it, and saves results as JSON for inspection.

Usage: uv run python scripts/docling_test.py test.pdf
Output: docling_chunks.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agentic_ra.core.config import get_settings
from agentic_ra.retrieval.embedder import EmbeddingPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/docling_test.py <file>")
        sys.exit(1)

    source = sys.argv[1]
    s = get_settings()

    pipeline = EmbeddingPipeline(
        model=s.embedding_model,
        api_key=s.llm_api_key.get_secret_value(),
        dimension=s.embedding_dimension,
        max_tokens=s.chunk_max_tokens,
        merge_peers=s.chunk_merge_peers,
    )

    print(f"Parsing: {source}")
    print(f"  max_tokens={s.chunk_max_tokens}, merge_peers={s.chunk_merge_peers}")
    start = time.time()
    result = pipeline.chunk_file(source)
    elapsed = time.time() - start
    print(f"  Parsed in {elapsed:.1f}s → {result.chunk_count} chunks")

    # Build output
    chunks = []
    for i, doc in enumerate(result.documents):
        txt = doc.meta.get("chunk_text", doc.content or "")
        headings = doc.meta.get("headings", [])
        chunks.append({
            "index": i,
            "chars": len(txt),
            "headings": headings,
            "text": txt,
        })

    # Stats
    sizes = [c["chars"] for c in chunks]
    print(f"\n  Min: {min(sizes)} chars")
    print(f"  Max: {max(sizes)} chars")
    print(f"  Avg: {sum(sizes) // len(sizes)} chars")
    print(f"  Under 50 chars: {sum(1 for s in sizes if s < 50)}")
    print(f"  Over 2000 chars: {sum(1 for s in sizes if s > 2000)}")

    out = Path("docling_chunks.json")
    out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

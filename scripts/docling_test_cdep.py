"""Test Docling HierarchicalChunker on crawled CDEP markdown.

Flow: CDEP markdown → strip boilerplate → temp .md file → Docling parse
      → DoclingDocument → HierarchicalChunker → structured chunks

Usage:
    uv run python scripts/docling_test_cdep.py                         # random page
    uv run python scripts/docling_test_cdep.py --page-type deputy_profile
    uv run python scripts/docling_test_cdep.py --page-id 42
    uv run python scripts/docling_test_cdep.py --search "comisia juridica"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import asyncpg

# Reuse the boilerplate stripping from ingest_cdep
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_cdep import strip_boilerplate

CRAWLER_DB_DSN = "postgresql://agentic_ra:changeme-postgres@localhost:1054/cdep_crawler"


async def fetch_page(
    page_id: int | None = None,
    page_type: str | None = None,
    search: str | None = None,
) -> dict | None:
    """Fetch a single page from cdep_crawler."""
    conn = await asyncpg.connect(CRAWLER_DB_DSN)
    try:
        conditions = [
            "pc.fetch_status = 'success'",
            "pc.markdown_text IS NOT NULL",
            "length(pc.markdown_text) > 200",
        ]

        if page_id:
            conditions.append(f"p.id = {page_id}")
        if page_type:
            conditions.append(f"p.page_type = '{page_type}'")
        if search:
            conditions.append(f"pc.markdown_text ILIKE '%{search}%'")

        where = " AND ".join(conditions)
        query = f"""
            SELECT p.id, p.normalized_url, p.page_type, p.canonical_id,
                   pc.markdown_text
            FROM pages p
            JOIN page_captures pc ON pc.page_id = p.id AND pc.is_latest = TRUE
            WHERE {where}
            ORDER BY random()
            LIMIT 1
        """
        row = await conn.fetchrow(query)
        return dict(row) if row else None
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Docling HierarchicalChunker on CDEP markdown"
    )
    parser.add_argument("--page-id", type=int, default=None, help="Specific page ID")
    parser.add_argument("--page-type", type=str, default=None, help="Filter by page type")
    parser.add_argument("--search", type=str, default=None, help="Search in markdown text")
    args = parser.parse_args()

    page = asyncio.run(fetch_page(args.page_id, args.page_type, args.search))
    if not page:
        print("No matching page found.")
        return

    raw_len = len(page["markdown_text"])
    clean_text = strip_boilerplate(page["markdown_text"], page_id=page.get("id"))
    clean_len = len(clean_text)

    print(f"Page: id={page['id']}, type={page['page_type']}")
    print(f"  URL: {page['normalized_url']}")
    print(f"  Raw: {raw_len} chars → Clean: {clean_len} chars ({100 - clean_len * 100 // raw_len}% stripped)")

    if not clean_text.strip():
        print("  Empty after boilerplate strip.")
        return

    # Write cleaned markdown to temp file for Docling
    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(clean_text)
        tmp_path = f.name

    try:
        import logging as _logging
        # Suppress Docling's verbose debug logging
        for _noisy in ("docling.backend.md_backend", "docling.document_converter",
                        "docling.pipeline", "docling.models", "docling.datamodel"):
            _logging.getLogger(_noisy).setLevel(_logging.WARNING)

        from docling.chunking import HierarchicalChunker
        from docling.document_converter import DocumentConverter

        # Parse markdown with Docling
        start = time.time()
        print("  Parsing markdown with Docling...")
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        doc = result.document
        parse_time = time.time() - start
        print(f"  Parsed in {parse_time:.1f}s")

        # Chunk with HierarchicalChunker
        print("  Chunking with HierarchicalChunker...")
        chunker = HierarchicalChunker()
        doc_chunks = list(chunker.chunk(doc))
        print(f"  Chunks: {len(doc_chunks)}")

        # Build output — merge tiny chunks (<100 chars) into next sibling
        raw_chunks = []
        for c in doc_chunks:
            text = c.text
            headings = []
            if hasattr(c, "meta") and c.meta:
                headings = getattr(c.meta, "headings", []) or []
            raw_chunks.append({"chars": len(text), "headings": headings, "text": text})

        # Merge pass: absorb tiny chunks into the next chunk
        chunks = []
        pending = None
        for rc in raw_chunks:
            if pending is not None:
                # Merge pending into this chunk
                rc["text"] = pending["text"] + "\n\n" + rc["text"]
                rc["chars"] = len(rc["text"])
                # Keep headings from both
                rc["headings"] = pending["headings"] + rc["headings"]
                pending = None
            if rc["chars"] < 100:
                pending = rc
            else:
                chunks.append(rc)
        # If last chunk was tiny, just add it
        if pending is not None:
            chunks.append(pending)

        # Re-index
        for i, c in enumerate(chunks):
            c["index"] = i

        # Stats
        sizes = [c["chars"] for c in chunks]
        if sizes:
            print(f"\n  Min: {min(sizes)} chars")
            print(f"  Max: {max(sizes)} chars")
            print(f"  Avg: {sum(sizes) // len(sizes)} chars")
            print(f"  Under 50 chars: {sum(1 for s in sizes if s < 50)}")
            print(f"  Over 2000 chars: {sum(1 for s in sizes if s > 2000)}")

        # Preview first 5 chunks
        print(f"\n=== FIRST 5 CHUNKS ===")
        for c in chunks[:5]:
            t = c["text"]
            print(f'  [{c["index"]}] {c["chars"]}c: {t[:120]}...')
            if c["headings"]:
                print(f'    Headings: {c["headings"]}')
            print(f'    END: ...{t[-60:]}')
            print()

        # Preview last 3 chunks
        if len(chunks) > 5:
            print(f"=== LAST 3 CHUNKS ===")
            for c in chunks[-3:]:
                t = c["text"]
                print(f'  [{c["index"]}] {c["chars"]}c: {t[:120]}...')
                print(f'    END: ...{t[-60:]}')
                print()

        out = Path("cdep_chunks.json")
        out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {out} ({out.stat().st_size // 1024} KB)")

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Save source for comparison
    src_out = Path("cdep_source.md")
    src_out.write_text(clean_text, encoding="utf-8")
    print(f"Source markdown saved to {src_out}")


if __name__ == "__main__":
    main()

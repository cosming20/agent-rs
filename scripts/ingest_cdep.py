"""Ingest crawled CDEP pages from the crawler PostgreSQL into Cognee.

Reads pages from cdep_crawler database, strips CDEP boilerplate (nav, footer,
cookie banners), chunks clean content with Docling HierarchicalChunker,
prepends metadata headers, and feeds them through cognee.add_chunks() + cognify.

Supports BFS depth-based page selection from homepage for scoped chatbot ingestion.

Usage:
    uv run python scripts/ingest_cdep.py --priority high --limit 5   # test
    uv run python scripts/ingest_cdep.py --priority high              # deputies, committees, declarations
    uv run python scripts/ingest_cdep.py --priority medium            # + expenses, stenograms, votes
    uv run python scripts/ingest_cdep.py --priority all               # everything
    uv run python scripts/ingest_cdep.py --page-types deputy_profile,committee
    uv run python scripts/ingest_cdep.py --dry-run --limit 3         # preview chunks, no Cognee
    uv run python scripts/ingest_cdep.py --depth 2                   # BFS: depth 0+1 all, depth 2 high-priority
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path

import asyncpg

from agentic_ra.core.config import get_settings
from agentic_ra.observability.cost_tracker import setup_cost_tracking
from agentic_ra.stores.cognee.client import CogneeStore

logger = logging.getLogger(__name__)

# --- Crawler DB (separate from agentic-rag Postgres) ---
CRAWLER_DB_DSN = "postgresql://agentic_ra:changeme-postgres@localhost:1054/cdep_crawler"

# --- Priority tiers ---
PRIORITY_HIGH = [
    "deputy_profile", "committee", "bureau", "group",
    "declaration", "deputy_list", "district",
]
PRIORITY_MEDIUM = [
    "expenses", "institutional", "interpellation", "agenda",
    "stenogram", "vote", "motion", "press_release",
    "petitions", "gallery", "law_project", "committee_docs",
    "legislative_acts",
]
PRIORITY_LOW = [
    "session", "foreign_affairs", "procurement", "eu_affairs",
]
# Excluded: "unknown", "navigation", "document"

PRIORITY_MAP = {
    "high": PRIORITY_HIGH,
    "medium": PRIORITY_HIGH + PRIORITY_MEDIUM,
    "all": PRIORITY_HIGH + PRIORITY_MEDIUM + PRIORITY_LOW,
}


# ---------------------------------------------------------------------------
# Boilerplate stripping (2-phase, fail-safe)
# ---------------------------------------------------------------------------

# Phase 1 header-end marker — #eCdep appears twice in the nav template:
# once at the top (line ~0-2) and once at the bottom (line ~46-48).
# The SECOND occurrence marks the end of the nav. Content starts after it.
_NAV_END_MARKER = "#eCdep"

# Footer markers — the address block and useful links section.
# The crawler's PruningContentFilter already removed the footer logo,
# but kept the address/links block because it has text density.
_FOOTER_MARKERS = [
    "#### Adresa poştală",
    "#### Legături utile",
    "Copyright © Camera Deputaţilor",
]

# Phase 2 — non-textual artifacts to remove from the content zone.
_ARTIFACT_PATTERNS = [
    re.compile(r"^\|\s+\|\s+\|\s+\|"),           # empty markdown table cells
    re.compile(r"^\| ---"),                        # table separator-only lines
    re.compile(r"\[Loading\.\.\.\]\(javascript"),  # JS loading placeholders
    re.compile(r"^!\[\]\(https?://"),              # bare images with no alt text
    re.compile(r"highslide/graphics"),             # highslide JS artifacts
    # Activity nav links: lines of just year links (e.g. "[2026](https://...)")
    re.compile(r"^(\[?\d{4}\]?\(https://www\.cdep\.ro/.*?\)\s*){3,}"),
    # Calendar/session/document nav lines
    re.compile(r"^\[Calendar lunar"),
    re.compile(r"^Şedinţe:\s+\["),
    re.compile(r"^Documente:\s+\["),
    # Breadcrumb nav lines
    re.compile(r"^\|\s*Sunteţi aici:"),
]


def strip_boilerplate(markdown: str, page_id: int | None = None) -> str:
    """Remove CDEP template header/footer, keep all content.

    Phase 1 — Structural cut:
      Find nav-end markers in first 120 lines → content starts after.
      Find footer logo in last 60 lines → content ends before.
      If no markers found → return text unchanged (safe fallback).
      If stripping removes >80% of text → return unchanged (sanity check).

    Phase 2 — Artifact cleanup (within content zone only):
      Remove empty table scaffolding, JS placeholders, bare images.
      Never removes lines containing readable text.
    """
    lines = markdown.split("\n")
    original_len = len(markdown)

    # --- Phase 1: find content boundaries ---

    # Header: find the SECOND occurrence of #eCdep (bottom of nav)
    content_start = 0
    ecdep_count = 0
    search_limit = min(120, len(lines))
    for i in range(search_limit):
        if _NAV_END_MARKER in lines[i]:
            ecdep_count += 1
            if ecdep_count >= 2:
                content_start = i + 1
                break

    # Footer: first occurrence of any footer marker in last 60 lines
    content_end = len(lines)
    footer_search_start = max(0, len(lines) - 60)
    for i in range(footer_search_start, len(lines)):
        if any(marker in lines[i] for marker in _FOOTER_MARKERS):
            content_end = i
            break

    # Fallback: no markers found → don't strip
    if content_start == 0 and content_end == len(lines):
        logger.debug("No boilerplate markers found for page %s, skipping strip", page_id)
        content_lines = lines
    else:
        content_lines = lines[content_start:content_end]

    # Sanity check: if we removed >80% of text, something is wrong
    content_text = "\n".join(content_lines)
    if original_len > 0 and len(content_text) < original_len * 0.2:
        logger.warning(
            "Boilerplate strip removed >80%% for page %s (%d → %d chars), aborting strip",
            page_id, original_len, len(content_text),
        )
        content_lines = lines

    # --- Phase 2: remove non-textual artifacts ---
    cleaned: list[str] = []
    for line in content_lines:
        if any(p.search(line) for p in _ARTIFACT_PATTERNS):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Collapse 3+ consecutive blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result


# ---------------------------------------------------------------------------
# Chunking (Docling HierarchicalChunker — heading-aware, no mid-sentence cuts)
# ---------------------------------------------------------------------------

# Lazy-init Docling imports (heavy, only load once)
_docling_converter = None
_docling_chunker = None


def _get_docling():
    """Lazy-init Docling converter + chunker (heavy imports)."""
    global _docling_converter, _docling_chunker
    if _docling_converter is None:
        # Suppress Docling's verbose debug logging
        for name in ("docling.backend.md_backend", "docling.document_converter",
                      "docling.pipeline", "docling.models", "docling.datamodel"):
            logging.getLogger(name).setLevel(logging.WARNING)
        from docling.chunking import HierarchicalChunker
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
        _docling_chunker = HierarchicalChunker()
    return _docling_converter, _docling_chunker


def chunk_page(page: dict) -> list[str]:
    """Strip boilerplate, chunk with Docling HierarchicalChunker, prepend metadata."""
    url = page["normalized_url"]
    page_type = page["page_type"]
    canonical_id = page.get("canonical_id") or "N/A"

    clean_text = strip_boilerplate(page["markdown_text"], page_id=page.get("id"))

    if not clean_text.strip():
        return []

    header = (
        f"Source: Camera Deputaților (cdep.ro)\n"
        f"URL: {url}\n"
        f"Page Type: {page_type}\n"
        f"Entity ID: {canonical_id}\n---\n\n"
    )
    header_len = len(header)

    # Parse markdown with Docling → structured document → hierarchical chunks
    converter, chunker = _get_docling()

    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(clean_text)
        tmp_path = f.name

    try:
        result = converter.convert(tmp_path)
        doc_chunks = list(chunker.chunk(result.document))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not doc_chunks:
        return []

    # Merge tiny chunks (<100 chars) into next sibling
    merged: list[str] = []
    pending: str | None = None
    for c in doc_chunks:
        text = c.text
        if pending is not None:
            text = pending + "\n\n" + text
            pending = None
        if len(text) < 100:
            pending = text
        else:
            merged.append(text)
    if pending is not None:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + pending
        else:
            merged.append(pending)

    # Split oversized chunks (>5000 chars) to stay under embedding token limit
    # text-embedding-3-large has 8192 token limit; URLs tokenize ~4x chars
    MAX_CHUNK_CHARS = 5000
    final: list[str] = []
    for text in merged:
        if not text.strip():
            continue
        while len(text) > MAX_CHUNK_CHARS:
            # Split at last paragraph break before limit
            cut = text[:MAX_CHUNK_CHARS].rfind("\n\n")
            if cut < MAX_CHUNK_CHARS // 3:
                # No good break point — split at last sentence
                cut = text[:MAX_CHUNK_CHARS].rfind(". ")
                if cut < MAX_CHUNK_CHARS // 3:
                    cut = MAX_CHUNK_CHARS  # hard cut as last resort
                else:
                    cut += 1  # include the period
            final.append(text[:cut].strip())
            text = text[cut:].strip()
        if text.strip():
            final.append(text)

    # Prepend metadata header to each chunk
    chunks = [header + chunk for chunk in final]

    # Filter out chunks that are just the header with no real content
    return [c for c in chunks if len(c) > header_len + 50]


# ---------------------------------------------------------------------------
# BFS depth from homepage (link graph traversal)
# ---------------------------------------------------------------------------

async def compute_bfs_depths(max_depth: int = 2) -> dict[int, int]:
    """BFS from cdep.ro homepage, return {page_id: depth}."""
    conn = await asyncpg.connect(CRAWLER_DB_DSN)
    try:
        # Build URL → page_id mapping
        rows = await conn.fetch("SELECT id, normalized_url FROM pages")
        url_to_id = {r["normalized_url"]: r["id"] for r in rows}
        id_to_url = {r["id"]: r["normalized_url"] for r in rows}

        # Build adjacency list via URL join (target_page_id is unpopulated)
        links = await conn.fetch(
            "SELECT source_page_id, normalized_target_url FROM links WHERE is_internal = true"
        )
        graph: dict[int, set[int]] = {}
        for link in links:
            target_id = url_to_id.get(link["normalized_target_url"])
            if target_id:
                graph.setdefault(link["source_page_id"], set()).add(target_id)

        # Seed: homepage variants
        seeds = set()
        for pid, url in id_to_url.items():
            if "site2015.home" in url or url in ("https://www.cdep.ro/", "https://www.cdep.ro/."):
                seeds.add(pid)

        # BFS
        visited: dict[int, int] = {}
        queue = list(seeds)
        for s in seeds:
            visited[s] = 0
        i = 0
        while i < len(queue):
            pid = queue[i]
            i += 1
            depth = visited[pid]
            if depth >= max_depth:
                continue
            for nb in graph.get(pid, []):
                if nb not in visited:
                    visited[nb] = depth + 1
                    queue.append(nb)

        return visited
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

async def fetch_pages(
    page_types: list[str] | None = None,
    page_ids: set[int] | None = None,
    min_length: int = 200,
    limit: int | None = None,
) -> list[dict]:
    """Fetch crawled pages from the cdep_crawler database."""
    conn = await asyncpg.connect(CRAWLER_DB_DSN)
    try:
        conditions = [
            "pc.fetch_status = 'success'",
            "pc.markdown_text IS NOT NULL",
            f"length(pc.markdown_text) > {min_length}",
            "p.page_type NOT IN ('navigation', 'unknown', 'document')",
        ]

        if page_types:
            types_str = ",".join(f"'{t}'" for t in page_types)
            conditions.append(f"p.page_type IN ({types_str})")

        if page_ids is not None:
            ids_str = ",".join(str(i) for i in page_ids)
            conditions.append(f"p.id IN ({ids_str})")

        where = " AND ".join(conditions)
        query = f"""
            SELECT p.id, p.normalized_url, p.page_type, p.canonical_id,
                   pc.markdown_text
            FROM pages p
            JOIN page_captures pc ON pc.page_id = p.id AND pc.is_latest = TRUE
            WHERE {where}
            ORDER BY p.page_type, p.id
        """
        if limit:
            query += f" LIMIT {limit}"

        rows = await conn.fetch(query)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest crawled CDEP pages into Cognee")
    parser.add_argument("--priority", choices=["high", "medium", "all"], default="high",
                        help="Priority tier (high=~270, medium=~570, all=~1770)")
    parser.add_argument("--page-types", type=str, default=None,
                        help="Override: comma-separated page types")
    parser.add_argument("--depth", type=int, default=None,
                        help="BFS depth from homepage (0+1=all types, depth N=high-priority only)")
    parser.add_argument("--limit", type=int, default=None, help="Max pages")
    parser.add_argument("--min-length", type=int, default=200, help="Min markdown length")
    parser.add_argument("--dataset", type=str, default=None, help="Cognee dataset name (default: from COGNEE_DATASET env)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview chunks without sending to Cognee")
    args = parser.parse_args()

    s = get_settings()
    dataset = args.dataset or s.cognee_dataset
    start = time.time()

    # Suppress Cognee's extremely verbose debug logging (aiosqlite, neo4j, instructor)
    for noisy in ("aiosqlite", "neo4j", "neo4j.io", "neo4j.pool", "instructor", "cognee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.dry_run:
        setup_cost_tracking(pg_dsn=s.postgres_dsn_async.get_secret_value())

        store = CogneeStore(dataset=dataset)
        store.configure(
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

    # --- Page selection ---
    page_ids: set[int] | None = None
    page_types: list[str] | None = None

    if args.depth is not None:
        # BFS mode: depth 0+1 = all types, deeper levels = high-priority only
        print(f"Computing BFS depths from homepage (max_depth={args.depth})...")
        depths = await compute_bfs_depths(max_depth=args.depth)

        # Depth 0+1: all page IDs (no type filter)
        near_ids = {pid for pid, d in depths.items() if d <= 1}
        # Depth 2+: only high-priority types — we'll filter after fetch
        deep_ids = {pid for pid, d in depths.items() if d >= 2}

        page_ids = near_ids | deep_ids  # fetch all, filter types below
        print(f"  Depth 0+1: {len(near_ids)} pages (all types)")
        print(f"  Depth 2+: {len(deep_ids)} pages (will filter to high-priority)")
        print(f"  Total candidates: {len(page_ids)}")
    else:
        # Priority/page-types mode
        if args.page_types:
            page_types = args.page_types.split(",")
        else:
            page_types = PRIORITY_MAP[args.priority]

    # Fetch pages
    mode = f"depth={args.depth}" if args.depth is not None else f"priority={args.priority}"
    print(f"Fetching pages ({mode})...")
    pages = await fetch_pages(page_types, page_ids, args.min_length, args.limit)

    # Post-fetch filter for BFS mode: depth 2+ only keeps high-priority types
    if args.depth is not None:
        before = len(pages)
        pages = [
            p for p in pages
            if depths.get(p["id"], 99) <= 1  # depth 0+1: keep all
            or p["page_type"] in PRIORITY_HIGH  # depth 2+: high-priority only
        ]
        filtered = before - len(pages)
        if filtered:
            print(f"  Filtered {filtered} non-high-priority pages from depth 2+")

    print(f"  Found {len(pages)} pages")

    if not pages:
        print("Nothing to ingest.")
        return

    # Distribution
    type_counts: dict[str, int] = {}
    for p in pages:
        type_counts[p["page_type"]] = type_counts.get(p["page_type"], 0) + 1
    for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")

    # Ingest
    ok = 0
    err = 0
    total_chunks = 0
    empty = 0

    for i, page in enumerate(pages):
        n = i + 1
        url = page["normalized_url"][:65]
        raw_len = len(page["markdown_text"])
        clean_len = len(strip_boilerplate(page["markdown_text"], page.get("id")))

        chunks = chunk_page(page)
        total_chunks += len(chunks)

        if not chunks:
            empty += 1
            if args.dry_run:
                print(f"[{n}/{len(pages)}] {page['page_type']:20s}   0 chunks (empty after cleanup) {url}")
            continue

        print(
            f"[{n}/{len(pages)}] {page['page_type']:20s} "
            f"{len(chunks):3d} chunks "
            f"({raw_len:6d} → {clean_len:5d} chars) "
            f"{url}"
        )

        if args.dry_run:
            for j, chunk in enumerate(chunks):
                # Strip header for display
                content = chunk.split("---\n\n", 1)[-1] if "---\n\n" in chunk else chunk
                print(f"    [{j}] {len(chunk)}c: {content[:100]}...")
            continue

        try:
            await store.add_chunks(chunks=chunks, dataset=dataset)
            ok += 1
        except Exception as e:
            err += 1
            print(f"  ERROR: {e!s:.200}")

        if n % 25 == 0:
            elapsed = time.time() - start
            print(f"  --- {ok} ok, {err} errors, {total_chunks} chunks, {elapsed:.0f}s ---")

    elapsed = time.time() - start
    if args.dry_run:
        print(f"\nDry run: {len(pages)} pages → {total_chunks} chunks, {empty} empty, {elapsed:.1f}s")
    else:
        print(f"\nDone! {ok} pages ({total_chunks} chunks) ingested, {err} errors, {empty} empty, {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

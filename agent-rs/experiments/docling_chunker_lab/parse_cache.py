"""Docling parse cache.

Parses each PDF once with `DocumentConverter` + accurate table structure,
caches the resulting `DoclingDocument` as a pickle alongside version
metadata (docling version, pipeline-option digest, source SHA-256).

Subsequent sweep iterations can skip the expensive parse step and reuse
the cached doc directly.

Usage:

    cache = ParseCache(cache_dir=Path("results/run_XXX/parse_cache"))
    parsed = cache.parse_or_load("D1", pdf_path=Path("fixtures/D1_test.pdf"))
    # parsed.doc is a DoclingDocument
    # parsed.markdown is the exported markdown (cached too)
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("docling_lab.parse_cache")


# ----------------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: Path
    source_sha256: str
    doc: Any  # DoclingDocument
    markdown: str
    parse_latency_ms: int
    docling_version: str
    docling_core_version: str


# ----------------------------------------------------------------------------
# Parse cache
# ----------------------------------------------------------------------------


class ParseCache:
    """On-disk pickle cache keyed by `(doc_id, source_sha, pipeline_digest)`."""

    CACHE_VERSION = "1"

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def parse_or_load(self, doc_id: str, pdf_path: Path) -> ParsedDocument:
        source_sha = _sha256_file(pdf_path)
        pipeline_digest = _pipeline_options_digest()
        key = f"{doc_id}__{source_sha[:12]}__{pipeline_digest[:8]}"
        pickle_path = self._cache_dir / f"{key}.pickle"
        meta_path = self._cache_dir / f"{key}.meta.json"

        if pickle_path.exists() and meta_path.exists():
            try:
                with pickle_path.open("rb") as fh:
                    parsed: ParsedDocument = pickle.load(fh)
                LOG.info("cache hit: %s (%d markdown chars)", doc_id, len(parsed.markdown))
                return parsed
            except Exception as exc:  # pickle corruption → reparse
                LOG.warning("cache load failed for %s (%s); reparsing", doc_id, exc)

        parsed = _parse_fresh(doc_id, pdf_path, source_sha)
        with pickle_path.open("wb") as fh:
            pickle.dump(parsed, fh, protocol=pickle.HIGHEST_PROTOCOL)
        meta_path.write_text(
            json.dumps(
                {
                    "cache_version": self.CACHE_VERSION,
                    "doc_id": doc_id,
                    "source_path": str(pdf_path),
                    "source_sha256": source_sha,
                    "pipeline_digest": pipeline_digest,
                    "docling_version": parsed.docling_version,
                    "docling_core_version": parsed.docling_core_version,
                    "parse_latency_ms": parsed.parse_latency_ms,
                    "markdown_len": len(parsed.markdown),
                },
                indent=2,
            )
        )
        return parsed


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _parse_fresh(doc_id: str, pdf_path: Path, source_sha: str) -> ParsedDocument:
    import importlib.metadata as _meta

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        force_backend_text=True,
    )
    pipeline.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)},
    )

    LOG.info("parsing %s (%s)", doc_id, pdf_path.name)
    t0 = time.monotonic()
    result = converter.convert(str(pdf_path))
    parse_latency_ms = int((time.monotonic() - t0) * 1000)
    doc = result.document
    markdown = doc.export_to_markdown()
    LOG.info(
        "parsed %s in %d ms (markdown=%d chars)", doc_id, parse_latency_ms, len(markdown)
    )

    try:
        docling_version = _meta.version("docling")
    except _meta.PackageNotFoundError:
        docling_version = "unknown"
    try:
        docling_core_version = _meta.version("docling-core")
    except _meta.PackageNotFoundError:
        docling_core_version = "unknown"

    return ParsedDocument(
        doc_id=doc_id,
        source_path=pdf_path,
        source_sha256=source_sha,
        doc=doc,
        markdown=markdown,
        parse_latency_ms=parse_latency_ms,
        docling_version=docling_version,
        docling_core_version=docling_core_version,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pipeline_options_digest() -> str:
    """Stable digest of the pipeline option set we use for parsing.

    Must change whenever we change converter options so stale caches are
    invalidated automatically.
    """
    payload = json.dumps(
        {
            "do_ocr": False,
            "do_table_structure": True,
            "force_backend_text": True,
            "table_mode": "ACCURATE",
            "format": "PDF",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()

"""Apply a chunker configuration to a cached DoclingDocument.

Produces a list of `Chunk` records with three text views:

- `raw_text`            -- the chunker's native `chunk.text`
- `contextualized_text` -- chunker.contextualize(chunk) (HybridChunker/
                           HierarchicalChunker only; legacy returns raw)
- `embed_text`          -- the exact string fed to the embedding model.
                           For Docling chunkers we use the contextualized
                           string so headings/captions flow into retrieval.
                           For the legacy chunker we use raw.

The runner also serialises per-chunk structural info that intrinsic_metrics
and retrieval_eval consume later.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from configs import CONFIG_BY_ID, build_chunker
from parse_cache import ParsedDocument

LOG = logging.getLogger("docling_lab.chunker_runner")


# ----------------------------------------------------------------------------
# Output shape
# ----------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    cfg_id: str
    index: int
    raw_text: str
    embed_text: str
    contextualized_text: str
    headings: list[str] = field(default_factory=list)
    doc_item_refs: list[str] = field(default_factory=list)
    n_doc_items: int = 0


@dataclass
class ChunkingResult:
    doc_id: str
    cfg_id: str
    chunks: list[Chunk]
    chunker_latency_ms: int


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def run_chunker(parsed: ParsedDocument, cfg_id: str) -> ChunkingResult:
    """Apply the config's chunker to a cached `ParsedDocument`."""
    cfg = CONFIG_BY_ID[cfg_id]
    LOG.info(
        "chunking doc=%s cfg=%s (family=%s, max_tokens=%s)",
        parsed.doc_id,
        cfg.cfg_id,
        cfg.family,
        cfg.max_tokens,
    )
    chunker = build_chunker(cfg)

    t0 = time.monotonic()
    raw_chunks = list(chunker.chunk(parsed.doc))
    dur_ms = int((time.monotonic() - t0) * 1000)

    out: list[Chunk] = []
    for idx, raw in enumerate(raw_chunks):
        raw_text = getattr(raw, "text", str(raw))

        # Contextualize produces the heading-prefixed text for Docling
        # chunkers.  For the legacy chunker we fall back to raw_text.
        if hasattr(chunker, "contextualize"):
            try:
                contextualized = chunker.contextualize(raw)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 -- contextualize is best-effort
                LOG.debug("contextualize failed (%s); using raw_text", exc)
                contextualized = raw_text
        else:
            contextualized = raw_text

        # Embed the contextualized string for Docling chunkers so heading
        # context flows into the vector space.  Legacy has no heading
        # context, so raw == contextualized.
        embed_text = contextualized if cfg.family != "legacy_nltk" else raw_text

        meta = getattr(raw, "meta", None)
        headings = _extract_headings(meta)
        doc_item_refs = _extract_doc_item_refs(meta)

        chunk_id = f"{parsed.doc_id}__{cfg.cfg_id}__{idx:05d}"
        out.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=parsed.doc_id,
                cfg_id=cfg.cfg_id,
                index=idx,
                raw_text=raw_text,
                embed_text=embed_text,
                contextualized_text=contextualized,
                headings=headings,
                doc_item_refs=doc_item_refs,
                n_doc_items=len(doc_item_refs),
            )
        )

    LOG.info(
        "chunker done: doc=%s cfg=%s -> %d chunks in %d ms",
        parsed.doc_id,
        cfg.cfg_id,
        len(out),
        dur_ms,
    )
    return ChunkingResult(
        doc_id=parsed.doc_id,
        cfg_id=cfg.cfg_id,
        chunks=out,
        chunker_latency_ms=dur_ms,
    )


# ----------------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------------


def dump_chunks(result: ChunkingResult, out_path: Path) -> None:
    """Write chunks as JSONL — one chunk per line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in result.chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    LOG.debug("wrote %s", out_path)


# ----------------------------------------------------------------------------
# Meta extraction helpers
# ----------------------------------------------------------------------------


def _extract_headings(meta: Any) -> list[str]:
    if meta is None:
        return []
    headings = getattr(meta, "headings", None)
    if headings is None:
        return []
    # HybridChunker yields plain strings; keep the first few to avoid
    # blowing up the payload.
    return [str(h) for h in headings][:8]


def _extract_doc_item_refs(meta: Any) -> list[str]:
    """Return stable references for the DocItems backing this chunk.

    Prefers `self_ref` / `cref` when present.  Falls back to `str(item)`
    otherwise.  Empty for chunkers with no structural provenance.
    """
    if meta is None:
        return []
    doc_items = getattr(meta, "doc_items", None)
    if not doc_items:
        return []
    refs: list[str] = []
    for item in doc_items:
        ref = (
            getattr(item, "self_ref", None)
            or getattr(item, "cref", None)
            or getattr(item, "ref", None)
        )
        if ref is None:
            ref = str(item)
        refs.append(str(ref))
    return refs

"""Marker-based document converter + token-budget chunker.

Produces the same ``ConversionResult`` shape that the Docling sidecar's
``ChunkerPool`` produces, so ``server.py`` can emit identical gRPC
Chunk messages regardless of which backend parsed the PDF.

Implementation notes:

- Uses Marker's ``ChunkRenderer`` which returns a flat list of
  ``FlatBlockOutput`` objects, each tagged with its page number. We avoid
  re-walking the raw JSON tree and trust Marker's reading-order flatten.
- Heading path is derived from ``FlatBlockOutput.section_hierarchy`` — a
  dict mapping heading level → SectionHeader block ID. We build a
  ``section_id → text`` lookup on the fly.
- Token counting uses ``cl100k_base`` (matching Docling) so chunk
  boundaries and embedding budgets are backend-invariant.
- Formula / table / picture / code detection is explicit via block type
  and a ``detect_formula`` fallback scan for inline ``$…$`` LaTeX inside
  Text blocks.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
from bs4 import BeautifulSoup

from .errors import MarkerParseError, UnsupportedMimetypeError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses — same field set as docling-sidecar's ConversionResult
# / ChunkResult. Keeping the shape identical means ``server.py`` does not
# need to branch by backend when it emits proto messages.
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    raw_text: str
    embed_text: str
    heading_path: list[str]
    page_start: int
    page_end: int
    token_count: int
    char_count: int
    doc_items: list[dict[str, Any]]
    has_table: bool
    has_picture: bool
    has_code: bool
    has_formula: bool
    section_title: str


@dataclass
class ConversionResult:
    marker_doc: Any
    chunks: list[ChunkResult]
    n_pages: int
    language: str
    title: str
    mimetype: str


# ---------------------------------------------------------------------------
# Mimetype sniffing — mirrors docling-sidecar to keep the accepted input set
# aligned. Marker itself accepts more, but we gate here so the contract is
# identical across backends.
# ---------------------------------------------------------------------------

_SUPPORTED_MIMETYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/html",
        "text/markdown",
        "text/plain",
        "image/png",
        "image/jpeg",
        "image/tiff",
    }
)

_MAGIC_PDF = b"%PDF"


def _sniff_mimetype(content: bytes, hint: str) -> str:
    if hint:
        return hint.lower().strip()
    if content[:4] == _MAGIC_PDF:
        return "application/pdf"
    return "application/octet-stream"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    logger.warning("env %s=%r not parseable as bool; default=%s", name, raw, default)
    return default


# ---------------------------------------------------------------------------
# Formula detection — cross-lingual, structural-glyph based. Mirrors
# `crates/retrieval/src/ingestion/metadata.rs::detect_formula` so both
# backends produce consistent `has_formula` flags.
# ---------------------------------------------------------------------------

_LATEX_MARKERS = (
    r"\begin{equation",
    r"\begin{align",
    r"\begin{gather",
    r"\[",
    r"\(",
    "<!-- formula-not-decoded -->",
)
_MATH_GLYPHS = set("=+−×÷≤≥≠≈∑∫∏∞√πθλμσω≡⇒∀∃∈∉⊂⊃∪∩")
_INLINE_MATH = re.compile(r"\$[^\$]{1,200}\$")


def _detect_formula_in_text(text: str) -> bool:
    if not text:
        return False
    for m in _LATEX_MARKERS:
        if m in text:
            return True
    if _INLINE_MATH.search(text):
        return True
    hits = sum(1 for c in text if c in _MATH_GLYPHS)
    if hits < 2:
        return False
    # Require density: two glyphs inside any 400-char window.
    if len(text) <= 400:
        return True
    for start in range(0, len(text) - 400, 200):
        local = sum(1 for c in text[start : start + 400] if c in _MATH_GLYPHS)
        if local >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Block-type → docling label mapping (Marker labels are stringified
# ``BlockTypes`` enum values; see marker.schema.BlockTypes).
# ---------------------------------------------------------------------------

_MARKER_TO_DOCLING_LABEL = {
    "SectionHeader": "section_header",
    "PageHeader": "page_header",
    "PageFooter": "page_footer",
    "Title": "title",
    "Text": "text",
    "TextInlineMath": "text",
    "ListItem": "list_item",
    "ListGroup": "list_item",
    "Caption": "caption",
    "Footnote": "footnote",
    "Equation": "formula",
    "Code": "code",
    "Table": "table",
    "TableOfContents": "document_index",
    "Picture": "picture",
    "PictureGroup": "picture",
    "Figure": "picture",
    "FigureGroup": "picture",
    "Form": "form",
    "Handwriting": "text",
    "Reference": "reference",
}


# ---------------------------------------------------------------------------
# Heading-level derivation. Marker's ``section_hierarchy`` on every block
# maps heading level (int) → SectionHeader block ID. We build a
# header-id → text dict as we encounter SectionHeader blocks in reading
# order, then look up each block's active heading path from its
# ``section_hierarchy``.
# ---------------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    """Strip HTML, return plain text. Preserves inline math delimiters."""
    if not html:
        return ""
    # BeautifulSoup handles nested tags, entity decoding, etc.
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _heading_path_from_hierarchy(
    section_hierarchy: dict[int, str] | None,
    header_text_by_id: dict[str, str],
) -> list[str]:
    """Resolve a block's active heading path via its section_hierarchy map."""
    if not section_hierarchy:
        return []
    # Sort by level ascending (1, 2, 3...) so output reflects nesting.
    levels = sorted(int(k) for k in section_hierarchy.keys())
    path: list[str] = []
    for lvl in levels:
        sid = section_hierarchy.get(lvl) or section_hierarchy.get(str(lvl))
        if not sid:
            continue
        text = header_text_by_id.get(str(sid))
        if text:
            path.append(text)
    return path


def _polygon_to_bbox(polygon: list[list[float]] | None) -> dict[str, float] | None:
    """Reduce a 4-corner polygon to an axis-aligned bbox."""
    if not polygon:
        return None
    try:
        xs = [float(pt[0]) for pt in polygon]
        ys = [float(pt[1]) for pt in polygon]
    except (IndexError, TypeError, ValueError):
        return None
    return {"l": min(xs), "t": min(ys), "r": max(xs), "b": max(ys)}


def _bbox_fallback(bbox: list[float] | None) -> dict[str, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    try:
        return {
            "l": float(bbox[0]),
            "t": float(bbox[1]),
            "r": float(bbox[2]),
            "b": float(bbox[3]),
        }
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Chunker pool — loads Marker models once and exposes the same
# ``convert_and_chunk`` method signature used by the docling sidecar.
# ---------------------------------------------------------------------------


_TOKEN_BUDGET = int(os.environ.get("MARKER_CHUNK_TOKEN_BUDGET", "1024"))
_MIN_CHUNK_TOKENS = int(os.environ.get("MARKER_CHUNK_MIN_TOKENS", "64"))


class ChunkerPool:
    """Single shared Marker converter + encoding.

    Thread-safe per call: Marker's ``PdfConverter`` is stateless after
    ``__init__`` (the model dict is immutable once loaded) and each
    ``__call__`` spawns its own short-lived state.
    """

    def __init__(self) -> None:
        logger.info("Initialising ChunkerPool — loading Marker models...")

        # Deferred imports so a `python -c "import marker_sidecar"` probe
        # doesn't trigger the full torch + marker load path.
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        use_llm = _env_bool("MARKER_USE_LLM", False)
        config = {
            "output_format": "chunks",
        }
        if use_llm:
            config["use_llm"] = True

        parser = ConfigParser(config)
        self._converter = PdfConverter(
            artifact_dict=create_model_dict(),
            processor_list=parser.get_processors(),
            renderer=parser.get_renderer(),
            llm_service=parser.get_llm_service() if use_llm else None,
            config=parser.generate_config_dict(),
        )
        self._enc = tiktoken.get_encoding("cl100k_base")
        logger.info("ChunkerPool ready (use_llm=%s).", use_llm)

    # ------------------------------------------------------------------
    # Entry point called from server.py
    # ------------------------------------------------------------------
    def convert_and_chunk(
        self,
        path: Path,
        mimetype: str,
        sha256: str,
        expected_sha256: str,
    ) -> ConversionResult:
        if mimetype and mimetype not in _SUPPORTED_MIMETYPES:
            raise UnsupportedMimetypeError(
                f"mimetype '{mimetype}' not supported by Marker sidecar"
            )
        if expected_sha256 and sha256 != expected_sha256:
            from .errors import InvalidMetadataError

            raise InvalidMetadataError(
                f"sha256 mismatch: computed {sha256[:16]}… expected {expected_sha256[:16]}…"
            )

        try:
            rendered = self._converter(str(path))
        except Exception as exc:  # noqa: BLE001
            raise MarkerParseError(f"Marker converter failed: {exc}") from exc

        # ChunkRenderer returns marker.renderers.chunk.ChunkOutput.
        blocks = list(getattr(rendered, "blocks", []) or [])
        page_info: dict[int, dict] = dict(getattr(rendered, "page_info", {}) or {})
        metadata: dict = dict(getattr(rendered, "metadata", {}) or {})

        n_pages = max(len(page_info), 0)
        if n_pages == 0 and blocks:
            n_pages = max((int(getattr(b, "page", 0) or 0) for b in blocks), default=0) + 1

        language = str(metadata.get("language") or metadata.get("lang") or "")
        title = str(metadata.get("title") or "")

        chunks = self._assemble_chunks(blocks)
        return ConversionResult(
            marker_doc=rendered,
            chunks=chunks,
            n_pages=n_pages,
            language=language,
            title=title,
            mimetype=mimetype or "application/pdf",
        )

    # ------------------------------------------------------------------
    # Internal: walk flat blocks and group into chunks
    # ------------------------------------------------------------------
    def _assemble_chunks(self, blocks: list[Any]) -> list[ChunkResult]:
        # Pre-pass: build header_id → text lookup.
        header_text_by_id: dict[str, str] = {}
        for b in blocks:
            if getattr(b, "block_type", "") == "SectionHeader":
                text = _html_to_text(getattr(b, "html", "") or "")
                if text:
                    header_text_by_id[str(b.id)] = text

        chunks: list[ChunkResult] = []
        acc = _ChunkAccumulator(self._enc)

        for b in blocks:
            bt = str(getattr(b, "block_type", "") or "")
            # SectionHeader is a boundary: flush current, record heading,
            # then start the next chunk inside the new section.
            if bt == "SectionHeader":
                acc.flush_into(chunks)
                header_text = _html_to_text(getattr(b, "html", "") or "")
                heading_path = _heading_path_from_hierarchy(
                    getattr(b, "section_hierarchy", None) or {},
                    header_text_by_id,
                )
                # The SectionHeader itself is not in its own section_hierarchy
                # until after its ID is registered — ensure we include it.
                if header_text and (not heading_path or heading_path[-1] != header_text):
                    heading_path = [*heading_path, header_text]
                acc.set_heading(heading_path)
                # Emit the header text as its own tiny chunk so retrieval
                # can surface it when users search for the section title.
                acc.record_doc_item(b, text=header_text)
                acc.append_text(header_text)
                acc.flush_into(chunks, force_small_ok=True)
                continue

            # Non-header: update heading_path if this block's hierarchy
            # introduces a deeper level the accumulator hasn't seen yet.
            sh = getattr(b, "section_hierarchy", None) or {}
            if sh:
                hp = _heading_path_from_hierarchy(sh, header_text_by_id)
                if hp and hp != acc.heading_path:
                    # Flush before changing heading path so chunks don't span
                    # section boundaries.
                    acc.flush_into(chunks)
                    acc.set_heading(hp)

            acc.record_doc_item(b)

            # Extract display text. Equation blocks typically embed LaTeX
            # inside `<math>` or `$…$`; grab the HTML text — LaTeX markers
            # survive get_text() untouched.
            text = _html_to_text(getattr(b, "html", "") or "")
            if not text:
                continue
            acc.append_text(text)
            if acc.token_count() >= _TOKEN_BUDGET:
                acc.flush_into(chunks)

        acc.flush_into(chunks)
        return chunks


# ---------------------------------------------------------------------------
# Accumulator — groups flat blocks into token-budgeted chunks and tracks
# page-range / flag aggregation.
# ---------------------------------------------------------------------------


@dataclass
class _ChunkAccumulator:
    enc: Any
    heading_path: list[str] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    doc_items: list[dict[str, Any]] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    has_table: bool = False
    has_picture: bool = False
    has_code: bool = False
    has_formula: bool = False

    # --- heading management --------------------------------------------
    def set_heading(self, heading_path: list[str]) -> None:
        self.heading_path = list(heading_path)

    # --- content collection --------------------------------------------
    def append_text(self, text: str) -> None:
        if text:
            self.text_parts.append(text)

    def record_doc_item(self, block: Any, text: str | None = None) -> None:
        bt = str(getattr(block, "block_type", "") or "")
        label = _MARKER_TO_DOCLING_LABEL.get(bt, "text")
        page_no = int(getattr(block, "page", 0) or 0) + 1  # 0-based → 1-based
        bbox = _polygon_to_bbox(getattr(block, "polygon", None)) or _bbox_fallback(
            getattr(block, "bbox", None)
        )
        block_text = text if text is not None else _html_to_text(getattr(block, "html", "") or "")

        # Flag aggregation via block type + inline math scan.
        if bt == "Table":
            self.has_table = True
        if bt in {"Picture", "Figure", "PictureGroup", "FigureGroup"}:
            self.has_picture = True
        if bt == "Code":
            self.has_code = True
        if bt == "Equation" or _detect_formula_in_text(block_text):
            self.has_formula = True

        # Page range update.
        if self.page_start == 0:
            self.page_start = page_no
        if page_no > self.page_end:
            self.page_end = page_no

        # Per-item metadata for the downstream Neo4j graph layer.
        self.doc_items.append(
            {
                "self_ref": str(getattr(block, "id", "") or ""),
                "label_str": label,
                "page_no": page_no,
                "bbox": bbox,
                "text": block_text,
                "table_markdown": getattr(block, "html", "") if bt == "Table" else "",
                "table_n_rows": 0,
                "table_n_cols": 0,
                "picture_caption": (
                    block_text if bt in {"Picture", "Figure"} else ""
                ),
            }
        )

    # --- chunk emission ------------------------------------------------
    def token_count(self) -> int:
        if not self.text_parts:
            return 0
        joined = "\n".join(self.text_parts)
        return len(self.enc.encode(joined, disallowed_special="all"))

    def flush_into(
        self,
        chunks: list[ChunkResult],
        force_small_ok: bool = False,
    ) -> None:
        """Emit the current buffer as a ChunkResult and reset.

        ``force_small_ok`` lets us emit short chunks (e.g. standalone
        section titles) that would otherwise be dropped.
        """
        raw_text = "\n".join(self.text_parts).strip()
        if not raw_text:
            self._reset_content()
            return

        tok = self.token_count()
        if tok < _MIN_CHUNK_TOKENS and not force_small_ok:
            # Too small on its own — leave the buffer intact so the next
            # block can merge with it. Caller decides when to force-flush.
            return

        heading_prefix = ""
        if self.heading_path:
            heading_prefix = " > ".join(self.heading_path) + "\n\n"
        embed_text = heading_prefix + raw_text

        char_count = len(embed_text)
        token_count = len(self.enc.encode(embed_text, disallowed_special="all"))
        section_title = self.heading_path[-1] if self.heading_path else ""

        chunks.append(
            ChunkResult(
                raw_text=raw_text,
                embed_text=embed_text,
                heading_path=list(self.heading_path),
                page_start=self.page_start,
                page_end=self.page_end or self.page_start,
                token_count=token_count,
                char_count=char_count,
                doc_items=list(self.doc_items),
                has_table=self.has_table,
                has_picture=self.has_picture,
                has_code=self.has_code,
                has_formula=self.has_formula,
                section_title=section_title,
            )
        )
        self._reset_content()

    def _reset_content(self) -> None:
        self.text_parts.clear()
        self.doc_items.clear()
        self.page_start = 0
        self.page_end = 0
        self.has_table = False
        self.has_picture = False
        self.has_code = False
        self.has_formula = False

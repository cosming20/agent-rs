"""Docling DocumentConverter + HybridChunker wrapper.

Wraps the Docling pipeline behind a simple synchronous interface. The gRPC
server calls these methods in a thread pool via asyncio.to_thread() since
Docling's converter is CPU-bound and synchronous.

All Docling objects are constructed once at server startup (ChunkerPool.__init__)
so that model weights are loaded into memory before the first request arrives.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from .errors import (
    ChunkingError,
    DoclingParseError,
    InvalidMetadataError,
    UnsupportedMimetypeError,
)

logger = logging.getLogger(__name__)

# Supported MIME types that Docling can handle.
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

# Magic bytes for mimetype sniffing when mimetype field is empty.
_MAGIC_PDF = b"%PDF"


def _resolve_formula_backend() -> str:
    """Which decoder to use for formulas.

    Values (case-insensitive, stripped):
      - ``pix2tex`` — local 80M-param specialist (~1-2 s/formula CPU).
        **Default** — optimised for laptops / CPU-only deploys.
      - ``codeformula`` — Docling's native 1.8B Idefics3 VLM
        (~60 s/formula CPU, 2-5 s/formula CUDA). Pick this when
        running the sidecar on a CUDA host.
      - ``off`` — no decoding; formula items keep the placeholder text
        `<!-- formula-not-decoded -->`. Fastest ingestion; formulas
        are not searchable.
    """
    raw = os.environ.get("DOCLING_FORMULA_BACKEND", "pix2tex").strip().lower()
    if raw in {"pix2tex", "codeformula", "off"}:
        return raw
    logger.warning(
        "DOCLING_FORMULA_BACKEND=%r not recognised; falling back to pix2tex", raw
    )
    return "pix2tex"


_FORMULA_BACKEND: str = _resolve_formula_backend()


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean-shaped env var with an explicit default.

    Accepts (case-insensitive): `1`/`true`/`yes`/`on` → True,
    `0`/`false`/`no`/`off` → False. Any other value falls back to
    `default` with a warning — silently accepting typos like
    `FORMULA=ture` would defeat the point of the toggle.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "env %s=%r not parseable as bool; falling back to default=%s",
        name,
        raw,
        default,
    )
    return default


def _sniff_mimetype(content: bytes, hint: str) -> str:
    """Return the effective mimetype from hint or magic bytes."""
    if hint:
        return hint.lower().strip()
    if content[:4] == _MAGIC_PDF:
        return "application/pdf"
    # Default fallback
    return "application/octet-stream"


@dataclass
class ChunkResult:
    """A single chunk produced by HybridChunker, with all fields needed for proto."""

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
    """Output of ChunkerPool.convert_and_chunk()."""

    docling_doc: Any  # docling_core.types.doc.DoclingDocument
    chunks: list[ChunkResult]
    n_pages: int
    language: str
    title: str
    mimetype: str


class ChunkerPool:
    """Single shared DocumentConverter + HybridChunker, configured per spec.

    Thread-safe: Docling's DocumentConverter is stateless across convert() calls.
    Construction pre-loads all ONNX model weights so the first request is fast.
    """

    def __init__(self) -> None:
        logger.info("Initialising ChunkerPool — loading Docling models...")
        self._converter = self._build_converter()
        self._chunker = self._build_chunker()
        self._enc = tiktoken.encoding_for_model("text-embedding-3-large")
        # Optional secondary formula decoder (pix2tex). Constructed lazy —
        # the model only loads on first formula encountered, so docs
        # without math pay zero cost. `None` when backend != "pix2tex".
        self._pix2tex_enricher: Any | None = None
        if _FORMULA_BACKEND == "pix2tex":
            from .formula_enrichment import Pix2TexEnricher

            self._pix2tex_enricher = Pix2TexEnricher()
            logger.info("Formula backend: pix2tex (model loads on first use)")
        elif _FORMULA_BACKEND == "codeformula":
            logger.info("Formula backend: codeformula (Docling's 1.8B VLM)")
        else:
            logger.info("Formula backend: off (formulas stay as placeholders)")
        logger.info("ChunkerPool ready.")

    # ------------------------------------------------------------------
    # Public interface (called synchronously, wrapped in to_thread by server)
    # ------------------------------------------------------------------

    def convert_and_chunk(
        self,
        path: Path,
        mimetype: str,
        sha256: str,
        expected_sha256: str,
    ) -> ConversionResult:
        """Convert a document and chunk it.

        Args:
            path: Filesystem path to the temporary document file.
            mimetype: Effective mimetype (already sniffed if not supplied).
            sha256: SHA-256 of the file bytes, computed by the caller.
            expected_sha256: SHA-256 from the request. Must match sha256.

        Raises:
            InvalidMetadataError: sha256 mismatch.
            UnsupportedMimetypeError: Docling cannot handle this mimetype.
            DoclingParseError: Docling raised during conversion.
            ChunkingError: HybridChunker raised.
        """
        if expected_sha256 and sha256 != expected_sha256:
            raise InvalidMetadataError(
                f"sha256 mismatch: got {sha256}, expected {expected_sha256}"
            )

        if mimetype not in _SUPPORTED_MIMETYPES:
            raise UnsupportedMimetypeError(
                f"Unsupported mimetype: {mimetype!r}. "
                f"Supported: {sorted(_SUPPORTED_MIMETYPES)}"
            )

        try:
            result = self._converter.convert(str(path))
        except Exception as exc:
            raise DoclingParseError(f"DocumentConverter failed: {exc}") from exc

        doc = result.document

        # --- page metadata ---
        n_pages = 0
        if hasattr(doc, "pages") and doc.pages:
            n_pages = len(doc.pages)

        # --- language ---
        language = ""
        if hasattr(doc, "detected_language") and doc.detected_language:
            language = str(doc.detected_language)

        # --- title ---
        title = ""
        if hasattr(doc, "name") and doc.name:
            title = str(doc.name)
        # Try to extract from metadata if present
        if not title and hasattr(doc, "metadata") and doc.metadata:
            meta = doc.metadata
            if hasattr(meta, "title") and meta.title:
                title = str(meta.title)

        # --- pix2tex formula enrichment (backend-gated) ---
        # Runs before chunking so HybridChunker picks up the decoded
        # LaTeX text on each FormulaItem. No-op when backend != pix2tex
        # or when the document contains zero formulas.
        if self._pix2tex_enricher is not None:
            try:
                self._pix2tex_enricher.enrich_document(doc)
            except Exception as exc:  # noqa: BLE001 — never fail ingest on enrichment
                logger.warning(
                    "pix2tex enrichment raised; formulas stay as placeholders: %s",
                    exc,
                )

        # --- chunking ---
        try:
            raw_chunks = list(self._chunker.chunk(doc))
        except Exception as exc:
            raise ChunkingError(f"HybridChunker failed: {exc}") from exc

        chunks = [self._map_chunk(doc, chunk) for chunk in raw_chunks]

        return ConversionResult(
            docling_doc=doc,
            chunks=chunks,
            n_pages=n_pages,
            language=language,
            title=title,
            mimetype=mimetype,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_chunk(self, doc: Any, chunk: Any) -> ChunkResult:
        """Map a docling HybridChunker chunk to our ChunkResult."""
        raw_text: str = chunk.text if hasattr(chunk, "text") else ""

        # contextualize() prefixes the chunk with its heading hierarchy for
        # better embedding quality.
        try:
            embed_text: str = self._chunker.contextualize(chunk)
        except Exception as ctx_exc:
            # W7 fix #9: log when contextualize() fails — silent fallback to
            # raw_text means embed_text loses heading hierarchy context, which
            # degrades retrieval quality silently.
            logger.warning(
                "contextualize() failed, using raw_text: %s", ctx_exc
            )
            embed_text = raw_text

        # Heading path
        heading_path: list[str] = []
        if hasattr(chunk, "meta") and chunk.meta:
            meta = chunk.meta
            if hasattr(meta, "headings") and meta.headings:
                heading_path = list(meta.headings)

        # Doc items
        doc_items_raw: list[Any] = []
        if hasattr(chunk, "meta") and chunk.meta:
            meta = chunk.meta
            if hasattr(meta, "doc_items") and meta.doc_items:
                doc_items_raw = list(meta.doc_items)

        # Page range + flags
        page_start = 0
        page_end = 0
        has_table = False
        has_picture = False
        has_code = False
        has_formula = False
        doc_items: list[dict[str, Any]] = []

        for item in doc_items_raw:
            label_str = ""
            if hasattr(item, "label"):
                label_val = item.label
                label_str = str(label_val.value) if hasattr(label_val, "value") else str(label_val)

            # Page provenance
            item_page = 0
            item_bbox: dict[str, float] | None = None
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no") and prov.page_no:
                        p = int(prov.page_no)
                        if item_page == 0:
                            item_page = p
                        if page_start == 0 or p < page_start:
                            page_start = p
                        if p > page_end:
                            page_end = p
                    if hasattr(prov, "bbox") and prov.bbox and item_bbox is None:
                        b = prov.bbox
                        item_bbox = {
                            "l": float(getattr(b, "l", 0)),
                            "t": float(getattr(b, "t", 0)),
                            "r": float(getattr(b, "r", 0)),
                            "b": float(getattr(b, "b", 0)),
                        }

            # Flags
            label_upper = label_str.upper()
            if "TABLE" in label_upper:
                has_table = True
            if "PICTURE" in label_upper:
                has_picture = True
            if "CODE" in label_upper:
                has_code = True
            if "FORMULA" in label_upper:
                has_formula = True

            # Table markdown
            table_markdown = ""
            table_n_rows = 0
            table_n_cols = 0
            if "TABLE" in label_upper and hasattr(item, "data") and item.data:
                try:
                    table_markdown = item.export_to_markdown() if hasattr(item, "export_to_markdown") else ""
                except Exception:
                    table_markdown = ""
                if hasattr(item.data, "num_rows"):
                    table_n_rows = int(item.data.num_rows)
                if hasattr(item.data, "num_cols"):
                    table_n_cols = int(item.data.num_cols)

            # Picture caption
            picture_caption = ""
            if "PICTURE" in label_upper:
                if hasattr(item, "captions") and item.captions:
                    picture_caption = " ".join(
                        str(c.text) if hasattr(c, "text") else str(c)
                        for c in item.captions
                    )

            doc_items.append(
                {
                    "self_ref": str(item.self_ref) if hasattr(item, "self_ref") else "",
                    "label_str": label_str,
                    "page_no": item_page,
                    "bbox": item_bbox,
                    "text": str(item.text) if hasattr(item, "text") else "",
                    "table_markdown": table_markdown,
                    "table_n_rows": table_n_rows,
                    "table_n_cols": table_n_cols,
                    "picture_caption": picture_caption,
                }
            )

        token_count = len(
            self._enc.encode(embed_text, disallowed_special="all")
        )
        char_count = len(embed_text)
        section_title = heading_path[-1] if heading_path else ""

        return ChunkResult(
            raw_text=raw_text,
            embed_text=embed_text,
            heading_path=heading_path,
            page_start=page_start,
            page_end=page_end,
            token_count=token_count,
            char_count=char_count,
            doc_items=doc_items,
            has_table=has_table,
            has_picture=has_picture,
            has_code=has_code,
            has_formula=has_formula,
            section_title=section_title,
        )

    @staticmethod
    def _resolve_accelerator():
        """Pick the best accelerator device for Docling's layout + table models.

        Precedence:
        1. Explicit DOCLING_ACCELERATOR env var: cpu | cuda | mps | auto.
        2. Auto-detect via torch: cuda → mps → cpu.

        Note on Docker-on-macOS: Apple's Metal Performance Shaders (MPS) is a
        macOS-only API. Docker Desktop on macOS runs a Linux VM that cannot
        see Metal, so MPS will NOT be available here regardless of host
        hardware — torch inside the container reports mps.is_available() ==
        False. To actually use MPS, run this sidecar natively on the host
        (outside Docker) with a Python venv. CUDA passthrough works on Linux
        hosts with nvidia-container-toolkit configured in docker-compose.
        """
        from docling.datamodel.pipeline_options import AcceleratorDevice

        override = os.environ.get("DOCLING_ACCELERATOR", "auto").strip().lower()
        explicit = {
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
            "auto": None,
        }
        if override in explicit:
            if explicit[override] is not None:
                logger.info("Accelerator: %s (explicit DOCLING_ACCELERATOR)", override)
                return explicit[override]
        else:
            logger.warning(
                "Unknown DOCLING_ACCELERATOR=%r; falling back to auto-detect", override
            )

        # Auto-detect via torch probes.
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("Accelerator: CUDA (auto-detected)")
                return AcceleratorDevice.CUDA
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Accelerator: MPS (auto-detected)")
                return AcceleratorDevice.MPS
        except Exception as exc:
            logger.warning(
                "torch not available for accelerator auto-detect (%s); using CPU",
                exc,
            )
        logger.info("Accelerator: CPU (no GPU available)")
        return AcceleratorDevice.CPU

    @staticmethod
    def _build_converter() -> DocumentConverter:
        """Build a DocumentConverter for production PDF ingestion.

        All enrichment stages default **on** so a freshly deployed
        sidecar captures formulas, code blocks, and scanned text out of
        the box — the defect this pipeline fought before was silently
        dropping formulas to `<!-- formula-not-decoded -->` placeholders
        because `do_formula_enrichment` was unset. Operators on memory-
        constrained hosts can opt out per stage via env vars below.

        Env-driven toggles (all accept `1`/`true` / `0`/`false`, case-
        insensitive; unset uses the documented default):

        - `DOCLING_TABLEFORMER_MODE` — `accurate` (default) or `fast`.
        - `DOCLING_DO_OCR`                   — default `true`.
        - `DOCLING_DO_FORMULA_ENRICHMENT`    — default `true`.
        - `DOCLING_DO_CODE_ENRICHMENT`       — default `true`.
        - `DOCLING_DO_PICTURE_CLASSIFICATION`— default `true`.
        - `DOCLING_DO_PICTURE_DESCRIPTION`   — default `false`
          (requires a VLM; enabling without one will fail at convert
          time).
        - `DOCLING_DO_CHART_EXTRACTION`      — default `false`. Added
          in Docling 2.91 — extracts bar / pie / line chart data as a
          table so retrieval can match on chart values. Default off
          because it depends on `ibm-granite/granite-4.0-3b-vision`
          (~8 GB download + LoRA adapters, needs `peft`), and Docling
          2.91's chart engine declares `supported_devices=[CPU, CUDA]`
          only — Apple Silicon falls back to CPU and the transformers
          sync load is prone to deadlocking inside the asyncio gRPC
          handler on macOS. Enable in Linux/CUDA deployments where
          chart data matters for retrieval.
        - `DOCLING_GENERATE_PICTURE_IMAGES`  — default `true`.
        - `DOCLING_IMAGES_SCALE`             — float, default `2.0`.
        - `DOCLING_ACCELERATOR`              — cpu | cuda | mps | auto.
        """
        from docling.datamodel.pipeline_options import (
            AcceleratorOptions,
            TableStructureOptions,
        )

        mode_env = os.environ.get("DOCLING_TABLEFORMER_MODE", "accurate").lower()
        mode = (
            TableFormerMode.ACCURATE
            if mode_env == "accurate"
            else TableFormerMode.FAST
        )
        logger.info("TableFormer mode: %s", mode.name)

        accelerator_device = ChunkerPool._resolve_accelerator()
        accelerator_options = AcceleratorOptions(device=accelerator_device)

        do_ocr = _env_bool("DOCLING_DO_OCR", True)
        # Backend-aware formula flag: let Docling's internal VLM run only
        # when explicitly selected. For `pix2tex` and `off` we skip the
        # slow path — pix2tex runs as a post-convert enrichment pass
        # instead (see formula_enrichment.py).
        if _FORMULA_BACKEND == "codeformula":
            do_formula = _env_bool("DOCLING_DO_FORMULA_ENRICHMENT", True)
        else:
            do_formula = False
        do_code = _env_bool("DOCLING_DO_CODE_ENRICHMENT", True)
        do_pic_class = _env_bool("DOCLING_DO_PICTURE_CLASSIFICATION", True)
        do_pic_desc = _env_bool("DOCLING_DO_PICTURE_DESCRIPTION", False)
        do_chart = _env_bool("DOCLING_DO_CHART_EXTRACTION", False)
        gen_pic_images = _env_bool("DOCLING_GENERATE_PICTURE_IMAGES", True)
        # pix2tex needs page-level raster images to crop formula bboxes.
        # Force page-image generation when pix2tex is active, regardless
        # of the individual env var the user might set.
        gen_page_images = (
            _FORMULA_BACKEND == "pix2tex"
            or _env_bool("DOCLING_GENERATE_PAGE_IMAGES", False)
        )
        try:
            images_scale = float(os.environ.get("DOCLING_IMAGES_SCALE", "2.0"))
        except ValueError:
            logger.warning(
                "DOCLING_IMAGES_SCALE not parseable as float; falling back to 2.0"
            )
            images_scale = 2.0

        logger.info(
            "Docling enrichment: ocr=%s formula_backend=%s (docling_formula=%s) "
            "code=%s picture_classification=%s picture_description=%s "
            "chart_extraction=%s generate_picture_images=%s "
            "generate_page_images=%s images_scale=%.2f",
            do_ocr,
            _FORMULA_BACKEND,
            do_formula,
            do_code,
            do_pic_class,
            do_pic_desc,
            do_chart,
            gen_pic_images,
            gen_page_images,
            images_scale,
        )

        pdf_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=True,
            do_formula_enrichment=do_formula,
            do_code_enrichment=do_code,
            do_picture_classification=do_pic_class,
            do_picture_description=do_pic_desc,
            do_chart_extraction=do_chart,
            generate_picture_images=gen_pic_images,
            generate_page_images=gen_page_images,
            images_scale=images_scale,
            table_structure_options=TableStructureOptions(mode=mode),
            accelerator_options=accelerator_options,
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            }
        )

    @staticmethod
    def _build_chunker() -> HybridChunker:
        """Build a HybridChunker with cl100k tokenizer, 1024-token chunks.

        OpenAITokenizer(tokenizer=enc, max_tokens=N) — the `tokenizer` field
        takes a tiktoken.Encoding object, not a model name string.
        """
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

        enc = tiktoken.encoding_for_model("text-embedding-3-large")
        tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=1024)
        return HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
        )

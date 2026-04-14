"""Chunker configuration factories + startup validators.

This module is deliberately side-effect free at import time.  The chunker
factories are functions so the expensive Docling imports happen only when a
chunker is actually constructed.

The 10 configurations map to the v2 plan:

    C1  cl100k/512  merge=True   md-tables  repeat-hdr        -- sweet spot
    C2  cl100k/1024 merge=True   md-tables  repeat-hdr        -- medium
    C3  cl100k/256  merge=True   md-tables  repeat-hdr        -- small
    C4  cl100k/768  merge=True   md-tables  repeat-hdr        -- middle
    C5  cl100k/512  merge=False  md-tables  repeat-hdr        -- peer-merge ablation
    C6  cl100k/512  merge=True   default    repeat-hdr        -- table serializer ablation
    C7  cl100k/1500 merge=True   md-tables  repeat-hdr        -- production-aligned
    C8  HierarchicalChunker                                    -- baseline
    C9  legacy NLTK TextTiling + sliding window                -- Python legacy
    C10 cl100k/512  merge=True   md-tables  repeat+overflow    -- overflow ablation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

LOG = logging.getLogger("docling_lab.configs")

# ----------------------------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkerConfig:
    """Fully-described chunker variant for the sweep."""

    cfg_id: str
    family: str  # "hybrid" | "hierarchical" | "legacy_nltk"
    max_tokens: int | None
    merge_peers: bool | None
    tokenizer_kind: str | None  # "openai_cl100k" | "hf_minilm" | None
    markdown_table_serializer: bool
    repeat_table_header: bool
    omit_header_on_overflow: bool
    rationale: str
    # Set True to exempt the config from structural gates that depend on
    # Docling's doc_items (legacy chunker has none).
    structural_gate_exempt: bool = False


CONFIGS: list[ChunkerConfig] = [
    ChunkerConfig(
        cfg_id="C1",
        family="hybrid",
        max_tokens=512,
        merge_peers=True,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
        rationale="sweet spot for text-embedding-3-large",
    ),
    ChunkerConfig(
        cfg_id="C2",
        family="hybrid",
        max_tokens=1024,
        merge_peers=True,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
        rationale="medium budget",
    ),
    ChunkerConfig(
        cfg_id="C3",
        family="hybrid",
        max_tokens=256,
        merge_peers=True,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
        rationale="small — fine-grained retrieval",
    ),
    ChunkerConfig(
        cfg_id="C4",
        family="hybrid",
        max_tokens=768,
        merge_peers=True,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
        rationale="middle ground",
    ),
    ChunkerConfig(
        cfg_id="C5",
        family="hybrid",
        max_tokens=512,
        merge_peers=False,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
        rationale="ablation: peer-merge disabled",
    ),
    ChunkerConfig(
        cfg_id="C6",
        family="hybrid",
        max_tokens=512,
        merge_peers=True,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=False,
        repeat_table_header=True,
        omit_header_on_overflow=False,
        rationale="ablation: default table serialization",
    ),
    ChunkerConfig(
        cfg_id="C7",
        family="hybrid",
        max_tokens=1500,
        merge_peers=True,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
        rationale="production-aligned with agent-rs Rust chunker (1500 budget)",
    ),
    ChunkerConfig(
        cfg_id="C8",
        family="hierarchical",
        max_tokens=None,
        merge_peers=None,
        tokenizer_kind=None,
        markdown_table_serializer=False,
        repeat_table_header=False,
        omit_header_on_overflow=False,
        rationale="HierarchicalChunker baseline — no token refinement",
    ),
    ChunkerConfig(
        cfg_id="C9",
        family="legacy_nltk",
        max_tokens=1024,
        merge_peers=None,
        tokenizer_kind=None,
        markdown_table_serializer=False,
        repeat_table_header=False,
        omit_header_on_overflow=False,
        rationale="reproduces Python legacy NLTK TextTiling + sliding window",
        structural_gate_exempt=True,
    ),
    ChunkerConfig(
        cfg_id="C10",
        family="hybrid",
        max_tokens=512,
        merge_peers=True,
        tokenizer_kind="openai_cl100k",
        markdown_table_serializer=True,
        repeat_table_header=True,
        omit_header_on_overflow=True,
        rationale="ablation: omit_header_on_overflow for wide tables",
    ),
]

CONFIG_BY_ID: dict[str, ChunkerConfig] = {c.cfg_id: c for c in CONFIGS}


# ----------------------------------------------------------------------------
# Chunker factories
# ----------------------------------------------------------------------------


def build_chunker(cfg: ChunkerConfig) -> Any:
    """Build a live chunker instance for the given config.

    Returns a chunker object that exposes `.chunk(doc) -> iterable[Chunk]`.
    For C9 the return is a thin wrapper that accepts a `DoclingDocument`
    and produces chunks analogous to HybridChunker's output shape (with
    `.text` and a lightweight `.meta` namespace).
    """
    if cfg.family == "hybrid":
        return _build_hybrid(cfg)
    if cfg.family == "hierarchical":
        return _build_hierarchical(cfg)
    if cfg.family == "legacy_nltk":
        return _build_legacy_nltk(cfg)
    raise ValueError(f"unknown family: {cfg.family}")


def _build_hybrid(cfg: ChunkerConfig) -> Any:
    import tiktoken
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    if cfg.tokenizer_kind != "openai_cl100k":
        raise ValueError(
            f"config {cfg.cfg_id}: only openai_cl100k tokenizer supported"
        )

    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=cfg.max_tokens)

    kwargs: dict[str, Any] = {
        "tokenizer": tokenizer,
        "merge_peers": bool(cfg.merge_peers),
    }
    # repeat_table_header flag is only supported when markdown table
    # serializer provider is set.  Passing the repeat flag without the
    # serializer is a no-op in docling 2.x.
    kwargs["repeat_table_header"] = cfg.repeat_table_header
    kwargs["omit_header_on_overflow"] = cfg.omit_header_on_overflow

    if cfg.markdown_table_serializer:
        serializer_provider = _markdown_table_serializer_provider()
        kwargs["serializer_provider"] = serializer_provider

    return HybridChunker(**kwargs)


def _build_hierarchical(cfg: ChunkerConfig) -> Any:
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

    return HierarchicalChunker()


def _build_legacy_nltk(cfg: ChunkerConfig) -> "LegacyNLTKChunker":
    return LegacyNLTKChunker(max_tokens=cfg.max_tokens or 1024)


def _markdown_table_serializer_provider() -> Any:
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
    )
    from docling_core.transforms.chunker.hybrid_chunker import (
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import (
        MarkdownParams,
        MarkdownTableSerializer,
    )

    class _Prov(ChunkingSerializerProvider):
        def get_serializer(self, doc: Any) -> Any:
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(compact_tables=True),
            )

    return _Prov()


# ----------------------------------------------------------------------------
# Legacy NLTK wrapper (replicates the Python agentic_ra chunker shape)
# ----------------------------------------------------------------------------


@dataclass
class _LegacyChunkMeta:
    headings: list[str] = field(default_factory=list)
    doc_items: list[Any] = field(default_factory=list)


@dataclass
class _LegacyChunk:
    text: str
    meta: _LegacyChunkMeta


class LegacyNLTKChunker:
    """Reproduces the old Python `chunk_text` (NLTK TextTiling + sliding).

    This chunker consumes a `DoclingDocument` by first calling
    `doc.export_to_markdown()` (mirroring the pipeline in
    `src/agentic_ra/retrieval/embedder.py`).  It then applies TextTiling for
    topic segmentation and a word-count sliding window for over-long topics.
    """

    def __init__(self, max_tokens: int = 1024) -> None:
        # Mirror legacy math: ~4 chars/token, ~5 chars/word.
        self._max_chars = max_tokens * 4
        self._max_words = self._max_chars // 5           # ~819 for 1024 tokens
        self._step_words = self._max_chars // 7          # ~585 for 1024 tokens

    # ``contextualize`` is called by the sweep runner for parity with the
    # Docling chunker API.  Legacy chunks have no structural context, so we
    # simply return the raw text.
    def contextualize(self, chunk: _LegacyChunk) -> str:
        return chunk.text

    def chunk(self, doc: Any):  # type: ignore[override]
        full_text = doc.export_to_markdown()
        texts = self._chunk_text(full_text)
        for text in texts:
            yield _LegacyChunk(text=text, meta=_LegacyChunkMeta())

    def _chunk_text(self, text: str) -> list[str]:
        import re

        if not text or not text.strip():
            return []

        topics = self._topic_segment(text)
        chunks: list[str] = []
        for topic in topics:
            words = topic.split()
            if len(words) <= self._max_words:
                chunks.append(topic)
            else:
                chunks.extend(self._sliding_window(topic))
        return [c for c in chunks if c.strip()]

    def _topic_segment(self, text: str) -> list[str]:
        import re

        try:
            import nltk
            from nltk.tokenize import TextTilingTokenizer

            tokenizer = TextTilingTokenizer()
            segments = tokenizer.tokenize(text)
            if segments:
                return [s.strip() for s in segments if s.strip()]
        except (ImportError, ValueError, LookupError) as exc:
            LOG.debug("TextTiling unavailable (%s); falling back to paragraphs", exc)

        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _sliding_window(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self._max_words:
            return [text]
        out: list[str] = []
        i = 0
        while i < len(words):
            out.append(" ".join(words[i : i + self._max_words]))
            if i + self._max_words >= len(words):
                break
            i += self._step_words
        return out


# ----------------------------------------------------------------------------
# Startup validators
# ----------------------------------------------------------------------------


def ensure_nltk_resources() -> None:
    """Ensure the NLTK resources used by C9 are installed (downloads if not)."""
    import nltk

    required = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for probe_key, download_key in required:
        try:
            nltk.data.find(probe_key)
        except LookupError:
            LOG.warning("downloading NLTK resource: %s", download_key)
            nltk.download(download_key, quiet=True)


def resolve_qa_model() -> str:
    """Verify the configured QA model exists, abort otherwise.

    Returns the model id that should be used for QA generation.
    """
    import os

    import openai

    model_id = os.environ.get("OPENAI_QA_MODEL", "gpt-5.4-mini-2026-03-17")
    client = openai.OpenAI()
    try:
        client.models.retrieve(model_id)
    except openai.NotFoundError as exc:
        raise SystemExit(
            f"OpenAI model '{model_id}' not available "
            f"(set OPENAI_QA_MODEL to override). Underlying error: {exc}"
        ) from exc
    LOG.info("QA model verified: %s", model_id)
    return model_id


def tokenizer_smoke_test(parsed_texts: dict[str, str]) -> None:
    """Ensure cl100k_base can encode every parsed document strictly.

    Per Codex round 2, use disallowed_special="all" so any BPE special
    token literal in user-supplied text triggers a ValueError we can
    surface before the sweep starts.  If this raises, the offending doc
    needs pre-sanitization (strip the offending marker).
    """
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    for doc_id, text in parsed_texts.items():
        try:
            enc.encode(text, disallowed_special="all")
        except ValueError as exc:
            raise SystemExit(
                f"doc {doc_id} contains special-token literal that would "
                f"poison tiktoken encode; sanitize before chunking. Error: {exc}"
            ) from exc
    LOG.info("tiktoken smoke test passed on %d docs", len(parsed_texts))

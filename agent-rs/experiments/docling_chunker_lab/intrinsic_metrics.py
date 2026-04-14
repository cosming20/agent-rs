"""Intrinsic (diagnostic) metrics for a chunking result.

These metrics do NOT drive winner selection (per Codex round 1 CRITICAL
#1).  Their only active role is as **hard gates** that disqualify a
config from the winner pool.

Hard gates (any failure -> disqualify):

    gate_embedding_limit
        No chunk in any doc has `embed_text` > 8191 tokens
        (text-embedding-3-large hard input limit).

    gate_doc_item_coverage
        At least 50% of chunks carry >= 1 doc_item_ref.
        Configs with `structural_gate_exempt = True` (C9 legacy) skip this.

    gate_table_integrity
        For D4 / D5 (the table-heavy docs), the estimated fraction of
        tables split across chunks is at most 60%.
        Configs with `structural_gate_exempt = True` skip this too, per
        Codex round 2.  Non-structural configs (C8 hierarchical) are still
        evaluated — HierarchicalChunker has doc_items.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Iterable

from chunker_runner import Chunk, ChunkingResult
from configs import CONFIG_BY_ID

LOG = logging.getLogger("docling_lab.intrinsic_metrics")

EMBED_MODEL_HARD_LIMIT = 8191           # text-embedding-3-large max input
STRUCTURAL_COVERAGE_MIN = 0.50          # >= 50% of chunks must carry doc items
TABLE_SPLIT_FAIL_RATIO = 0.60           # > 60% split -> fail
TABLE_HEAVY_DOC_IDS = {"D4", "D5"}

_PUNCT_ENDINGS = re.compile(r"[.!?:;]\s*$")
_HEADING_ENDINGS = re.compile(r"(^|\n)(#+ |\d+\.\s|[A-Z][A-Z0-9 _-]{2,}$)")


# ----------------------------------------------------------------------------
# Output shapes
# ----------------------------------------------------------------------------


@dataclass
class TokenStats:
    mean: float
    median: float
    p25: float
    p75: float
    p95: float
    min: int
    max: int
    stddev: float


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float | int | str
    detail: str


@dataclass
class IntrinsicMetrics:
    cfg_id: str
    doc_id: str
    n_chunks: int
    embed_tokens: TokenStats
    pct_over_embedding_limit: float
    pct_over_declared_budget: float | None
    size_uniformity: float
    boundary_quality: float
    pct_chunks_with_headings: float
    pct_chunks_with_doc_items: float
    chunker_latency_ms: int
    table_stats: dict[str, float | int] = field(default_factory=dict)
    gates: list[GateResult] = field(default_factory=list)

    @property
    def all_gates_passed(self) -> bool:
        return all(g.passed for g in self.gates)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def compute(
    result: ChunkingResult,
    *,
    source_table_count: int,
) -> IntrinsicMetrics:
    """Compute all intrinsic metrics + run hard gates for a chunk set."""
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    cfg = CONFIG_BY_ID[result.cfg_id]

    embed_token_counts = [
        len(enc.encode(c.embed_text, disallowed_special="all"))
        for c in result.chunks
    ]

    token_stats = _token_stats(embed_token_counts)

    pct_over_limit = _pct(
        embed_token_counts, lambda n: n > EMBED_MODEL_HARD_LIMIT
    )
    pct_over_budget: float | None = None
    if cfg.max_tokens is not None:
        pct_over_budget = _pct(
            embed_token_counts, lambda n: n > (cfg.max_tokens or 0)
        )

    size_uniformity = _size_uniformity(embed_token_counts)
    boundary_quality = _boundary_quality(result.chunks)
    pct_with_headings = _pct_chunks_matching(
        result.chunks, lambda c: bool(c.headings)
    )
    pct_with_doc_items = _pct_chunks_matching(
        result.chunks, lambda c: c.n_doc_items > 0
    )

    table_stats = _table_stats(result.chunks, source_table_count)

    gates = _run_gates(
        cfg_id=cfg.cfg_id,
        doc_id=result.doc_id,
        cfg_structural_exempt=cfg.structural_gate_exempt,
        max_embed_tokens=max(embed_token_counts) if embed_token_counts else 0,
        pct_with_doc_items=pct_with_doc_items,
        table_split_ratio=float(table_stats.get("pct_tables_split", 0.0)),
    )

    return IntrinsicMetrics(
        cfg_id=result.cfg_id,
        doc_id=result.doc_id,
        n_chunks=len(result.chunks),
        embed_tokens=token_stats,
        pct_over_embedding_limit=pct_over_limit,
        pct_over_declared_budget=pct_over_budget,
        size_uniformity=size_uniformity,
        boundary_quality=boundary_quality,
        pct_chunks_with_headings=pct_with_headings,
        pct_chunks_with_doc_items=pct_with_doc_items,
        chunker_latency_ms=result.chunker_latency_ms,
        table_stats=table_stats,
        gates=gates,
    )


# ----------------------------------------------------------------------------
# Token stats
# ----------------------------------------------------------------------------


def _token_stats(counts: list[int]) -> TokenStats:
    if not counts:
        return TokenStats(0, 0, 0, 0, 0, 0, 0, 0)
    sorted_counts = sorted(counts)
    mean = sum(counts) / len(counts)
    return TokenStats(
        mean=round(mean, 2),
        median=round(median(counts), 2),
        p25=round(_percentile(sorted_counts, 0.25), 2),
        p75=round(_percentile(sorted_counts, 0.75), 2),
        p95=round(_percentile(sorted_counts, 0.95), 2),
        min=min(counts),
        max=max(counts),
        stddev=round(pstdev(counts), 2) if len(counts) > 1 else 0.0,
    )


def _percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _pct(values: list[int], pred) -> float:
    if not values:
        return 0.0
    return round(sum(1 for v in values if pred(v)) / len(values), 4)


def _pct_chunks_matching(chunks: Iterable[Chunk], pred) -> float:
    items = list(chunks)
    if not items:
        return 0.0
    return round(sum(1 for c in items if pred(c)) / len(items), 4)


def _size_uniformity(counts: list[int]) -> float:
    if not counts or len(counts) < 2:
        return 1.0
    mean = sum(counts) / len(counts)
    if mean == 0:
        return 0.0
    sigma = pstdev(counts)
    return round(max(0.0, 1.0 - (sigma / mean)), 4)


def _boundary_quality(chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    ok = 0
    for c in chunks:
        text = c.raw_text.rstrip()
        if not text:
            continue
        if _PUNCT_ENDINGS.search(text) or _HEADING_ENDINGS.search(text):
            ok += 1
    return round(ok / len(chunks), 4)


# ----------------------------------------------------------------------------
# Table integrity
# ----------------------------------------------------------------------------


def _table_stats(
    chunks: list[Chunk], source_table_count: int
) -> dict[str, float | int]:
    """Heuristic estimate of how tables were split across chunks.

    A doc_item_ref pointing to a table element typically starts with
    '#/tables/' in docling 2.x.  We count distinct table indices referenced
    by at least one chunk, and how many chunks reference each.  A table
    appearing in more than one chunk is considered 'split'.
    """
    table_refs: dict[str, set[str]] = {}
    for c in chunks:
        for ref in c.doc_item_refs:
            if "tables" in ref:
                table_refs.setdefault(ref, set()).add(c.chunk_id)

    referenced = len(table_refs)
    split_tables = sum(1 for chunks_for_table in table_refs.values() if len(chunks_for_table) > 1)
    pct_split = round(split_tables / referenced, 4) if referenced > 0 else 0.0

    return {
        "source_table_count": source_table_count,
        "chunks_referencing_tables": sum(1 for c in chunks if any("tables" in r for r in c.doc_item_refs)),
        "referenced_table_count": referenced,
        "split_table_count": split_tables,
        "pct_tables_split": pct_split,
    }


# ----------------------------------------------------------------------------
# Hard gates
# ----------------------------------------------------------------------------


def _run_gates(
    *,
    cfg_id: str,
    doc_id: str,
    cfg_structural_exempt: bool,
    max_embed_tokens: int,
    pct_with_doc_items: float,
    table_split_ratio: float,
) -> list[GateResult]:
    """Run all hard gates in order.  Any failure excludes the config."""
    gates: list[GateResult] = []

    # 1. Embedding-limit gate -- never exempted.
    gates.append(
        GateResult(
            name="embedding_limit",
            passed=max_embed_tokens <= EMBED_MODEL_HARD_LIMIT,
            value=max_embed_tokens,
            detail=(
                f"max embed_text tokens: {max_embed_tokens} "
                f"(limit {EMBED_MODEL_HARD_LIMIT})"
            ),
        )
    )

    # 2. Doc-item coverage gate -- exempt structural-less configs (C9).
    if cfg_structural_exempt:
        gates.append(
            GateResult(
                name="doc_item_coverage",
                passed=True,
                value="exempt",
                detail=f"{cfg_id} is structural_gate_exempt",
            )
        )
    else:
        gates.append(
            GateResult(
                name="doc_item_coverage",
                passed=pct_with_doc_items >= STRUCTURAL_COVERAGE_MIN,
                value=pct_with_doc_items,
                detail=(
                    f"pct_chunks_with_doc_items={pct_with_doc_items:.2f} "
                    f"(min {STRUCTURAL_COVERAGE_MIN})"
                ),
            )
        )

    # 3. Table integrity gate -- exempt structural-less configs (C9) too.
    # Only active on table-heavy docs D4/D5.
    if cfg_structural_exempt:
        gates.append(
            GateResult(
                name="table_integrity",
                passed=True,
                value="exempt",
                detail=f"{cfg_id} is structural_gate_exempt",
            )
        )
    elif doc_id not in TABLE_HEAVY_DOC_IDS:
        gates.append(
            GateResult(
                name="table_integrity",
                passed=True,
                value="n/a",
                detail=f"doc {doc_id} is not table-heavy",
            )
        )
    else:
        gates.append(
            GateResult(
                name="table_integrity",
                passed=table_split_ratio <= TABLE_SPLIT_FAIL_RATIO,
                value=table_split_ratio,
                detail=(
                    f"pct_tables_split={table_split_ratio:.2f} "
                    f"(max {TABLE_SPLIT_FAIL_RATIO})"
                ),
            )
        )

    LOG.debug(
        "gate results doc=%s cfg=%s: %s",
        doc_id,
        cfg_id,
        [(g.name, g.passed) for g in gates],
    )
    return gates

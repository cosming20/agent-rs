"""Dedup, reciprocal rank fusion, and evidence pack assembly."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from haystack import Document

from agentic_ra.core.errors import RankingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidencePack:
    """Ranked, deduplicated evidence from retrieval.

    Attributes:
        documents: Ranked documents with fused scores.
        query: Original user query.
        strategy: Retrieval strategy used (vector/keyword/hybrid).
        source_counts: Number of results per source before dedup.
    """

    documents: tuple[Document, ...]
    query: str
    strategy: str
    source_counts: dict[str, int] = field(default_factory=dict)


class ResultRanker:
    """Deduplicates and ranks retrieval results using Reciprocal Rank Fusion.

    Args:
        rrf_k: RRF constant that dampens high-rank contributions. Default 60.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        self._rrf_k = rrf_k

    def rank(
        self,
        results: list[Document],
        query: str,
        strategy: str,
        limit: int = 10,
    ) -> EvidencePack:
        """Deduplicate and rank results using RRF.

        Args:
            results: Combined results from retrieval router.
                Each document must have ``meta["score"]`` and ``meta["source"]``.
            query: Original user query.
            strategy: Retrieval strategy used.
            limit: Maximum documents in the evidence pack.

        Returns:
            Ranked, deduplicated evidence pack.
        """
        if not results:
            return EvidencePack(documents=(), query=query, strategy=strategy)

        try:
            # Count sources before dedup
            source_counts: dict[str, int] = defaultdict(int)
            for doc in results:
                source = doc.meta.get("source", "unknown")
                source_counts[source] += 1

            # Group by source for RRF scoring
            by_source: dict[str, list[Document]] = defaultdict(list)
            for doc in results:
                source = doc.meta.get("source", "unknown")
                by_source[source].append(doc)

            # Sort each source's results by their original score (descending)
            for source, source_docs in by_source.items():
                for doc in source_docs:
                    if "score" not in doc.meta:
                        logger.warning(
                            "Document missing 'score' in meta, defaulting to 0.0",
                            extra={"source": source, "doc_id": doc.id},
                        )
                source_docs.sort(key=lambda d: d.meta.get("score", 0.0), reverse=True)

            # Compute RRF scores — dedup key uses doc.id for empty content
            rrf_scores: dict[str, float] = defaultdict(float)
            content_to_doc: dict[str, Document] = {}

            for source_docs in by_source.values():
                for rank_idx, doc in enumerate(source_docs):
                    content_key = doc.content if doc.content else f"__empty_{doc.id}"
                    rrf_scores[content_key] += 1.0 / (self._rrf_k + rank_idx + 1)
                    if content_key not in content_to_doc:
                        content_to_doc[content_key] = doc

            # Sort by fused RRF score descending, take top limit
            ranked_keys = sorted(
                rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
            )[:limit]

            # Build final documents with fused score in metadata
            ranked_docs = []
            for key in ranked_keys:
                doc = content_to_doc[key]
                ranked_docs.append(
                    Document(
                        content=doc.content,
                        meta={
                            **doc.meta,
                            "rrf_score": rrf_scores[key],
                        },
                    )
                )

            return EvidencePack(
                documents=tuple(ranked_docs),
                query=query,
                strategy=strategy,
                source_counts=dict(source_counts),
            )

        except Exception as e:
            raise RankingError(
                "Ranking failed",
                operation="rank",
                document_count=len(results),
                original_error=str(e),
            ) from e

    def fuse_evidence(
        self,
        kb_documents: list[Document],
        web_documents: list[Document],
        query: str,
        kb_weight: float = 1.5,
        limit: int = 15,
    ) -> EvidencePack:
        """Fuse KB and web evidence into a single ranked evidence pack.

        Deduplicates across sources, applies cross-source RRF with a
        weight boost for KB results (trusted source), and preserves
        provenance metadata.

        Args:
            kb_documents: Evidence from knowledge base retrieval.
            web_documents: Evidence from web search and URL fetch.
            query: Original user query.
            kb_weight: Multiplier for KB document RRF scores. Default 1.5.
            limit: Maximum documents in the fused evidence pack.

        Returns:
            Fused, ranked evidence pack with provenance tags.
        """
        if not kb_documents and not web_documents:
            return EvidencePack(documents=(), query=query, strategy="fused")

        try:
            # Tag provenance if not already tagged
            for doc in kb_documents:
                if "provider" not in doc.meta:
                    doc.meta["provider"] = "cognee"
            for doc in web_documents:
                if "provider" not in doc.meta:
                    doc.meta["provider"] = "web"

            # Dedup by content prefix (first 200 chars)
            seen_content: set[str] = set()
            unique_kb: list[Document] = []
            for doc in kb_documents:
                key = (doc.content or "")[:200]
                if key not in seen_content:
                    seen_content.add(key)
                    unique_kb.append(doc)

            unique_web: list[Document] = []
            for doc in web_documents:
                key = (doc.content or "")[:200]
                if key not in seen_content:
                    seen_content.add(key)
                    unique_web.append(doc)

            # RRF scoring with source weighting
            rrf_scores: dict[str, float] = {}
            content_to_doc: dict[str, Document] = {}

            # Score KB documents (with weight boost)
            for rank_idx, doc in enumerate(unique_kb):
                content_key = doc.content if doc.content else f"__empty_{doc.id}"
                rrf_scores[content_key] = kb_weight / (self._rrf_k + rank_idx + 1)
                if content_key not in content_to_doc:
                    content_to_doc[content_key] = doc

            # Score web documents (standard weight)
            for rank_idx, doc in enumerate(unique_web):
                content_key = doc.content if doc.content else f"__empty_{doc.id}"
                rrf_scores[content_key] = 1.0 / (self._rrf_k + rank_idx + 1)
                if content_key not in content_to_doc:
                    content_to_doc[content_key] = doc

            # Sort by fused score descending
            ranked_keys = sorted(
                rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
            )[:limit]

            ranked_docs = []
            for content_key in ranked_keys:
                doc = content_to_doc[content_key]
                ranked_docs.append(
                    Document(
                        content=doc.content,
                        meta={
                            **doc.meta,
                            "rrf_score": rrf_scores[content_key],
                        },
                    )
                )

            source_counts: dict[str, int] = {}
            source_counts["kb"] = len(unique_kb)
            source_counts["web"] = len(unique_web)

            logger.info(
                "Evidence fusion: %d KB + %d web -> %d fused documents",
                len(unique_kb),
                len(unique_web),
                len(ranked_docs),
            )

            return EvidencePack(
                documents=tuple(ranked_docs),
                query=query,
                strategy="fused",
                source_counts=source_counts,
            )

        except Exception as e:
            raise RankingError(
                "Evidence fusion failed",
                operation="fuse_evidence",
                kb_count=len(kb_documents),
                web_count=len(web_documents),
                original_error=str(e),
            ) from e

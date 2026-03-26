"""Dedup, reciprocal rank fusion, and evidence pack assembly."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from haystack import Document

from agentic_ra.core.errors import RankingError


@dataclass(frozen=True)
class EvidencePack:
    """Ranked, deduplicated evidence from retrieval.

    Attributes:
        documents: Ranked documents with fused scores.
        query: Original user query.
        strategy: Retrieval strategy used (vector/keyword/hybrid).
        source_counts: Number of results per source before dedup.
    """

    documents: list[Document]
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
            query: Original user query.
            strategy: Retrieval strategy used.
            limit: Maximum documents in the evidence pack.

        Returns:
            Ranked, deduplicated evidence pack.
        """
        if not results:
            return EvidencePack(documents=[], query=query, strategy=strategy)

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
            for source_docs in by_source.values():
                source_docs.sort(
                    key=lambda d: d.meta.get("score", 0.0), reverse=True
                )

            # Compute RRF scores
            rrf_scores: dict[str, float] = defaultdict(float)
            content_to_doc: dict[str, Document] = {}

            for source_docs in by_source.values():
                for rank_idx, doc in enumerate(source_docs):
                    content_key = doc.content or ""
                    rrf_scores[content_key] += 1.0 / (self._rrf_k + rank_idx + 1)
                    # Keep the version with higher accumulated RRF score
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
                documents=ranked_docs,
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

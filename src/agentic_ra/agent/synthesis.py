"""Response synthesis with citations and provenance."""

from __future__ import annotations

from dataclasses import dataclass

from haystack import Document

from agentic_ra.core.errors import SynthesisError
from agentic_ra.agent.verifier import VerificationResult


@dataclass(frozen=True)
class Citation:
    """A citation linking a claim to its source evidence.

    Attributes:
        index: Citation number (1-based).
        content_snippet: Relevant excerpt from the source document.
        source: Origin of the evidence (e.g., "qdrant", "opensearch").
        score: Relevance score from retrieval.
    """

    index: int
    content_snippet: str
    source: str
    score: float


@dataclass(frozen=True)
class SynthesizedResponse:
    """Final agent response with citations and provenance.

    Attributes:
        answer: The synthesized answer text.
        citations: Source citations supporting the answer.
        query: Original user query.
        confidence: Verification confidence score.
        is_grounded: Whether the answer passed grounding verification.
        strategy: Retrieval strategy used.
    """

    answer: str
    citations: tuple[Citation, ...]
    query: str
    confidence: float
    is_grounded: bool
    strategy: str = ""


class ResponseSynthesizer:
    """Builds final responses with citations and provenance metadata.

    Takes an answer, evidence documents, and verification result,
    and assembles a structured response with citations linking
    claims to source evidence.

    Args:
        max_snippet_length: Maximum character length for citation snippets.
    """

    def __init__(self, max_snippet_length: int = 200) -> None:
        self._max_snippet = max_snippet_length

    def synthesize(
        self,
        answer: str,
        evidence: tuple[Document, ...] | list[Document],
        query: str,
        verification: VerificationResult,
        strategy: str = "",
    ) -> SynthesizedResponse:
        """Build a response with citations from evidence documents.

        Args:
            answer: The generated answer text.
            evidence: Documents used to generate the answer.
            query: Original user query.
            verification: Grounding verification result.
            strategy: Retrieval strategy that was used.

        Returns:
            Structured response with citations and metadata.
        """
        try:
            citations = self._build_citations(evidence)

            return SynthesizedResponse(
                answer=answer,
                citations=citations,
                query=query,
                confidence=verification.confidence,
                is_grounded=verification.is_grounded,
                strategy=strategy,
            )

        except Exception as e:
            raise SynthesisError(
                "Response synthesis failed",
                operation="synthesize",
                original_error=str(e),
            ) from e

    def _build_citations(
        self,
        evidence: tuple[Document, ...] | list[Document],
    ) -> tuple[Citation, ...]:
        """Extract citations from evidence documents."""
        citations: list[Citation] = []
        for idx, doc in enumerate(evidence, start=1):
            snippet = (doc.content or "")[:self._max_snippet]
            if len(doc.content or "") > self._max_snippet:
                snippet += "..."

            citations.append(
                Citation(
                    index=idx,
                    content_snippet=snippet,
                    source=doc.meta.get("source", "unknown"),
                    score=doc.meta.get("rrf_score", doc.meta.get("score", 0.0)),
                )
            )

        return tuple(citations)

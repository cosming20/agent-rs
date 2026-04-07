"""Response synthesis with citations and provenance."""

from __future__ import annotations

import re
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
        source: Origin of the evidence (e.g., "cognee_graph", "cognee_chunks").
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
        confidence: User-facing answer quality score. Semantics:
            - 0.0: no evidence available (unknown quality)
            - close to 0.0: verifier is confident the answer is ungrounded
            - close to 1.0: verifier is confident the answer is well-grounded
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
            all_citations = self._build_citations(evidence)

            # Only include citations the LLM actually referenced in the answer
            # (e.g., [1], [4], [7]).  Unreferenced evidence is noise.
            referenced = self._find_referenced_indices(answer)
            if referenced:
                citations = tuple(c for c in all_citations if c.index in referenced)
            else:
                # If no explicit references found, keep top 5 for context
                citations = all_citations[:5]

            # The verifier's confidence is meta-confidence ("how certain am I
            # about this grounding assessment"), not answer quality.  When the
            # answer is ungrounded, high verifier confidence means "I'm very
            # sure this is bad" — invert for user-facing semantics.
            #
            # Special case: confidence=0.0 means "no evidence to assess" —
            # keep it at 0.0 (unknown quality), don't invert to 1.0.
            if verification.is_grounded:
                answer_confidence = verification.confidence
            elif verification.confidence < 1e-9:
                answer_confidence = 0.0
            else:
                answer_confidence = 1.0 - verification.confidence

            return SynthesizedResponse(
                answer=answer,
                citations=citations,
                query=query,
                confidence=answer_confidence,
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
            content = doc.meta.get("chunk_text", doc.content or "")
            snippet = content[: self._max_snippet]
            if len(content) > self._max_snippet:
                snippet += "..."

            # Prefer original retrieval score (cosine similarity 0-1) for
            # display.  RRF scores are tiny fractions useful only for
            # ordering — showing them as percentages misleads users.
            citations.append(
                Citation(
                    index=idx,
                    content_snippet=snippet,
                    source=doc.meta.get("source", "unknown"),
                    score=doc.meta.get("score", doc.meta.get("rrf_score", 0.0)),
                )
            )

        return tuple(citations)

    @staticmethod
    def _find_referenced_indices(answer: str) -> set[int]:
        """Extract citation indices like [1], [4] from the answer text."""
        return {int(m) for m in re.findall(r"\[(\d+)\]", answer)}

"""Faithfulness and grounding verification for agent responses."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from haystack import Document

from agentic_ra.core.errors import VerificationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    """Result of a faithfulness/grounding check.

    Attributes:
        is_grounded: Whether the answer is grounded in the evidence.
        confidence: Confidence score between 0.0 and 1.0.
        reasoning: Explanation of the verification decision.
        ungrounded_claims: Claims in the answer not supported by evidence.
    """

    is_grounded: bool
    confidence: float
    reasoning: str
    ungrounded_claims: tuple[str, ...] = ()


class GroundingVerifier:
    """Verifies that an answer is faithfully grounded in retrieved evidence.

    Uses heuristic overlap scoring for initial implementation.
    Future: replace with LLM-based faithfulness evaluation.

    Args:
        confidence_threshold: Minimum confidence to consider answer grounded.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self._threshold = confidence_threshold

    def verify(
        self,
        answer: str,
        evidence: tuple[Document, ...] | list[Document],
    ) -> VerificationResult:
        """Check if the answer is grounded in the provided evidence.

        Args:
            answer: The generated answer text.
            evidence: Retrieved documents used to generate the answer.

        Returns:
            Verification result with grounding assessment.
        """
        if not answer:
            return VerificationResult(
                is_grounded=False,
                confidence=0.0,
                reasoning="Empty answer cannot be verified.",
            )

        if not evidence:
            return VerificationResult(
                is_grounded=False,
                confidence=0.0,
                reasoning="No evidence provided for grounding check.",
            )

        try:
            # Heuristic: compute token overlap between answer and evidence
            answer_tokens = set(answer.lower().split())
            evidence_tokens: set[str] = set()
            for doc in evidence:
                if doc.content:
                    evidence_tokens.update(doc.content.lower().split())

            if not answer_tokens:
                return VerificationResult(
                    is_grounded=False,
                    confidence=0.0,
                    reasoning="Answer contains no tokens.",
                )

            overlap = answer_tokens & evidence_tokens
            confidence = len(overlap) / len(answer_tokens)

            is_grounded = confidence >= self._threshold

            # Identify ungrounded tokens (simplified: tokens not in evidence)
            ungrounded = answer_tokens - evidence_tokens
            # Filter out common stop words for cleaner reporting
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                          "at", "to", "for", "of", "and", "or", "but", "not", "with",
                          "that", "this", "it", "be", "as", "by", "from", "has", "have"}
            meaningful_ungrounded = ungrounded - stop_words

            return VerificationResult(
                is_grounded=is_grounded,
                confidence=round(confidence, 3),
                reasoning=(
                    f"Token overlap: {len(overlap)}/{len(answer_tokens)} "
                    f"({confidence:.1%}). Threshold: {self._threshold:.1%}."
                ),
                ungrounded_claims=tuple(sorted(meaningful_ungrounded)[:10]),
            )

        except Exception as e:
            raise VerificationError(
                "Grounding verification failed",
                operation="verify",
                original_error=str(e),
            ) from e

    def needs_retry(self, result: VerificationResult) -> bool:
        """Determine if the answer should trigger a Stage-2 retrieval retry.

        Args:
            result: Verification result from a previous check.

        Returns:
            True if confidence is too low and a retry is warranted.
        """
        return not result.is_grounded

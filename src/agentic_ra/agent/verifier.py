"""Faithfulness and grounding verification for agent responses."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from agentic_ra.core.errors import VerificationError

logger = logging.getLogger(__name__)

_VERIFICATION_PROMPT = """\
You are a faithfulness evaluator. Given an ANSWER and EVIDENCE, determine \
if the answer is fully grounded in the evidence.

Respond with a JSON object only, no other text:
{
  "is_grounded": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "ungrounded_claims": ["claim1", "claim2"]
}

Rules:
- is_grounded: true ONLY if every factual claim in the answer is supported by the evidence.
- confidence: how confident you are in the grounding assessment (1.0 = certain).
- ungrounded_claims: list any claims in the answer NOT supported by the evidence. Empty if grounded.
- If the answer says "no evidence found" or similar, is_grounded=false, confidence=1.0.
"""


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

    Uses LLM-based evaluation when an API key is provided,
    falls back to heuristic token overlap otherwise.

    Args:
        confidence_threshold: Minimum confidence to consider answer grounded.
        api_key: OpenAI API key for LLM-based verification. If empty, uses heuristic.
        model: OpenAI model for verification.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        api_key: str = "",
        model: str = "gpt-5.4-mini-2026-03-17",
    ) -> None:
        self._threshold = confidence_threshold
        self._llm: OpenAIChatGenerator | None = None

        if api_key:
            self._llm = OpenAIChatGenerator(
                model=model,
                api_key=Secret.from_token(api_key),
                generation_kwargs={"temperature": 0.0, "max_completion_tokens": 512},
            )

    def verify(
        self,
        answer: str,
        evidence: tuple[Document, ...] | list[Document],
    ) -> VerificationResult:
        """Check if the answer is grounded in the provided evidence.

        Uses LLM-based evaluation if configured, otherwise falls back to heuristic.

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

        if self._llm is not None:
            return self._verify_with_llm(answer, evidence)
        return self._verify_heuristic(answer, evidence)

    def needs_retry(self, result: VerificationResult) -> bool:
        """Determine if the answer should trigger a Stage-2 retrieval retry."""
        return not result.is_grounded

    def _verify_with_llm(
        self,
        answer: str,
        evidence: tuple[Document, ...] | list[Document],
    ) -> VerificationResult:
        """LLM-based faithfulness verification."""
        evidence_text = "\n\n".join(
            f"[{i}] {doc.meta.get('chunk_text', doc.content or '')}"
            for i, doc in enumerate(evidence, 1)
        )

        messages = [
            ChatMessage.from_system(_VERIFICATION_PROMPT),
            ChatMessage.from_user(
                f"EVIDENCE:\n{evidence_text}\n\nANSWER:\n{answer}"
            ),
        ]

        assert self._llm is not None  # guaranteed by caller check
        try:
            result = self._llm.run(messages=messages)
            replies: list[ChatMessage] = result["replies"]
            if not replies or not replies[0].text:
                return self._verify_heuristic(answer, evidence)

            parsed = json.loads(replies[0].text)
            return VerificationResult(
                is_grounded=bool(parsed.get("is_grounded", False)),
                confidence=float(parsed.get("confidence", 0.0)),
                reasoning=str(parsed.get("reasoning", "")),
                ungrounded_claims=tuple(parsed.get("ungrounded_claims", [])),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("LLM verification parse failed, using heuristic: %s", e)
            return self._verify_heuristic(answer, evidence)
        except Exception as e:
            raise VerificationError(
                "LLM verification failed",
                operation="verify_llm",
                original_error=str(e),
            ) from e

    def _verify_heuristic(
        self,
        answer: str,
        evidence: tuple[Document, ...] | list[Document],
    ) -> VerificationResult:
        """Heuristic token overlap verification (fallback)."""
        answer_tokens = set(answer.lower().split())
        evidence_tokens: set[str] = set()
        for doc in evidence:
            text = doc.meta.get("chunk_text", doc.content or "")
            evidence_tokens.update(text.lower().split())

        if not answer_tokens:
            return VerificationResult(
                is_grounded=False, confidence=0.0,
                reasoning="Answer contains no tokens.",
            )

        overlap = answer_tokens & evidence_tokens
        confidence = len(overlap) / len(answer_tokens)
        is_grounded = confidence >= self._threshold

        return VerificationResult(
            is_grounded=is_grounded,
            confidence=round(confidence, 3),
            reasoning=(
                f"Heuristic: token overlap {len(overlap)}/{len(answer_tokens)} "
                f"({confidence:.1%}). Threshold: {self._threshold:.1%}."
            ),
        )

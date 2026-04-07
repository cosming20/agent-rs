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
from agentic_ra.observability.cost_tracker import get_tracker

logger = logging.getLogger(__name__)

_VERIFICATION_PROMPT = """\
You are a faithfulness evaluator. Given an ANSWER and EVIDENCE, determine \
if the answer is fully grounded in the evidence.

Respond with a JSON object only, no other text.
The EVIDENCE is delimited by <evidence> tags.
The ANSWER is delimited by <answer> tags.

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
        model: str = "gpt-4o-mini",
    ) -> None:
        self._threshold = confidence_threshold
        self._llm: OpenAIChatGenerator | None = None

        if api_key:
            # Strip provider prefix (e.g. "openai/gpt-4o-mini" -> "gpt-4o-mini")
            bare_model = model.split("/", 1)[-1] if "/" in model else model
            self._llm = OpenAIChatGenerator(
                model=bare_model,
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
        """Determine if the answer should trigger a Stage-2 retrieval retry.

        Retries on any ungrounded result — including zero-evidence cases,
        where a broader retrieval strategy may succeed.
        """
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
                f"<evidence>\n{evidence_text}\n</evidence>\n\n<answer>\n{answer}\n</answer>"
            ),
        ]

        if self._llm is None:
            raise VerificationError(
                "LLM verifier not configured", operation="verify_llm"
            )
        try:
            import time as _time

            _v_start = _time.monotonic()
            result = self._llm.run(messages=messages)
            _v_latency = int((_time.monotonic() - _v_start) * 1000)
            replies: list[ChatMessage] = result["replies"]

            # Record verifier LLM cost
            if replies:
                meta = replies[0]._meta if hasattr(replies[0], "_meta") else {}
                usage = meta.get("usage")
                model = meta.get("model", "")
                tracker = get_tracker()
                if tracker and usage and model:
                    tracker.record_call(
                        model=model, purpose="verifier", usage=usage,
                        latency_ms=_v_latency,
                    )

            if not replies or not replies[0].text:
                return self._verify_heuristic(answer, evidence)

            parsed = json.loads(replies[0].text)
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            return VerificationResult(
                is_grounded=bool(parsed.get("is_grounded", False)),
                confidence=confidence,
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

    _STOP_WORDS: frozenset[str] = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "but",
            "and",
            "or",
            "if",
            "while",
            "about",
            "up",
            "its",
            "it",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "they",
            "them",
            "their",
            "what",
            "which",
            "who",
            "whom",
        }
    )

    def _verify_heuristic(
        self,
        answer: str,
        evidence: tuple[Document, ...] | list[Document],
    ) -> VerificationResult:
        """Heuristic token overlap verification (fallback)."""
        answer_tokens = set(answer.lower().split()) - self._STOP_WORDS
        evidence_tokens: set[str] = set()
        for doc in evidence:
            text = doc.meta.get("chunk_text", doc.content or "")
            evidence_tokens.update(set(text.lower().split()) - self._STOP_WORDS)

        if not answer_tokens:
            return VerificationResult(
                is_grounded=False,
                confidence=0.0,
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

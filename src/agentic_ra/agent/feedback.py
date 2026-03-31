"""Feedback adapter — bridges verifier scores to Cognee's session feedback system."""

from __future__ import annotations

import logging

import cognee  # type: ignore[import-untyped]

from agentic_ra.core.config import Settings

logger = logging.getLogger(__name__)


def confidence_to_score(confidence: float) -> int:
    """Map verifier confidence (0.0-1.0) to Cognee feedback score (1-5).

    Mapping:
        0.0-0.2 → 1 (very poor)
        0.2-0.4 → 2 (poor)
        0.4-0.6 → 3 (neutral)
        0.6-0.8 → 4 (good)
        0.8-1.0 → 5 (excellent)
    """
    if confidence >= 0.8:
        return 5
    if confidence >= 0.6:
        return 4
    if confidence >= 0.4:
        return 3
    if confidence >= 0.2:
        return 2
    return 1


class FeedbackAdapter:
    """Submits verifier confidence as Cognee session feedback.

    After the verifier scores a response, this adapter:
    1. Retrieves the latest QA entry from the Cognee session
    2. Converts confidence (0.0-1.0) to feedback score (1-5)
    3. Calls cognee.session.add_feedback()

    Cognee internally stores used_graph_element_ids per search,
    so the feedback pipeline knows which graph nodes/edges to update.

    Args:
        settings: Application settings for feedback config.
    """

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.feedback_enabled
        self._dry_run = settings.feedback_dry_run

    async def submit(
        self,
        session_id: str,
        confidence: float,
        is_grounded: bool,
    ) -> bool:
        """Submit feedback for the latest QA in a Cognee session.

        Args:
            session_id: Cognee session ID (same as passed to search).
            confidence: Verifier confidence score (0.0-1.0).
            is_grounded: Whether the answer passed grounding check.

        Returns:
            True if feedback was submitted, False if skipped.
        """
        if not self._enabled:
            return False

        score = confidence_to_score(confidence)

        if self._dry_run:
            logger.info(
                "Feedback dry-run: session=%s, confidence=%.2f, score=%d",
                session_id,
                confidence,
                score,
            )
            return False

        try:
            entries = await cognee.session.get_session(
                session_id=session_id, last_n=1
            )
            if not entries:
                logger.warning(
                    "No session entries for feedback: session=%s", session_id
                )
                return False

            latest = entries[-1]
            qa_id = latest.qa_id

            feedback_text = (
                f"Grounded={is_grounded}, confidence={confidence:.2f}"
            )

            ok = await cognee.session.add_feedback(
                session_id=session_id,
                qa_id=qa_id,
                feedback_text=feedback_text,
                feedback_score=score,
            )

            if ok:
                logger.info(
                    "Feedback submitted: session=%s, qa_id=%s, score=%d, confidence=%.2f",
                    session_id,
                    qa_id,
                    score,
                    confidence,
                )
            else:
                logger.warning(
                    "Feedback submission returned False: session=%s, qa_id=%s",
                    session_id,
                    qa_id,
                )

            return bool(ok)

        except Exception:
            # Broad catch is intentional — feedback submission must never crash
            # the request pipeline. Errors are logged and surfaced via the
            # asyncio task's done_callback (see orchestrator._background_tasks).
            logger.exception(
                "Failed to submit feedback: session=%s", session_id
            )
            return False

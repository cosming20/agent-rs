"""Grounding gate — verifies answer grounding, retries if needed."""


import asyncio
import logging
from typing import TYPE_CHECKING

from haystack.dataclasses import ChatMessage

from agentic_ra.agent.context import QueryContext
from agentic_ra.agent.steps.agent_runner import AgentRunner
from agentic_ra.agent.verifier import VerificationResult
from agentic_ra.observability.tracing import get_tracer

if TYPE_CHECKING:
    from agentic_ra.agent.verifier import GroundingVerifier
    from agentic_ra.observability.metrics import RAGMetrics
    from agentic_ra.retrieval.ranker import ResultRanker

logger = logging.getLogger(__name__)


class GroundingGate:
    """Verifies answer grounding and retries if the answer is ungrounded.

    Responsibilities:
    - Run verifier.verify(answer, evidence) via asyncio.to_thread
    - Record verification metrics
    - Retry loop: if not grounded and retries < max, re-run agent with
      feedback, re-fuse, re-verify
    - Handle retry failures with forced ungrounded VerificationResult
    """

    def __init__(
        self,
        verifier: GroundingVerifier,
        ranker: ResultRanker,
        max_retries: int,
        metrics: RAGMetrics | None,
    ) -> None:
        self._verifier = verifier
        self._ranker = ranker
        self._max_retries = max_retries
        self._metrics = metrics

    async def verify_and_retry(self, ctx: QueryContext) -> None:
        """Verify grounding and retry if needed. Populates ctx.verification and ctx.retries."""
        tracer = get_tracer()
        runner = AgentRunner()

        # Initial verification
        with tracer.start_as_current_span("orchestrator.verification") as ver_span:
            ctx.verification = await asyncio.to_thread(
                self._verifier.verify, answer=ctx.answer, evidence=ctx.evidence
            )
            ver_span.set_attribute("is_grounded", ctx.verification.is_grounded)
            ver_span.set_attribute("confidence", ctx.verification.confidence)
            ver_span.set_attribute("evidence_count", len(ctx.evidence))
        if self._metrics:
            self._metrics.record_verification(
                ctx.verification.confidence, ctx.verification.is_grounded
            )

        # Retry loop
        ctx.retries = 0
        while (
            self._verifier.needs_retry(ctx.verification)
            and ctx.retries < self._max_retries
        ):
            ctx.retries += 1
            with tracer.start_as_current_span(
                "orchestrator.retry",
                attributes={"retry_number": ctx.retries},
            ) as retry_span:
                logger.warning(
                    "Grounding failed — confidence=%.3f, is_grounded=%s, "
                    "evidence=%d, retry %d/%d, query=%.100s",
                    ctx.verification.confidence,
                    ctx.verification.is_grounded,
                    len(ctx.evidence),
                    ctx.retries,
                    self._max_retries,
                    ctx.query,
                )
                if self._metrics:
                    self._metrics.record_stage2_retry()

                feedback = ChatMessage.from_user(
                    "Your previous answer was not well-grounded in the evidence. "
                    "Please search again with a broader strategy (try 'graph_hybrid') "
                    "and provide a more evidence-based answer."
                )
                retry_messages = ctx.chat_messages + [
                    ChatMessage.from_assistant(ctx.answer),
                    feedback,
                ]
                try:
                    await runner.rerun(ctx, retry_messages)
                    fused_pack = self._ranker.fuse_evidence(
                        kb_documents=ctx.kb_evidence,
                        web_documents=ctx.web_evidence,
                        query=ctx.query,
                    )
                    ctx.evidence = list(fused_pack.documents)
                    ctx.fused_pack = fused_pack
                    retry_span.set_attribute("retry_evidence", len(ctx.evidence))
                    logger.info(
                        "Retry %d evidence: kb=%d, web=%d, fused=%d",
                        ctx.retries,
                        len(ctx.kb_evidence),
                        len(ctx.web_evidence),
                        len(ctx.evidence),
                    )
                    ctx.verification = await asyncio.to_thread(
                        self._verifier.verify,
                        answer=ctx.answer,
                        evidence=ctx.evidence,
                    )
                    retry_span.set_attribute(
                        "retry_grounded", ctx.verification.is_grounded
                    )
                except Exception as retry_err:
                    retry_span.set_attribute("error", True)
                    logger.error(
                        "Retry %d failed (%s): %s",
                        ctx.retries,
                        type(retry_err).__name__,
                        str(retry_err)[:200],
                    )
                    # Force ungrounded state so synthesis doesn't claim grounded
                    ctx.verification = VerificationResult(
                        is_grounded=False,
                        confidence=0.0,
                        reasoning=f"Retry {ctx.retries} failed: {type(retry_err).__name__}",
                    )
                    break

"""Result persister — synthesizes response, persists run, stores messages."""


import asyncio
import logging
from typing import TYPE_CHECKING, Any

from agentic_ra.agent.context import QueryContext
from agentic_ra.agent.feedback import confidence_to_score
from agentic_ra.agent.synthesis import SynthesizedResponse
from agentic_ra.observability.tracing import get_tracer
from agentic_ra.stores.postgres.models import MessageRole

if TYPE_CHECKING:
    from agentic_ra.agent.feedback import FeedbackAdapter
    from agentic_ra.agent.memory import MemoryStore
    from agentic_ra.agent.synthesis import ResponseSynthesizer
    from agentic_ra.observability.metrics import RAGMetrics
    from agentic_ra.retrieval.strategy_learner import StrategyLearner
    from agentic_ra.stores.postgres.runs import RunStore

logger = logging.getLogger(__name__)


class ResultPersister:
    """Synthesizes response, persists run data, stores messages.

    Responsibilities:
    - Synthesize citations via ResponseSynthesizer
    - Persist pipeline run to RunStore
    - Record strategy outcome to StrategyLearner
    - Submit feedback to Cognee (async background task)
    - Store assistant response in memory
    - Return SynthesizedResponse
    """

    def __init__(
        self,
        synthesizer: ResponseSynthesizer,
        memory: MemoryStore,
        run_store: RunStore | None,
        feedback: FeedbackAdapter | None,
        strategy_learner: StrategyLearner | None,
        metrics: RAGMetrics | None,
    ) -> None:
        self._synthesizer = synthesizer
        self._memory = memory
        self._run_store = run_store
        self._feedback = feedback
        self._strategy_learner = strategy_learner
        self._metrics = metrics
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def persist(self, ctx: QueryContext) -> SynthesizedResponse:
        """Synthesize, persist, and return the final response."""
        tracer = get_tracer()
        assert ctx.verification is not None  # noqa: S101

        # 8. Synthesize citations
        with tracer.start_as_current_span("orchestrator.synthesis") as syn_span:
            response = self._synthesizer.synthesize(
                answer=ctx.answer,
                evidence=ctx.evidence,
                query=ctx.query,
                verification=ctx.verification,
                strategy=ctx.strategy,
            )
            syn_span.set_attribute("citation_count", len(response.citations))
            syn_span.set_attribute("is_grounded", response.is_grounded)
            syn_span.set_attribute("confidence", response.confidence)

        # 9. Persist pipeline run
        if self._run_store:
            await self._run_store.save(
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                query=ctx.query,
                rewritten_query=ctx.query,
                strategy=ctx.strategy,
                answer=response.answer,
                confidence=response.confidence,
                is_grounded=response.is_grounded,
                citation_count=len(response.citations),
                retrieval_ms=ctx.elapsed_ms,
                evidence_count=len(ctx.evidence),
                retries=ctx.retries,
            )

        # 9a. Record strategy outcome (Loop 2)
        if self._strategy_learner:
            try:
                query_embedding = ctx.query_embedding
                if query_embedding is None:
                    query_embedding = self._strategy_learner.embed_query(ctx.query)
                await self._strategy_learner.record(
                    query_embedding=query_embedding,
                    strategy=ctx.strategy,
                    confidence=ctx.verification.confidence,
                    is_grounded=ctx.verification.is_grounded,
                    retrieval_ms=ctx.elapsed_ms,
                    evidence_count=len(ctx.evidence),
                )
            except Exception:
                logger.exception("Failed to record strategy outcome")

        # 9b. Submit feedback to Cognee (async, non-blocking)
        if self._feedback:
            task = asyncio.create_task(
                self._feedback.submit(
                    session_id=str(ctx.session_id),
                    confidence=ctx.verification.confidence,
                    is_grounded=ctx.verification.is_grounded,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            if self._metrics:
                self._metrics.record_feedback(
                    confidence_to_score(ctx.verification.confidence)
                )

        # 10. Store assistant response
        await self._memory.add_message(
            session_id=ctx.session_id,
            role=MessageRole.ASSISTANT,
            content=response.answer,
            metadata={
                "confidence": response.confidence,
                "is_grounded": response.is_grounded,
                "citation_count": len(response.citations),
            },
        )

        return response

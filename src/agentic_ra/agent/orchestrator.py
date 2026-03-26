"""Agent orchestrator — wires memory, retrieval, LLM generation, verification, and synthesis."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from agentic_ra.agent.generator import LLMGenerator
from agentic_ra.agent.memory import MemoryStore
from agentic_ra.agent.synthesis import ResponseSynthesizer, SynthesizedResponse
from agentic_ra.agent.verifier import GroundingVerifier
from agentic_ra.core.errors import OrchestratorError
from agentic_ra.observability.metrics import RAGMetrics
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import RetrievalRouter, RetrievalStrategy
from agentic_ra.stores.postgres.models import MessageRole

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates the full RAG pipeline.

    Flow: memory → query rewrite → retrieval → rank → LLM generate →
    verify → (Stage-2 retry if needed) → synthesize → persist.

    Args:
        memory: MemoryStore for conversation history and long-term facts.
        router: RetrievalRouter for search dispatch.
        ranker: ResultRanker for RRF fusion and dedup.
        generator: LLMGenerator for answer generation and query rewriting.
        verifier: GroundingVerifier for faithfulness checks.
        synthesizer: ResponseSynthesizer for citation assembly.
        metrics: RAGMetrics for observability.
        max_retries: Maximum Stage-2 retrieval retries on verification failure.
    """

    def __init__(
        self,
        memory: MemoryStore,
        router: RetrievalRouter,
        ranker: ResultRanker,
        generator: LLMGenerator,
        verifier: GroundingVerifier,
        synthesizer: ResponseSynthesizer,
        metrics: RAGMetrics | None = None,
        max_retries: int = 1,
    ) -> None:
        self._memory = memory
        self._router = router
        self._ranker = ranker
        self._generator = generator
        self._verifier = verifier
        self._synthesizer = synthesizer
        self._metrics = metrics
        self._max_retries = max_retries

    async def process_query(
        self,
        query: str,
        session_id: uuid.UUID,
        user_id: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        limit: int = 10,
    ) -> SynthesizedResponse:
        """Process a user query through the full RAG pipeline.

        Args:
            query: User query text.
            session_id: Conversation session identifier.
            user_id: User identifier for memory access.
            strategy: Retrieval strategy to use.
            limit: Maximum evidence documents per retrieval.

        Returns:
            Synthesized response with citations and confidence.
        """
        try:
            if self._metrics:
                self._metrics.record_query(strategy.value)

            # 1. Store user message in session memory
            await self._memory.add_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=query,
            )

            # 2. Load conversation history for context
            messages = await self._memory.get_messages(session_id, limit=10)
            history = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages[:-1]  # Exclude the message we just added
            ]

            # 3. Rewrite query for better retrieval
            rewritten_query = await asyncio.to_thread(
                self._generator.rewrite_query, query, history or None
            )

            # 4. Retrieve and rank evidence
            retrieval_start = time.monotonic()
            raw_results = await self._router.retrieve(
                query=rewritten_query,
                strategy=strategy,
                limit=limit,
            )
            evidence_pack = self._ranker.rank(
                results=raw_results,
                query=rewritten_query,
                strategy=strategy.value,
                limit=limit,
            )
            retrieval_ms = (time.monotonic() - retrieval_start) * 1000

            if self._metrics:
                self._metrics.record_retrieval(
                    retrieval_ms, len(evidence_pack.documents), strategy.value
                )

            # 5. Generate answer with LLM
            answer = await asyncio.to_thread(
                self._generator.generate_answer,
                query,
                evidence_pack.documents,
                history or None,
            )

            # 6. Verify grounding
            verification = self._verifier.verify(
                answer=answer,
                evidence=evidence_pack.documents,
            )

            if self._metrics:
                self._metrics.record_verification(
                    verification.confidence, verification.is_grounded
                )

            # 7. Stage-2 retry if verification fails
            retries = 0
            while self._verifier.needs_retry(verification) and retries < self._max_retries:
                retries += 1
                logger.info(
                    "Verification failed (confidence=%.3f), Stage-2 retry %d/%d",
                    verification.confidence,
                    retries,
                    self._max_retries,
                )

                if self._metrics:
                    self._metrics.record_stage2_retry()

                retry_strategy = RetrievalStrategy.HYBRID
                raw_results = await self._router.retrieve(
                    query=rewritten_query,
                    strategy=retry_strategy,
                    limit=limit * 2,
                )
                evidence_pack = self._ranker.rank(
                    results=raw_results,
                    query=rewritten_query,
                    strategy=retry_strategy.value,
                    limit=limit * 2,
                )
                strategy = retry_strategy

                answer = await asyncio.to_thread(
                    self._generator.generate_answer,
                    query,
                    evidence_pack.documents,
                    history or None,
                )
                verification = self._verifier.verify(
                    answer=answer,
                    evidence=evidence_pack.documents,
                )

            # 8. Synthesize final response
            response = self._synthesizer.synthesize(
                answer=answer,
                evidence=evidence_pack.documents,
                query=query,
                verification=verification,
                strategy=strategy.value,
            )

            # 9. Store assistant response in session memory
            await self._memory.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=response.answer,
                metadata={
                    "confidence": response.confidence,
                    "is_grounded": response.is_grounded,
                    "citation_count": len(response.citations),
                    "rewritten_query": rewritten_query,
                },
            )

            return response

        except OrchestratorError:
            raise
        except Exception as e:
            raise OrchestratorError(
                "Query processing failed",
                query=query,
                session_id=str(session_id),
                operation="process_query",
                original_error=str(e),
            ) from e

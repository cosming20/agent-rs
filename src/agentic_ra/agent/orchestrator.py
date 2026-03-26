"""Agent orchestrator — wires memory, retrieval, verification, and synthesis."""

from __future__ import annotations

import logging
import uuid

from haystack import Document

from agentic_ra.agent.memory import MemoryStore
from agentic_ra.agent.synthesis import ResponseSynthesizer, SynthesizedResponse
from agentic_ra.agent.verifier import GroundingVerifier
from agentic_ra.core.errors import OrchestratorError
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import RetrievalRouter, RetrievalStrategy
from agentic_ra.stores.postgres.models import MessageRole

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates the full RAG pipeline: memory → retrieval → verify → synthesize.

    This is the main entry point for processing user queries. It:
    1. Loads conversation history from memory
    2. Dispatches retrieval via the router
    3. Ranks and deduplicates results
    4. Generates an answer (placeholder — LLM integration is Phase 8)
    5. Verifies grounding against evidence
    6. If verification fails, retries with broadened retrieval (Stage 2)
    7. Synthesizes final response with citations

    Args:
        memory: MemoryStore for conversation history and long-term facts.
        router: RetrievalRouter for search dispatch.
        ranker: ResultRanker for RRF fusion and dedup.
        verifier: GroundingVerifier for faithfulness checks.
        synthesizer: ResponseSynthesizer for citation assembly.
        max_retries: Maximum Stage-2 retrieval retries on verification failure.
    """

    def __init__(
        self,
        memory: MemoryStore,
        router: RetrievalRouter,
        ranker: ResultRanker,
        verifier: GroundingVerifier,
        synthesizer: ResponseSynthesizer,
        max_retries: int = 1,
    ) -> None:
        self._memory = memory
        self._router = router
        self._ranker = ranker
        self._verifier = verifier
        self._synthesizer = synthesizer
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
            # 1. Store user message in session memory
            await self._memory.add_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=query,
            )

            # 2. Retrieve and rank evidence
            raw_results = await self._router.retrieve(
                query=query,
                strategy=strategy,
                limit=limit,
            )
            evidence_pack = self._ranker.rank(
                results=raw_results,
                query=query,
                strategy=strategy.value,
                limit=limit,
            )

            # 3. Generate answer (placeholder — LLM integration in Phase 8)
            answer = self._generate_answer(query, evidence_pack.documents)

            # 4. Verify grounding
            verification = self._verifier.verify(
                answer=answer,
                evidence=evidence_pack.documents,
            )

            # 5. Stage-2 retry if verification fails
            retries = 0
            while self._verifier.needs_retry(verification) and retries < self._max_retries:
                retries += 1
                logger.info(
                    "Verification failed, retrying retrieval (attempt %d/%d)",
                    retries,
                    self._max_retries,
                    extra={"confidence": verification.confidence, "query": query},
                )

                # Broaden search with higher limit
                raw_results = await self._router.retrieve(
                    query=query,
                    strategy=RetrievalStrategy.HYBRID,
                    limit=limit * 2,
                )
                evidence_pack = self._ranker.rank(
                    results=raw_results,
                    query=query,
                    strategy="hybrid",
                    limit=limit * 2,
                )

                answer = self._generate_answer(query, evidence_pack.documents)
                verification = self._verifier.verify(
                    answer=answer,
                    evidence=evidence_pack.documents,
                )

            # 6. Synthesize final response
            response = self._synthesizer.synthesize(
                answer=answer,
                evidence=evidence_pack.documents,
                query=query,
                verification=verification,
                strategy=strategy.value,
            )

            # 7. Store assistant response in session memory
            await self._memory.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=response.answer,
                metadata={
                    "confidence": response.confidence,
                    "is_grounded": response.is_grounded,
                    "citation_count": len(response.citations),
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

    def _generate_answer(
        self,
        query: str,
        evidence: tuple[Document, ...],
    ) -> str:
        """Generate an answer from query and evidence.

        Placeholder implementation — returns a summary of evidence content.
        Will be replaced with LLM-based generation in Phase 8 (pipeline assembly).

        Args:
            query: User query text.
            evidence: Ranked evidence documents.

        Returns:
            Generated answer text.
        """
        if not evidence:
            return f"No relevant evidence found for: {query}"

        # Placeholder: concatenate top evidence snippets
        snippets = []
        for doc in evidence[:3]:
            if doc.content:
                snippet = doc.content[:300]
                snippets.append(snippet)

        if not snippets:
            return f"Evidence retrieved but no content available for: {query}"

        return f"Based on the retrieved evidence: {' | '.join(snippets)}"

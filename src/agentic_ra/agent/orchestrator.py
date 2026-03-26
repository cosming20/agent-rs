"""Agent orchestrator — runs Haystack Agent, then hardcoded quality gate."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from agentic_ra.agent.agent import build_agent_with_tools
from agentic_ra.agent.memory import MemoryStore
from agentic_ra.agent.synthesis import ResponseSynthesizer, SynthesizedResponse
from agentic_ra.agent.tools import (
    get_evidence_buffer,
    make_ingest_tool,
    make_remember_tool,
    make_search_knowledge_tool,
    make_search_web_tool,
    reset_evidence_buffer,
)
from agentic_ra.agent.verifier import GroundingVerifier
from agentic_ra.core.errors import OrchestratorError
from agentic_ra.observability.metrics import RAGMetrics
from agentic_ra.stores.postgres.models import MessageRole
from agentic_ra.stores.postgres.runs import RunStore

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Runs the Haystack Agent and applies a hardcoded quality gate.

    Flow:
    1. Load conversation history -> build ChatMessage list
    2. Run Haystack Agent (ReACT tool-calling loop)
    3. Extract answer and evidence from agent result
    4. Verify grounding (hardcoded, agent cannot skip)
    5. If not grounded, re-run agent with feedback (max retries)
    6. Synthesize citations
    7. Persist run + store messages in memory
    """

    def __init__(
        self,
        base_agent: Agent,
        memory: MemoryStore,
        verifier: GroundingVerifier,
        synthesizer: ResponseSynthesizer,
        metrics: RAGMetrics | None = None,
        run_store: RunStore | None = None,
        max_retries: int = 1,
        router: Any = None,
        ranker: Any = None,
        cognee_store: Any = None,
        document_store: Any = None,
    ) -> None:
        self._base_agent = base_agent
        self._memory = memory
        self._verifier = verifier
        self._synthesizer = synthesizer
        self._metrics = metrics
        self._run_store = run_store
        self._max_retries = max_retries
        self._router = router
        self._ranker = ranker
        self._cognee_store = cognee_store
        self._document_store = document_store

    async def process_query(
        self,
        query: str,
        session_id: uuid.UUID,
        user_id: str,
        strategy: str = "hybrid",
        limit: int = 10,
        role: str = "user",
    ) -> SynthesizedResponse:
        """Process a user query through the agent + quality gate pipeline."""
        try:
            if self._metrics:
                self._metrics.record_query(strategy)

            start = time.monotonic()

            # 1. Store user message
            await self._memory.add_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=query,
            )

            # 2. Load conversation history as ChatMessages
            db_messages = await self._memory.get_messages(session_id, limit=10)
            chat_messages: list[ChatMessage] = []
            for msg in db_messages[:-1]:
                if msg.role == MessageRole.USER:
                    chat_messages.append(ChatMessage.from_user(msg.content))
                elif msg.role == MessageRole.ASSISTANT:
                    chat_messages.append(ChatMessage.from_assistant(msg.content))
            # Inject strategy hint into the user message
            hint = (
                f"\n\n[System hint: prefer '{strategy}' search strategy for this query]"
            )
            chat_messages.append(ChatMessage.from_user(query + hint))

            # 3. Build per-request tools scoped to this user's dataset
            dataset = f"user_{user_id}"
            tools_config = [
                make_search_knowledge_tool(
                    self._router,
                    self._ranker,
                    cognee_store=self._cognee_store,
                    document_store=self._document_store,
                    dataset=dataset,
                    role=role,
                ),
                make_search_web_tool(),
                make_remember_tool(self._memory, user_id),
            ]
            if self._cognee_store is not None:
                tools_config.append(
                    make_ingest_tool(self._cognee_store, dataset=dataset)
                )

            agent = build_agent_with_tools(self._base_agent, tools_config)

            # 4. Run Haystack Agent (offloaded to thread — Agent.run is sync)
            reset_evidence_buffer()
            agent_result = await asyncio.to_thread(agent.run, messages=chat_messages)

            # 5. Extract answer and evidence
            answer = self._extract_answer(agent_result)
            evidence: list[Document] = get_evidence_buffer()
            elapsed_ms = (time.monotonic() - start) * 1000

            if self._metrics:
                self._metrics.record_retrieval(elapsed_ms, len(evidence), strategy)

            # 6. Verify grounding (offloaded — verify() calls LLM synchronously)
            verification = await asyncio.to_thread(
                self._verifier.verify, answer=answer, evidence=evidence
            )
            if self._metrics:
                self._metrics.record_verification(
                    verification.confidence, verification.is_grounded
                )

            # 7. Retry if not grounded
            retries = 0
            while (
                self._verifier.needs_retry(verification) and retries < self._max_retries
            ):
                retries += 1
                logger.info(
                    "Grounding failed (confidence=%.3f), retry %d/%d",
                    verification.confidence,
                    retries,
                    self._max_retries,
                )
                if self._metrics:
                    self._metrics.record_stage2_retry()

                feedback = ChatMessage.from_user(
                    "Your previous answer was not well-grounded in the evidence. "
                    "Please search again with a broader strategy (try 'graph_hybrid') "
                    "and provide a more evidence-based answer."
                )
                retry_messages = chat_messages + [
                    ChatMessage.from_assistant(answer),
                    feedback,
                ]
                reset_evidence_buffer()
                agent_result = await asyncio.to_thread(
                    agent.run, messages=retry_messages
                )
                answer = self._extract_answer(agent_result)
                evidence = get_evidence_buffer()
                verification = await asyncio.to_thread(
                    self._verifier.verify, answer=answer, evidence=evidence
                )

            # 8. Synthesize citations
            response = self._synthesizer.synthesize(
                answer=answer,
                evidence=evidence,
                query=query,
                verification=verification,
                strategy=strategy,
            )

            # 9. Persist pipeline run
            if self._run_store:
                await self._run_store.save(
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    rewritten_query=query,
                    strategy=strategy,
                    answer=response.answer,
                    confidence=response.confidence,
                    is_grounded=response.is_grounded,
                    citation_count=len(response.citations),
                    retrieval_ms=elapsed_ms,
                    evidence_count=len(evidence),
                    retries=retries,
                )

            # 10. Store assistant response
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

    @staticmethod
    def _extract_answer(agent_result: dict[str, Any]) -> str:
        """Extract the final text answer from agent result."""
        messages: list[ChatMessage] = agent_result.get("messages", [])
        for msg in reversed(messages):
            if msg.role.value == "assistant" and msg.text and not msg.tool_calls:
                return msg.text
        return "I was unable to generate an answer."

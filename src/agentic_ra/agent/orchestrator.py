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
    _sanitize_user_id,
    make_fetch_url_tool,
    make_ingest_tool,
    make_remember_tool,
    make_search_knowledge_tool,
    make_search_news_tool,
    make_search_web_tool,
)
from agentic_ra.agent.feedback import FeedbackAdapter, confidence_to_score
from agentic_ra.agent.verifier import GroundingVerifier
from agentic_ra.core.config import get_settings
from agentic_ra.core.errors import OrchestratorError
from agentic_ra.retrieval.strategy_learner import StrategyLearner
from agentic_ra.tools.budgets import WebToolBudget
from agentic_ra.tools.mcp_http_client import MCPHttpClient
from agentic_ra.tools.url_safety import URLValidator
from agentic_ra.observability.cost_tracker import get_tracker
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
        brave_client: MCPHttpClient | None = None,
        url_validator: URLValidator | None = None,
        feedback: FeedbackAdapter | None = None,
        strategy_learner: StrategyLearner | None = None,
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
        self._brave_client = brave_client
        self._url_validator = url_validator or URLValidator()
        self._feedback = feedback
        self._strategy_learner = strategy_learner

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
            # Strategy prediction (Loop 2)
            if self._strategy_learner and strategy == "hybrid":
                predicted = await self._strategy_learner.predict(query)
                if predicted != strategy:
                    logger.info(
                        "Strategy learner: %s → %s", strategy, predicted
                    )
                    if self._metrics:
                        self._metrics.record_strategy_prediction(
                            predicted, strategy, was_override=True
                        )
                    strategy = predicted

            # Inject strategy hint into the user message
            hint = (
                f"\n\n[System hint: prefer '{strategy}' search strategy for this query]"
            )
            chat_messages.append(ChatMessage.from_user(query + hint))

            # 3. Build per-request tools scoped to this user's dataset
            dataset = f"user_{_sanitize_user_id(user_id)}"
            kb_evidence: list[Document] = []
            web_evidence: list[Document] = []

            settings = get_settings()
            budget = WebToolBudget(
                max_web_searches=settings.web_budget_max_searches,
                max_news_searches=settings.web_budget_max_news,
                max_fetch_calls=settings.web_budget_max_fetches,
                total_web_timeout_s=settings.web_budget_timeout_s,
                total_fetch_bytes=settings.web_budget_max_bytes,
            )

            tools_config = [
                make_search_knowledge_tool(
                    self._router,
                    self._ranker,
                    cognee_store=self._cognee_store,
                    document_store=self._document_store,
                    dataset=dataset,
                    role=role,
                    evidence_accumulator=kb_evidence,
                    session_id=str(session_id),
                ),
                make_search_web_tool(
                    brave_client=self._brave_client,
                    evidence_accumulator=web_evidence,
                    budget=budget,
                ),
                make_search_news_tool(
                    brave_client=self._brave_client,
                    evidence_accumulator=web_evidence,
                    budget=budget,
                ),
                make_fetch_url_tool(
                    url_validator=self._url_validator,
                    evidence_accumulator=web_evidence,
                    budget=budget,
                    max_length=settings.fetch_max_length,
                    max_bytes=settings.fetch_max_bytes,
                    timeout=settings.fetch_timeout,
                    user_agent=settings.fetch_user_agent,
                ),
                make_remember_tool(self._memory, user_id),
            ]
            if self._cognee_store is not None:
                tools_config.append(
                    make_ingest_tool(self._cognee_store, dataset=dataset)
                )

            agent = build_agent_with_tools(self._base_agent, tools_config)

            # 4. Run Haystack Agent (offloaded to thread — Agent.run is sync)
            #    Haystack's agent snapshot code tries to serialize tool functions
            #    on pipeline errors.  Our closure-based tools can't be serialized,
            #    so we catch SerializationError and fall back to evidence already
            #    accumulated by the tool closures.
            kb_evidence.clear()
            web_evidence.clear()
            agent_result: dict[str, Any] | None = None
            agent_start = time.monotonic()
            try:
                agent_result = await asyncio.to_thread(
                    agent.run, messages=chat_messages
                )
            except Exception as agent_err:
                err_name = type(agent_err).__name__
                logger.warning(
                    "Agent run failed (%s): %s — using accumulated evidence",
                    err_name,
                    str(agent_err)[:200],
                )
            agent_latency_ms = int((time.monotonic() - agent_start) * 1000)

            # 4b. Record agent LLM costs from response metadata
            self._record_agent_costs(
                agent_result, session_id, "agent", agent_latency_ms
            )

            # 5. Extract answer and evidence (accumulated via closure)
            answer = (
                self._extract_answer(agent_result)
                if agent_result is not None
                else "I was unable to generate an answer due to a processing error."
            )

            # 5b. Fuse KB + web evidence
            fused_pack = self._ranker.fuse_evidence(
                kb_documents=kb_evidence,
                web_documents=web_evidence,
                query=query,
            )
            evidence = list(fused_pack.documents)

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
                kb_evidence.clear()
                web_evidence.clear()
                agent_result = await asyncio.to_thread(
                    agent.run, messages=retry_messages
                )
                answer = self._extract_answer(agent_result)
                fused_pack = self._ranker.fuse_evidence(
                    kb_documents=kb_evidence,
                    web_documents=web_evidence,
                    query=query,
                )
                evidence = list(fused_pack.documents)
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

            # 9a. Record strategy outcome (Loop 2)
            if self._strategy_learner:
                try:
                    query_embedding = self._strategy_learner.embed_query(query)
                    await self._strategy_learner.record(
                        query_embedding=query_embedding,
                        strategy=strategy,
                        confidence=verification.confidence,
                        is_grounded=verification.is_grounded,
                        retrieval_ms=elapsed_ms,
                        evidence_count=len(evidence),
                    )
                except Exception:
                    logger.exception("Failed to record strategy outcome")

            # 9b. Submit feedback to Cognee (async, non-blocking)
            if self._feedback:
                asyncio.create_task(
                    self._feedback.submit(
                        session_id=str(session_id),
                        confidence=verification.confidence,
                        is_grounded=verification.is_grounded,
                    )
                )
                if self._metrics:
                    self._metrics.record_feedback(
                        confidence_to_score(verification.confidence)
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

    @staticmethod
    def _record_agent_costs(
        agent_result: dict[str, Any] | None,
        session_id: uuid.UUID,
        purpose: str,
        total_latency_ms: int = 0,
    ) -> None:
        """Extract token usage from Haystack agent result and record to cost tracker."""
        tracker = get_tracker()
        if tracker is None or agent_result is None:
            return

        messages: list[ChatMessage] = agent_result.get("messages", [])
        llm_msgs = [
            m for m in messages
            if hasattr(m, "_meta") and m._meta.get("usage") and m._meta.get("model")
        ]
        n = len(llm_msgs)

        for i, msg in enumerate(llm_msgs):
            meta = msg._meta
            usage = meta["usage"]
            # Split total latency proportionally across LLM calls
            latency = total_latency_ms // n if n > 0 else 0
            tracker.record_call(
                model=meta["model"],
                purpose=purpose,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency,
                session_id=str(session_id),
            )

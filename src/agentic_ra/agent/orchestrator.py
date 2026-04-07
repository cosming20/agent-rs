"""Agent orchestrator — runs Haystack Agent, then hardcoded quality gate."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from opentelemetry import trace

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from agentic_ra.agent.agent import build_agent_with_tools
from agentic_ra.agent.memory import MemoryStore
from agentic_ra.agent.synthesis import ResponseSynthesizer, SynthesizedResponse
from agentic_ra.agent.tools import (
    make_fetch_url_tool,
    make_ingest_tool,
    make_remember_tool,
    make_search_knowledge_tool,
    make_search_news_tool,
    make_search_web_tool,
)
from agentic_ra.agent.feedback import FeedbackAdapter, confidence_to_score
from agentic_ra.agent.verifier import GroundingVerifier, VerificationResult
from agentic_ra.core.config import get_settings
from agentic_ra.core.errors import OrchestratorError
from agentic_ra.retrieval.router import RetrievalStrategy
from agentic_ra.retrieval.strategy_learner import StrategyLearner
from agentic_ra.tools.budgets import WebToolBudget
from agentic_ra.tools.mcp_http_client import MCPHttpClient
from agentic_ra.tools.url_safety import URLValidator
from agentic_ra.observability.cost_tracker import get_tracker
from agentic_ra.observability.metrics import RAGMetrics
from agentic_ra.observability.tracing import get_tracer
from agentic_ra.stores.postgres.models import MessageRole
from agentic_ra.stores.postgres.runs import RunStore

logger = logging.getLogger(__name__)

# Tuning constants for context building
_CONTEXT_MAX_MESSAGES = 4
_CONTEXT_MAX_CHARS_PER_MSG = 150


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
        self._background_tasks: set[asyncio.Task[Any]] = set()

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
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "orchestrator.process_query",
            attributes={
                "query": query[:200],
                "session_id": str(session_id),
                "user_id": user_id,
                "strategy": strategy,
            },
        ) as root_span:
            return await self._process_query_traced(
                query, session_id, user_id, strategy, limit, role, root_span,
            )

    async def _process_query_traced(
        self,
        query: str,
        session_id: uuid.UUID,
        user_id: str,
        strategy: str,
        limit: int,
        role: str,
        root_span: trace.Span,
    ) -> SynthesizedResponse:
        """Inner query processing with tracing spans."""
        tracer = get_tracer()
        try:
            # Validate strategy against enum to prevent prompt injection
            try:
                RetrievalStrategy(strategy)
            except ValueError:
                logger.warning("Invalid strategy %r, falling back to hybrid", strategy)
                strategy = "hybrid"

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
            with tracer.start_as_current_span("orchestrator.load_memory") as mem_span:
                db_messages = await self._memory.get_messages(session_id, limit=10)
                chat_messages: list[ChatMessage] = []
                for msg in db_messages[:-1]:
                    if msg.role == MessageRole.USER:
                        chat_messages.append(ChatMessage.from_user(msg.content))
                    elif msg.role == MessageRole.ASSISTANT:
                        chat_messages.append(ChatMessage.from_assistant(msg.content))
                mem_span.set_attribute("history_messages", len(chat_messages))

                # 2b. Inject long-term memories as context
                try:
                    memories = await self._memory.get_memories(user_id=user_id, limit=20)
                    if memories:
                        memory_lines = [f"- {m.key}: {m.value}" for m in memories]
                        memory_context = (
                            "[User's stored memories]\n"
                            + "\n".join(memory_lines)
                        )
                        chat_messages.insert(0, ChatMessage.from_system(memory_context))
                        logger.info("Injected %d long-term memories", len(memories))
                    mem_span.set_attribute("long_term_memories", len(memories) if memories else 0)
                except Exception:
                    logger.warning("Failed to load long-term memories", exc_info=True)
                    mem_span.set_attribute("long_term_memories", 0)
                    mem_span.set_attribute("memory_load_error", True)

            # Strategy prediction (Loop 2)
            query_embedding: list[float] | None = None
            strategy_reason = ""
            if self._strategy_learner and strategy == "hybrid":
                with tracer.start_as_current_span("orchestrator.strategy_prediction") as strat_span:
                    query_embedding = self._strategy_learner.embed_query(query)
                    prediction = await self._strategy_learner.predict(
                        query, query_embedding=query_embedding
                    )
                    predicted_str = prediction.strategy.value
                    strat_span.set_attribute("original_strategy", strategy)
                    strat_span.set_attribute("predicted_strategy", predicted_str)
                    strat_span.set_attribute("prediction_reason", prediction.reason)
                    strategy_reason = prediction.reason
                    if predicted_str != strategy:
                        logger.info(
                            "Strategy learner: %s → %s (%s)",
                            strategy, predicted_str, prediction.reason,
                        )
                        if self._metrics:
                            self._metrics.record_strategy_prediction(
                                predicted_str, strategy, was_override=True
                            )
                        strategy = predicted_str

            # Failure memory lookup (FB → AT: warn about similar past failures)
            failure_warning = ""
            if self._strategy_learner and query_embedding is not None:
                try:
                    failures = await self._strategy_learner.find_similar_failures(
                        query_embedding, limit=3,
                    )
                    if failures:
                        # FailureRecord.strategy is already validated RetrievalStrategy
                        failed_strategies = {f.strategy.value for f in failures}
                        failure_warning = (
                            f"Warning: similar queries previously failed with "
                            f"strategies: {', '.join(sorted(failed_strategies))}. "
                            f"Try a different strategy or broaden your search."
                        )
                        logger.info(
                            "Failure memory: %d similar failures found (%s)",
                            len(failures), ", ".join(sorted(failed_strategies)),
                        )
                except Exception:
                    logger.warning("Failure memory lookup failed", exc_info=True)

            # Build conversation context for search augmentation (S → C)
            # Filter to USER/ASSISTANT only — exclude system messages (e.g. memories)
            conversation_context = ""
            if chat_messages:
                ctx_parts: list[str] = []
                user_assistant_msgs = [
                    m for m in chat_messages
                    if hasattr(m, "role") and m.role.value in ("user", "assistant")
                ]
                for msg in user_assistant_msgs[-_CONTEXT_MAX_MESSAGES:]:
                    text = msg.text if hasattr(msg, "text") and msg.text else ""
                    if text:
                        ctx_parts.append(text[:_CONTEXT_MAX_CHARS_PER_MSG])
                conversation_context = " | ".join(ctx_parts)

            # Inject strategy hint with reasoning into the user message (FB → AT)
            hint_parts = [f"prefer '{strategy}' search strategy"]
            if strategy_reason:
                hint_parts.append(strategy_reason)
            if failure_warning:
                hint_parts.append(failure_warning)
            hint = "\n\n[System hint: " + ". ".join(hint_parts) + "]"
            chat_messages.append(ChatMessage.from_user(query + hint))

            # 3. Build per-request tools — use configured dataset
            dataset = get_settings().cognee_dataset
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
                    conversation_context=conversation_context,
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
            #    Haystack's auto-tracing covers Agent.run() and tool calls.
            kb_evidence.clear()
            web_evidence.clear()
            agent_result: dict[str, Any] | None = None
            with tracer.start_as_current_span("orchestrator.agent_run") as agent_span:
                agent_start = time.monotonic()
                try:
                    agent_result = await asyncio.to_thread(
                        agent.run, messages=chat_messages
                    )
                except Exception as agent_err:
                    err_name = type(agent_err).__name__
                    agent_span.set_attribute("error", True)
                    agent_span.set_attribute("error.type", err_name)
                    logger.error(
                        "Agent run failed (%s): %s — using accumulated evidence",
                        err_name,
                        str(agent_err)[:200],
                    )
                agent_latency_ms = int((time.monotonic() - agent_start) * 1000)
                agent_span.set_attribute("latency_ms", agent_latency_ms)
                agent_span.set_attribute("success", agent_result is not None)

            # 4b. Record agent LLM costs from response metadata
            self._record_agent_costs(
                agent_result, session_id, "agent", agent_latency_ms
            )

            # 5. Extract answer and evidence (accumulated via closure)
            if agent_result is not None:
                answer = self._extract_answer(agent_result)
            else:
                answer = "I was unable to generate an answer due to a processing error."
                root_span.set_attribute("agent_crashed", True)

            # 5b. Fuse KB + web evidence
            with tracer.start_as_current_span("orchestrator.evidence_fusion") as fuse_span:
                fused_pack = self._ranker.fuse_evidence(
                    kb_documents=kb_evidence,
                    web_documents=web_evidence,
                    query=query,
                )
                evidence = list(fused_pack.documents)
                fuse_span.set_attribute("kb_raw", len(kb_evidence))
                fuse_span.set_attribute("web_raw", len(web_evidence))
                fuse_span.set_attribute("fused", len(evidence))
                fuse_span.set_attribute("strategy", strategy)

            elapsed_ms = int((time.monotonic() - start) * 1000)

            if not evidence:
                logger.warning(
                    "ZERO EVIDENCE after retrieval — "
                    "kb_raw=%d, web_raw=%d, fused=%d, strategy=%s, "
                    "agent_ok=%s, query=%.100s",
                    len(kb_evidence),
                    len(web_evidence),
                    len(fused_pack.documents),
                    strategy,
                    agent_result is not None,
                    query,
                )
            else:
                logger.info(
                    "Evidence collected: kb=%d, web=%d, fused=%d, strategy=%s",
                    len(kb_evidence), len(web_evidence), len(evidence), strategy,
                )

            if self._metrics:
                self._metrics.record_retrieval(elapsed_ms, len(evidence), strategy)

            # 6. Verify grounding (offloaded — verify() calls LLM synchronously)
            with tracer.start_as_current_span("orchestrator.verification") as ver_span:
                verification = await asyncio.to_thread(
                    self._verifier.verify, answer=answer, evidence=evidence
                )
                ver_span.set_attribute("is_grounded", verification.is_grounded)
                ver_span.set_attribute("confidence", verification.confidence)
                ver_span.set_attribute("evidence_count", len(evidence))
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
                with tracer.start_as_current_span(
                    "orchestrator.retry",
                    attributes={"retry_number": retries},
                ) as retry_span:
                    logger.warning(
                        "Grounding failed — confidence=%.3f, is_grounded=%s, "
                        "evidence=%d, retry %d/%d, query=%.100s",
                        verification.confidence,
                        verification.is_grounded,
                        len(evidence),
                        retries,
                        self._max_retries,
                        query,
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
                    try:
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
                        retry_span.set_attribute("retry_evidence", len(evidence))
                        logger.info(
                            "Retry %d evidence: kb=%d, web=%d, fused=%d",
                            retries, len(kb_evidence), len(web_evidence), len(evidence),
                        )
                        verification = await asyncio.to_thread(
                            self._verifier.verify, answer=answer, evidence=evidence
                        )
                        retry_span.set_attribute("retry_grounded", verification.is_grounded)
                    except Exception as retry_err:
                        retry_span.set_attribute("error", True)
                        logger.error(
                            "Retry %d failed (%s): %s",
                            retries, type(retry_err).__name__, str(retry_err)[:200],
                        )
                        # Force ungrounded state so synthesis doesn't claim grounded
                        verification = VerificationResult(
                            is_grounded=False,
                            confidence=0.0,
                            reasoning=f"Retry {retries} failed: {type(retry_err).__name__}",
                        )
                        break

            # 8. Synthesize citations
            with tracer.start_as_current_span("orchestrator.synthesis") as syn_span:
                response = self._synthesizer.synthesize(
                    answer=answer,
                    evidence=evidence,
                    query=query,
                    verification=verification,
                    strategy=strategy,
                )
                syn_span.set_attribute("citation_count", len(response.citations))
                syn_span.set_attribute("is_grounded", response.is_grounded)
                syn_span.set_attribute("confidence", response.confidence)

            # Record final outcome on root span
            root_span.set_attribute("result.confidence", response.confidence)
            root_span.set_attribute("result.is_grounded", response.is_grounded)
            root_span.set_attribute("result.citations", len(response.citations))
            root_span.set_attribute("result.evidence", len(evidence))
            root_span.set_attribute("result.retries", retries)
            root_span.set_attribute("result.strategy", strategy)
            root_span.set_attribute("result.elapsed_ms", elapsed_ms)

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
                    if query_embedding is None:
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
                task = asyncio.create_task(
                    self._feedback.submit(
                        session_id=str(session_id),
                        confidence=verification.confidence,
                        is_grounded=verification.is_grounded,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
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

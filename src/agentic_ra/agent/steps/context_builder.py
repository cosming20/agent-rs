"""Context builder — loads memory, predicts strategy, builds chat messages."""


import logging
from typing import TYPE_CHECKING

from haystack.dataclasses import ChatMessage

from agentic_ra.agent.context import QueryContext
from agentic_ra.observability.tracing import get_tracer
from agentic_ra.stores.postgres.models import MessageRole

if TYPE_CHECKING:
    from agentic_ra.agent.memory import MemoryStore
    from agentic_ra.core.config import Settings
    from agentic_ra.observability.metrics import RAGMetrics
    from agentic_ra.retrieval.strategy_learner import StrategyLearner

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds the chat message context for the agent.

    Responsibilities:
    - Store user message in memory
    - Load conversation history (short-term messages)
    - Inject long-term memories
    - Predict strategy via StrategyLearner (if enabled)
    - Look up failure history
    - Build conversation context string
    - Inject strategy hints into chat messages
    """

    def __init__(
        self,
        memory: MemoryStore,
        strategy_learner: StrategyLearner | None,
        metrics: RAGMetrics | None,
        settings: Settings,
    ) -> None:
        self._memory = memory
        self._strategy_learner = strategy_learner
        self._metrics = metrics
        self._context_max_messages = settings.context_max_messages
        self._context_max_chars_per_msg = settings.context_max_chars_per_msg

    async def build(self, ctx: QueryContext) -> None:
        """Build chat messages and strategy context on the QueryContext."""
        tracer = get_tracer()

        if self._metrics:
            self._metrics.record_query(ctx.strategy)

        # 1. Store user message
        await self._memory.add_message(
            session_id=ctx.session_id,
            role=MessageRole.USER,
            content=ctx.query,
        )

        # 2. Load conversation history as ChatMessages
        with tracer.start_as_current_span("orchestrator.load_memory") as mem_span:
            db_messages = await self._memory.get_messages(ctx.session_id, limit=10)
            chat_messages: list[ChatMessage] = []
            for msg in db_messages[:-1]:
                if msg.role == MessageRole.USER:
                    chat_messages.append(ChatMessage.from_user(msg.content))
                elif msg.role == MessageRole.ASSISTANT:
                    chat_messages.append(ChatMessage.from_assistant(msg.content))
            mem_span.set_attribute("history_messages", len(chat_messages))

            # 2b. Inject long-term memories as context
            try:
                memories = await self._memory.get_memories(
                    user_id=ctx.user_id, limit=20
                )
                if memories:
                    memory_lines = [f"- {m.key}: {m.value}" for m in memories]
                    memory_context = (
                        "[User's stored memories]\n" + "\n".join(memory_lines)
                    )
                    chat_messages.insert(0, ChatMessage.from_system(memory_context))
                    logger.info("Injected %d long-term memories", len(memories))
                mem_span.set_attribute(
                    "long_term_memories", len(memories) if memories else 0
                )
            except Exception:
                logger.warning("Failed to load long-term memories", exc_info=True)
                mem_span.set_attribute("long_term_memories", 0)
                mem_span.set_attribute("memory_load_error", True)

        # Strategy prediction (Loop 2)
        if self._strategy_learner and ctx.strategy == "hybrid":
            with tracer.start_as_current_span(
                "orchestrator.strategy_prediction"
            ) as strat_span:
                ctx.query_embedding = self._strategy_learner.embed_query(ctx.query)
                prediction = await self._strategy_learner.predict(
                    ctx.query, query_embedding=ctx.query_embedding
                )
                predicted_str = prediction.strategy.value
                strat_span.set_attribute("original_strategy", ctx.strategy)
                strat_span.set_attribute("predicted_strategy", predicted_str)
                strat_span.set_attribute("prediction_reason", prediction.reason)
                ctx.strategy_reason = prediction.reason
                if predicted_str != ctx.strategy:
                    logger.info(
                        "Strategy learner: %s -> %s (%s)",
                        ctx.strategy,
                        predicted_str,
                        prediction.reason,
                    )
                    if self._metrics:
                        self._metrics.record_strategy_prediction(
                            predicted_str, ctx.strategy, was_override=True
                        )
                    ctx.strategy = predicted_str

        # Failure memory lookup (FB -> AT: warn about similar past failures)
        if self._strategy_learner and ctx.query_embedding is not None:
            try:
                failures = await self._strategy_learner.find_similar_failures(
                    ctx.query_embedding,
                    limit=3,
                )
                if failures:
                    failed_strategies = {f.strategy.value for f in failures}
                    ctx.failure_warning = (
                        f"Warning: similar queries previously failed with "
                        f"strategies: {', '.join(sorted(failed_strategies))}. "
                        f"Try a different strategy or broaden your search."
                    )
                    logger.info(
                        "Failure memory: %d similar failures found (%s)",
                        len(failures),
                        ", ".join(sorted(failed_strategies)),
                    )
            except Exception:
                logger.warning("Failure memory lookup failed", exc_info=True)

        # Build conversation context for search augmentation (S -> C)
        # Filter to USER/ASSISTANT only — exclude system messages (e.g. memories)
        if chat_messages:
            ctx_parts: list[str] = []
            user_assistant_msgs = [
                m
                for m in chat_messages
                if hasattr(m, "role") and m.role.value in ("user", "assistant")
            ]
            for msg in user_assistant_msgs[-self._context_max_messages :]:
                text = msg.text if hasattr(msg, "text") and msg.text else ""
                if text:
                    ctx_parts.append(text[: self._context_max_chars_per_msg])
            ctx.conversation_context = " | ".join(ctx_parts)

        # Inject strategy hint with reasoning into the user message (FB -> AT)
        hint_parts = [f"prefer '{ctx.strategy}' search strategy"]
        if ctx.strategy_reason:
            hint_parts.append(ctx.strategy_reason)
        if ctx.failure_warning:
            hint_parts.append(ctx.failure_warning)
        hint = "\n\n[System hint: " + ". ".join(hint_parts) + "]"
        chat_messages.append(ChatMessage.from_user(ctx.query + hint))

        ctx.chat_messages = chat_messages

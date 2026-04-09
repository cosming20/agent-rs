"""Agent runner — executes Haystack Agent and extracts answer."""


import asyncio
import logging
import time
import uuid
from typing import Any

from haystack.dataclasses import ChatMessage

from agentic_ra.agent.context import QueryContext
from agentic_ra.observability.cost_tracker import get_tracker
from agentic_ra.observability.tracing import get_tracer

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs the Haystack Agent and extracts the answer.

    Responsibilities:
    - Clear evidence lists
    - Run Haystack Agent via asyncio.to_thread
    - Handle agent failures
    - Record costs
    - Extract answer
    """

    async def run(self, ctx: QueryContext) -> None:
        """Execute the agent and populate ctx.agent_result and ctx.answer."""
        tracer = get_tracer()

        ctx.kb_evidence.clear()
        ctx.web_evidence.clear()

        with tracer.start_as_current_span("orchestrator.agent_run") as agent_span:
            agent_start = time.monotonic()
            try:
                ctx.agent_result = await asyncio.to_thread(
                    ctx.agent.run, messages=ctx.chat_messages  # type: ignore[union-attr]
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
            ctx.agent_latency_ms = int((time.monotonic() - agent_start) * 1000)
            agent_span.set_attribute("latency_ms", ctx.agent_latency_ms)
            agent_span.set_attribute("success", ctx.agent_result is not None)

        # Record agent LLM costs from response metadata
        _record_agent_costs(
            ctx.agent_result, ctx.session_id, "agent", ctx.agent_latency_ms
        )

        # Extract answer
        if ctx.agent_result is not None:
            ctx.answer = _extract_answer(ctx.agent_result)
        else:
            ctx.answer = (
                "I was unable to generate an answer due to a processing error."
            )

    async def rerun(self, ctx: QueryContext, retry_messages: list[ChatMessage]) -> None:
        """Re-run the agent with retry messages (used by GroundingGate)."""
        ctx.kb_evidence.clear()
        ctx.web_evidence.clear()
        ctx.agent_result = await asyncio.to_thread(
            ctx.agent.run, messages=retry_messages  # type: ignore[union-attr]
        )
        ctx.answer = _extract_answer(ctx.agent_result)


def _extract_answer(agent_result: dict[str, Any]) -> str:
    """Extract the final text answer from agent result."""
    messages: list[ChatMessage] = agent_result.get("messages", [])
    for msg in reversed(messages):
        if msg.role.value == "assistant" and msg.text and not msg.tool_calls:
            return msg.text
    return "I was unable to generate an answer."


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
        m
        for m in messages
        if hasattr(m, "_meta") and m._meta.get("usage") and m._meta.get("model")
    ]
    n = len(llm_msgs)

    for msg in llm_msgs:
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

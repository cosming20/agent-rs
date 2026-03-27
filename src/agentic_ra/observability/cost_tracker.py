"""LiteLLM cost tracking callback — writes every LLM call to Postgres + logs."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)

# Dedicated event loop for async DB writes from sync callbacks
_DB_LOOP: asyncio.AbstractEventLoop | None = None
_DB_THREAD: Any = None
_DB_LOCK = __import__("threading").Lock()


def _get_db_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated event loop for async DB writes."""
    global _DB_LOOP, _DB_THREAD  # noqa: PLW0603
    import threading

    with _DB_LOCK:
        if _DB_LOOP is None or _DB_LOOP.is_closed():
            _DB_LOOP = asyncio.new_event_loop()
            _DB_THREAD = threading.Thread(
                target=_DB_LOOP.run_forever, daemon=True, name="cost-tracker-db"
            )
            _DB_THREAD.start()
    return _DB_LOOP


class CostTracker(CustomLogger):
    """LiteLLM callback that logs every LLM call to Postgres and structlog.

    Register at startup:
        tracker = CostTracker(pg_dsn="postgresql+asyncpg://...")
        litellm.callbacks = [tracker]

    Args:
        pg_dsn: Async Postgres DSN for writing llm_calls rows.
    """

    def __init__(self, pg_dsn: str) -> None:
        self._pg_dsn = pg_dsn
        self._engine: Any = None

    def _get_engine(self) -> Any:
        """Lazy-create async SQLAlchemy engine."""
        if self._engine is None:
            from sqlalchemy.ext.asyncio import create_async_engine

            self._engine = create_async_engine(self._pg_dsn, pool_size=2)
        return self._engine

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Called by LiteLLM after a successful completion."""
        self._record(kwargs, response_obj, start_time, end_time, success=True)

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Called by LiteLLM after a failed completion."""
        error = str(kwargs.get("exception", "unknown"))
        self._record(
            kwargs, response_obj, start_time, end_time, success=False, error=error
        )

    def _record(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record an LLM call to Postgres + structured log."""
        model = kwargs.get("model", "unknown")
        metadata = (
            kwargs.get("litellm_params", {}).get("metadata", {})
            if isinstance(kwargs.get("litellm_params"), dict)
            else {}
        )
        purpose = metadata.get("purpose", "unknown")
        cost = kwargs.get("response_cost", 0.0) or 0.0

        usage: dict[str, int] = {}
        if response_obj and hasattr(response_obj, "usage") and response_obj.usage:
            usage = {
                "prompt_tokens": getattr(response_obj.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response_obj.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response_obj.usage, "total_tokens", 0) or 0,
            }

        latency_ms = 0
        if start_time and end_time:
            delta = end_time - start_time
            latency_ms = int(delta.total_seconds() * 1000)

        session_id = metadata.get("session_id")

        # Structured log (always, sync)
        log_fn = logger.info if success else logger.error
        log_fn(
            "llm_call model=%s purpose=%s tokens=%d/%d cost=$%.6f latency=%dms%s",
            model, purpose,
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            cost, latency_ms,
            f" error={error}" if error else "",
        )

        # Postgres insert (async, fire-and-forget)
        row = {
            "id": uuid.uuid4(),
            "model": model,
            "purpose": purpose,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": cost,
            "latency_ms": latency_ms,
            "session_id": uuid.UUID(session_id) if session_id else None,
            "success": success,
            "error": error,
        }
        loop = _get_db_loop()
        asyncio.run_coroutine_threadsafe(self._insert_row(row), loop)

    async def _insert_row(self, row: dict[str, Any]) -> None:
        """Insert a row into llm_calls table."""
        try:
            engine = self._get_engine()
            from sqlalchemy import text

            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO llm_calls "
                        "(id, model, purpose, prompt_tokens, completion_tokens, "
                        "total_tokens, cost_usd, latency_ms, session_id, success, error) "
                        "VALUES (:id, :model, :purpose, :prompt_tokens, :completion_tokens, "
                        ":total_tokens, :cost_usd, :latency_ms, :session_id, :success, :error)"
                    ),
                    row,
                )
        except Exception:
            logger.exception("Failed to insert llm_call row")


    def record_call(
        self,
        model: str,
        purpose: str,
        usage: dict[str, int],
        latency_ms: int = 0,
        session_id: str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Manually record an LLM call (for non-LiteLLM generators like Haystack).

        Use this when the LLM call goes through OpenAI SDK directly
        (not LiteLLM) and the callback doesn't fire.

        Args:
            model: Model name (e.g. "gpt-4o").
            purpose: Cost label ("agent", "verifier", etc.).
            usage: Token usage dict with prompt_tokens, completion_tokens, total_tokens.
            latency_ms: Call latency in milliseconds.
            session_id: Optional session UUID string.
            success: Whether the call succeeded.
            error: Error message if failed.
        """
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Estimate cost using LiteLLM's pricing database
        cost = 0.0
        try:
            # Strip date suffix (gpt-4o-2024-08-06 → gpt-4o) for pricing lookup
            base_model = model.rsplit("-202", 1)[0] if "-202" in model else model
            pt_cost, ct_cost = litellm.cost_per_token(
                model=base_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            cost = pt_cost + ct_cost
        except Exception:
            pass  # Unknown model — record 0 cost

        log_fn = logger.info if success else logger.error
        log_fn(
            "llm_call model=%s purpose=%s tokens=%d/%d cost=$%.6f latency=%dms%s",
            model, purpose, prompt_tokens, completion_tokens,
            cost, latency_ms,
            f" error={error}" if error else "",
        )

        row = {
            "id": uuid.uuid4(),
            "model": model,
            "purpose": purpose,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "latency_ms": latency_ms,
            "session_id": uuid.UUID(session_id) if session_id else None,
            "success": success,
            "error": error,
        }
        loop = _get_db_loop()
        asyncio.run_coroutine_threadsafe(self._insert_row(row), loop)


# Module-level tracker instance (set by setup_cost_tracking)
_tracker: CostTracker | None = None


def get_tracker() -> CostTracker | None:
    """Get the global CostTracker instance, or None if not initialized."""
    return _tracker


def setup_cost_tracking(pg_dsn: str) -> CostTracker:
    """Initialize cost tracking and register with LiteLLM.

    Call once at application startup.

    Args:
        pg_dsn: Async Postgres DSN.

    Returns:
        The CostTracker instance (for testing/inspection).
    """
    global _tracker  # noqa: PLW0603
    tracker = CostTracker(pg_dsn=pg_dsn)
    litellm.callbacks = [tracker]
    _tracker = tracker
    logger.info("LiteLLM cost tracking initialized (Postgres + structured logs)")
    return tracker

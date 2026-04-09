"""Dedicated background event loop for bridging async tool operations to sync.

Haystack's ``Tool`` class expects synchronous callables.  Internal async
operations (Cognee, Postgres, MCP) are bridged via ``run_async`` using
a persistent background event loop thread.
"""


import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_LOOP: asyncio.AbstractEventLoop | None = None
_TOOL_THREAD: threading.Thread | None = None
_TOOL_LOOP_LOCK = threading.Lock()


def _get_tool_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated event loop for tool async operations."""
    global _TOOL_LOOP, _TOOL_THREAD  # noqa: PLW0603
    with _TOOL_LOOP_LOCK:
        if (
            _TOOL_LOOP is None
            or _TOOL_LOOP.is_closed()
            or (_TOOL_THREAD is not None and not _TOOL_THREAD.is_alive())
        ):
            if _TOOL_THREAD is not None and not _TOOL_THREAD.is_alive():
                logger.warning("Tool event loop thread died -- recreating")
            _TOOL_LOOP = asyncio.new_event_loop()
            _TOOL_THREAD = threading.Thread(
                target=_TOOL_LOOP.run_forever, daemon=True, name="tool-event-loop"
            )
            _TOOL_THREAD.start()
    return _TOOL_LOOP


def run_async(coro: Any) -> Any:
    """Bridge async to sync using a dedicated background event loop.

    Always uses the persistent ``_TOOL_LOOP`` thread so that all async
    operations (Cognee search, Neo4j graph queries, Qdrant vector search)
    share a single event loop.  This prevents "Future attached to a
    different loop" errors from Neo4j/asyncio drivers.
    """
    loop = _get_tool_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)


def shutdown_tool_loop() -> None:
    """Gracefully shut down the background event loop (for clean process exit)."""
    global _TOOL_LOOP, _TOOL_THREAD  # noqa: PLW0603
    with _TOOL_LOOP_LOCK:
        if _TOOL_LOOP is not None and not _TOOL_LOOP.is_closed():
            _TOOL_LOOP.call_soon_threadsafe(_TOOL_LOOP.stop)
            if _TOOL_THREAD is not None:
                _TOOL_THREAD.join(timeout=5)
            _TOOL_LOOP.close()
            _TOOL_LOOP = None
            _TOOL_THREAD = None

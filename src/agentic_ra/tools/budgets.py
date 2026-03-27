"""Per-query resource budgets for web tools.

Prevents the ReACT loop from making unbounded web calls.
Each tool closure checks the shared budget before executing.
"""

from __future__ import annotations

import logging
import time

from agentic_ra.core.errors import WebBudgetExhaustedError

logger = logging.getLogger(__name__)


class WebToolBudget:
    """Per-query resource budget for web tools.

    Enforces hard limits on search calls, fetch calls, bytes fetched,
    and total wall-clock time for all web operations in a single query.

    Create one instance per ``process_query()`` call and pass it to
    all web tool closures.

    Args:
        max_web_searches: Maximum ``brave_web_search`` calls.
        max_news_searches: Maximum ``brave_news_search`` calls.
        max_fetch_calls: Maximum ``fetch_url`` calls.
        total_web_timeout_s: Wall-clock seconds budget for all web ops.
        total_fetch_bytes: Total bytes budget across all fetches.
    """

    def __init__(
        self,
        max_web_searches: int = 3,
        max_news_searches: int = 2,
        max_fetch_calls: int = 3,
        total_web_timeout_s: float = 45.0,
        total_fetch_bytes: int = 6_000_000,
    ) -> None:
        self._max_web_searches = max_web_searches
        self._max_news_searches = max_news_searches
        self._max_fetch_calls = max_fetch_calls
        self._total_web_timeout_s = total_web_timeout_s
        self._total_fetch_bytes = total_fetch_bytes

        self._web_searches_used: int = 0
        self._news_searches_used: int = 0
        self._fetch_calls_used: int = 0
        self._bytes_fetched: int = 0
        self._start_time: float = time.monotonic()

    def _check_timeout(self) -> None:
        """Raise if total web timeout has been exceeded."""
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self._total_web_timeout_s:
            raise WebBudgetExhaustedError(
                "Web tool timeout budget exceeded",
                elapsed_s=round(elapsed, 1),
                budget_s=self._total_web_timeout_s,
            )

    def check_web_search(self) -> None:
        """Raise if web search budget is exhausted.

        Raises:
            WebBudgetExhaustedError: If limit reached or timeout exceeded.
        """
        self._check_timeout()
        if self._web_searches_used >= self._max_web_searches:
            raise WebBudgetExhaustedError(
                "Web search budget exhausted",
                used=self._web_searches_used,
                max=self._max_web_searches,
            )

    def check_news_search(self) -> None:
        """Raise if news search budget is exhausted.

        Raises:
            WebBudgetExhaustedError: If limit reached or timeout exceeded.
        """
        self._check_timeout()
        if self._news_searches_used >= self._max_news_searches:
            raise WebBudgetExhaustedError(
                "News search budget exhausted",
                used=self._news_searches_used,
                max=self._max_news_searches,
            )

    def check_fetch(self) -> None:
        """Raise if fetch budget is exhausted (calls or bytes).

        Raises:
            WebBudgetExhaustedError: If limit reached or timeout exceeded.
        """
        self._check_timeout()
        if self._fetch_calls_used >= self._max_fetch_calls:
            raise WebBudgetExhaustedError(
                "Fetch call budget exhausted",
                used=self._fetch_calls_used,
                max=self._max_fetch_calls,
            )
        if self._bytes_fetched >= self._total_fetch_bytes:
            raise WebBudgetExhaustedError(
                "Fetch bytes budget exhausted",
                bytes_fetched=self._bytes_fetched,
                max_bytes=self._total_fetch_bytes,
            )

    def record_web_search(self) -> None:
        """Record one web search call."""
        self._web_searches_used += 1
        logger.debug(
            "Web search budget: %d/%d used",
            self._web_searches_used,
            self._max_web_searches,
        )

    def record_news_search(self) -> None:
        """Record one news search call."""
        self._news_searches_used += 1
        logger.debug(
            "News search budget: %d/%d used",
            self._news_searches_used,
            self._max_news_searches,
        )

    def record_fetch(self, bytes_fetched: int) -> None:
        """Record one fetch call with byte count.

        Args:
            bytes_fetched: Number of bytes downloaded in this fetch.
        """
        self._fetch_calls_used += 1
        self._bytes_fetched += bytes_fetched
        logger.debug(
            "Fetch budget: %d/%d calls, %d/%d bytes",
            self._fetch_calls_used,
            self._max_fetch_calls,
            self._bytes_fetched,
            self._total_fetch_bytes,
        )

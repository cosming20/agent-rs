"""URL fetch tool -- downloads and extracts page content."""


import logging
from typing import Any

from haystack import Document

from agentic_ra.core.errors import ToolError, WebBudgetExhaustedError
from agentic_ra.tools.budgets import WebToolBudget
from agentic_ra.tools.url_safety import URLValidator
from agentic_ra.tools.web import fetch_url

from ._async_bridge import run_async
from ._types import ToolConfig

logger = logging.getLogger(__name__)


def make_fetch_url_tool(
    url_validator: URLValidator,
    evidence_accumulator: list[Document] | None = None,
    budget: WebToolBudget | None = None,
    max_length: int = 5000,
    max_bytes: int = 2_000_000,
    timeout: int = 30,
    user_agent: str = "AgenticRA/1.0",
) -> ToolConfig:
    """Create the ``fetch_url`` tool for the Haystack Agent.

    Fetches a URL, extracts main content via trafilatura, and returns
    clean text. All URLs are SSRF-validated before any network I/O.

    Args:
        url_validator: URLValidator for SSRF protection.
        evidence_accumulator: Shared list to collect web evidence documents.
        budget: Per-query web tool budget.
        max_length: Maximum characters in returned content.
        max_bytes: Maximum bytes to download.
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header value.

    Returns:
        Tool descriptor dict.
    """

    def fetch(url: str) -> dict[str, Any]:
        """Fetch a URL and extract its main content."""
        if budget is not None:
            try:
                budget.check_fetch()
            except WebBudgetExhaustedError as e:
                return {"text": str(e)}

        try:
            content, bytes_downloaded = run_async(
                fetch_url(
                    url=url,
                    validator=url_validator,
                    max_length=max_length,
                    max_bytes=max_bytes,
                    timeout=timeout,
                    user_agent=user_agent,
                )
            )

            if budget is not None:
                budget.record_fetch(bytes_downloaded)

            doc = Document(
                content=content,
                meta={"source": url, "provider": "fetch"},
            )
            if evidence_accumulator is not None:
                evidence_accumulator.append(doc)

            logger.info(
                "fetch_url downloaded %d bytes, extracted %d chars (url=%s)",
                bytes_downloaded,
                len(content),
                url,
            )
            return {"text": content}

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "URL fetch failed",
                url=url,
                operation="fetch_url",
                original_error=str(e),
            ) from e

    return {
        "name": "fetch_url",
        "description": (
            "Fetch a URL and extract its main content as clean text. "
            "Use to get full page content from URLs found in search results."
        ),
        "function": fetch,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and extract content from.",
                },
            },
            "required": ["url"],
        },
    }

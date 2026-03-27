"""URL fetching with main-content extraction.

Pipeline: SSRF validate -> HTTP GET (streaming, size-capped)
-> content-type detect -> main-content extraction -> truncate.

Uses trafilatura for HTML main-content extraction (strips nav, sidebars,
footers, cookie banners). Falls back to markdownify if trafilatura
returns None.
"""

from __future__ import annotations

import json
import logging
import re

import httpx
import trafilatura
from markdownify import markdownify

from agentic_ra.core.errors import SSRFError, WebSearchError
from agentic_ra.tools.url_safety import URLValidator

logger = logging.getLogger(__name__)


async def fetch_url(
    url: str,
    validator: URLValidator,
    max_length: int = 5000,
    max_bytes: int = 2_000_000,
    timeout: int = 30,
    user_agent: str = "AgenticRA/1.0",
) -> tuple[str, int]:
    """Fetch a URL, extract main content, return as clean text.

    Args:
        url: The URL to fetch.
        validator: URLValidator for SSRF protection.
        max_length: Maximum characters in returned content.
        max_bytes: Maximum bytes to download before aborting.
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header value.

    Returns:
        Tuple of (extracted_content, bytes_downloaded).

    Raises:
        SSRFError: If URL fails SSRF validation.
        WebSearchError: If fetch or extraction fails.
    """
    # SSRF validation — must pass before any network I/O
    validated_url = validator.validate(url)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            max_redirects=validator.max_redirects,
            headers={"User-Agent": user_agent},
        ) as client:
            # Streaming download with size cap
            async with client.stream("GET", validated_url) as response:
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                chunks: list[bytes] = []
                total_bytes = 0

                async for chunk in response.aiter_bytes(chunk_size=8192):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        logger.warning(
                            "Fetch aborted: response exceeds %d bytes (url=%s)",
                            max_bytes,
                            url,
                        )
                        break
                    chunks.append(chunk)

                raw_bytes = b"".join(chunks)

    except SSRFError:
        raise
    except httpx.HTTPStatusError as e:
        raise WebSearchError(
            f"HTTP {e.response.status_code} fetching URL",
            url=url,
            status_code=e.response.status_code,
        ) from e
    except httpx.TimeoutException as e:
        raise WebSearchError(
            "URL fetch timed out",
            url=url,
            timeout=timeout,
        ) from e
    except httpx.RequestError as e:
        raise WebSearchError(
            "URL fetch failed",
            url=url,
            original_error=str(e),
        ) from e

    # Decode content
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = raw_bytes.decode("latin-1", errors="replace")

    # Content extraction based on content-type
    extracted = _extract_content(text, content_type, url)

    # Truncate to max_length
    if len(extracted) > max_length:
        extracted = extracted[:max_length] + "\n\n... (truncated)"

    return extracted, total_bytes


def _extract_content(text: str, content_type: str, url: str) -> str:
    """Extract main content from raw text based on content type.

    Args:
        text: Raw response text.
        content_type: HTTP Content-Type header value.
        url: Source URL for logging.

    Returns:
        Cleaned, extracted content string.
    """
    ct_lower = content_type.lower()

    if "json" in ct_lower:
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return text

    if "html" in ct_lower or "<html" in text[:500].lower():
        return _extract_html(text, url)

    # Plain text or other — return as-is with whitespace normalization
    return _normalize_whitespace(text)


def _extract_html(html: str, url: str) -> str:
    """Extract main content from HTML using trafilatura, fallback to markdownify.

    Args:
        html: Raw HTML string.
        url: Source URL (used by trafilatura for context).

    Returns:
        Extracted main content as plain text or markdown.
    """
    # Primary: trafilatura (strips nav, sidebars, footers, cookie banners)
    extracted = trafilatura.extract(
        html,
        url=url,
        include_links=True,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,
    )

    if extracted and len(extracted.strip()) > 50:
        return _normalize_whitespace(extracted)

    # Fallback: markdownify (raw HTML -> markdown, less clean)
    logger.debug("trafilatura returned empty for %s, falling back to markdownify", url)
    md = markdownify(html, heading_style="ATX", strip=["script", "style", "nav"])
    return _normalize_whitespace(md)


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace and blank lines.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (not newlines) to single space
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()

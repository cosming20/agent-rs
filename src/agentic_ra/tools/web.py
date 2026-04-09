"""URL fetching with main-content extraction.

Pipeline: SSRF validate -> HTTP GET (streaming, size-capped)
-> content-type detect -> main-content extraction -> truncate.

Uses trafilatura for HTML main-content extraction (strips nav, sidebars,
footers, cookie banners). Falls back to markdownify if trafilatura
returns None.
"""


import json
import logging
import re
from urllib.parse import urljoin

import httpx
import trafilatura
from markdownify import markdownify

from agentic_ra.core.errors import SSRFError, WebSearchError
from agentic_ra.tools.url_safety import URLValidator, ValidatedURL

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

    Uses IP-pinned URLs to prevent DNS rebinding TOCTOU attacks:
    the validator resolves DNS and returns a ``ValidatedURL`` whose
    ``safe_url`` has the hostname replaced by the resolved IP. The
    original hostname is sent via the ``Host`` header.

    Redirects are followed manually (up to ``validator.max_redirects``)
    so that every redirect target is re-validated against the SSRF
    policy.

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
    # SSRF validation — returns a pinned ValidatedURL (IP-substituted)
    validated = validator.validate(url)

    try:
        raw_bytes, content_type = await _fetch_with_redirect_validation(
            validated=validated,
            validator=validator,
            max_bytes=max_bytes,
            timeout=timeout,
            user_agent=user_agent,
        )
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

    # Decode content — errors="replace" never raises (Issue 4: dead code removed)
    text = raw_bytes.decode("utf-8", errors="replace")

    # Content extraction based on content-type
    extracted = _extract_content(text, content_type, url)

    # Truncate to max_length
    if len(extracted) > max_length:
        extracted = extracted[:max_length] + "\n\n... (truncated)"

    return extracted, len(raw_bytes)


async def _fetch_with_redirect_validation(
    validated: ValidatedURL,
    validator: URLValidator,
    max_bytes: int,
    timeout: int,
    user_agent: str,
) -> tuple[bytes, str]:
    """Fetch URL with manual redirect following and SSRF re-validation.

    Each redirect ``Location`` header is validated through the
    ``URLValidator`` before following, preventing open-redirect SSRF.

    Args:
        validated: The SSRF-validated URL to fetch.
        validator: URLValidator for re-validating redirect targets.
        max_bytes: Maximum bytes to download before aborting.
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header value.

    Returns:
        Tuple of (raw_bytes, content_type_header).

    Raises:
        SSRFError: If a redirect target fails SSRF validation.
        WebSearchError: If too many redirects occur.
        httpx.HTTPStatusError: On non-redirect HTTP errors.
        httpx.TimeoutException: On request timeout.
        httpx.RequestError: On other request failures.
    """
    current = validated
    redirects_remaining = validator.max_redirects

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,  # Manual redirect handling for re-validation
        headers={"User-Agent": user_agent},
    ) as client:
        while True:
            request_headers = {"Host": current.hostname}
            async with client.stream(
                "GET", current.safe_url, headers=request_headers
            ) as response:
                # Handle redirects manually
                if response.is_redirect:
                    if redirects_remaining <= 0:
                        raise WebSearchError(
                            "Too many redirects",
                            url=current.original_url,
                            max_redirects=validator.max_redirects,
                        )
                    redirects_remaining -= 1

                    location = response.headers.get("location")
                    if not location:
                        raise WebSearchError(
                            "Redirect with no Location header",
                            url=current.original_url,
                        )

                    # Resolve relative redirects against the original URL
                    absolute_location = urljoin(current.original_url, location)

                    # Re-validate redirect target against SSRF policy
                    current = validator.validate(absolute_location)
                    continue

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
                            current.original_url,
                        )
                        break
                    chunks.append(chunk)

                return b"".join(chunks), content_type


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
    logger.warning(
        "trafilatura returned empty for %s, falling back to markdownify", url
    )
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

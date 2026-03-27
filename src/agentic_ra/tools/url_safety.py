"""SSRF protection for URL fetching.

Validates URLs against private IPs, blocked schemes, cloud metadata
endpoints, and DNS rebinding attacks. Every URL must pass validation
before any network I/O occurs.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from agentic_ra.core.errors import SSRFError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatedURL:
    """URL that has passed SSRF validation with a pinned resolved IP."""

    original_url: str
    safe_url: str  # URL with hostname replaced by resolved IP
    hostname: str  # Original hostname for Host header


# IP networks that must never be reached by fetch_url
_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    # Loopback
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv6Network("::1/128"),
    # Unspecified
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv6Network("::/128"),
    # Private RFC1918
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    # Link-local (includes cloud metadata 169.254.169.254)
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv6Network("fe80::/10"),
    # Docker default bridge
    ipaddress.IPv4Network("172.17.0.0/16"),
)


class URLValidator:
    """Validates URLs against SSRF attack vectors.

    Enforces scheme, host, IP, and domain restrictions.
    All validation happens before any network I/O.

    Args:
        allowed_schemes: URL schemes permitted. Default http and https only.
        domain_denylist: Domains that are always blocked.
        domain_allowlist: If set, ONLY these domains are allowed.
        max_redirects: Maximum redirect hops permitted.
    """

    def __init__(
        self,
        allowed_schemes: frozenset[str] = frozenset({"http", "https"}),
        domain_denylist: frozenset[str] = frozenset(),
        domain_allowlist: frozenset[str] | None = None,
        max_redirects: int = 3,
    ) -> None:
        self._allowed_schemes = allowed_schemes
        self._domain_denylist = domain_denylist
        self._domain_allowlist = domain_allowlist
        self.max_redirects = max_redirects

    def validate(self, url: str) -> ValidatedURL:
        """Validate a URL for safe fetching. Returns a pinned ValidatedURL.

        Checks scheme, domain allow/deny lists, and resolves DNS to
        verify the target IP is not private or internal. The returned
        ``ValidatedURL`` contains a ``safe_url`` with the hostname replaced
        by the resolved IP so that the caller connects to the *validated*
        address, preventing DNS-rebinding TOCTOU attacks.

        Args:
            url: The URL to validate.

        Returns:
            A ``ValidatedURL`` with the original URL, pinned safe URL, and
            original hostname (for the ``Host`` header).

        Raises:
            SSRFError: If the URL fails any validation check.
        """
        parsed = urlparse(url)

        # Scheme check
        if parsed.scheme not in self._allowed_schemes:
            raise SSRFError(
                f"Blocked URL scheme: {parsed.scheme!r}",
                url=url,
                allowed_schemes=str(self._allowed_schemes),
            )

        # Hostname check
        hostname = parsed.hostname
        if not hostname:
            raise SSRFError("URL has no hostname", url=url)

        hostname_lower = hostname.lower()

        # Domain allowlist (if set, only these are permitted)
        if self._domain_allowlist is not None:
            if hostname_lower not in self._domain_allowlist:
                raise SSRFError(
                    f"Domain not in allowlist: {hostname_lower!r}",
                    url=url,
                )

        # Domain denylist
        if hostname_lower in self._domain_denylist:
            raise SSRFError(
                f"Domain is denied: {hostname_lower!r}",
                url=url,
            )

        # Block 'localhost' string variants
        if hostname_lower in ("localhost", "localhost.localdomain"):
            raise SSRFError("Blocked localhost access", url=url)

        # DNS resolution + IP validation — returns the first safe IP
        resolved_ip = self._validate_resolved_ips(hostname, url)

        # Build safe_url with the resolved IP replacing the hostname
        # so httpx connects to the validated address (no re-resolution).
        port_suffix = f":{parsed.port}" if parsed.port else ""
        # For IPv6 addresses, wrap in brackets for URL netloc
        ip_host = f"[{resolved_ip}]" if ":" in resolved_ip else resolved_ip
        safe_netloc = f"{ip_host}{port_suffix}"
        safe_url = urlunparse(
            (parsed.scheme, safe_netloc, parsed.path, parsed.params, parsed.query, "")
        )

        return ValidatedURL(
            original_url=url,
            safe_url=safe_url,
            hostname=hostname,
        )

    def _validate_resolved_ips(self, hostname: str, url: str) -> str:
        """Resolve hostname and verify all IPs are public.

        Args:
            hostname: The hostname to resolve.
            url: Original URL (for error context).

        Returns:
            The first safe resolved IP address as a string.

        Raises:
            SSRFError: If any resolved IP is private/internal, or if an
                IP string from DNS cannot be parsed (fail-closed).
        """
        # Check if hostname is already an IP literal
        try:
            addr = ipaddress.ip_address(hostname)
            self._check_ip(addr, url)
            return hostname
        except ValueError:
            pass  # Not an IP literal, proceed with DNS resolution

        try:
            addrinfos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except socket.gaierror as e:
            raise SSRFError(
                f"DNS resolution failed for {hostname!r}",
                url=url,
                original_error=str(e),
            ) from e

        if not addrinfos:
            raise SSRFError(f"DNS returned no results for {hostname!r}", url=url)

        first_ip: str | None = None
        for addrinfo in addrinfos:
            ip_str = addrinfo[4][0]
            try:
                addr = ipaddress.ip_address(ip_str)
            except ValueError:
                # Issue 2: unparseable IP from DNS is suspicious — fail closed
                logger.warning(
                    "Unparseable IP %r from DNS for %r — blocking (url=%s)",
                    ip_str,
                    hostname,
                    url,
                )
                raise SSRFError(
                    f"DNS returned unparseable IP {ip_str!r} for {hostname!r}",
                    url=url,
                )
            self._check_ip(addr, url)
            if first_ip is None:
                first_ip = ip_str

        if first_ip is None:
            raise SSRFError(f"DNS returned no usable IPs for {hostname!r}", url=url)

        return first_ip

    @staticmethod
    def _check_ip(
        addr: ipaddress.IPv4Address | ipaddress.IPv6Address, url: str
    ) -> None:
        """Check a single IP address against blocked networks.

        Args:
            addr: The IP address to check.
            url: Original URL (for error context).

        Raises:
            SSRFError: If the IP falls within a blocked network.
        """
        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise SSRFError(
                    f"Blocked private/internal IP: {addr}",
                    url=url,
                    network=str(network),
                )

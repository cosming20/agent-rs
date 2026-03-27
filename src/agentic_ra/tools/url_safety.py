"""SSRF protection for URL fetching.

Validates URLs against private IPs, blocked schemes, cloud metadata
endpoints, and DNS rebinding attacks. Every URL must pass validation
before any network I/O occurs.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from agentic_ra.core.errors import SSRFError

logger = logging.getLogger(__name__)

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

    def validate(self, url: str) -> str:
        """Validate a URL for safe fetching. Returns normalized URL.

        Checks scheme, domain allow/deny lists, and resolves DNS to
        verify the target IP is not private or internal.

        Args:
            url: The URL to validate.

        Returns:
            The normalized URL string.

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

        # DNS resolution + IP validation
        self._validate_resolved_ips(hostname, url)

        return url

    def _validate_resolved_ips(self, hostname: str, url: str) -> None:
        """Resolve hostname and verify all IPs are public.

        Args:
            hostname: The hostname to resolve.
            url: Original URL (for error context).

        Raises:
            SSRFError: If any resolved IP is private/internal.
        """
        # Check if hostname is already an IP literal
        try:
            addr = ipaddress.ip_address(hostname)
            self._check_ip(addr, url)
            return
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

        for addrinfo in addrinfos:
            ip_str = addrinfo[4][0]
            try:
                addr = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            self._check_ip(addr, url)

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

"""
URL normalization and validation for the web crawler.

Responsibilities:
- Resolve relative URLs against a base URL
- Normalize scheme/host to lowercase
- Remove URL fragments
- Reject unsupported schemes (only http/https allowed)
- Reject private/local IP addresses (SSRF prevention)
- Extract domain from a URL (stripping www.)
"""

from __future__ import annotations

from urllib.parse import urljoin, urlparse, urlunparse
import ipaddress
import re
import socket


# Allowed URL schemes.
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Rejected hostnames.
_BLOCKED_HOSTS = frozenset({
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
})


def normalize_url(url: str, base_url: str | None = None) -> str | None:
    """
    Normalize a URL for crawling.

    - Resolves relative URLs against *base_url*.
    - Lowercases scheme and host.
    - Removes fragments.
    - Preserves path and query string.
    - Returns ``None`` for invalid or unsupported URLs.

    Parameters
    ----------
    url : str
        The URL to normalize.
    base_url : str | None
        Base URL for resolving relative links.

    Returns
    -------
    str | None
        Normalized URL or None if invalid.
    """
    url = url.strip()
    if not url:
        return None

    # Resolve relative URLs.
    if base_url:
        url = urljoin(base_url, url)

    parsed = urlparse(url)

    # Validate scheme.
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return None

    # Validate host.
    hostname = parsed.hostname
    if not hostname:
        return None

    hostname_lower = hostname.lower()

    # Block local/private hosts.
    if hostname_lower in _BLOCKED_HOSTS:
        return None

    if _is_private_ip(hostname_lower):
        return None

    # Rebuild URL without fragment.
    normalized = urlunparse((
        scheme,
        parsed.netloc.lower(),  # lowercase host + port
        parsed.path or "/",
        parsed.params,
        parsed.query,
        "",  # remove fragment
    ))

    return normalized


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid for crawling."""
    return normalize_url(url) is not None


def extract_domain(url: str) -> str:
    """
    Extract the domain from a URL, stripping 'www.' prefix.

    Consistent with DocumentRepository._extract_domain.
    """
    hostname = urlparse(url).hostname or ""
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname.lower()


def is_same_domain(url: str, seed_url: str) -> bool:
    """
    Check if *url* belongs to the same domain as *seed_url*.

    Both URLs are compared after stripping 'www.' prefix.
    """
    return extract_domain(url) == extract_domain(seed_url)


def _is_private_ip(hostname: str) -> bool:
    """
    Check if a hostname resolves to a private or reserved IP.

    Only checks if the hostname looks like an IP address.
    Does NOT perform DNS resolution (to avoid blocking).
    """
    try:
        addr = ipaddress.ip_address(hostname)
        return addr.is_private or addr.is_reserved or addr.is_loopback
    except ValueError:
        # Not an IP address — it's a hostname, which is fine.
        return False

"""
URL validation and SSRF protection.

Blocks requests to internal networks, localhost, cloud metadata endpoints,
and non-HTTP schemes.
"""
import ipaddress
import socket
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_BLOCKED_HOSTS = {
    'localhost', '0.0.0.0', 'metadata.google.internal',
    'metadata.google', 'metadata', 'kubernetes.default',
}

_BLOCKED_SCHEMES = {'file', 'ftp', 'gopher', 'data', 'javascript'}


def validate_url(url: str) -> str:
    """
    Validate a URL for safe external fetching.

    Raises ValueError if the URL targets internal resources.
    Returns the validated URL.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL is required")

    parsed = urlparse(url.strip())

    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a hostname")

    hostname_lower = hostname.lower()
    if hostname_lower in _BLOCKED_HOSTS:
        raise ValueError("URLs targeting localhost or internal hosts are not allowed")

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("URLs targeting private/internal IP addresses are not allowed")
    except ValueError as e:
        if "not allowed" in str(e):
            raise
        # Not an IP address — resolve and check
        try:
            resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for family, _, _, _, sockaddr in resolved:
                addr = sockaddr[0]
                ip = ipaddress.ip_address(addr)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise ValueError(f"URL resolves to private IP address")
        except socket.gaierror:
            pass  # DNS resolution failed — let the fetcher handle it
        except ValueError as ve:
            if "private" in str(ve).lower() or "not allowed" in str(ve).lower():
                raise

    # Block cloud metadata endpoints
    if '169.254.169.254' in url or 'metadata.google' in url:
        raise ValueError("URLs targeting cloud metadata endpoints are not allowed")

    return url

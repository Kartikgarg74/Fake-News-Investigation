"""SSRF protection — validate external URLs before fetching.

Rejects:
- Non-http/https schemes (file://, ftp://, data://, etc.)
- Private/reserved IP ranges (RFC 1918, loopback, link-local, CGNAT, etc.)
- IPv6 loopback and ULA ranges

DNS is resolved first; the resolved IP is then checked against the blocklist
so that DNS rebinding attacks (hostname -> private IP) are caught.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from typing import Optional

# Schemes that are acceptable for user-supplied media URLs
_ALLOWED_SCHEMES = {"http", "https"}

# RFC 1918 + special-purpose ranges that must never be contacted
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("10.0.0.0/8"),        # RFC 1918 private
    ipaddress.ip_network("172.16.0.0/12"),     # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"),    # RFC 1918 private
    ipaddress.ip_network("169.254.0.0/16"),    # link-local / APIPA
    ipaddress.ip_network("100.64.0.0/10"),     # CGNAT (RFC 6598)
    ipaddress.ip_network("192.0.2.0/24"),      # TEST-NET-1 (RFC 5737)
    ipaddress.ip_network("198.51.100.0/24"),   # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),    # TEST-NET-3
    ipaddress.ip_network("0.0.0.0/8"),         # "this" network
    ipaddress.ip_network("255.255.255.255/32"),# broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),           # loopback
    ipaddress.ip_network("fc00::/7"),          # ULA (RFC 4193)
    ipaddress.ip_network("fe80::/10"),         # link-local
    ipaddress.ip_network("::ffff:0:0/96"),     # IPv4-mapped
]


def _is_private_ip(addr: str) -> bool:
    """Return True if addr is in a blocked/private range."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        # Unparseable — block it to be safe
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS)


def validate_url(url: str) -> Optional[str]:
    """Validate a user-supplied URL for SSRF safety.

    Returns None if the URL is safe, or an error string describing why it
    was rejected. The caller should treat any non-None return as a blocked
    request.

    DNS resolution is performed here so that hostnames that resolve to
    private IPs (a common SSRF vector) are caught before the connection
    is opened.
    """
    if not url or not isinstance(url, str):
        return "empty or non-string URL"

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        return f"URL parse error: {exc}"

    # 1. Scheme check
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return f"scheme '{scheme}' is not allowed (only http/https)"

    # 2. Hostname must be present
    host = parsed.hostname
    if not host:
        return "URL has no hostname"

    # 3. Resolve DNS and check every returned address
    try:
        # getaddrinfo returns (family, type, proto, canonname, sockaddr)
        # sockaddr[0] is the IP string for both AF_INET and AF_INET6
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return f"DNS resolution failed for '{host}': {exc}"

    if not results:
        return f"DNS returned no results for '{host}'"

    for family, _type, _proto, _canon, sockaddr in results:
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            return (
                f"host '{host}' resolves to private/reserved address '{ip_str}'"
            )

    return None  # URL is safe

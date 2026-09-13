# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared paths, constants, and socket framing for Muse-style auth.

Both the daemon (vault + sentinel + network boundary) and the
agent-side client import from here; this module must not import any
other ``muse_auth`` module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home

_SERVICE_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")


def valid_service_name(service: Any) -> bool:
    """Return whether *service* is a safe connector service name.

    Service names become vault file names, so anything that could
    traverse directories (``..``, ``/``, absolute paths) or smuggle a
    trailing newline is rejected.  ``fullmatch`` (not ``match``) is
    used so ``"gmail\\n"`` cannot slip through the ``$`` end-anchor.

    Args:
        service: Candidate service name (any type; non-strings fail).

    Returns:
        True for lowercase ``[a-z0-9_-]`` strings of at most 64 chars.
    """
    return isinstance(service, str) and _SERVICE_NAME_RE.fullmatch(service) is not None

# Hosts each connector service is allowed to reach.  Mirrors Muse's
# per-worker credential/host ACLs: a Drive surrogate cannot be spent
# against the Gmail API.  Hostnames only (ports are ignored).
SERVICE_HOSTS: dict[str, tuple[str, ...]] = {
    "gmail": ("gmail.googleapis.com", "www.googleapis.com"),
    "google_drive": ("www.googleapis.com",),
    "google_calendar": ("www.googleapis.com",),
    "google_docs": ("docs.googleapis.com", "www.googleapis.com"),
    "google_sheets": ("sheets.googleapis.com", "www.googleapis.com"),
    "googlechat": ("chat.googleapis.com",),
    "notion": ("api.notion.com",),
    "github": ("api.github.com",),
    "slack": ("slack.com", "files.slack.com"),
    "brave_search": ("api.search.brave.com",),
    "discord": ("discord.com",),
    "govee": ("openapi.api.govee.com",),
    # Home Assistant is always self-hosted; an ntfy access token and a
    # Firecrawl API key each belong to exactly one server (the public
    # cloud OR a self-hosted instance).  These services have NO built-in
    # host: the credential is bound to the one origin enrolled with it,
    # so a self-hosted key can never be spent against the cloud service
    # (or vice versa) — strict origin binding.
    "homeassistant": (),
    "ntfy": (),
    "firecrawl": (),
}

# Services that enroll one vault entry per workspace/account; a
# ``<root>-<workspace>`` service name inherits the root's hosts.
_WORKSPACE_SERVICE_ROOTS = ("slack",)


def builtin_hosts(service: str) -> tuple[str, ...]:
    """Return the built-in host allowlist for *service*.

    Exact :data:`SERVICE_HOSTS` entries win; otherwise a workspace-keyed
    name such as ``slack-myteam`` inherits its root service's hosts.

    Args:
        service: Connector service name.

    Returns:
        Tuple of allowed hostnames (possibly empty).
    """
    hosts = SERVICE_HOSTS.get(service)
    if hosts is not None:
        return hosts
    root = service.split("-", 1)[0]
    if root in _WORKSPACE_SERVICE_ROOTS:
        return SERVICE_HOSTS.get(root, ())
    return ()


_HEADER_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{0,63}")

# Headers a vault credential may never occupy: hop-by-hop and
# framing headers would corrupt the boundary request itself.
_FORBIDDEN_CREDENTIAL_HEADERS = frozenset(
    {"host", "connection", "keep-alive", "transfer-encoding", "content-length", "content-type"}
)


def valid_credential_header(name: Any) -> bool:
    """Return whether *name* may carry a real credential at the boundary.

    Used for ``{"kind": "header"}`` vault credentials (e.g. Brave's
    ``X-Subscription-Token``): the name must be a syntactically valid
    HTTP header token and must not be a hop-by-hop or framing header.

    Args:
        name: Candidate header name (any type; non-strings fail).

    Returns:
        True when the header may carry the credential.
    """
    return (
        isinstance(name, str)
        and _HEADER_NAME_RE.fullmatch(name) is not None
        and name.lower() not in _FORBIDDEN_CREDENTIAL_HEADERS
    )


# Visible-ASCII runs separated by single spaces: covers bearer tokens
# and scheme-prefixed values like ``Bot <token>``, while rejecting
# control characters (header injection) and leading/trailing/duplicate
# whitespace (which some HTTP stacks reflect verbatim into errors).
_CREDENTIAL_VALUE_RE = re.compile(r"[\x21-\x7e]+( [\x21-\x7e]+)*")


def valid_credential_value(value: Any) -> bool:
    """Return whether *value* may be sent as a credential header value.

    Args:
        value: Candidate token/header value (any type; non-strings fail).

    Returns:
        True when the value is a safe, canonical header value.
    """
    return isinstance(value, str) and _CREDENTIAL_VALUE_RE.fullmatch(value) is not None


SURROGATE_PREFIX = "muse-sgt."

# Daemon wire-protocol version.  Bumped whenever the daemon gains
# semantics an older daemon would silently mishandle (v2: header-kind
# credentials, enrollment hosts, service-aware action classes; v3:
# consent-scoped insecure enrollment hosts, which a v2 daemon would
# silently drop from ``store_credentials`` and then deny every plain-
# HTTP Home Assistant request).  The client restarts a running daemon
# whose ``status`` reports an older protocol, so a detached pre-upgrade
# daemon cannot serve new clients.
PROTOCOL_VERSION = 3

# One JSON object per line; requests carrying request/response bodies
# are base64-encoded, so cap the frame to keep the daemon safe from
# unbounded allocations (64 MiB of base64 ~ 48 MiB of payload).
MAX_FRAME_BYTES = 64 * 1024 * 1024

READ_METHODS = ("GET", "HEAD", "OPTIONS")


def muse_auth_enabled() -> bool:
    """Return True when Muse-style auth is switched on via ``KISS_MUSE_AUTH``.

    Returns:
        True when the env var is a truthy string ("1", "true", "yes", "on").
    """
    return os.environ.get("KISS_MUSE_AUTH", "").strip().lower() in ("1", "true", "yes", "on")


def muse_auth_dir() -> Path:
    """Return the Muse-auth state directory, honoring ``KISS_HOME``.

    Returns:
        Path to ``$KISS_HOME/muse_auth``.
    """
    return kiss_home() / "muse_auth"


def socket_path() -> Path:
    """Return the daemon's Unix socket path.

    Unix socket paths are limited to ~104 bytes; when the natural
    location under ``KISS_HOME`` is too long (deep pytest temp dirs),
    fall back to a per-user, per-home path in ``/tmp`` derived from a
    hash of ``KISS_HOME`` so client and daemon always agree.

    Returns:
        Path the daemon binds and clients connect to.
    """
    natural = muse_auth_dir() / "authd.sock"
    if len(str(natural)) <= 90:
        return natural
    digest = hashlib.sha256(str(kiss_home()).encode()).hexdigest()[:12]
    return Path(f"/tmp/kiss-muse-{os.getuid()}-{digest}.sock")


def canonical_host(host: str) -> str:
    """Return the canonical form of a hostname for allowlist matching.

    Lowercases and strips exactly one trailing DNS root dot:
    ``Example.COM.`` and ``example.com`` are the same authority, and
    rejecting the fully-qualified spelling would only produce false
    denials.  Only one dot is removed — ``example.com..`` contains an
    empty DNS label, so it canonicalizes to the still-malformed
    ``example.com.`` and fails :func:`valid_hostname` instead of
    collapsing into a valid name.

    Args:
        host: Hostname or IP literal.

    Returns:
        The canonical hostname.
    """
    return host.lower().removesuffix(".")


def canonical_host_entry(entry: str) -> str:
    """Canonicalize an allowlist entry that may carry a ``:port`` suffix.

    Splits any ``host:port`` / ``[ipv6]:port`` suffix, canonicalizes the
    host part (lowercase, drop a trailing DNS dot), and reassembles, so
    a fully-qualified ``localhost.:8123`` becomes ``localhost:8123``.

    Args:
        entry: A bare host or a ``host:port`` allowlist entry.

    Returns:
        The canonical entry.
    """
    lowered = entry.strip().lower()
    # The bracket alternative accepts any IPv6 spelling — IPv4-mapped
    # (``::ffff:127.0.0.1``) and zone/scoped literals with RFC-6874
    # ZoneIDs (``fe80::1%25eth-0``); the daemon validates the bracket
    # contents as a real IP before enrolling it.
    match = re.fullmatch(r"(?P<host>\[[^\]]+\]|[^:]+):(?P<port>[0-9]{1,5})", lowered)
    if not match:
        return canonical_host(lowered)
    # Normalize the port to its integer form so a leading-zero entry
    # (``h:00080``) matches a request whose parsed port is ``80``.
    port = int(match.group("port"))
    host = match.group("host")
    if host.startswith("["):
        return f"{host}:{port}"
    return f"{canonical_host(host)}:{port}"


def host_port_entry(host: str, port: int) -> str:
    """Return the allowlist entry that pins a host to one port.

    Args:
        host: Canonical hostname or IP literal.
        port: TCP port.

    Returns:
        ``host:port``, bracketing IPv6 literals (``[::1]:8080``).
    """
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def url_origin(url_scheme: str, url_host: str | None, url_port: int | None) -> tuple[str, int]:
    """Return the effective (host, port) origin of parsed URL parts.

    Args:
        url_scheme: URL scheme (``http``/``https``).
        url_host: Parsed hostname (may be None).
        url_port: Parsed explicit port (may be None).

    Returns:
        ``(canonical_host, effective_port)`` with the scheme default
        port applied when none is explicit.
    """
    host = canonical_host(url_host or "")
    port = url_port if url_port is not None else (443 if url_scheme == "https" else 80)
    return host, port


def _safe_port(parsed: Any) -> int | None:
    """Return a parsed URL's port, treating a malformed port as absent.

    ``urllib.parse.ParseResult.port`` raises ``ValueError`` for a
    non-numeric or out-of-range port; callers on the enrollment path
    validate the URL up front with :func:`valid_http_url`, but this
    keeps the origin helpers total for any already-stored base URL.

    Args:
        parsed: A ``urlparse`` result.

    Returns:
        The port, or None when absent or malformed.
    """
    try:
        port: int | None = parsed.port
    except ValueError:
        return None
    return port


_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _is_ip_literal(candidate: str) -> bool:
    """Return whether *candidate* is a valid IPv4/IPv6 literal.

    Args:
        candidate: Host string (no brackets, may carry an IPv6 zone id).

    Returns:
        True for a valid IP address literal.
    """
    try:
        import ipaddress

        ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return True


def valid_hostname(host: str) -> bool:
    """Return whether *host* is a syntactically valid hostname or IP literal.

    Hostnames are validated per DNS label (each 1-63 chars, no empty
    labels), so a malformed ``bad..example`` is rejected; IPv4/IPv6
    literals (including zone/scoped forms) are delegated to
    :mod:`ipaddress`.

    Args:
        host: Candidate host (canonicalized before checking).

    Returns:
        True for RFC-1123 hostnames and IPv4/IPv6 literals.
    """
    candidate = canonical_host(host)
    if candidate and len(candidate) <= 253 and all(
        _LABEL_RE.fullmatch(label) for label in candidate.split(".")
    ):
        return True
    return _is_ip_literal(candidate)


def valid_http_url(url: str) -> bool:
    """Return whether *url* is a usable ``http(s)://`` URL.

    Requires an ``http``/``https`` scheme, a valid hostname/IP literal,
    and a well-formed port; used to reject a base URL before any
    credential state changes (a malformed host or port must not
    partially migrate a vault).

    Args:
        url: Candidate base URL.

    Returns:
        True when the URL is safe to enroll against.
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in ("http", "https")
        and parsed.hostname is not None
        and valid_hostname(parsed.hostname)
        # Userinfo makes the HTTP stack derive Basic auth that clobbers
        # the swapped credential header, and the password would be
        # persisted as "non-secret" metadata — reject it up front.
        and not parsed.username
        and not parsed.password
    )


def url_origin_entry(url: str) -> str:
    """Return the port-pinned allowlist entry for a base URL.

    Self-hosted connector enrollments (Home Assistant, ntfy, Firecrawl)
    bind their credential to one origin: the same hostname on another
    port is a different server.

    Args:
        url: Base URL the user configured.

    Returns:
        ``host:port``, or ``""`` when the URL has no host.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host, port = url_origin(parsed.scheme, parsed.hostname, _safe_port(parsed))
    return host_port_entry(host, port) if host else ""


def is_loopback_host(host: str) -> bool:
    """Return whether *host* is a loopback destination.

    Args:
        host: Hostname or IP literal (lowercase).

    Returns:
        True for ``localhost`` and loopback IP addresses.
    """
    if host == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def action_class(method: str) -> str:
    """Classify an HTTP method into Muse's read/write action classes.

    Args:
        method: HTTP method name (any case).

    Returns:
        ``"read"`` for GET/HEAD/OPTIONS, ``"write"`` otherwise.
    """
    return "read" if method.upper() in READ_METHODS else "write"


# Slack's Web API is RPC over POST, so the HTTP method says nothing
# about the action.  These API methods only read workspace state; any
# method not listed here classifies as a write (safe default).
_SLACK_READ_API_METHODS = frozenset(
    {
        "auth.test",
        "bots.info",
        "conversations.history",
        "conversations.info",
        "conversations.list",
        "conversations.members",
        "conversations.replies",
        "emoji.list",
        "files.info",
        "files.list",
        "reactions.get",
        "search.messages",
        "team.info",
        "usergroups.list",
        "users.info",
        "users.list",
        "users.lookupByEmail",
    }
)

# Firecrawl endpoints that only retrieve data (POST bodies carry the
# query).  Starting or cancelling a crawl job stays a write.
_FIRECRAWL_READ_PATHS = ("/v2/scrape", "/v2/map", "/v2/search")

# Govee's state query is a POST whose body only names the device;
# /device/control (the actual actuation) stays a write.
_GOVEE_READ_PATHS = ("/device/state",)


def request_action(service: str, method: str, path: str) -> str:
    """Classify a concrete request into Muse's read/write action classes.

    Most REST connectors are classified by HTTP method via
    :func:`action_class`.  RPC-style APIs need service-specific rules:
    Slack sends every call as POST (the API method name is the last URL
    path segment), Firecrawl's retrieval endpoints take POST bodies,
    and Govee's device-state query is a POST naming the device.

    Args:
        service: Connector service name the surrogate is bound to.
        method: HTTP method of the request.
        path: URL path of the effective (post-normalization) request.

    Returns:
        ``"read"`` or ``"write"``.
    """
    if service == "slack" or service.startswith("slack-"):
        api_method = path.rstrip("/").rsplit("/", 1)[-1]
        return "read" if api_method in _SLACK_READ_API_METHODS else "write"
    if service == "firecrawl":
        if method.upper() in READ_METHODS:
            return "read"
        # Suffix match keeps self-hosted instances behind a reverse
        # proxy (e.g. /proxy/firecrawl/v2/scrape) classified as reads.
        if method.upper() == "POST" and path.rstrip("/").endswith(_FIRECRAWL_READ_PATHS):
            return "read"
        return "write"
    if service == "govee":
        if method.upper() == "POST" and path.rstrip("/").endswith(_GOVEE_READ_PATHS):
            return "read"
        return action_class(method)
    if service == "discord":
        # The typing indicator is an ephemeral, harmless POST; treating
        # it as a write would burn one-shot write grants before the
        # actual message send they were meant for.
        if method.upper() == "POST" and path.rstrip("/").endswith("/typing"):
            return "read"
        return action_class(method)
    return action_class(method)


def send_frame(sock: socket.socket, obj: dict[str, Any]) -> None:
    """Send one newline-terminated JSON frame over *sock*.

    Args:
        sock: Connected Unix socket.
        obj: JSON-serializable payload.
    """
    sock.sendall(json.dumps(obj, separators=(",", ":")).encode() + b"\n")


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    """Receive one newline-terminated JSON frame from *sock*.

    Args:
        sock: Connected Unix socket.

    Returns:
        The decoded JSON object.

    Raises:
        ConnectionError: When the peer closes before a full frame arrives.
        ValueError: When the frame exceeds :data:`MAX_FRAME_BYTES`.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(1 << 16)
        if not chunk:
            raise ConnectionError("muse-auth peer closed the connection mid-frame")
        total += len(chunk)
        if total > MAX_FRAME_BYTES:
            raise ValueError("muse-auth frame exceeds the 64 MiB limit")
        chunks.append(chunk)
        if chunk.endswith(b"\n"):
            return dict(json.loads(b"".join(chunks).decode()))

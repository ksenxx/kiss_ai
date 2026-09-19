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
    # Cloud messaging APIs with one fixed endpoint host.
    "twitch": ("api.twitch.tv",),
    "zalo": ("openapi.zalo.me",),
    "line": ("api.line.me",),
    # Self-hosted messaging servers: like Home Assistant, the credential
    # is bound to the one origin enrolled with it.
    "mattermost": (),
    "nextcloud": (),
    "bluebubbles": (),
    "synology": (),
    # Token-exchange connectors with one fixed endpoint host.
    "msteams": ("graph.microsoft.com",),
    "telegram": ("api.telegram.org",),
}

# Where a service's ``oauth2_client_credentials`` vault entry may send
# its client_secret to acquire an access token.  The daemon performs
# the token exchange itself (agents never see the secret), so the
# token endpoint must be pinned per service: a credential whose
# ``token_url`` points anywhere else would POST the secret to an
# attacker-chosen host.  Loopback endpoints are additionally accepted
# at store time (same-machine development/test token servers have no
# egress risk — the same exception Sentinel makes for plaintext HTTP).
TOKEN_ENDPOINT_HOSTS: dict[str, tuple[str, ...]] = {
    "msteams": ("login.microsoftonline.com",),
    # ``oauth2_refresh_token`` entries (device-authorization sign-ins
    # with a public client) refresh against these endpoints; the
    # refresh token is a bearer-equivalent secret and must never be
    # POSTed anywhere else.
    "github": ("github.com",),
    "twitch": ("id.twitch.tv",),
}


def valid_token_endpoint(service: str, url: Any) -> bool:
    """Return whether *url* may serve as *service*'s OAuth token endpoint.

    Args:
        service: Connector service name the credential is stored under.
        url: Candidate ``token_url`` (any type; non-strings fail).

    Returns:
        True for an ``https://`` URL on the service's pinned token
        host, or any valid http(s) URL to a loopback host.
    """
    from urllib.parse import urlparse

    if not (isinstance(url, str) and valid_http_url(url)):
        return False
    parsed = urlparse(url)
    host = canonical_host(parsed.hostname or "")
    if is_loopback_host(host):
        return True
    allowed = TOKEN_ENDPOINT_HOSTS.get(service)
    if allowed is None:
        # Scratch validation enrollments (msteams-pending) pin the same
        # token endpoints as their root service.
        allowed = TOKEN_ENDPOINT_HOSTS.get(service_root(service), ())
    return parsed.scheme == "https" and host in allowed

# Services that enroll one vault entry per workspace/account (slack)
# or use a scratch ``<root>-pending`` enrollment to validate a
# candidate credential without touching the live one (telegram,
# msteams); a ``<root>-<suffix>`` service name inherits the root's
# hosts, token endpoints, and action-classification rules.
_WORKSPACE_SERVICE_ROOTS = ("slack", "telegram", "msteams")


def service_root(service: str) -> str:
    """Return the base service a (possibly suffixed) name derives from.

    Args:
        service: Connector service name (e.g. ``"telegram-pending"``).

    Returns:
        The root name when the prefix is a known workspace/scratch
        root, else *service* unchanged.
    """
    root = service.split("-", 1)[0]
    return root if root in _WORKSPACE_SERVICE_ROOTS else service


# The exact grammar a candidate-validation scratch service is minted
# with: ``<root>-pending-<16 lowercase hex>`` for a token-exchange
# root.  Matched precisely so a legitimate per-workspace name that
# merely contains ``-pending-`` (e.g. a Slack workspace literally
# called "pending", ``slack-pending-<hash>``) is NOT mistaken for a
# scratch service and does not inherit another service's policy.
_SCRATCH_SERVICE_RE = re.compile(r"(?P<root>telegram|msteams)-pending-[0-9a-f]{16}")


def scratch_root(service: str) -> str:
    """Return the root a candidate-validation scratch name belongs to.

    Args:
        service: Connector service name.

    Returns:
        The root service (``"telegram"``/``"msteams"``) when *service*
        matches the exact scratch grammar, else ``""``.
    """
    match = _SCRATCH_SERVICE_RE.fullmatch(service)
    return match.group("root") if match else ""


def policy_service(service: str) -> str:
    """Return the service whose Sentinel policy governs *service*.

    Candidate-validation scratch names (``<root>-pending-<hex>``) must
    obey the LIVE root service's policy exactly — an explicit
    ``telegram`` deny must also deny validating a telegram candidate,
    and a ``telegram`` grant must approve it.  Every other name
    (including per-workspace ``slack-<ws>`` services, which are
    independent identities with their own isolated policy) governs
    itself.

    Args:
        service: Connector service name.

    Returns:
        The service name to look policy and grants up under.
    """
    return scratch_root(service) or service


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
    return SERVICE_HOSTS.get(service_root(service), ())


_HEADER_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")

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


def valid_credential_param(name: Any) -> bool:
    """Return whether *name* may carry a real credential as a query parameter.

    Used for ``{"kind": "query"}`` vault credentials (BlueBubbles'
    ``password``, Synology Chat's webhook ``token``): the boundary
    splices ``name=<real value>`` into the request URL's query string,
    so the name must be a plain token that cannot break out of its
    key/value slot.

    Args:
        name: Candidate query parameter name (any type; non-strings fail).

    Returns:
        True when the parameter may carry the credential.
    """
    return isinstance(name, str) and _HEADER_NAME_RE.fullmatch(name) is not None


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


# Path-placed credentials travel inside one URL path segment (Telegram's
# ``/bot<token>/<method>``), so beyond the header-value rules they must
# be path-safe without percent-encoding: RFC 3986 unreserved characters
# plus ``:`` (a pchar, and part of every Telegram bot token).  ``/``
# (segment escape), ``%`` (double-encoding ambiguity), ``?``/``#``
# (component escapes) and spaces are all rejected, which also makes the
# boundary's literal splice/scrub substitutions unambiguous.
_PATH_CREDENTIAL_VALUE_RE = re.compile(r"[A-Za-z0-9:_.~-]{1,256}")

# Stands in for a path-placed credential (or its surrogate) in every
# URL Sentinel sees: decisions and the audit log stay capability-free.
PATH_CREDENTIAL_PLACEHOLDER = "muse-path-credential"


def valid_credential_path_value(value: Any) -> bool:
    """Return whether *value* may be spliced into a URL path segment.

    Used for ``{"kind": "path"}`` vault credentials (Telegram's
    ``/bot<token>/...``).

    Args:
        value: Candidate token value (any type; non-strings fail).

    Returns:
        True when the value is path-safe without encoding.
    """
    return isinstance(value, str) and _PATH_CREDENTIAL_VALUE_RE.fullmatch(value) is not None


SURROGATE_PREFIX = "muse-sgt."

# Daemon wire-protocol version.  Bumped whenever the daemon gains
# semantics an older daemon would silently mishandle (v2: header-kind
# credentials, enrollment hosts, service-aware action classes; v3:
# consent-scoped insecure enrollment hosts, which a v2 daemon would
# silently drop from ``store_credentials`` and then deny every plain-
# HTTP Home Assistant request; v4: query-kind credentials, which a v3
# daemon would fail to resolve at the boundary; v5: path-kind
# credentials and daemon-side ``oauth2_client_credentials`` token
# acquisition, which a v4 daemon would fail to resolve; v6: the atomic
# store-if-absent critical section and the transport-write generation
# gate — a v5 daemon could fail a concurrent auto-migration and could
# emit a rotated-away credential resolved before connection setup;
# v7: daemon-side ``oauth2_refresh_token`` credentials from device-
# authorization sign-ins, which a v6 daemon would fail to resolve).
# The client restarts a running daemon whose ``status`` reports an
# older protocol, so a detached pre-upgrade daemon cannot serve new
# clients.
PROTOCOL_VERSION = 7

# One JSON object per line; requests carrying request/response bodies
# are base64-encoded, so cap the frame to keep the daemon safe from
# unbounded allocations (64 MiB of base64 ~ 48 MiB of payload).
MAX_FRAME_BYTES = 64 * 1024 * 1024

READ_METHODS = ("GET", "HEAD", "OPTIONS")


def platform_supports_muse_daemon() -> bool:
    """Return whether this platform can run the Muse-auth daemon.

    The daemon needs ``SO_PEERCRED`` peer authentication on Unix
    sockets (Linux; macOS exposes ``LOCAL_PEERCRED`` instead and
    Windows has neither).  On unsupported platforms the daemon cannot
    authenticate its first client, so defaulting Muse-auth on there
    would break every connector instead of protecting it.

    Returns:
        True when ``socket.SO_PEERCRED`` exists.
    """
    return hasattr(socket, "SO_PEERCRED")


def muse_auth_enabled() -> bool:
    """Return True when Muse-style auth is switched on (the default).

    On platforms that can run the daemon (see
    :func:`platform_supports_muse_daemon`), Muse-auth is enabled unless
    ``KISS_MUSE_AUTH`` is explicitly set to a falsy string ("0",
    "false", "no", "off") — unset, empty, or unrecognized values keep
    the secure default on, so a typo cannot silently fall back to
    plaintext credentials in the agent process.  An explicit truthy
    string ("1", "true", "yes", "on") forces Muse-auth on even on
    unsupported platforms, preserving the historical opt-in behavior.

    Returns:
        False when ``KISS_MUSE_AUTH`` is "0", "false", "no", or "off"
        (case-insensitive, surrounding whitespace ignored); True when
        it is "1", "true", "yes", or "on"; otherwise the platform
        default (True where the daemon can run, False elsewhere).
    """
    value = os.environ.get("KISS_MUSE_AUTH", "").strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    return platform_supports_muse_daemon()


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
    # Windows has no os.getuid (and no Unix sockets): there the path is
    # only ever probed for existence, never bound.
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(f"/tmp/kiss-muse-{uid}-{digest}.sock")


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


def origin_hosts(base_url: str) -> tuple[str, ...]:
    """Return the Muse enrollment origins for a self-hosted base URL.

    Origin-bound services (Mattermost, Nextcloud Talk, BlueBubbles,
    Synology Chat, ...) have no built-in allowlist: the configured
    origin — host AND port, because the same hostname on another port
    is a different server — is enrolled with the credential.

    Args:
        base_url: The configured server base URL.

    Returns:
        ``("host:port",)``, or ``()`` when the URL has no host.
    """
    entry = url_origin_entry(base_url)
    return (entry,) if entry else ()


def insecure_origin_hosts(base_url: str) -> tuple[str, ...]:
    """Return the origins to enroll as consent-scoped plain-HTTP origins.

    Only an explicit ``http://`` base URL to a non-loopback host needs
    the exception (loopback plaintext is always allowed, and HTTPS
    needs none).

    Args:
        base_url: The configured server base URL.

    Returns:
        ``("host:port",)`` for a plain-HTTP non-loopback base URL,
        else ``()``.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = canonical_host(parsed.hostname or "")
    if parsed.scheme == "http" and host and not is_loopback_host(host):
        return (url_origin_entry(base_url),)
    return ()


def strip_url_query_param(url: str, name: str) -> str:
    """Return *url* with every query parameter named *name* removed.

    Shared by the daemon boundary (which refuses to send a
    caller-supplied or server-echoed copy of a query-kind credential
    parameter next to the real one) and by connectors that scrub an
    embedded credential out of a configured URL (Synology Chat's
    webhook ``token``).

    Args:
        url: Absolute URL.
        name: Credential query parameter name.

    Returns:
        The URL without any ``name=...`` query pairs.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    if not parts.query:
        return url
    pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != name]
    return urlunsplit(parts._replace(query=urlencode(pairs)))


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

# Telegram Bot API methods that only read bot/chat state, plus the
# ephemeral typing indicator (sendChatAction) which must not burn
# one-shot write grants.  Method names are case-insensitive on
# Telegram's side, so the set is matched lowercased.  Every method not
# listed here classifies as a write — the HTTP verb is meaningless
# (``GET /bot<token>/sendMessage?...`` sends a message).
_TELEGRAM_READ_METHODS = frozenset(
    {
        "getchat",
        "getchatadministrators",
        "getchatmember",
        "getchatmembercount",
        "getchatmemberscount",
        "getfile",
        "getme",
        "getmycommands",
        "getupdates",
        "getwebhookinfo",
        "sendchataction",
    }
)


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
    if service == "mattermost":
        # Same reasoning as Discord: the typing indicator is ephemeral
        # presence, not a workspace mutation.
        if method.upper() == "POST" and path.rstrip("/").endswith("/users/me/typing"):
            return "read"
        return action_class(method)
    if service == "bluebubbles":
        # BlueBubbles' message search is a POST whose body carries the
        # query filters; it only retrieves messages.  Sending
        # (/message/text) and mark-read stay writes.
        if method.upper() == "POST" and path.rstrip("/").endswith("/message/query"):
            return "read"
        return action_class(method)
    if service_root(service) == "telegram":
        # Telegram is RPC over the URL path: /bot<token>/<Method>
        # answers to ANY HTTP verb (GET sendMessage sends a message),
        # so the verb-based default would misclassify writes as reads.
        # Classification comes from the API method name alone; the
        # /file/bot<token>/<file_path> download namespace is a read.
        segments = [s for s in path.split("/") if s]
        if segments and segments[0] == "file":
            return "read"
        api_method = segments[-1].lower() if segments else ""
        return "read" if api_method in _TELEGRAM_READ_METHODS else "write"
    # Nextcloud's POST /room/{token}/participants/active ("join a
    # conversation") creates or replaces an active participant session —
    # a server-side state change — so it deliberately stays a write and
    # needs a write grant even though the poll loop uses it.
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

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Muse-auth daemon: authd + Sentinel + network boundary in one process.

Listens on a Unix domain socket (one JSON frame in, one out, per
connection), authenticated with ``SO_PEERCRED`` (peer UID must match
the daemon's UID).  Agent processes talk to it through
:mod:`.client`; they never see real credentials.

Request ops:

* ``status`` — enrolled services.
* ``store_credentials`` — enroll a service's OAuth token into the vault.
* ``clear_credentials`` — remove a service's token and surrogates.
* ``mint_surrogate`` — get an opaque per-service surrogate token.
* ``http_request`` — the boundary: Sentinel authorizes the concrete
  request, the surrogate Authorization header is swapped for the real
  bearer token, and the HTTPS call is executed here in the daemon.
* ``grant`` / ``revoke`` — manage user approvals.
* ``stop`` — shut the daemon down.

Run with ``python -m kiss.agents.third_party_agents.muse_auth.daemon``.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import math
import os
import re
import socket
import struct
import threading
import time
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

import requests
import urllib3.connection
from requests.adapters import HTTPAdapter
from urllib3 import connectionpool

from kiss.agents.third_party_agents.muse_auth._common import (
    PATH_CREDENTIAL_PLACEHOLDER,
    PROTOCOL_VERSION,
    SURROGATE_PREFIX,
    _is_ip_literal,
    canonical_host,
    canonical_host_entry,
    muse_auth_dir,
    recv_frame,
    scratch_root,
    send_frame,
    socket_path,
    strip_url_query_param,
    url_origin,
    valid_credential_header,
    valid_credential_param,
    valid_credential_path_value,
    valid_credential_value,
    valid_hostname,
    valid_service_name,
    valid_token_endpoint,
)
from kiss.agents.third_party_agents.muse_auth.sentinel import Sentinel
from kiss.agents.third_party_agents.muse_auth.vault import CredentialVault
from kiss.core.file_lock import lock_exclusive, unlock

_HOP_HEADERS = ("connection", "keep-alive", "transfer-encoding", "content-length", "host")
_UNDECODED_HEADERS = ("content-encoding", "transfer-encoding", "content-length")


# The bracket alternative accepts any IPv6 spelling; the bracket
# contents are validated as a real IP by _valid_host_part.
_PORT_SUFFIX_RE = re.compile(r"(?P<host>\[[^\]]+\]|[^:]+):(?P<port>[0-9]{1,5})")


def _valid_host_part(host_part: str) -> bool:
    """Return whether a (possibly bracketed) host part is acceptable.

    A bracketed host part must be a real IPv6/IP literal (a bracketed
    hostname is malformed); an unbracketed part may be a hostname or a
    bare IP literal.

    Args:
        host_part: Host portion of an enrollment entry, lowercased.

    Returns:
        True when the host part is a valid host.
    """
    if host_part.startswith("[") and host_part.endswith("]"):
        return _is_ip_literal(canonical_host(host_part[1:-1]))
    return valid_hostname(host_part)


def _invalid_hosts_reason(hosts: Any) -> str:
    """Validate enrollment-time extra hosts; return the rejection reason.

    Args:
        hosts: The ``hosts`` list from a ``store_credentials`` frame.

    Returns:
        A human-readable error, or ``""`` when the hosts are acceptable
        (a list of at most 16 lowercase hostnames/IP literals, each
        optionally port-pinned as ``host:port`` / ``[ipv6]:port``).
    """
    if not isinstance(hosts, list):
        return "hosts must be a list of hostnames"
    if len(hosts) > 16:
        return "at most 16 enrollment hosts are allowed"
    for host in hosts:
        if not isinstance(host, str):
            return f"invalid enrollment host {str(host)[:80]!r}"
        candidate = host.strip().lower()
        pinned = _PORT_SUFFIX_RE.fullmatch(candidate)
        if pinned:
            if int(pinned.group("port")) > 65535:
                return f"invalid enrollment host {str(host)[:80]!r}"
            candidate = pinned.group("host")
        if not _valid_host_part(candidate):
            return f"invalid enrollment host {str(host)[:80]!r}"
    return ""


def _credential_regex(value: str) -> re.Pattern[str]:
    """Return a regex matching every wire spelling of a path credential.

    A path-placed credential can be echoed back by the API origin either
    verbatim or with any subset of its characters percent-encoded (and
    the hex digits in either case): ``:`` as ``%3A`` or ``%3a``, an
    unreserved ``A`` as ``%41``.  The pattern matches each character as
    itself OR as its case-insensitive percent escape, so no encoding
    variant slips through a scrub.

    Args:
        value: The real credential value (path-safe charset).

    Returns:
        A compiled pattern (anchored at neither end).
    """
    atoms = []
    for char in value:
        hexcode = format(ord(char), "02X")
        hex_ci = "".join(f"[{d}{d.lower()}]" if d.isalpha() else d for d in hexcode)
        # ``%(?:25)*<HH>`` matches the char percent-encoded at any
        # depth: ``%3A``, and the multiply-encoded ``%253A`` (``%25``
        # is an encoded ``%``), ``%25253A``, ...  A single-level regex
        # would let a doubly-encoded echo slip a reversible spelling
        # past the scrub.
        atoms.append(f"(?:{re.escape(char)}|%(?:25)*{hex_ci})")
    return re.compile("".join(atoms))


def _scrub_credential_text(text: str, value: str, replacement: str) -> str:
    """Replace every wire spelling of *value* in *text* with *replacement*.

    Normalizes accidental credential reflections in header values,
    reason phrases, and URLs.  The API origin legitimately knows the
    credential and could theoretically re-encode it in ways no scrubber
    anticipates (e.g. double percent-encoding), but every single-level
    encoding an ordinary echo produces is covered.

    Args:
        text: Header value, reason phrase, or URL text.
        value: The real credential value.
        replacement: The surrogate or placeholder to substitute.

    Returns:
        The normalized text.
    """
    return _credential_regex(value).sub(replacement, text)


def _scrub_credential_bytes(body: bytes, value: str, replacement: str) -> bytes:
    """Replace every wire spelling of *value* in a response body.

    The body may be binary, so it is treated as latin-1 (a 1:1 byte
    mapping that round-trips every byte); the credential is ASCII, so
    the pattern only matches ASCII byte runs and unrelated binary data
    is preserved exactly.

    Args:
        body: Response body bytes.
        value: The real credential value.
        replacement: The surrogate to substitute.

    Returns:
        The normalized body bytes.
    """
    if value.encode("latin-1", "ignore") not in body and "%" not in value:
        # Fast path: the raw value is absent and it has no literal
        # percent (so a raw substring miss is conclusive only when the
        # value itself cannot appear pre-encoded).  Fall through to the
        # regex whenever a percent byte is present, which is where
        # encoded spellings live.
        if b"%" not in body:
            return body
    decoded = body.decode("latin-1")
    scrubbed = _credential_regex(value).sub(replacement, decoded)
    return scrubbed.encode("latin-1")


def _scrub_url_path_token(url: str, value: str, replacement: str) -> str:
    """Replace every spelling of *value* in *url* with *replacement*.

    Args:
        url: Absolute URL (e.g. a redirect target).
        value: The real credential value.
        replacement: The surrogate or placeholder to substitute.

    Returns:
        The normalized URL.
    """
    return _scrub_credential_text(url, value, replacement)


def _path_credential_url(url: str, surrogate: str, value: str) -> str:
    """Return *url* with the path-placed surrogate replaced by *value*.

    Only the path component is substituted, so a surrogate string that
    somehow also appears in the query can never turn into the real
    credential.  An empty *value* redacts the segment instead: a hop
    that is not entitled to the credential must not reveal a live
    surrogate to a foreign host either.

    Args:
        url: Absolute request URL carrying the surrogate in its path.
        surrogate: The surrogate token embedded in the path.
        value: Real credential value, or ``""`` to redact.

    Returns:
        The URL to actually send.
    """
    parts = urlsplit(url)
    replacement = value or "muse-redacted"
    return urlunsplit(parts._replace(path=parts.path.replace(surrogate, replacement)))


def _invalid_client_credentials_reason(service: str, info: dict[str, Any]) -> str:
    """Validate an ``oauth2_client_credentials`` enrollment payload.

    Args:
        service: Connector service name the credential is stored under.
        info: The ``authorized_user_info`` dict from the store frame.

    Returns:
        A human-readable rejection reason, or ``""`` when acceptable.
    """
    if not valid_token_endpoint(service, info.get("token_url")):
        # The daemon POSTs the client_secret to this URL; anything but
        # the service's pinned https endpoint (or a same-machine
        # loopback endpoint) would exfiltrate the secret.
        return f"invalid or unpinned OAuth token endpoint URL for '{service}'"
    if not valid_credential_value(info.get("client_id")):
        return "invalid client_id value"
    if not valid_credential_value(info.get("client_secret")):
        return "invalid client_secret value"
    scope = info.get("token_scope")
    if scope is not None and not valid_credential_value(scope):
        return "invalid token_scope value"
    return ""


def _invalid_refresh_token_reason(service: str, info: dict[str, Any]) -> str:
    """Validate an ``oauth2_refresh_token`` enrollment payload.

    These entries come from a device-authorization sign-in with a
    public OAuth client: the daemon later POSTs the refresh token to
    ``token_url``, so the endpoint must be the service's pinned host
    (or loopback), and every value must be a clean credential string.

    Args:
        service: Connector service name the credential is stored under.
        info: The ``authorized_user_info`` dict from the store frame.

    Returns:
        A human-readable rejection reason, or ``""`` when acceptable.
    """
    if not valid_token_endpoint(service, info.get("token_url")):
        return f"invalid or unpinned OAuth token endpoint URL for '{service}'"
    for key in ("client_id", "access_token", "refresh_token"):
        if not valid_credential_value(info.get(key)):
            return f"invalid {key} value"
    scope = info.get("token_scope")
    if scope is not None and not valid_credential_value(scope):
        return "invalid token_scope value"
    expires_at = info.get("expires_at")
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        return "invalid expires_at value"
    if not math.isfinite(float(expires_at)):
        return "invalid expires_at value"
    return ""


def _with_query_param(url: str, name: str, value: str) -> str:
    """Return *url* with ``name=value`` appended to its query string.

    Args:
        url: Absolute request URL (already stripped of *name*).
        name: Credential query parameter name.
        value: Real credential value (percent-encoded on the way in).

    Returns:
        The URL carrying the credential parameter.
    """
    from urllib.parse import urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    encoded = urlencode([(name, value)])
    query = f"{parts.query}&{encoded}" if parts.query else encoded
    return urlunsplit(parts._replace(query=query))


def _spliced_url(
    url: str, placement: str, cred_name: str, cred_value: str, path_surrogate: str
) -> str:
    """Return the URL to actually send for one authorized hop.

    Args:
        url: The credential-free tracked URL of the hop.
        placement: ``"header"``, ``"query"``, or ``"path"``.
        cred_name: Query parameter name (query placement only).
        cred_value: Real credential value; empty when the hop is not
            entitled to carry it.
        path_surrogate: Surrogate marker in the path (path placement).

    Returns:
        The send URL with the credential spliced in (or, for path
        placement without a credential, the surrogate redacted).
    """
    if placement == "query" and cred_value:
        return _with_query_param(url, cred_name, cred_value)
    if placement == "path":
        return _path_credential_url(url, path_surrogate, cred_value)
    return url


class _CredentialRotatedError(Exception):
    """The pinned credential generation changed before the head write.

    Raised by the transport-write gate (under the vault lock, before
    any byte of the credential-bearing request head reaches the
    socket), so a rotation that completes during connection setup —
    DNS, TCP, TLS — aborts the request instead of emitting the
    old-generation credential.
    """


# The transport-write gate for the request currently being sent on
# this thread: a callable returning a context manager that holds the
# vault lock and re-checks the pinned credential generation.  Set by
# _execute around its session use; each boundary request runs on its
# own daemon thread, so the gate can never leak across requests.
_SEND_GATE = threading.local()


class _GatedSendMixin(urllib3.connection.HTTPConnection):
    """Connection mixin: emit the request head under the send gate.

    ``urllib3`` composes the whole request head (request line — which
    carries path/query-placed credentials — plus all headers) via
    ``putrequest``/``putheader`` and writes it to the socket in
    ``endheaders``; body chunks follow in separate ``send`` calls.
    Overriding ``endheaders`` therefore brackets exactly the moment the
    credential bytes are emitted:

    1. the connection is established FIRST (DNS/TCP/TLS happen outside
       any lock — for HTTPS pools ``_validate_conn`` already connected);
    2. the gate then acquires the vault lock, re-checks the pinned
       credential generation, and the head is written to the
       already-connected socket while the lock is still held.

    A rotation's ``store()`` takes the same lock, so it either
    completes before the check (the request aborts) or waits until the
    head bytes have been handed to the kernel.  The head is small and
    the socket send buffer of a fresh or idle keep-alive connection is
    empty, so this write does not wait on the peer; even a pathological
    stall is bounded by the socket timeout.  The lock is NOT held while
    the body is sent or the response is awaited (a peer-triggered
    rotation would deadlock against a lock held across that wait).
    """

    def endheaders(
        self, message_body: Any = None, *, encode_chunked: bool = False
    ) -> None:
        """Write the buffered request head, gated on the vault generation.

        Args:
            message_body: Optional body handed through to http.client.
            encode_chunked: Chunked-encoding flag handed through.
        """
        gate = getattr(_SEND_GATE, "check", None)
        if gate is None:
            super().endheaders(message_body, encode_chunked=encode_chunked)
            return
        if self.sock is None:
            # Plain-HTTP pools connect lazily inside send(); do the
            # DNS/TCP setup now, before the gate takes the vault lock.
            self.connect()
        with gate():
            super().endheaders(message_body, encode_chunked=encode_chunked)


class _GatedHTTPConnection(_GatedSendMixin, urllib3.connection.HTTPConnection):
    """Plain-HTTP boundary connection with the transport-write gate."""


class _GatedHTTPSConnection(_GatedSendMixin, urllib3.connection.HTTPSConnection):
    """TLS boundary connection with the transport-write gate."""


class _GatedHTTPConnectionPool(connectionpool.HTTPConnectionPool):
    """Pool producing :class:`_GatedHTTPConnection` connections."""

    # The gated connection is a genuine urllib3 HTTPConnection subclass;
    # pyright cannot see it satisfies the pool's structural protocol.
    ConnectionCls = _GatedHTTPConnection  # pyright: ignore[reportAssignmentType]


class _GatedHTTPSConnectionPool(connectionpool.HTTPSConnectionPool):
    """Pool producing :class:`_GatedHTTPSConnection` connections."""

    ConnectionCls = _GatedHTTPSConnection  # pyright: ignore[reportAssignmentType]


class _GatedSendAdapter(HTTPAdapter):
    """Requests adapter whose connections honor the send gate."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        """Build the pool manager, then swap in the gated pool classes.

        Args:
            *args: Positional pool-manager options from HTTPAdapter.
            **kwargs: Keyword pool-manager options from HTTPAdapter.
        """
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _GatedHTTPConnectionPool,
            "https": _GatedHTTPSConnectionPool,
        }


def _peer_uid(conn: socket.socket) -> int:
    """Return the connecting peer's UID via ``SO_PEERCRED``.

    Args:
        conn: Accepted Unix-socket connection.

    Returns:
        The peer's numeric UID.
    """
    # ``SO_PEERCRED`` is Linux-only; look it up at runtime so the module
    # type-checks on macOS, where ``platform_supports_muse_daemon`` keeps
    # the daemon off and this function is never reached.
    so_peercred = getattr(socket, "SO_PEERCRED")
    creds = conn.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
    _pid, uid, _gid = struct.unpack("3i", creds)
    return int(uid)


class MuseAuthDaemon:
    """Socket server hosting the vault, Sentinel, and network boundary."""

    def __init__(self) -> None:
        self.vault = CredentialVault()
        # Sentinel extends each service's allowlist with the hosts
        # enrolled alongside its vault credential (self-hosted bases),
        # and honors consent-time plain-HTTP exceptions for hosts the
        # user enrolled from an http:// base URL.
        self.sentinel = Sentinel(
            hosts_provider=self.vault.enrolled_hosts,
            insecure_hosts_provider=self.vault.enrolled_insecure_hosts,
        )
        self._server: socket.socket | None = None
        self._stop = threading.Event()

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one request frame to its op handler.

        Args:
            request: Decoded request frame.

        Returns:
            Response frame (always carries an ``ok`` bool).
        """
        op = request.get("op", "")
        service_arg = request.get("service")
        if service_arg is not None and not valid_service_name(str(service_arg)):
            # Service names become vault file names; reject traversal.
            return {"ok": False, "error": f"invalid service name {str(service_arg)!r}"}
        if op == "status":
            vault_dir = muse_auth_dir() / "vault"
            services = (
                sorted(p.stem for p in vault_dir.glob("*.json")) if vault_dir.exists() else []
            )
            return {"ok": True, "services": services, "protocol": PROTOCOL_VERSION}
        if op == "store_credentials":
            return self._store_credentials(request)
        if op == "clear_credentials":
            self.vault.clear(request["service"])
            return {"ok": True}
        if op == "mint_surrogate":
            service = request["service"]
            if not self.vault.has_credentials(service):
                return {"ok": False, "error": f"service '{service}' is not enrolled in the vault"}
            return {"ok": True, "surrogate": self.vault.mint_surrogate(service)}
        if op == "grant":
            grant_id = self.sentinel.add_grant(
                request["service"],
                request["action"],
                request["scope"],
                float(request.get("ttl", 0.0)),
            )
            return {"ok": True, "grant_id": grant_id}
        if op == "revoke":
            removed = self.sentinel.revoke_grants(request["service"], request.get("action", ""))
            return {"ok": True, "removed": removed}
        if op == "http_request":
            return self._boundary(request)
        if op == "stop":
            self._stop.set()
            return {"ok": True}
        return {"ok": False, "error": f"unknown op '{op}'"}

    def _store_credentials(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle the ``store_credentials`` op as one vault critical section.

        The presence short-circuit, the candidate validation, and the
        conditional write all run under the vault lock: for a
        store-if-absent, presence and validation cannot be separated —
        a concurrent authoritative store either lands before this
        section (so the now-irrelevant candidate is never validated and
        the caller gets ``created=False``) or after it, never in
        between.  Everything inside the section is pure CPU work.

        Args:
            request: Decoded ``store_credentials`` frame.

        Returns:
            Response frame (``ok`` plus ``created`` or ``error``).
        """
        with self.vault.locked():
            info = request["authorized_user_info"]
            # For an atomic store-if-absent, an existing credential wins
            # BEFORE the candidate is validated: a stale/malformed
            # config value that would never be stored (the vault already
            # holds the authoritative credential) must not raise a
            # validation error that fails the whole connect.
            if bool(request.get("only_if_absent", False)) and self.vault.has_credentials(
                request["service"]
            ):
                return {"ok": True, "created": False}
            if (
                isinstance(info, dict)
                and info.get("kind") == "header"
                and not valid_credential_header(info.get("header"))
            ):
                return {"ok": False, "error": "invalid credential header name"}
            if (
                isinstance(info, dict)
                and info.get("kind") == "query"
                and not valid_credential_param(info.get("param"))
            ):
                return {"ok": False, "error": "invalid credential query parameter name"}
            if (
                isinstance(info, dict)
                and info.get("kind") in ("header", "bearer", "query")
                and not valid_credential_value(info.get("token"))
            ):
                # A malformed value (control chars, stray whitespace)
                # would make the HTTP stack raise errors that reflect
                # the credential verbatim; refuse it at enrollment.
                return {"ok": False, "error": "invalid credential token value"}
            if (
                isinstance(info, dict)
                and info.get("kind") == "path"
                and not valid_credential_path_value(info.get("token"))
            ):
                # Path-placed values must be splice-safe without
                # percent-encoding (see valid_credential_path_value).
                return {"ok": False, "error": "invalid path credential token value"}
            if isinstance(info, dict) and info.get("kind") == "oauth2_client_credentials":
                reason = _invalid_client_credentials_reason(request["service"], info)
                if reason:
                    return {"ok": False, "error": reason}
            if isinstance(info, dict) and info.get("kind") == "oauth2_refresh_token":
                reason = _invalid_refresh_token_reason(request["service"], info)
                if reason:
                    return {"ok": False, "error": reason}
            hosts = request.get("hosts", [])
            insecure_hosts = request.get("insecure_hosts", [])
            hosts_error = _invalid_hosts_reason(hosts) or _invalid_hosts_reason(insecure_hosts)
            if hosts_error:
                return {"ok": False, "error": hosts_error}
            # Enrolling a candidate-validation scratch entry is a good
            # moment to sweep any scratch files a killed validation
            # caller abandoned (they are removed in a finally that a
            # SIGKILL cannot run).
            if scratch_root(str(request["service"])):
                self.vault.sweep_stale_pending()
            created = self.vault.store(
                request["service"],
                info,
                request.get("scopes", []),
                hosts=[canonical_host_entry(str(h)) for h in hosts],
                insecure_hosts=[canonical_host_entry(str(h)) for h in insecure_hosts],
                only_if_absent=bool(request.get("only_if_absent", False)),
            )
            return {"ok": True, "created": created}

    def _boundary(self, request: dict[str, Any]) -> dict[str, Any]:
        """Authorize and execute one outbound API request.

        Validates the surrogate, lets Sentinel decide, swaps the
        surrogate Authorization header for the real bearer token, and
        performs the HTTPS call from the daemon process.

        Args:
            request: Frame with ``service``, ``method``, ``url``,
                ``headers``, ``body_b64``, and ``timeout``.

        Returns:
            ``{"ok": True, "status", "reason", "headers", "body_b64"}``
            on an executed request (any HTTP status), or
            ``{"ok": False, "denied": True, "error": ...}`` when
            Sentinel refuses, or ``{"ok": False, "error": ...}`` on
            surrogate/credential problems.
        """
        headers = {str(k): str(v) for k, v in request.get("headers", {}).items()}
        surrogate = ""
        for key, value in headers.items():
            if key.lower() == "authorization" and value.lower().startswith("bearer "):
                surrogate = value[7:].strip()
                break
        if not surrogate.startswith(SURROGATE_PREFIX):
            return {"ok": False, "error": "request carries no surrogate bearer token"}
        service = self.vault.surrogate_service(surrogate)
        if service is None:
            return {
                "ok": False,
                "error": "unknown or stale surrogate token; re-connect the agent backend",
            }
        declared = request.get("service", service)
        if declared != service:
            return {
                "ok": False,
                "error": f"surrogate is bound to '{service}' but the client claims '{declared}'",
            }
        method = str(request["method"]).upper()
        raw_url = str(request["url"])
        body = base64.b64decode(request.get("body_b64", "")) or None
        # Normalize the URL exactly the way requests will send it, so
        # Sentinel and the eventual connection agree on the host
        # (closes urlparse-vs-requests parser-differential bypasses).
        # Sentinel inspects the raw URL for userinfo/scheme and the
        # normalized URL for the destination host.
        try:
            prepared = requests.Request(method, raw_url, data=body).prepare()
        except Exception as e:
            return {"ok": False, "error": f"malformed request URL: {e}"}
        url = str(prepared.url)
        # For a path-placed credential the URL itself must reference the
        # surrogate; Sentinel then sees (and audits) the URL with that
        # capability handle redacted.  A rotation between this peek and
        # the resolution below invalidates the surrogate, so the
        # re-check after the generation read catches any disagreement.
        display_raw, display_url = raw_url, url
        if self.vault.placement(service) == "path":
            if surrogate not in urlsplit(url).path:
                return {
                    "ok": False,
                    "error": "path-credential request URL does not reference the surrogate",
                }
            display_raw = raw_url.replace(surrogate, PATH_CREDENTIAL_PLACEHOLDER)
            display_url = url.replace(surrogate, PATH_CREDENTIAL_PLACEHOLDER)
        decision = self.sentinel.decide(service, method, display_raw, effective_url=display_url)
        if decision.verdict != "allow":
            return {"ok": False, "denied": True, "error": decision.reason}
        # Pin the whole request (initial send and every redirect hop) to
        # the credential generation that was current while the surrogate
        # was still live: a rotation that lands mid-request aborts it
        # instead of letting an old-generation capability spend the new
        # credential.  (Re-check the surrogate AFTER reading the
        # generation: store() bumps the generation and invalidates
        # surrogates under one vault lock, so a live surrogate here
        # proves the generation read is current.)
        generation = self.vault.generation(service)
        if self.vault.surrogate_service(surrogate) != service:
            return {
                "ok": False,
                "error": "unknown or stale surrogate token; re-connect the agent backend",
            }
        try:
            placement, cred_name, cred_value = self.vault.resolve_credential(service, generation)
        except Exception as e:
            return {"ok": False, "error": f"credential resolution failed: {e}"}
        if placement == "header" and not valid_credential_header(cred_name):
            # Defense in depth: enrollment already validates this.
            return {
                "ok": False,
                "error": f"vault credential for '{service}' names an unsafe header",
            }
        if placement == "query" and not valid_credential_param(cred_name):
            return {
                "ok": False,
                "error": f"vault credential for '{service}' names an unsafe query parameter",
            }
        if placement == "path" and not valid_credential_path_value(cred_value):
            # Defense in depth: enrollment already validates this.
            return {
                "ok": False,
                "error": f"vault credential for '{service}' is not path-splice-safe",
            }
        # Preserve the caller's own headers (Notion-Version, Accept,
        # multipart Content-Type, ...); only drop hop-by-hop headers,
        # every Authorization variant, and every copy of the credential
        # header, then set exactly one real credential so a duplicate
        # header cannot smuggle a value past the swap.
        dropped = {"authorization"}
        if placement == "header":
            dropped.add(cred_name.lower())
        out_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in _HOP_HEADERS and k.lower() not in dropped
        }
        if placement == "header":
            out_headers[cred_name] = cred_value
        else:
            # Query-kind: the credential travels in the URL, injected
            # per hop by _execute.  A caller-supplied copy of the
            # parameter must not survive next to the real one.
            url = strip_url_query_param(url, cred_name)
        try:
            resp = self._execute(
                service,
                method,
                url,
                out_headers,
                body,
                float(request.get("timeout", 120.0)),
                placement=placement,
                cred_name=cred_name,
                cred_value=cred_value,
                generation=generation,
                path_surrogate=surrogate if placement == "path" else "",
            )
        except Exception as e:
            if placement in ("query", "path"):
                # The sent URL carries the (percent-encoded or
                # path-spliced) credential and the HTTP stack embeds
                # that URL verbatim in exception text; rather than
                # enumerating encoding variants, return only the
                # exception class and the credential-free URL (the
                # param-stripped URL for query placement, the
                # surrogate-redacted URL for path placement).
                safe_url = url if placement == "query" else display_url
                message = f"{type(e).__name__} contacting {safe_url}"
            else:
                # Exception text from the HTTP stack can reflect header
                # values verbatim; never let the real credential cross
                # back to the agent inside an error message.
                message = str(e).replace(cred_value, "<redacted-credential>")
            return {"ok": False, "error": f"network boundary request failed: {message}"}
        if isinstance(resp, str):
            return {"ok": False, "denied": True, "error": resp}
        # requests already decoded any content-encoding; drop headers
        # describing the raw wire form so clients trust the body as-is.
        resp_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in _UNDECODED_HEADERS
        }
        reason = resp.reason or ""
        content = resp.content
        if placement == "path":
            # The credentialed request URI is server-visible, and APIs
            # routinely echo it (an error description repeating the
            # path, a redirect-limit Location header).  Normalize every
            # wire spelling back to the agent's own surrogate before the
            # reply crosses out of the daemon.
            resp_headers = {
                k: _scrub_credential_text(v, cred_value, surrogate)
                for k, v in resp_headers.items()
            }
            reason = _scrub_credential_text(reason, cred_value, surrogate)
            content = _scrub_credential_bytes(content, cred_value, surrogate)
        return {
            "ok": True,
            "status": resp.status_code,
            "reason": reason,
            "headers": resp_headers,
            "body_b64": base64.b64encode(content).decode(),
        }

    def _execute(
        self,
        service: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        placement: str = "header",
        cred_name: str = "Authorization",
        cred_value: str = "",
        generation: str | None = None,
        path_surrogate: str = "",
    ) -> requests.Response | str:
        """Execute a request, following redirects with per-hop authorization.

        Each hop is evaluated by Sentinel.  The real credential is
        only sent to hosts on the service's allowlist; a redirect to any
        other host (e.g. a Google download CDN or signed URL) is
        followed without the credential so the token can never leak off
        an allowlisted host.  Method/body are downgraded to GET on
        301/302/303 exactly as a normal HTTP client would.

        Args:
            service: Connector service name.
            method: HTTP method.
            url: Absolute request URL (already authorized for hop 0;
                for query placement, already stripped of *cred_name*).
            headers: Outgoing headers (for header placement, including
                the real credential).
            body: Request body bytes, or None.
            timeout: Per-request timeout in seconds.
            placement: ``"header"`` (credential in *cred_name* header),
                ``"query"`` (credential spliced into the URL query as
                ``cred_name=cred_value`` just before each send), or
                ``"path"`` (credential substituted for *path_surrogate*
                inside the URL path just before each send).
            cred_name: Header or query parameter carrying the real
                credential (empty for path placement).
            cred_value: The real credential value (used for query and
                path placement; header placement already carries it in
                *headers*).
            generation: Vault credential generation the request is
                pinned to; redirect hops abort when it changes.
            path_surrogate: For path placement, the surrogate token the
                URL path references (its splice marker).

        Returns:
            The final :class:`requests.Response`, or a Sentinel denial
            reason string when a redirect hop is refused.
        """

        # The boundary must be immune to ambient environment configs:
        # an HTTP(S)_PROXY would re-route credentialed requests through
        # an unauthorized intermediary and a ~/.netrc would overwrite
        # the swapped Authorization header after the swap.
        session = requests.Session()
        session.trust_env = False
        # Gated connections: every send in this session re-checks the
        # pinned credential generation under the vault lock at the
        # instant the request head is written to the connected socket
        # (see _GatedSendMixin), so a rotation that completes during
        # connection setup — DNS, TCP, TLS — aborts the request instead
        # of emitting the old-generation credential.
        session.mount("http://", _GatedSendAdapter())
        session.mount("https://", _GatedSendAdapter())
        _SEND_GATE.check = functools.partial(self._generation_gate, service, generation)
        try:
            return self._send_and_follow(
                session,
                service,
                method,
                url,
                headers,
                body,
                timeout,
                placement,
                cred_name,
                cred_value,
                generation,
                path_surrogate,
            )
        except _CredentialRotatedError:
            return (
                f"the '{service}' credential changed mid-request; "
                "re-connect the agent backend and retry"
            )
        finally:
            _SEND_GATE.check = None

    @contextlib.contextmanager
    def _generation_gate(self, service: str, generation: str | None) -> Any:
        """Hold the vault lock and re-check a request's pinned generation.

        The transport writes the credential-bearing request head inside
        this context (see :class:`_GatedSendMixin`): the check and the
        head write form one critical section with respect to
        :meth:`CredentialVault.store`, so a rotation either completes
        before the check (aborting the request) or after the credential
        bytes were emitted — never in between.

        Args:
            service: Connector service name.
            generation: The vault generation the request is pinned to;
                ``None`` skips the check (nothing was resolved).

        Raises:
            _CredentialRotatedError: When the credential was replaced
                after this request was authorized.

        Yields:
            None while the vault lock is held.
        """
        with self.vault.locked():
            if generation is not None and self.vault.generation(service) != generation:
                raise _CredentialRotatedError(service)
            yield

    def _send_and_follow(
        self,
        session: requests.Session,
        service: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        placement: str,
        cred_name: str,
        cred_value: str,
        generation: str | None,
        path_surrogate: str,
    ) -> requests.Response | str:
        """Send one authorized request and follow its redirect chain.

        Runs inside :meth:`_execute`'s send-gate scope; every argument
        has the meaning documented there.

        Args:
            session: The gated, ``trust_env=False`` session to send on.
            service: Connector service name.
            method: HTTP method.
            url: Credential-free absolute request URL (hop 0 was
                already authorized by Sentinel).
            headers: Outgoing headers.
            body: Request body bytes, or None.
            timeout: Per-request timeout in seconds.
            placement: ``"header"``, ``"query"``, or ``"path"``.
            cred_name: Header or query parameter carrying the credential.
            cred_value: The real credential value.
            generation: Vault generation the request is pinned to.
            path_surrogate: Path-placement surrogate splice marker.

        Returns:
            The final :class:`requests.Response`, or a Sentinel denial
            reason string when a redirect hop is refused.
        """
        with session:
            # For query/path placement the credential is spliced into
            # the sent URL just before each send; ``url`` itself stays
            # credential-free (it feeds Sentinel and origin checks).
            # ``path_token`` remembers the real path credential so
            # server-echoed copies can be scrubbed out of redirect
            # targets even on hops not entitled to carry it.
            path_token = cred_value if placement == "path" else ""
            # No generation check is needed here: the send gate performs
            # it under the vault lock at the moment each hop's request
            # head is written (after DNS/TCP/TLS setup), which closes
            # both the token-exchange window (resolve_credential can
            # take a network round-trip) and the connection-setup
            # window.  The lock is never held across a response wait,
            # so a peer-triggered rotation cannot deadlock.
            sent = _spliced_url(url, placement, cred_name, cred_value, path_surrogate)
            resp = session.request(
                method,
                sent,
                headers=headers,
                data=body,
                timeout=timeout,
                allow_redirects=False,
            )
            # The credential stays pinned to the origin of the initially
            # authorized request for the WHOLE redirect chain: comparing
            # against the immediately previous hop would let an
            # allowlisted foreign origin regain the credential with one
            # extra same-origin redirect (A -> B -> B).
            first_parsed = urlparse(url)
            pinned_origin = url_origin(
                first_parsed.scheme, first_parsed.hostname, first_parsed.port
            )
            # Follow up to 5 redirects, re-authorizing each hop.
            for _hop in range(5):
                location = resp.headers.get("Location")
                if not (resp.is_redirect and location):
                    return resp
                # Resolve the redirect against the credential-free form
                # of the current URL, so a relative Location can never
                # inherit the real credential from the sent URL's path.
                next_url = requests.compat.urljoin(url, location)  # type: ignore[attr-defined]
                if placement == "query":
                    # Never treat a server-echoed (or attacker-chosen)
                    # copy of the credential parameter as part of the
                    # redirect target; authorized hops re-inject it.
                    next_url = strip_url_query_param(next_url, cred_name)
                if placement == "path" and path_token:
                    # Same rule for path placement: normalize any echoed
                    # copy of the real credential (raw or any
                    # percent-encoded spelling) back to the surrogate
                    # marker; authorized pinned-origin hops re-inject it.
                    next_url = _scrub_url_path_token(next_url, path_token, path_surrogate)
                next_parsed = urlparse(next_url)
                next_host = canonical_host(next_parsed.hostname or "")
                # Same ORIGIN (scheme, host, and effective port) as the
                # initially authorized request: the same hostname on
                # another port is a different server and must re-qualify
                # through the allowlist.
                same_host = url_origin(
                    next_parsed.scheme, next_parsed.hostname, next_parsed.port
                ) == pinned_origin
                if resp.status_code in (301, 302, 303) and method not in ("GET", "HEAD"):
                    method, body = "GET", None
                headers = dict(headers)
                # Sentinel (and its audit log) sees path-placed
                # capability handles redacted, exactly like hop 0 (the
                # real value was already normalized to the surrogate
                # above; scrub again as defense in depth).
                display_next = next_url
                if placement == "path":
                    display_next = _scrub_url_path_token(
                        display_next.replace(path_surrogate, PATH_CREDENTIAL_PLACEHOLDER),
                        path_token,
                        PATH_CREDENTIAL_PLACEHOLDER,
                    )
                if not (same_host or self.sentinel.origin_allowed(service, next_url)):
                    # Cross-host redirect off the allowlist: only bodyless
                    # GET/HEAD hops (download CDNs, signed URLs) may be
                    # followed, and never with the real credential.  A
                    # 307/308 keeps the request body, so following it would
                    # ship content to a host Sentinel denied.
                    self.sentinel.decide(service, method, display_next, effective_url=display_next)
                    if method not in ("GET", "HEAD") or body is not None:
                        return (
                            f"cross-host redirect to '{next_host}' would carry the request "
                            f"body off the '{service}' allowlist; refusing to follow it"
                        )
                    headers.pop(cred_name, None)
                    cred_value = ""
                else:
                    decision = self.sentinel.decide(
                        service, method, display_next, effective_url=display_next
                    )
                    if decision.verdict != "allow":
                        return decision.reason
                    # Re-resolve the credential for every authorized
                    # hop, pinned to the request's vault generation: a
                    # rotation landing mid-request aborts the request
                    # instead of shipping either generation's token
                    # under the other generation's host scope.
                    try:
                        hop = self.vault.resolve_credential(service, generation)
                    except Exception:
                        return (
                            f"the '{service}' credential changed mid-request; "
                            "re-connect the agent backend and retry"
                        )
                    if hop[:2] != (placement, cred_name):
                        return (
                            f"the '{service}' credential changed its placement mid-request; "
                            "re-connect the agent backend and retry"
                        )
                    if placement in ("query", "path") and not same_host:
                        # A URL-placed credential binds to the exact
                        # origin it was consented for: a separately
                        # allowlisted sibling origin may be followed,
                        # but never with the credential spliced into
                        # its URL.
                        cred_value = ""
                    else:
                        cred_value = hop[2]
                        if placement == "header":
                            headers[cred_name] = cred_value
                url = next_url
                sent = _spliced_url(url, placement, cred_name, cred_value, path_surrogate)
                resp = session.request(
                    method,
                    sent,
                    headers=headers,
                    data=body,
                    timeout=timeout,
                    allow_redirects=False,
                )
            return resp

    def _serve_connection(self, conn: socket.socket) -> None:
        """Serve one accepted connection: authenticate, read, reply.

        Args:
            conn: Accepted Unix-socket connection.
        """
        with conn:
            # A client sends its single frame immediately after connect;
            # a timeout stops idle connections from pinning threads.
            conn.settimeout(120.0)
            if _peer_uid(conn) != os.getuid():
                # Kernel-authenticated rejection of other users.
                with contextlib.suppress(Exception):
                    send_frame(conn, {"ok": False, "error": "peer uid mismatch"})
                return
            try:
                request = recv_frame(conn)
            except Exception as e:
                with contextlib.suppress(Exception):
                    send_frame(conn, {"ok": False, "error": f"bad frame: {e}"})
                return
            try:
                response = self._handle(request)
            except Exception as e:
                response = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            with contextlib.suppress(Exception):
                send_frame(conn, response)

    def run(self) -> None:
        """Bind the socket and serve until a ``stop`` op arrives."""
        state_dir = muse_auth_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            state_dir.chmod(0o700)
        path = socket_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent == state_dir:
            # Never chmod a shared parent like /tmp (long-path fallback).
            with contextlib.suppress(OSError):
                path.parent.chmod(0o700)
        # The probe-unlink-bind-listen sequence runs under an exclusive
        # file lock (held THROUGH listen) so two simultaneously spawned
        # daemons cannot observe each other bound-but-not-listening and
        # unlink each other's live socket.
        lock_file = open(state_dir / "daemon.lock", "a+b")
        lock_exclusive(lock_file)
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(path))
                probe.close()
                return
            except OSError:
                probe.close()
                with contextlib.suppress(OSError):
                    path.unlink()
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(16)
        finally:
            unlock(lock_file)
            lock_file.close()
        server.settimeout(0.5)
        self._server = server
        # Sweep abandoned candidate-validation scratch files at startup
        # (a validation caller killed before its finally ran would have
        # left one) and periodically thereafter, so a scratch secret's
        # lifetime is bounded even if no later validation ever runs.
        with contextlib.suppress(Exception):
            self.vault.sweep_stale_pending()
        last_sweep = time.monotonic()
        # Bound worker slots cap concurrent connection threads so a
        # same-UID client cannot exhaust threads with a burst of idle
        # connections (each also carries a 120s read timeout).
        slots = threading.BoundedSemaphore(64)
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_sweep >= 300.0:
                    with contextlib.suppress(Exception):
                        self.vault.sweep_stale_pending()
                    last_sweep = time.monotonic()
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                if not slots.acquire(blocking=False):
                    conn.close()
                    continue
                threading.Thread(
                    target=self._serve_with_slot, args=(conn, slots), daemon=True
                ).start()
        finally:
            server.close()
            with contextlib.suppress(OSError):
                path.unlink()

    def _serve_with_slot(self, conn: socket.socket, slots: threading.BoundedSemaphore) -> None:
        """Serve one connection then release its worker slot.

        Args:
            conn: Accepted Unix-socket connection.
            slots: Semaphore whose slot is released when serving ends.
        """
        try:
            self._serve_connection(conn)
        finally:
            slots.release()


def main() -> None:
    """Run the daemon in the foreground (``python -m ...muse_auth.daemon``)."""
    MuseAuthDaemon().run()


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent-side client for the Muse-auth daemon.

Everything the connector agents touch lives here: surrogate-bearing
credential handles, a ``requests``-compatible session and an
``httplib2``-compatible object that both route API traffic through the
daemon's network boundary, and thin wrappers over the daemon ops.
This module never reads the vault and never holds a real token
(enrollment via :func:`store_credentials` hands the token straight to
the daemon and keeps no copy).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any

import httplib2  # type: ignore[import-untyped]
import requests
from requests.structures import CaseInsensitiveDict
from requests.utils import get_encoding_from_headers

from kiss.agents.third_party_agents.muse_auth._common import (
    PROTOCOL_VERSION,
    muse_auth_dir,
    muse_auth_enabled,
    platform_supports_muse_daemon,
    recv_frame,
    send_frame,
    socket_path,
)
from kiss.agents.third_party_agents.muse_auth.sentinel import grant_command
from kiss.core.file_lock import lock_exclusive
from kiss.core.processes import popen_process_group

__all__ = [
    "MuseAuthError",
    "MuseBoundarySession",
    "MuseHttp",
    "SurrogateCredentials",
    "clear_credentials",
    "enrolled_services",
    "grant",
    "mint_surrogate",
    "mint_surrogate_migrating",
    "muse_auth_enabled",
    "platform_supports_muse_daemon",
    "revoke",
    "stop_daemon",
    "store_credentials",
    "vault_has_credentials",
]

_DEFAULT_TIMEOUT = 120.0


class MuseAuthError(RuntimeError):
    """Raised when the Muse-auth daemon cannot serve a request."""


class SurrogateCredentials:
    """Credential handle carrying only a surrogate token.

    Quacks enough like :class:`google.oauth2.credentials.Credentials`
    for the connector backends (``valid``, ``expired``,
    ``refresh_token``, ``token``): the surrogate never expires on the
    agent side because the daemon refreshes the real credential at the
    boundary.

    Attributes:
        service: Connector service the surrogate is bound to.
        token: The opaque surrogate token.
    """

    valid = True
    expired = False
    refresh_token = None

    def __init__(self, service: str, token: str) -> None:
        self.service = service
        self.token = token


def _daemon_running() -> bool:
    """Return whether the daemon socket accepts connections.

    Returns:
        True when a connect() to the socket succeeds; always False where
        the platform has no Unix-domain sockets (Windows), so callers
        such as :func:`stop_daemon` stay clean no-ops there.
    """
    if not hasattr(socket, "AF_UNIX"):
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(socket_path()))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _raw_op(payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any] | None:
    """Send one frame to an already-running daemon without auto-spawn.

    Args:
        payload: Request frame (must include ``op``).
        timeout: Socket timeout in seconds.

    Returns:
        The response frame, or None when no daemon answers (or when the
        platform has no Unix-domain sockets at all).
    """
    if not hasattr(socket, "AF_UNIX"):
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path()))
        send_frame(sock, payload)
        return recv_frame(sock)
    except (OSError, ValueError):
        return None
    finally:
        sock.close()


def _daemon_protocol() -> int | None:
    """Return the running daemon's protocol version.

    Returns:
        The version from its ``status`` reply (pre-versioning daemons
        report none and count as 1), 0 for a listener that answers
        garbage, or None when no daemon is running.
    """
    reply = _raw_op({"op": "status"})
    if reply is None:
        return None
    if not reply.get("ok"):
        return 0
    try:
        return int(reply.get("protocol", 1))
    except (TypeError, ValueError):
        return 0


# Identity of the socket file whose daemon this process verified as
# protocol-compatible.  A daemon binds a fresh socket file, so any
# listener replacement changes the identity (the ctime guards against
# inode-number reuse after unlink) and forces a new handshake; the
# per-op fast path stays a cheap stat + connect probe.
_verified_socket_id: tuple[int, int, int] | None = None


def _socket_id() -> tuple[int, int, int] | None:
    """Return the daemon socket file's identity.

    Returns:
        ``(st_dev, st_ino, st_ctime_ns)``, or None when the socket path
        is absent.
    """
    try:
        stat = os.stat(socket_path())
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_ctime_ns)


def ensure_daemon() -> None:
    """Start a protocol-compatible Muse-auth daemon if none is running.

    A running daemon whose ``status`` reports an older protocol (a
    detached pre-upgrade survivor that would mishandle header-kind
    credentials or enrollment hosts) is stopped first.  Spawns
    ``python -m kiss.agents.third_party_agents.muse_auth.daemon`` fully
    detached (its log goes to ``$KISS_HOME/muse_auth/daemon.log``) and
    waits for a compatible daemon to accept connections.

    Raises:
        MuseAuthError: When the platform has no Unix-domain sockets
            (the daemon's only transport), or when a compatible daemon
            does not come up within 15s.
    """
    global _verified_socket_id
    if not hasattr(socket, "AF_UNIX"):
        raise MuseAuthError(
            "the muse-auth daemon needs Unix-domain sockets, which this platform lacks; "
            "set KISS_MUSE_AUTH=0 to keep connector credentials in the agent process"
        )
    socket_id = _socket_id()
    if socket_id is not None and socket_id == _verified_socket_id and _daemon_running():
        return
    protocol = _daemon_protocol()
    if protocol == PROTOCOL_VERSION:
        _verified_socket_id = _socket_id()
        return
    if protocol is not None:
        # Incompatible daemon: ask it to stop and wait for the socket
        # to die before spawning the current version.
        _raw_op({"op": "stop"})
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _daemon_running():
            time.sleep(0.05)
    directory = muse_auth_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        directory.chmod(0o700)
    with open(directory / "daemon.log", "ab") as log:
        popen_process_group(
            [sys.executable, "-m", "kiss.agents.third_party_agents.muse_auth.daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=os.environ.copy(),
        )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if _daemon_protocol() == PROTOCOL_VERSION:
            _verified_socket_id = _socket_id()
            return
        time.sleep(0.05)
    raise MuseAuthError(
        f"muse-auth daemon did not start; see {directory / 'daemon.log'}"
    )


def _op(payload: dict[str, Any], timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Send one op frame to the daemon and return its response frame.

    Args:
        payload: Request frame (must include ``op``).
        timeout: Socket timeout in seconds.

    Returns:
        The daemon's response frame.

    Raises:
        MuseAuthError: On transport failure.
    """
    ensure_daemon()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout + 30.0)
    try:
        sock.connect(str(socket_path()))
        send_frame(sock, payload)
        return recv_frame(sock)
    except (OSError, ValueError) as e:
        raise MuseAuthError(f"muse-auth daemon transport failure: {e}") from e
    finally:
        sock.close()


def _checked(payload: dict[str, Any], timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Run :func:`_op` and raise on a non-ok response.

    Args:
        payload: Request frame.
        timeout: Socket timeout in seconds.

    Returns:
        The ok response frame.

    Raises:
        MuseAuthError: When the daemon reports an error.
    """
    reply = _op(payload, timeout)
    if not reply.get("ok"):
        raise MuseAuthError(str(reply.get("error", "muse-auth daemon error")))
    return reply


def store_credentials(
    service: str,
    creds: Any,
    scopes: list[str],
    hosts: tuple[str, ...] = (),
    insecure_hosts: tuple[str, ...] = (),
    only_if_absent: bool = False,
) -> bool:
    """Enroll a freshly obtained OAuth credential into the daemon vault.

    The real token crosses into the daemon once, right after the OAuth
    consent flow, and no plaintext copy is kept on the agent side.

    Args:
        service: Connector service name (e.g. ``"gmail"``).
        creds: ``google.oauth2.credentials.Credentials`` or an
            authorized-user info dict.
        scopes: OAuth scopes the credential carries.
        hosts: Extra hostnames the credential may be spent against
            (consent-time allowlist extension for self-hosted bases).
        insecure_hosts: Hostnames the credential may reach over plain
            HTTP (consent-time exception when the user configured an
            ``http://`` base URL, e.g. a LAN Home Assistant instance).
        only_if_absent: When True, the daemon stores ATOMICALLY only if
            the service has no credential yet (auto-migration's
            store-if-absent), so a concurrent authoritative writer
            cannot be clobbered by a stale config candidate.

    Returns:
        True when the daemon wrote the credential; False only when
        *only_if_absent* was set and a credential already existed.
    """
    info = creds if isinstance(creds, dict) else json.loads(creds.to_json())
    frame = {"op": "store_credentials", "service": service,
             "authorized_user_info": info, "scopes": scopes}
    if hosts:
        frame["hosts"] = list(hosts)
    if insecure_hosts:
        frame["insecure_hosts"] = list(insecure_hosts)
    if only_if_absent:
        frame["only_if_absent"] = True
    return bool(_checked(frame).get("created", True))


def vault_has_credentials(service: str) -> bool:
    """Return whether the daemon vault holds a credential for *service*.

    Args:
        service: Connector service name.

    Returns:
        True when the service is enrolled.
    """
    return service in enrolled_services()


def enrolled_services() -> list[str]:
    """Return every service enrolled in the daemon vault.

    Includes workspace-keyed names such as ``slack-<slug>-<hash>``.

    Returns:
        Sorted service names.
    """
    return [str(s) for s in _checked({"op": "status"}).get("services", [])]


def mint_surrogate(service: str) -> SurrogateCredentials | None:
    """Mint a surrogate credential handle for *service*.

    Args:
        service: Connector service name.

    Returns:
        A :class:`SurrogateCredentials`, or ``None`` when the service
        is not enrolled in the vault.
    """
    reply = _op({"op": "mint_surrogate", "service": service})
    if not reply.get("ok"):
        return None
    return SurrogateCredentials(service, str(reply["surrogate"]))


def _migratable_google_info(token_file: Any, scopes: list[str]) -> dict[str, Any] | None:
    """Return the authorized-user info in *token_file* when it is loadable.

    Validated with the same ``google-auth`` parser the legacy loader
    and the daemon vault use, so the migration accepts exactly the
    credentials that worked before the Muse-auth default (including
    quirks like empty-but-present required fields) and keeps every
    other file on disk for diagnosis instead of enrolling a credential
    that can only fail at the boundary.

    Args:
        token_file: :class:`~pathlib.Path` of the legacy ``token.json``.
        scopes: OAuth scopes the connector requests.

    Returns:
        The parsed info dict, or ``None`` when the file is unreadable,
        malformed, or not loadable as an authorized-user credential.
    """
    try:
        info = json.loads(token_file.read_text())
        if not isinstance(info, dict):
            return None
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_info(info, scopes)
        # Match the legacy loader's usability contract, not just its
        # parser: a credential that is neither currently valid nor
        # refreshable (expired with no refresh token) always yielded
        # None before, so vaulting it would only manufacture a
        # surrogate that can never resolve.
        if not creds.valid and not (creds.expired and creds.refresh_token):
            return None
    except Exception:
        return None
    return info


def mint_surrogate_migrating(
    service: str, token_file: Any, scopes: list[str]
) -> SurrogateCredentials | None:
    """Mint a surrogate, migrating a legacy ``token.json`` if needed.

    A user upgrading with a working Google OAuth token on disk but no
    vault enrollment must keep working under the Muse-auth default:
    when the vault has no credential for *service* and *token_file*
    exists, its authorized-user info is enrolled into the vault, the
    plaintext file is removed, and a surrogate is minted — the same
    store-then-unlink migration Google Chat has always done.

    Args:
        service: Connector service name (e.g. ``"google_drive"``).
        token_file: :class:`~pathlib.Path` of the legacy ``token.json``.
        scopes: OAuth scopes the migrated credential carries.

    Returns:
        A :class:`SurrogateCredentials`, or ``None`` when the service
        is not enrolled and no migratable ``token.json`` exists.
    """
    handle = mint_surrogate(service)
    if handle is not None:
        # Deliberately no scrub retry here: a token.json that exists
        # alongside a vault enrollment may be a NEWER credential (the
        # user re-ran a legacy consent flow), and deleting it without
        # migrating would destroy it.  The rare leftover from an
        # unlink that failed in a read-only directory is finished
        # manually with the ``import`` CLI.
        return handle
    if token_file.exists() and _migratable_google_info(token_file, scopes) is not None:
        # Serialize concurrent upgrades on a lock file next to
        # token.json: without it, several first-connect callers each
        # store the same credential (every store starts a new
        # generation, invalidating the surrogates the others just
        # minted) and race on the unlink.  Under the lock exactly one
        # caller migrates; the rest re-mint against the finished
        # enrollment.  The empty lock file is left behind on purpose —
        # unlinking it would reopen the race it exists to close.
        lock_path = token_file.with_name(token_file.name + ".muse-migrate.lock")
        try:
            with open(lock_path, "w") as lock_fh:
                lock_exclusive(lock_fh)
                handle = mint_surrogate(service)
                if handle is None and token_file.exists():
                    info = _migratable_google_info(token_file, scopes)
                    if info is not None:
                        store_credentials(service, info, scopes)
                        handle = mint_surrogate(service)
                        # The vault holds the credential and the handle
                        # is live: a failed unlink (e.g. a directory
                        # turned read-only) must not turn this success
                        # into "no credentials".
                        with contextlib.suppress(OSError):
                            token_file.unlink(missing_ok=True)
        except Exception:
            # A daemon hiccup or an unwritable directory means "no
            # usable credentials" for this caller; the next connect
            # retries the migration.
            handle = None
        if handle is not None:
            return handle
    # Close the pre-lock TOCTOU window: between the first mint and the
    # checks above, a concurrent caller may have finished the migration
    # (the store precedes the unlink, so a vanished token.json implies
    # the enrollment completed).  One final vault check answers for
    # every "could not migrate" path.
    return mint_surrogate(service)


def clear_credentials(service: str) -> None:
    """Remove a service's credential (and surrogates) from the vault.

    Args:
        service: Connector service name.
    """
    _checked({"op": "clear_credentials", "service": service})


def grant(service: str, action: str, scope: str, ttl: float = 0.0) -> str:
    """Record a user approval for asked actions (Muse grant semantics).

    Args:
        service: Connector service name.
        action: ``"read"`` or ``"write"``.
        scope: ``"once"``, ``"session"``, ``"perpetual"``, or ``"ttl"``.
        ttl: Lifetime in seconds for the ``ttl`` scope.

    Returns:
        The new grant's ID.
    """
    reply = _checked({"op": "grant", "service": service, "action": action,
                      "scope": scope, "ttl": ttl})
    return str(reply["grant_id"])


def revoke(service: str, action: str = "") -> int:
    """Revoke grants for a service (optionally one action class).

    Args:
        service: Connector service name.
        action: Action class filter; empty revokes all.

    Returns:
        Number of grants removed.
    """
    return int(_checked({"op": "revoke", "service": service, "action": action})["removed"])


def stop_daemon() -> None:
    """Ask a running daemon to shut down; a no-op when none is running."""
    if _daemon_running():
        with contextlib.suppress(MuseAuthError):
            _op({"op": "stop"})


def _denial_body(message: str) -> bytes:
    """Build a Google-API-shaped JSON error body for a Sentinel denial.

    Args:
        message: Sentinel's reason (includes grant instructions).

    Returns:
        UTF-8 JSON bytes.
    """
    return json.dumps(
        {"error": {"code": 403, "message": message, "status": "MUSE_AUTH_DENIED"}}
    ).encode()


def _boundary_call(
    service: str, method: str, url: str, headers: dict[str, str], body: bytes, timeout: float
) -> dict[str, Any]:
    """Ship one prepared request through the daemon's network boundary.

    Args:
        service: Connector service name.
        method: HTTP method.
        url: Absolute request URL.
        headers: Request headers (surrogate Authorization included).
        body: Request body bytes (may be empty).
        timeout: Request timeout in seconds.

    Returns:
        The daemon's response frame (ok, denied, or error).
    """
    return _op(
        {
            "op": "http_request",
            "service": service,
            "method": method,
            "url": url,
            "headers": headers,
            "body_b64": base64.b64encode(body).decode(),
            "timeout": timeout,
        },
        timeout=timeout,
    )


class MuseBoundarySession:
    """``requests``-compatible session that executes at the daemon boundary.

    Drop-in for the ``requests`` module in the connector backends:
    supports ``request/get/post/put/patch/delete`` and returns real
    :class:`requests.Response` objects.  Sentinel denials come back as
    HTTP 403 responses carrying the grant instructions; daemon
    transport problems raise :class:`MuseAuthError`.
    """

    def __init__(self, service: str) -> None:
        self.service = service

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: Any = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        timeout: float = _DEFAULT_TIMEOUT,
        **_ignored: Any,
    ) -> requests.Response:
        """Prepare and execute one API request through the boundary.

        Args:
            method: HTTP method.
            url: Absolute request URL.
            headers: Request headers (should carry the surrogate bearer).
            params: Optional query parameters.
            json: Optional JSON body.
            data: Optional raw/form body.
            files: Optional ``requests``-style multipart file mapping;
                encoded here so the daemon ships the finished multipart
                body (with its boundary Content-Type) verbatim.
            timeout: Request timeout in seconds.

        Returns:
            A reconstructed :class:`requests.Response`.

        Raises:
            MuseAuthError: When the daemon cannot execute the request.
        """
        prep = requests.Request(
            method=method.upper(), url=url, headers=headers or {}, params=params,
            json=json, data=data, files=files,
        ).prepare()
        body = prep.body or b""
        if isinstance(body, str):
            body = body.encode()
        reply = _boundary_call(
            self.service, method.upper(), str(prep.url),
            {str(k): str(v) for k, v in prep.headers.items()}, body, timeout,
        )
        if not reply.get("ok"):
            if reply.get("denied"):
                return _build_response(
                    403, "Forbidden",
                    {"Content-Type": "application/json"},
                    _denial_body(str(reply.get("error", ""))), str(prep.url),
                )
            raise MuseAuthError(str(reply.get("error", "muse-auth boundary error")))
        return _build_response(
            int(reply["status"]), str(reply.get("reason", "")),
            dict(reply.get("headers", {})),
            base64.b64decode(reply.get("body_b64", "")), str(prep.url),
        )

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue a GET through the boundary; see :meth:`request`."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue a POST through the boundary; see :meth:`request`."""
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue a PUT through the boundary; see :meth:`request`."""
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue a PATCH through the boundary; see :meth:`request`."""
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """Issue a DELETE through the boundary; see :meth:`request`."""
        return self.request("DELETE", url, **kwargs)


def _build_response(
    status: int, reason: str, headers: dict[str, str], content: bytes, url: str
) -> requests.Response:
    """Reconstruct a :class:`requests.Response` from boundary frame data.

    Args:
        status: HTTP status code.
        reason: HTTP reason phrase.
        headers: Response headers (already decoded form).
        content: Response body bytes.
        url: The request URL.

    Returns:
        A fully usable Response (``.json()``, ``.text``, ``.content``).
    """
    resp = requests.Response()
    resp.status_code = status
    resp.reason = reason
    resp.headers = CaseInsensitiveDict(headers)
    resp.url = url
    resp._content = content
    # Mark the body as fully consumed so iter_content() serves the
    # embedded bytes instead of reaching for a raw stream.
    resp._content_consumed = True  # type: ignore[attr-defined]
    resp.encoding = get_encoding_from_headers(resp.headers)
    return resp


class MuseHttp:
    """``httplib2``-compatible transport for ``googleapiclient.build``.

    Injects the surrogate bearer token and routes every request through
    the daemon boundary, so Gmail's ``build("gmail", "v1", http=...)``
    service object works without the agent process ever holding a real
    token.  Sentinel denials surface as HTTP 403 (googleapiclient
    raises ``HttpError`` carrying the grant instructions).
    """

    def __init__(self, service: str, surrogate: str) -> None:
        self.service = service
        self.surrogate = surrogate
        # googleapiclient pokes these attributes on http objects.
        self.timeout = _DEFAULT_TIMEOUT
        self.credentials = None

    def request(
        self,
        uri: str,
        method: str = "GET",
        body: Any = None,
        headers: dict[str, str] | None = None,
        redirections: int = 5,
        connection_type: Any = None,
    ) -> tuple[httplib2.Response, bytes]:
        """Execute one googleapiclient HTTP request via the boundary.

        Args:
            uri: Absolute request URL.
            method: HTTP method.
            body: Request body (str or bytes).
            headers: Request headers.
            redirections: Ignored; the daemon follows redirects.
            connection_type: Ignored (httplib2 signature compatibility).

        Returns:
            ``(httplib2.Response, content_bytes)`` as googleapiclient
            expects.

        Raises:
            MuseAuthError: When the daemon cannot execute the request.
        """
        out_headers = {str(k): str(v) for k, v in (headers or {}).items()}
        if not any(k.lower() == "authorization" for k in out_headers):
            out_headers["Authorization"] = f"Bearer {self.surrogate}"
        raw = body or b""
        if isinstance(raw, str):
            raw = raw.encode()
        reply = _boundary_call(
            self.service, method.upper(), uri, out_headers, raw, self.timeout
        )
        if not reply.get("ok"):
            if reply.get("denied"):
                content = _denial_body(str(reply.get("error", "")))
                info = httplib2.Response(
                    {"status": "403", "content-type": "application/json"}
                )
                info.reason = "Forbidden"
                return info, content
            raise MuseAuthError(str(reply.get("error", "muse-auth boundary error")))
        resp_headers = {k.lower(): v for k, v in dict(reply.get("headers", {})).items()}
        resp_headers["status"] = str(reply["status"])
        info = httplib2.Response(resp_headers)
        info.reason = str(reply.get("reason", ""))
        return info, base64.b64decode(reply.get("body_b64", ""))


def bearer_surrogate(
    service: str,
    legacy_token: str,
    header: str = "",
    hosts: tuple[str, ...] = (),
    insecure_hosts: tuple[str, ...] = (),
) -> str:
    """Return a surrogate for a plain token-authenticated service.

    Prefers an existing vault enrollment; otherwise enrolls
    *legacy_token* (read once from the service's legacy config) into
    the vault so future connects never need the plaintext again
    (delete the legacy file with the ``import`` CLI to finish the
    migration).

    Args:
        service: Connector service name (e.g. ``"notion"``).
        legacy_token: Real token from legacy storage; may be empty.
        header: Header the real token is sent in when it is not an
            ``Authorization: Bearer`` credential (e.g. Brave Search's
            ``X-Subscription-Token``); empty means bearer.
        hosts: Extra hostnames to enroll with the credential (e.g. a
            self-hosted Firecrawl base URL's host).
        insecure_hosts: Hostnames the credential may reach over plain
            HTTP (consent-time exception for ``http://`` base URLs).

    Returns:
        The surrogate token, or ``""`` when the service is not
        enrolled and no legacy token was supplied.
    """
    if not vault_has_credentials(service):
        if not legacy_token:
            return ""
        if header:
            info = {"kind": "header", "header": header, "token": legacy_token}
        else:
            info = {"kind": "bearer", "token": legacy_token}
        store_credentials(service, info, [], hosts=hosts, insecure_hosts=insecure_hosts)
    handle = mint_surrogate(service)
    return handle.token if handle else ""


def approval_hint(service: str, action: str) -> str:
    """Return the user-facing instruction for approving an asked action.

    Args:
        service: Connector service name.
        action: Action class.

    Returns:
        The grant CLI command string.
    """
    return grant_command(service, action)

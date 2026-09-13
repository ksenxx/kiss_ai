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
import fcntl
import os
import re
import socket
import struct
import threading
from typing import Any
from urllib.parse import urlparse

import requests

from kiss.agents.third_party_agents.muse_auth._common import (
    PROTOCOL_VERSION,
    SURROGATE_PREFIX,
    _is_ip_literal,
    canonical_host,
    canonical_host_entry,
    muse_auth_dir,
    recv_frame,
    send_frame,
    socket_path,
    url_origin,
    valid_credential_header,
    valid_credential_value,
    valid_hostname,
    valid_service_name,
)
from kiss.agents.third_party_agents.muse_auth.sentinel import Sentinel
from kiss.agents.third_party_agents.muse_auth.vault import CredentialVault

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


def _peer_uid(conn: socket.socket) -> int:
    """Return the connecting peer's UID via ``SO_PEERCRED``.

    Args:
        conn: Accepted Unix-socket connection.

    Returns:
        The peer's numeric UID.
    """
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
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
            info = request["authorized_user_info"]
            if (
                isinstance(info, dict)
                and info.get("kind") == "header"
                and not valid_credential_header(info.get("header"))
            ):
                return {"ok": False, "error": "invalid credential header name"}
            if (
                isinstance(info, dict)
                and info.get("kind") in ("header", "bearer")
                and not valid_credential_value(info.get("token"))
            ):
                # A malformed value (control chars, stray whitespace)
                # would make the HTTP stack raise errors that reflect
                # the credential verbatim; refuse it at enrollment.
                return {"ok": False, "error": "invalid credential token value"}
            hosts = request.get("hosts", [])
            insecure_hosts = request.get("insecure_hosts", [])
            hosts_error = _invalid_hosts_reason(hosts) or _invalid_hosts_reason(insecure_hosts)
            if hosts_error:
                return {"ok": False, "error": hosts_error}
            self.vault.store(
                request["service"],
                info,
                request.get("scopes", []),
                hosts=[canonical_host_entry(str(h)) for h in hosts],
                insecure_hosts=[canonical_host_entry(str(h)) for h in insecure_hosts],
            )
            return {"ok": True}
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
        decision = self.sentinel.decide(service, method, raw_url, effective_url=url)
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
            cred_header, cred_value = self.vault.resolve_header(service, generation)
        except Exception as e:
            return {"ok": False, "error": f"credential resolution failed: {e}"}
        if not valid_credential_header(cred_header):
            # Defense in depth: enrollment already validates this.
            return {"ok": False, "error": f"vault credential for '{service}' "
                                          f"names an unsafe header"}
        # Preserve the caller's own headers (Notion-Version, Accept,
        # multipart Content-Type, ...); only drop hop-by-hop headers,
        # every Authorization variant, and every copy of the credential
        # header, then set exactly one real credential so a duplicate
        # header cannot smuggle a value past the swap.
        out_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in _HOP_HEADERS
            and k.lower() not in ("authorization", cred_header.lower())
        }
        out_headers[cred_header] = cred_value
        try:
            resp = self._execute(service, method, url, out_headers, body,
                                 float(request.get("timeout", 120.0)),
                                 cred_header=cred_header, generation=generation)
        except Exception as e:
            # Exception text from the HTTP stack can reflect header
            # values verbatim; never let the real credential cross back
            # to the agent inside an error message.
            message = str(e).replace(cred_value, "<redacted-credential>")
            return {"ok": False, "error": f"network boundary request failed: {message}"}
        if isinstance(resp, str):
            return {"ok": False, "denied": True, "error": resp}
        # requests already decoded any content-encoding; drop headers
        # describing the raw wire form so clients trust the body as-is.
        resp_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in _UNDECODED_HEADERS
        }
        return {
            "ok": True,
            "status": resp.status_code,
            "reason": resp.reason or "",
            "headers": resp_headers,
            "body_b64": base64.b64encode(resp.content).decode(),
        }

    def _execute(
        self,
        service: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        cred_header: str = "Authorization",
        generation: str | None = None,
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
            url: Absolute request URL (already authorized for hop 0).
            headers: Outgoing headers including the real credential.
            body: Request body bytes, or None.
            timeout: Per-request timeout in seconds.
            cred_header: Header carrying the real credential
                (``Authorization`` or a header-kind credential's name
                such as ``X-Subscription-Token``).
            generation: Vault credential generation the request is
                pinned to; redirect hops abort when it changes.

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
        with session:
            resp = session.request(
                method, url, headers=headers, data=body,
                timeout=timeout, allow_redirects=False,
            )
            # Follow up to 5 redirects, re-authorizing each hop.
            for _hop in range(5):
                location = resp.headers.get("Location")
                if not (resp.is_redirect and location):
                    return resp
                next_url = requests.compat.urljoin(resp.url, location)  # type: ignore[attr-defined]
                next_parsed = urlparse(next_url)
                prev_parsed = urlparse(url)
                next_host = canonical_host(next_parsed.hostname or "")
                # Same ORIGIN (host and effective port): the same
                # hostname on another port is a different server and
                # must re-qualify through the allowlist.
                same_host = url_origin(
                    next_parsed.scheme, next_parsed.hostname, next_parsed.port
                ) == url_origin(prev_parsed.scheme, prev_parsed.hostname, prev_parsed.port)
                if resp.status_code in (301, 302, 303) and method not in ("GET", "HEAD"):
                    method, body = "GET", None
                headers = dict(headers)
                if not (same_host or self.sentinel.origin_allowed(service, next_url)):
                    # Cross-host redirect off the allowlist: only bodyless
                    # GET/HEAD hops (download CDNs, signed URLs) may be
                    # followed, and never with the real credential.  A
                    # 307/308 keeps the request body, so following it would
                    # ship content to a host Sentinel denied.
                    self.sentinel.decide(service, method, next_url, effective_url=next_url)
                    if method not in ("GET", "HEAD") or body is not None:
                        return (
                            f"cross-host redirect to '{next_host}' would carry the request "
                            f"body off the '{service}' allowlist; refusing to follow it"
                        )
                    headers.pop(cred_header, None)
                else:
                    decision = self.sentinel.decide(service, method, next_url,
                                                    effective_url=next_url)
                    if decision.verdict != "allow":
                        return decision.reason
                    # Re-resolve the credential for every authorized
                    # hop, pinned to the request's vault generation: a
                    # rotation landing mid-request aborts the request
                    # instead of shipping either generation's token
                    # under the other generation's host scope.
                    try:
                        hop_header, hop_value = self.vault.resolve_header(service, generation)
                    except Exception:
                        return (
                            f"the '{service}' credential changed mid-request; "
                            "re-connect the agent backend and retry"
                        )
                    if hop_header != cred_header:
                        return (
                            f"the '{service}' credential changed its header mid-request; "
                            "re-connect the agent backend and retry"
                        )
                    headers[cred_header] = hop_value
                url = next_url
                resp = session.request(
                    method, url, headers=headers, data=body,
                    timeout=timeout, allow_redirects=False,
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
        fcntl.flock(lock_file, fcntl.LOCK_EX)
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
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
        server.settimeout(0.5)
        self._server = server
        # Bound worker slots cap concurrent connection threads so a
        # same-UID client cannot exhaust threads with a burst of idle
        # connections (each also carries a 120s read timeout).
        slots = threading.BoundedSemaphore(64)
        try:
            while not self._stop.is_set():
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

    def _serve_with_slot(
        self, conn: socket.socket, slots: threading.BoundedSemaphore
    ) -> None:
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

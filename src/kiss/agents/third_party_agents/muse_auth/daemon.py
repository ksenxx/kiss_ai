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
import socket
import struct
import threading
from typing import Any
from urllib.parse import urlparse

import requests

from kiss.agents.third_party_agents.muse_auth._common import (
    SURROGATE_PREFIX,
    muse_auth_dir,
    recv_frame,
    send_frame,
    socket_path,
    valid_service_name,
)
from kiss.agents.third_party_agents.muse_auth.sentinel import Sentinel
from kiss.agents.third_party_agents.muse_auth.vault import CredentialVault

_HOP_HEADERS = ("connection", "keep-alive", "transfer-encoding", "content-length", "host")
_UNDECODED_HEADERS = ("content-encoding", "transfer-encoding", "content-length")


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
        self.sentinel = Sentinel()
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
            return {"ok": True, "services": services}
        if op == "store_credentials":
            self.vault.store(
                request["service"], request["authorized_user_info"], request.get("scopes", [])
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
        try:
            real_token = self.vault.resolve_token(service)
        except Exception as e:
            return {"ok": False, "error": f"credential resolution failed: {e}"}
        # Preserve the caller's own headers (Notion-Version, Accept,
        # multipart Content-Type, ...); only drop hop-by-hop headers and
        # every Authorization variant, then set exactly one real bearer
        # so a duplicate header cannot smuggle a value past the swap.
        out_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in _HOP_HEADERS and k.lower() != "authorization"
        }
        out_headers["Authorization"] = f"Bearer {real_token}"
        try:
            resp = self._execute(service, method, url, out_headers, body,
                                 float(request.get("timeout", 120.0)))
        except Exception as e:
            return {"ok": False, "error": f"network boundary request failed: {e}"}
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
    ) -> requests.Response | str:
        """Execute a request, following redirects with per-hop authorization.

        Each hop is evaluated by Sentinel.  The real bearer token is
        only sent to hosts on the service's allowlist; a redirect to any
        other host (e.g. a Google download CDN or signed URL) is
        followed without the credential so the token can never leak off
        an allowlisted host.  Method/body are downgraded to GET on
        301/302/303 exactly as a normal HTTP client would.

        Args:
            service: Connector service name.
            method: HTTP method.
            url: Absolute request URL (already authorized for hop 0).
            headers: Outgoing headers including the real bearer token.
            body: Request body bytes, or None.
            timeout: Per-request timeout in seconds.

        Returns:
            The final :class:`requests.Response`, or a Sentinel denial
            reason string when a redirect hop is refused.
        """
        real_auth = headers.get("Authorization", "")
        resp = requests.request(
            method, url, headers=headers, data=body, timeout=timeout, allow_redirects=False,
        )
        # Follow up to 5 redirects, re-authorizing each hop.
        for _hop in range(5):
            location = resp.headers.get("Location")
            if not (resp.is_redirect and location):
                return resp
            next_url = requests.compat.urljoin(resp.url, location)  # type: ignore[attr-defined]
            next_host = (urlparse(next_url).hostname or "").lower()
            same_host = next_host == (urlparse(url).hostname or "").lower()
            if resp.status_code in (301, 302, 303) and method not in ("GET", "HEAD"):
                method, body = "GET", None
            headers = dict(headers)
            if not (same_host or next_host in self.sentinel.allowed_hosts(service)):
                # Cross-host redirect: never forward the real credential;
                # audit it as a followed egress but do not re-swap a token.
                headers.pop("Authorization", None)
                self.sentinel.decide(service, method, next_url, effective_url=next_url)
            else:
                decision = self.sentinel.decide(service, method, next_url,
                                                effective_url=next_url)
                if decision.verdict != "allow":
                    return decision.reason
                headers["Authorization"] = real_auth
            url = next_url
            resp = requests.request(
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

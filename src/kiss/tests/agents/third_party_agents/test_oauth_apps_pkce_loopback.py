# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the KISS OAuth app registry and the PKCE loopback flow.

Covers ``kiss.agents.third_party_agents._oauth_apps`` and the
``PkceProvider`` / ``LoopbackPkceSession`` / ``_loopback_step`` additions
to ``kiss.agents.third_party_agents._device_auth``.  Every test runs a
real local token endpoint (``http.server`` on a thread) and plays the
browser by issuing real HTTP GETs to the session's loopback callback.

Each test binds its redirect server on a free port through the
``redirect_uri`` constructor parameter instead of the fixed production
port 53682, so parallel test processes never collide.

Unreachable branch: ``redirect.port or 80`` in ``LoopbackPkceSession``
falls back to port 80 for a redirect URI without a port; binding port 80
needs root, so that fallback is not exercised here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, parse_qsl, urlsplit

import pytest
import requests

from kiss.agents.third_party_agents import _oauth_apps
from kiss.agents.third_party_agents._device_auth import (
    ConsentSession,
    LoopbackPkceSession,
    PkceProvider,
    TokenGrant,
    _loopback_step,
    consent_instructions,
)

# ---------------------------------------------------------------------------
# Local token endpoint
# ---------------------------------------------------------------------------


class _TokenHandler(BaseHTTPRequestHandler):
    """Records every POSTed form and answers with the configured reply."""

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length") or 0)
        form = dict(parse_qsl(self.rfile.read(length).decode()))
        server = cast(_TokenServer, self.server)
        server.forms.append(form)
        status, body = server.reply
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence per-request logging."""


class _TokenServer(ThreadingHTTPServer):
    """Token endpoint whose reply a test can choose."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _TokenHandler)
        self.forms: list[dict[str, str]] = []
        self.reply: tuple[int, dict[str, Any]] = (
            200,
            {"access_token": "at-123", "refresh_token": "rt-456", "expires_in": 3600},
        )

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture
def token_server() -> Iterator[_TokenServer]:
    server = _TokenServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def service() -> Iterator[str]:
    """A unique service name whose session is cancelled after the test."""
    name = f"pkce_{uuid.uuid4().hex[:8]}"
    try:
        yield name
    finally:
        ConsentSession.cancel_active(name)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start(
    service: str, server: _TokenServer, port: int, path: str = "/callback", lifetime: float = 30.0
) -> LoopbackPkceSession:
    provider = PkceProvider(
        authorize_url=f"{server.base}/authorize", token_url=f"{server.base}/token"
    )
    session = LoopbackPkceSession(
        service,
        provider,
        "public-client-id",
        f"http://localhost:{port}{path}",
        {"scope": "chat:write users:read", "user_scope": "chat:write"},
        lifetime=lifetime,
    )
    session.register()
    return session


def _query(session: LoopbackPkceSession) -> dict[str, str]:
    parts = urlsplit(session.verification_uri)
    return {k: v[0] for k, v in parse_qs(parts.query).items()}


def _can_bind(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


# ---------------------------------------------------------------------------
# _oauth_apps
# ---------------------------------------------------------------------------


def test_loopback_redirect_constant() -> None:
    assert _oauth_apps.LOOPBACK_PORT == 53682
    assert _oauth_apps.LOOPBACK_REDIRECT_URI == "http://localhost:53682/callback"


def test_oauth_client_id_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KISS_SLACK_CLIENT_ID", "  12345.67890  ")
    assert _oauth_apps.oauth_client_id("slack") == "12345.67890"


def test_oauth_client_id_blank_env_falls_back_to_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KISS_GITHUB_CLIENT_ID", "   ")
    assert _oauth_apps.oauth_client_id("github") == _oauth_apps.KISS_OAUTH_CLIENT_IDS["github"]
    monkeypatch.delenv("KISS_GITHUB_CLIENT_ID")
    assert _oauth_apps.oauth_client_id("github") == _oauth_apps.KISS_OAUTH_CLIENT_IDS["github"]


def test_oauth_client_id_unknown_provider_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KISS_NOSUCHPROVIDER_CLIENT_ID", raising=False)
    assert _oauth_apps.oauth_client_id("nosuchprovider") == ""


def test_missing_client_id_error_names_env_var() -> None:
    message = _oauth_apps.missing_client_id_error("msteams", "Microsoft Teams")
    assert "KISS_MSTEAMS_CLIENT_ID" in message
    assert "Microsoft Teams OAuth app" in message
    assert "_oauth_apps.py" in message


# ---------------------------------------------------------------------------
# LoopbackPkceSession
# ---------------------------------------------------------------------------


def test_pkce_happy_path(token_server: _TokenServer, service: str) -> None:
    port = _free_port()
    session = _start(service, token_server, port)
    query = _query(session)
    assert session.verification_uri.startswith(f"{token_server.base}/authorize?")
    assert query["response_type"] == "code"
    assert query["client_id"] == "public-client-id"
    assert query["redirect_uri"] == f"http://localhost:{port}/callback"
    assert query["code_challenge_method"] == "S256"
    assert query["scope"] == "chat:write users:read"
    assert query["user_scope"] == "chat:write"
    assert session.loopback is True

    resp = requests.get(
        f"http://127.0.0.1:{port}/callback",
        params={"code": "auth-code-1", "state": query["state"]},
        timeout=10,
    )
    assert resp.status_code == 200
    assert "Sign-in received" in resp.text

    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert status == "ok"
    assert finished is session
    assert session.result == {
        "access_token": "at-123",
        "refresh_token": "rt-456",
        "expires_in": 3600,
    }
    assert TokenGrant.from_session(session).access_token == "at-123"

    assert len(token_server.forms) == 1
    form = token_server.forms[0]
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == "auth-code-1"
    assert form["client_id"] == "public-client-id"
    assert form["redirect_uri"] == f"http://localhost:{port}/callback"
    assert "client_secret" not in form
    digest = hashlib.sha256(form["code_verifier"].encode()).digest()
    assert base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == query["code_challenge"]

    # The finished session released the redirect port and its ownership.
    assert LoopbackPkceSession._port_owner is not session
    assert _can_bind(port)


def test_pkce_state_mismatch_refused(token_server: _TokenServer, service: str) -> None:
    port = _free_port()
    _start(service, token_server, port)
    requests.get(
        f"http://127.0.0.1:{port}/callback", params={"code": "c", "state": "forged"}, timeout=10
    )
    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert finished is None
    assert status == "sign-in refused (state mismatch)"
    assert token_server.forms == []


def test_pkce_access_denied_redirect(token_server: _TokenServer, service: str) -> None:
    port = _free_port()
    session = _start(service, token_server, port)
    requests.get(
        f"http://127.0.0.1:{port}/callback",
        params={
            "error": "access_denied",
            "error_description": "The user denied the request",
            "state": _query(session)["state"],
        },
        timeout=10,
    )
    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert finished is None
    assert status == "sign-in refused (access_denied: The user denied the request)"
    assert token_server.forms == []


def test_pkce_slack_style_ok_false(token_server: _TokenServer, service: str) -> None:
    token_server.reply = (200, {"ok": False, "error": "invalid_code"})
    port = _free_port()
    session = _start(service, token_server, port)
    requests.get(
        f"http://127.0.0.1:{port}/callback",
        params={"code": "bad", "state": _query(session)["state"]},
        timeout=10,
    )
    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert finished is None
    assert status == "code exchange refused (invalid_code)"
    assert token_server.forms[0]["code"] == "bad"


def test_pkce_slack_style_ok_true_is_success(token_server: _TokenServer, service: str) -> None:
    body = {"ok": True, "authed_user": {"access_token": "xoxp-1"}}
    token_server.reply = (200, body)
    port = _free_port()
    session = _start(service, token_server, port)
    requests.get(
        f"http://127.0.0.1:{port}/callback",
        params={"code": "good", "state": _query(session)["state"]},
        timeout=10,
    )
    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert status == "ok"
    assert finished is session
    assert session.result == body


def test_pkce_http_error_and_200_error_field(token_server: _TokenServer) -> None:
    for reply, expected in [
        ((400, {"error": "invalid_grant"}), "code exchange refused (invalid_grant)"),
        ((200, {"error": "bad_verifier"}), "code exchange refused (bad_verifier)"),
        ((500, {}), "code exchange refused (HTTP 500)"),
    ]:
        name = f"pkce_{uuid.uuid4().hex[:8]}"
        token_server.reply = reply
        port = _free_port()
        session = _start(name, token_server, port)
        requests.get(
            f"http://127.0.0.1:{port}/callback",
            params={"code": "c", "state": _query(session)["state"]},
            timeout=10,
        )
        finished, status = ConsentSession.finish(name, wait_seconds=15)
        assert finished is None
        assert status == expected


def test_pkce_wrong_path_404_then_callback_completes(
    token_server: _TokenServer, service: str
) -> None:
    port = _free_port()
    session = _start(service, token_server, port)
    wrong = requests.get(f"http://127.0.0.1:{port}/not-the-callback?code=x", timeout=10)
    assert wrong.status_code == 404
    finished, status = ConsentSession.finish(service, wait_seconds=1.0)
    assert (finished, status) == (None, "pending")
    assert session._callback is None

    requests.get(
        f"http://127.0.0.1:{port}/callback",
        params={"code": "c2", "state": _query(session)["state"]},
        timeout=10,
    )
    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert status == "ok"
    assert token_server.forms[0]["code"] == "c2"


def test_pkce_redirect_uri_without_path_uses_root(token_server: _TokenServer, service: str) -> None:
    port = _free_port()
    session = _start(service, token_server, port, path="")
    assert _query(session)["redirect_uri"] == f"http://localhost:{port}"
    requests.get(
        f"http://127.0.0.1:{port}/",
        params={"code": "root", "state": _query(session)["state"]},
        timeout=10,
    )
    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert status == "ok"
    assert token_server.forms[0]["redirect_uri"] == f"http://localhost:{port}"


def test_pkce_cancel_releases_port(token_server: _TokenServer, service: str) -> None:
    port = _free_port()
    first = _start(service, token_server, port)
    assert LoopbackPkceSession._port_owner is first
    first.cancel()
    assert LoopbackPkceSession._port_owner is None
    assert _can_bind(port)

    other = f"{service}_b"
    try:
        second = _start(other, token_server, port)
        assert LoopbackPkceSession._port_owner is second
        requests.get(
            f"http://127.0.0.1:{port}/callback",
            params={"code": "after-cancel", "state": _query(second)["state"]},
            timeout=10,
        )
        finished, status = ConsentSession.finish(other, wait_seconds=15)
        assert status == "ok"
        assert finished is second
    finally:
        ConsentSession.cancel_active(other)


def test_pkce_second_session_replaces_first(token_server: _TokenServer, service: str) -> None:
    port = _free_port()
    first = _start(service, token_server, port)
    other = f"{service}_b"
    try:
        # A second session on the same fixed port (another service) takes
        # the port over; the first is cancelled rather than failing to bind.
        second = _start(other, token_server, port)
        assert first._cancelled is True
        assert LoopbackPkceSession._port_owner is second

        finished, status = ConsentSession.finish(service, wait_seconds=15)
        assert (finished, status) == (None, "sign-in failed")

        requests.get(
            f"http://127.0.0.1:{port}/callback",
            params={"code": "second", "state": _query(second)["state"]},
            timeout=10,
        )
        finished, status = ConsentSession.finish(other, wait_seconds=15)
        assert status == "ok"
        assert finished is second
        assert [f["code"] for f in token_server.forms] == ["second"]
    finally:
        ConsentSession.cancel_active(other)


def test_pkce_port_in_use_raises_oserror(token_server: _TokenServer, service: str) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        provider = PkceProvider(f"{token_server.base}/authorize", f"{token_server.base}/token")
        with pytest.raises(OSError):
            LoopbackPkceSession(service, provider, "cid", f"http://localhost:{port}/callback", {})


def test_pkce_expires_without_redirect(token_server: _TokenServer, service: str) -> None:
    port = _free_port()
    _start(service, token_server, port, lifetime=1.0)
    finished, status = ConsentSession.finish(service, wait_seconds=15)
    assert finished is None
    assert status == "the sign-in request expired before it was approved"
    assert _can_bind(port)


def test_pkce_half_open_connection_does_not_pin_the_port(
    token_server: _TokenServer, service: str
) -> None:
    port = _free_port()
    session = _start(service, token_server, port)
    # A browser that opens the connection but never finishes its request:
    # the handler's socket timeout must return the serving thread to its
    # stop check instead of blocking on the missing headers forever.
    half_open = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        half_open.sendall(b"GET /callback HTTP/1.1\r\n")
        time.sleep(0.5)  # the redirect server has accepted the connection by now
        started = time.monotonic()
        session.cancel()  # joins the serving thread
        assert time.monotonic() - started < 10
        assert not session._server_thread.is_alive()
        assert LoopbackPkceSession._port_owner is None
        assert half_open.recv(1) == b""  # the server closed the half-open connection
        assert _can_bind(port)

        other = f"{service}_b"
        try:
            replacement = _start(other, token_server, port)
            assert LoopbackPkceSession._port_owner is replacement
            requests.get(
                f"http://127.0.0.1:{port}/callback",
                params={"code": "after-half-open", "state": _query(replacement)["state"]},
                timeout=10,
            )
            finished, status = ConsentSession.finish(other, wait_seconds=15)
            assert (finished, status) == (replacement, "ok")
        finally:
            ConsentSession.cancel_active(other)
    finally:
        half_open.close()


def test_pkce_concurrent_constructors_leave_one_port_owner(token_server: _TokenServer) -> None:
    port = _free_port()
    provider = PkceProvider(f"{token_server.base}/authorize", f"{token_server.base}/token")
    redirect_uri = f"http://localhost:{port}/callback"
    sessions: list[LoopbackPkceSession] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def construct(name: str, go: threading.Event) -> None:
        go.wait()
        try:
            session = LoopbackPkceSession(name, provider, "cid", redirect_uri, {})
        except BaseException as e:  # noqa: BLE001 - every failure is a test failure
            with lock:
                errors.append(e)
            return
        with lock:
            sessions.append(session)

    try:
        for round_no in range(20):
            go = threading.Event()
            threads = [
                threading.Thread(target=construct, args=(f"race_{round_no}_{i}", go))
                for i in range(2)
            ]
            for thread in threads:
                thread.start()
            go.set()
            for thread in threads:
                thread.join(30)
                assert not thread.is_alive()
            assert errors == []
            pair = sessions[-2:]
            assert len(pair) == 2
            owner = LoopbackPkceSession._port_owner
            assert owner in pair
            loser = pair[0] if pair[1] is owner else pair[1]
            assert loser._cancelled is True
            assert owner._cancelled is False
            assert not loser._server_thread.is_alive()
            assert owner._server_thread.is_alive()
            # Only the owner serves the port.
            assert requests.get(f"http://127.0.0.1:{port}/nope", timeout=10).status_code == 404
    finally:
        for session in sessions:
            session.cancel()
    assert LoopbackPkceSession._port_owner is None
    assert all(not s._server_thread.is_alive() for s in sessions)
    assert _can_bind(port)


# ---------------------------------------------------------------------------
# _loopback_step / consent_instructions
# ---------------------------------------------------------------------------


def test_loopback_step_only_for_loopback_sessions(token_server: _TokenServer, service: str) -> None:
    plain = ConsentSession("plain", 60.0, 1.0)
    assert _loopback_step(plain) == ""
    plain.verification_uri = "https://example.test/device"
    assert "curl" not in consent_instructions("plain", "Plain", plain)

    session = _start(service, token_server, _free_port())
    step = _loopback_step(session)
    assert "ANOTHER device" in step
    assert "curl -s '<pasted URL>'" in step
    text = consent_instructions(service, "Slack", session, browser_opened=True)
    assert text.endswith(step)
    assert session.verification_uri in text
    assert f"finish_{service}_auth()" in text

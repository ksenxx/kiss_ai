# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the official QQ bot channel agent.

No mocks or test doubles: a real in-process HTTP server emulates the
QQ open-platform API (token + group/C2C message endpoints), and the
backend's embedded webhook server is exercised with real HTTP requests
signed with the same Ed25519 key the backend derives from the secret.
Config state is isolated by the session-wide temp ``KISS_HOME`` set in
``src/kiss/tests/conftest.py``.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kiss.agents.third_party_agents.qq_agent import (
    QQAgent,
    QQChannelBackend,
    _config,
    _derive_signing_key,
    get_tools,
)

_SECRET = "kiss-qq-test-secret"


def _free_port() -> int:
    """Reserve and return a free TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _test_key() -> Ed25519PrivateKey:
    """Derive the Ed25519 key exactly as the official QQ scheme does."""
    seed = _SECRET.encode()
    while len(seed) < 32:
        seed += _SECRET.encode()
    return Ed25519PrivateKey.from_private_bytes(seed[:32])


def _signed_headers(ts: str, raw: bytes) -> dict[str, str]:
    """Build valid Ed25519 webhook signature headers for a raw body."""
    return {
        "X-Signature-Ed25519": _test_key().sign(ts.encode() + raw).hex(),
        "X-Signature-Timestamp": ts,
        "Content-Type": "application/json",
    }


def _raw_post_status(port: int, headers: str, path: str = "/") -> int:
    """Send a raw HTTP POST with custom headers and return the status code."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        request = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{headers}Connection: close\r\n\r\n"
        sock.sendall(request.encode())
        sock.settimeout(10)
        data = b""
        while b"\r\n" not in data:
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk
        return int(data.split(b" ")[1])


class _QQApiHandler(BaseHTTPRequestHandler):
    """Emulates the QQ open-platform token and message endpoints."""

    token_hits = 0
    messages: list[tuple[str, str, dict[str, Any]]] = []

    def do_POST(self) -> None:  # noqa: N802
        """Serve the token endpoint and the v2 message endpoints."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/getAppAccessToken":
            type(self).token_hits += 1
            assert body == {"appId": "qq_app", "clientSecret": _SECRET}
            self._json(200, {"access_token": "QQTOKEN", "expires_in": "7200"})
            return
        assert self.headers.get("Authorization") == "QQBot QQTOKEN"
        type(self).messages.append((self.path, self.headers.get("Authorization", ""), body))
        self._json(200, {"id": "msg1", "timestamp": 1712345678})

    def _json(self, status: int, body: dict[str, Any]) -> None:
        """Send a JSON response."""
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


class _ApiServer:
    """Run the emulated QQ API server for the duration of a test."""

    def __init__(self) -> None:
        _QQApiHandler.token_hits = 0
        _QQApiHandler.messages = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _QQApiHandler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> _ApiServer:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()


def _authenticated_agent(api_base: str = "", token_url: str = "", port: str = "") -> QQAgent:
    """Authenticate a fresh agent against the given endpoints."""
    _config.clear()
    agent = QQAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["authenticate_qq"](
        appid="qq_app",
        secret=_SECRET,
        port=port,
        api_base=api_base,
        token_url=token_url,
    )
    assert json.loads(result)["ok"] is True
    return agent


def test_unauthenticated_state() -> None:
    """Fresh agent exposes only the auth trio and reports unconfigured."""
    _config.clear()
    agent = QQAgent()
    assert agent.name == "QQ Agent"
    assert agent._is_authenticated() is False
    tools = {t.__name__ for t in agent._get_tools()}
    assert tools == {"check_qq_auth", "authenticate_qq", "clear_qq_auth"}
    check = {t.__name__: t for t in agent._get_tools()}["check_qq_auth"]
    assert "not configured" in check().lower()


def test_auth_trio_persistence() -> None:
    """authenticate persists 0600 config; check reports it; clear removes it."""
    _config.clear()
    agent = _authenticated_agent(api_base="http://127.0.0.1:1/")
    assert _config.path.exists()
    if sys.platform != "win32":
        assert _config.path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(_config.path.read_text())
    assert saved["appid"] == "qq_app"
    assert saved["api_base"] == "http://127.0.0.1:1"

    tools = {t.__name__: t for t in agent._get_tools()}
    check = json.loads(tools["check_qq_auth"]())
    assert check["ok"] is True and check["appid"] == "qq_app"
    assert {"send_group_message", "send_c2c_message"} <= set(tools)

    assert "cleared" in tools["clear_qq_auth"]().lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False


def test_get_tools_module_function() -> None:
    """Module-level get_tools() returns a non-empty tool list."""
    _config.clear()
    assert len(get_tools()) >= 3


def test_send_messages_with_cached_token() -> None:
    """Group and C2C sends use QQBot auth and share one cached token."""
    with _ApiServer() as api:
        agent = _authenticated_agent(
            api_base=api.base_url, token_url=f"{api.base_url}/getAppAccessToken"
        )
        backend = agent._backend
        assert json.loads(backend.send_group_message("G1", "hi group"))["ok"] is True
        assert json.loads(backend.send_c2c_message("U1", "hi user"))["ok"] is True
        assert _QQApiHandler.token_hits == 1
        assert _QQApiHandler.messages == [
            ("/v2/groups/G1/messages", "QQBot QQTOKEN", {"content": "hi group", "msg_type": 0}),
            ("/v2/users/U1/messages", "QQBot QQTOKEN", {"content": "hi user", "msg_type": 0}),
        ]
    _config.clear()


def test_webhook_validation_challenge() -> None:
    """The op-13 challenge response verifies with the derived public key."""
    port = _free_port()
    _authenticated_agent(api_base="http://127.0.0.1:1/", port=str(port))
    backend = QQChannelBackend()
    assert backend.connect() is True
    try:
        challenge = json.dumps(
            {"op": 13, "d": {"plain_token": "PT0kEn", "event_ts": "1712345678"}}
        ).encode()
        resp = requests.post(
            f"http://127.0.0.1:{port}/",
            data=challenge,
            headers=_signed_headers("1712345678", challenge),
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["plain_token"] == "PT0kEn"
        public_key = _test_key().public_key()
        public_key.verify(bytes.fromhex(body["signature"]), b"1712345678PT0kEn")
        assert _derive_signing_key(_SECRET).public_key().public_bytes_raw() == (
            public_key.public_bytes_raw()
        )
        unsigned = requests.post(f"http://127.0.0.1:{port}/", data=challenge, timeout=10)
        assert unsigned.status_code == 401
    finally:
        backend.disconnect()
        _config.clear()


def test_webhook_signed_events_and_bad_signature() -> None:
    """Signed events are queued and polled; bad signatures get 401."""
    port = _free_port()
    _authenticated_agent(api_base="http://127.0.0.1:1/", port=str(port))
    backend = QQChannelBackend()
    assert backend.connect() is True
    try:
        url = f"http://127.0.0.1:{port}/"

        group_event = json.dumps(
            {
                "op": 0,
                "t": "GROUP_AT_MESSAGE_CREATE",
                "d": {
                    "id": "MSG_G1",
                    "group_openid": "G42",
                    "content": "  hello from group  ",
                    "timestamp": "1712345001",
                    "author": {"member_openid": "M7"},
                },
            }
        ).encode()
        resp = requests.post(
            url, data=group_event, headers=_signed_headers("1712345002", group_event), timeout=10
        )
        assert resp.status_code == 200 and resp.json() == {"op": 12}

        c2c_event = json.dumps(
            {
                "op": 0,
                "t": "C2C_MESSAGE_CREATE",
                "d": {
                    "id": "MSG_C1",
                    "content": "hi bot",
                    "timestamp": "1712345003",
                    "author": {"user_openid": "U9"},
                },
            }
        ).encode()
        resp = requests.post(
            url, data=c2c_event, headers=_signed_headers("1712345004", c2c_event), timeout=10
        )
        assert resp.status_code == 200 and resp.json() == {"op": 12}

        bad = requests.post(
            url,
            data=c2c_event,
            headers={
                "X-Signature-Ed25519": "ab" * 64,
                "X-Signature-Timestamp": "1712345004",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        assert bad.status_code == 401

        missing = requests.post(url, data=c2c_event, timeout=10)
        assert missing.status_code == 401

        garbage = requests.post(url, data=b"{not json", timeout=10)
        assert garbage.status_code == 401

        signed_garbage = requests.post(
            url, data=b"{not json", headers=_signed_headers("1712345005", b"{not json"), timeout=10
        )
        assert signed_garbage.status_code == 400

        signed_non_dict = requests.post(
            url, data=b"[1, 2]", headers=_signed_headers("1712345006", b"[1, 2]"), timeout=10
        )
        assert signed_non_dict.status_code == 400

        messages, _ = backend.poll_messages("", "0")
        assert messages == [
            {
                "ts": "1712345001",
                "user": "M7",
                "text": "hello from group",
                "channel_id": "G42",
                "msg_id": "MSG_G1",
            },
            {
                "ts": "1712345003",
                "user": "U9",
                "text": "hi bot",
                "channel_id": "U9",
                "msg_id": "MSG_C1",
            },
        ]
        assert "G42" in backend._group_ids
    finally:
        backend.disconnect()
        _config.clear()


def test_webhook_bad_content_length() -> None:
    """Missing/negative/non-decimal Content-Length gets 400; oversized gets 413."""
    port = _free_port()
    _authenticated_agent(api_base="http://127.0.0.1:1/", port=str(port))
    backend = QQChannelBackend()
    assert backend.connect() is True
    try:
        assert _raw_post_status(port, "") == 400
        assert _raw_post_status(port, "Content-Length: -5\r\n") == 400
        assert _raw_post_status(port, "Content-Length: 12abc\r\n") == 400
        assert _raw_post_status(port, f"Content-Length: {2 * 1024 * 1024}\r\n") == 413
        messages, _ = backend.poll_messages("", "0")
        assert messages == []
    finally:
        backend.disconnect()
        _config.clear()


def test_authenticate_rejects_invalid_port() -> None:
    """authenticate_qq rejects non-numeric and out-of-range ports."""
    _config.clear()
    agent = QQAgent()
    authenticate = {t.__name__: t for t in agent._get_tools()}["authenticate_qq"]
    for bad_port in ("abc", "0", "-1", "65536", "1.5", "1e3"):
        result = authenticate(appid="qq_app", secret=_SECRET, port=bad_port)
        assert "invalid port" in result.lower(), bad_port
        assert not _config.path.exists()
    assert json.loads(authenticate(appid="qq_app", secret=_SECRET, port="18086"))["ok"]
    _config.clear()


def test_send_message_routes_group_vs_c2c() -> None:
    """send_message uses the group endpoint for known group channel ids."""
    with _ApiServer() as api:
        agent = _authenticated_agent(
            api_base=api.base_url, token_url=f"{api.base_url}/getAppAccessToken"
        )
        backend = agent._backend
        backend._group_ids.add("G42")
        backend.send_message("G42", "to group")
        backend.send_message("U9", "to user")
        assert [path for path, _, _ in _QQApiHandler.messages] == [
            "/v2/groups/G42/messages",
            "/v2/users/U9/messages",
        ]
    _config.clear()


def test_connect_unconfigured_fails() -> None:
    """connect() fails cleanly when no config is stored."""
    _config.clear()
    backend = QQChannelBackend()
    assert backend.connect() is False
    assert "no qq config" in backend.connection_info.lower()

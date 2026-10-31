# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the official QQ bot channel agent.

No mocks or test doubles: a real in-process HTTP server emulates the
QQ open-platform API (token + group/C2C message endpoints), and the
backend's embedded webhook server is exercised with real HTTP requests
signed with the same Ed25519 key the backend derives from the secret.
Config state is isolated per test by ``isolated_kiss_home``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kiss.agents.third_party_agents.qq.qq_sea import (
    QQAgent,
    QQChannelBackend,
    _config,
    _derive_signing_key,
)
from kiss.tests.agents.third_party_agents.conftest import connect_on_ephemeral_port
from kiss.tests.agents.third_party_agents.muse_test_utils import tools_by_name
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    raw_post_status,
    recording_server,
)

_SECRET = "kiss-qq-test-secret"

pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


def _derive_key(secret: str) -> Ed25519PrivateKey:
    """Derive the Ed25519 key exactly as the official QQ scheme does."""
    seed = secret.encode()
    while len(seed) < 32:
        seed += secret.encode()
    return Ed25519PrivateKey.from_private_bytes(seed[:32])


_TEST_KEY = _derive_key(_SECRET)


def _signed_headers(ts: str, raw: bytes) -> dict[str, str]:
    """Build valid Ed25519 webhook signature headers for a raw body."""
    return {
        "X-Signature-Ed25519": _TEST_KEY.sign(ts.encode() + raw).hex(),
        "X-Signature-Timestamp": ts,
        "Content-Type": "application/json",
    }


_TOKEN_PATH = "/getAppAccessToken"


class _QQApiHandler(RecordingHandler):
    """Emulates the QQ open-platform token and message endpoints.

    Every request is recorded by :meth:`RecordingHandler.record_request`
    with its parsed JSON body; see :func:`_sent`.  Wrong
    credentials are answered like the real API (an error body naming
    what arrived) so the test body, not a handler-thread traceback,
    reports the mismatch.
    """

    def do_POST(self) -> None:  # noqa: N802
        """Serve the token endpoint and the v2 message endpoints."""
        body = json.loads(self._read_raw().decode("utf-8"))
        authorization = self.headers.get("Authorization", "")
        if self.record_request(body)["path"] == _TOKEN_PATH:
            if body != {"appId": "qq_app", "clientSecret": _SECRET}:
                self._json(400, {"code": 100007, "message": f"bad credentials: {body}"})
                return
            self._json(200, {"access_token": "QQTOKEN", "expires_in": "7200"})
            return
        if authorization != "QQBot QQTOKEN":
            self._json(401, {"code": 11244, "message": f"bad token: {authorization!r}"})
            return
        self._json(200, {"id": "msg1", "timestamp": 1712345678})


def _token_hits(api: RecordingServer) -> int:
    """Count the token-endpoint requests the emulated API received."""
    return sum(r["path"] == _TOKEN_PATH for r in api.requests)


def _sent(api: RecordingServer) -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(path, authorization, body)`` of every message request received."""
    return [
        (r["path"], r["authorization"], r["body"]) for r in api.requests if r["path"] != _TOKEN_PATH
    ]


def _authenticated_agent(api_base: str = "", token_url: str = "", port: str = "") -> QQAgent:
    """Authenticate a fresh agent against the given endpoints."""
    agent = QQAgent()
    tools = tools_by_name(agent)
    result = tools["authenticate_qq"](
        appid="qq_app",
        secret=_SECRET,
        port=port,
        api_base=api_base,
        token_url=token_url,
    )
    assert json.loads(result)["ok"] is True
    return agent


def test_auth_trio_persistence() -> None:
    """authenticate persists 0600 config; check reports it; clear removes it.

    The fresh-agent auth-trio contract is pinned for every channel by
    test_base_channel_agent.py; only the name and the unconfigured check
    wording (checked after clear) are channel-specific.
    """
    agent = _authenticated_agent(api_base="http://127.0.0.1:1/")
    assert agent.name == "QQ Agent"
    assert _config.path.exists()
    if sys.platform != "win32":
        assert _config.path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(_config.path.read_text())
    assert saved["appid"] == "qq_app"
    assert saved["api_base"] == "http://127.0.0.1:1"

    tools = tools_by_name(agent)
    check = json.loads(tools["check_qq_auth"]())
    assert check["ok"] is True and check["appid"] == "qq_app"
    assert {"send_group_message", "send_c2c_message"} <= set(tools)

    assert "cleared" in tools["clear_qq_auth"]().lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert "not configured" in tools["check_qq_auth"]().lower()


def test_send_messages_with_cached_token() -> None:
    """Group and C2C sends use QQBot auth and share one cached token."""
    with recording_server(_QQApiHandler) as api:
        agent = _authenticated_agent(api_base=api.base_url, token_url=api.base_url + _TOKEN_PATH)
        backend = agent._backend
        assert json.loads(backend.send_group_message("G1", "hi group"))["ok"] is True
        assert json.loads(backend.send_c2c_message("U1", "hi user"))["ok"] is True
        assert _token_hits(api) == 1
        assert api.requests[0]["body"] == {"appId": "qq_app", "clientSecret": _SECRET}
        assert _sent(api) == [
            ("/v2/groups/G1/messages", "QQBot QQTOKEN", {"content": "hi group", "msg_type": 0}),
            ("/v2/users/U1/messages", "QQBot QQTOKEN", {"content": "hi user", "msg_type": 0}),
        ]


def test_webhook_validation_challenge() -> None:
    """The op-13 challenge response verifies with the derived public key."""
    _authenticated_agent(api_base="http://127.0.0.1:1/")
    backend = QQChannelBackend()
    port = connect_on_ephemeral_port(_config, backend, "_webhook_server")
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
        public_key = _TEST_KEY.public_key()
        public_key.verify(bytes.fromhex(body["signature"]), b"1712345678PT0kEn")
        assert _derive_signing_key(_SECRET).public_key().public_bytes_raw() == (
            public_key.public_bytes_raw()
        )
        unsigned = requests.post(f"http://127.0.0.1:{port}/", data=challenge, timeout=10)
        assert unsigned.status_code == 401
    finally:
        backend.disconnect()


def test_webhook_signed_events_and_bad_signature() -> None:
    """Signed events are queued and polled; bad signatures get 401."""
    _authenticated_agent(api_base="http://127.0.0.1:1/")
    backend = QQChannelBackend()
    port = connect_on_ephemeral_port(_config, backend, "_webhook_server")
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


def test_webhook_bad_content_length() -> None:
    """Missing/negative/non-decimal Content-Length gets 400; oversized gets 413."""
    _authenticated_agent(api_base="http://127.0.0.1:1/")
    backend = QQChannelBackend()
    port = connect_on_ephemeral_port(_config, backend, "_webhook_server")
    try:
        assert raw_post_status(port, "") == 400
        assert raw_post_status(port, "Content-Length: -5\r\n") == 400
        assert raw_post_status(port, "Content-Length: 12abc\r\n") == 400
        assert raw_post_status(port, f"Content-Length: {2 * 1024 * 1024}\r\n") == 413
        messages, _ = backend.poll_messages("", "0")
        assert messages == []
    finally:
        backend.disconnect()


def test_authenticate_rejects_invalid_port() -> None:
    """authenticate_qq rejects non-numeric and out-of-range ports."""
    agent = QQAgent()
    authenticate = tools_by_name(agent)["authenticate_qq"]
    for bad_port in ("abc", "0", "-1", "65536", "1.5", "1e3"):
        result = authenticate(appid="qq_app", secret=_SECRET, port=bad_port)
        assert "invalid port" in result.lower(), bad_port
        assert not _config.path.exists()
    assert json.loads(authenticate(appid="qq_app", secret=_SECRET, port="18086"))["ok"]


def test_send_message_routes_group_vs_c2c() -> None:
    """send_message uses the group endpoint for known group channel ids."""
    with recording_server(_QQApiHandler) as api:
        agent = _authenticated_agent(api_base=api.base_url, token_url=api.base_url + _TOKEN_PATH)
        backend = agent._backend
        backend._group_ids.add("G42")
        backend.send_message("G42", "to group")
        backend.send_message("U9", "to user")
        assert [path for path, _, _ in _sent(api)] == [
            "/v2/groups/G42/messages",
            "/v2/users/U9/messages",
        ]


def test_connect_unconfigured_fails() -> None:
    """connect() fails cleanly when no config is stored."""
    backend = QQChannelBackend()
    assert backend.connect() is False
    assert "no qq config" in backend.connection_info.lower()

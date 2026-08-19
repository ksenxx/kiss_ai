# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the DingTalk channel agent — no mocks or test doubles.

Runs a real local HTTP server as the DingTalk webhook receiver to
assert outbound payload shapes and HMAC query signatures, exercises
the embedded inbound callback server with real signed HTTP POSTs, and
verifies the auth-trio config persistence.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

import kiss.agents.third_party_agents.dingtalk_agent as dingtalk_mod
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer
from kiss.agents.third_party_agents.dingtalk_agent import (
    DingTalkAgent,
    DingTalkChannelBackend,
    get_tools,
)

_AUTH_TRIO = {"check_dingtalk_auth", "authenticate_dingtalk", "clear_dingtalk_auth"}


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dingtalk_sign(key: str, timestamp_ms: str) -> str:
    """Independently recompute DingTalk's base64 HMAC-SHA256 signature."""
    digest = hmac.new(
        key.encode("utf-8"), f"{timestamp_ms}\n{key}".encode(), hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


class _WebhookReceiver:
    """Real local HTTP server standing in for DingTalk's webhook endpoint.

    Records every request's path/query/body and answers with a
    configurable JSON body (``errcode: 0`` by default).
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response_body: dict[str, Any] = {"errcode": 0, "errmsg": "ok"}
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                split = urlsplit(self.path)
                receiver.requests.append(
                    {
                        "path": split.path,
                        "query": parse_qs(split.query),
                        "json": json.loads(body.decode("utf-8")),
                    }
                )
                payload = json.dumps(receiver.response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.server = ThreadedHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/robot/send?access_token=testtoken"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Shut the receiver down."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)


@pytest.fixture()
def receiver() -> Any:
    """A running local webhook receiver, stopped after the test."""
    rec = _WebhookReceiver()
    yield rec
    rec.stop()


@pytest.fixture(autouse=True)
def _clean_config() -> Any:
    """Start and finish every test with no persisted DingTalk config."""
    dingtalk_mod._config.clear()
    yield
    dingtalk_mod._config.clear()


def test_agent_unauthenticated_exposes_only_auth_trio() -> None:
    """Unauthenticated agents expose exactly the auth tool trio."""
    agent = DingTalkAgent()
    assert agent.name == "DingTalk Agent"
    assert agent._is_authenticated() is False
    tool_names = {t.__name__ for t in agent._get_tools()}
    assert tool_names == _AUTH_TRIO
    check = next(t for t in agent._get_tools() if t.__name__ == "check_dingtalk_auth")
    assert "authenticate_dingtalk" in check()


def test_authenticate_persists_and_clear_removes() -> None:
    """authenticate_dingtalk persists 0600 config; clear removes it."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["authenticate_dingtalk"](
        "https://oapi.dingtalk.com/robot/send?access_token=x",
        secret="SECabc",
        outgoing_token="OUTtok",
        port="18099",
    )
    assert json.loads(result)["ok"] is True

    path = dingtalk_mod._config.path
    assert path.exists()
    if sys.platform != "win32":
        assert (path.stat().st_mode & 0o777) == 0o600
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["webhook_url"] == "https://oapi.dingtalk.com/robot/send?access_token=x"
    assert saved["secret"] == "SECabc"
    assert saved["outgoing_token"] == "OUTtok"
    assert saved["port"] == "18099"

    assert agent._is_authenticated() is True
    tool_names = {t.__name__ for t in agent._get_tools()}
    assert {"post_message", "post_markdown"} <= tool_names
    assert json.loads(tools["check_dingtalk_auth"]())["ok"] is True

    assert "cleared" in tools["clear_dingtalk_auth"]().lower()
    assert not path.exists()
    assert agent._is_authenticated() is False
    assert {t.__name__ for t in agent._get_tools()} == _AUTH_TRIO


def test_authenticate_rejects_empty_url_and_bad_port() -> None:
    """Bad authenticate arguments are rejected without persisting config."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "empty" in tools["authenticate_dingtalk"]("   ")
    assert "port" in tools["authenticate_dingtalk"]("https://x", port="not-a-number")
    assert not dingtalk_mod._config.path.exists()


def test_authenticate_rejects_out_of_range_port() -> None:
    """Out-of-range ports are rejected without mutating state or persisting."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    for bad_port in ("0", "-1", "65536", "999999"):
        result = tools["authenticate_dingtalk"]("https://x", port=bad_port)
        assert "1..65535" in result
    assert not dingtalk_mod._config.path.exists()
    assert agent._is_authenticated() is False


def test_fresh_agent_loads_persisted_config() -> None:
    """A new agent instance picks up the persisted config."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_dingtalk"]("https://x?access_token=1", secret="s")
    fresh = DingTalkAgent()
    assert fresh._is_authenticated() is True
    assert fresh._backend._secret == "s"


def test_get_tools_module_function() -> None:
    """The module-level get_tools() returns a non-empty tool list."""
    assert len(get_tools()) >= 3


def test_post_message_shape_and_signed_query(receiver: _WebhookReceiver) -> None:
    """post_message sends the DingTalk text payload with a valid HMAC signature."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_dingtalk"](receiver.url, secret="SECret123")

    before_ms = int(time.time() * 1000)
    raw = agent._backend.post_message("hello 团队", at_mobiles="123, 456", at_all=True)
    after_ms = int(time.time() * 1000)
    result = json.loads(raw)
    assert result["ok"] is True

    req = receiver.requests[-1]
    assert req["json"] == {
        "msgtype": "text",
        "text": {"content": "hello 团队"},
        "at": {"atMobiles": ["123", "456"], "isAtAll": True},
    }
    assert req["query"]["access_token"] == ["testtoken"]
    timestamp = req["query"]["timestamp"][0]
    assert before_ms <= int(timestamp) <= after_ms
    assert req["query"]["sign"][0] == _dingtalk_sign("SECret123", timestamp)


def test_post_markdown_shape(receiver: _WebhookReceiver) -> None:
    """post_markdown sends the DingTalk markdown payload."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_dingtalk"](receiver.url, secret="s2")

    assert json.loads(agent._backend.post_markdown("Title", "# body"))["ok"] is True
    req = receiver.requests[-1]
    assert req["json"] == {"msgtype": "markdown", "markdown": {"title": "Title", "text": "# body"}}
    assert req["query"]["sign"][0] == _dingtalk_sign("s2", req["query"]["timestamp"][0])


def test_send_message_unsigned_without_secret(receiver: _WebhookReceiver) -> None:
    """Without a secret, send_message posts to the raw webhook URL."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_dingtalk"](receiver.url)

    agent._backend.send_message("ignored-channel", "plain text")
    req = receiver.requests[-1]
    assert req["json"] == {"msgtype": "text", "text": {"content": "plain text"}}
    assert "timestamp" not in req["query"]
    assert "sign" not in req["query"]


def test_errcode_nonzero_raises_and_tools_report(receiver: _WebhookReceiver) -> None:
    """A non-zero errcode raises from send_message and yields ok:false tools."""
    agent = DingTalkAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_dingtalk"](receiver.url)
    receiver.response_body = {"errcode": 310000, "errmsg": "sign not match"}

    with pytest.raises(RuntimeError, match="310000"):
        agent._backend.send_message("", "boom")
    assert json.loads(agent._backend.post_message("boom"))["ok"] is False
    assert json.loads(agent._backend.post_markdown("t", "boom"))["ok"] is False


def _post_callback(port: int, payload: dict[str, Any], headers: dict[str, str]) -> int:
    """POST a raw outgoing-robot callback to the embedded server."""
    resp = requests.post(f"http://127.0.0.1:{port}/", json=payload, headers=headers, timeout=10)
    return resp.status_code


def test_inbound_callback_signed_queue_and_poll() -> None:
    """A correctly signed callback is queued, normalized, and pollable."""
    port = _free_port()
    dingtalk_mod._config.save(
        {
            "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=x",
            "secret": "",
            "outgoing_token": "OUTtoken",
            "port": str(port),
        }
    )
    backend = DingTalkChannelBackend()
    assert backend.connect() is True
    try:
        ts = str(int(time.time() * 1000))
        payload = {
            "createAt": 1700000000000,
            "senderStaffId": "staff42",
            "senderId": "raw-id",
            "senderNick": "Alice",
            "conversationId": "cidABC",
            "text": {"content": "@bot do the thing"},
        }
        status = _post_callback(
            port, payload, {"timestamp": ts, "sign": _dingtalk_sign("OUTtoken", ts)}
        )
        assert status == 200

        messages, oldest = backend.poll_messages("cidABC", "0")
        assert oldest == "0"
        assert messages == [
            {
                "ts": "1700000000000",
                "user": "staff42",
                "username": "Alice",
                "text": "@bot do the thing",
                "channel_id": "cidABC",
            }
        ]
    finally:
        backend.disconnect()


def test_inbound_callback_bad_sign_and_stale_timestamp_rejected() -> None:
    """Wrong signatures and >1h-old timestamps get 401 and are not queued."""
    port = _free_port()
    dingtalk_mod._config.save(
        {
            "webhook_url": "https://x?access_token=1",
            "secret": "",
            "outgoing_token": "OUTtoken",
            "port": str(port),
        }
    )
    backend = DingTalkChannelBackend()
    assert backend.connect() is True
    try:
        payload = {"conversationId": "c", "text": {"content": "hi"}}
        ts = str(int(time.time() * 1000))
        assert _post_callback(port, payload, {"timestamp": ts, "sign": "bogus"}) == 401
        assert _post_callback(port, payload, {"timestamp": "not-a-number", "sign": "x"}) == 401
        stale = str(int(time.time() * 1000) - 2 * 3600 * 1000)
        assert (
            _post_callback(
                port, payload, {"timestamp": stale, "sign": _dingtalk_sign("OUTtoken", stale)}
            )
            == 401
        )
        messages, _ = backend.poll_messages("", "0")
        assert messages == []
    finally:
        backend.disconnect()


def test_inbound_callback_unverified_when_no_token_and_channel_filter() -> None:
    """Without outgoing_token callbacks are accepted; poll filters by channel."""
    port = _free_port()
    dingtalk_mod._config.save(
        {
            "webhook_url": "https://x?access_token=1",
            "secret": "",
            "outgoing_token": "",
            "port": str(port),
        }
    )
    backend = DingTalkChannelBackend()
    assert backend.connect() is True
    try:
        for cid, sender in (("c1", "u1"), ("c2", "u2")):
            payload = {
                "createAt": 1,
                "senderId": sender,
                "senderNick": sender,
                "conversationId": cid,
                "text": {"content": "hi"},
            }
            assert _post_callback(port, payload, {}) == 200
        messages, _ = backend.poll_messages("c2", "0")
        assert [m["channel_id"] for m in messages] == ["c2"]
        assert messages[0]["user"] == "u2"
    finally:
        backend.disconnect()


def _raw_http_post(port: int, content_length: str | None, body: bytes = b"") -> str:
    """POST over a raw socket with a hand-crafted Content-Length header.

    Returns the HTTP status line of the response.
    """
    lines = ["POST / HTTP/1.1", "Host: 127.0.0.1", "Connection: close"]
    if content_length is not None:
        lines.append(f"Content-Length: {content_length}")
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8") + body
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        data = b""
        while chunk := sock.recv(4096):
            data += chunk
    return data.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")


def test_inbound_callback_bad_content_length_handled_and_server_stays_up() -> None:
    """Bad Content-Length values get 400/413 without an unbounded read.

    Negative, non-decimal, missing, and >1 MiB Content-Length headers are
    rejected before the body is read, and the server keeps serving valid
    signed callbacks afterwards.
    """
    port = _free_port()
    dingtalk_mod._config.save(
        {
            "webhook_url": "https://x?access_token=1",
            "secret": "",
            "outgoing_token": "OUTtoken",
            "port": str(port),
        }
    )
    backend = DingTalkChannelBackend()
    assert backend.connect() is True
    try:
        assert "400" in _raw_http_post(port, "-1")
        assert "400" in _raw_http_post(port, "nope")
        assert "400" in _raw_http_post(port, None)
        assert "413" in _raw_http_post(port, str(2 * 1024 * 1024))

        ts = str(int(time.time() * 1000))
        payload = {
            "createAt": 1,
            "senderId": "u1",
            "senderNick": "U1",
            "conversationId": "c1",
            "text": {"content": "still alive"},
        }
        status = _post_callback(
            port, payload, {"timestamp": ts, "sign": _dingtalk_sign("OUTtoken", ts)}
        )
        assert status == 200
        messages, _ = backend.poll_messages("c1", "0")
        assert [m["text"] for m in messages] == ["still alive"]
    finally:
        backend.disconnect()


def test_connect_fails_without_config() -> None:
    """connect() fails cleanly when no config is persisted."""
    backend = DingTalkChannelBackend()
    assert backend.connect() is False
    assert "No DingTalk config" in backend.connection_info

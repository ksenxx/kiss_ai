# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Weixin (WeChat Official Account) channel agent.

No mocks or test doubles: a real in-process HTTP server emulates the
WeChat API (token + custom-send + user-info endpoints), and the
backend's own callback server is exercised with real HTTP requests.
Config state is isolated by the session-wide temp ``KISS_HOME`` set in
``src/kiss/tests/conftest.py``.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from kiss.agents.third_party_agents.weixin_agent import (
    WeixinAgent,
    WeixinChannelBackend,
    _config,
    get_tools,
)


def _free_port() -> int:
    """Reserve and return a free TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _signed_params(ts: str = "1712345678", nonce: str = "n0nce") -> dict[str, str]:
    """Build valid signature query parameters for the 'cbtok' callback token."""
    sig = hashlib.sha1("".join(sorted(["cbtok", ts, nonce])).encode()).hexdigest()
    return {"signature": sig, "timestamp": ts, "nonce": nonce}


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


class _WeixinApiHandler(BaseHTTPRequestHandler):
    """Emulates the WeChat Official Account HTTP API."""

    token_hits = 0
    send_hits = 0
    sent_payloads: list[dict[str, Any]] = []
    send_errcode = 0
    send_http_status = 200

    def do_GET(self) -> None:  # noqa: N802
        """Serve the token and user-info endpoints."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/cgi-bin/token":
            type(self).token_hits += 1
            assert params.get("grant_type") == ["client_credential"]
            self._json(200, {"access_token": "WXTOKEN", "expires_in": 7200})
        elif parsed.path == "/cgi-bin/user/info":
            assert params.get("access_token") == ["WXTOKEN"]
            self._json(200, {"openid": params.get("openid", [""])[0], "nickname": "Alice"})
        else:
            self._json(404, {"errcode": 404, "errmsg": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        """Serve the custom-send endpoint."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if parsed.path == "/cgi-bin/message/custom/send":
            assert params.get("access_token") == ["WXTOKEN"]
            type(self).send_hits += 1
            type(self).sent_payloads.append(body)
            if type(self).send_http_status != 200:
                self._json(type(self).send_http_status, {})
            elif type(self).send_errcode:
                self._json(200, {"errcode": type(self).send_errcode, "errmsg": "denied"})
            else:
                self._json(200, {"errcode": 0, "errmsg": "ok"})
        else:
            self._json(404, {"errcode": 404, "errmsg": "not found"})

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
    """Run the emulated WeChat API server for the duration of a test."""

    def __init__(self) -> None:
        _WeixinApiHandler.token_hits = 0
        _WeixinApiHandler.send_hits = 0
        _WeixinApiHandler.sent_payloads = []
        _WeixinApiHandler.send_errcode = 0
        _WeixinApiHandler.send_http_status = 200
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _WeixinApiHandler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> _ApiServer:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()


def _authenticated_agent(
    api_base: str = "", port: str = "", callback_token: str = "cbtok"
) -> WeixinAgent:
    """Authenticate a fresh agent against the given API base."""
    _config.clear()
    agent = WeixinAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["authenticate_weixin"](
        appid="wx_app",
        appsecret="wx_secret",
        callback_token=callback_token,
        port=port,
        api_base=api_base,
    )
    assert json.loads(result)["ok"] is True
    return agent


def test_unauthenticated_state() -> None:
    """Fresh agent exposes only the auth trio and reports unconfigured."""
    _config.clear()
    agent = WeixinAgent()
    assert agent.name == "Weixin Agent"
    assert agent._is_authenticated() is False
    tools = {t.__name__ for t in agent._get_tools()}
    assert tools == {"check_weixin_auth", "authenticate_weixin", "clear_weixin_auth"}
    check = {t.__name__: t for t in agent._get_tools()}["check_weixin_auth"]
    assert "not configured" in check().lower()


def test_auth_trio_persistence() -> None:
    """authenticate persists 0600 config; check reports it; clear removes it."""
    _config.clear()
    agent = _authenticated_agent(api_base="http://127.0.0.1:1/")
    assert _config.path.exists()
    if sys.platform != "win32":
        assert _config.path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(_config.path.read_text())
    assert saved["appid"] == "wx_app"
    assert saved["api_base"] == "http://127.0.0.1:1"

    tools = {t.__name__: t for t in agent._get_tools()}
    check = json.loads(tools["check_weixin_auth"]())
    assert check["ok"] is True and check["appid"] == "wx_app"
    assert {"send_text_message", "get_user_info"} <= set(tools)

    assert "cleared" in tools["clear_weixin_auth"]().lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False


def test_get_tools_module_function() -> None:
    """Module-level get_tools() returns a non-empty tool list."""
    _config.clear()
    assert len(get_tools()) >= 3


def test_send_reuses_cached_token() -> None:
    """Two sends fetch the access token only once (in-memory cache)."""
    with _ApiServer() as api:
        agent = _authenticated_agent(api_base=api.base_url)
        backend = agent._backend
        assert json.loads(backend.send_text_message("openid1", "hi"))["ok"] is True
        assert json.loads(backend.send_text_message("openid2", "there"))["ok"] is True
        assert _WeixinApiHandler.token_hits == 1
        assert _WeixinApiHandler.send_hits == 2
        assert _WeixinApiHandler.sent_payloads[0] == {
            "touser": "openid1",
            "msgtype": "text",
            "text": {"content": "hi"},
        }
    _config.clear()


def test_send_message_raises_on_errcode() -> None:
    """send_message raises RuntimeError when the API reports an errcode."""
    with _ApiServer() as api:
        agent = _authenticated_agent(api_base=api.base_url)
        _WeixinApiHandler.send_errcode = 45015
        try:
            agent._backend.send_message("openid1", "hi")
            raise AssertionError("send_message should have raised RuntimeError")
        except RuntimeError as e:
            assert "45015" in str(e)
        err = json.loads(agent._backend.send_text_message("openid1", "hi"))
        assert err["ok"] is False and "45015" in err["error"]
    _config.clear()


def test_get_user_info() -> None:
    """get_user_info returns the profile from the user-info endpoint."""
    with _ApiServer() as api:
        agent = _authenticated_agent(api_base=api.base_url)
        result = json.loads(agent._backend.get_user_info("openid9"))
        assert result["ok"] is True
        assert result["user"]["openid"] == "openid9"
        assert result["user"]["nickname"] == "Alice"
    _config.clear()


def test_callback_verification_and_inbound_xml() -> None:
    """The callback server handles GET verification and normalizes POST XML."""
    port = _free_port()
    _authenticated_agent(api_base="http://127.0.0.1:1/", port=str(port))
    backend = WeixinChannelBackend()
    assert backend.connect() is True
    try:
        url = f"http://127.0.0.1:{port}/"
        ts, nonce = "1712345678", "n0nce"
        good = hashlib.sha1("".join(sorted(["cbtok", ts, nonce])).encode()).hexdigest()
        resp = requests.get(
            url,
            params={"signature": good, "timestamp": ts, "nonce": nonce, "echostr": "e-c-h-o"},
            timeout=10,
        )
        assert resp.status_code == 200 and resp.text == "e-c-h-o"

        bad = requests.get(
            url,
            params={"signature": "0" * 40, "timestamp": ts, "nonce": nonce, "echostr": "x"},
            timeout=10,
        )
        assert bad.status_code == 401

        xml = (
            "<xml><ToUserName><![CDATA[gh_acct]]></ToUserName>"
            "<FromUserName><![CDATA[openid42]]></FromUserName>"
            "<CreateTime>1712345678</CreateTime>"
            "<MsgType><![CDATA[text]]></MsgType>"
            "<Content><![CDATA[hello kiss]]></Content>"
            "<MsgId>9001</MsgId></xml>"
        )
        posted = requests.post(url, params=_signed_params(), data=xml.encode("utf-8"), timeout=10)
        assert posted.status_code == 200 and posted.text == "success"

        dtd = '<?xml version="1.0"?><!DOCTYPE xml [<!ENTITY x "y">]><xml></xml>'
        assert (
            requests.post(url, params=_signed_params(), data=dtd.encode(), timeout=10).status_code
            == 200
        )
        assert (
            requests.post(
                url, params=_signed_params(), data=b"not xml at all", timeout=10
            ).status_code
            == 200
        )

        messages, _ = backend.poll_messages("", "0")
        assert messages == [
            {
                "ts": "1712345678",
                "user": "openid42",
                "text": "hello kiss",
                "channel_id": "openid42",
                "msg_type": "text",
                "msg_id": "9001",
            }
        ]
    finally:
        backend.disconnect()
        _config.clear()


def test_callback_post_requires_signature_when_token_configured() -> None:
    """Unsigned or badly signed POSTs are rejected with 401 and never queued."""
    port = _free_port()
    _authenticated_agent(api_base="http://127.0.0.1:1/", port=str(port))
    backend = WeixinChannelBackend()
    assert backend.connect() is True
    try:
        url = f"http://127.0.0.1:{port}/"
        xml = b"<xml><FromUserName>evil</FromUserName><Content>inject</Content></xml>"
        unsigned = requests.post(url, data=xml, timeout=10)
        assert unsigned.status_code == 401
        bad = requests.post(
            url,
            params={"signature": "0" * 40, "timestamp": "1712345678", "nonce": "n0nce"},
            data=xml,
            timeout=10,
        )
        assert bad.status_code == 401
        messages, _ = backend.poll_messages("", "0")
        assert messages == []
    finally:
        backend.disconnect()
        _config.clear()


def test_callback_post_without_token_allows_unsigned() -> None:
    """With no callback token configured, unsigned POSTs are still queued."""
    port = _free_port()
    _authenticated_agent(api_base="http://127.0.0.1:1/", port=str(port), callback_token="")
    backend = WeixinChannelBackend()
    assert backend.connect() is True
    try:
        xml = (
            "<xml><FromUserName>openid7</FromUserName><CreateTime>1</CreateTime>"
            "<MsgType>text</MsgType><Content>hi</Content><MsgId>42</MsgId></xml>"
        )
        posted = requests.post(f"http://127.0.0.1:{port}/", data=xml.encode(), timeout=10)
        assert posted.status_code == 200
        messages, _ = backend.poll_messages("", "0")
        assert len(messages) == 1 and messages[0]["msg_id"] == "42"
    finally:
        backend.disconnect()
        _config.clear()


def test_callback_post_bad_content_length() -> None:
    """Missing/negative/non-decimal Content-Length gets 400; oversized gets 413."""
    port = _free_port()
    _authenticated_agent(api_base="http://127.0.0.1:1/", port=str(port), callback_token="")
    backend = WeixinChannelBackend()
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


def test_send_fails_on_http_error_status() -> None:
    """A non-200 HTTP response is a failure even when the body has no errcode."""
    with _ApiServer() as api:
        agent = _authenticated_agent(api_base=api.base_url)
        _WeixinApiHandler.send_http_status = 500
        err = json.loads(agent._backend.send_text_message("openid1", "hi"))
        assert err["ok"] is False and "HTTP 500" in err["error"]
        try:
            agent._backend.send_message("openid1", "hi")
            raise AssertionError("send_message should have raised RuntimeError")
        except RuntimeError as e:
            assert "HTTP 500" in str(e)
    _config.clear()


def test_authenticate_rejects_invalid_port() -> None:
    """authenticate_weixin rejects non-numeric and out-of-range ports."""
    _config.clear()
    agent = WeixinAgent()
    authenticate = {t.__name__: t for t in agent._get_tools()}["authenticate_weixin"]
    for bad_port in ("abc", "0", "-1", "65536", "1.5", "1e3"):
        result = authenticate(appid="wx_app", appsecret="wx_secret", port=bad_port)
        assert "invalid port" in result.lower(), bad_port
        assert not _config.path.exists()
    assert json.loads(authenticate(appid="wx_app", appsecret="wx_secret", port="18085"))["ok"]
    _config.clear()


def test_poll_messages_filters_by_channel() -> None:
    """poll_messages keeps only messages for the requested channel_id."""
    backend = WeixinChannelBackend()
    backend._message_queue.put({"ts": "1", "user": "a", "text": "t1", "channel_id": "a"})
    backend._message_queue.put({"ts": "2", "user": "b", "text": "t2", "channel_id": "b"})
    messages, _ = backend.poll_messages("b", "0")
    assert [m["user"] for m in messages] == ["b"]


def test_connect_unconfigured_fails() -> None:
    """connect() fails cleanly when no config is stored."""
    _config.clear()
    backend = WeixinChannelBackend()
    assert backend.connect() is False
    assert "no weixin config" in backend.connection_info.lower()

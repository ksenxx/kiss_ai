# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Weixin Agent — channel agent for WeChat Official Accounts.

Provides access to a WeChat Official Account via the customer-service
message API and a plaintext callback server for inbound messages.
Stores config in ``~/.kiss/third_party_agents/weixin/config.json``.

Outbound messages use the cached ``cgi-bin/token`` access token and the
``cgi-bin/message/custom/send`` endpoint.  Inbound messages arrive on an
embedded HTTP callback server: GET requests answer the WeChat URL
verification handshake (SHA-1 over the sorted ``[token, timestamp,
nonce]`` triple), POST requests carry plaintext WeChat XML which is
normalized and queued for :meth:`WeixinChannelBackend.poll_messages`.
When a callback token is configured, POST requests must carry the same
valid query-string signature as the GET handshake or they are rejected
with 401.  XML bodies containing DTDs are dropped without parsing.

Usage::

    agent = WeixinAgent()
    agent.run(prompt_template="Send 'Hello!' to openid oX1234")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import queue
import sys
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    drain_queue_messages,
    drain_request_body,
    stop_http_server,
)
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_CALLBACK_PORT = "18085"
_DEFAULT_API_BASE = "https://api.weixin.qq.com"
_MAX_BODY_BYTES = 1024 * 1024

_WEIXIN_DIR = Path.home() / ".kiss" / "third_party_agents" / "weixin"
_config = ChannelConfig(_WEIXIN_DIR, ("appid", "appsecret"))


def _verification_signature(token: str, timestamp: str, nonce: str) -> str:
    """Compute the WeChat callback verification signature.

    Args:
        token: The configured callback token.
        timestamp: The ``timestamp`` query parameter.
        nonce: The ``nonce`` query parameter.

    Returns:
        SHA-1 hex digest over the sorted, concatenated triple.
    """
    return hashlib.sha1("".join(sorted([token, timestamp, nonce])).encode("utf-8")).hexdigest()


def _parse_content_length(value: str | None) -> int | None:
    """Parse a ``Content-Length`` header value.

    Args:
        value: The raw header value, or None if the header is missing.

    Returns:
        The non-negative integer length, or None when the header is
        missing, negative, not a plain decimal integer, or longer than
        10 digits (an absurd length; unguarded ``int()`` would raise
        past Python's integer-string conversion limit on a header of
        thousands of digits).
    """
    if value is None:
        return None
    value = value.strip()
    if not value.isascii() or not value.isdigit() or len(value) > 10:
        return None
    return int(value)


def _is_valid_port(port: str) -> bool:
    """Return True if ``port`` is a decimal integer between 1 and 65535."""
    return port.isascii() and port.isdigit() and 1 <= int(port) <= 65535


class WeixinChannelBackend(ToolMethodBackend):
    """Channel backend for the WeChat Official Account platform.

    Sends messages via the customer-service message API (with an
    in-memory cached access token) and receives messages via an
    embedded plaintext callback HTTP server.
    """

    def __init__(self) -> None:
        self._appid: str = ""
        self._appsecret: str = ""
        self._callback_token: str = ""
        self._port: str = _DEFAULT_CALLBACK_PORT
        self._api_base: str = _DEFAULT_API_BASE
        self._access_token: str = ""
        self._token_expiry: float = 0.0
        self._token_lock = threading.Lock()
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._callback_server: ThreadedHTTPServer | None = None
        self._callback_thread: threading.Thread | None = None
        self._connection_info: str = ""

    def _apply_config(self, cfg: dict[str, str]) -> None:
        """Copy persisted config values onto the backend."""
        self._appid = cfg["appid"]
        self._appsecret = cfg["appsecret"]
        self._callback_token = cfg.get("callback_token", "")
        self._port = cfg.get("port", "") or _DEFAULT_CALLBACK_PORT
        self._api_base = (cfg.get("api_base", "") or _DEFAULT_API_BASE).rstrip("/")

    def connect(self) -> bool:
        """Load the Weixin config and start the inbound callback server."""
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No Weixin config found."
            return False
        self._apply_config(cfg)
        self._connection_info = f"Weixin configured for appid {self._appid}"
        if not self._start_callback_server():
            return False
        return True

    def _start_callback_server(self) -> bool:
        """Start the WeChat callback HTTP server on the configured port."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                """Answer the WeChat URL verification handshake."""
                params = parse_qs(urlparse(self.path).query)
                signature = params.get("signature", [""])[0]
                timestamp = params.get("timestamp", [""])[0]
                nonce = params.get("nonce", [""])[0]
                echostr = params.get("echostr", [""])[0]
                expected = _verification_signature(backend._callback_token, timestamp, nonce)
                if not hmac.compare_digest(expected, signature):
                    self.send_response(401)
                    self.end_headers()
                    return
                body = echostr.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                """Authenticate and queue an inbound plaintext WeChat XML message.

                A rejection path with a parseable ``Content-Length``
                drains the (bounded) unread request body after sending
                the response — see ``drain_request_body`` — so the
                client reads the status code instead of a connection
                reset.
                """
                length = _parse_content_length(self.headers.get("Content-Length"))
                if backend._callback_token:
                    params = parse_qs(urlparse(self.path).query)
                    signature = params.get("signature", [""])[0]
                    timestamp = params.get("timestamp", [""])[0]
                    nonce = params.get("nonce", [""])[0]
                    expected = _verification_signature(backend._callback_token, timestamp, nonce)
                    if not hmac.compare_digest(expected, signature):
                        self.send_response(401)
                        self.end_headers()
                        drain_request_body(self, length)
                        return
                if length is None:
                    self.send_response(400)
                    self.end_headers()
                    return
                if length > _MAX_BODY_BYTES:
                    self.send_response(413)
                    self.end_headers()
                    drain_request_body(self, length)
                    return
                raw = self.rfile.read(length)
                backend._queue_xml_message(raw)
                body = b"success"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        try:
            self._callback_server = ThreadedHTTPServer(("0.0.0.0", int(self._port)), Handler)
            self._callback_thread = threading.Thread(
                target=self._callback_server.serve_forever, daemon=True
            )
            self._callback_thread.start()
            logger.info("Weixin callback server started on port %s", self._port)
            return True
        except (OSError, ValueError, OverflowError) as e:
            self._connection_info = f"Weixin callback bind failed: {e}"
            logger.warning("Could not start Weixin callback server: %s", e)
            self._callback_server = None
            self._callback_thread = None
            return False

    def _queue_xml_message(self, raw: bytes) -> None:
        """Parse a plaintext WeChat XML payload and queue the message.

        DTD-bearing payloads are dropped without parsing (entity-expansion
        defusing); malformed XML is silently ignored.
        """
        if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
            logger.warning("Dropped Weixin callback payload containing a DTD")
            return
        try:
            root = ET.fromstring(raw.decode("utf-8"))
        except (ET.ParseError, UnicodeDecodeError):
            logger.warning("Dropped malformed Weixin callback XML")
            return
        if root.tag != "xml":
            return
        fields = {child.tag: (child.text or "") for child in root}
        from_user = fields.get("FromUserName", "")
        self._message_queue.put(
            {
                "ts": fields.get("CreateTime", ""),
                "user": from_user,
                "text": fields.get("Content", ""),
                "channel_id": from_user,
                "msg_type": fields.get("MsgType", ""),
                "msg_id": fields.get("MsgId", ""),
            }
        )

    def _get_access_token(self) -> str:
        """Return a cached Official Account access token, refreshing if stale.

        Raises:
            RuntimeError: If the token endpoint returns an error.
        """
        with self._token_lock:
            now = time.time()
            if self._access_token and now < self._token_expiry:
                return self._access_token
            resp = requests.get(
                f"{self._api_base}/cgi-bin/token",
                params={
                    "grant_type": "client_credential",
                    "appid": self._appid,
                    "secret": self._appsecret,
                },
                timeout=30,
            )
            data = resp.json()
            token = data.get("access_token", "")
            if not token:
                raise RuntimeError(f"Weixin access token fetch failed: {data}")
            self._access_token = str(token)
            self._token_expiry = now + float(data.get("expires_in", 7200)) - 60.0
            return self._access_token

    def _send_custom_text(self, openid: str, text: str) -> dict[str, Any]:
        """Send a customer-service text message, raising on API errors.

        Raises:
            RuntimeError: If the HTTP status is not 200 or the API
                reports a non-zero ``errcode``.
        """
        token = self._get_access_token()
        resp = requests.post(
            f"{self._api_base}/cgi-bin/message/custom/send",
            params={"access_token": token},
            json={"touser": openid, "msgtype": "text", "text": {"content": text}},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Weixin send failed: HTTP {resp.status_code}")
        data: dict[str, Any] = resp.json()
        if data.get("errcode", 0) != 0:
            raise RuntimeError(f"Weixin send failed: {data}")
        return data

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the callback message queue.

        Drained messages not matching ``channel_id`` are discarded.
        """
        messages = drain_queue_messages(
            self._message_queue,
            limit=limit,
            keep=lambda msg: not channel_id or msg.get("channel_id") == channel_id,
        )
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a text message to a follower's openid.

        WeChat customer-service messages have no threads, so
        ``thread_ts`` is ignored.

        Raises:
            RuntimeError: If the API reports a non-zero ``errcode``.
        """
        self._send_custom_text(channel_id, text)

    def disconnect(self) -> None:
        """Stop the embedded callback server and release backend resources."""
        self._callback_server, self._callback_thread = stop_http_server(
            self._callback_server, self._callback_thread
        )

    def send_text_message(self, openid: str, text: str) -> str:
        """Send a text message to a WeChat follower.

        Args:
            openid: The follower's openid.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            self._send_custom_text(openid, text)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_user_info(self, openid: str) -> str:
        """Fetch a WeChat follower's profile.

        Args:
            openid: The follower's openid.

        Returns:
            JSON string with ok status and the user profile.
        """
        try:
            token = self._get_access_token()
            resp = requests.get(
                f"{self._api_base}/cgi-bin/user/info",
                params={"access_token": token, "openid": openid, "lang": "en"},
                timeout=30,
            )
            data = resp.json()
            if data.get("errcode", 0) != 0:
                return json.dumps({"ok": False, "error": str(data)})
            return json.dumps({"ok": True, "user": data})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class WeixinAgent(BaseChannelAgent):
    """Channel agent with WeChat Official Account tools."""

    channel_system_prompt = (
        "You are chatting through a WeChat Official Account. Messages are "
        "plain text sent to follower openids via the customer-service API; "
        "there are no threads or rich formatting."
    )

    def __init__(self) -> None:
        super().__init__("Weixin Agent")
        self._backend = WeixinChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._apply_config(cfg)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._appid and self._backend._appsecret)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_weixin_auth() -> str:
            """Check if the WeChat Official Account is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for Weixin. Use authenticate_weixin() to configure.\n"
                    "You need the AppID and AppSecret of a WeChat Official Account "
                    "from https://mp.weixin.qq.com (Settings > Basic Configuration)."
                )
            return json.dumps(
                {
                    "ok": True,
                    "appid": agent._backend._appid,
                    "port": agent._backend._port,
                    "api_base": agent._backend._api_base,
                }
            )

        def authenticate_weixin(
            appid: str,
            appsecret: str,
            callback_token: str = "",
            port: str = "",
            api_base: str = "",
        ) -> str:
            """Configure the WeChat Official Account credentials.

            Args:
                appid: Official Account AppID.
                appsecret: Official Account AppSecret.
                callback_token: Optional token for callback URL verification.
                port: Optional inbound callback server port (default 18085).
                api_base: Optional API base URL (default https://api.weixin.qq.com).

            Returns:
                Configuration result or error message.
            """
            if not appid.strip() or not appsecret.strip():
                return "appid and appsecret cannot be empty."
            port_value = port.strip() or _DEFAULT_CALLBACK_PORT
            if not _is_valid_port(port_value):
                return f"Invalid port {port_value!r}: must be an integer between 1 and 65535."
            cfg = {
                "appid": appid.strip(),
                "appsecret": appsecret.strip(),
                "callback_token": callback_token.strip(),
                "port": port_value,
                "api_base": (api_base.strip() or _DEFAULT_API_BASE).rstrip("/"),
            }
            _config.save(cfg)
            agent._backend._apply_config(cfg)
            return json.dumps({"ok": True, "message": "Weixin configured."})

        def clear_weixin_auth() -> str:
            """Clear the stored Weixin configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._appid = ""
            agent._backend._appsecret = ""
            agent._backend._callback_token = ""
            agent._backend._access_token = ""
            agent._backend._token_expiry = 0.0
            return "Weixin configuration cleared."

        return [check_weixin_auth, authenticate_weixin, clear_weixin_auth]


def _make_backend() -> WeixinChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = WeixinChannelBackend()
    cfg = _config.load()
    if not cfg:
        print("Not configured. Run: kiss-weixin -t 'authenticate'")
        sys.exit(1)
    backend._apply_config(cfg)
    return backend


def main() -> None:
    """Run the WeixinAgent from the command line with chat persistence."""
    channel_main(
        WeixinAgent,
        "kiss-weixin",
        channel_name="Weixin",
        make_backend=_make_backend,
    )


def get_tools() -> list:
    """Return the Weixin channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return WeixinAgent()._get_tools()


if __name__ == "__main__":
    main()

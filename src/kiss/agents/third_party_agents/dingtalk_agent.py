# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""DingTalk Agent — channel agent for DingTalk group robots.

Sends messages through a DingTalk custom-robot incoming webhook
(optionally signed with the robot's ``secret``) and receives messages
through an embedded HTTP callback server for DingTalk outgoing robots
(verified with the robot's ``outgoing_token`` HMAC signature).
Stores config in ``~/.kiss/third_party_agents/dingtalk/config.json``.

Usage::

    agent = DingTalkAgent()
    agent.run(prompt_template="Send a message to the team")
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    drain_queue_messages,
    start_http_server,
    stop_http_server,
)
from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_PORT = 18084
_MAX_TIMESTAMP_SKEW_MS = 60 * 60 * 1000
_MAX_CALLBACK_BODY_BYTES = 1024 * 1024

_DINGTALK_DIR = Path.home() / ".kiss" / "third_party_agents" / "dingtalk"
_config = ChannelConfig(_DINGTALK_DIR, ("webhook_url",))


def _compute_sign(secret: str, timestamp_ms: str) -> str:
    """Compute the DingTalk HMAC-SHA256 signature.

    Both the custom-robot outbound signature and the outgoing-robot
    inbound signature use ``base64(HMAC-SHA256(key, "{timestamp}\\n{key}"))``.

    Args:
        secret: Robot secret (outbound) or outgoing token (inbound).
        timestamp_ms: Millisecond timestamp string.

    Returns:
        Base64-encoded signature string.
    """
    string_to_sign = f"{timestamp_ms}\n{secret}"
    digest = hmac.new(
        secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _port_from_config(cfg: dict[str, Any]) -> int:
    """Return the configured callback port, falling back to the default.

    Defensively handles non-numeric and out-of-range values persisted in
    the config file.

    Args:
        cfg: Loaded DingTalk config dictionary.

    Returns:
        A valid port in 1..65535 (``_DEFAULT_WEBHOOK_PORT`` on invalid input).
    """
    try:
        port = int(cfg.get("port") or _DEFAULT_WEBHOOK_PORT)
    except ValueError:
        return _DEFAULT_WEBHOOK_PORT
    if not 1 <= port <= 65535:
        return _DEFAULT_WEBHOOK_PORT
    return port


class DingTalkChannelBackend(ToolMethodBackend):
    """Channel backend for DingTalk group robots.

    Sends messages via the custom-robot incoming webhook URL, signing
    requests when a ``secret`` is configured.  Receives messages via an
    embedded HTTP server acting as the outgoing-robot callback URL,
    verifying the ``sign`` header when an ``outgoing_token`` is
    configured.
    """

    def __init__(self) -> None:
        self._webhook_url: str = ""
        self._secret: str = ""
        self._outgoing_token: str = ""
        self._port: int = _DEFAULT_WEBHOOK_PORT
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load DingTalk config and start the outgoing-robot callback server."""
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No DingTalk config found."
            return False
        self._webhook_url = cfg["webhook_url"]
        self._secret = cfg.get("secret", "")
        self._outgoing_token = cfg.get("outgoing_token", "")
        self._port = _port_from_config(cfg)
        self._connection_info = "DingTalk configured"
        return self._start_webhook_server(self._port)

    def _start_webhook_server(self, port: int) -> bool:
        """Start the outgoing-robot callback HTTP server."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                raw_length = str(self.headers.get("Content-Length", "")).strip()
                if not raw_length.isdecimal():
                    self.send_response(400)
                    self.end_headers()
                    return
                length = int(raw_length)
                if length > _MAX_CALLBACK_BODY_BYTES:
                    self.send_response(413)
                    self.end_headers()
                    return
                body = self.rfile.read(length)
                if not backend._verify_inbound(
                    str(self.headers.get("timestamp", "")),
                    str(self.headers.get("sign", "")),
                ):
                    self.send_response(401)
                    self.end_headers()
                    return
                try:
                    payload: dict[str, Any] = json.loads(body.decode("utf-8"))
                    text = payload.get("text") or {}
                    backend._message_queue.put(
                        {
                            "ts": str(payload.get("createAt", "")),
                            "user": str(
                                payload.get("senderStaffId") or payload.get("senderId") or ""
                            ),
                            "username": str(payload.get("senderNick", "")),
                            "text": str(text.get("content", "")),
                            "channel_id": str(payload.get("conversationId", "")),
                        }
                    )
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        self._webhook_server, self._webhook_thread, error = start_http_server(
            ("0.0.0.0", port),
            Handler,
            log=logger,
            started_log="DingTalk callback server started on port %d",
            error_prefix="DingTalk callback bind failed",
            error_log="Could not start DingTalk callback server: %s",
        )
        if error is not None:
            self._connection_info = error
            return False
        return True

    def _verify_inbound(self, timestamp_ms: str, sign: str) -> bool:
        """Verify an outgoing-robot callback signature and timestamp skew.

        Accepts everything when no ``outgoing_token`` is configured.
        """
        if not self._outgoing_token:
            return True
        try:
            ts = int(timestamp_ms)
        except ValueError:
            return False
        if abs(int(time.time() * 1000) - ts) > _MAX_TIMESTAMP_SKEW_MS:
            return False
        expected = _compute_sign(self._outgoing_token, timestamp_ms)
        return hmac.compare_digest(expected, sign)

    def _signed_webhook_url(self) -> str:
        """Return the webhook URL, signed when a secret is configured."""
        if not self._secret:
            return self._webhook_url
        timestamp_ms = str(int(time.time() * 1000))
        sign = quote_plus(_compute_sign(self._secret, timestamp_ms))
        return f"{self._webhook_url}&timestamp={timestamp_ms}&sign={sign}"

    def _post_payload(self, payload: dict[str, Any]) -> None:
        """POST a message payload to the incoming webhook.

        Raises:
            RuntimeError: If the HTTP request fails or DingTalk returns
                a non-zero ``errcode``.
        """
        with self._send_lock:
            resp = requests.post(self._signed_webhook_url(), json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"DingTalk send failed: HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("errcode", 0) != 0:
            raise RuntimeError(f"DingTalk send failed: {data}")

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the outgoing-robot callback message queue.

        Drained messages not matching ``channel_id`` are discarded.
        """
        messages = drain_queue_messages(
            self._message_queue,
            limit=limit,
            keep=lambda msg: not channel_id or msg.get("channel_id") == channel_id,
        )
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a DingTalk text message via the incoming webhook.

        Custom-robot webhooks are bound to a fixed group chat at
        creation, so ``channel_id`` and ``thread_ts`` are ignored.

        Raises:
            RuntimeError: If the webhook request fails or DingTalk
                returns a non-zero ``errcode``.
        """
        self._post_payload({"msgtype": "text", "text": {"content": text}})

    def disconnect(self) -> None:
        """Stop the embedded callback server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

    def post_message(self, text: str, at_mobiles: str = "", at_all: bool = False) -> str:
        """Send a text message to the DingTalk group via the robot webhook.

        Args:
            text: Message text.
            at_mobiles: Comma-separated mobile numbers to @-mention (optional).
            at_all: Whether to @-mention everyone in the group.

        Returns:
            JSON string with ok status.
        """
        try:
            payload: dict[str, Any] = {
                "msgtype": "text",
                "text": {"content": text},
                "at": {
                    "atMobiles": [m.strip() for m in at_mobiles.split(",") if m.strip()],
                    "isAtAll": at_all,
                },
            }
            self._post_payload(payload)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def post_markdown(self, title: str, text: str) -> str:
        """Send a markdown message to the DingTalk group via the robot webhook.

        Args:
            title: Message title shown in the chat list.
            text: Markdown-formatted message body.

        Returns:
            JSON string with ok status.
        """
        try:
            payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
            self._post_payload(payload)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class DingTalkAgent(BaseChannelAgent):
    """Channel agent with DingTalk group-robot tools."""

    channel_system_prompt = (
        "You are chatting via a DingTalk group robot. Messages support "
        "plain text and DingTalk-flavored markdown; use post_message for "
        "text (with optional @-mentions by mobile number) and "
        "post_markdown for formatted content."
    )

    def __init__(self) -> None:
        super().__init__("DingTalk Agent")
        self._backend = DingTalkChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._webhook_url = cfg["webhook_url"]
            self._backend._secret = cfg.get("secret", "")
            self._backend._outgoing_token = cfg.get("outgoing_token", "")
            self._backend._port = _port_from_config(cfg)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._webhook_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_dingtalk_auth() -> str:
            """Check if DingTalk is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._webhook_url:
                return (
                    "Not configured for DingTalk. Use authenticate_dingtalk() to configure.\n"
                    "You need the custom robot webhook URL from DingTalk group settings > "
                    "Group Assistant > Add Robot > Custom, plus its signing secret if "
                    "signature security is enabled."
                )
            return json.dumps(
                {
                    "ok": True,
                    "webhook_url": agent._backend._webhook_url[:50] + "...",
                }
            )

        def authenticate_dingtalk(
            webhook_url: str, secret: str = "", outgoing_token: str = "", port: str = ""
        ) -> str:
            """Configure the DingTalk group robot.

            Args:
                webhook_url: DingTalk custom-robot incoming webhook URL.
                secret: Optional robot signing secret for outbound requests.
                outgoing_token: Optional outgoing-robot token for verifying
                    inbound callback signatures.
                port: Optional local port for the inbound callback server
                    (default 18084).

            Returns:
                Configuration result or error message.
            """
            if not webhook_url.strip():
                return "webhook_url cannot be empty."
            try:
                port_num = int(port.strip() or _DEFAULT_WEBHOOK_PORT)
            except ValueError:
                return f"port must be an integer, got {port!r}."
            if not 1 <= port_num <= 65535:
                return f"port must be an integer in 1..65535, got {port!r}."
            agent._backend._webhook_url = webhook_url.strip()
            agent._backend._secret = secret.strip()
            agent._backend._outgoing_token = outgoing_token.strip()
            agent._backend._port = port_num
            _config.save(
                {
                    "webhook_url": webhook_url.strip(),
                    "secret": secret.strip(),
                    "outgoing_token": outgoing_token.strip(),
                    "port": str(agent._backend._port),
                }
            )
            return json.dumps({"ok": True, "message": "DingTalk configured."})

        def clear_dingtalk_auth() -> str:
            """Clear the stored DingTalk configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._webhook_url = ""
            agent._backend._secret = ""
            agent._backend._outgoing_token = ""
            agent._backend._port = _DEFAULT_WEBHOOK_PORT
            return "DingTalk configuration cleared."

        return [check_dingtalk_auth, authenticate_dingtalk, clear_dingtalk_auth]


def _make_backend() -> DingTalkChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = DingTalkChannelBackend()
    cfg = _config.load()
    if not cfg:
        print("Not configured. Run: kiss-dingtalk -t 'authenticate'")
        sys.exit(1)
    backend._webhook_url = cfg["webhook_url"]
    backend._secret = cfg.get("secret", "")
    backend._outgoing_token = cfg.get("outgoing_token", "")
    backend._port = _port_from_config(cfg)
    return backend


def main() -> None:
    """Run the DingTalkAgent from the command line with chat persistence."""
    channel_main(
        DingTalkAgent,
        "kiss-dingtalk",
        channel_name="DingTalk",
        make_backend=_make_backend,
    )


def get_tools() -> list:
    """Return the DingTalk channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return DingTalkAgent()._get_tools()


if __name__ == "__main__":
    main()

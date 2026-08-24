# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""QQ Agent — channel agent for the official QQ bot platform.

Provides access to a QQ bot via the official open-platform HTTP API
(group and C2C messages) and an embedded Ed25519-verified webhook
server for inbound events.  Stores config in
``~/.kiss/third_party_agents/qq/config.json``.

Outbound messages use a cached app access token (``Authorization:
QQBot <token>``).  Inbound events arrive on the webhook server, which
verifies the ``X-Signature-Ed25519`` header of every request (including
the op-13 URL-validation challenge, which it answers) with a key pair
derived from the bot secret (the secret repeated to 32 bytes is the
Ed25519 signing seed, per the official QQ bot documentation).

Usage::

    agent = QQAgent()
    agent.run(prompt_template="Send 'Hello!' to group ABCD1234")
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

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

_DEFAULT_WEBHOOK_PORT = "18086"
_DEFAULT_API_BASE = "https://api.sgroup.qq.com"
_DEFAULT_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
_MAX_BODY_BYTES = 1024 * 1024

_OP_DISPATCH = 0
_OP_HTTP_CALLBACK_ACK = 12
_OP_VALIDATION = 13

_QQ_DIR = Path.home() / ".kiss" / "third_party_agents" / "qq"
_config = ChannelConfig(_QQ_DIR, ("appid", "secret"))


def _derive_signing_key(secret: str) -> Ed25519PrivateKey:
    """Derive the webhook Ed25519 signing key from the bot secret.

    Per the official QQ bot documentation, the secret is repeated until
    it reaches 32 bytes and truncated to 32 bytes to form the seed.

    Args:
        secret: The bot's app secret.

    Returns:
        The derived Ed25519 private key.
    """
    seed = secret.encode("utf-8")
    while len(seed) < 32:
        seed += secret.encode("utf-8")
    return Ed25519PrivateKey.from_private_bytes(seed[:32])


def _parse_content_length(value: str | None) -> int | None:
    """Parse a ``Content-Length`` header value.

    Args:
        value: The raw header value, or None if the header is missing.

    Returns:
        The non-negative integer length, or None when the header is
        missing, negative, or not a plain decimal integer.
    """
    if value is None:
        return None
    value = value.strip()
    if not value.isascii() or not value.isdigit():
        return None
    return int(value)


def _is_valid_port(port: str) -> bool:
    """Return True if ``port`` is a decimal integer between 1 and 65535."""
    return port.isascii() and port.isdigit() and 1 <= int(port) <= 65535


class QQChannelBackend(ToolMethodBackend):
    """Channel backend for the official QQ bot open platform.

    Sends group and C2C messages via the v2 HTTP API (with an
    in-memory cached app access token) and receives events via an
    embedded Ed25519-verified webhook HTTP server.
    """

    def __init__(self) -> None:
        self._appid: str = ""
        self._secret: str = ""
        self._port: str = _DEFAULT_WEBHOOK_PORT
        self._api_base: str = _DEFAULT_API_BASE
        self._token_url: str = _DEFAULT_TOKEN_URL
        self._access_token: str = ""
        self._token_expiry: float = 0.0
        self._token_lock = threading.Lock()
        self._private_key: Ed25519PrivateKey | None = None
        self._public_key: Ed25519PublicKey | None = None
        self._group_ids: set[str] = set()
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._connection_info: str = ""

    def _apply_config(self, cfg: dict[str, str]) -> None:
        """Copy persisted config values onto the backend."""
        self._appid = cfg["appid"]
        self._secret = cfg["secret"]
        self._port = cfg.get("port", "") or _DEFAULT_WEBHOOK_PORT
        self._api_base = (cfg.get("api_base", "") or _DEFAULT_API_BASE).rstrip("/")
        self._token_url = cfg.get("token_url", "") or _DEFAULT_TOKEN_URL
        self._private_key = _derive_signing_key(self._secret)
        self._public_key = self._private_key.public_key()

    def connect(self) -> bool:
        """Load the QQ config and start the inbound webhook server."""
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No QQ config found."
            return False
        self._apply_config(cfg)
        self._connection_info = f"QQ bot configured for appid {self._appid}"
        if not self._start_webhook_server():
            return False
        return True

    def _start_webhook_server(self) -> bool:
        """Start the QQ webhook HTTP server on the configured port."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                """Validate and dispatch an inbound QQ webhook payload."""
                length = _parse_content_length(self.headers.get("Content-Length"))
                if length is None:
                    self.send_response(400)
                    self.end_headers()
                    return
                if length > _MAX_BODY_BYTES:
                    self.send_response(413)
                    self.end_headers()
                    return
                raw = self.rfile.read(length)
                if not backend._verify_event_signature(
                    self.headers.get("X-Signature-Ed25519", ""),
                    self.headers.get("X-Signature-Timestamp", ""),
                    raw,
                ):
                    self.send_response(401)
                    self.end_headers()
                    return
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._respond(400, {"message": "invalid JSON"})
                    return
                if not isinstance(payload, dict):
                    self._respond(400, {"message": "invalid payload"})
                    return
                if payload.get("op") == _OP_VALIDATION:
                    self._respond(200, backend._sign_validation(payload.get("d") or {}))
                    return
                if payload.get("op") == _OP_DISPATCH:
                    backend._queue_event(payload)
                self._respond(200, {"op": _OP_HTTP_CALLBACK_ACK})

            def _respond(self, status: int, body: dict[str, Any]) -> None:
                """Send a JSON response."""
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        self._webhook_server, self._webhook_thread, error = start_http_server(
            ("0.0.0.0", self._port),
            Handler,
            log=logger,
            started_log="QQ webhook server started on port %s",
            error_prefix="QQ webhook bind failed",
            error_log="Could not start QQ webhook server: %s",
            catch=(OSError, ValueError, OverflowError),
        )
        if error is not None:
            self._connection_info = error
            return False
        return True

    def _sign_validation(self, data: dict[str, Any]) -> dict[str, str]:
        """Answer the op-13 URL-validation challenge.

        Args:
            data: The challenge payload (``plain_token``, ``event_ts``).

        Returns:
            Dict with the plain token and the hex Ed25519 signature over
            ``event_ts + plain_token``.
        """
        assert self._private_key is not None
        plain_token = str(data.get("plain_token", ""))
        event_ts = str(data.get("event_ts", ""))
        signature = self._private_key.sign((event_ts + plain_token).encode("utf-8")).hex()
        return {"plain_token": plain_token, "signature": signature}

    def _verify_event_signature(self, signature_hex: str, timestamp: str, body: bytes) -> bool:
        """Verify a webhook event's Ed25519 signature.

        Args:
            signature_hex: The ``X-Signature-Ed25519`` header (hex).
            timestamp: The ``X-Signature-Timestamp`` header.
            body: The raw request body.

        Returns:
            True if the signature over ``timestamp + body`` is valid.
        """
        if self._public_key is None or not signature_hex or not timestamp:
            return False
        try:
            self._public_key.verify(bytes.fromhex(signature_hex), timestamp.encode("utf-8") + body)
            return True
        except Exception:
            return False

    def _queue_event(self, payload: dict[str, Any]) -> None:
        """Normalize and queue a dispatch (op 0) event."""
        event_type = str(payload.get("t", ""))
        data = payload.get("d") or {}
        author = data.get("author") or {}
        if event_type == "GROUP_AT_MESSAGE_CREATE":
            channel_id = str(data.get("group_openid", ""))
            user = str(author.get("member_openid", ""))
            self._group_ids.add(channel_id)
        elif event_type == "C2C_MESSAGE_CREATE":
            user = str(author.get("user_openid", ""))
            channel_id = user
        else:
            return
        self._message_queue.put(
            {
                "ts": str(data.get("timestamp", "")) or str(data.get("id", "")),
                "user": user,
                "text": str(data.get("content", "")).strip(),
                "channel_id": channel_id,
                "msg_id": str(data.get("id", "")),
            }
        )

    def _get_access_token(self) -> str:
        """Return a cached app access token, refreshing if stale.

        Raises:
            RuntimeError: If the token endpoint returns an error.
        """
        with self._token_lock:
            now = time.time()
            if self._access_token and now < self._token_expiry:
                return self._access_token
            resp = requests.post(
                self._token_url,
                json={"appId": self._appid, "clientSecret": self._secret},
                timeout=30,
            )
            data = resp.json()
            token = data.get("access_token", "")
            if not token:
                raise RuntimeError(f"QQ access token fetch failed: {data}")
            self._access_token = str(token)
            self._token_expiry = now + float(data.get("expires_in", 7200)) - 60.0
            return self._access_token

    def _post_text(self, path: str, text: str) -> dict[str, Any]:
        """POST a plain-text message to a v2 message endpoint.

        Raises:
            RuntimeError: If the API reports an error.
        """
        token = self._get_access_token()
        resp = requests.post(
            f"{self._api_base}{path}",
            json={"content": text, "msg_type": 0},
            headers={"Authorization": f"QQBot {token}"},
            timeout=30,
        )
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400 or (isinstance(data, dict) and data.get("code")):
            raise RuntimeError(f"QQ send failed: HTTP {resp.status_code} {data}")
        return data if isinstance(data, dict) else {}

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue.

        Drained messages not matching ``channel_id`` are discarded.
        """
        messages = drain_queue_messages(
            self._message_queue,
            limit=limit,
            keep=lambda msg: not channel_id or msg.get("channel_id") == channel_id,
        )
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a text message to a group or C2C conversation.

        ``channel_id`` values seen in group webhook events are routed to
        the group endpoint; anything else is treated as a user openid.
        QQ bot messages have no threads, so ``thread_ts`` is ignored.

        Raises:
            RuntimeError: If the API reports an error.
        """
        if channel_id in self._group_ids:
            self._post_text(f"/v2/groups/{channel_id}/messages", text)
        else:
            self._post_text(f"/v2/users/{channel_id}/messages", text)

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

    def send_group_message(self, group_openid: str, text: str) -> str:
        """Send a text message to a QQ group.

        Args:
            group_openid: The group's openid.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            self._post_text(f"/v2/groups/{group_openid}/messages", text)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_c2c_message(self, openid: str, text: str) -> str:
        """Send a text message to a QQ user (C2C).

        Args:
            openid: The user's openid.
            text: Message text.

        Returns:
            JSON string with ok status.
        """
        try:
            self._post_text(f"/v2/users/{openid}/messages", text)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class QQAgent(BaseChannelAgent):
    """Channel agent with official QQ bot tools."""

    channel_system_prompt = (
        "You are chatting through an official QQ bot. Messages are plain "
        "text sent to group openids or user openids; there are no threads."
    )

    def __init__(self) -> None:
        super().__init__("QQ Agent")
        self._backend = QQChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._apply_config(cfg)

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._appid and self._backend._secret)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_qq_auth() -> str:
            """Check if the QQ bot is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for QQ. Use authenticate_qq() to configure.\n"
                    "You need the AppID and AppSecret of a QQ bot from "
                    "https://q.qq.com (Developer Settings)."
                )
            return json.dumps(
                {
                    "ok": True,
                    "appid": agent._backend._appid,
                    "port": agent._backend._port,
                    "api_base": agent._backend._api_base,
                }
            )

        def authenticate_qq(
            appid: str,
            secret: str,
            port: str = "",
            api_base: str = "",
            token_url: str = "",
        ) -> str:
            """Configure the QQ bot credentials.

            Args:
                appid: QQ bot AppID.
                secret: QQ bot AppSecret.
                port: Optional inbound webhook server port (default 18086).
                api_base: Optional API base URL (default https://api.sgroup.qq.com).
                token_url: Optional access-token URL
                    (default https://bots.qq.com/app/getAppAccessToken).

            Returns:
                Configuration result or error message.
            """
            if not appid.strip() or not secret.strip():
                return "appid and secret cannot be empty."
            port_value = port.strip() or _DEFAULT_WEBHOOK_PORT
            if not _is_valid_port(port_value):
                return f"Invalid port {port_value!r}: must be an integer between 1 and 65535."
            cfg = {
                "appid": appid.strip(),
                "secret": secret.strip(),
                "port": port_value,
                "api_base": (api_base.strip() or _DEFAULT_API_BASE).rstrip("/"),
                "token_url": token_url.strip() or _DEFAULT_TOKEN_URL,
            }
            _config.save(cfg)
            agent._backend._apply_config(cfg)
            return json.dumps({"ok": True, "message": "QQ bot configured."})

        def clear_qq_auth() -> str:
            """Clear the stored QQ bot configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._appid = ""
            agent._backend._secret = ""
            agent._backend._access_token = ""
            agent._backend._token_expiry = 0.0
            agent._backend._private_key = None
            agent._backend._public_key = None
            return "QQ configuration cleared."

        return [check_qq_auth, authenticate_qq, clear_qq_auth]


def _make_backend() -> QQChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = QQChannelBackend()
    cfg = _config.load()
    if not cfg:
        print("Not configured. Run: kiss-qq -t 'authenticate'")
        sys.exit(1)
    backend._apply_config(cfg)
    return backend


def main() -> None:
    """Run the QQAgent from the command line with chat persistence."""
    channel_main(
        QQAgent,
        "kiss-qq",
        channel_name="QQ",
        make_backend=_make_backend,
    )


def get_tools() -> list:
    """Return the QQ channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return QQAgent()._get_tools()


if __name__ == "__main__":
    main()

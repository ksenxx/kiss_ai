"""Synology Chat Agent — StatefulSorcarAgent extension with Synology Chat webhook API.

Provides access to Synology Chat via incoming and outgoing webhooks.
Stores config in ``~/.kiss/channels/synology/config.json``.

Usage::

    agent = SynologyChatAgent()
    agent.run(prompt_template="Send a message to the team")
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import requests

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
    wait_for_matching_message,
)
from kiss.channels._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_PORT = 18083

_SYNOLOGY_DIR = Path.home() / ".kiss" / "channels" / "synology"
_config = ChannelConfig(_SYNOLOGY_DIR, ("webhook_url",))


class SynologyChatChannelBackend(ToolMethodBackend):
    """Channel backend for Synology Chat webhooks.

    Sends messages via the incoming webhook URL. Receives messages
    via a webhook server (outgoing webhook from Synology Chat).
    """

    def __init__(self) -> None:
        self._webhook_url: str = ""
        self._token: str = ""
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._webhook_server: ThreadedHTTPServer | None = None
        self._webhook_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load Synology config and start webhook server."""
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Synology Chat config found."
            return False
        self._webhook_url = cfg["webhook_url"]
        self._token = cfg.get("token", "")
        self._connection_info = "Synology Chat configured"
        if not self._start_webhook_server():  # pragma: no branch
            return False
        return True

    def _start_webhook_server(self, port: int = _DEFAULT_WEBHOOK_PORT) -> bool:
        """Start the outgoing webhook HTTP server."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    # Synology Chat sends URL-encoded payload
                    params = parse_qs(body.decode("utf-8"))
                    payload_str = params.get("payload", ["{}"])[0]
                    payload = json.loads(payload_str)
                    token = payload.get("token", "")
                    if not backend._token or token == backend._token:  # pragma: no branch
                        backend._message_queue.put(
                            {
                                "ts": str(payload.get("timestamp", "")),
                                "user": payload.get("user_id", ""),
                                "text": payload.get("text", ""),
                                "channel_id": payload.get("channel_id", ""),
                            }
                        )
                except Exception:
                    pass
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.disconnect()
        try:
            self._webhook_server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
            self._webhook_thread = threading.Thread(
                target=self._webhook_server.serve_forever, daemon=True
            )
            self._webhook_thread.start()
            logger.info("Synology Chat webhook server started on port %d", port)
            return True
        except OSError as e:
            self._connection_info = f"Synology webhook bind failed: {e}"
            logger.warning("Could not start Synology webhook server: %s", e)
            self._webhook_server = None
            self._webhook_thread = None
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Drain the webhook message queue."""
        messages: list[dict[str, Any]] = []
        while not self._message_queue.empty() and len(messages) < limit:  # pragma: no branch
            msg = self._message_queue.get_nowait()
            if not channel_id or msg.get("channel_id") == channel_id:  # pragma: no branch
                messages.append(msg)
        return messages, oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Synology Chat message via incoming webhook."""
        with self._send_lock:
            payload = {"text": text}
            if channel_id:  # pragma: no branch
                payload["channel_id"] = channel_id
            requests.post(self._webhook_url, params={"payload": json.dumps(payload)}, timeout=30)

    def wait_for_reply(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        timeout_seconds: float = 300.0,
    ) -> str | None:
        """Poll for a reply from a specific user."""
        return wait_for_matching_message(
            poll=lambda: self.poll_messages(channel_id, "")[0],
            matches=lambda msg: msg.get("user") == user_id,
            extract_text=lambda msg: str(msg.get("text", "")),
            timeout_seconds=timeout_seconds,
            poll_interval=2.0,
        )

    def disconnect(self) -> None:
        """Stop the embedded webhook server and release backend resources."""
        self._webhook_server, self._webhook_thread = stop_http_server(
            self._webhook_server, self._webhook_thread
        )

    def post_message(self, text: str, user_ids: str = "") -> str:
        """Send a message to Synology Chat via incoming webhook.

        Args:
            text: Message text.
            user_ids: Comma-separated user IDs to send to (optional).
                If empty, sends to the default channel.

        Returns:
            JSON string with ok status.
        """
        try:
            payload: dict[str, Any] = {"text": text}
            if user_ids:  # pragma: no branch
                payload["user_ids"] = [u.strip() for u in user_ids.split(",") if u.strip()]
            with self._send_lock:
                resp = requests.post(
                    self._webhook_url,
                    params={"payload": json.dumps(payload)},
                    timeout=30,
                )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_file_message(self, text: str, file_url: str) -> str:
        """Send a message with a file attachment.

        Args:
            text: Message text.
            file_url: URL of the file to attach.

        Returns:
            JSON string with ok status.
        """
        try:
            payload = {"text": text, "file_url": file_url}
            with self._send_lock:
                resp = requests.post(
                    self._webhook_url,
                    params={"payload": json.dumps(payload)},
                    timeout=30,
                )
            return json.dumps({"ok": resp.status_code == 200})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class SynologyChatAgent(BaseChannelAgent, StatefulSorcarAgent):
    """StatefulSorcarAgent extended with Synology Chat webhook tools."""

    def __init__(self) -> None:
        super().__init__("Synology Chat Agent")
        self._backend = SynologyChatChannelBackend()
        cfg = _config.load()
        if cfg:  # pragma: no branch
            self._backend._webhook_url = cfg["webhook_url"]
            self._backend._token = cfg.get("token", "")

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._webhook_url)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_synology_auth() -> str:
            """Check if Synology Chat is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._backend._webhook_url:  # pragma: no branch
                return (
                    "Not configured for Synology Chat. Use authenticate_synology() to configure.\n"
                    "You need the incoming webhook URL from Synology Chat integration settings."
                )
            return json.dumps(
                {
                    "ok": True,
                    "webhook_url": agent._backend._webhook_url[:50] + "...",
                }
            )

        def authenticate_synology(webhook_url: str, token: str = "") -> str:
            """Configure Synology Chat webhook.

            Args:
                webhook_url: Synology Chat incoming webhook URL.
                token: Optional outgoing webhook token for verification.

            Returns:
                Configuration result or error message.
            """
            if not webhook_url.strip():  # pragma: no branch
                return "webhook_url cannot be empty."
            agent._backend._webhook_url = webhook_url.strip()
            agent._backend._token = token.strip()
            _config.save({"webhook_url": webhook_url.strip(), "token": token.strip()})
            return json.dumps({"ok": True, "message": "Synology Chat configured."})

        def clear_synology_auth() -> str:
            """Clear the stored Synology Chat configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._webhook_url = ""
            agent._backend._token = ""
            return "Synology Chat configuration cleared."

        return [check_synology_auth, authenticate_synology, clear_synology_auth]


def _make_backend() -> SynologyChatChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = SynologyChatChannelBackend()
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not configured. Run: kiss-synology -t 'authenticate'")
        sys.exit(1)
    backend._webhook_url = cfg["webhook_url"]
    backend._token = cfg.get("token", "")
    return backend


def main() -> None:
    """Run the SynologyChatAgent from the command line with chat persistence."""
    channel_main(
        SynologyChatAgent,
        "kiss-synology",
        channel_name="Synology Chat",
        make_backend=_make_backend,
    )


if __name__ == "__main__":
    main()

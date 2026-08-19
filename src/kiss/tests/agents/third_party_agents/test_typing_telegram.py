# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Telegram typing indicator — no mocks or test doubles.

Runs a real local HTTP server standing in for the Telegram Bot API,
points the backend's ``_api_base`` at it (the same pattern the QQ and
Discord backend tests use), and asserts the exact ``sendChatAction``
request that ``send_typing`` emits.  Failure paths (HTTP 500, an
unreachable server, a missing token) are exercised against the same
real server / a genuinely closed port.

Config state is isolated by the session-wide temp ``KISS_HOME`` set in
the root conftest; each test that persists a token clears it afterward.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer
from kiss.agents.third_party_agents.telegram_agent import (
    TelegramChannelBackend,
    _config,
)

_TOKEN = "123456:TEST-telegram-token"


def _free_closed_port() -> int:
    """Return an OS-assigned TCP port that is closed once this returns."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _BotApiReceiver:
    """Real local HTTP server standing in for ``https://api.telegram.org``.

    Records every POST's path and JSON body and answers with a
    configurable HTTP status (200 with ``{"ok": true}`` by default).
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response_status: int = 200
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                receiver.requests.append(
                    {"path": self.path, "json": json.loads(body.decode("utf-8"))}
                )
                payload = json.dumps({"ok": receiver.response_status == 200}).encode("utf-8")
                self.send_response(receiver.response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadedHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        """Base URL of the running receiver, e.g. ``http://127.0.0.1:PORT``."""
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        """Shut down the server and join its thread."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture()
def receiver() -> Iterator[_BotApiReceiver]:
    """Yield a running Bot API receiver, stopping it afterward."""
    server = _BotApiReceiver()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture()
def configured_backend(receiver: _BotApiReceiver) -> Iterator[TelegramChannelBackend]:
    """Yield a backend with a persisted token, pointed at the local receiver."""
    _config.save({"bot_token": _TOKEN})
    backend = TelegramChannelBackend()
    backend._api_base = receiver.base_url
    try:
        yield backend
    finally:
        _config.clear()


def test_send_typing_posts_send_chat_action(
    receiver: _BotApiReceiver, configured_backend: TelegramChannelBackend
) -> None:
    """send_typing POSTs sendChatAction with action=typing and the int chat id."""
    configured_backend.send_typing("123456789")
    assert len(receiver.requests) == 1
    request = receiver.requests[0]
    assert request["path"] == f"/bot{_TOKEN}/sendChatAction"
    assert request["json"] == {"chat_id": 123456789, "action": "typing"}


def test_send_typing_negative_and_username_chat_ids(
    receiver: _BotApiReceiver, configured_backend: TelegramChannelBackend
) -> None:
    """Numeric ids (including negative group ids) become ints; @usernames stay strings."""
    configured_backend.send_typing("-100987654321")
    configured_backend.send_typing("@somechannel")
    assert [r["json"]["chat_id"] for r in receiver.requests] == [
        -100987654321,
        "@somechannel",
    ]
    assert all(r["json"]["action"] == "typing" for r in receiver.requests)


def test_send_typing_ignores_thread_ts(
    receiver: _BotApiReceiver, configured_backend: TelegramChannelBackend
) -> None:
    """thread_ts is accepted for interface parity but never sent to the API."""
    configured_backend.send_typing("42", thread_ts="777")
    assert len(receiver.requests) == 1
    assert receiver.requests[0]["json"] == {"chat_id": 42, "action": "typing"}


def test_send_typing_uses_live_bot_token_before_config(
    receiver: _BotApiReceiver,
) -> None:
    """A token exposed by the connected Bot object wins over the stored config."""

    class _TokenHolder:
        token = "999:LIVE-bot-token"

    backend = TelegramChannelBackend()
    backend._api_base = receiver.base_url
    backend._bot = _TokenHolder()
    backend.send_typing("5")
    assert len(receiver.requests) == 1
    assert receiver.requests[0]["path"] == "/bot999:LIVE-bot-token/sendChatAction"


def test_send_typing_swallows_http_500(
    receiver: _BotApiReceiver, configured_backend: TelegramChannelBackend
) -> None:
    """A 500 response from the Bot API never raises."""
    receiver.response_status = 500
    configured_backend.send_typing("123456789")
    assert len(receiver.requests) == 1
    assert receiver.requests[0]["json"]["action"] == "typing"


def test_send_typing_swallows_unreachable_server() -> None:
    """An unreachable API host (closed port) never raises."""
    _config.save({"bot_token": _TOKEN})
    try:
        backend = TelegramChannelBackend()
        backend._api_base = f"http://127.0.0.1:{_free_closed_port()}"
        backend.send_typing("123456789")
    finally:
        _config.clear()


def test_send_typing_without_token_sends_nothing(receiver: _BotApiReceiver) -> None:
    """With no Bot and no stored config, send_typing is a silent no-op."""
    _config.clear()
    backend = TelegramChannelBackend()
    backend._api_base = receiver.base_url
    backend.send_typing("123456789")
    assert receiver.requests == []

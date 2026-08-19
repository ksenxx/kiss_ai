# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Telegram poll_messages cursor contract — no mocks.

Runs a real local HTTP server standing in for the Telegram Bot API
(the same pattern as ``test_typing_telegram.py``), points the backend's
``_api_base`` at it, and asserts the exact ``getUpdates`` request that
``poll_messages`` emits plus the cursor it returns:

- ``oldest="0"`` sends no offset (fresh backend) and returns
  ``str(highest update_id + 1)``;
- a digit-string ``oldest`` is sent as the ``getUpdates`` offset;
- no updates -> cursor returned unchanged;
- server error / unreachable host -> ``([], oldest)`` without raising.
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


def _update(update_id: int, chat_id: int, message_id: int, text: str) -> dict[str, Any]:
    """Build a minimal Telegram ``getUpdates`` result entry."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": 1700000000,
            "text": text,
            "chat": {"id": chat_id},
            "from": {"id": 777},
        },
    }


class _BotApiReceiver:
    """Real local HTTP server standing in for ``https://api.telegram.org``.

    Records every POST's path and JSON body and answers ``getUpdates``
    with a configurable update list and HTTP status.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response_status: int = 200
        self.updates: list[dict[str, Any]] = []
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                receiver.requests.append(
                    {"path": self.path, "json": json.loads(body.decode("utf-8"))}
                )
                ok = receiver.response_status == 200
                payload = json.dumps({"ok": ok, "result": receiver.updates}).encode("utf-8")
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
def backend(receiver: _BotApiReceiver) -> Iterator[TelegramChannelBackend]:
    """Yield a backend with a persisted token, pointed at the local receiver."""
    _config.save({"bot_token": _TOKEN})
    instance = TelegramChannelBackend()
    instance._api_base = receiver.base_url
    try:
        yield instance
    finally:
        _config.clear()


def test_oldest_zero_sends_no_offset_and_returns_next_cursor(
    receiver: _BotApiReceiver, backend: TelegramChannelBackend
) -> None:
    """oldest='0' on a fresh backend omits offset; cursor = highest update_id + 1."""
    receiver.updates = [
        _update(100, 42, 1, "hello"),
        _update(101, 42, 2, "world"),
    ]
    messages, new_cursor = backend.poll_messages("42", "0", limit=10)
    assert len(receiver.requests) == 1
    request = receiver.requests[0]
    assert request["path"] == f"/bot{_TOKEN}/getUpdates"
    assert "offset" not in request["json"]
    assert new_cursor == "102"
    assert [m["text"] for m in messages] == ["hello", "world"]
    assert messages[0] == {
        "ts": "1",
        "date": "1700000000.0",
        "user": "777",
        "text": "hello",
        "message_id": "1",
        "chat_id": "42",
    }


def test_numeric_oldest_is_sent_as_offset(
    receiver: _BotApiReceiver, backend: TelegramChannelBackend
) -> None:
    """oldest='42' is forwarded as offset=42 in the getUpdates request."""
    receiver.updates = [_update(42, 7, 5, "resumed")]
    messages, new_cursor = backend.poll_messages("", "42", limit=10)
    assert len(receiver.requests) == 1
    assert receiver.requests[0]["json"]["offset"] == 42
    assert new_cursor == "43"
    assert [m["text"] for m in messages] == ["resumed"]


def test_no_updates_returns_cursor_unchanged(
    receiver: _BotApiReceiver, backend: TelegramChannelBackend
) -> None:
    """When Telegram returns no updates, the passed-in cursor comes back verbatim."""
    receiver.updates = []
    messages, new_cursor = backend.poll_messages("42", "42", limit=10)
    assert messages == []
    assert new_cursor == "42"
    assert receiver.requests[0]["json"]["offset"] == 42


def test_server_error_returns_empty_and_cursor_without_raising(
    receiver: _BotApiReceiver, backend: TelegramChannelBackend
) -> None:
    """An HTTP 500 from the Bot API yields ([], oldest) and never raises."""
    receiver.response_status = 500
    messages, new_cursor = backend.poll_messages("42", "42", limit=10)
    assert messages == []
    assert new_cursor == "42"


def test_unreachable_server_returns_empty_and_cursor_without_raising() -> None:
    """An unreachable API host (closed port) yields ([], oldest) and never raises."""
    _config.save({"bot_token": _TOKEN})
    try:
        backend = TelegramChannelBackend()
        backend._api_base = f"http://127.0.0.1:{_free_closed_port()}"
        messages, new_cursor = backend.poll_messages("42", "7", limit=10)
        assert messages == []
        assert new_cursor == "7"
    finally:
        _config.clear()


def test_process_local_cursor_stays_monotonic_with_stale_oldest(
    receiver: _BotApiReceiver, backend: TelegramChannelBackend
) -> None:
    """A stale numeric oldest never rewinds past the in-process _last_update_id."""
    receiver.updates = [_update(200, 42, 9, "first")]
    _, first_cursor = backend.poll_messages("42", "0", limit=10)
    assert first_cursor == "201"
    receiver.updates = []
    receiver.requests.clear()
    _, second_cursor = backend.poll_messages("42", "5", limit=10)
    assert receiver.requests[0]["json"]["offset"] == 201
    assert second_cursor == "5"


def test_non_numeric_oldest_uses_legacy_behavior(
    receiver: _BotApiReceiver, backend: TelegramChannelBackend
) -> None:
    """A non-numeric oldest is ignored for the offset and returned on failure paths."""
    receiver.updates = []
    messages, new_cursor = backend.poll_messages("42", "not-a-number", limit=10)
    assert messages == []
    assert new_cursor == "not-a-number"
    assert "offset" not in receiver.requests[0]["json"]


def test_channel_filter_still_confirms_all_updates(
    receiver: _BotApiReceiver, backend: TelegramChannelBackend
) -> None:
    """Updates for other chats are filtered out but still advance the cursor."""
    receiver.updates = [
        _update(300, 42, 1, "mine"),
        _update(301, 99, 2, "other-chat"),
    ]
    messages, new_cursor = backend.poll_messages("42", "0", limit=10)
    assert [m["text"] for m in messages] == ["mine"]
    assert new_cursor == "302"

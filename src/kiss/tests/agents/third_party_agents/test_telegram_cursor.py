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

from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.third_party_agents.telegram.telegram_sea import (
    TelegramChannelBackend,
    _config,
)
from kiss.tests.agents.third_party_agents.telegram_bot_api import (
    TOKEN,
    BotApiServer,
    configured_backend,
    serve_bot_api,
)


@pytest.fixture()
def receiver() -> Iterator[BotApiServer]:
    """Yield a running Bot API receiver, stopping it afterward."""
    yield from serve_bot_api()


@pytest.fixture()
def backend(receiver: BotApiServer) -> Iterator[TelegramChannelBackend]:
    """Yield a backend with a persisted token, pointed at the local receiver."""
    yield from configured_backend(receiver)


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


def test_oldest_zero_sends_no_offset_and_returns_next_cursor(
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """oldest='0' on a fresh backend omits offset; cursor = highest update_id + 1."""
    receiver.updates = [
        _update(100, 42, 1, "hello"),
        _update(101, 42, 2, "world"),
    ]
    messages, new_cursor = backend.poll_messages("42", "0", limit=10)
    assert len(receiver.requests) == 1
    request = receiver.requests[0]
    assert request["path"] == f"/bot{TOKEN}/getUpdates"
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
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """oldest='42' is forwarded as offset=42 in the getUpdates request."""
    receiver.updates = [_update(42, 7, 5, "resumed")]
    messages, new_cursor = backend.poll_messages("", "42", limit=10)
    assert len(receiver.requests) == 1
    assert receiver.requests[0]["json"]["offset"] == 42
    assert new_cursor == "43"
    assert [m["text"] for m in messages] == ["resumed"]


def test_no_updates_returns_cursor_unchanged(
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """When Telegram returns no updates, the passed-in cursor comes back verbatim."""
    receiver.updates = []
    messages, new_cursor = backend.poll_messages("42", "42", limit=10)
    assert messages == []
    assert new_cursor == "42"
    assert receiver.requests[0]["json"]["offset"] == 42


def test_server_error_returns_empty_and_cursor_without_raising(
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """An HTTP 500 from the Bot API yields ([], oldest) and never raises."""
    receiver.response_status = 500
    messages, new_cursor = backend.poll_messages("42", "42", limit=10)
    assert messages == []
    assert new_cursor == "42"


def test_unreachable_server_returns_empty_and_cursor_without_raising(
    refusing_port: int,
) -> None:
    """An unreachable API host (refused port) yields ([], oldest) and never raises."""
    _config.save({"bot_token": TOKEN})
    try:
        backend = TelegramChannelBackend()
        backend._api_base = f"http://127.0.0.1:{refusing_port}"
        messages, new_cursor = backend.poll_messages("42", "7", limit=10)
        assert messages == []
        assert new_cursor == "7"
    finally:
        _config.clear()


def test_process_local_cursor_stays_monotonic_with_stale_oldest(
    receiver: BotApiServer, backend: TelegramChannelBackend
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
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """A non-numeric oldest is ignored for the offset and returned on failure paths."""
    receiver.updates = []
    messages, new_cursor = backend.poll_messages("42", "not-a-number", limit=10)
    assert messages == []
    assert new_cursor == "not-a-number"
    assert "offset" not in receiver.requests[0]["json"]


def test_channel_filter_still_confirms_all_updates(
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """Updates for other chats are filtered out but still advance the cursor."""
    receiver.updates = [
        _update(300, 42, 1, "mine"),
        _update(301, 99, 2, "other-chat"),
    ]
    messages, new_cursor = backend.poll_messages("42", "0", limit=10)
    assert [m["text"] for m in messages] == ["mine"]
    assert new_cursor == "302"

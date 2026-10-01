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

from collections.abc import Iterator

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


def test_send_typing_posts_send_chat_action(
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """send_typing POSTs sendChatAction with action=typing and the int chat id."""
    backend.send_typing("123456789")
    assert len(receiver.requests) == 1
    request = receiver.requests[0]
    assert request["path"] == f"/bot{TOKEN}/sendChatAction"
    assert request["json"] == {"chat_id": 123456789, "action": "typing"}


def test_send_typing_negative_and_username_chat_ids(
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """Numeric ids (including negative group ids) become ints; @usernames stay strings."""
    backend.send_typing("-100987654321")
    backend.send_typing("@somechannel")
    assert [r["json"]["chat_id"] for r in receiver.requests] == [
        -100987654321,
        "@somechannel",
    ]
    assert all(r["json"]["action"] == "typing" for r in receiver.requests)


def test_send_typing_ignores_thread_ts(
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """thread_ts is accepted for interface parity but never sent to the API."""
    backend.send_typing("42", thread_ts="777")
    assert len(receiver.requests) == 1
    assert receiver.requests[0]["json"] == {"chat_id": 42, "action": "typing"}


def test_send_typing_uses_live_bot_token_before_config(
    receiver: BotApiServer,
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
    receiver: BotApiServer, backend: TelegramChannelBackend
) -> None:
    """A 500 response from the Bot API never raises."""
    receiver.response_status = 500
    backend.send_typing("123456789")
    assert len(receiver.requests) == 1
    assert receiver.requests[0]["json"]["action"] == "typing"


def test_send_typing_swallows_unreachable_server(refusing_port: int) -> None:
    """An unreachable API host (refused port) never raises."""
    _config.save({"bot_token": TOKEN})
    try:
        backend = TelegramChannelBackend()
        backend._api_base = f"http://127.0.0.1:{refusing_port}"
        backend.send_typing("123456789")
    finally:
        _config.clear()


def test_send_typing_without_token_sends_nothing(receiver: BotApiServer) -> None:
    """With no Bot and no stored config, send_typing is a silent no-op."""
    _config.clear()
    backend = TelegramChannelBackend()
    backend._api_base = receiver.base_url
    backend.send_typing("123456789")
    assert receiver.requests == []

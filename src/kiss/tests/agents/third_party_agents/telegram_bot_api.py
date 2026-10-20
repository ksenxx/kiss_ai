# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local stand-in for ``https://api.telegram.org`` shared by the Telegram tests.

``BotApiServer`` records every POST (see ``JsonWebhookServer.requests``)
and answers ``{"ok": <status == 200>, "result": updates}`` with the
configurable ``response_status``.  Test modules wrap :func:`serve_bot_api`
and :func:`configured_backend` in their ``receiver`` / ``backend`` fixtures
and run under ``isolated_kiss_home`` so the persisted token is per test.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from kiss.agents.third_party_agents.telegram.telegram_sea import TelegramChannelBackend, _config
from kiss.tests.agents.third_party_agents.recording_http import (
    JsonWebhookHandler,
    JsonWebhookServer,
    serve_recording,
)

TOKEN = "123456:TEST-telegram-token"


class BotApiServer(JsonWebhookServer):
    """Recording Bot API server with ``response_status`` and ``updates`` knobs."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.updates: list[dict[str, Any]] = []


class _BotApiHandler(JsonWebhookHandler):
    def do_POST(self) -> None:  # noqa: N802
        """Record the call and answer it Bot-API style from the server's knobs."""
        server = cast(BotApiServer, self.server)
        self.record()
        ok = server.response_status == 200
        self._json(server.response_status, {"ok": ok, "result": server.updates})


def serve_bot_api() -> Iterator[BotApiServer]:
    """Run a Bot API receiver for one fixture (``yield from serve_bot_api()``)."""
    yield from serve_recording(_BotApiHandler, BotApiServer)


def configured_backend(receiver: BotApiServer) -> TelegramChannelBackend:
    """Return a backend with a persisted ``TOKEN`` aimed at *receiver*."""
    _config.save({"bot_token": TOKEN})
    instance = TelegramChannelBackend()
    instance._api_base = receiver.base_url
    return instance

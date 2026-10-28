# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local stand-in for ``https://api.telegram.org`` shared by the Telegram tests.

``BotApiServer`` records every POST (see ``JsonWebhookServer.requests``,
plus ``token``, ``api_method``, the raw ``body`` and lower-cased ``headers``)
and answers ``getMe`` / ``sendMessage`` / ``sendPhoto`` like
api.telegram.org; every other method gets ``{"ok": <status == 200>,
"result": updates}`` with the configurable ``response_status``.  With
``expected_token`` set, any other token is answered 401.  ``badJson``,
``errorObject`` and ``listBody`` are malformed-reply probes for the
transport tests.  Test modules wrap :func:`serve_bot_api`
and :func:`configured_backend` in their ``receiver`` / ``backend`` fixtures
and run under ``isolated_kiss_home`` so the persisted token is per test.
"""

from __future__ import annotations

import json
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
    """Recording Bot API server; knobs: ``response_status``, ``updates``, ``expected_token``."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.updates: list[dict[str, Any]] = []
        self.expected_token = ""


class _BotApiHandler(JsonWebhookHandler):
    def do_POST(self) -> None:  # noqa: N802
        """Record the call and answer ``/bot<token>/<method>`` like api.telegram.org."""
        server = cast(BotApiServer, self.server)
        _, _, rest = self.path.partition("/bot")
        token, _, method = rest.partition("/")
        raw = self._read_raw()
        entry = self.record_request(
            raw,
            token=token,
            api_method=method,
            headers={k.lower(): v for k, v in self.headers.items()},
        )
        try:
            entry["json"] = json.loads(raw) if raw else {}
        except ValueError:
            entry["json"] = None
        if server.expected_token and token != server.expected_token:
            self._json(401, {"ok": False, "description": "Unauthorized"})
        elif method == "getMe":
            me = {"id": 123456, "username": "emu_bot", "first_name": "Emu"}
            self._json(200, {"ok": True, "result": me})
        elif method in ("sendMessage", "sendPhoto"):
            self._json(200, {"ok": True, "result": {"message_id": 77}})
        elif method == "badJson":
            self._reply(200, "<html>not json</html>")
        elif method == "errorObject":
            self._json(500, {"error": {"message": "boom"}})
        elif method == "listBody":
            self._reply(200, "[]")
        else:
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

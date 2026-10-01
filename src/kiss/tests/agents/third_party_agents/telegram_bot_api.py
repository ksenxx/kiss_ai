# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local stand-in for ``https://api.telegram.org`` shared by the Telegram tests.

``BotApiServer`` records every POST's path and JSON body in ``requests``
and answers ``{"ok": <status == 200>, "result": updates}`` with the
configurable ``response_status``.  Test modules wrap :func:`serve_bot_api`
and :func:`configured_backend` in their ``receiver`` / ``backend`` fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any, cast

from kiss.agents.third_party_agents.telegram.telegram_sea import TelegramChannelBackend, _config
from kiss.tests.agents.third_party_agents.recording_http import RecordingServer, serve_recording

TOKEN = "123456:TEST-telegram-token"


class BotApiServer(RecordingServer):
    """Recording Bot API server with ``response_status`` and ``updates`` knobs."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.response_status = 200
        self.updates: list[dict[str, Any]] = []


class _BotApiHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        server = cast(BotApiServer, self.server)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        server.requests.append({"path": self.path, "json": json.loads(body.decode("utf-8"))})
        ok = server.response_status == 200
        payload = json.dumps({"ok": ok, "result": server.updates}).encode("utf-8")
        self.send_response(server.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def serve_bot_api() -> Iterator[BotApiServer]:
    """Run a Bot API receiver for one fixture (``yield from serve_bot_api()``)."""
    yield from serve_recording(_BotApiHandler, BotApiServer)


def configured_backend(receiver: BotApiServer) -> Iterator[TelegramChannelBackend]:
    """Yield a backend with a persisted ``TOKEN`` aimed at *receiver*; clear the token after."""
    _config.save({"bot_token": TOKEN})
    instance = TelegramChannelBackend()
    instance._api_base = receiver.base_url
    try:
        yield instance
    finally:
        _config.clear()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Slack Web API stand-in that answers every call with ``invalid_auth``.

Returns exactly what ``https://slack.com/api/auth.test`` returns for a bad
token, so ``SlackChannelBackend.connect()`` follows its real
``SlackApiError`` path offline.  Serve it with
``recording_server(InvalidAuthHandler)``; every request is recorded as
``{"method", "path"}`` in ``server.requests``.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, cast

from kiss.tests.agents.third_party_agents.recording_http import RecordingServer


class InvalidAuthHandler(BaseHTTPRequestHandler):
    """Record the request and reply 200 with Slack's invalid-token error body."""

    def _respond(self) -> None:
        cast(RecordingServer, self.server).requests.append(
            {"method": self.command, "path": self.path.split("?", 1)[0]}
        )
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        data = json.dumps({"ok": False, "error": "invalid_auth"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests (e.g. conversations.list)."""
        self._respond()

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests (e.g. auth.test, chat.postMessage)."""
        self._respond()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""

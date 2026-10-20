# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Slack Web API stand-in that answers every call with ``invalid_auth``.

Returns exactly what ``https://slack.com/api/auth.test`` returns for a bad
token, so ``SlackChannelBackend.connect()`` follows its real
``SlackApiError`` path offline.  Serve it with
``recording_server(InvalidAuthHandler)``; every request is recorded by
``RecordingHandler.record_request`` (``{"method", "path", "query",
"authorization", "body"}``) in ``server.requests``.
"""

from __future__ import annotations

from kiss.tests.agents.third_party_agents.recording_http import RecordingHandler


def slack_method(path: str) -> str:
    """Return the Slack Web API method name (``conversations.info``) of a request *path*."""
    return path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]


class InvalidAuthHandler(RecordingHandler):
    """Record the request and reply 200 with Slack's invalid-token error body."""

    def _respond(self) -> None:
        self.record_request(None)
        self._read_raw()
        self._json(200, {"ok": False, "error": "invalid_auth"})

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests (e.g. conversations.list)."""
        self._respond()

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests (e.g. auth.test, chat.postMessage)."""
        self._respond()

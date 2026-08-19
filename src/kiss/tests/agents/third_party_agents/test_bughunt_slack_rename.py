# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing the "channels" -> "third_party_agents" rename bugs.

A mechanical rename corrupted Slack API-facing strings in slack_agent.py:
response key ``resp.get("third_party_agents")`` (must be ``"channels"``) and
the ``files_upload_v2(third_party_agents=...)`` kwarg (must be ``channels=``).

These tests run WITHOUT network and WITHOUT mocks: a real in-process HTTP
server returns Slack-shaped JSON and records every request path, query string,
and body. A real ``slack_sdk.WebClient`` pointed at that server is injected
into a real ``SlackChannelBackend`` (same injection pattern as
``test_slack_channel_backend.py``).
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

from slack_sdk import WebClient

from kiss.agents.third_party_agents.slack_agent import SlackChannelBackend

_CHANNELS_JSON = {
    "ok": True,
    "channels": [
        {
            "id": "C1",
            "name": "general",
            "is_private": False,
            "purpose": {"value": "company-wide"},
            "num_members": 3,
        }
    ],
    "response_metadata": {"next_cursor": ""},
}


class _SlackHandler(BaseHTTPRequestHandler):
    """Minimal Slack Web API emulator that records every request."""

    server: Any

    def _record_and_respond(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        parsed = urllib.parse.urlparse(self.path)
        self.server.requests.append(
            {"path": parsed.path, "query": parsed.query, "body": body}
        )
        port = self.server.server_address[1]
        if parsed.path.endswith("/auth.test"):
            payload: dict[str, Any] = {
                "ok": True,
                "user_id": "UBOT",
                "user": "bot",
                "team": "testteam",
            }
        elif parsed.path.endswith("/conversations.list"):
            payload = _CHANNELS_JSON
        elif parsed.path.endswith("/files.getUploadURLExternal"):
            payload = {
                "ok": True,
                "upload_url": f"http://127.0.0.1:{port}/upload/v1/F1",
                "file_id": "F1",
            }
        elif "/upload/v1/" in parsed.path:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK - upload complete")
            return
        elif parsed.path.endswith("/files.completeUploadExternal"):
            payload = {"ok": True, "files": [{"id": "F1", "title": "a.txt"}]}
        else:
            payload = {"ok": True}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests (e.g. conversations.list)."""
        self._record_and_respond()

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests (e.g. files.completeUploadExternal)."""
        self._record_and_respond()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


def _all_sent_params(requests: list[dict[str, str]]) -> set[str]:
    """Collect every parameter name sent in any query string or form body."""
    names: set[str] = set()
    for req in requests:
        for raw in (req["query"], req["body"]):
            if not raw or raw.startswith(("{", "OK")):
                continue
            try:
                names.update(urllib.parse.parse_qs(raw).keys())
            except ValueError:
                continue
    return names


class TestSlackRenameBugs:
    """Reproduce the rename-corruption bugs against a local Slack emulator."""

    server: ClassVar[ThreadingHTTPServer]
    thread: ClassVar[threading.Thread]

    @classmethod
    def setup_class(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _SlackHandler)
        cls.server.requests = []  # type: ignore[attr-defined]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def teardown_class(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setup_method(self) -> None:
        self.server.requests.clear()  # type: ignore[attr-defined]
        port = self.server.server_address[1]
        self.backend = SlackChannelBackend()
        self.backend._client = WebClient(
            token="xoxb-test", base_url=f"http://127.0.0.1:{port}/", retry_handlers=[]
        )
        self.backend._bot_user_id = "UBOT"

    def _requests(self) -> list[dict[str, str]]:
        return list(self.server.requests)  # type: ignore[attr-defined]

    def test_find_channel_reads_channels_key(self) -> None:
        """find_channel must read the 'channels' key of conversations.list."""
        assert self.backend.find_channel("general") == "C1"
        paths = [r["path"] for r in self._requests()]
        assert any(p.endswith("/conversations.list") for p in paths)

    def test_list_channels_tool_returns_channels(self) -> None:
        """The list-channels tool must surface channels from the API response."""
        result = json.loads(self.backend.list_third_party_agents())
        assert result["ok"] is True
        entries = result.get("channels") or result.get("third_party_agents") or []
        assert len(entries) == 1
        assert entries[0]["id"] == "C1"
        assert entries[0]["name"] == "general"
        assert entries[0]["purpose"] == "company-wide"

    def test_upload_file_shares_to_channels(self) -> None:
        """upload_file must send channel ids under 'channels', never
        'third_party_agents'."""
        out = json.loads(self.backend.upload_file("C1", "hello world", "a.txt"))
        assert out["ok"] is True, out
        assert out["file_id"] == "F1"
        requests = self._requests()
        complete = [
            r for r in requests if r["path"].endswith("/files.completeUploadExternal")
        ]
        assert len(complete) == 1
        params = urllib.parse.parse_qs(complete[0]["body"])
        assert "third_party_agents" not in params
        assert "C1" in "".join(params.get("channels", []) + params.get("channel_id", []))
        assert "third_party_agents" not in _all_sent_params(requests)

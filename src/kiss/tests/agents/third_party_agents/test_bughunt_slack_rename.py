# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing the "channels" -> "third_party_agents" rename bugs.

A mechanical rename corrupted Slack API-facing strings in slack_sea.py:
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
import urllib.parse
from collections.abc import Iterator
from typing import Any

import pytest
from slack_sdk import WebClient

from kiss.agents.third_party_agents.slack.slack_sea import SlackChannelBackend
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    serve_recording,
)

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


class _SlackHandler(RecordingHandler):
    """Minimal Slack Web API emulator that records every request."""

    server: RecordingServer  # type: ignore[assignment]

    def _record_and_respond(self) -> None:
        body = self._read_raw().decode("utf-8", errors="replace")
        path = self.record_request(body)["path"]
        port = self.server.server_address[1]
        if path.endswith("/auth.test"):
            payload: dict[str, Any] = {
                "ok": True,
                "user_id": "UBOT",
                "user": "bot",
                "team": "testteam",
            }
        elif path.endswith("/conversations.list"):
            payload = _CHANNELS_JSON
        elif path.endswith("/files.getUploadURLExternal"):
            payload = {
                "ok": True,
                "upload_url": f"http://127.0.0.1:{port}/upload/v1/F1",
                "file_id": "F1",
            }
        elif "/upload/v1/" in path:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK - upload complete")
            return
        elif path.endswith("/files.completeUploadExternal"):
            payload = {"ok": True, "files": [{"id": "F1", "title": "a.txt"}]}
        else:
            payload = {"ok": True}
        self._json(200, payload)

    do_GET = do_POST = _record_and_respond  # noqa: N815


def _all_sent_params(requests: list[dict[str, Any]]) -> set[str]:
    """Collect every parameter name sent in any query string or form body."""
    names: set[str] = set()
    for req in requests:
        names.update(req["query"])
        body = req["body"]
        if body and not body.startswith(("{", "OK")):
            names.update(urllib.parse.parse_qs(body).keys())
    return names


@pytest.fixture(scope="module")
def slack_server() -> Iterator[RecordingServer]:
    """The Slack Web API emulator, shared by the module."""
    yield from serve_recording(_SlackHandler)


class TestSlackRenameBugs:
    """Reproduce the rename-corruption bugs against a local Slack emulator."""

    @pytest.fixture(autouse=True)
    def _backend(self, slack_server: RecordingServer) -> None:
        """A backend whose client talks to *slack_server*; the request log starts empty."""
        slack_server.requests.clear()
        self.server = slack_server
        self.backend = SlackChannelBackend()
        self.backend._client = WebClient(
            token="xoxb-test",
            base_url=f"{slack_server.base_url}/",
            retry_handlers=[],
        )
        self.backend._bot_user_id = "UBOT"

    def _requests(self) -> list[dict[str, str]]:
        return list(self.server.requests)

    @pytest.mark.parametrize(
        "name,channel_id,used,unused",
        [
            ("general", "C1", "/conversations.list", "/conversations.info"),
            ("C0AKYSNLB7W", "C0AKYSNLB7W", "/conversations.info", "/conversations.list"),
        ],
        ids=["name_is_listed", "id_is_verified_via_info"],
    )
    def test_find_channel_endpoint(
        self, name: str, channel_id: str, used: str, unused: str
    ) -> None:
        """A name is looked up in conversations.list; an ID is confirmed with conversations.info."""
        assert self.backend.find_channel(name) == channel_id
        paths = [r["path"] for r in self._requests()]
        assert any(p.endswith(used) for p in paths)
        assert not any(p.endswith(unused) for p in paths)

    def test_list_channels_tool_returns_channels(self) -> None:
        """The list-channels tool surfaces the API's channels under its own ``channels`` key."""
        result = json.loads(self.backend.list_third_party_agents())
        assert result["ok"] is True
        assert "third_party_agents" not in result
        entries = result["channels"]
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
        complete = [r for r in requests if r["path"].endswith("/files.completeUploadExternal")]
        assert len(complete) == 1
        params = urllib.parse.parse_qs(complete[0]["body"])
        assert "third_party_agents" not in params
        assert "C1" in "".join(params.get("channels", []) + params.get("channel_id", []))
        assert "third_party_agents" not in _all_sent_params(requests)

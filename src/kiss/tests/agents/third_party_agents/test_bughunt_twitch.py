# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing verified bugs in twitch_sea.

Runs a real in-process HTTP server that records request paths/bodies and
returns Helix-shaped JSON. No mocks, patches, or test doubles.  (The Tlon
bugs found in the same hunt live in test_bughunt_round2_tlon_phone_wa.py.)

Bugs covered:
- Twitch Helix endpoint corruption: ``/third_party_agents`` -> ``/channels``
  and ``/search/third_party_agents`` -> ``/search/channels``.
- Twitch ``send_message`` omitting the required ``sender_id`` field.
- Twitch ``_get``/``_post`` silently swallowing HTTP 4xx errors.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from kiss.agents.third_party_agents.twitch.twitch_sea import TwitchChannelBackend
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    recording_server,
)


class _HelixHandler(RecordingHandler):
    """Serves the Twitch Helix endpoints the backend uses."""

    def do_GET(self) -> None:  # noqa: N802
        self.record_request(None)
        if self.path.startswith("/users"):
            self._json(200, {"data": [{"id": "botid123", "login": "kissbot"}]})
        elif self.path.startswith("/channels"):
            self._json(200, {"data": [{"broadcaster_id": "123", "broadcaster_login": "streamer"}]})
        elif self.path.startswith("/search/channels"):
            self._json(200, {"data": [{"broadcaster_login": "streamer"}]})
        else:
            self._json(404, {"error": "Not Found", "status": 404})

    def do_POST(self) -> None:  # noqa: N802
        self.record_request(self._read_raw().decode())
        if self.path.startswith("/chat/messages"):
            self._json(200, {"data": [{"message_id": "m1", "is_sent": True}]})
        elif self.path.startswith("/moderation/bans"):
            self._json(400, {"error": "Bad Request", "status": 400, "message": "missing field"})
        else:
            self._json(404, {"error": "Not Found", "status": 404})


@pytest.fixture()
def server() -> Iterator[RecordingServer]:
    """A recording server emulating Twitch Helix."""
    with recording_server(_HelixHandler) as srv:
        yield srv


def _paths(server: RecordingServer, method: str | None = None) -> list[str]:
    return [r["path"] for r in server.requests if method is None or r["method"] == method]


class TestTwitchBackendBugs:
    """Twitch Helix endpoint corruption, missing sender_id, silent HTTP errors."""

    @pytest.fixture(autouse=True)
    def _backend(self, server: RecordingServer) -> None:
        self.server = server
        self.backend = TwitchChannelBackend(helix_base=server.base_url)
        self.backend._client_id = "cid"
        self.backend._access_token = "tok"

    def test_get_channel_info_hits_channels_endpoint(self) -> None:
        """get_channel_info must call GET /channels?broadcaster_id=..."""
        result = json.loads(self.backend.get_channel_info("123"))
        gets = [r for r in self.server.requests if r["method"] == "GET"]
        assert any(
            r["path"] == "/channels" and r["query"].get("broadcaster_id") == ["123"] for r in gets
        ), f"expected GET /channels?broadcaster_id=123, saw {gets}"
        assert not any("third_party_agents" in r["path"] for r in gets), gets
        assert result["ok"] is True
        assert result["channel"]["broadcaster_login"] == "streamer"

    def test_search_hits_search_channels_endpoint(self) -> None:
        """Channel search must call GET /search/channels."""
        result = json.loads(self.backend.search_channels("streamer"))
        paths = _paths(self.server, "GET")
        assert "/search/channels" in paths, f"expected GET /search/channels, saw {paths}"
        assert not any("third_party_agents" in p for p in paths), paths
        assert result["ok"] is True

    def test_send_message_includes_sender_id(self) -> None:
        """Contract send_message must POST /chat/messages with sender_id."""
        self.backend.send_message("bcast1", "hello there")
        posts = [
            r
            for r in self.server.requests
            if r["method"] == "POST" and r["path"].startswith("/chat/messages")
        ]
        assert posts, f"no POST /chat/messages seen: {self.server.requests}"
        body = json.loads(posts[-1]["body"])
        assert body["broadcaster_id"] == "bcast1"
        assert body["message"] == "hello there"
        assert body.get("sender_id") == "botid123", f"sender_id missing/wrong in body: {body}"

    def test_send_chat_message_still_includes_sender_id(self) -> None:
        """Explicit send_chat_message keeps passing sender_id through."""
        result = json.loads(self.backend.send_chat_message("bcast1", "sender9", "hi"))
        body = json.loads(self.server.requests[-1]["body"])
        assert body["sender_id"] == "sender9"
        assert result["ok"] is True

    def test_post_http_error_is_not_reported_ok(self) -> None:
        """A 4xx Helix response must not be reported as ok: true."""
        result = json.loads(self.backend.ban_user("b1", "m1", "u1", duration=10, reason="spam"))
        assert result["ok"] is False, f"4xx error reported as success: {result}"

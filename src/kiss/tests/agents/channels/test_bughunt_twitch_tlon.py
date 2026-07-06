# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing verified bugs in twitch_agent and tlon_agent.

Runs a real in-process HTTP server that records request paths/bodies and
returns Helix/Eyre-shaped JSON. No mocks, patches, or test doubles.

Bugs covered:
- Twitch Helix endpoint corruption: ``/third_party_agents`` -> ``/channels``
  and ``/search/third_party_agents`` -> ``/search/channels``.
- Twitch ``send_message`` omitting the required ``sender_id`` field.
- Twitch ``_get``/``_post`` silently swallowing HTTP 4xx errors.
- Tlon gall agent/scry path corruption: ``third_party_agents`` -> ``channels``.
- Tlon ``authenticate_tlon`` persisting an un-normalized ``ship_url``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kiss.agents.third_party_agents.tlon_agent import TlonChannelBackend
from kiss.agents.third_party_agents.tlon_agent import _config as _tlon_config
from kiss.agents.third_party_agents.twitch_agent import TwitchChannelBackend


class _RecordingHandler(BaseHTTPRequestHandler):
    """Handler that records requests and returns service-shaped JSON."""

    requests_seen: list[dict[str, Any]] = []

    def _record(self, body: bytes) -> None:
        self.requests_seen.append(
            {"method": self.command, "path": self.path, "body": body.decode() if body else ""}
        )

    def _respond_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        """Serve Helix GET and Eyre scry endpoints."""
        self._record(b"")
        if self.path.startswith("/users"):
            self._respond_json(200, {"data": [{"id": "botid123", "login": "kissbot"}]})
        elif self.path.startswith("/channels"):
            self._respond_json(
                200, {"data": [{"broadcaster_id": "123", "broadcaster_login": "streamer"}]}
            )
        elif self.path.startswith("/search/channels"):
            self._respond_json(200, {"data": [{"broadcaster_login": "streamer"}]})
        elif self.path.startswith("/~/scry/channels/"):
            self._respond_json(200, {"channels": {}})
        else:
            self._respond_json(404, {"error": "Not Found", "status": 404})

    def do_PUT(self) -> None:  # noqa: N802
        """Serve Eyre channel PUT (poke) endpoints."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._record(body)
        if self.path.startswith("/~/channel"):
            self.send_response(204)
            self.end_headers()
        else:
            self._respond_json(404, {"error": "Not Found", "status": 404})

    def do_POST(self) -> None:  # noqa: N802
        """Serve Helix POST and Eyre login/poke endpoints."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._record(body)
        if self.path == "/~/login" or self.path.startswith("/~/channel"):
            self.send_response(204)
            self.end_headers()
        elif self.path.startswith("/chat/messages"):
            self._respond_json(200, {"data": [{"message_id": "m1", "is_sent": True}]})
        elif self.path.startswith("/moderation/bans"):
            self._respond_json(
                400, {"error": "Bad Request", "status": 400, "message": "missing field"}
            )
        else:
            self._respond_json(404, {"error": "Not Found", "status": 404})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _paths(method: str | None = None) -> list[str]:
    seen = _RecordingHandler.requests_seen
    return [r["path"] for r in seen if method is None or r["method"] == method]


class TestTwitchBackendBugs:
    """Twitch Helix endpoint corruption, missing sender_id, silent HTTP errors."""

    def setup_method(self) -> None:
        _RecordingHandler.requests_seen = []
        self._server, base = _start_server()
        self.backend = TwitchChannelBackend(helix_base=base)
        self.backend._client_id = "cid"
        self.backend._access_token = "tok"

    def teardown_method(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def test_get_channel_info_hits_channels_endpoint(self) -> None:
        """get_channel_info must call GET /channels?broadcaster_id=..."""
        result = json.loads(self.backend.get_channel_info("123"))
        paths = _paths("GET")
        assert any(p.startswith("/channels?") and "broadcaster_id=123" in p for p in paths), (
            f"expected GET /channels?broadcaster_id=123, saw {paths}"
        )
        assert not any("third_party_agents" in p for p in paths), paths
        assert result["ok"] is True
        assert result["channel"]["broadcaster_login"] == "streamer"

    def test_search_hits_search_channels_endpoint(self) -> None:
        """Channel search must call GET /search/channels."""
        result = json.loads(self.backend.search_third_party_agents("streamer"))
        paths = _paths("GET")
        assert any(p.startswith("/search/channels?") for p in paths), (
            f"expected GET /search/channels, saw {paths}"
        )
        assert not any("third_party_agents" in p for p in paths), paths
        assert result["ok"] is True

    def test_send_message_includes_sender_id(self) -> None:
        """Contract send_message must POST /chat/messages with sender_id."""
        self.backend.send_message("bcast1", "hello there")
        posts = [
            r
            for r in _RecordingHandler.requests_seen
            if r["method"] == "POST" and r["path"].startswith("/chat/messages")
        ]
        assert posts, f"no POST /chat/messages seen: {_RecordingHandler.requests_seen}"
        body = json.loads(posts[-1]["body"])
        assert body["broadcaster_id"] == "bcast1"
        assert body["message"] == "hello there"
        assert body.get("sender_id") == "botid123", f"sender_id missing/wrong in body: {body}"

    def test_send_chat_message_still_includes_sender_id(self) -> None:
        """Explicit send_chat_message keeps passing sender_id through."""
        result = json.loads(self.backend.send_chat_message("bcast1", "sender9", "hi"))
        body = json.loads(_RecordingHandler.requests_seen[-1]["body"])
        assert body["sender_id"] == "sender9"
        assert result["ok"] is True

    def test_post_http_error_is_not_reported_ok(self) -> None:
        """A 4xx Helix response must not be reported as ok: true."""
        result = json.loads(self.backend.ban_user("b1", "m1", "u1", duration=10, reason="spam"))
        assert result["ok"] is False, f"4xx error reported as success: {result}"


class TestTlonBackendBugs:
    """Tlon gall agent name/scry path corruption and ship_url persistence."""

    def setup_method(self) -> None:
        _RecordingHandler.requests_seen = []
        self._server, self.base = _start_server()
        self._backup = None
        if _tlon_config.path.exists():
            self._backup = _tlon_config.path.read_text()
        _tlon_config.save({"ship_url": self.base, "code": "lidlut-tabwed", "ship": "~zod"})
        self.backend = TlonChannelBackend()
        assert self.backend.connect() is True

    def teardown_method(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._backup is not None:
            _tlon_config.path.parent.mkdir(parents=True, exist_ok=True)
            _tlon_config.path.write_text(self._backup)
        elif _tlon_config.path.exists():
            _tlon_config.path.unlink()

    def test_list_channels_scries_channels_agent(self) -> None:
        """Group channel listing must scry the 'channels' gall agent."""
        result = json.loads(self.backend.list_third_party_agents("~zod/my-group"))
        scries = [p for p in _paths("GET") if p.startswith("/~/scry/")]
        assert any(p.startswith("/~/scry/channels/channels/~zod/my-group/light") for p in scries), (
            f"expected scry of channels agent at /channels/..., saw {scries}"
        )
        assert not any("third_party_agents" in p for p in scries), scries
        assert result["ok"] is True

    def test_get_messages_scries_channels_agent(self) -> None:
        """Message fetch must scry the 'channels' gall agent."""
        result = json.loads(self.backend.get_messages("~zod/my-group", "chat", count=5))
        scries = [p for p in _paths("GET") if p.startswith("/~/scry/")]
        expected = "/~/scry/channels/channel/~zod/my-group/chat/posts/"
        assert any(p.startswith(expected) for p in scries), f"expected channels scry, saw {scries}"
        assert not any("third_party_agents" in p for p in scries), scries
        assert result["ok"] is True

    def test_post_message_pokes_channels_agent(self) -> None:
        """post_message must poke the 'channels' gall agent."""
        result = json.loads(self.backend.post_message("~zod/my-group", "chat", "hello"))
        pokes = [
            r
            for r in _RecordingHandler.requests_seen
            if r["method"] in ("POST", "PUT") and r["path"].startswith("/~/channel")
        ]
        assert pokes, f"no poke request seen: {_RecordingHandler.requests_seen}"
        body = json.loads(pokes[-1]["body"])
        poke_obj = body[0] if isinstance(body, list) else body
        assert poke_obj["app"] == "channels", f"poked wrong gall agent: {poke_obj}"
        assert result["ok"] is True

    def test_authenticate_tlon_persists_normalized_ship_url(self) -> None:
        """authenticate_tlon must save the rstrip('/')-normalized ship_url."""
        from kiss.agents.third_party_agents.tlon_agent import TlonAgent

        agent = TlonAgent()
        agent.web_use_tool = None
        tools = {t.__name__: t for t in agent._get_auth_tools()}
        result = json.loads(tools["authenticate_tlon"](ship_url=self.base + "/", code="code123"))
        assert result["ok"] is True
        cfg = _tlon_config.load()
        assert cfg is not None
        assert cfg["ship_url"] == self.base, (
            f"persisted ship_url not normalized: {cfg['ship_url']!r} != {self.base!r}"
        )

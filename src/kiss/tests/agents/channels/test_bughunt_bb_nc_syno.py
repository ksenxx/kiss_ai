# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Round-2 bug-hunt integration tests for BlueBubbles, Nextcloud Talk, and
Synology Chat channel backends.

No mock/patch libraries: real in-process ``ThreadingHTTPServer`` instances
record every request (method/path/query/body) and return BlueBubbles-,
Nextcloud-OCS-, and Synology-shaped JSON.  Backends read their server URL
from ``~/.kiss/third_party_agents/*/config.json``; configs touched by the
tests are backed up and restored around each test.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from kiss.agents.third_party_agents import bluebubbles_agent as bb_mod
from kiss.agents.third_party_agents import nextcloud_talk_agent as nc_mod
from kiss.agents.third_party_agents import synology_chat_agent as syno_mod
from kiss.agents.third_party_agents.bluebubbles_agent import BlueBubblesChannelBackend
from kiss.agents.third_party_agents.nextcloud_talk_agent import NextcloudTalkChannelBackend
from kiss.agents.third_party_agents.synology_chat_agent import SynologyChatChannelBackend

Responder = Callable[[str, str, dict[str, list[str]], bytes], tuple[int, dict[str, Any]]]


class _RecordingServer(ThreadingHTTPServer):
    """HTTP server that records requests and serves configurable JSON."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[dict[str, Any]] = []
        self.responder: Responder = lambda method, path, query, body: (200, {})


class _Handler(BaseHTTPRequestHandler):
    def _handle(self) -> None:
        srv = cast(_RecordingServer, self.server)
        parsed = urlsplit(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        srv.requests.append(
            {
                "method": self.command,
                "path": parsed.path,
                "raw_path": self.path,
                "query": parse_qs(parsed.query),
                "body": body.decode("utf-8", "replace"),
            }
        )
        status, payload = srv.responder(self.command, parsed.path, parse_qs(parsed.query), body)
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


def _start_server() -> tuple[_RecordingServer, str]:
    server = _RecordingServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _stop_server(server: _RecordingServer) -> None:
    server.shutdown()
    server.server_close()


def _backup_config(path: Path) -> str | None:
    if path.exists():
        backup = path.read_text()
        path.unlink()
        return backup
    return None


def _restore_config(path: Path, backup: str | None) -> None:
    if backup is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# BlueBubbles
# ---------------------------------------------------------------------------

_BB_MSG_PENDING = {
    "guid": "m-guid-1",
    "text": "hello agent",
    "dateCreated": 1700000000123,
    "isFromMe": False,
    "sender": {"address": "+15550001111"},
    "chats": [{"guid": "iMessage;-;+15550001111"}],
}
_BB_MSG_FROM_ME = {
    "guid": "m-guid-2",
    "text": "my own reply",
    "dateCreated": 1700000000456,
    "isFromMe": True,
    "sender": {"address": "me@example.com"},
    "chats": [{"guid": "iMessage;-;+15550001111"}],
}


def _bb_responder(messages: list[dict[str, Any]], send_status: int = 200) -> Responder:
    def respond(
        method: str, path: str, query: dict[str, list[str]], body: bytes
    ) -> tuple[int, dict[str, Any]]:
        if path == "/api/v1/server/info":
            return 200, {"status": 200, "data": {"os_version": "test"}}
        if path == "/api/v1/message/query":
            return 200, {"status": 200, "data": messages}
        if path == "/api/v1/message/text":
            if send_status != 200:
                return send_status, {"status": send_status, "error": {"message": "send failed"}}
            return 200, {"status": 200, "data": {"guid": "sent-guid"}}
        return 404, {"status": 404, "error": {"message": f"unknown path {path}"}}

    return respond


class TestBlueBubbles:
    """BlueBubbles backend: cursor handling, chat filtering, send errors."""

    def setup_method(self) -> None:
        self._backup = _backup_config(bb_mod._config.path)
        self._server, self._url = _start_server()

    def teardown_method(self) -> None:
        _stop_server(self._server)
        _restore_config(bb_mod._config.path, self._backup)

    def _make_backend(self) -> BlueBubblesChannelBackend:
        backend = BlueBubblesChannelBackend()
        backend._server_url = self._url
        backend._password = "pw"
        return backend

    @pytest.mark.skipif(sys.platform != "darwin", reason="BlueBubbles connect() is macOS-only")
    def test_poll_oldest_zero_after_connect_returns_pending_message(self) -> None:
        """poll_messages(channel, '0') right after connect() must return the
        pending message the server serves (no after-NOW filter), via the
        documented POST /api/v1/message/query endpoint filtered by chatGuid."""
        bb_mod._config.save({"server_url": self._url, "password": "pw"})
        self._server.responder = _bb_responder([_BB_MSG_PENDING])
        backend = BlueBubblesChannelBackend()
        assert backend.connect() is True
        messages, cursor = backend.poll_messages("iMessage;-;+15550001111", "0", limit=50)
        assert len(messages) == 1
        assert messages[0]["text"] == "hello agent"
        assert messages[0]["user"] == "+15550001111"
        assert messages[0]["chat_guid"] == "iMessage;-;+15550001111"
        assert cursor == "1700000000123"
        query_reqs = [r for r in self._server.requests if r["path"] == "/api/v1/message/query"]
        assert len(query_reqs) == 1
        req = query_reqs[0]
        assert req["method"] == "POST"
        assert req["query"].get("password") == ["pw"]
        req_body = json.loads(req["body"])
        assert req_body["chatGuid"] == "iMessage;-;+15550001111"
        assert "after" not in req_body

    def test_poll_with_real_cursor_sends_after(self) -> None:
        """A non-zero oldest cursor must be forwarded as the 'after' filter."""
        self._server.responder = _bb_responder([_BB_MSG_PENDING])
        backend = self._make_backend()
        backend.poll_messages("iMessage;-;+15550001111", "1600000000000", limit=10)
        req = [r for r in self._server.requests if r["path"] == "/api/v1/message/query"][0]
        req_body = json.loads(req["body"])
        assert req_body["after"] == 1600000000000

    def test_poll_captures_is_from_me_and_is_from_bot(self) -> None:
        """isFromMe must be captured and is_from_bot must flag the bot's own
        messages so the bot does not reply to itself."""
        self._server.responder = _bb_responder([_BB_MSG_FROM_ME, _BB_MSG_PENDING])
        backend = self._make_backend()
        messages, _ = backend.poll_messages("iMessage;-;+15550001111", "0", limit=10)
        assert len(messages) == 2
        by_guid = {m["guid"]: m for m in messages}
        assert backend.is_from_bot(by_guid["m-guid-2"]) is True
        assert backend.is_from_bot(by_guid["m-guid-1"]) is False

    def test_send_message_raises_on_error(self) -> None:
        """send_message must raise RuntimeError on failure (runner retry contract)."""
        self._server.responder = _bb_responder([], send_status=401)
        backend = self._make_backend()
        with pytest.raises(RuntimeError):
            backend.send_message("iMessage;-;+15550001111", "hi")

    def test_send_message_ok_on_success(self) -> None:
        """send_message must not raise when the server reports status 200."""
        self._server.responder = _bb_responder([])
        backend = self._make_backend()
        backend.send_message("iMessage;-;+15550001111", "hi")
        req = [r for r in self._server.requests if r["path"] == "/api/v1/message/text"][0]
        assert json.loads(req["body"])["message"] == "hi"


# ---------------------------------------------------------------------------
# Nextcloud Talk
# ---------------------------------------------------------------------------

_NC_API_PREFIX = "/ocs/v2.php/apps/spreed/api/v4"


def _nc_ocs(status: int, data: Any) -> dict[str, Any]:
    return {"ocs": {"meta": {"status": "ok", "statuscode": status}, "data": data}}


def _nc_responder(
    chat_messages: list[dict[str, Any]], post_statuscode: int = 201
) -> Responder:
    def respond(
        method: str, path: str, query: dict[str, list[str]], body: bytes
    ) -> tuple[int, dict[str, Any]]:
        if path == f"{_NC_API_PREFIX}/room" and method == "GET":
            return 200, _nc_ocs(200, [{"token": "roomtok", "displayName": "Room"}])
        if path == f"{_NC_API_PREFIX}/room/roomtok/participants/active":
            return 200, _nc_ocs(200, {})
        if path == f"{_NC_API_PREFIX}/chat/roomtok" and method == "GET":
            return 200, _nc_ocs(200, chat_messages)
        if path == f"{_NC_API_PREFIX}/chat/roomtok" and method == "POST":
            if post_statuscode >= 400:
                return post_statuscode, _nc_ocs(post_statuscode, {})
            return 201, _nc_ocs(post_statuscode, {"id": 99})
        return 404, _nc_ocs(404, {})

    return respond


_NC_MESSAGES = [
    {"id": 7, "actorId": "alice", "message": "newest", "timestamp": 1700000700},
    {"id": 5, "actorId": "bob", "message": "older", "timestamp": 1700000500},
]


class TestNextcloudTalk:
    """Nextcloud Talk backend: URL normalization, poll cursor/ts, send/join."""

    def setup_method(self) -> None:
        self._backup = _backup_config(nc_mod._config.path)
        self._server, self._url = _start_server()

    def teardown_method(self) -> None:
        _stop_server(self._server)
        _restore_config(nc_mod._config.path, self._backup)

    def _save_config_with_trailing_slash(self) -> None:
        nc_mod._config.save(
            {"url": self._url + "/", "username": "bot", "password": "pw"}
        )

    def test_urls_single_slash_with_trailing_slash_config(self) -> None:
        """All load sites must rstrip('/') the configured URL: no '//' after host."""
        self._save_config_with_trailing_slash()
        self._server.responder = _nc_responder(_NC_MESSAGES)
        backend = NextcloudTalkChannelBackend()
        assert backend.connect() is True
        assert backend._url == self._url, "connect() must rstrip('/') the URL"
        backend.poll_messages("roomtok", "0", limit=10)
        made_backend = nc_mod._make_backend()
        assert made_backend._url == self._url, "_make_backend() must rstrip('/') the URL"
        made_backend.poll_messages("roomtok", "0", limit=10)
        agent = nc_mod.NextcloudTalkAgent()
        assert agent._backend._url == self._url, "__init__ must rstrip('/') the URL"
        assert self._server.requests, "no requests recorded"
        for req in self._server.requests:
            assert "//" not in req["path"], f"double slash in {req['raw_path']}"

    def test_authenticate_persists_normalized_url(self) -> None:
        """authenticate_nextcloud must persist the rstripped URL."""
        self._server.responder = _nc_responder(_NC_MESSAGES)
        agent = nc_mod.NextcloudTalkAgent()
        auth_tools = {t.__name__: t for t in agent._get_auth_tools()}
        result = json.loads(
            auth_tools["authenticate_nextcloud"](self._url + "/", "bot", "pw")
        )
        assert result["ok"] is True
        saved = json.loads(nc_mod._config.path.read_text())
        assert saved["url"] == self._url

    def test_poll_returns_message_id_ts_and_max_cursor(self) -> None:
        """poll_messages must fetch latest (lookIntoFuture=0, no
        lastKnownMessageId), emit ts = Talk message id string, keep the unix
        epoch under 'timestamp', and return max id as the new cursor."""
        self._save_config_with_trailing_slash()
        self._server.responder = _nc_responder(_NC_MESSAGES)
        backend = NextcloudTalkChannelBackend()
        assert backend.connect() is True
        messages, cursor = backend.poll_messages("roomtok", "0", limit=10)
        assert cursor == "7"
        assert sorted(m["ts"] for m in messages) == ["5", "7"]
        newest = [m for m in messages if m["ts"] == "7"][0]
        assert newest["user"] == "alice"
        assert newest["timestamp"] == "1700000700"
        chat_reqs = [
            r
            for r in self._server.requests
            if r["path"] == f"{_NC_API_PREFIX}/chat/roomtok" and r["method"] == "GET"
        ]
        assert chat_reqs
        assert chat_reqs[-1]["query"].get("lookIntoFuture") == ["0"]
        assert "lastKnownMessageId" not in chat_reqs[-1]["query"]
        # Client-side filter: ids <= cursor are not returned again.
        messages2, cursor2 = backend.poll_messages("roomtok", cursor, limit=10)
        assert messages2 == []
        assert cursor2 == "7"

    def test_send_message_posts_reply_to_as_int_message_id(self) -> None:
        """send_message with thread_ts must post replyTo as the int message id."""
        self._save_config_with_trailing_slash()
        self._server.responder = _nc_responder(_NC_MESSAGES)
        backend = NextcloudTalkChannelBackend()
        assert backend.connect() is True
        backend.send_message("roomtok", "reply text", thread_ts="7")
        post_reqs = [
            r
            for r in self._server.requests
            if r["path"] == f"{_NC_API_PREFIX}/chat/roomtok" and r["method"] == "POST"
        ]
        assert len(post_reqs) == 1
        body = json.loads(post_reqs[0]["body"])
        assert body["message"] == "reply text"
        assert body["replyTo"] == 7

    def test_send_message_raises_on_ocs_error(self) -> None:
        """send_message must raise RuntimeError on OCS failure."""
        self._save_config_with_trailing_slash()
        self._server.responder = _nc_responder(_NC_MESSAGES, post_statuscode=400)
        backend = NextcloudTalkChannelBackend()
        assert backend.connect() is True
        with pytest.raises(RuntimeError):
            backend.send_message("roomtok", "reply text")

    def test_join_channel_uses_self_join_endpoint(self) -> None:
        """join_channel must POST to /room/{token}/participants/active."""
        self._save_config_with_trailing_slash()
        self._server.responder = _nc_responder(_NC_MESSAGES)
        backend = NextcloudTalkChannelBackend()
        assert backend.connect() is True
        backend.join_channel("roomtok")
        join_reqs = [r for r in self._server.requests if "/participants" in r["path"]]
        assert len(join_reqs) == 1
        assert join_reqs[0]["path"].endswith("/room/roomtok/participants/active")
        assert join_reqs[0]["method"] == "POST"


# ---------------------------------------------------------------------------
# Synology Chat
# ---------------------------------------------------------------------------


class TestSynologyChat:
    """Synology Chat backend: webhook form parsing, send body, poll drain."""

    def setup_method(self) -> None:
        self._backup = _backup_config(syno_mod._config.path)
        self._server, self._url = _start_server()
        self._backend: SynologyChatChannelBackend | None = None

    def teardown_method(self) -> None:
        if self._backend is not None:
            self._backend.disconnect()
        _stop_server(self._server)
        _restore_config(syno_mod._config.path, self._backup)

    def _post_webhook(self, fields: dict[str, str]) -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{syno_mod._DEFAULT_WEBHOOK_PORT}/",
            data=urlencode(fields).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200

    def test_webhook_parses_synology_form_fields(self) -> None:
        """Synology outgoing webhooks POST form fields directly (not a
        Slack-style payload= JSON blob); the handler must parse them and
        verify the configured token."""
        syno_mod._config.save({"webhook_url": self._url + "/webhook", "token": "sekret"})
        self._backend = SynologyChatChannelBackend()
        assert self._backend.connect() is True
        self._post_webhook(
            {
                "token": "sekret",
                "user_id": "42",
                "username": "alice",
                "post_id": "100001",
                "timestamp": "1700000000",
                "text": "hello bot",
            }
        )
        # A message with a wrong token must be dropped.
        self._post_webhook({"token": "wrong", "user_id": "13", "text": "spoofed"})
        messages, _ = self._backend.poll_messages("", "0", limit=10)
        assert len(messages) == 1
        assert messages[0]["user"] == "42"
        assert messages[0]["text"] == "hello bot"
        assert messages[0]["ts"] == "1700000000"

    def test_send_message_posts_payload_in_body_without_channel_id(self) -> None:
        """send_message must send payload= in the POST body (not URL params)
        and must not include the unsupported channel_id key."""
        self._server.responder = lambda method, path, query, body: (200, {"success": True})
        backend = SynologyChatChannelBackend()
        backend._webhook_url = self._url + "/webhook"
        long_text = "x" * 5000
        backend.send_message("channel-7", long_text)
        req = self._server.requests[0]
        assert req["query"] == {}, "payload must not be sent as URL params"
        form = parse_qs(req["body"])
        payload = json.loads(form["payload"][0])
        assert payload["text"] == long_text
        assert "channel_id" not in payload

    def test_send_message_raises_on_failure(self) -> None:
        """send_message must raise RuntimeError when Synology reports failure."""
        backend = SynologyChatChannelBackend()
        backend._webhook_url = self._url + "/webhook"
        self._server.responder = lambda method, path, query, body: (
            200,
            {"success": False, "error": {"code": 404, "errors": "no such webhook"}},
        )
        with pytest.raises(RuntimeError):
            backend.send_message("", "hi")
        self._server.responder = lambda method, path, query, body: (500, {})
        with pytest.raises(RuntimeError):
            backend.send_message("", "hi")

    def test_poll_messages_uses_shared_queue_drain(self) -> None:
        """poll_messages must drain via drain_queue_messages semantics:
        matching messages returned up to limit."""
        backend = SynologyChatChannelBackend()
        for i in range(3):
            backend._message_queue.put(
                {"ts": str(i), "user": "u", "text": f"m{i}", "channel_id": "chanA"}
            )
        messages, _ = backend.poll_messages("chanA", "0", limit=2)
        assert [m["text"] for m in messages] == ["m0", "m1"]
        messages, _ = backend.poll_messages("chanA", "0", limit=10)
        assert [m["text"] for m in messages] == ["m2"]

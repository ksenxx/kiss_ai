# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ntfy channel agent — no mocks or test doubles.

Runs a real local HTTP server emulating the ntfy API (POST stores
messages per topic, ``GET /{topic}/json?poll=1`` returns them as
newline-delimited JSON) and exercises the backend's publish, poll,
loop-prevention and auth-header code paths against it.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests

import kiss.agents.third_party_agents.ntfy.ntfy_sea as ntfy_agent_mod
from kiss.agents.third_party_agents.ntfy.ntfy_sea import (
    NtfyAgent,
    NtfyChannelBackend,
    _config,
)
from kiss.core.config import kiss_home
from kiss.tests.agents.third_party_agents.muse_test_utils import tools_by_name
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    serve_recording,
)


class _NtfyServer(RecordingServer):
    """State of the ntfy emulator: stored messages per topic and the auth headers seen."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.auth_headers: list[str] = []
        self._next_time = 0
        self._lock = threading.Lock()


class _NtfyHandler(RecordingHandler):
    """Emulates the ntfy publish/poll API.

    POST ``/{topic}`` stores the body as a message (403 for the topic
    ``forbidden``); GET ``/{topic}/json`` returns an ``open`` event
    followed by stored messages newer than ``since`` as JSON lines.
    Records every Authorization header it sees.
    """

    server: _NtfyServer  # type: ignore[assignment]

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Store the posted message under its topic."""
        emulator = self.server
        topic = self.path.strip("/")
        emulator.auth_headers.append(self.headers.get("Authorization", ""))
        body = self._read_raw().decode("utf-8")
        if topic == "forbidden":
            self.send_response(403)
            self.end_headers()
            return
        tags = [t for t in self.headers.get("X-Tags", "").split(",") if t]
        with emulator._lock:
            emulator._next_time += 1
            msg = {
                "id": f"id{emulator._next_time}",
                "time": emulator._next_time,
                "event": "message",
                "topic": topic,
                "message": body,
                "title": self.headers.get("X-Title", ""),
                "tags": tags,
            }
            emulator.messages.setdefault(topic, []).append(msg)
        self._json(200, msg)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Stream the topic's messages newer than ``since`` as JSON lines."""
        emulator = self.server
        emulator.auth_headers.append(self.headers.get("Authorization", ""))
        parsed = urlparse(self.path)
        topic = parsed.path.strip("/").removesuffix("/json")
        if topic == "forbidden":
            self.send_response(403)
            self.end_headers()
            return
        since = parse_qs(parsed.query).get("since", ["all"])[0]
        with emulator._lock:
            stored = list(emulator.messages.get(topic, []))
        if since != "all":
            stored = [m for m in stored if m["time"] > int(since)]
        lines = [json.dumps({"id": "open1", "time": 0, "event": "open", "topic": topic})]
        lines.extend(json.dumps(m) for m in stored)
        self._reply(200, "\n".join(lines) + "\n", "application/x-ndjson")


@pytest.fixture()
def emulator() -> Iterator[_NtfyServer]:
    """Run a local ntfy emulator on a free port for the duration of a test."""
    yield from serve_recording(_NtfyHandler, _NtfyServer)


pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


def _authenticated_agent(emulator: _NtfyServer, topic: str = "alerts", token: str = "") -> Any:
    """Return an NtfyAgent authenticated against the emulator."""
    agent = NtfyAgent()
    tools = tools_by_name(agent)
    result = json.loads(tools["authenticate_ntfy"](topic, server=emulator.base_url, token=token))
    assert result["ok"] is True
    return agent


def test_unauthenticated_agent_exposes_only_auth_trio() -> None:
    """A fresh agent is unauthenticated and offers exactly the auth trio."""
    agent = NtfyAgent()
    assert agent.name == "Ntfy Agent"
    assert agent._is_authenticated() is False
    names = sorted(t.__name__ for t in agent._get_tools())
    assert names == ["authenticate_ntfy", "check_ntfy_auth", "clear_ntfy_auth"]
    assert "Not configured" in agent._get_auth_tools()[0]()


def test_authenticate_persists_config_and_clear_removes_it(emulator: _NtfyServer) -> None:
    """authenticate_ntfy persists 0600 config under KISS_HOME; clear removes it."""
    agent = _authenticated_agent(emulator, token="tok123")
    path = _config.path
    assert path.exists()
    assert kiss_home() in path.parents
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(path.read_text())
    assert saved["topic"] == "alerts"
    assert saved["server"] == emulator.base_url
    assert saved["token"] == "tok123"
    assert saved["echo_tag"] == "kiss-sorcar"

    status = json.loads(agent._get_auth_tools()[0]())
    assert status["ok"] is True and status["topic"] == "alerts"

    tool_names = {t.__name__ for t in agent._get_tools()}
    assert {"publish_notification", "poll_topic"} <= tool_names

    tools = tools_by_name(agent)
    assert "cleared" in tools["clear_ntfy_auth"]()
    assert not path.exists()
    assert agent._is_authenticated() is False


def test_authenticate_rejects_empty_topic() -> None:
    """authenticate_ntfy refuses an empty topic and persists nothing."""
    agent = NtfyAgent()
    tools = tools_by_name(agent)
    assert "empty" in tools["authenticate_ntfy"]("   ")
    assert not _config.path.exists()


def test_publish_then_poll_end_to_end(emulator: _NtfyServer) -> None:
    """publish_notification stores a message the backend polls back normalized."""
    agent = _authenticated_agent(emulator)
    result = json.loads(agent._backend.publish_notification("hello world", title="Build"))
    assert result["ok"] is True and result["id"]

    messages, newest = agent._backend.poll_messages("", "")
    assert len(messages) == 1
    msg = messages[0]
    assert msg["text"] == "Build\n\nhello world"
    assert msg["title"] == "Build"
    assert msg["user"] == "alerts"
    assert msg["channel_id"] == "alerts"
    assert msg["ts"] == newest
    assert isinstance(newest, str) and int(newest) > 0
    assert "kiss-sorcar" in msg["tags"]

    # The non-message "open" event emitted by the emulator was filtered out.
    listed = json.loads(agent._backend.poll_topic())
    assert listed["ok"] is True
    assert [m["text"] for m in listed["messages"]] == ["Build\n\nhello world"]


def test_incremental_poll_with_since_cursor(emulator: _NtfyServer) -> None:
    """Polling with the returned newest cursor yields only newer messages."""
    agent = _authenticated_agent(emulator)
    agent._backend.send_message("", "first")
    _, newest = agent._backend.poll_messages("", "")
    agent._backend.send_message("", "second")
    messages, newer = agent._backend.poll_messages("", newest)
    assert [m["text"] for m in messages] == ["second"]
    assert int(newer) > int(newest)
    assert agent._backend.poll_messages("", newer) == ([], newer)


def test_echo_tagged_messages_are_from_bot(emulator: _NtfyServer) -> None:
    """Messages published by the agent carry the echo tag; foreign ones do not."""
    agent = _authenticated_agent(emulator)
    agent._backend.send_message("", "bot says hi")
    requests.post(f"{emulator.base_url}/alerts", data=b"human says hi", timeout=10)

    messages, _ = agent._backend.poll_messages("", "")
    by_text = {m["text"]: m for m in messages}
    assert agent._backend.is_from_bot(by_text["bot says hi"]) is True
    assert agent._backend.is_from_bot(by_text["human says hi"]) is False


def test_token_sent_as_bearer_authorization(emulator: _NtfyServer) -> None:
    """The configured token is sent as an Authorization header on publish and poll."""
    agent = _authenticated_agent(emulator, token="secret-token")
    agent._backend.publish_notification("with auth")
    agent._backend.poll_messages("", "")
    assert emulator.auth_headers.count("Bearer secret-token") >= 2


def test_publish_to_forbidden_topic_reports_error(emulator: _NtfyServer) -> None:
    """A non-2xx publish surfaces as ok:false from the tool and raises from send."""
    agent = _authenticated_agent(emulator, topic="forbidden")
    result = json.loads(agent._backend.publish_notification("nope"))
    assert result["ok"] is False
    assert "403" in result["error"]
    with pytest.raises(RuntimeError, match="403"):
        agent._backend.send_message("", "nope")


def test_connect_uses_persisted_config(emulator: _NtfyServer) -> None:
    """A fresh backend connects from the persisted config; without it, fails."""
    _authenticated_agent(emulator, topic="ops")
    backend = NtfyChannelBackend()
    assert backend.connect() is True
    assert "ops" in backend.connection_info
    _config.clear()
    fresh = NtfyChannelBackend()
    assert fresh.connect() is False
    assert "No ntfy config" in fresh.connection_info


def test_poll_uses_channel_id_when_unconfigured(emulator: _NtfyServer) -> None:
    """An unconfigured backend polls the topic given as channel_id."""
    requests.post(f"{emulator.base_url}/adhoc", data=b"direct", timeout=10)
    backend = NtfyChannelBackend()
    backend._server = emulator.base_url
    messages, _ = backend.poll_messages("adhoc", "")
    assert [m["text"] for m in messages] == ["direct"]


def test_poll_network_error_returns_empty(emulator: _NtfyServer, refusing_port: int) -> None:
    """Poll failures are swallowed by poll_messages but reported by poll_topic."""
    agent = _authenticated_agent(emulator)
    # A port that is bound but never accepts: unlike stopping the emulator
    # (whose released port another process could be handed), it refuses
    # deterministically.
    agent._backend._server = f"http://127.0.0.1:{refusing_port}"
    assert agent._backend.poll_messages("", "5") == ([], "5")
    result = json.loads(agent._backend.poll_topic(since="5"))
    assert result["ok"] is False
    assert result["error"]


def test_poll_topic_reports_http_error(emulator: _NtfyServer) -> None:
    """poll_topic returns ok:false on a non-2xx poll and ok:true on success."""
    agent = _authenticated_agent(emulator, topic="forbidden")
    result = json.loads(agent._backend.poll_topic())
    assert result["ok"] is False
    assert "403" in result["error"]

    ok_agent = _authenticated_agent(emulator, topic="alerts")
    requests.post(f"{emulator.base_url}/alerts", data=b"hello", timeout=10)
    result = json.loads(ok_agent._backend.poll_topic())
    assert result["ok"] is True
    assert [m["text"] for m in result["messages"]] == ["hello"]


def test_comma_echo_tag_round_trip_is_from_bot(emulator: _NtfyServer) -> None:
    """A comma-separated echo_tag still filters the agent's own messages."""
    agent = NtfyAgent()
    tools = tools_by_name(agent)
    result = json.loads(
        tools["authenticate_ntfy"]("alerts", server=emulator.base_url, echo_tag="kiss, sorcar-bot")
    )
    assert result["ok"] is True
    agent._backend.send_message("", "multi-tag bot message")
    requests.post(f"{emulator.base_url}/alerts", data=b"human message", timeout=10)

    messages, _ = agent._backend.poll_messages("", "")
    by_text = {m["text"]: m for m in messages}
    assert set(by_text["multi-tag bot message"]["tags"]) == {"kiss", "sorcar-bot"}
    assert agent._backend.is_from_bot(by_text["multi-tag bot message"]) is True
    assert agent._backend.is_from_bot(by_text["human message"]) is False

    # Even a message carrying only one echo sub-tag is treated as from the bot.
    requests.post(
        f"{emulator.base_url}/alerts", data=b"partial", headers={"X-Tags": "sorcar-bot"}, timeout=10
    )
    messages, _ = agent._backend.poll_messages("", "")
    partial = next(m for m in messages if m["text"] == "partial")
    assert agent._backend.is_from_bot(partial) is True


def test_make_backend_exits_when_unconfigured() -> None:
    """_make_backend exits with an instruction when no config exists."""
    with pytest.raises(SystemExit):
        ntfy_agent_mod._make_backend()


def test_make_backend_loads_config(emulator: _NtfyServer) -> None:
    """_make_backend returns a backend loaded from the persisted config."""
    _authenticated_agent(emulator, topic="ops")
    backend = ntfy_agent_mod._make_backend()
    assert backend._topic == "ops"
    assert backend._server == emulator.base_url

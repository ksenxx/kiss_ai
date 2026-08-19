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
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests

import kiss.agents.third_party_agents.ntfy_agent as ntfy_agent_mod
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
from kiss.agents.third_party_agents.ntfy_agent import (
    NtfyAgent,
    NtfyChannelBackend,
    _config,
    get_tools,
)
from kiss.core.config import kiss_home


class _NtfyEmulator:
    """Local HTTP server emulating the ntfy publish/poll API.

    POST ``/{topic}`` stores the body as a message (403 for the topic
    ``forbidden``); GET ``/{topic}/json`` returns an ``open`` event
    followed by stored messages newer than ``since`` as JSON lines.
    Records every Authorization header it sees.
    """

    def __init__(self) -> None:
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.auth_headers: list[str] = []
        self._next_time = 0
        self._lock = threading.Lock()
        emulator = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                topic = self.path.strip("/")
                emulator.auth_headers.append(self.headers.get("Authorization", ""))
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
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
                payload = json.dumps(msg).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
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
                payload = ("\n".join(lines) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.server = ThreadedHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut down the emulator server."""
        stop_http_server(self.server, self._thread)


@pytest.fixture()
def emulator() -> Any:
    """Start a local ntfy emulator on a free port and stop it after the test."""
    server = _NtfyEmulator()
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def _clean_config() -> Any:
    """Clear the (KISS_HOME-isolated) ntfy config before and after each test."""
    _config.clear()
    yield
    _config.clear()


def _authenticated_agent(emulator: _NtfyEmulator, topic: str = "alerts", token: str = "") -> Any:
    """Return an NtfyAgent authenticated against the emulator."""
    agent = NtfyAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_ntfy"](topic, server=emulator.url, token=token))
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


def test_authenticate_persists_config_and_clear_removes_it(emulator: _NtfyEmulator) -> None:
    """authenticate_ntfy persists 0600 config under KISS_HOME; clear removes it."""
    agent = _authenticated_agent(emulator, token="tok123")
    path = _config.path
    assert path.exists()
    assert kiss_home() in path.parents
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(path.read_text())
    assert saved["topic"] == "alerts"
    assert saved["server"] == emulator.url
    assert saved["token"] == "tok123"
    assert saved["echo_tag"] == "kiss-sorcar"

    status = json.loads(agent._get_auth_tools()[0]())
    assert status["ok"] is True and status["topic"] == "alerts"

    tool_names = {t.__name__ for t in agent._get_tools()}
    assert {"publish_notification", "poll_topic"} <= tool_names

    tools = {t.__name__: t for t in agent._get_tools()}
    assert "cleared" in tools["clear_ntfy_auth"]()
    assert not path.exists()
    assert agent._is_authenticated() is False


def test_authenticate_rejects_empty_topic() -> None:
    """authenticate_ntfy refuses an empty topic and persists nothing."""
    agent = NtfyAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "empty" in tools["authenticate_ntfy"]("   ")
    assert not _config.path.exists()


def test_get_tools_module_function() -> None:
    """The module-level get_tools() returns a non-empty tool list."""
    tools = get_tools()
    assert tools
    assert "check_ntfy_auth" in {t.__name__ for t in tools}


def test_publish_then_poll_end_to_end(emulator: _NtfyEmulator) -> None:
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


def test_incremental_poll_with_since_cursor(emulator: _NtfyEmulator) -> None:
    """Polling with the returned newest cursor yields only newer messages."""
    agent = _authenticated_agent(emulator)
    agent._backend.send_message("", "first")
    _, newest = agent._backend.poll_messages("", "")
    agent._backend.send_message("", "second")
    messages, newer = agent._backend.poll_messages("", newest)
    assert [m["text"] for m in messages] == ["second"]
    assert int(newer) > int(newest)
    assert agent._backend.poll_messages("", newer) == ([], newer)


def test_echo_tagged_messages_are_from_bot(emulator: _NtfyEmulator) -> None:
    """Messages published by the agent carry the echo tag; foreign ones do not."""
    agent = _authenticated_agent(emulator)
    agent._backend.send_message("", "bot says hi")
    requests.post(f"{emulator.url}/alerts", data=b"human says hi", timeout=10)

    messages, _ = agent._backend.poll_messages("", "")
    by_text = {m["text"]: m for m in messages}
    assert agent._backend.is_from_bot(by_text["bot says hi"]) is True
    assert agent._backend.is_from_bot(by_text["human says hi"]) is False


def test_token_sent_as_bearer_authorization(emulator: _NtfyEmulator) -> None:
    """The configured token is sent as an Authorization header on publish and poll."""
    agent = _authenticated_agent(emulator, token="secret-token")
    agent._backend.publish_notification("with auth")
    agent._backend.poll_messages("", "")
    assert emulator.auth_headers.count("Bearer secret-token") >= 2


def test_publish_to_forbidden_topic_reports_error(emulator: _NtfyEmulator) -> None:
    """A non-2xx publish surfaces as ok:false from the tool and raises from send."""
    agent = _authenticated_agent(emulator, topic="forbidden")
    result = json.loads(agent._backend.publish_notification("nope"))
    assert result["ok"] is False
    assert "403" in result["error"]
    with pytest.raises(RuntimeError, match="403"):
        agent._backend.send_message("", "nope")


def test_connect_uses_persisted_config(emulator: _NtfyEmulator) -> None:
    """A fresh backend connects from the persisted config; without it, fails."""
    _authenticated_agent(emulator, topic="ops")
    backend = NtfyChannelBackend()
    assert backend.connect() is True
    assert "ops" in backend.connection_info
    _config.clear()
    fresh = NtfyChannelBackend()
    assert fresh.connect() is False
    assert "No ntfy config" in fresh.connection_info


def test_poll_uses_channel_id_when_unconfigured(emulator: _NtfyEmulator) -> None:
    """An unconfigured backend polls the topic given as channel_id."""
    requests.post(f"{emulator.url}/adhoc", data=b"direct", timeout=10)
    backend = NtfyChannelBackend()
    backend._server = emulator.url
    messages, _ = backend.poll_messages("adhoc", "")
    assert [m["text"] for m in messages] == ["direct"]


def test_poll_network_error_returns_empty(emulator: _NtfyEmulator) -> None:
    """Poll failures are swallowed by poll_messages but reported by poll_topic."""
    agent = _authenticated_agent(emulator)
    emulator.stop()
    assert agent._backend.poll_messages("", "5") == ([], "5")
    result = json.loads(agent._backend.poll_topic(since="5"))
    assert result["ok"] is False
    assert result["error"]


def test_poll_topic_reports_http_error(emulator: _NtfyEmulator) -> None:
    """poll_topic returns ok:false on a non-2xx poll and ok:true on success."""
    agent = _authenticated_agent(emulator, topic="forbidden")
    result = json.loads(agent._backend.poll_topic())
    assert result["ok"] is False
    assert "403" in result["error"]

    ok_agent = _authenticated_agent(emulator, topic="alerts")
    requests.post(f"{emulator.url}/alerts", data=b"hello", timeout=10)
    result = json.loads(ok_agent._backend.poll_topic())
    assert result["ok"] is True
    assert [m["text"] for m in result["messages"]] == ["hello"]


def test_comma_echo_tag_round_trip_is_from_bot(emulator: _NtfyEmulator) -> None:
    """A comma-separated echo_tag still filters the agent's own messages."""
    agent = NtfyAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(
        tools["authenticate_ntfy"]("alerts", server=emulator.url, echo_tag="kiss, sorcar-bot")
    )
    assert result["ok"] is True
    agent._backend.send_message("", "multi-tag bot message")
    requests.post(f"{emulator.url}/alerts", data=b"human message", timeout=10)

    messages, _ = agent._backend.poll_messages("", "")
    by_text = {m["text"]: m for m in messages}
    assert set(by_text["multi-tag bot message"]["tags"]) == {"kiss", "sorcar-bot"}
    assert agent._backend.is_from_bot(by_text["multi-tag bot message"]) is True
    assert agent._backend.is_from_bot(by_text["human message"]) is False

    # Even a message carrying only one echo sub-tag is treated as from the bot.
    requests.post(
        f"{emulator.url}/alerts", data=b"partial", headers={"X-Tags": "sorcar-bot"}, timeout=10
    )
    messages, _ = agent._backend.poll_messages("", "")
    partial = next(m for m in messages if m["text"] == "partial")
    assert agent._backend.is_from_bot(partial) is True


def test_title_prepended_to_poll_text(emulator: _NtfyEmulator) -> None:
    """A published title is prepended to the normalized message text."""
    agent = _authenticated_agent(emulator)
    requests.post(
        f"{emulator.url}/alerts", data=b"disk almost full", headers={"X-Title": "Alert"}, timeout=10
    )
    messages, _ = agent._backend.poll_messages("", "")
    assert messages[0]["text"] == "Alert\n\ndisk almost full"
    assert messages[0]["title"] == "Alert"


def test_make_backend_exits_when_unconfigured() -> None:
    """_make_backend exits with an instruction when no config exists."""
    with pytest.raises(SystemExit):
        ntfy_agent_mod._make_backend()


def test_make_backend_loads_config(emulator: _NtfyEmulator) -> None:
    """_make_backend returns a backend loaded from the persisted config."""
    _authenticated_agent(emulator, topic="ops")
    backend = ntfy_agent_mod._make_backend()
    assert backend._topic == "ops"
    assert backend._server == emulator.url


def test_main_exits_with_no_args() -> None:
    """main() prints usage and exits when called without arguments."""
    original_argv = sys.argv
    sys.argv = ["kiss-ntfy"]
    try:
        with pytest.raises(SystemExit):
            ntfy_agent_mod.main()
    finally:
        sys.argv = original_argv

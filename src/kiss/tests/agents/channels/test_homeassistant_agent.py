# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Home Assistant channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Home Assistant REST API endpoints — no mocks, patches, or fakes.
The server asserts the ``Authorization: Bearer`` header on every call,
returns canned JSON, and records POSTed bodies for verification.

Config state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and
``ChannelConfig.path`` resolves ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import json
import stat
import sys
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

import kiss.agents.third_party_agents.homeassistant_agent as ha_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.homeassistant_agent import (
    HomeAssistantAgent,
    HomeAssistantChannelBackend,
    _config,
)

_TOKEN = "test-ha-token"

_STATE_KITCHEN = {"entity_id": "light.kitchen", "state": "on", "attributes": {"brightness": 200}}
_ALL_STATES = [_STATE_KITCHEN, {"entity_id": "switch.fan", "state": "off", "attributes": {}}]
_SERVICES = [{"domain": "light", "services": {"turn_on": {}, "turn_off": {}}}]
_HISTORY = [[{"entity_id": "light.kitchen", "state": "on"}]]


class _HARequestHandler(BaseHTTPRequestHandler):
    """Emulates the Home Assistant REST API and records requests."""

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {_TOKEN}"

    def _reply(self, status: int, body: str, content_type: str = "application/json") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, body: dict[str, Any] | None) -> None:
        cast(_HAServer, self.server).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "body": body,
            }
        )

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._record(None)
        if not self._authorized():
            self._reply(401, json.dumps({"message": "Unauthorized"}))
            return
        path = urlparse(self.path).path
        if path == "/api/fail":
            self._reply(500, json.dumps({"message": "Internal Server Error"}))
        elif path == "/api/states":
            self._reply(200, json.dumps(_ALL_STATES))
        elif path == "/api/states/light.kitchen":
            self._reply(200, json.dumps(_STATE_KITCHEN))
        elif path.startswith("/api/history/period/"):
            self._reply(200, json.dumps(_HISTORY))
        elif path == "/api/services":
            self._reply(200, json.dumps(_SERVICES))
        else:
            self._reply(404, json.dumps({"message": "Not found"}))

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        body: dict[str, Any] | None
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            body = None
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"message": "Unauthorized"}))
            return
        path = urlparse(self.path).path
        if path == "/api/template":
            self._reply(200, "rendered: 21.5", content_type="text/plain")
        elif path.startswith("/api/services/"):
            self._reply(200, json.dumps([]))
        elif path.startswith("/api/events/"):
            self._reply(200, json.dumps({"message": "Event fired."}))
        else:
            self._reply(404, json.dumps({"message": "Not found"}))

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _HAServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _HARequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture()
def ha_server():
    """Start the emulated HA REST server on a free port; yield (base_url, server)."""
    server = _HAServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(ha_server):
    """A backend pointed at the emulated server with the valid token."""
    base_url, server = ha_server
    b = HomeAssistantChannelBackend()
    b._base_url = base_url
    b._token = _TOKEN
    return b, server


@pytest.fixture(autouse=True)
def _fresh_config():
    """Start and end every test with no persisted HA config."""
    _config.clear()
    yield
    _config.clear()


def test_agent_instantiation_unauthenticated() -> None:
    """A fresh agent is unauthenticated and exposes only the auth trio."""
    agent = HomeAssistantAgent()
    assert agent.name == "Home Assistant Agent"
    assert agent._is_authenticated() is False
    names = [t.__name__ for t in agent._get_tools()]
    assert names == [
        "check_homeassistant_auth",
        "authenticate_homeassistant",
        "clear_homeassistant_auth",
    ]


def test_check_auth_unauthenticated_message() -> None:
    """check_homeassistant_auth explains how to configure when unauthenticated."""
    agent = HomeAssistantAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_homeassistant_auth"]()
    assert "authenticate_homeassistant" in msg
    assert "long-lived" in msg.lower()


def test_authenticate_persists_config_and_exposes_tools() -> None:
    """authenticate_homeassistant persists config (0600) and unlocks backend tools."""
    agent = HomeAssistantAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_homeassistant"]("http://ha.local:8123", _TOKEN))
    assert result["ok"] is True

    assert _config.path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(_config.path.stat().st_mode) == 0o600
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {"base_url": "http://ha.local:8123", "token": _TOKEN}

    assert agent._is_authenticated() is True
    checked = json.loads(tools["check_homeassistant_auth"]())
    assert checked == {"ok": True, "base_url": "http://ha.local:8123"}

    names = {t.__name__ for t in agent._get_tools()}
    assert {
        "ha_get_states",
        "ha_call_service",
        "ha_list_services",
        "ha_get_history",
        "ha_render_template",
        "ha_fire_event",
    } <= names
    assert "send_message" not in names  # channel protocol method, not an LLM tool

    cleared = tools["clear_homeassistant_auth"]()
    assert "cleared" in cleared.lower()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False
    assert len(agent._get_tools()) == 3


def test_authenticate_rejects_empty_values() -> None:
    """authenticate_homeassistant refuses empty base_url or token."""
    agent = HomeAssistantAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "cannot be empty" in tools["authenticate_homeassistant"]("", _TOKEN)
    assert "cannot be empty" in tools["authenticate_homeassistant"]("http://x", "  ")
    assert not _config.path.exists()


def test_new_agent_loads_persisted_config() -> None:
    """A new agent picks up previously persisted credentials."""
    _config.save({"base_url": "http://ha.local:8123", "token": _TOKEN})
    agent = HomeAssistantAgent()
    assert agent._is_authenticated() is True
    assert agent._backend._base_url == "http://ha.local:8123"
    assert agent._backend._token == _TOKEN


def test_get_tools_module_function() -> None:
    """Module-level get_tools() returns a non-empty tool list."""
    tools = ha_mod.get_tools()
    assert len(tools) >= 3
    assert all(callable(t) for t in tools)


def test_connect_without_config_fails() -> None:
    """connect() fails cleanly when no config is persisted."""
    b = HomeAssistantChannelBackend()
    assert b.connect() is False
    assert "No Home Assistant config" in b.connection_info


def test_connect_with_config_succeeds(ha_server) -> None:
    """connect() loads persisted config into the backend."""
    base_url, _ = ha_server
    _config.save({"base_url": base_url, "token": _TOKEN})
    b = HomeAssistantChannelBackend()
    assert b.connect() is True
    assert b._base_url == base_url
    assert b._token == _TOKEN
    assert base_url in b.connection_info


def test_poll_messages_returns_empty(backend) -> None:
    """poll_messages returns no messages: HA REST has no inbound stream."""
    b, server = backend
    messages, cursor = b.poll_messages("anything", "42", limit=5)
    assert messages == []
    assert cursor == "42"
    assert server.requests == []  # no HTTP traffic


def test_ha_get_states_all(backend) -> None:
    """ha_get_states() with no entity_id hits /api/states with the Bearer header."""
    b, server = backend
    result = json.loads(b.ha_get_states())
    assert result["ok"] is True
    assert result["result"] == _ALL_STATES
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/api/states")
    assert req["authorization"] == f"Bearer {_TOKEN}"


def test_ha_get_states_single_entity(backend) -> None:
    """ha_get_states('light.kitchen') hits /api/states/light.kitchen."""
    b, server = backend
    result = json.loads(b.ha_get_states("light.kitchen"))
    assert result["ok"] is True
    assert result["result"] == _STATE_KITCHEN
    assert server.requests[-1]["path"] == "/api/states/light.kitchen"


def test_ha_call_service_merges_entity_id(backend) -> None:
    """ha_call_service merges entity_id into the posted JSON body."""
    b, server = backend
    result = json.loads(
        b.ha_call_service("light", "turn_on", "light.kitchen", '{"brightness": 100}')
    )
    assert result["ok"] is True
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/api/services/light/turn_on")
    assert req["authorization"] == f"Bearer {_TOKEN}"
    assert req["body"] == {"brightness": 100, "entity_id": "light.kitchen"}


def test_ha_call_service_without_data(backend) -> None:
    """ha_call_service with no data_json posts just the entity_id."""
    b, server = backend
    result = json.loads(b.ha_call_service("switch", "toggle", "switch.fan"))
    assert result["ok"] is True
    assert server.requests[-1]["body"] == {"entity_id": "switch.fan"}


def test_ha_call_service_invalid_data_json(backend) -> None:
    """Invalid data_json yields ok:false without raising or making a request."""
    b, server = backend
    result = json.loads(b.ha_call_service("light", "turn_on", "", "{not json"))
    assert result["ok"] is False
    assert server.requests == []
    result = json.loads(b.ha_call_service("light", "turn_on", "", "[1, 2]"))
    assert result == {"ok": False, "error": "data_json must be a JSON object"}
    assert server.requests == []


def test_ha_list_services(backend) -> None:
    """ha_list_services returns the service catalog from GET /api/services."""
    b, server = backend
    result = json.loads(b.ha_list_services())
    assert result["ok"] is True
    assert result["result"] == _SERVICES
    assert server.requests[-1]["path"] == "/api/services"


def test_ha_get_history(backend) -> None:
    """ha_get_history queries /api/history/period/{iso}?filter_entity_id=..."""
    b, server = backend
    before = datetime.now(UTC)
    result = json.loads(b.ha_get_history("light.kitchen", hours=2))
    assert result["ok"] is True
    assert result["result"] == _HISTORY
    parsed = urlparse(server.requests[-1]["path"])
    assert parsed.path.startswith("/api/history/period/")
    assert parse_qs(parsed.query)["filter_entity_id"] == ["light.kitchen"]
    iso_start = parsed.path.removeprefix("/api/history/period/")
    start = datetime.fromisoformat(iso_start.replace("Z", "+00:00"))
    expected = before - timedelta(hours=2)
    assert abs((start - expected).total_seconds()) < 60


def test_ha_render_template(backend) -> None:
    """ha_render_template posts the template and returns the plain-text result."""
    b, server = backend
    template = "{{ states('sensor.temperature') }}"
    result = json.loads(b.ha_render_template(template))
    assert result == {"ok": True, "result": "rendered: 21.5"}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/api/template")
    assert req["body"] == {"template": template}


def test_ha_fire_event(backend) -> None:
    """ha_fire_event posts event data to /api/events/{event_type}."""
    b, server = backend
    result = json.loads(b.ha_fire_event("kiss_alert", '{"level": "info"}'))
    assert result["ok"] is True
    assert result["result"] == {"message": "Event fired."}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/api/events/kiss_alert")
    assert req["body"] == {"level": "info"}


def test_ha_fire_event_invalid_data_json(backend) -> None:
    """ha_fire_event rejects malformed event data without raising."""
    b, server = backend
    result = json.loads(b.ha_fire_event("kiss_alert", "not json"))
    assert result["ok"] is False
    assert server.requests == []


def test_path_traversal_values_rejected_before_any_request(backend) -> None:
    """Traversal attempts like '../services' are refused up front — no HTTP request."""
    b, server = backend
    for attempt in (
        b.ha_get_states("../services"),
        b.ha_get_states("light/../../api/services"),
        b.ha_call_service("../states", "x"),
        b.ha_call_service("../states", "x", "sensor.injected"),
        b.ha_call_service("light", "../../states/sensor.injected"),
        b.ha_fire_event("../states"),
        b.ha_get_history("../services"),
        b.ha_get_states("a\\b"),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid" in result["error"]
    assert server.requests == []  # nothing ever reached the server


def test_entity_id_is_encoded_as_single_path_segment(backend) -> None:
    """Special characters in entity_id are percent-encoded, never path-interpreted."""
    b, server = backend
    result = json.loads(b.ha_get_states("light.a b%c"))
    assert result["ok"] is False  # emulator 404s the unknown entity — that's fine
    assert server.requests[-1]["path"] == "/api/states/light.a%20b%25c"


def test_history_filter_entity_id_is_url_encoded(backend) -> None:
    """The filter_entity_id query value is URL-encoded in the raw request."""
    b, server = backend
    result = json.loads(b.ha_get_history("light.a b", hours=1))
    assert result["ok"] is True
    parsed = urlparse(server.requests[-1]["path"])
    assert "filter_entity_id=light.a%20b" in parsed.query
    assert parse_qs(parsed.query)["filter_entity_id"] == ["light.a b"]


def test_send_message_creates_persistent_notification(backend) -> None:
    """send_message posts to persistent_notification/create with message and title."""
    b, server = backend
    b.send_message("Alerts", "Door left open")
    req = server.requests[-1]
    assert (req["method"], req["path"]) == (
        "POST",
        "/api/services/persistent_notification/create",
    )
    assert req["body"] == {"message": "Door left open", "title": "Alerts"}


def test_send_message_default_title(backend) -> None:
    """send_message uses the 'KISS Sorcar' title when channel_id is empty."""
    b, server = backend
    b.send_message("", "hello", thread_ts="ignored")
    assert server.requests[-1]["body"] == {"message": "hello", "title": "KISS Sorcar"}


def test_unauthorized_token_returns_ok_false(ha_server) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    base_url, _ = ha_server
    b = HomeAssistantChannelBackend()
    b._base_url = base_url
    b._token = "wrong-token"
    for call in (
        lambda: b.ha_get_states(),
        lambda: b.ha_call_service("light", "turn_on", "light.kitchen"),
        lambda: b.ha_list_services(),
        lambda: b.ha_get_history("light.kitchen"),
        lambda: b.ha_render_template("{{ 1 }}"),
        lambda: b.ha_fire_event("e"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the server yields ok:false JSON, not an exception."""
    b, _ = backend
    result = json.loads(b._request("GET", "/api/fail"))
    assert result["ok"] is False
    assert "500" in result["error"]


def test_connection_refused_returns_ok_false() -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = HomeAssistantChannelBackend()
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._token = _TOKEN
    result = json.loads(b.ha_get_states())
    assert result["ok"] is False
    assert result["error"]


def test_send_message_raises_on_failure(ha_server) -> None:
    """send_message raises RuntimeError on failure so ChannelRunner can retry."""
    base_url, _ = ha_server
    b = HomeAssistantChannelBackend()
    b._base_url = base_url
    b._token = "wrong-token"
    with pytest.raises(RuntimeError, match="401"):
        b.send_message("Alerts", "text")

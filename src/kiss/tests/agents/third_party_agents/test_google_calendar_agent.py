# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Google Calendar channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Google Calendar v3 REST API — no mocks, patches, or fakes.  The
server asserts the ``Authorization: Bearer`` header on every call,
returns canned JSON, and records every request for verification.

Token state is isolated because the session conftest points
``KISS_HOME`` at a temporary directory; an autouse fixture additionally
clears the google_calendar token around every test.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

import kiss.agents.third_party_agents.gcal_sea as gcal_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    clear_google_credentials,
    token_path,
)
from kiss.agents.third_party_agents.gcal_sea import (
    _SCOPES,
    _SERVICE,
    GoogleCalendarAgent,
    GoogleCalendarChannelBackend,
)

_TOKEN = "test-token"

_EVENT = {
    "id": "ev1",
    "summary": "Standup",
    "start": {"dateTime": "2025-06-02T09:00:00-07:00"},
    "end": {"dateTime": "2025-06-02T09:15:00-07:00"},
    "location": "Room 1",
    "status": "confirmed",
    "organizer": {"email": "boss@example.com"},
    "attendees": [{"email": "dev@example.com", "responseStatus": "accepted"}],
    "htmlLink": "https://calendar.google.com/event?eid=ev1",
    "etag": "should-be-dropped",
    "iCalUID": "should-be-dropped-too",
}

_CONDENSED_EVENT = {
    "id": "ev1",
    "summary": "Standup",
    "start": {"dateTime": "2025-06-02T09:00:00-07:00"},
    "end": {"dateTime": "2025-06-02T09:15:00-07:00"},
    "location": "Room 1",
    "status": "confirmed",
    "organizer": {"email": "boss@example.com"},
    "attendees": [{"email": "dev@example.com", "responseStatus": "accepted"}],
    "htmlLink": "https://calendar.google.com/event?eid=ev1",
}

_CALENDARS = {
    "items": [
        {"id": "primary", "summary": "Main", "primary": True, "accessRole": "owner"},
        {"id": "team@example.com", "summary": "Team", "accessRole": "reader"},
    ]
}

_AUTH_TOOL_NAMES = [
    "check_google_calendar_auth",
    "authenticate_google_calendar",
    "clear_google_calendar_auth",
    "start_google_calendar_browser_setup",
    "finish_google_calendar_auth",
]


def write_synthetic_token() -> None:
    """Persist a synthetic, never-expiring OAuth token for google_calendar."""
    path = token_path(_SERVICE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "token": "synthetic-access-token",
                "refresh_token": "synthetic-refresh-token",
                "client_id": "synthetic-client-id",
                "client_secret": "synthetic-client-secret",
                "scopes": _SCOPES,
                "expiry": "2099-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


class _CalendarRequestHandler(BaseHTTPRequestHandler):
    """Emulates the Google Calendar v3 REST API and records requests."""

    def _reply(self, status: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if not raw:
            return None
        try:
            return cast(dict[str, Any], json.loads(raw.decode("utf-8")))
        except ValueError:
            return None

    def _record(self, body: dict[str, Any] | None) -> None:
        cast(_CalendarServer, self.server).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "body": body,
            }
        )

    def _handle(self) -> None:
        body = self._read_body()
        self._record(body)
        if self.headers.get("Authorization") != f"Bearer {_TOKEN}":
            self._reply(401, json.dumps({"error": {"code": 401, "message": "Unauthorized"}}))
            return
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        if "/calendars/boom/" in path:
            self._reply(500, json.dumps({"error": {"code": 500, "message": "boom"}}))
        elif self.command == "GET" and path == "/plaintext":
            payload = b"pong"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.command == "GET" and path == "/users/me/calendarList":
            self._reply(200, json.dumps(_CALENDARS))
        elif self.command == "GET" and path.endswith("/events"):
            page: dict[str, Any] = {"items": [_EVENT]}
            if query.get("pageToken") == ["tok-1"]:
                page["nextPageToken"] = "tok-2"
            self._reply(200, json.dumps(page))
        elif self.command == "GET" and "/events/" in path:
            self._reply(200, json.dumps(_EVENT))
        elif self.command == "POST" and path.endswith("/events/quickAdd"):
            text = query.get("text", [""])[0]
            self._reply(200, json.dumps({"id": "qa1", "summary": text, "status": "confirmed"}))
        elif self.command == "POST" and path.endswith("/events"):
            self._reply(200, json.dumps({"id": "new-ev"} | (body or {})))
        elif self.command == "PATCH" and "/events/" in path:
            event_id = path.rsplit("/", 1)[1]
            self._reply(200, json.dumps({"id": event_id} | (body or {})))
        elif self.command == "DELETE" and "/events/" in path:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._reply(404, json.dumps({"error": {"code": 404, "message": "Not found"}}))

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._handle()

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._handle()

    def do_PATCH(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._handle()

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _CalendarServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _CalendarRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _fresh_token():
    """Start and end every test with no persisted google_calendar token."""
    clear_google_credentials(_SERVICE)
    yield
    clear_google_credentials(_SERVICE)


@pytest.fixture()
def gcal_server():
    """Start the emulated Calendar server on a free port; yield (base_url, server)."""
    server = _CalendarServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(gcal_server):
    """A backend pointed at the emulated server with the valid token."""
    base_url, server = gcal_server
    b = GoogleCalendarChannelBackend()
    b._base_url = base_url
    b._token = _TOKEN
    return b, server


def _last(server: _CalendarServer) -> dict[str, Any]:
    return server.requests[-1]


def test_agent_unauthenticated_exposes_only_auth_tools() -> None:
    """A fresh agent is unauthenticated and exposes exactly the 4 auth tools."""
    agent = GoogleCalendarAgent()
    assert agent.name == "Google Calendar Agent"
    assert agent._is_authenticated() is False
    assert [t.__name__ for t in agent._get_tools()] == _AUTH_TOOL_NAMES


def test_check_auth_unauthenticated_explains_setup() -> None:
    """check_google_calendar_auth explains how to set up credentials."""
    agent = GoogleCalendarAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_google_calendar_auth"]()
    assert "Not authenticated with Google Calendar" in msg
    assert "start_google_calendar_browser_setup" in msg
    assert "authenticate_google_calendar" in msg


def test_synthetic_token_authenticates_new_agent() -> None:
    """A synthetic token.json makes a new agent authenticated end-to-end."""
    write_synthetic_token()
    agent = GoogleCalendarAgent()
    assert agent._is_authenticated() is True
    names = {t.__name__ for t in agent._get_tools()}
    assert set(_AUTH_TOOL_NAMES) <= names
    assert {
        "gcal_list_calendars",
        "gcal_list_events",
        "gcal_get_event",
        "gcal_create_event",
        "gcal_update_event",
        "gcal_delete_event",
        "gcal_quick_add",
    } <= names
    assert "connect" not in names  # channel protocol method, not an LLM tool
    tools = {t.__name__: t for t in agent._get_tools()}
    assert json.loads(tools["check_google_calendar_auth"]())["ok"] is True


def test_clear_auth_removes_token_and_relocks_tools() -> None:
    """clear_google_calendar_auth deletes the token and re-locks backend tools."""
    write_synthetic_token()
    agent = GoogleCalendarAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["clear_google_calendar_auth"]()
    assert "cleared" in result.lower()
    assert not token_path(_SERVICE).exists()
    assert agent._is_authenticated() is False
    assert [t.__name__ for t in agent._get_tools()] == _AUTH_TOOL_NAMES


def test_tools_module_function() -> None:
    """Module-level tools() returns the auth tool set when locked."""
    tools = gcal_mod.tools()
    assert [t.__name__ for t in tools] == _AUTH_TOOL_NAMES
    assert all(callable(t) for t in tools)


def test_connect_without_token_fails() -> None:
    """connect() fails cleanly when no token is persisted."""
    b = GoogleCalendarChannelBackend()
    assert b.connect() is False
    assert "No Google Calendar credentials" in b.connection_info


def test_connect_with_synthetic_token_succeeds() -> None:
    """connect() loads persisted synthetic credentials."""
    write_synthetic_token()
    b = GoogleCalendarChannelBackend()
    assert b.connect() is True
    assert b._creds is not None
    assert "credentials loaded" in b.connection_info


def test_headers_use_stored_credentials_when_no_token_override() -> None:
    """Without a _token override, the bearer comes from the stored credentials."""
    write_synthetic_token()
    b = GoogleCalendarChannelBackend()
    assert b.connect() is True
    assert b._headers() == {"Authorization": "Bearer synthetic-access-token"}


def test_list_calendars(backend) -> None:
    """gcal_list_calendars hits /users/me/calendarList and condenses items."""
    b, server = backend
    result = json.loads(b.gcal_list_calendars())
    assert result["ok"] is True
    assert result["calendars"] == [
        {"id": "primary", "summary": "Main", "primary": True, "access_role": "owner"},
        {"id": "team@example.com", "summary": "Team", "primary": False, "access_role": "reader"},
    ]
    req = _last(server)
    assert req["method"] == "GET"
    assert urlparse(req["path"]).path == "/users/me/calendarList"
    assert req["authorization"] == f"Bearer {_TOKEN}"


def test_list_events_defaults(backend) -> None:
    """gcal_list_events sends singleEvents/orderBy and omits empty filters."""
    b, server = backend
    result = json.loads(b.gcal_list_events())
    assert result["ok"] is True
    assert result["events"] == [_CONDENSED_EVENT]
    parsed = urlparse(_last(server)["path"])
    assert parsed.path == "/calendars/primary/events"
    query = parse_qs(parsed.query)
    assert query["singleEvents"] == ["true"]
    assert query["orderBy"] == ["startTime"]
    assert query["maxResults"] == ["20"]
    assert "timeMin" not in query
    assert "timeMax" not in query
    assert "q" not in query


def test_list_events_with_filters(backend) -> None:
    """gcal_list_events forwards time_min/time_max/query/max_results."""
    b, server = backend
    result = json.loads(
        b.gcal_list_events(
            calendar_id="team@example.com",
            time_min="2025-06-01T00:00:00Z",
            time_max="2025-06-08T00:00:00Z",
            query="standup",
            max_results=5,
        )
    )
    assert result["ok"] is True
    parsed = urlparse(_last(server)["path"])
    assert parsed.path == "/calendars/team%40example.com/events"
    query = parse_qs(parsed.query)
    assert query["timeMin"] == ["2025-06-01T00:00:00Z"]
    assert query["timeMax"] == ["2025-06-08T00:00:00Z"]
    assert query["q"] == ["standup"]
    assert query["maxResults"] == ["5"]


def test_list_events_pagination(backend) -> None:
    """gcal_list_events forwards page_token as pageToken and surfaces next_page_token."""
    b, server = backend
    result = json.loads(b.gcal_list_events(page_token="tok-1"))
    assert result["ok"] is True
    assert result["events"] == [_CONDENSED_EVENT]
    assert result["next_page_token"] == "tok-2"
    query = parse_qs(urlparse(_last(server)["path"]).query)
    assert query["pageToken"] == ["tok-1"]
    # Without a page_token, none is forwarded and the last page (no
    # nextPageToken from the API) omits next_page_token from the result.
    result = json.loads(b.gcal_list_events())
    assert result["ok"] is True
    assert "next_page_token" not in result
    assert "pageToken" not in parse_qs(urlparse(_last(server)["path"]).query)


def test_get_event(backend) -> None:
    """gcal_get_event fetches one event and condenses it."""
    b, server = backend
    result = json.loads(b.gcal_get_event("ev1"))
    assert result["ok"] is True
    assert result["event"] == _CONDENSED_EVENT
    req = _last(server)
    assert req["method"] == "GET"
    assert urlparse(req["path"]).path == "/calendars/primary/events/ev1"


def test_create_timed_event_with_all_fields(backend) -> None:
    """gcal_create_event posts dateTime objects, attendees, and timeZone."""
    b, server = backend
    result = json.loads(
        b.gcal_create_event(
            summary="Design review",
            start="2025-06-03T14:00:00",
            end="2025-06-03T15:00:00",
            description="Quarterly design review",
            location="HQ",
            attendees="a@example.com, b@example.com,",
            timezone="America/Los_Angeles",
        )
    )
    assert result["ok"] is True
    assert result["event"]["id"] == "new-ev"
    req = _last(server)
    assert req["method"] == "POST"
    assert urlparse(req["path"]).path == "/calendars/primary/events"
    assert req["body"] == {
        "summary": "Design review",
        "start": {"dateTime": "2025-06-03T14:00:00", "timeZone": "America/Los_Angeles"},
        "end": {"dateTime": "2025-06-03T15:00:00", "timeZone": "America/Los_Angeles"},
        "description": "Quarterly design review",
        "location": "HQ",
        "attendees": [{"email": "a@example.com"}, {"email": "b@example.com"}],
    }


def test_create_all_day_event_minimal(backend) -> None:
    """A start/end without 'T' becomes an all-day date, optional fields omitted."""
    b, server = backend
    result = json.loads(b.gcal_create_event("Offsite", "2025-06-05", "2025-06-06"))
    assert result["ok"] is True
    assert _last(server)["body"] == {
        "summary": "Offsite",
        "start": {"date": "2025-06-05"},
        "end": {"date": "2025-06-06"},
    }


def test_update_event_patches_only_supplied_fields(backend) -> None:
    """gcal_update_event PATCHes only the supplied fields."""
    b, server = backend
    result = json.loads(b.gcal_update_event("ev1", summary="New title", end="2025-06-07"))
    assert result["ok"] is True
    assert result["event"]["id"] == "ev1"
    req = _last(server)
    assert req["method"] == "PATCH"
    assert urlparse(req["path"]).path == "/calendars/primary/events/ev1"
    assert req["body"] == {"summary": "New title", "end": {"date": "2025-06-07"}}


def test_update_event_other_fields_with_timezone(backend) -> None:
    """gcal_update_event covers description/location/start with timezone."""
    b, server = backend
    result = json.loads(
        b.gcal_update_event(
            "ev1",
            description="moved",
            location="Room 2",
            start="2025-06-07T09:00:00",
            timezone="UTC",
        )
    )
    assert result["ok"] is True
    assert _last(server)["body"] == {
        "description": "moved",
        "location": "Room 2",
        "start": {"dateTime": "2025-06-07T09:00:00", "timeZone": "UTC"},
    }


def test_delete_event_handles_204(backend) -> None:
    """gcal_delete_event treats the 204 empty response as success."""
    b, server = backend
    result = json.loads(b.gcal_delete_event("ev1"))
    assert result == {"ok": True}
    req = _last(server)
    assert req["method"] == "DELETE"
    assert urlparse(req["path"]).path == "/calendars/primary/events/ev1"
    assert req["authorization"] == f"Bearer {_TOKEN}"


def test_quick_add(backend) -> None:
    """gcal_quick_add posts to /events/quickAdd with the text query param."""
    b, server = backend
    result = json.loads(b.gcal_quick_add("Lunch with Ada Friday at noon"))
    assert result["ok"] is True
    assert result["event"]["summary"] == "Lunch with Ada Friday at noon"
    parsed = urlparse(_last(server)["path"])
    assert parsed.path == "/calendars/primary/events/quickAdd"
    assert parse_qs(parsed.query)["text"] == ["Lunch with Ada Friday at noon"]


def test_path_unsafe_ids_rejected_before_any_request(backend) -> None:
    """Path-unsafe calendar/event IDs are refused up front — no HTTP request."""
    b, server = backend
    for attempt in (
        b.gcal_list_events(calendar_id="../users/me"),
        b.gcal_get_event("../ev", "primary"),
        b.gcal_get_event("ev1", "a/b"),
        b.gcal_create_event("x", "2025-06-05", "2025-06-06", calendar_id="a\\b"),
        b.gcal_update_event("ev..1"),
        b.gcal_update_event("ev1", calendar_id="c/../d"),
        b.gcal_delete_event("ev/1"),
        b.gcal_delete_event("ev1", calendar_id="..\\x"),
        b.gcal_quick_add("hi", calendar_id="a/b"),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid" in result["error"]
    assert server.requests == []


def test_wrong_token_returns_ok_false(gcal_server) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    base_url, _ = gcal_server
    b = GoogleCalendarChannelBackend()
    b._base_url = base_url
    b._token = "wrong-token"
    for call in (
        b.gcal_list_calendars,
        b.gcal_list_events,
        lambda: b.gcal_get_event("ev1"),
        lambda: b.gcal_create_event("x", "2025-06-05", "2025-06-06"),
        lambda: b.gcal_update_event("ev1", summary="x"),
        lambda: b.gcal_delete_event("ev1"),
        lambda: b.gcal_quick_add("hi"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the server yields ok:false JSON, not an exception."""
    b, _ = backend
    result = json.loads(b.gcal_list_events(calendar_id="boom"))
    assert result["ok"] is False
    assert "500" in result["error"]


def test_request_wraps_non_json_success_body(backend) -> None:
    """A 200 with a non-JSON body is returned as plain text, not an error."""
    b, _ = backend
    assert json.loads(b._request("GET", "/plaintext")) == {"ok": True, "result": "pong"}


def test_main_without_args_prints_usage(monkeypatch, capsys) -> None:
    """main() with no CLI arguments prints the usage line and exits 1."""
    monkeypatch.setattr(sys, "argv", ["kiss-gcal"])
    with pytest.raises(SystemExit) as excinfo:
        gcal_mod.main()
    assert excinfo.value.code == 1
    assert "Usage: kiss-gcal" in capsys.readouterr().out


def test_connection_refused_returns_ok_false() -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = GoogleCalendarChannelBackend()
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._token = _TOKEN
    for call in (
        b.gcal_list_calendars,
        b.gcal_list_events,
        lambda: b.gcal_get_event("ev1"),
        lambda: b.gcal_create_event("x", "2025-06-05", "2025-06-06"),
        lambda: b.gcal_update_event("ev1", summary="x"),
        lambda: b.gcal_delete_event("ev1"),
        lambda: b.gcal_quick_add("hi"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert result["error"]

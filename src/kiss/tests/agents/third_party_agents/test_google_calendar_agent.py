# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Google Calendar channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Google Calendar v3 REST API, reached through a real local Composio
API emulator (``composio_test_utils``) whose proxy injects the bearer
token — no mocks or patches.  The Calendar server asserts the
``Authorization: Bearer`` header on every call, returns canned JSON,
and records every request for verification.

State is isolated per test: ``isolated_kiss_home`` points ``KISS_HOME`` at a
fresh temporary directory and the config paths resolve ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiss.agents.third_party_agents.gcal.gcal_sea import (
    _SERVICE,
    GoogleCalendarChannelBackend,
)
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    TOKEN,
    connect,
)
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingHandler,
    RecordingServer,
    serve_recording,
)

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


class _CalendarRequestHandler(RecordingHandler):
    """Emulates the Google Calendar v3 REST API and records requests."""

    def _handle(self) -> None:
        body = self._read_body()
        entry = self.record_request(body)
        if not self.bearer_ok(TOKEN):
            self._reply(401, json.dumps({"error": {"code": 401, "message": "Unauthorized"}}))
            return
        path, query = entry["path"], entry["query"]
        if "/calendars/boom/" in path:
            self._reply(500, json.dumps({"error": {"code": 500, "message": "boom"}}))
        elif self.command == "GET" and path == "/plaintext":
            payload = b"pong"
            self._reply(200, payload, "text/plain")
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

    do_GET = do_POST = do_PATCH = do_DELETE = _handle  # noqa: N815


pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


@pytest.fixture()
def gcal_server():
    """Start the emulated Calendar server on a free port; yield the recording server."""
    yield from serve_recording(_CalendarRequestHandler)


@pytest.fixture()
def backend(gcal_server, composio):
    """A connected backend pointed at the emulated Calendar server."""
    server = gcal_server
    base_url = server.base_url
    connect(composio, _SERVICE)
    b = GoogleCalendarChannelBackend()
    b._base_url = base_url
    return b, server


def _last(server: RecordingServer) -> dict[str, Any]:
    return server.requests[-1]


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
    assert req["path"] == "/users/me/calendarList"
    assert req["authorization"] == f"Bearer {TOKEN}"


def test_list_events_defaults(backend) -> None:
    """gcal_list_events sends singleEvents/orderBy and omits empty filters."""
    b, server = backend
    result = json.loads(b.gcal_list_events())
    assert result["ok"] is True
    assert result["events"] == [_CONDENSED_EVENT]
    assert _last(server)["path"] == "/calendars/primary/events"
    query = _last(server)["query"]
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
    assert _last(server)["path"] == "/calendars/team%40example.com/events"
    query = _last(server)["query"]
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
    query = _last(server)["query"]
    assert query["pageToken"] == ["tok-1"]
    # Without a page_token, none is forwarded and the last page (no
    # nextPageToken from the API) omits next_page_token from the result.
    result = json.loads(b.gcal_list_events())
    assert result["ok"] is True
    assert "next_page_token" not in result
    assert "pageToken" not in _last(server)["query"]


def test_get_event(backend) -> None:
    """gcal_get_event fetches one event and condenses it."""
    b, server = backend
    result = json.loads(b.gcal_get_event("ev1"))
    assert result["ok"] is True
    assert result["event"] == _CONDENSED_EVENT
    req = _last(server)
    assert req["method"] == "GET"
    assert req["path"] == "/calendars/primary/events/ev1"


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
    assert req["path"] == "/calendars/primary/events"
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
    assert req["path"] == "/calendars/primary/events/ev1"
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
    assert req["path"] == "/calendars/primary/events/ev1"
    assert req["authorization"] == f"Bearer {TOKEN}"


def test_quick_add(backend) -> None:
    """gcal_quick_add posts to /events/quickAdd with the text query param."""
    b, server = backend
    result = json.loads(b.gcal_quick_add("Lunch with Ada Friday at noon"))
    assert result["ok"] is True
    assert result["event"]["summary"] == "Lunch with Ada Friday at noon"
    assert _last(server)["path"] == "/calendars/primary/events/quickAdd"
    assert _last(server)["query"]["text"] == ["Lunch with Ada Friday at noon"]


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


def _call_every_tool(b: GoogleCalendarChannelBackend) -> list[str]:
    """Invoke every backend tool once with valid arguments; return the results."""
    return [
        b.gcal_list_calendars(),
        b.gcal_list_events(),
        b.gcal_get_event("ev1"),
        b.gcal_create_event("x", "2025-06-05", "2025-06-06"),
        b.gcal_update_event("ev1", summary="x"),
        b.gcal_delete_event("ev1"),
        b.gcal_quick_add("hi"),
    ]


def test_wrong_token_returns_ok_false(backend, composio) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    b, _ = backend
    composio.token = "wrong-token"
    for result_str in _call_every_tool(b):
        result = json.loads(result_str)
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


def test_connection_refused_returns_ok_false(backend, refusing_port: int) -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b, _ = backend
    b._base_url = f"http://127.0.0.1:{refusing_port}"
    for result_str in _call_every_tool(b):
        result = json.loads(result_str)
        assert result["ok"] is False
        assert result["error"]

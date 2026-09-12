# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Google Calendar Agent — channel agent for the Google Calendar REST API.

Provides authenticated access to Google Calendar via OAuth2 using plain
REST calls against ``https://www.googleapis.com/calendar/v3`` (endpoint
paths per https://developers.google.com/calendar/api/v3/reference).
Credentials are handled by the shared Google Workspace OAuth helpers
and persisted under
``~/.kiss/third_party_agents/google_calendar/token.json``.

The Calendar REST API has no inbound message stream, so this adapter is
outbound-only: ``main`` passes ``make_backend=None`` to ``channel_main``
so the ``--channel`` poll mode is disabled.

Usage::

    agent = GoogleCalendarAgent()
    agent.run(prompt_template="List my events for tomorrow")
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    fresh_access_token,
    load_google_credentials,
    make_google_auth_tools,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_SERVICE = "google_calendar"
_SCOPES = ["https://www.googleapis.com/auth/calendar"]

_EVENT_FIELDS = (
    "id",
    "summary",
    "start",
    "end",
    "location",
    "status",
    "organizer",
    "attendees",
    "htmlLink",
)


def _bad_segment(value: str, name: str) -> str | None:
    """Reject *value* if it cannot safely form a single URL path segment.

    Values containing a path separator or a ``..`` sequence could
    traverse out of the intended API endpoint, so they are refused up
    front (defense in depth on top of ``quote(value, safe="")``).

    Args:
        value: Caller-supplied identifier destined for a URL path segment.
        name: Parameter name used in the error message.

    Returns:
        An ``{"ok": false, "error": ...}`` JSON string if *value* is
        unsafe, otherwise None.
    """
    if "/" in value or "\\" in value or ".." in value:
        return json.dumps(
            {"ok": False, "error": f"invalid {name}: must not contain path separators or '..'"}
        )
    return None


def _event_time(value: str, timezone: str) -> dict[str, str]:
    """Build a Calendar API event time object from a user-supplied string.

    Args:
        value: RFC3339 date-time (contains ``"T"``) or all-day date
            (``YYYY-MM-DD``).
        timezone: Optional IANA time zone name attached as ``timeZone``.

    Returns:
        A ``{"dateTime": ...}`` or ``{"date": ...}`` dict, plus
        ``timeZone`` when *timezone* is non-empty.
    """
    time_obj = {"dateTime": value} if "T" in value else {"date": value}
    if timezone:
        time_obj["timeZone"] = timezone
    return time_obj


def _condense_event(event: dict[str, Any]) -> dict[str, Any]:
    """Reduce a Calendar API event resource to its useful fields.

    Args:
        event: Full event dict from the Calendar API.

    Returns:
        Dict with only the ``_EVENT_FIELDS`` keys that are present.
    """
    return {key: event[key] for key in _EVENT_FIELDS if key in event}


class GoogleCalendarChannelBackend(ToolMethodBackend):
    """Channel backend for the Google Calendar REST API.

    Talks to the Calendar v3 API over HTTP with an OAuth2 bearer token.
    Outbound-only: there is no inbound message stream over plain REST.
    """

    def __init__(self) -> None:
        self._creds: Any = None
        self._token: str = ""
        self._base_url: str = "https://www.googleapis.com/calendar/v3"
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load stored Google Calendar OAuth2 credentials from disk.

        Returns:
            True if valid credentials were loaded.
        """
        self._creds = load_google_credentials(_SERVICE, _SCOPES)
        if self._creds is None:
            self._connection_info = "No Google Calendar credentials found."
            return False
        self._connection_info = "Google Calendar credentials loaded."
        return True

    def _headers(self) -> dict[str, str]:
        """Return the Authorization header for an API request.

        Returns:
            Header dict with the bearer token (the direct test override
            ``_token`` wins over the stored credentials).
        """
        return {"Authorization": f"Bearer {self._token or fresh_access_token(self._creds)}"}

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Issue an authenticated Calendar API request.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, ``"PATCH"``,
                ``"DELETE"``).
            path: API path relative to the base URL, starting with ``/``.
            params: Optional query parameters.
            payload: Optional JSON body.

        Returns:
            JSON string ``{"ok": true, "result": ...}`` on success or
            ``{"ok": false, "error": ...}`` on an HTTP error status.
        """
        url = self._base_url.rstrip("/") + path
        resp = requests.request(
            method,
            url,
            headers=self._headers(),
            params=params,
            json=payload,
            timeout=_TIMEOUT,
        )
        if resp.status_code >= 400:
            return json.dumps({"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"})
        if resp.status_code == 204 or not resp.content:
            return json.dumps({"ok": True, "result": None})
        try:
            result: Any = resp.json()
        except ValueError:
            result = resp.text
        return json.dumps({"ok": True, "result": result})

    def _event_result(self, raw: str) -> str:
        """Condense a single-event API response to its useful fields.

        Args:
            raw: JSON string from :meth:`_request`.

        Returns:
            JSON string with ok status and the condensed ``event``.
        """
        parsed = json.loads(raw)
        if not parsed.get("ok"):
            return raw
        return json.dumps({"ok": True, "event": _condense_event(parsed["result"] or {})}, indent=2)

    def gcal_list_calendars(self) -> str:
        """List the calendars on the user's calendar list.

        Returns:
            JSON string with ok status and the calendars (id, summary,
            primary flag, access role).
        """
        try:
            parsed = json.loads(self._request("GET", "/users/me/calendarList"))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            calendars = [
                {
                    "id": cal.get("id", ""),
                    "summary": cal.get("summary", ""),
                    "primary": cal.get("primary", False),
                    "access_role": cal.get("accessRole", ""),
                }
                for cal in (parsed["result"] or {}).get("items", [])
            ]
            return json.dumps({"ok": True, "calendars": calendars}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gcal_list_events(
        self,
        calendar_id: str = "primary",
        time_min: str = "",
        time_max: str = "",
        query: str = "",
        max_results: int = 20,
        page_token: str = "",
    ) -> str:
        """List events on a calendar, expanded and ordered by start time.

        Args:
            calendar_id: Calendar ID (default ``"primary"``).
            time_min: Lower bound (exclusive) for an event's end time,
                RFC3339 (e.g. ``"2025-01-01T00:00:00Z"``). Optional.
            time_max: Upper bound (exclusive) for an event's start time,
                RFC3339. Optional.
            query: Free-text search over event fields. Optional.
            max_results: Maximum number of events to return (default 20).
            page_token: Page token from a previous response's
                ``next_page_token`` to fetch the next page. Optional.

        Returns:
            JSON string with ok status, the condensed events (id,
            summary, start, end, location, status, organizer, attendees,
            htmlLink), and ``next_page_token`` when more results exist.
        """
        try:
            err = _bad_segment(calendar_id, "calendar_id")
            if err:
                return err
            params = {
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": str(max_results),
            }
            if time_min:
                params["timeMin"] = time_min
            if time_max:
                params["timeMax"] = time_max
            if query:
                params["q"] = query
            if page_token:
                params["pageToken"] = page_token
            path = f"/calendars/{quote(calendar_id, safe='')}/events"
            parsed = json.loads(self._request("GET", path, params=params))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            result = parsed["result"] or {}
            events = [_condense_event(e) for e in result.get("items", [])]
            out: dict[str, Any] = {"ok": True, "events": events}
            if result.get("nextPageToken"):
                out["next_page_token"] = result["nextPageToken"]
            return json.dumps(out, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gcal_get_event(self, event_id: str, calendar_id: str = "primary") -> str:
        """Get a single event by ID.

        Args:
            event_id: Event ID (from gcal_list_events).
            calendar_id: Calendar ID (default ``"primary"``).

        Returns:
            JSON string with ok status and the condensed event.
        """
        try:
            err = _bad_segment(event_id, "event_id") or _bad_segment(calendar_id, "calendar_id")
            if err:
                return err
            path = (
                f"/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}"
            )
            return self._event_result(self._request("GET", path))
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gcal_create_event(
        self,
        summary: str,
        start: str,
        end: str,
        calendar_id: str = "primary",
        description: str = "",
        location: str = "",
        attendees: str = "",
        timezone: str = "",
    ) -> str:
        """Create a calendar event.

        Args:
            summary: Event title.
            start: Start time — RFC3339 date-time (contains ``"T"``,
                e.g. ``"2025-06-01T10:00:00-07:00"``) or all-day date
                (``"2025-06-01"``).
            end: End time, same formats as *start*.
            calendar_id: Calendar ID (default ``"primary"``).
            description: Event description. Optional.
            location: Event location. Optional.
            attendees: Comma-separated attendee email addresses. Optional.
            timezone: IANA time zone (e.g. ``"America/Los_Angeles"``)
                attached to start and end. Optional.

        Returns:
            JSON string with ok status and the condensed created event.
        """
        try:
            err = _bad_segment(calendar_id, "calendar_id")
            if err:
                return err
            body: dict[str, Any] = {
                "summary": summary,
                "start": _event_time(start, timezone),
                "end": _event_time(end, timezone),
            }
            if description:
                body["description"] = description
            if location:
                body["location"] = location
            if attendees:
                body["attendees"] = [
                    {"email": email.strip()} for email in attendees.split(",") if email.strip()
                ]
            path = f"/calendars/{quote(calendar_id, safe='')}/events"
            return self._event_result(self._request("POST", path, payload=body))
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gcal_update_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        summary: str = "",
        description: str = "",
        location: str = "",
        start: str = "",
        end: str = "",
        timezone: str = "",
    ) -> str:
        """Update an event, changing only the supplied fields (PATCH semantics).

        Args:
            event_id: Event ID to update.
            calendar_id: Calendar ID (default ``"primary"``).
            summary: New title. Optional.
            description: New description. Optional.
            location: New location. Optional.
            start: New start time — RFC3339 date-time or all-day date.
                Optional.
            end: New end time, same formats. Optional.
            timezone: IANA time zone attached to a supplied start/end.
                Optional.

        Returns:
            JSON string with ok status and the condensed updated event.
        """
        try:
            err = _bad_segment(event_id, "event_id") or _bad_segment(calendar_id, "calendar_id")
            if err:
                return err
            body: dict[str, Any] = {}
            if summary:
                body["summary"] = summary
            if description:
                body["description"] = description
            if location:
                body["location"] = location
            if start:
                body["start"] = _event_time(start, timezone)
            if end:
                body["end"] = _event_time(end, timezone)
            path = (
                f"/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}"
            )
            return self._event_result(self._request("PATCH", path, payload=body))
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gcal_delete_event(self, event_id: str, calendar_id: str = "primary") -> str:
        """Delete an event.

        Args:
            event_id: Event ID to delete.
            calendar_id: Calendar ID (default ``"primary"``).

        Returns:
            JSON string with ok status.
        """
        try:
            err = _bad_segment(event_id, "event_id") or _bad_segment(calendar_id, "calendar_id")
            if err:
                return err
            path = (
                f"/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}"
            )
            parsed = json.loads(self._request("DELETE", path))
            if not parsed.get("ok"):
                return json.dumps(parsed)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gcal_quick_add(self, text: str, calendar_id: str = "primary") -> str:
        """Create an event from a natural-language text string.

        Args:
            text: Description such as ``"Lunch with Ada Friday at noon"``.
            calendar_id: Calendar ID (default ``"primary"``).

        Returns:
            JSON string with ok status and the condensed created event.
        """
        try:
            err = _bad_segment(calendar_id, "calendar_id")
            if err:
                return err
            path = f"/calendars/{quote(calendar_id, safe='')}/events/quickAdd"
            return self._event_result(self._request("POST", path, params={"text": text}))
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class GoogleCalendarAgent(BaseChannelAgent):
    """Channel agent with Google Calendar REST API tools."""

    channel_system_prompt = (
        "\n\n## Google Calendar Authentication\n"
        "If credentials.json is missing, call start_google_calendar_browser_setup() to "
        "open Google Cloud Console, then use browser tools to create OAuth credentials "
        "autonomously. If credentials.json exists, call authenticate_google_calendar() "
        "directly. Use ask_user_question() if you need user help with Google account "
        "login screens. Do NOT instruct the user to do these steps manually. "
        "Do all the steps on the user's behalf and ask the user's help ONLY if you are "
        "stuck on login or captcha."
    )

    def __init__(self) -> None:
        super().__init__("Google Calendar Agent")
        self._backend = GoogleCalendarChannelBackend()
        self._backend._creds = load_google_credentials(_SERVICE, _SCOPES)

    def _is_authenticated(self) -> bool:
        """Return True if the backend has credentials or a direct token."""
        return bool(self._backend._creds is not None or self._backend._token)

    def _get_auth_tools(self) -> list:
        """Return the standard Google OAuth tool quartet for Calendar."""
        backend = self._backend

        def on_credentials(creds: Any) -> None:
            """Wire new (or cleared) OAuth credentials into the backend.

            Args:
                creds: New credentials, or None after clearing.
            """
            backend._creds = creds

        return make_google_auth_tools(
            self, _SERVICE, "Google Calendar", _SCOPES, on_credentials=on_credentials
        )


def main() -> None:
    """Run the GoogleCalendarAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): the Calendar REST API
    has no inbound message stream to poll.
    """
    channel_main(
        GoogleCalendarAgent,
        "kiss-gcal",
        channel_name="Google Calendar",
        make_backend=None,
    )


def tools() -> list:
    """Return the Google Calendar channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return GoogleCalendarAgent()._get_tools()


if __name__ == "__main__":
    main()

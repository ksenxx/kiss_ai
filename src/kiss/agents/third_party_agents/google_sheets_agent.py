# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Google Sheets Agent — channel agent for the Google Sheets REST API.

Provides authenticated access to Google Sheets via OAuth2 (see
``_google_workspace_utils`` for the shared credential flow; the token
is persisted under ``$KISS_HOME/third_party_agents/google_sheets/``).
Spreadsheets are read and edited through plain ``requests`` calls
against ``https://sheets.googleapis.com/v4`` and listed through the
Drive v3 ``files`` endpoint.

The Sheets REST API has no inbound message stream, so this adapter is
outbound-only: ``poll_messages`` always returns no messages and poll
mode is disabled (``main`` passes ``make_backend=None``).

Usage::

    agent = GoogleSheetsAgent()
    agent.run(prompt_template="Create a budget spreadsheet with a Summary sheet")
"""

from __future__ import annotations

import json
import threading
from typing import Any
from urllib.parse import quote

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ToolMethodBackend,
    channel_main,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    fresh_access_token,
    google_api_session,
    load_google_credentials,
    make_google_auth_tools,
)

_SERVICE = "google_sheets"
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
_TIMEOUT = 30


def _bad_segment(value: str, name: str) -> str | None:
    """Reject *value* if it cannot safely form a single URL path segment.

    Values containing a path separator or a ``..`` sequence could
    traverse out of the intended API endpoint, so they are refused up
    front (defense in depth on top of ``quote(value, safe="")``).
    A1 ranges legitimately contain ``!``, ``:``, and quotes — those are
    allowed and percent-encoded.

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


def _escape_drive_query(query: str) -> str:
    """Escape a value for embedding in a Drive ``q`` single-quoted string.

    Args:
        query: Raw user-supplied search text.

    Returns:
        The text with backslashes and single quotes backslash-escaped.
    """
    return query.replace("\\", "\\\\").replace("'", "\\'")


def _parse_values_json(values_json: str) -> tuple[list[list[Any]] | None, str]:
    """Parse and validate a JSON 2D array of cell values.

    Args:
        values_json: JSON string that must parse to a list of lists.

    Returns:
        ``(values, "")`` on success or ``(None, error_message)`` when
        the string is not valid JSON or not a list of lists.
    """
    try:
        values = json.loads(values_json)
    except ValueError as e:
        return None, f"values_json is not valid JSON: {e}"
    if not isinstance(values, list) or not all(isinstance(row, list) for row in values):
        return None, "values_json must be a JSON array of arrays (2D array of cell values)"
    return values, ""


class GoogleSheetsChannelBackend(ToolMethodBackend):
    """Channel backend for the Google Sheets REST API.

    Talks to ``sheets.googleapis.com`` (and Drive v3 for listing) over
    HTTP with an OAuth2 bearer token.  Outbound-only: there is no
    inbound message stream, so :meth:`poll_messages` always returns no
    messages.
    """

    def __init__(self) -> None:
        self._creds: Any = None
        self._http: Any = google_api_session(_SERVICE)
        self._token: str = ""
        self._base_url: str = "https://sheets.googleapis.com/v4"
        self._drive_base_url: str = "https://www.googleapis.com/drive/v3"
        self._request_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load stored Google Sheets OAuth2 credentials from disk.

        Returns:
            True if valid credentials were loaded (or a token was
            already injected), False otherwise.
        """
        creds = load_google_credentials(_SERVICE, _SCOPES)
        if creds is not None:
            self._creds = creds
        if self._creds is None and not self._token:
            self._connection_info = (
                "No Google Sheets credentials found. Please authenticate first."
            )
            return False
        self._connection_info = "Google Sheets credentials loaded."
        return True

    def _headers(self) -> dict[str, str]:
        """Build the Authorization header from the injected token or credentials.

        Returns:
            Headers dict with a Bearer access token.
        """
        return {"Authorization": f"Bearer {self._token or fresh_access_token(self._creds)}"}

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue an authenticated Google API request.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, or ``"PUT"``).
            url: Full request URL.
            params: Optional query parameters.
            payload: Optional JSON body.

        Returns:
            ``{"ok": True, "result": <parsed JSON>}`` on success or
            ``{"ok": False, "error": ...}`` on an HTTP error status.
        """
        with self._request_lock:
            resp = self._http.request(
                method, url, headers=self._headers(), params=params, json=payload,
                timeout=_TIMEOUT,
            )
        if resp.status_code >= 400:
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
        try:
            result: Any = resp.json()
        except ValueError:
            result = resp.text
        return {"ok": True, "result": result}

    def _values_url(self, spreadsheet_id: str, range_a1: str, suffix: str = "") -> str:
        """Build a ``spreadsheets.values`` endpoint URL.

        Args:
            spreadsheet_id: The spreadsheet ID (already path-checked).
            range_a1: The A1 range (already path-checked); URL-quoted here.
            suffix: Optional method suffix such as ``":append"``.

        Returns:
            The full request URL.
        """
        return (
            f"{self._base_url}/spreadsheets/{quote(spreadsheet_id, safe='')}"
            f"/values/{quote(range_a1, safe='')}{suffix}"
        )

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Return no messages: the Sheets REST API has no inbound message stream.

        Args:
            channel_id: Ignored.
            oldest: Cursor, returned unchanged.
            limit: Ignored.

        Returns:
            ``([], oldest)``.
        """
        return [], oldest

    def gsheets_create_spreadsheet(self, title: str, sheet_titles: str = "") -> str:
        """Create a new Google Sheets spreadsheet.

        Args:
            title: Title of the new spreadsheet.
            sheet_titles: Optional comma-separated titles for the
                initial sheets (tabs). Empty creates the default single
                sheet.

        Returns:
            JSON string with ok status, spreadsheet_id, url, and sheet names.
        """
        try:
            body: dict[str, Any] = {"properties": {"title": title}}
            titles = [t.strip() for t in sheet_titles.split(",") if t.strip()]
            if titles:
                body["sheets"] = [{"properties": {"title": t}} for t in titles]
            resp = self._request("POST", f"{self._base_url}/spreadsheets", payload=body)
            if not resp["ok"]:
                return json.dumps(resp)
            sheet = resp["result"]
            return json.dumps(
                {
                    "ok": True,
                    "spreadsheet_id": sheet.get("spreadsheetId", ""),
                    "url": sheet.get("spreadsheetUrl", ""),
                    "sheets": [
                        s.get("properties", {}).get("title", "")
                        for s in sheet.get("sheets", [])
                    ],
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_get_info(self, spreadsheet_id: str) -> str:
        """Get a spreadsheet's title and its sheets (names, IDs, grid sizes).

        Args:
            spreadsheet_id: The ID of the spreadsheet to inspect.

        Returns:
            JSON string with ok status, title, and per-sheet properties.
        """
        try:
            err = _bad_segment(spreadsheet_id, "spreadsheet_id")
            if err:
                return err
            resp = self._request(
                "GET",
                f"{self._base_url}/spreadsheets/{quote(spreadsheet_id, safe='')}",
                params={"fields": "spreadsheetId,properties.title,sheets.properties"},
            )
            if not resp["ok"]:
                return json.dumps(resp)
            sheet = resp["result"]
            sheets = []
            for entry in sheet.get("sheets", []):
                props = entry.get("properties", {})
                grid = props.get("gridProperties", {})
                sheets.append(
                    {
                        "sheet_id": props.get("sheetId", 0),
                        "title": props.get("title", ""),
                        "rows": grid.get("rowCount", 0),
                        "columns": grid.get("columnCount", 0),
                    }
                )
            return json.dumps(
                {
                    "ok": True,
                    "spreadsheet_id": sheet.get("spreadsheetId", ""),
                    "title": sheet.get("properties", {}).get("title", ""),
                    "sheets": sheets,
                }
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_get_values(self, spreadsheet_id: str, range_a1: str) -> str:
        """Read cell values from a spreadsheet range.

        Args:
            spreadsheet_id: The ID of the spreadsheet to read.
            range_a1: The A1 range to read, e.g. ``"Sheet1!A1:C10"``.

        Returns:
            JSON string with ok status, the resolved range, and the 2D
            array of values.
        """
        try:
            err = _bad_segment(spreadsheet_id, "spreadsheet_id") or _bad_segment(
                range_a1, "range_a1"
            )
            if err:
                return err
            resp = self._request("GET", self._values_url(spreadsheet_id, range_a1))
            if not resp["ok"]:
                return json.dumps(resp)
            result = resp["result"]
            return json.dumps(
                {
                    "ok": True,
                    "range": result.get("range", ""),
                    "values": result.get("values", []),
                }
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_update_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        values_json: str,
        value_input_option: str = "USER_ENTERED",
    ) -> str:
        """Write cell values into a spreadsheet range (overwriting).

        Args:
            spreadsheet_id: The ID of the spreadsheet to update.
            range_a1: The A1 range to write, e.g. ``"Sheet1!A1:B2"``.
            values_json: JSON 2D array string of cell values, e.g.
                ``'[["Name", "Age"], ["Ada", 36]]'``.
            value_input_option: ``"USER_ENTERED"`` (parse as if typed,
                default) or ``"RAW"`` (store verbatim).

        Returns:
            JSON string with ok status and the update counts.
        """
        try:
            err = _bad_segment(spreadsheet_id, "spreadsheet_id") or _bad_segment(
                range_a1, "range_a1"
            )
            if err:
                return err
            values, parse_err = _parse_values_json(values_json)
            if values is None:
                return json.dumps({"ok": False, "error": parse_err})
            resp = self._request(
                "PUT",
                self._values_url(spreadsheet_id, range_a1),
                params={"valueInputOption": value_input_option},
                payload={"values": values},
            )
            if not resp["ok"]:
                return json.dumps(resp)
            result = resp["result"]
            return json.dumps(
                {
                    "ok": True,
                    "updated_range": result.get("updatedRange", ""),
                    "updated_cells": result.get("updatedCells", 0),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_append_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        values_json: str,
        value_input_option: str = "USER_ENTERED",
    ) -> str:
        """Append rows of values after the last row of a table.

        Args:
            spreadsheet_id: The ID of the spreadsheet to update.
            range_a1: The A1 range locating the table to append to,
                e.g. ``"Sheet1!A1"``.
            values_json: JSON 2D array string of cell values, e.g.
                ``'[["Ada", 36]]'``.
            value_input_option: ``"USER_ENTERED"`` (parse as if typed,
                default) or ``"RAW"`` (store verbatim).

        Returns:
            JSON string with ok status and the appended range.
        """
        try:
            err = _bad_segment(spreadsheet_id, "spreadsheet_id") or _bad_segment(
                range_a1, "range_a1"
            )
            if err:
                return err
            values, parse_err = _parse_values_json(values_json)
            if values is None:
                return json.dumps({"ok": False, "error": parse_err})
            resp = self._request(
                "POST",
                self._values_url(spreadsheet_id, range_a1, ":append"),
                params={"valueInputOption": value_input_option},
                payload={"values": values},
            )
            if not resp["ok"]:
                return json.dumps(resp)
            updates = resp["result"].get("updates", {})
            return json.dumps(
                {
                    "ok": True,
                    "updated_range": updates.get("updatedRange", ""),
                    "updated_cells": updates.get("updatedCells", 0),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_clear_values(self, spreadsheet_id: str, range_a1: str) -> str:
        """Clear cell values in a range (formatting is kept).

        Args:
            spreadsheet_id: The ID of the spreadsheet to update.
            range_a1: The A1 range to clear, e.g. ``"Sheet1!A1:C10"``.

        Returns:
            JSON string with ok status and the cleared range.
        """
        try:
            err = _bad_segment(spreadsheet_id, "spreadsheet_id") or _bad_segment(
                range_a1, "range_a1"
            )
            if err:
                return err
            resp = self._request(
                "POST", self._values_url(spreadsheet_id, range_a1, ":clear")
            )
            if not resp["ok"]:
                return json.dumps(resp)
            return json.dumps(
                {"ok": True, "cleared_range": resp["result"].get("clearedRange", "")}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_add_sheet(self, spreadsheet_id: str, title: str) -> str:
        """Add a new sheet (tab) to an existing spreadsheet.

        Args:
            spreadsheet_id: The ID of the spreadsheet to update.
            title: Title of the new sheet.

        Returns:
            JSON string with ok status and the new sheet's ID and title.
        """
        try:
            err = _bad_segment(spreadsheet_id, "spreadsheet_id")
            if err:
                return err
            url = f"{self._base_url}/spreadsheets/{quote(spreadsheet_id, safe='')}:batchUpdate"
            resp = self._request(
                "POST",
                url,
                payload={"requests": [{"addSheet": {"properties": {"title": title}}}]},
            )
            if not resp["ok"]:
                return json.dumps(resp)
            replies = resp["result"].get("replies", [])
            props = {}
            if replies:
                props = replies[0].get("addSheet", {}).get("properties", {})
            return json.dumps(
                {
                    "ok": True,
                    "sheet_id": props.get("sheetId", 0),
                    "title": props.get("title", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_batch_update(self, spreadsheet_id: str, requests_json: str) -> str:
        """Apply a raw list of Sheets API batchUpdate requests to a spreadsheet.

        Use for operations without a dedicated tool (formatting,
        deleting sheets, resizing, charts, ...). See the Sheets API
        ``Request`` reference for the available request objects.

        Args:
            spreadsheet_id: The ID of the spreadsheet to update.
            requests_json: JSON array string of Sheets ``Request``
                objects, e.g.
                ``'[{"deleteSheet": {"sheetId": 123}}]'``.

        Returns:
            JSON string with ok status and the batchUpdate replies.
        """
        try:
            err = _bad_segment(spreadsheet_id, "spreadsheet_id")
            if err:
                return err
            try:
                requests_list = json.loads(requests_json)
            except ValueError as e:
                return json.dumps({"ok": False, "error": f"requests_json is not valid JSON: {e}"})
            if not isinstance(requests_list, list):
                return json.dumps({"ok": False, "error": "requests_json must be a JSON array"})
            url = f"{self._base_url}/spreadsheets/{quote(spreadsheet_id, safe='')}:batchUpdate"
            resp = self._request("POST", url, payload={"requests": requests_list})
            if not resp["ok"]:
                return json.dumps(resp)
            return json.dumps(
                {"ok": True, "replies": resp["result"].get("replies", [])}
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gsheets_list_spreadsheets(self, query: str = "", max_results: int = 20) -> str:
        """List Google Sheets in the user's Drive, optionally filtered by name.

        Args:
            query: Optional substring to match against spreadsheet names.
            max_results: Maximum number of spreadsheets to return. Default: 20.

        Returns:
            JSON string with ok status and files (id, name,
            modifiedTime, webViewLink).
        """
        try:
            q = "mimeType='application/vnd.google-apps.spreadsheet'"
            if query:
                q += f" and name contains '{_escape_drive_query(query)}'"
            resp = self._request(
                "GET",
                f"{self._drive_base_url}/files",
                params={
                    "q": q,
                    "pageSize": max_results,
                    "fields": "files(id,name,modifiedTime,webViewLink)",
                },
            )
            if not resp["ok"]:
                return json.dumps(resp)
            return json.dumps(
                {"ok": True, "files": resp["result"].get("files", [])}
            )[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class GoogleSheetsAgent(BaseChannelAgent):
    """Channel agent with Google Sheets REST API tools."""

    channel_system_prompt = (
        "\n\n## Google Sheets Authentication\n"
        "If credentials.json is missing, call start_google_sheets_browser_setup() to open "
        "Google Cloud Console, then use browser tools to create OAuth credentials "
        "autonomously. If credentials.json exists, call authenticate_google_sheets() "
        "directly. Use ask_user_question() if you need user help with Google account "
        "login screens. Do NOT instruct the user to do these steps manually. "
        "Do all the steps on the user's behalf and ask the user's help ONLY if you "
        "are stuck on login or captcha."
    )

    def __init__(self) -> None:
        super().__init__("Google Sheets Agent")
        self._backend = GoogleSheetsChannelBackend()
        self._backend._creds = load_google_credentials(_SERVICE, _SCOPES)

    def _is_authenticated(self) -> bool:
        """Return True if the backend has credentials or an injected token."""
        return bool(self._backend._creds is not None or self._backend._token)

    def _get_auth_tools(self) -> list:
        """Return the Google Sheets authentication tool functions."""
        backend = self._backend

        def on_credentials(creds: Any) -> None:
            """Wire new (or cleared) OAuth credentials into the backend.

            Args:
                creds: New credentials, or None after clearing.
            """
            backend._creds = creds

        return make_google_auth_tools(self, _SERVICE, "Google Sheets", _SCOPES, on_credentials)


def main() -> None:
    """Run the GoogleSheetsAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): the Sheets REST API
    has no inbound message stream to poll.
    """
    channel_main(
        GoogleSheetsAgent,
        "kiss-gsheets",
        channel_name="Google Sheets",
        make_backend=None,
    )


def tools() -> list:
    """Return the Google Sheets channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return GoogleSheetsAgent()._get_tools()


if __name__ == "__main__":
    main()

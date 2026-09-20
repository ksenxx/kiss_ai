# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Google Sheets channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Google Sheets v4 and Drive v3 REST endpoints — no mocks, patches,
or fakes.  The server asserts the ``Authorization: Bearer`` header on
every call, returns canned JSON, and records requests (method, path,
query, body) for verification.

Credential state is isolated per pytest process because the session
conftest points ``KISS_HOME`` at a temporary directory and every
``_google_workspace_utils`` path helper resolves ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

import kiss.agents.third_party_agents.gsheets_sea as gsheets_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    clear_google_credentials,
    fresh_access_token,
    token_path,
)
from kiss.agents.third_party_agents.gsheets_sea import (
    _SCOPES,
    _SERVICE,
    GoogleSheetsAgent,
    GoogleSheetsChannelBackend,
)

_TOKEN = "test-token"
_SYNTHETIC_TOKEN = "synthetic-access-token"

_AUTH_TOOL_NAMES = [
    "check_google_sheets_auth",
    "authenticate_google_sheets",
    "clear_google_sheets_auth",
    "start_google_sheets_browser_setup",
    "finish_google_sheets_auth",
]

_BACKEND_TOOL_NAMES = {
    "gsheets_create_spreadsheet",
    "gsheets_get_info",
    "gsheets_get_values",
    "gsheets_update_values",
    "gsheets_append_values",
    "gsheets_clear_values",
    "gsheets_add_sheet",
    "gsheets_batch_update",
    "gsheets_list_spreadsheets",
}

_SPREADSHEET_INFO = {
    "spreadsheetId": "ss1",
    "properties": {"title": "Budget"},
    "sheets": [
        {
            "properties": {
                "sheetId": 0,
                "title": "Sheet1",
                "gridProperties": {"rowCount": 1000, "columnCount": 26},
            }
        },
        {
            "properties": {
                "sheetId": 42,
                "title": "Summary",
                "gridProperties": {"rowCount": 50, "columnCount": 10},
            }
        },
    ],
}

_DRIVE_FILES = [
    {
        "id": "s1",
        "name": "Budget 2026",
        "modifiedTime": "2026-01-01T00:00:00Z",
        "webViewLink": "https://docs.google.com/spreadsheets/d/s1",
    }
]


class _SheetsRequestHandler(BaseHTTPRequestHandler):
    """Emulates the Google Sheets v4 + Drive v3 REST APIs and records requests."""

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {_TOKEN}"

    def _reply(self, status: int, body: str, content_type: str = "application/json") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, body: Any) -> None:
        parsed = urlparse(self.path)
        cast(_SheetsServer, self.server).requests.append(
            {
                "method": self.command,
                "path": parsed.path,
                "query": parse_qs(parsed.query),
                "authorization": self.headers.get("Authorization", ""),
                "body": body,
            }
        )

    def _read_body(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._record(None)
        if not self._authorized():
            self._reply(401, json.dumps({"error": {"message": "Unauthorized"}}))
            return
        path = urlparse(self.path).path
        if path == "/spreadsheets/ss1":
            self._reply(200, json.dumps(_SPREADSHEET_INFO))
        elif path == "/spreadsheets/fail500":
            self._reply(500, json.dumps({"error": {"message": "Internal error"}}))
        elif path.startswith("/spreadsheets/ss1/values/"):
            self._reply(
                200,
                json.dumps({"range": "Sheet1!A1:B2", "values": [["a", "b"], ["c", "d"]]}),
            )
        elif path == "/drive/v3/files":
            self._reply(200, json.dumps({"files": _DRIVE_FILES}))
        elif path == "/plain":
            self._reply(200, "just text", content_type="text/plain")
        else:
            self._reply(404, json.dumps({"error": {"message": "Not found"}}))

    def do_PUT(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        body = self._read_body()
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"error": {"message": "Unauthorized"}}))
            return
        path = urlparse(self.path).path
        if path.startswith("/spreadsheets/ss1/values/"):
            self._reply(200, json.dumps({"updatedRange": "Sheet1!A1:B2", "updatedCells": 4}))
        else:
            self._reply(404, json.dumps({"error": {"message": "Not found"}}))

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        body = self._read_body()
        self._record(body)
        if not self._authorized():
            self._reply(401, json.dumps({"error": {"message": "Unauthorized"}}))
            return
        path = urlparse(self.path).path
        if path == "/spreadsheets":
            sheets = (body or {}).get(
                "sheets", [{"properties": {"title": "Sheet1"}}]
            )
            self._reply(
                200,
                json.dumps(
                    {
                        "spreadsheetId": "newss",
                        "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/newss",
                        "sheets": sheets,
                    }
                ),
            )
        elif path.startswith("/spreadsheets/ss1/values/") and path.endswith(":append"):
            self._reply(
                200,
                json.dumps(
                    {
                        "spreadsheetId": "ss1",
                        "tableRange": "Sheet1!A1:B2",
                        "updates": {"updatedRange": "Sheet1!A3:B3", "updatedCells": 2},
                    }
                ),
            )
        elif path.startswith("/spreadsheets/ss1/values/") and path.endswith(":clear"):
            self._reply(
                200,
                json.dumps({"spreadsheetId": "ss1", "clearedRange": "Sheet1!A1:C10"}),
            )
        elif path == "/spreadsheets/ss1:batchUpdate":
            replies = []
            for request in (body or {}).get("requests", []):
                if "addSheet" in request:
                    title = request["addSheet"].get("properties", {}).get("title", "")
                    replies.append(
                        {"addSheet": {"properties": {"sheetId": 77, "title": title}}}
                    )
                else:
                    replies.append({})
            self._reply(200, json.dumps({"spreadsheetId": "ss1", "replies": replies}))
        elif path == "/spreadsheets/emptyreply:batchUpdate":
            self._reply(200, json.dumps({"spreadsheetId": "emptyreply"}))
        elif path == "/spreadsheets/fail500:batchUpdate":
            self._reply(500, json.dumps({"error": {"message": "Internal error"}}))
        else:
            self._reply(404, json.dumps({"error": {"message": "Not found"}}))

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _SheetsServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _SheetsRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture()
def sheets_server():
    """Start the emulated Sheets/Drive server on a free port; yield (base_url, server)."""
    server = _SheetsServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(sheets_server):
    """A backend pointed at the emulated server with a directly injected token."""
    base_url, server = sheets_server
    b = GoogleSheetsChannelBackend()
    b._token = _TOKEN
    b._base_url = base_url
    b._drive_base_url = f"{base_url}/drive/v3"
    return b, server


@pytest.fixture(autouse=True)
def _fresh_credentials():
    """Start and end every test with no persisted Google Sheets token."""
    clear_google_credentials(_SERVICE)
    yield
    clear_google_credentials(_SERVICE)


def _write_synthetic_token() -> None:
    """Persist a synthetic, non-expiring OAuth2 user token for the service."""
    path = token_path(_SERVICE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "token": _SYNTHETIC_TOKEN,
                "refresh_token": "synthetic-refresh",
                "client_id": "synthetic-client-id",
                "client_secret": "synthetic-client-secret",
                "scopes": _SCOPES,
                "expiry": "2099-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def test_unauthenticated_agent_exposes_only_auth_tools() -> None:
    """A fresh agent is unauthenticated and exposes exactly the auth tool set."""
    agent = GoogleSheetsAgent()
    assert agent.name == "Google Sheets Agent"
    assert agent._is_authenticated() is False
    assert [t.__name__ for t in agent._get_tools()] == _AUTH_TOOL_NAMES


def test_synthetic_token_authenticates_new_agent() -> None:
    """A synthetic token.json makes a NEW agent authenticated without network."""
    _write_synthetic_token()
    agent = GoogleSheetsAgent()
    assert agent._is_authenticated() is True
    creds = agent._backend._creds
    assert creds is not None
    assert creds.valid is True
    assert fresh_access_token(creds) == _SYNTHETIC_TOKEN
    names = {t.__name__ for t in agent._get_tools()}
    assert set(_AUTH_TOOL_NAMES) <= names
    assert _BACKEND_TOOL_NAMES <= names
    assert "poll_messages" not in names  # channel protocol method, not an LLM tool
    assert "connect" not in names


def test_clear_auth_removes_token_and_relocks() -> None:
    """clear_google_sheets_auth deletes the token file and re-locks the agent."""
    _write_synthetic_token()
    agent = GoogleSheetsAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["clear_google_sheets_auth"]()
    assert "cleared" in result.lower()
    assert not token_path(_SERVICE).exists()
    assert agent._is_authenticated() is False
    assert len(agent._get_tools()) == len(_AUTH_TOOL_NAMES)


def test_tools_module_function() -> None:
    """Module-level tools() returns the auth tools of a fresh agent."""
    tools = gsheets_mod.tools()
    assert [t.__name__ for t in tools] == _AUTH_TOOL_NAMES
    assert all(callable(t) for t in tools)


def test_connect_without_credentials_fails() -> None:
    """connect() fails cleanly when no token is persisted or injected."""
    b = GoogleSheetsChannelBackend()
    assert b.connect() is False
    assert "No Google Sheets credentials" in b.connection_info


def test_connect_with_persisted_token_succeeds() -> None:
    """connect() loads a persisted synthetic token into the backend."""
    _write_synthetic_token()
    b = GoogleSheetsChannelBackend()
    assert b.connect() is True
    assert b._creds is not None
    assert "credentials loaded" in b.connection_info.lower()


def test_connect_with_injected_token_succeeds() -> None:
    """connect() accepts a directly injected bearer token without a token file."""
    b = GoogleSheetsChannelBackend()
    b._token = _TOKEN
    assert b.connect() is True
    assert b._creds is None


def test_poll_messages_returns_empty(backend) -> None:
    """poll_messages returns no messages: the Sheets REST API has no inbound stream."""
    b, server = backend
    messages, cursor = b.poll_messages("anything", "42", limit=5)
    assert messages == []
    assert cursor == "42"
    assert server.requests == []


def test_create_spreadsheet_default_sheet(backend) -> None:
    """gsheets_create_spreadsheet with no sheet_titles posts only the title."""
    b, server = backend
    result = json.loads(b.gsheets_create_spreadsheet("Budget"))
    assert result == {
        "ok": True,
        "spreadsheet_id": "newss",
        "url": "https://docs.google.com/spreadsheets/d/newss",
        "sheets": ["Sheet1"],
    }
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/spreadsheets")
    assert req["authorization"] == f"Bearer {_TOKEN}"
    assert req["body"] == {"properties": {"title": "Budget"}}


def test_create_spreadsheet_with_sheet_titles(backend) -> None:
    """Comma-separated sheet_titles become initial sheets in the request body."""
    b, server = backend
    result = json.loads(b.gsheets_create_spreadsheet("Budget", "Income, Expenses"))
    assert result["ok"] is True
    assert result["sheets"] == ["Income", "Expenses"]
    req = server.requests[-1]
    assert req["body"] == {
        "properties": {"title": "Budget"},
        "sheets": [
            {"properties": {"title": "Income"}},
            {"properties": {"title": "Expenses"}},
        ],
    }


def test_get_info(backend) -> None:
    """gsheets_get_info returns sheet names, IDs, and grid sizes via a fields mask."""
    b, server = backend
    result = json.loads(b.gsheets_get_info("ss1"))
    assert result == {
        "ok": True,
        "spreadsheet_id": "ss1",
        "title": "Budget",
        "sheets": [
            {"sheet_id": 0, "title": "Sheet1", "rows": 1000, "columns": 26},
            {"sheet_id": 42, "title": "Summary", "rows": 50, "columns": 10},
        ],
    }
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/spreadsheets/ss1")
    assert req["query"]["fields"] == ["spreadsheetId,properties.title,sheets.properties"]


def test_get_values(backend) -> None:
    """gsheets_get_values GETs the URL-encoded range and returns the 2D array."""
    b, server = backend
    result = json.loads(b.gsheets_get_values("ss1", "Sheet1!A1:B2"))
    assert result == {
        "ok": True,
        "range": "Sheet1!A1:B2",
        "values": [["a", "b"], ["c", "d"]],
    }
    req = server.requests[-1]
    assert (req["method"], req["path"]) == (
        "GET",
        "/spreadsheets/ss1/values/Sheet1%21A1%3AB2",
    )
    assert req["authorization"] == f"Bearer {_TOKEN}"


def test_get_values_quoted_sheet_name_encoded(backend) -> None:
    """Quoted sheet names with spaces are percent-encoded in the path."""
    b, server = backend
    result = json.loads(b.gsheets_get_values("ss1", "'My Sheet'!A1"))
    assert result["ok"] is True
    assert server.requests[-1]["path"] == (
        "/spreadsheets/ss1/values/%27My%20Sheet%27%21A1"
    )


def test_update_values(backend) -> None:
    """gsheets_update_values PUTs the parsed 2D array with valueInputOption."""
    b, server = backend
    result = json.loads(
        b.gsheets_update_values("ss1", "Sheet1!A1:B2", '[["x", 1], ["y", 2]]')
    )
    assert result == {"ok": True, "updated_range": "Sheet1!A1:B2", "updated_cells": 4}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == (
        "PUT",
        "/spreadsheets/ss1/values/Sheet1%21A1%3AB2",
    )
    assert req["query"]["valueInputOption"] == ["USER_ENTERED"]
    assert req["body"] == {"values": [["x", 1], ["y", 2]]}


def test_update_values_raw_option(backend) -> None:
    """A non-default value_input_option is forwarded as the query parameter."""
    b, server = backend
    result = json.loads(
        b.gsheets_update_values("ss1", "A1", '[["=SUM(1,2)"]]', value_input_option="RAW")
    )
    assert result["ok"] is True
    assert server.requests[-1]["query"]["valueInputOption"] == ["RAW"]


def test_append_values(backend) -> None:
    """gsheets_append_values POSTs to the :append endpoint and condenses updates."""
    b, server = backend
    result = json.loads(b.gsheets_append_values("ss1", "Sheet1!A1", '[["z", 3]]'))
    assert result == {"ok": True, "updated_range": "Sheet1!A3:B3", "updated_cells": 2}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == (
        "POST",
        "/spreadsheets/ss1/values/Sheet1%21A1:append",
    )
    assert req["query"]["valueInputOption"] == ["USER_ENTERED"]
    assert req["body"] == {"values": [["z", 3]]}


def test_clear_values(backend) -> None:
    """gsheets_clear_values POSTs to the :clear endpoint with no body."""
    b, server = backend
    result = json.loads(b.gsheets_clear_values("ss1", "Sheet1!A1:C10"))
    assert result == {"ok": True, "cleared_range": "Sheet1!A1:C10"}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == (
        "POST",
        "/spreadsheets/ss1/values/Sheet1%21A1%3AC10:clear",
    )


def test_add_sheet(backend) -> None:
    """gsheets_add_sheet posts an addSheet batchUpdate and returns the new sheet."""
    b, server = backend
    result = json.loads(b.gsheets_add_sheet("ss1", "Q3"))
    assert result == {"ok": True, "sheet_id": 77, "title": "Q3"}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/spreadsheets/ss1:batchUpdate")
    assert req["body"] == {"requests": [{"addSheet": {"properties": {"title": "Q3"}}}]}


def test_add_sheet_without_replies(backend) -> None:
    """A batchUpdate response with no replies yields empty sheet properties."""
    b, _ = backend
    result = json.loads(b.gsheets_add_sheet("emptyreply", "Q3"))
    assert result == {"ok": True, "sheet_id": 0, "title": ""}


def test_batch_update(backend) -> None:
    """gsheets_batch_update forwards a caller-supplied request list verbatim."""
    b, server = backend
    requests_json = '[{"deleteSheet": {"sheetId": 42}}]'
    result = json.loads(b.gsheets_batch_update("ss1", requests_json))
    assert result == {"ok": True, "replies": [{}]}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("POST", "/spreadsheets/ss1:batchUpdate")
    assert req["body"] == {"requests": json.loads(requests_json)}


def test_batch_update_rejects_bad_json(backend) -> None:
    """Malformed or non-array requests_json is refused without any request."""
    b, server = backend
    result = json.loads(b.gsheets_batch_update("ss1", "{not json"))
    assert result["ok"] is False
    assert "not valid JSON" in result["error"]
    result = json.loads(b.gsheets_batch_update("ss1", '{"addSheet": {}}'))
    assert result == {"ok": False, "error": "requests_json must be a JSON array"}
    assert server.requests == []


def test_values_json_validation(backend) -> None:
    """Bad values_json (invalid JSON, non-list, non-2D) is refused up front."""
    b, server = backend
    for bad in ("{not json", '{"a": 1}', "[1, 2]", '["x", ["y"]]'):
        result = json.loads(b.gsheets_update_values("ss1", "A1", bad))
        assert result["ok"] is False
        assert "values_json" in result["error"]
        result = json.loads(b.gsheets_append_values("ss1", "A1", bad))
        assert result["ok"] is False
        assert "values_json" in result["error"]
    assert server.requests == []


def test_list_spreadsheets(backend) -> None:
    """gsheets_list_spreadsheets queries Drive for Sheets files with a fields mask."""
    b, server = backend
    result = json.loads(b.gsheets_list_spreadsheets())
    assert result == {"ok": True, "files": _DRIVE_FILES}
    req = server.requests[-1]
    assert (req["method"], req["path"]) == ("GET", "/drive/v3/files")
    assert req["query"]["q"] == ["mimeType='application/vnd.google-apps.spreadsheet'"]
    assert req["query"]["pageSize"] == ["20"]
    assert req["query"]["fields"] == ["files(id,name,modifiedTime,webViewLink)"]


def test_list_spreadsheets_escapes_query(backend) -> None:
    """Single quotes in the name filter are backslash-escaped in the Drive q."""
    b, server = backend
    result = json.loads(b.gsheets_list_spreadsheets("Bob's books", max_results=3))
    assert result["ok"] is True
    req = server.requests[-1]
    assert req["query"]["q"] == [
        "mimeType='application/vnd.google-apps.spreadsheet'"
        " and name contains 'Bob\\'s books'"
    ]
    assert req["query"]["pageSize"] == ["3"]


def test_path_unsafe_params_rejected(backend) -> None:
    """Path traversal attempts in ids and ranges are refused — no HTTP request."""
    b, server = backend
    for attempt in (
        b.gsheets_get_info("../ss"),
        b.gsheets_get_info("a/b"),
        b.gsheets_get_info("a\\b"),
        b.gsheets_get_values("../ss", "A1"),
        b.gsheets_get_values("ss1", "A1/../B2"),
        b.gsheets_update_values("ss1", "A1\\B2", "[[1]]"),
        b.gsheets_append_values("..", "A1", "[[1]]"),
        b.gsheets_clear_values("ss1", "../A1"),
        b.gsheets_add_sheet("../ss", "T"),
        b.gsheets_batch_update("a/b", "[]"),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid" in result["error"]
    assert server.requests == []


def test_unauthorized_token_returns_ok_false(sheets_server) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    base_url, _ = sheets_server
    b = GoogleSheetsChannelBackend()
    b._token = "wrong-token"
    b._base_url = base_url
    b._drive_base_url = f"{base_url}/drive/v3"
    for call in (
        lambda: b.gsheets_create_spreadsheet("t"),
        lambda: b.gsheets_get_info("ss1"),
        lambda: b.gsheets_get_values("ss1", "A1"),
        lambda: b.gsheets_update_values("ss1", "A1", "[[1]]"),
        lambda: b.gsheets_append_values("ss1", "A1", "[[1]]"),
        lambda: b.gsheets_clear_values("ss1", "A1"),
        lambda: b.gsheets_add_sheet("ss1", "T"),
        lambda: b.gsheets_batch_update("ss1", "[]"),
        lambda: b.gsheets_list_spreadsheets(),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend) -> None:
    """A 500 from the server yields ok:false JSON, not an exception."""
    b, _ = backend
    result = json.loads(b.gsheets_get_info("fail500"))
    assert result["ok"] is False
    assert "500" in result["error"]
    result = json.loads(b.gsheets_add_sheet("fail500", "T"))
    assert result["ok"] is False
    assert "500" in result["error"]


def test_connection_refused_returns_ok_false() -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = GoogleSheetsChannelBackend()
    b._token = _TOKEN
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._drive_base_url = "http://127.0.0.1:9/drive/v3"
    for call in (
        lambda: b.gsheets_create_spreadsheet("t"),
        lambda: b.gsheets_get_info("ss1"),
        lambda: b.gsheets_get_values("ss1", "A1"),
        lambda: b.gsheets_update_values("ss1", "A1", "[[1]]"),
        lambda: b.gsheets_append_values("ss1", "A1", "[[1]]"),
        lambda: b.gsheets_clear_values("ss1", "A1"),
        lambda: b.gsheets_add_sheet("ss1", "T"),
        lambda: b.gsheets_batch_update("ss1", "[]"),
        lambda: b.gsheets_list_spreadsheets(),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert result["error"]


def test_request_returns_plain_text_body(backend) -> None:
    """_request falls back to the raw text body when the response is not JSON."""
    b, _ = backend
    result = b._request("GET", f"{b._base_url}/plain")
    assert result == {"ok": True, "result": "just text"}


def test_main_exits_with_no_args() -> None:
    """main() prints usage and exits when called with no arguments."""
    original_argv = sys.argv
    sys.argv = ["kiss-gsheets"]
    try:
        with pytest.raises(SystemExit):
            gsheets_mod.main()
    finally:
        sys.argv = original_argv

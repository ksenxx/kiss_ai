# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Google Drive channel agent.

Runs a REAL local HTTP server (stdlib ``ThreadedHTTPServer``) emulating
the Google Drive v3 REST API — no mocks, patches, or fakes.  The server
asserts the ``Authorization: Bearer`` header on every call, serves both
file metadata and file content (``alt=media`` and ``/export``), accepts
multipart uploads, and records every request for verification.

Token state is isolated because the session conftest points
``KISS_HOME`` at a temporary directory; an autouse fixture additionally
clears the google_drive token around every test.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

import kiss.agents.third_party_agents.gdrive_sea as gdrive_mod
from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents._google_workspace_utils import (
    clear_google_credentials,
    token_path,
)
from kiss.agents.third_party_agents.gdrive_sea import (
    _SCOPES,
    _SERVICE,
    GoogleDriveAgent,
    GoogleDriveChannelBackend,
)

_TOKEN = "test-token"

_FILES: dict[str, dict[str, Any]] = {
    "gdoc1": {
        "id": "gdoc1",
        "name": "Design Doc",
        "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "2025-06-01T00:00:00Z",
        "webViewLink": "https://docs.google.com/document/d/gdoc1",
        "parents": ["old1"],
    },
    "sheet1": {
        "id": "sheet1",
        "name": "Budget",
        "mimeType": "application/vnd.google-apps.spreadsheet",
        "parents": ["old1"],
    },
    "bin1": {
        "id": "bin1",
        "name": "notes.txt",
        "mimeType": "text/plain",
        "size": "19",
        "parents": ["old1", "old2"],
    },
    "orphan1": {
        "id": "orphan1",
        "name": "orphan.txt",
        "mimeType": "text/plain",
    },
    "cfail": {
        "id": "cfail",
        "name": "cursed.txt",
        "mimeType": "text/plain",
    },
    "pboom": {
        "id": "pboom",
        "name": "patchfail.txt",
        "mimeType": "text/plain",
        "parents": ["old1"],
    },
}

_BIN_CONTENT = b"hello plain content"

_AUTH_TOOL_NAMES = [
    "check_google_drive_auth",
    "authenticate_google_drive",
    "clear_google_drive_auth",
    "start_google_drive_browser_setup",
    "finish_google_drive_auth",
]

_TOOL_NAMES = {
    "gdrive_search_files",
    "gdrive_get_file",
    "gdrive_read_file",
    "gdrive_download_file",
    "gdrive_upload_file",
    "gdrive_create_folder",
    "gdrive_share_file",
    "gdrive_move_file",
    "gdrive_trash_file",
}


def write_synthetic_token() -> None:
    """Persist a synthetic, never-expiring OAuth token for google_drive."""
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


class _DriveRequestHandler(BaseHTTPRequestHandler):
    """Emulates the Google Drive v3 REST API and records requests."""

    def _reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reply_json(self, status: int, obj: Any) -> None:
        self._reply(status, json.dumps(obj).encode("utf-8"))

    def _record(self, raw: bytes) -> None:
        body: Any = None
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                body = raw
        cast(_DriveServer, self.server).requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "content_type": self.headers.get("Content-Type", ""),
                "body": body,
            }
        )

    def _handle(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._record(raw)
        if self.headers.get("Authorization") != f"Bearer {_TOKEN}":
            self._reply_json(401, {"error": {"code": 401, "message": "Unauthorized"}})
            return
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        if path == "/files/boom" or path.startswith("/files/boom/"):
            self._reply_json(500, {"error": {"code": 500, "message": "boom"}})
        elif self.command == "PATCH" and path == "/files/pboom":
            self._reply_json(500, {"error": {"code": 500, "message": "patch boom"}})
        elif self.command == "GET" and path == "/plaintext":
            self._reply(200, b"pong", content_type="text/plain")
        elif self.command == "GET" and path == "/files":
            if query.get("q") == ["name = 'none'"]:
                self._reply_json(200, {"files": []})
            else:
                self._reply_json(
                    200, {"files": list(_FILES.values()), "nextPageToken": "tok-next"}
                )
        elif self.command == "GET" and path.endswith("/export"):
            file_id = path.split("/")[-2]
            mime = query.get("mimeType", [""])[0]
            self._reply(200, f"exported {file_id} as {mime}".encode(), content_type=mime)
        elif self.command == "GET" and path.startswith("/files/"):
            file_id = path.split("/")[2]
            if file_id not in _FILES:
                self._reply_json(404, {"error": {"code": 404, "message": "Not found"}})
            elif query.get("alt") == ["media"]:
                if file_id == "cfail":
                    self._reply_json(500, {"error": {"code": 500, "message": "content boom"}})
                else:
                    self._reply(200, _BIN_CONTENT, content_type="text/plain")
            else:
                self._reply_json(200, _FILES[file_id])
        elif self.command == "POST" and path == "/upload/files":
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/related"):
                self._reply_json(400, {"error": {"code": 400, "message": "not multipart/related"}})
                return
            self._reply_json(200, {"id": "up1", "name": "uploaded"})
        elif self.command == "POST" and path.endswith("/permissions"):
            self._reply_json(200, {"id": "perm1"} | json.loads(raw.decode("utf-8")))
        elif self.command == "POST" and path == "/files":
            self._reply_json(200, {"id": "folder-new"} | json.loads(raw.decode("utf-8")))
        elif self.command == "PATCH" and path.startswith("/files/"):
            file_id = path.split("/")[2]
            body = json.loads(raw.decode("utf-8")) if raw else {}
            self._reply_json(200, _FILES.get(file_id, {"id": file_id}) | body)
        else:
            self._reply_json(404, {"error": {"code": 404, "message": "Not found"}})

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._handle()

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._handle()

    def do_PATCH(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        self._handle()

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _DriveServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _DriveRequestHandler)
        self.requests: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _fresh_token():
    """Start and end every test with no persisted google_drive token."""
    clear_google_credentials(_SERVICE)
    yield
    clear_google_credentials(_SERVICE)


@pytest.fixture()
def drive_server():
    """Start the emulated Drive server on a free port; yield (base_url, server)."""
    server = _DriveServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


@pytest.fixture()
def backend(drive_server):
    """A backend pointed at the emulated server with the valid token."""
    base_url, server = drive_server
    b = GoogleDriveChannelBackend()
    b._base_url = base_url
    b._upload_base_url = base_url + "/upload"
    b._token = _TOKEN
    return b, server


def _last(server: _DriveServer) -> dict[str, Any]:
    return server.requests[-1]


def test_agent_unauthenticated_exposes_only_auth_tools() -> None:
    """A fresh agent is unauthenticated and exposes exactly the 4 auth tools."""
    agent = GoogleDriveAgent()
    assert agent.name == "Google Drive Agent"
    assert agent._is_authenticated() is False
    assert [t.__name__ for t in agent._get_tools()] == _AUTH_TOOL_NAMES


def test_check_auth_unauthenticated_explains_setup() -> None:
    """check_google_drive_auth explains how to set up credentials."""
    agent = GoogleDriveAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_google_drive_auth"]()
    assert "Not authenticated with Google Drive" in msg
    assert "start_google_drive_browser_setup" in msg
    assert "authenticate_google_drive" in msg


def test_synthetic_token_authenticates_new_agent() -> None:
    """A synthetic token.json makes a new agent authenticated end-to-end."""
    write_synthetic_token()
    agent = GoogleDriveAgent()
    assert agent._is_authenticated() is True
    names = {t.__name__ for t in agent._get_tools()}
    assert set(_AUTH_TOOL_NAMES) <= names
    assert _TOOL_NAMES <= names
    assert "connect" not in names  # channel protocol method, not an LLM tool
    tools = {t.__name__: t for t in agent._get_tools()}
    assert json.loads(tools["check_google_drive_auth"]())["ok"] is True


def test_clear_auth_removes_token_and_relocks_tools() -> None:
    """clear_google_drive_auth deletes the token and re-locks backend tools."""
    write_synthetic_token()
    agent = GoogleDriveAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["clear_google_drive_auth"]()
    assert "cleared" in result.lower()
    assert not token_path(_SERVICE).exists()
    assert agent._is_authenticated() is False
    assert [t.__name__ for t in agent._get_tools()] == _AUTH_TOOL_NAMES


def test_tools_module_function() -> None:
    """Module-level tools() returns the auth tool set when locked."""
    tools = gdrive_mod.tools()
    assert [t.__name__ for t in tools] == _AUTH_TOOL_NAMES
    assert all(callable(t) for t in tools)


def test_connect_without_token_fails() -> None:
    """connect() fails cleanly when no token is persisted."""
    b = GoogleDriveChannelBackend()
    assert b.connect() is False
    assert "No Google Drive credentials" in b.connection_info


def test_connect_with_synthetic_token_succeeds() -> None:
    """connect() loads persisted synthetic credentials into the backend."""
    write_synthetic_token()
    b = GoogleDriveChannelBackend()
    assert b.connect() is True
    assert b._creds is not None
    assert b._headers() == {"Authorization": "Bearer synthetic-access-token"}


def test_search_files_defaults(backend) -> None:
    """gdrive_search_files lists files with the condensed fields projection."""
    b, server = backend
    result = json.loads(b.gdrive_search_files())
    assert result["ok"] is True
    assert result["files"] == list(_FILES.values())
    assert result["next_page_token"] == "tok-next"
    req = _last(server)
    assert req["method"] == "GET"
    parsed = urlparse(req["path"])
    assert parsed.path == "/files"
    query = parse_qs(parsed.query)
    assert query["pageSize"] == ["20"]
    assert query["fields"] == [
        "files(id,name,mimeType,size,modifiedTime,webViewLink,parents),nextPageToken"
    ]
    assert "q" not in query
    assert "pageToken" not in query
    assert "orderBy" not in query
    assert req["authorization"] == f"Bearer {_TOKEN}"


def test_search_files_with_query_and_paging(backend) -> None:
    """gdrive_search_files forwards q, pageToken, orderBy, and pageSize."""
    b, server = backend
    result = json.loads(
        b.gdrive_search_files(
            query="name contains 'report'",
            max_results=5,
            page_token="tok-1",
            order_by="modifiedTime desc",
        )
    )
    assert result["ok"] is True
    query = parse_qs(urlparse(_last(server)["path"]).query)
    assert query["q"] == ["name contains 'report'"]
    assert query["pageSize"] == ["5"]
    assert query["pageToken"] == ["tok-1"]
    assert query["orderBy"] == ["modifiedTime desc"]


def test_get_file(backend) -> None:
    """gdrive_get_file fetches metadata with the fields projection."""
    b, server = backend
    result = json.loads(b.gdrive_get_file("bin1"))
    assert result["ok"] is True
    assert result["file"] == _FILES["bin1"]
    parsed = urlparse(_last(server)["path"])
    assert parsed.path == "/files/bin1"
    assert parse_qs(parsed.query)["fields"] == [
        "id,name,mimeType,size,modifiedTime,webViewLink,parents"
    ]


def test_read_google_doc_uses_export(backend) -> None:
    """Reading a Google Workspace doc goes through /export with text/plain."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("gdoc1"))
    assert result["ok"] is True
    assert result["name"] == "Design Doc"
    assert result["mime_type"] == "application/vnd.google-apps.document"
    assert result["content"] == "exported gdoc1 as text/plain"
    parsed = urlparse(_last(server)["path"])
    assert parsed.path == "/files/gdoc1/export"
    assert parse_qs(parsed.query)["mimeType"] == ["text/plain"]


def test_read_spreadsheet_defaults_to_csv_export(backend) -> None:
    """Reading a Google Sheet defaults the export MIME type to text/csv."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("sheet1"))
    assert result["ok"] is True
    assert result["content"] == "exported sheet1 as text/csv"
    assert parse_qs(urlparse(_last(server)["path"]).query)["mimeType"] == ["text/csv"]


def test_read_with_export_mime_type_override(backend) -> None:
    """An explicit export_mime_type overrides the default export format."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("gdoc1", export_mime_type="text/markdown"))
    assert result["ok"] is True
    assert result["content"] == "exported gdoc1 as text/markdown"
    assert parse_qs(urlparse(_last(server)["path"]).query)["mimeType"] == ["text/markdown"]


def test_read_binary_file_uses_alt_media(backend) -> None:
    """Reading a regular file downloads it with alt=media, not /export."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("bin1"))
    assert result["ok"] is True
    assert result["content"] == _BIN_CONTENT.decode()
    parsed = urlparse(_last(server)["path"])
    assert parsed.path == "/files/bin1"
    assert parse_qs(parsed.query)["alt"] == ["media"]
    # The preceding metadata request used the fields projection, not alt=media.
    meta_req = server.requests[-2]
    assert parse_qs(urlparse(meta_req["path"]).query)["fields"] == ["id,name,mimeType"]


def test_download_file_writes_bytes(backend, tmp_path) -> None:
    """gdrive_download_file writes the fetched bytes to save_path."""
    b, _ = backend
    target = tmp_path / "sub" / "notes.txt"
    result = json.loads(b.gdrive_download_file("bin1", str(target)))
    assert result == {"ok": True, "path": str(target), "size": len(_BIN_CONTENT)}
    assert target.read_bytes() == _BIN_CONTENT


def test_download_google_doc_exports(backend, tmp_path) -> None:
    """gdrive_download_file exports Google Workspace docs before saving."""
    b, _ = backend
    target = tmp_path / "doc.txt"
    result = json.loads(b.gdrive_download_file("gdoc1", str(target)))
    assert result["ok"] is True
    assert target.read_text() == "exported gdoc1 as text/plain"


def test_upload_file_multipart(backend, tmp_path) -> None:
    """gdrive_upload_file builds a multipart/related body the server accepts."""
    b, server = backend
    source = tmp_path / "report.txt"
    source.write_bytes(b"report body bytes")
    result = json.loads(
        b.gdrive_upload_file(str(source), name="Q2 Report.txt", folder_id="folder9")
    )
    assert result["ok"] is True
    assert result["file"] == {"id": "up1", "name": "uploaded"}
    req = _last(server)
    assert req["method"] == "POST"
    parsed = urlparse(req["path"])
    assert parsed.path == "/upload/files"
    assert parse_qs(parsed.query)["uploadType"] == ["multipart"]
    assert req["authorization"] == f"Bearer {_TOKEN}"
    content_type = req["content_type"]
    assert content_type.startswith("multipart/related; boundary=")
    boundary = content_type.split("boundary=", 1)[1]
    body = cast(bytes, req["body"])
    parts = body.split(f"--{boundary}".encode())
    assert body.rstrip().endswith(f"--{boundary}--".encode().rstrip())
    metadata_part, media_part = parts[1], parts[2]
    assert b"Content-Type: application/json; charset=UTF-8" in metadata_part
    metadata = json.loads(metadata_part.split(b"\r\n\r\n", 1)[1])
    assert metadata == {"name": "Q2 Report.txt", "parents": ["folder9"]}
    assert b"Content-Type: text/plain" in media_part
    assert b"report body bytes" in media_part


def test_upload_file_defaults_name_and_guessed_mime(backend, tmp_path) -> None:
    """Upload defaults to the local file name and a guessed MIME type."""
    b, server = backend
    source = tmp_path / "diagram.png"
    source.write_bytes(b"\x89PNG fake")
    result = json.loads(b.gdrive_upload_file(str(source)))
    assert result["ok"] is True
    body = cast(bytes, _last(server)["body"])
    metadata = json.loads(body.split(b"\r\n\r\n", 1)[1].split(b"\r\n", 1)[0])
    assert metadata == {"name": "diagram.png"}
    assert b"Content-Type: image/png" in body


def test_upload_file_explicit_mime_and_unknown_extension(backend, tmp_path) -> None:
    """An explicit mime_type wins; unknown extensions fall back to octet-stream."""
    b, server = backend
    source = tmp_path / "data.unknownext"
    source.write_bytes(b"x")
    assert json.loads(b.gdrive_upload_file(str(source), mime_type="application/x-custom"))["ok"]
    assert b"Content-Type: application/x-custom" in cast(bytes, _last(server)["body"])
    assert json.loads(b.gdrive_upload_file(str(source)))["ok"]
    assert b"Content-Type: application/octet-stream" in cast(bytes, _last(server)["body"])


def test_upload_file_over_5mb_rejected(backend, tmp_path) -> None:
    """A local file over 5 MB is refused before any HTTP request (multipart limit)."""
    b, server = backend
    big = tmp_path / "big.bin"
    with big.open("wb") as f:  # sparse/seek-created: 5 MB + 1 byte without real I/O
        f.seek(5 * 1024 * 1024)
        f.write(b"\0")
    assert big.stat().st_size == 5 * 1024 * 1024 + 1
    result = json.loads(b.gdrive_upload_file(str(big)))
    assert result == {"ok": False, "error": "file exceeds the 5 MB multipart upload limit"}
    assert server.requests == []


def test_upload_file_exactly_5mb_allowed(backend, tmp_path) -> None:
    """A file of exactly 5 MB is within the documented multipart limit."""
    b, server = backend
    exact = tmp_path / "exact.bin"
    with exact.open("wb") as f:
        f.seek(5 * 1024 * 1024 - 1)
        f.write(b"\0")
    assert exact.stat().st_size == 5 * 1024 * 1024
    result = json.loads(b.gdrive_upload_file(str(exact)))
    assert result["ok"] is True
    assert _last(server)["method"] == "POST"


def test_upload_missing_local_file(backend, tmp_path) -> None:
    """Uploading a nonexistent local file fails without any HTTP request."""
    b, server = backend
    result = json.loads(b.gdrive_upload_file(str(tmp_path / "missing.txt")))
    assert result["ok"] is False
    assert "not found" in result["error"]
    assert server.requests == []


def test_create_folder(backend) -> None:
    """gdrive_create_folder posts the folder MIME type and parent."""
    b, server = backend
    result = json.loads(b.gdrive_create_folder("Reports", parent_id="root9"))
    assert result["ok"] is True
    assert result["folder"]["id"] == "folder-new"
    req = _last(server)
    assert req["method"] == "POST"
    assert urlparse(req["path"]).path == "/files"
    assert req["body"] == {
        "name": "Reports",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": ["root9"],
    }


def test_create_folder_without_parent(backend) -> None:
    """gdrive_create_folder omits parents when parent_id is empty."""
    b, server = backend
    assert json.loads(b.gdrive_create_folder("Loose"))["ok"] is True
    assert _last(server)["body"] == {
        "name": "Loose",
        "mimeType": "application/vnd.google-apps.folder",
    }


def test_share_file(backend) -> None:
    """gdrive_share_file posts a user permission with role and email."""
    b, server = backend
    result = json.loads(b.gdrive_share_file("bin1", "ada@example.com", role="writer"))
    assert result["ok"] is True
    assert result["permission"]["id"] == "perm1"
    req = _last(server)
    assert req["method"] == "POST"
    assert urlparse(req["path"]).path == "/files/bin1/permissions"
    assert req["body"] == {"type": "user", "role": "writer", "emailAddress": "ada@example.com"}


def test_move_file_swaps_parents(backend) -> None:
    """gdrive_move_file fetches current parents then PATCHes the swap."""
    b, server = backend
    result = json.loads(b.gdrive_move_file("bin1", "newfolder"))
    assert result["ok"] is True
    get_req, patch_req = server.requests[-2], server.requests[-1]
    assert get_req["method"] == "GET"
    assert parse_qs(urlparse(get_req["path"]).query)["fields"] == ["parents"]
    assert patch_req["method"] == "PATCH"
    assert urlparse(patch_req["path"]).path == "/files/bin1"
    query = parse_qs(urlparse(patch_req["path"]).query)
    assert query["addParents"] == ["newfolder"]
    assert query["removeParents"] == ["old1,old2"]


def test_move_file_without_existing_parents(backend) -> None:
    """Moving a parentless file omits removeParents."""
    b, server = backend
    result = json.loads(b.gdrive_move_file("orphan1", "newfolder"))
    assert result["ok"] is True
    query = parse_qs(urlparse(_last(server)["path"]).query)
    assert query["addParents"] == ["newfolder"]
    assert "removeParents" not in query


def test_trash_file(backend) -> None:
    """gdrive_trash_file PATCHes {"trashed": true}."""
    b, server = backend
    result = json.loads(b.gdrive_trash_file("bin1"))
    assert result == {"ok": True}
    req = _last(server)
    assert req["method"] == "PATCH"
    assert urlparse(req["path"]).path == "/files/bin1"
    assert req["body"] == {"trashed": True}


def test_path_unsafe_ids_rejected_before_any_request(backend, tmp_path) -> None:
    """Path-unsafe file/folder IDs are refused up front — no HTTP request."""
    b, server = backend
    source = tmp_path / "f.txt"
    source.write_text("x")
    for attempt in (
        b.gdrive_get_file("../about"),
        b.gdrive_read_file("a/b"),
        b.gdrive_download_file("a\\b", str(tmp_path / "out")),
        b.gdrive_upload_file(str(source), folder_id="../root"),
        b.gdrive_create_folder("x", parent_id="a/b"),
        b.gdrive_share_file("fil..e", "a@example.com"),
        b.gdrive_move_file("../x", "f"),
        b.gdrive_move_file("bin1", "a/b"),
        b.gdrive_trash_file("bin1/.."),
    ):
        result = json.loads(attempt)
        assert result["ok"] is False
        assert "invalid" in result["error"]
    assert server.requests == []


def test_wrong_token_returns_ok_false(drive_server, tmp_path) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    base_url, _ = drive_server
    b = GoogleDriveChannelBackend()
    b._base_url = base_url
    b._upload_base_url = base_url + "/upload"
    b._token = "wrong-token"
    source = tmp_path / "f.txt"
    source.write_text("x")
    for call in (
        b.gdrive_search_files,
        lambda: b.gdrive_get_file("bin1"),
        lambda: b.gdrive_read_file("bin1"),
        lambda: b.gdrive_download_file("bin1", str(tmp_path / "out")),
        lambda: b.gdrive_upload_file(str(source)),
        lambda: b.gdrive_create_folder("x"),
        lambda: b.gdrive_share_file("bin1", "a@example.com"),
        lambda: b.gdrive_move_file("bin1", "f"),
        lambda: b.gdrive_trash_file("bin1"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "401" in result["error"]


def test_server_error_returns_ok_false(backend, tmp_path) -> None:
    """A 500 from the server yields ok:false JSON, not an exception."""
    b, _ = backend
    for call in (
        lambda: b.gdrive_get_file("boom"),
        lambda: b.gdrive_read_file("boom"),
        lambda: b.gdrive_download_file("boom", str(tmp_path / "out")),
        lambda: b.gdrive_move_file("boom", "f"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert "500" in result["error"]


def test_read_metadata_404_returns_ok_false(backend) -> None:
    """A metadata 404 fails before any content fetch is attempted."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("nosuchfile"))
    assert result["ok"] is False
    assert "404" in result["error"]
    assert len(server.requests) == 1  # failed at the metadata step, no content fetch


def test_read_content_error_after_good_metadata(backend) -> None:
    """A content fetch that 500s after good metadata surfaces ok:false."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("cfail"))
    assert result["ok"] is False
    assert "500" in result["error"]
    assert len(server.requests) == 2  # metadata succeeded, content fetch failed


def test_search_without_next_page_token(backend) -> None:
    """A last-page response omits next_page_token from the tool result."""
    b, _ = backend
    result = json.loads(b.gdrive_search_files(query="name = 'none'"))
    assert result == {"ok": True, "files": []}


def test_move_file_patch_failure_after_good_get(backend) -> None:
    """A PATCH failure after a successful parents GET surfaces ok:false."""
    b, server = backend
    result = json.loads(b.gdrive_move_file("pboom", "newfolder"))
    assert result["ok"] is False
    assert "500" in result["error"]
    assert server.requests[-2]["method"] == "GET"
    assert server.requests[-1]["method"] == "PATCH"


def test_request_wraps_non_json_success_body(backend) -> None:
    """A 200 with a non-JSON body is returned as plain text, not an error."""
    b, _ = backend
    assert json.loads(b._request("GET", "/plaintext")) == {"ok": True, "result": "pong"}


def test_main_without_args_prints_usage(monkeypatch, capsys) -> None:
    """main() with no CLI arguments prints the usage line and exits 1."""
    monkeypatch.setattr(sys, "argv", ["kiss-gdrive"])
    with pytest.raises(SystemExit) as excinfo:
        gdrive_mod.main()
    assert excinfo.value.code == 1
    assert "Usage: kiss-gdrive" in capsys.readouterr().out


def test_connection_refused_returns_ok_false(tmp_path) -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b = GoogleDriveChannelBackend()
    b._base_url = "http://127.0.0.1:9"  # discard port; nothing listens
    b._upload_base_url = "http://127.0.0.1:9/upload"
    b._token = _TOKEN
    source = tmp_path / "f.txt"
    source.write_text("x")
    for call in (
        b.gdrive_search_files,
        lambda: b.gdrive_get_file("bin1"),
        lambda: b.gdrive_read_file("bin1"),
        lambda: b.gdrive_download_file("bin1", str(tmp_path / "out")),
        lambda: b.gdrive_upload_file(str(source)),
        lambda: b.gdrive_create_folder("x"),
        lambda: b.gdrive_share_file("bin1", "a@example.com"),
        lambda: b.gdrive_move_file("bin1", "f"),
        lambda: b.gdrive_trash_file("bin1"),
    ):
        result = json.loads(call())
        assert result["ok"] is False
        assert result["error"]

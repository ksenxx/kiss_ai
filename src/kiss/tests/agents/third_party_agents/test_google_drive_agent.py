# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Google Drive channel agent.

Runs a REAL local HTTP server (``recording_http.RecordingServer``) emulating
the Google Drive v3 REST API, reached through a real local Composio API
emulator (``composio_test_utils``) whose proxy injects the bearer token
— no mocks or patches.  The Drive server asserts the
``Authorization: Bearer`` header on every call, serves both file
metadata and file content (``alt=media`` and ``/export``), and records
every request for verification.

State is isolated per test: ``isolated_kiss_home`` points ``KISS_HOME`` at a
fresh temporary directory and the config paths resolve ``$KISS_HOME`` lazily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.third_party_agents.gdrive.gdrive_sea import (
    _SERVICE,
    GoogleDriveChannelBackend,
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


class _DriveRequestHandler(RecordingHandler):
    """Emulates the Google Drive v3 REST API and records requests."""

    def _handle(self) -> None:
        raw = self._read_raw()
        body: Any = None
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                body = raw
        req = self.record_request(body, content_type=self.headers.get("Content-Type", ""))
        if not self.bearer_ok(TOKEN):
            self._json(401, {"error": {"code": 401, "message": "Unauthorized"}})
            return
        path, query = req["path"], req["query"]
        if path == "/files/boom" or path.startswith("/files/boom/"):
            self._json(500, {"error": {"code": 500, "message": "boom"}})
        elif self.command == "PATCH" and path == "/files/pboom":
            self._json(500, {"error": {"code": 500, "message": "patch boom"}})
        elif self.command == "GET" and path == "/plaintext":
            self._reply(200, b"pong", content_type="text/plain")
        elif self.command == "GET" and path == "/files":
            if query.get("q") == ["name = 'none'"]:
                self._json(200, {"files": []})
            else:
                self._json(200, {"files": list(_FILES.values()), "nextPageToken": "tok-next"})
        elif self.command == "GET" and path.endswith("/export"):
            file_id = path.split("/")[-2]
            mime = query.get("mimeType", [""])[0]
            self._reply(200, f"exported {file_id} as {mime}".encode(), content_type=mime)
        elif self.command == "GET" and path.startswith("/files/"):
            file_id = path.split("/")[2]
            if file_id not in _FILES:
                self._json(404, {"error": {"code": 404, "message": "Not found"}})
            elif query.get("alt") == ["media"]:
                if file_id == "cfail":
                    self._json(500, {"error": {"code": 500, "message": "content boom"}})
                else:
                    self._reply(200, _BIN_CONTENT, content_type="application/octet-stream")
            else:
                self._json(200, _FILES[file_id])
        elif self.command == "POST" and path.endswith("/permissions"):
            self._json(200, {"id": "perm1"} | json.loads(raw.decode("utf-8")))
        elif self.command == "POST" and path == "/upload/files":
            if b"multipart/related" not in self.headers.get("Content-Type", "").encode():
                self._json(400, {"error": {"code": 400, "message": "not multipart"}})
            else:
                self._json(200, {"id": "up1", "name": "uploaded"})
        elif self.command == "POST" and path == "/files":
            self._json(200, {"id": "folder-new"} | json.loads(raw.decode("utf-8")))
        elif self.command == "PATCH" and path.startswith("/files/"):
            file_id = path.split("/")[2]
            body = json.loads(raw.decode("utf-8")) if raw else {}
            self._json(200, _FILES.get(file_id, {"id": file_id}) | body)
        else:
            self._json(404, {"error": {"code": 404, "message": "Not found"}})

    do_GET = do_POST = do_PATCH = _handle  # noqa: N815


pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


@pytest.fixture()
def drive_server():
    """Start the emulated Drive server on a free port."""
    yield from serve_recording(_DriveRequestHandler)


@pytest.fixture()
def backend(drive_server, composio):
    """A connected backend pointed at the emulated Drive server."""
    connect(composio, _SERVICE)
    b = GoogleDriveChannelBackend()
    b._base_url = drive_server.base_url
    b._upload_base_url = drive_server.base_url + "/upload"
    return b, drive_server


def _last(server: RecordingServer) -> dict[str, Any]:
    return server.requests[-1]


def test_search_files_defaults(backend) -> None:
    """gdrive_search_files lists files with the condensed fields projection."""
    b, server = backend
    result = json.loads(b.gdrive_search_files())
    assert result["ok"] is True
    assert result["files"] == list(_FILES.values())
    assert result["next_page_token"] == "tok-next"
    req = _last(server)
    assert req["method"] == "GET"
    assert req["path"] == "/files"
    query = req["query"]
    assert query["pageSize"] == ["20"]
    assert query["fields"] == [
        "files(id,name,mimeType,size,modifiedTime,webViewLink,parents),nextPageToken"
    ]
    assert "q" not in query
    assert "pageToken" not in query
    assert "orderBy" not in query
    assert req["authorization"] == f"Bearer {TOKEN}"


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
    query = _last(server)["query"]
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
    req = _last(server)
    assert req["path"] == "/files/bin1"
    assert req["query"]["fields"] == ["id,name,mimeType,size,modifiedTime,webViewLink,parents"]


def test_read_google_doc_uses_export(backend) -> None:
    """Reading a Google Workspace doc goes through /export with text/plain."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("gdoc1"))
    assert result["ok"] is True
    assert result["name"] == "Design Doc"
    assert result["mime_type"] == "application/vnd.google-apps.document"
    assert result["content"] == "exported gdoc1 as text/plain"
    req = _last(server)
    assert req["path"] == "/files/gdoc1/export"
    assert req["query"]["mimeType"] == ["text/plain"]


def test_read_spreadsheet_defaults_to_csv_export(backend) -> None:
    """Reading a Google Sheet defaults the export MIME type to text/csv."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("sheet1"))
    assert result["ok"] is True
    assert result["content"] == "exported sheet1 as text/csv"
    assert _last(server)["query"]["mimeType"] == ["text/csv"]


def test_read_with_export_mime_type_override(backend) -> None:
    """An explicit export_mime_type overrides the default export format."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("gdoc1", export_mime_type="text/markdown"))
    assert result["ok"] is True
    assert result["content"] == "exported gdoc1 as text/markdown"
    assert _last(server)["query"]["mimeType"] == ["text/markdown"]


def test_read_binary_file_uses_alt_media(backend) -> None:
    """Reading a regular file downloads it with alt=media, not /export."""
    b, server = backend
    result = json.loads(b.gdrive_read_file("bin1"))
    assert result["ok"] is True
    assert result["content"] == _BIN_CONTENT.decode()
    req = _last(server)
    assert req["path"] == "/files/bin1"
    assert req["query"]["alt"] == ["media"]
    # The preceding metadata request used the fields projection, not alt=media.
    meta_req = server.requests[-2]
    assert meta_req["query"]["fields"] == ["id,name,mimeType"]


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
    """gdrive_upload_file sends an RFC 2387 multipart body as the proxy's binary body."""
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
    assert req["path"] == "/upload/files"
    assert req["query"]["uploadType"] == ["multipart"]
    content_type = req["content_type"]
    assert content_type.startswith("multipart/related; boundary=")
    boundary = content_type.split("boundary=", 1)[1]
    body = cast(bytes, req["body"])
    parts = body.split(f"--{boundary}".encode())
    metadata_part, media_part = parts[1], parts[2]
    metadata = json.loads(metadata_part.split(b"\r\n\r\n", 1)[1])
    assert metadata == {"name": "Q2 Report.txt", "parents": ["folder9"]}
    assert b"Content-Type: text/plain" in media_part and b"report body bytes" in media_part


def test_upload_file_defaults_name_and_guessed_mime(backend, tmp_path) -> None:
    """Upload defaults to the local file name and a guessed MIME type."""
    b, server = backend
    source = tmp_path / "diagram.png"
    source.write_bytes(b"\x89PNG fake")
    assert json.loads(b.gdrive_upload_file(str(source)))["ok"] is True
    body = cast(bytes, _last(server)["body"])
    metadata = json.loads(body.split(b"\r\n\r\n", 1)[1].split(b"\r\n", 1)[0])
    assert metadata == {"name": "diagram.png"}
    assert b"Content-Type: image/png" in body
    source = tmp_path / "data.unknownext"
    source.write_bytes(b"x")
    assert json.loads(b.gdrive_upload_file(str(source), mime_type="application/x-custom"))["ok"]
    assert b"Content-Type: application/x-custom" in cast(bytes, _last(server)["body"])
    assert json.loads(b.gdrive_upload_file(str(source)))["ok"]
    assert b"Content-Type: application/octet-stream" in cast(bytes, _last(server)["body"])


def test_upload_file_size_limit_and_missing_file(backend, tmp_path) -> None:
    """Over 5 MB is refused before any request; exactly 5 MB and errors are handled."""
    b, server = backend
    big = tmp_path / "big.bin"
    with big.open("wb") as f:
        f.seek(5 * 1024 * 1024)
        f.write(b"\0")
    result = json.loads(b.gdrive_upload_file(str(big)))
    assert result == {"ok": False, "error": "file exceeds the 5 MB multipart upload limit"}
    assert server.requests == []
    exact = tmp_path / "exact.bin"
    with exact.open("wb") as f:
        f.seek(5 * 1024 * 1024 - 1)
        f.write(b"\0")
    assert json.loads(b.gdrive_upload_file(str(exact)))["ok"] is True
    missing = json.loads(b.gdrive_upload_file(str(tmp_path / "nope.txt")))
    assert missing["ok"] is False and "local file not found" in missing["error"]
    assert json.loads(b.gdrive_upload_file(str(exact), folder_id="../x"))["ok"] is False
    b._upload_base_url = b._base_url + "/files/boom"
    assert json.loads(b.gdrive_upload_file(str(exact)))["ok"] is False


def test_create_folder(backend) -> None:
    """gdrive_create_folder posts the folder MIME type and parent."""
    b, server = backend
    result = json.loads(b.gdrive_create_folder("Reports", parent_id="root9"))
    assert result["ok"] is True
    assert result["folder"]["id"] == "folder-new"
    req = _last(server)
    assert req["method"] == "POST"
    assert req["path"] == "/files"
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
    assert req["path"] == "/files/bin1/permissions"
    assert req["body"] == {"type": "user", "role": "writer", "emailAddress": "ada@example.com"}


def test_move_file_swaps_parents(backend) -> None:
    """gdrive_move_file fetches current parents then PATCHes the swap."""
    b, server = backend
    result = json.loads(b.gdrive_move_file("bin1", "newfolder"))
    assert result["ok"] is True
    get_req, patch_req = server.requests[-2], server.requests[-1]
    assert get_req["method"] == "GET"
    assert get_req["query"]["fields"] == ["parents"]
    assert patch_req["method"] == "PATCH"
    assert patch_req["path"] == "/files/bin1"
    query = patch_req["query"]
    assert query["addParents"] == ["newfolder"]
    assert query["removeParents"] == ["old1,old2"]


def test_move_file_without_existing_parents(backend) -> None:
    """Moving a parentless file omits removeParents."""
    b, server = backend
    result = json.loads(b.gdrive_move_file("orphan1", "newfolder"))
    assert result["ok"] is True
    query = _last(server)["query"]
    assert query["addParents"] == ["newfolder"]
    assert "removeParents" not in query


def test_trash_file(backend) -> None:
    """gdrive_trash_file PATCHes {"trashed": true}."""
    b, server = backend
    result = json.loads(b.gdrive_trash_file("bin1"))
    assert result == {"ok": True}
    req = _last(server)
    assert req["method"] == "PATCH"
    assert req["path"] == "/files/bin1"
    assert req["body"] == {"trashed": True}


def test_path_unsafe_ids_rejected_before_any_request(backend, tmp_path) -> None:
    """Path-unsafe file/folder IDs are refused up front — no HTTP request."""
    b, server = backend
    for attempt in (
        b.gdrive_get_file("../about"),
        b.gdrive_read_file("a/b"),
        b.gdrive_download_file("a\\b", str(tmp_path / "out")),
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


def _call_every_tool(b: GoogleDriveChannelBackend, tmp_path: Path) -> list[str]:
    """Invoke every backend tool once with valid arguments; return the results."""
    return [
        b.gdrive_search_files(),
        b.gdrive_get_file("bin1"),
        b.gdrive_read_file("bin1"),
        b.gdrive_download_file("bin1", str(tmp_path / "out")),
        b.gdrive_create_folder("x"),
        b.gdrive_share_file("bin1", "a@example.com"),
        b.gdrive_move_file("bin1", "f"),
        b.gdrive_trash_file("bin1"),
    ]


def test_wrong_token_returns_ok_false(backend, composio, tmp_path) -> None:
    """A 401 from the server yields ok:false JSON from every tool — no exception."""
    b, _ = backend
    composio.token = "wrong-token"
    for result_str in _call_every_tool(b, tmp_path):
        result = json.loads(result_str)
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


def test_connection_refused_returns_ok_false(backend, tmp_path, refusing_port: int) -> None:
    """Tools return ok:false when the server is unreachable — never raise."""
    b, _ = backend
    b._base_url = f"http://127.0.0.1:{refusing_port}"
    for result_str in _call_every_tool(b, tmp_path):
        result = json.loads(result_str)
        assert result["ok"] is False
        assert result["error"]

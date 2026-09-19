# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for gmail_agent — no mocks or test doubles.

Tests token persistence, tool creation, GmailAgent construction,
authentication workflows, body extraction, and tool function signatures.
"""

from __future__ import annotations

import base64
import json
import stat
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast

import google_auth_httplib2  # type: ignore[import-untyped]
import httplib2  # type: ignore[import-untyped]
import pytest
from googleapiclient.discovery import build

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.gmail_agent import (
    GmailAgent,
    GmailChannelBackend,
    _credentials_path,
    _extract_attachments,
    _extract_body,
    _save_credentials,
    _token_path,
    main,
)
from kiss.tests.conftest import IS_WINDOWS


def _backup_and_clear() -> tuple[str | None, str | None]:
    """Back up existing token and credentials files and remove them."""
    token_backup = None
    creds_backup = None
    tp = _token_path()
    cp = _credentials_path()
    if tp.exists():
        token_backup = tp.read_text()
        tp.unlink()
    if cp.exists():
        creds_backup = cp.read_text()
        cp.unlink()
    return token_backup, creds_backup


def _restore(token_backup: str | None, creds_backup: str | None) -> None:
    """Restore previously backed-up token and credentials files."""
    tp = _token_path()
    cp = _credentials_path()
    if token_backup is not None:
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(token_backup)
    elif tp.exists():
        tp.unlink()
    if creds_backup is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(creds_backup)
    elif cp.exists():
        cp.unlink()


class TestTokenPersistence:
    """Tests for credential loading, saving, and clearing."""

    def setup_method(self) -> None:
        self._token_backup, self._creds_backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._token_backup, self._creds_backup)

    def test_save_sets_permissions(self) -> None:
        from google.oauth2.credentials import Credentials

        creds = Credentials(token="fake-perm-test")
        _save_credentials(creds)
        path = _token_path()
        assert path.exists()
        # NTFS has no POSIX mode bits: chmod(0o600) is a no-op there.
        mode = path.stat().st_mode
        assert IS_WINDOWS or mode & stat.S_IRWXG == 0
        assert IS_WINDOWS or mode & stat.S_IRWXO == 0


class TestBodyExtraction:
    """Tests for _extract_body and _extract_attachments helpers."""

    def test_plain_text_body(self) -> None:
        data = base64.urlsafe_b64encode(b"Hello world").decode()
        payload = {"mimeType": "text/plain", "body": {"data": data}}
        assert _extract_body(payload) == "Hello world"

    def test_html_body_fallback(self) -> None:
        data = base64.urlsafe_b64encode(b"<p>Hello</p>").decode()
        payload = {"mimeType": "text/html", "body": {"data": data}}
        assert _extract_body(payload) == "<p>Hello</p>"

    def test_multipart_plain(self) -> None:
        data = base64.urlsafe_b64encode(b"Multipart text").decode()
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": data}},
                {"mimeType": "text/html", "body": {"data": "aW50ZXJuZXQ="}},
            ],
        }
        assert _extract_body(payload) == "Multipart text"

    def test_multipart_html_only(self) -> None:
        data = base64.urlsafe_b64encode(b"<b>HTML</b>").decode()
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html", "body": {"data": data}},
            ],
        }
        assert _extract_body(payload) == "<b>HTML</b>"

    def test_nested_multipart(self) -> None:
        data = base64.urlsafe_b64encode(b"Nested text").decode()
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": data}},
                    ],
                },
            ],
        }
        assert _extract_body(payload) == "Nested text"

    def test_plain_text_no_data(self) -> None:
        payload = {"mimeType": "text/plain", "body": {}}
        assert _extract_body(payload) == ""

    def test_html_no_data(self) -> None:
        payload = {"mimeType": "text/html", "body": {}}
        assert _extract_body(payload) == ""

    def test_multipart_plain_no_data(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [{"mimeType": "text/plain", "body": {}}],
        }
        assert _extract_body(payload) == ""

    def test_extract_attachments_nested(self) -> None:
        payload = {
            "parts": [
                {
                    "mimeType": "multipart/mixed",
                    "parts": [
                        {
                            "filename": "nested.txt",
                            "mimeType": "text/plain",
                            "body": {"size": 42, "attachmentId": "att-456"},
                        },
                    ],
                },
            ],
        }
        result = _extract_attachments(payload)
        assert len(result) == 1
        assert result[0]["filename"] == "nested.txt"

    def test_extract_attachments_skip_non_files(self) -> None:
        payload = {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "dGVzdA=="}},
                {
                    "filename": "image.png",
                    "mimeType": "image/png",
                    "body": {"size": 2048, "attachmentId": "att-789"},
                },
            ],
        }
        result = _extract_attachments(payload)
        assert len(result) == 1
        assert result[0]["filename"] == "image.png"


class _GmailErrorHandler(BaseHTTPRequestHandler):
    """Replies 401 with a Gmail-shaped JSON error body to every request."""

    def _reply(self) -> None:
        cast(_GmailErrorServer, self.server).requests.append(
            {"method": self.command, "path": self.path}
        )
        body = json.dumps(
            {
                "error": {
                    "code": 401,
                    "message": "Invalid Credentials",
                    "errors": [
                        {
                            "message": "Invalid Credentials",
                            "domain": "global",
                            "reason": "authError",
                        }
                    ],
                    "status": "UNAUTHENTICATED",
                }
            }
        ).encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "application/json; charset=UTF-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        """Reply 401 to GET requests."""
        self._reply()

    def do_POST(self) -> None:  # noqa: N802
        """Reply 401 to POST requests."""
        self._reply()

    def do_PUT(self) -> None:  # noqa: N802
        """Reply 401 to PUT requests."""
        self._reply()

    def do_PATCH(self) -> None:  # noqa: N802
        """Reply 401 to PATCH requests."""
        self._reply()

    def do_DELETE(self) -> None:  # noqa: N802
        """Reply 401 to DELETE requests."""
        self._reply()

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


class _GmailErrorServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _GmailErrorHandler)
        self.requests: list[dict[str, str]] = []


@pytest.fixture()
def gmail_error_server():
    """Start a local 401-only Gmail endpoint; yield (base_url, server)."""
    server = _GmailErrorServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        yield base_url, server
    finally:
        stop_http_server(server, thread)


def _make_error_backend(base_url: str) -> GmailChannelBackend:
    """Create a GmailChannelBackend whose invalid token is rejected locally.

    Uses the real googleapiclient against a local server that answers 401
    for every request, so API calls fail like they do on an invalid token
    without ever reaching the real gmail.googleapis.com (and with a bounded
    transport timeout, so a black-holed network cannot hang the tests).
    """
    from google.oauth2.credentials import Credentials

    creds = Credentials(token="invalid-token-for-test")
    http = google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=10))
    backend = GmailChannelBackend()
    backend._service = build(
        "gmail",
        "v1",
        http=http,
        static_discovery=True,
        client_options={"api_endpoint": base_url},
    )
    return backend


_GMAIL_TOOL_ERROR_CASES = [
    ("get_profile", {}),
    ("list_messages", {}),
    ("get_message", {"message_id": "fake-id"}),
    ("send_email", {"to": "test@example.com", "subject": "Test", "body": "Hello"}),
    ("reply_to_message", {"message_id": "fake-id", "body": "Reply"}),
    ("create_draft", {"to": "test@example.com", "subject": "Test", "body": "Draft"}),
    ("trash_message", {"message_id": "fake-id"}),
    ("untrash_message", {"message_id": "fake-id"}),
    ("delete_message", {"message_id": "fake-id"}),
    ("modify_labels", {"message_id": "fake-id", "add_label_ids": "STARRED"}),
    ("list_labels", {}),
    ("create_label", {"name": "TestLabel"}),
    ("get_attachment", {"message_id": "fake-id", "attachment_id": "att-fake"}),
    ("get_thread", {"thread_id": "fake-thread"}),
]


class TestGmailTools:
    """Tests for GmailChannelBackend tool creation and error handling."""

    @pytest.mark.parametrize("tool_name,kwargs", _GMAIL_TOOL_ERROR_CASES)
    def test_tool_returns_error_on_invalid_token(
        self, gmail_error_server, tool_name: str, kwargs: dict
    ) -> None:
        """Every Gmail tool returns {ok: false, error: ...} with invalid credentials."""
        base_url, server = gmail_error_server
        backend = _make_error_backend(base_url)
        tools = backend.get_tool_methods()
        fn = next(t for t in tools if t.__name__ == tool_name)
        result = json.loads(fn(**kwargs))
        assert result["ok"] is False
        assert "error" in result
        assert server.requests, "tool call never reached the local Gmail endpoint"


class TestGmailAgent:
    """Tests for GmailAgent construction and tool integration."""

    def setup_method(self) -> None:
        self._token_backup, self._creds_backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._token_backup, self._creds_backup)

    def test_check_auth_unauthenticated_no_creds_file(self) -> None:
        agent = GmailAgent()
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_gmail_auth")
        result = check()
        assert "Not authenticated" in result
        assert "start_gmail_browser_setup()" in result

    def test_check_auth_unauthenticated_with_creds_file(self) -> None:
        cp = _credentials_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"installed": {"client_id": "fake"}}))
        agent = GmailAgent()
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_gmail_auth")
        result = check()
        assert "Not authenticated" in result
        assert "authenticate_gmail()" in result

    def test_authenticate_no_creds_file(self) -> None:
        agent = GmailAgent()
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_gmail")
        result = auth()
        assert "credentials.json not found" in result

    def test_clear_auth(self) -> None:
        tp = _token_path()
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text("{}")
        agent = GmailAgent()
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_gmail_auth")
        result = clear()
        assert "cleared" in result.lower()
        assert not tp.exists()
        assert agent._backend._service is None

    def test_clear_auth_when_not_authenticated(self) -> None:
        agent = GmailAgent()
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_gmail_auth")
        result = clear()
        assert "cleared" in result.lower()

    def test_check_auth_with_invalid_token(self, gmail_error_server) -> None:
        """check_gmail_auth with an invalid token returns an error."""
        base_url, server = gmail_error_server
        agent = GmailAgent()
        agent._backend = _make_error_backend(base_url)
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_gmail_auth")
        result = json.loads(check())
        assert result["ok"] is False
        assert "error" in result
        assert server.requests, "check_gmail_auth never reached the local Gmail endpoint"


class TestCLIMain:
    def test_main_missing_task_exits(self) -> None:
        import sys

        original_argv = sys.argv
        sys.argv = ["gmail_agent"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 1
        finally:
            sys.argv = original_argv

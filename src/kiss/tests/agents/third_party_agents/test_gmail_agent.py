# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for gmail_sea — no mocks or test doubles.

Tests tool creation, GmailAgent construction, the Composio sign-in
workflow, body extraction, and tool error handling.  Gmail API calls go
through the real googleapiclient and Composio SDK to a real local
Composio emulator (``composio_test_utils``), whose proxy forwards them
to a local Gmail endpoint.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, cast

import pytest

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents._composio_google import connected_account_id
from kiss.agents.third_party_agents.gmail.gmail_sea import (
    _SERVICE,
    GmailAgent,
    GmailChannelBackend,
    _extract_attachments,
    _extract_body,
    main,
)
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    TOKEN,
    connect,
    reset_state,
    start_fake_composio,
)


@pytest.fixture(autouse=True)
def _fresh_state():
    """Start and end every test with no recorded Gmail connection."""
    reset_state(_SERVICE)
    yield
    reset_state(_SERVICE)


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


class _GmailHandler(BaseHTTPRequestHandler):
    """Answers the profile call for the Composio-injected token, else 401."""

    def _reply(self) -> None:
        cast(_GmailServer, self.server).requests.append(
            {"method": self.command, "path": self.path}
        )
        if self.path.split("?", 1)[0].endswith("/users/me/profile") and (
            self.headers.get("Authorization") == f"Bearer {TOKEN}"
        ):
            profile = json.dumps({"emailAddress": "me@example.com", "messagesTotal": 3})
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=UTF-8")
            self.send_header("Content-Length", str(len(profile)))
            self.end_headers()
            self.wfile.write(profile.encode())
            return
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


class _GmailServer(ThreadedHTTPServer):
    """ThreadedHTTPServer that records every request it receives."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _GmailHandler)
        self.requests: list[dict[str, str]] = []


@pytest.fixture()
def gmail_server(monkeypatch):
    """Run a local Gmail endpoint behind the local Composio emulator.

    Yields:
        ``(composio, server)``: the emulator (Gmail connected, requests
        to gmail.googleapis.com rerouted to the local endpoint) and the
        local Gmail server.
    """
    server = _GmailServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for composio in start_fake_composio(monkeypatch):
            composio.upstream_overrides["https://gmail.googleapis.com"] = (
                f"http://127.0.0.1:{server.server_address[1]}"
            )
            connect(composio, _SERVICE)
            yield composio, server
    finally:
        stop_http_server(server, thread)


def _make_error_backend(composio) -> GmailChannelBackend:
    """Create a connected GmailChannelBackend whose token Gmail rejects."""
    composio.token = "invalid-token-for-test"
    backend = GmailChannelBackend()
    assert backend.connect() is False
    assert "Gmail auth failed" in backend.connection_info
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
        self, gmail_server, tool_name: str, kwargs: dict
    ) -> None:
        """Every Gmail tool returns {ok: false, error: ...} with invalid credentials."""
        composio, server = gmail_server
        backend = _make_error_backend(composio)
        tools = backend.get_tool_methods()
        fn = next(t for t in tools if t.__name__ == tool_name)
        result = json.loads(fn(**kwargs))
        assert result["ok"] is False
        assert "error" in result
        assert server.requests, "tool call never reached the local Gmail endpoint"


class TestGmailAgent:
    """Tests for GmailAgent construction and the Composio sign-in tools."""

    def test_check_auth_unauthenticated(self) -> None:
        agent = GmailAgent()
        assert agent._is_authenticated() is False
        assert agent._backend._service is None
        check = next(t for t in agent._get_tools() if t.__name__ == "check_gmail_auth")
        result = check()
        assert "Not authenticated with Gmail" in result
        assert "authenticate_gmail()" in result

    def test_connect_without_connection_fails(self) -> None:
        backend = GmailChannelBackend()
        assert backend.connect() is False
        assert "not connected" in backend.connection_info

    def test_connect_link_flow_builds_service(self, gmail_server) -> None:
        """authenticate -> approve -> finish wires a working Gmail service."""
        composio, _ = gmail_server
        reset_state(_SERVICE)
        agent = GmailAgent()
        tools = {t.__name__: t for t in agent._get_tools()}
        started = json.loads(tools["authenticate_gmail"]())
        composio.accounts[started["verification_uri"].rsplit("/", 1)[1]] = "ACTIVE"
        assert json.loads(tools["finish_gmail_auth"]())["ok"] is True
        assert agent._is_authenticated() is True
        assert agent._backend.connection_info == "Authenticated as me@example.com"
        assert json.loads(agent._backend.get_profile())["email"] == "me@example.com"

    def test_new_agent_uses_recorded_connection(self, gmail_server) -> None:
        """A new agent builds its service from the recorded connection."""
        agent = GmailAgent()
        assert agent._is_authenticated() is True
        assert json.loads(agent._backend.get_profile())["email"] == "me@example.com"

    def test_clear_auth(self, gmail_server) -> None:
        composio, _ = gmail_server
        account = connected_account_id(_SERVICE)
        agent = GmailAgent()
        clear = next(t for t in agent._get_tools() if t.__name__ == "clear_gmail_auth")
        assert "cleared" in clear().lower()
        assert composio.deleted == [account]
        assert agent._is_authenticated() is False


class TestCLIMain:
    def test_main_missing_task_exits(self) -> None:
        import sys

        original_argv = sys.argv
        sys.argv = ["gmail_sea"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 1
        finally:
            sys.argv = original_argv

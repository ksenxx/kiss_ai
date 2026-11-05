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

import pytest

from kiss.agents.third_party_agents._composio_google import connected_account_id
from kiss.agents.third_party_agents.gmail.gmail_sea import (
    _SERVICE,
    GmailAgent,
    GmailChannelBackend,
    _extract_attachments,
    _extract_body,
)
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    TOKEN,
    connected_endpoint,
    reset_state,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import tools_by_name
from kiss.tests.agents.third_party_agents.recording_http import RecordingHandler

pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


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


class _GmailHandler(RecordingHandler):
    """Answers the profile call for the Composio-injected token, else 401."""

    def _handle(self) -> None:
        self.record_request(None)
        if self.path.split("?", 1)[0].endswith("/users/me/profile") and (
            self.headers.get("Authorization") == f"Bearer {TOKEN}"
        ):
            profile = {"emailAddress": "me@example.com", "messagesTotal": 3}
            self._json(200, profile, "application/json; charset=UTF-8")
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
        self._reply(401, body, "application/json; charset=UTF-8")

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle  # noqa: N815


@pytest.fixture()
def gmail_server(monkeypatch):
    """Run a local Gmail endpoint behind the local Composio emulator.

    Yields:
        ``(composio, server)``: the emulator (Gmail connected, requests
        to gmail.googleapis.com rerouted to the local endpoint) and the
        local Gmail server.
    """
    with connected_endpoint(
        monkeypatch, _GmailHandler, _SERVICE, "https://gmail.googleapis.com"
    ) as wiring:
        yield wiring


def _make_error_backend(composio) -> GmailChannelBackend:
    """Create a connected GmailChannelBackend whose token Gmail rejects."""
    composio.token = "invalid-token-for-test"
    backend = GmailChannelBackend()
    assert backend.connect() is False
    assert "Gmail auth failed" in backend.connection_info
    return backend


_ME = "/gmail/v1/users/me"
# (tool, arguments, the Gmail request the tool makes and the endpoint rejects).
# A reply fetches the message it answers before sending, so that fetch is
# the rejected request.
_GMAIL_TOOL_ERROR_CASES = [
    ("get_profile", {}, ("GET", f"{_ME}/profile")),
    ("list_messages", {}, ("GET", f"{_ME}/messages")),
    ("get_message", {"message_id": "fake-id"}, ("GET", f"{_ME}/messages/fake-id")),
    (
        "send_email",
        {"to": "test@example.com", "subject": "Test", "body": "Hello"},
        ("POST", f"{_ME}/messages/send"),
    ),
    (
        "reply_to_message",
        {"message_id": "fake-id", "body": "Reply"},
        ("GET", f"{_ME}/messages/fake-id"),
    ),
    (
        "create_draft",
        {"to": "test@example.com", "subject": "Test", "body": "Draft"},
        ("POST", f"{_ME}/drafts"),
    ),
    ("trash_message", {"message_id": "fake-id"}, ("POST", f"{_ME}/messages/fake-id/trash")),
    ("untrash_message", {"message_id": "fake-id"}, ("POST", f"{_ME}/messages/fake-id/untrash")),
    (
        "modify_labels",
        {"message_id": "fake-id", "add_label_ids": "STARRED"},
        ("POST", f"{_ME}/messages/fake-id/modify"),
    ),
    ("list_labels", {}, ("GET", f"{_ME}/labels")),
    ("create_label", {"name": "TestLabel"}, ("POST", f"{_ME}/labels")),
    (
        "get_attachment",
        {"message_id": "fake-id", "attachment_id": "att-fake"},
        ("GET", f"{_ME}/messages/fake-id/attachments/att-fake"),
    ),
    ("get_thread", {"thread_id": "fake-thread"}, ("GET", f"{_ME}/threads/fake-thread")),
]


class TestGmailTools:
    """Tests for GmailChannelBackend tool creation and error handling."""

    @pytest.mark.parametrize("tool_name,kwargs,rejected", _GMAIL_TOOL_ERROR_CASES)
    def test_tool_returns_error_on_invalid_token(
        self, gmail_server, tool_name: str, kwargs: dict, rejected: tuple[str, str]
    ) -> None:
        """Every Gmail tool returns {ok: false, error: ...} with invalid credentials.

        The error must be the endpoint's own 401, reached through the tool's
        one request: a tool refused earlier (Composio, argument validation)
        also answers ``ok: false`` but proves nothing about the Gmail call.
        """
        composio, server = gmail_server
        backend = _make_error_backend(composio)
        server.requests.clear()  # connect()'s own profile call
        fn = next(t for t in backend.get_tool_methods() if t.__name__ == tool_name)
        result = json.loads(fn(**kwargs))
        assert result["ok"] is False
        assert "HttpError 401" in result["error"] and "Invalid Credentials" in result["error"]
        assert [(r["method"], r["path"]) for r in server.requests] == [rejected]


class TestGmailAgent:
    """Tests for GmailAgent construction and the Composio sign-in tools."""

    def test_check_auth_unauthenticated(self) -> None:
        agent = GmailAgent()
        assert agent._is_authenticated() is False
        assert agent._backend._service is None
        check = tools_by_name(agent)["check_gmail_auth"]
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
        tools = tools_by_name(agent)
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
        clear = tools_by_name(agent)["clear_gmail_auth"]
        assert "cleared" in clear().lower()
        assert composio.deleted == [account]
        assert agent._is_authenticated() is False

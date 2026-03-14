"""Integration tests for gmail_agent — no mocks or test doubles.

Tests token persistence, tool creation, GmailAgent construction,
authentication workflows, body extraction, and tool function signatures.
"""

from __future__ import annotations

import base64
import json
import os
import stat

from googleapiclient.discovery import build

from kiss.channels.gmail_agent import (
    GmailAgent,
    _cli_ask_user_question,
    _cli_wait_for_user,
    _credentials_path,
    _extract_attachments,
    _extract_body,
    _make_gmail_tools,
    _save_credentials,
    _token_path,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


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
        mode = path.stat().st_mode
        if os.name == "nt":
            assert path.exists()
        else:
            assert mode & stat.S_IRWXG == 0
            assert mode & stat.S_IRWXO == 0

# ---------------------------------------------------------------------------
# Body extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gmail tools (with fake service that returns HttpError)
# ---------------------------------------------------------------------------


def _make_error_service() -> object:
    """Create a Gmail service with invalid credentials for error testing.

    Uses the real googleapiclient to test error handling — API calls
    will fail with HttpError because the token is invalid.
    """
    from google.oauth2.credentials import Credentials

    creds = Credentials(token="invalid-token-for-test")
    return build("gmail", "v1", credentials=creds)


class TestGmailTools:
    """Tests for _make_gmail_tools tool creation and error handling."""

    def test_get_profile_returns_error_on_invalid_token(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        get_profile = next(t for t in tools if t.__name__ == "get_profile")
        result = json.loads(get_profile())
        assert result["ok"] is False
        assert "error" in result

    def test_list_messages_returns_error_on_invalid_token(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        list_msgs = next(t for t in tools if t.__name__ == "list_messages")
        result = json.loads(list_msgs())
        assert result["ok"] is False

    def test_get_message_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        get_msg = next(t for t in tools if t.__name__ == "get_message")
        result = json.loads(get_msg(message_id="fake-id"))
        assert result["ok"] is False

    def test_send_message_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        send = next(t for t in tools if t.__name__ == "send_message")
        result = json.loads(send(to="test@example.com", subject="Test", body="Hello"))
        assert result["ok"] is False

    def test_reply_to_message_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        reply = next(t for t in tools if t.__name__ == "reply_to_message")
        result = json.loads(reply(message_id="fake-id", body="Reply"))
        assert result["ok"] is False

    def test_create_draft_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        draft = next(t for t in tools if t.__name__ == "create_draft")
        result = json.loads(draft(to="test@example.com", subject="Test", body="Draft"))
        assert result["ok"] is False

    def test_trash_message_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        trash = next(t for t in tools if t.__name__ == "trash_message")
        result = json.loads(trash(message_id="fake-id"))
        assert result["ok"] is False

    def test_untrash_message_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        untrash = next(t for t in tools if t.__name__ == "untrash_message")
        result = json.loads(untrash(message_id="fake-id"))
        assert result["ok"] is False

    def test_delete_message_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        delete = next(t for t in tools if t.__name__ == "delete_message")
        result = json.loads(delete(message_id="fake-id"))
        assert result["ok"] is False

    def test_modify_labels_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        modify = next(t for t in tools if t.__name__ == "modify_labels")
        result = json.loads(modify(message_id="fake-id", add_label_ids="STARRED"))
        assert result["ok"] is False

    def test_list_labels_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        labels = next(t for t in tools if t.__name__ == "list_labels")
        result = json.loads(labels())
        assert result["ok"] is False

    def test_create_label_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        create = next(t for t in tools if t.__name__ == "create_label")
        result = json.loads(create(name="TestLabel"))
        assert result["ok"] is False

    def test_get_attachment_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        get_att = next(t for t in tools if t.__name__ == "get_attachment")
        result = json.loads(get_att(message_id="fake-id", attachment_id="att-fake"))
        assert result["ok"] is False

    def test_get_thread_returns_error(self) -> None:
        service = _make_error_service()
        tools = _make_gmail_tools(service)
        get_thread = next(t for t in tools if t.__name__ == "get_thread")
        result = json.loads(get_thread(thread_id="fake-thread"))
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# GmailAgent
# ---------------------------------------------------------------------------


class TestGmailAgent:
    """Tests for GmailAgent construction and tool integration."""

    def setup_method(self) -> None:
        self._token_backup, self._creds_backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._token_backup, self._creds_backup)

    def test_check_auth_unauthenticated_no_creds_file(self) -> None:
        agent = GmailAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_gmail_auth")
        result = check()
        assert "Not authenticated" in result
        assert "console.cloud.google.com" in result

    def test_check_auth_unauthenticated_with_creds_file(self) -> None:
        # Create a dummy credentials.json
        cp = _credentials_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"installed": {"client_id": "fake"}}))
        agent = GmailAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_gmail_auth")
        result = check()
        assert "Not authenticated" in result
        assert "authenticate_gmail()" in result

    def test_authenticate_no_creds_file(self) -> None:
        agent = GmailAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_gmail")
        result = auth()
        assert "credentials.json not found" in result

    def test_clear_auth(self) -> None:
        # Create a dummy token file
        tp = _token_path()
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text("{}")
        agent = GmailAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_gmail_auth")
        result = clear()
        assert "cleared" in result.lower()
        assert not tp.exists()
        assert agent._gmail_service is None

    def test_clear_auth_when_not_authenticated(self) -> None:
        agent = GmailAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_gmail_auth")
        result = clear()
        assert "cleared" in result.lower()

    def test_check_auth_with_invalid_token(self) -> None:
        """check_gmail_auth with an invalid token returns an error."""
        agent = GmailAgent()
        agent.web_use_tool = None
        agent._gmail_service = _make_error_service()
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_gmail_auth")
        result = json.loads(check())
        assert result["ok"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# CLI helpers and main
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_main_is_callable(self) -> None:
        assert callable(main)

    def test_main_missing_task_exits(self) -> None:
        import sys

        original_argv = sys.argv
        sys.argv = ["gmail_agent"]
        try:
            main()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 2
        finally:
            sys.argv = original_argv

    def test_cli_callbacks_are_callable(self) -> None:
        assert callable(_cli_wait_for_user)
        assert callable(_cli_ask_user_question)

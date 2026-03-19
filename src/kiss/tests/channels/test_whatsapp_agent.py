"""Integration tests for whatsapp_agent — no mocks or test doubles.

Tests config persistence, tool creation, WhatsAppAgent construction,
authentication workflows, and tool function signatures against the real
Meta Graph API (with invalid tokens to test error paths).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

from kiss.channels.whatsapp_agent import (
    WhatsAppAgent,
    _cli_ask_user_question,
    _cli_wait_for_user,
    _config_path,
    _load_config,
    _make_whatsapp_tools,
    _save_config,
    main,
)


def _backup_and_clear() -> str | None:
    """Back up existing config file and remove it."""
    path = _config_path()
    backup = None
    if path.exists():
        backup = path.read_text()
        path.unlink()
    return backup


def _restore(backup: str | None) -> None:
    """Restore a previously backed-up config file."""
    path = _config_path()
    if backup is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


class TestConfigPersistence:
    """Tests for _load_config, _save_config, _clear_config."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_load_corrupt_json(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json!!")
        assert _load_config() is None

    def test_load_non_dict_json(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"just a string"')
        assert _load_config() is None

    def test_load_missing_fields(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"access_token": "tok"}))
        assert _load_config() is None

    def test_load_empty_token(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"access_token": "", "phone_number_id": "123"}))
        assert _load_config() is None


_INTERACTIVE_JSON = json.dumps({
    "type": "button",
    "body": {"text": "Choose:"},
    "action": {"buttons": [
        {"type": "reply", "reply": {"id": "1", "title": "Yes"}},
    ]},
})
_CONTACTS_JSON = json.dumps([{
    "name": {"formatted_name": "John Doe"},
    "phones": [{"phone": "+14155238886"}],
}])

_WHATSAPP_TOOL_ERROR_CASES = [
    ("send_text_message", {"to": "+1234567890", "body": "test"}),
    ("send_template_message", {"to": "+1234567890", "template_name": "hello_world"}),
    ("send_template_message", {"to": "+1234567890", "template_name": "hello",
     "components": '[{"type":"body","parameters":[{"type":"text","text":"John"}]}]'}),
    ("send_media_message", {"to": "+1234567890", "media_type": "image", "media_id": "123"}),
    ("send_media_message", {"to": "+1234567890", "media_type": "document",
     "link": "https://example.com/doc.pdf", "caption": "A doc", "filename": "doc.pdf"}),
    ("send_media_message", {"to": "+1234567890", "media_type": "image"}),
    ("send_reaction", {"to": "+1234567890", "message_id": "wamid.123", "emoji": "👍"}),
    ("send_location_message", {"to": "+1234567890", "latitude": "37.7749",
     "longitude": "-122.4194", "name": "SF", "address": "San Francisco, CA"}),
    ("send_location_message", {"to": "+1234567890", "latitude": "37.7749",
     "longitude": "-122.4194"}),
    ("send_interactive_message", {"to": "+1234567890", "interactive_json": _INTERACTIVE_JSON}),
    ("send_contact_message", {"to": "+1234567890", "contacts_json": _CONTACTS_JSON}),
    ("mark_as_read", {"message_id": "wamid.123"}),
    ("get_business_profile", {}),
    ("update_business_profile", {"about": "test", "address": "123 Main",
     "description": "desc", "email": "a@b.com", "websites": "https://example.com",
     "vertical": "OTHER"}),
    ("update_business_profile", {}),
    ("upload_media", {"file_path": "/nonexistent/file.jpg", "mime_type": "image/jpeg"}),
    ("get_media_url", {"media_id": "invalid-media-id"}),
    ("delete_media", {"media_id": "invalid-media-id"}),
]


class TestWhatsAppTools:
    """Tests for _make_whatsapp_tools — all return errors with invalid tokens."""

    def _get_tool(self, name: str) -> object:
        tools = _make_whatsapp_tools("invalid-token", "invalid-phone-id", "")
        return next(t for t in tools if t.__name__ == name)

    @pytest.mark.parametrize("tool_name,kwargs", _WHATSAPP_TOOL_ERROR_CASES)
    def test_tool_returns_error_on_invalid_token(
        self, tool_name: str, kwargs: dict
    ) -> None:
        """Every WhatsApp tool returns {ok: false} with invalid credentials."""
        fn = self._get_tool(tool_name)
        result = json.loads(fn(**kwargs))  # type: ignore[operator]
        assert result["ok"] is False

    def test_upload_media_invalid_token(self) -> None:
        fn = self._get_tool("upload_media")
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            f.flush()
            try:
                result = json.loads(fn(file_path=f.name, mime_type="text/plain"))  # type: ignore[operator]
                assert result["ok"] is False
            finally:
                os.unlink(f.name)

    def test_list_templates_no_waba_id(self) -> None:
        fn = self._get_tool("list_message_templates")
        result = json.loads(fn())  # type: ignore[operator]
        assert result["ok"] is False
        assert "waba_id" in result["error"]

    def test_list_templates_with_waba_id(self) -> None:
        tools = _make_whatsapp_tools("invalid-token", "invalid-phone-id", "waba-123")
        fn = next(t for t in tools if t.__name__ == "list_message_templates")
        for kwargs in [{"status": "APPROVED"}, {}]:
            result = json.loads(fn(**kwargs))  # type: ignore[operator]
            assert result["ok"] is False


class TestWhatsAppAgent:
    """Tests for WhatsAppAgent construction and tool integration."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_check_auth_unauthenticated(self) -> None:
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_whatsapp_auth")
        result = check()
        assert "Not authenticated" in result
        assert "developers.facebook.com" in result

    def test_check_auth_with_invalid_config(self) -> None:
        _save_config("invalid-token", "invalid-phone-id")
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_whatsapp_auth")
        result = json.loads(check())
        assert result["ok"] is False

    def test_authenticate_empty_fields(self) -> None:
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_whatsapp")
        result = auth(access_token="  ", phone_number_id="  ")
        assert "required" in result.lower()

    def test_authenticate_invalid_token(self) -> None:
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth = next(t for t in tools if t.__name__ == "authenticate_whatsapp")
        result = json.loads(auth(access_token="bad-token", phone_number_id="123"))
        assert result["ok"] is False
        assert _load_config() is None

    def test_clear_auth(self) -> None:
        _save_config("token", "12345")
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_whatsapp_auth")
        result = clear()
        assert "cleared" in result.lower()
        assert _load_config() is None
        assert agent._whatsapp_config is None

    def test_clear_auth_when_not_authenticated(self) -> None:
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_whatsapp_auth")
        result = clear()
        assert "cleared" in result.lower()


class TestCLIMain:
    def test_main_is_callable(self) -> None:
        assert callable(main)

    def test_main_missing_task_exits(self) -> None:
        original_argv = sys.argv
        sys.argv = ["whatsapp_agent"]
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

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

from kiss.channels.whatsapp_agent import (
    WhatsAppAgent,
    _api_request,
    _clear_config,
    _cli_ask_user_question,
    _cli_wait_for_user,
    _config_path,
    _load_config,
    _make_whatsapp_tools,
    _save_config,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    """Tests for _load_config, _save_config, _clear_config."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear()

    def teardown_method(self) -> None:
        _restore(self._backup)

    def test_load_missing_file(self) -> None:
        assert _load_config() is None

    def test_save_and_load(self) -> None:
        _save_config("test-token", "12345", "67890")
        cfg = _load_config()
        assert cfg is not None
        assert cfg["access_token"] == "test-token"
        assert cfg["phone_number_id"] == "12345"
        assert cfg["waba_id"] == "67890"
        # Check file permissions
        assert oct(_config_path().stat().st_mode)[-3:] == "600"

    def test_save_without_waba_id(self) -> None:
        _save_config("token", "12345")
        cfg = _load_config()
        assert cfg is not None
        assert cfg["waba_id"] == ""

    def test_clear_config(self) -> None:
        _save_config("token", "12345")
        _clear_config()
        assert _load_config() is None

    def test_clear_nonexistent(self) -> None:
        _clear_config()  # should not raise
        assert _load_config() is None

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


# ---------------------------------------------------------------------------
# API request helper
# ---------------------------------------------------------------------------


class TestApiRequest:
    """Tests for _api_request with invalid credentials."""

    def test_get_request_returns_error(self) -> None:
        result = _api_request(
            "GET",
            "https://graph.facebook.com/v21.0/me",
            "invalid-token",
        )
        assert "error" in result

    def test_post_request_returns_error(self) -> None:
        result = _api_request(
            "POST",
            "https://graph.facebook.com/v21.0/12345/messages",
            "invalid-token",
            json_body={"messaging_product": "whatsapp", "to": "1234", "type": "text"},
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# WhatsApp tools
# ---------------------------------------------------------------------------


class TestWhatsAppTools:
    """Tests for _make_whatsapp_tools — all return errors with invalid tokens."""

    def _get_tool(self, name: str) -> object:
        tools = _make_whatsapp_tools("invalid-token", "invalid-phone-id", "")
        return next(t for t in tools if t.__name__ == name)

    def test_send_text_message_returns_error(self) -> None:
        fn = self._get_tool("send_text_message")
        result = json.loads(fn(to="+1234567890", body="test"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_send_template_message_returns_error(self) -> None:
        fn = self._get_tool("send_template_message")
        result = json.loads(fn(to="+1234567890", template_name="hello_world"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_send_template_message_with_components(self) -> None:
        fn = self._get_tool("send_template_message")
        components = '[{"type":"body","parameters":[{"type":"text","text":"John"}]}]'
        result = json.loads(fn(to="+1234567890", template_name="hello", components=components))  # type: ignore[operator]
        assert result["ok"] is False

    def test_send_media_message_with_id(self) -> None:
        fn = self._get_tool("send_media_message")
        result = json.loads(fn(to="+1234567890", media_type="image", media_id="123"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_send_media_message_with_link(self) -> None:
        fn = self._get_tool("send_media_message")
        result = json.loads(fn(  # type: ignore[operator]
            to="+1234567890", media_type="document",
            link="https://example.com/doc.pdf", caption="A doc", filename="doc.pdf",
        ))
        assert result["ok"] is False

    def test_send_media_message_no_caption_no_filename(self) -> None:
        fn = self._get_tool("send_media_message")
        result = json.loads(fn(  # type: ignore[operator]
            to="+1234567890", media_type="audio", media_id="123",
        ))
        assert result["ok"] is False

    def test_send_media_message_no_source(self) -> None:
        fn = self._get_tool("send_media_message")
        result = json.loads(fn(  # type: ignore[operator]
            to="+1234567890", media_type="image",
        ))
        assert result["ok"] is False

    def test_send_reaction_returns_error(self) -> None:
        fn = self._get_tool("send_reaction")
        result = json.loads(fn(to="+1234567890", message_id="wamid.123", emoji="👍"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_send_location_message(self) -> None:
        fn = self._get_tool("send_location_message")
        result = json.loads(fn(  # type: ignore[operator]
            to="+1234567890", latitude="37.7749", longitude="-122.4194",
            name="SF", address="San Francisco, CA",
        ))
        assert result["ok"] is False

    def test_send_location_message_no_optional(self) -> None:
        fn = self._get_tool("send_location_message")
        result = json.loads(fn(  # type: ignore[operator]
            to="+1234567890", latitude="37.7749", longitude="-122.4194",
        ))
        assert result["ok"] is False

    def test_send_interactive_message(self) -> None:
        fn = self._get_tool("send_interactive_message")
        interactive = json.dumps({
            "type": "button",
            "body": {"text": "Choose:"},
            "action": {"buttons": [
                {"type": "reply", "reply": {"id": "1", "title": "Yes"}},
            ]},
        })
        result = json.loads(fn(to="+1234567890", interactive_json=interactive))  # type: ignore[operator]
        assert result["ok"] is False

    def test_send_contact_message(self) -> None:
        fn = self._get_tool("send_contact_message")
        contacts = json.dumps([{
            "name": {"formatted_name": "John Doe"},
            "phones": [{"phone": "+14155238886"}],
        }])
        result = json.loads(fn(to="+1234567890", contacts_json=contacts))  # type: ignore[operator]
        assert result["ok"] is False

    def test_mark_as_read_returns_error(self) -> None:
        fn = self._get_tool("mark_as_read")
        result = json.loads(fn(message_id="wamid.123"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_get_business_profile_returns_error(self) -> None:
        fn = self._get_tool("get_business_profile")
        result = json.loads(fn())  # type: ignore[operator]
        assert result["ok"] is False

    def test_update_business_profile_returns_error(self) -> None:
        fn = self._get_tool("update_business_profile")
        result = json.loads(fn(  # type: ignore[operator]
            about="test", address="123 Main", description="desc",
            email="a@b.com", websites="https://example.com", vertical="OTHER",
        ))
        assert result["ok"] is False

    def test_update_business_profile_no_optional(self) -> None:
        fn = self._get_tool("update_business_profile")
        result = json.loads(fn())  # type: ignore[operator]
        assert result["ok"] is False

    def test_upload_media_file_not_found(self) -> None:
        fn = self._get_tool("upload_media")
        result = json.loads(fn(file_path="/nonexistent/file.jpg", mime_type="image/jpeg"))  # type: ignore[operator]
        assert result["ok"] is False
        assert "error" in result

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

    def test_get_media_url_returns_error(self) -> None:
        fn = self._get_tool("get_media_url")
        result = json.loads(fn(media_id="invalid-media-id"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_delete_media_returns_error(self) -> None:
        fn = self._get_tool("delete_media")
        result = json.loads(fn(media_id="invalid-media-id"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_list_templates_no_waba_id(self) -> None:
        fn = self._get_tool("list_message_templates")
        result = json.loads(fn())  # type: ignore[operator]
        assert result["ok"] is False
        assert "waba_id" in result["error"]

    def test_list_templates_with_waba_id(self) -> None:
        tools = _make_whatsapp_tools("invalid-token", "invalid-phone-id", "waba-123")
        fn = next(t for t in tools if t.__name__ == "list_message_templates")
        result = json.loads(fn(status="APPROVED"))  # type: ignore[operator]
        assert result["ok"] is False

    def test_list_templates_with_waba_id_no_status(self) -> None:
        tools = _make_whatsapp_tools("invalid-token", "invalid-phone-id", "waba-123")
        fn = next(t for t in tools if t.__name__ == "list_message_templates")
        result = json.loads(fn())  # type: ignore[operator]
        assert result["ok"] is False

    def test_all_tools_present(self) -> None:
        tools = _make_whatsapp_tools("t", "p", "w")
        names = {t.__name__ for t in tools}
        expected = {
            "send_text_message", "send_template_message", "send_media_message",
            "send_reaction", "send_location_message", "send_interactive_message",
            "send_contact_message", "mark_as_read", "get_business_profile",
            "update_business_profile", "upload_media", "get_media_url",
            "delete_media", "list_message_templates",
        }
        assert names == expected

    def test_all_tools_have_docstrings(self) -> None:
        tools = _make_whatsapp_tools("t", "p", "w")
        for tool in tools:
            assert tool.__doc__, f"{tool.__name__} missing docstring"


# ---------------------------------------------------------------------------
# WhatsAppAgent
# ---------------------------------------------------------------------------


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

    def test_tools_include_api_tools_when_configured(self) -> None:
        _save_config("some-token", "some-phone-id", "waba-id")
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        assert "send_text_message" in names
        assert "send_template_message" in names
        assert "check_whatsapp_auth" in names

    def test_tools_exclude_api_tools_when_not_configured(self) -> None:
        agent = WhatsAppAgent()
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        assert "send_text_message" not in names
        assert "check_whatsapp_auth" in names


# ---------------------------------------------------------------------------
# CLI helpers and main
# ---------------------------------------------------------------------------


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

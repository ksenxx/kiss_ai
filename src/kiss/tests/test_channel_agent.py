"""Integration tests for channel_agent — no mocks or test doubles.

Tests token persistence, dynamic app creation, ChannelAgent lifecycle,
generic REST tools, AppAgent construction, agent file generation,
OAuth2 config persistence, and CLI entry point.
"""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.channels.channel_agent import (
    _CHANNELS_DIR,
    AppAgent,
    ChannelAgent,
    _clear_token,
    _generic_rest_tools,
    _load_oauth2_config,
    _load_token,
    _oauth2_config_path,
    _render_agent_template,
    _save_oauth2_config,
    _save_token,
    _token_path,
    create_channel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_APP = "github"


def _backup_and_clear(app_name: str) -> str | None:
    """Back up existing token file and remove it. Returns backup content."""
    path = _token_path(app_name)
    backup = None
    if path.exists():
        backup = path.read_text()
        path.unlink()
    return backup


def _restore(app_name: str, backup: str | None) -> None:
    """Restore a previously backed-up token file."""
    path = _token_path(app_name)
    if backup is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(backup)
    elif path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


class TestTokenPersistence:
    """Tests for _load_token, _save_token, _clear_token."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear(_TEST_APP)

    def teardown_method(self) -> None:
        _restore(_TEST_APP, self._backup)

    def test_load_missing_returns_none(self) -> None:
        assert _load_token(_TEST_APP) is None

    def test_save_and_load_roundtrip(self) -> None:
        data = {"access_token": "tok123", "extra": "value"}
        _save_token(_TEST_APP, data)
        loaded = _load_token(_TEST_APP)
        assert loaded == data

    def test_save_creates_directory(self) -> None:
        _save_token("_test_nonexistent_app", {"access_token": "x"})
        try:
            assert _load_token("_test_nonexistent_app") == {
                "access_token": "x"
            }
        finally:
            _clear_token("_test_nonexistent_app")
            d = _CHANNELS_DIR / "_test_nonexistent_app"
            if d.exists():
                d.rmdir()

    def test_save_sets_permissions(self) -> None:
        _save_token(_TEST_APP, {"access_token": "secret"})
        path = _token_path(_TEST_APP)
        mode = path.stat().st_mode
        assert mode & stat.S_IRWXG == 0
        assert mode & stat.S_IRWXO == 0

    def test_clear_removes_file(self) -> None:
        _save_token(_TEST_APP, {"access_token": "tok"})
        _clear_token(_TEST_APP)
        assert _load_token(_TEST_APP) is None

    def test_clear_missing_is_noop(self) -> None:
        _clear_token(_TEST_APP)

    def test_load_corrupt_json(self) -> None:
        path = _token_path(_TEST_APP)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json!!")
        assert _load_token(_TEST_APP) is None

    def test_load_non_dict_json(self) -> None:
        path = _token_path(_TEST_APP)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"just a string"')
        assert _load_token(_TEST_APP) is None

    def test_token_path_structure(self) -> None:
        path = _token_path("myapp")
        assert path.parent.name == "myapp"
        assert path.name == "token.json"
        assert _CHANNELS_DIR in path.parents


# ---------------------------------------------------------------------------
# OAuth2 config persistence
# ---------------------------------------------------------------------------


class TestOAuth2ConfigPersistence:
    """Tests for _load_oauth2_config, _save_oauth2_config."""

    def setup_method(self) -> None:
        self._path = _oauth2_config_path("_test_oauth2_app")
        self._backup: str | None = None
        if self._path.exists():
            self._backup = self._path.read_text()

    def teardown_method(self) -> None:
        if self._backup is not None:
            self._path.write_text(self._backup)
        elif self._path.exists():
            self._path.unlink()
        parent = self._path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    def test_load_missing_returns_none(self) -> None:
        assert _load_oauth2_config("_test_oauth2_app") is None

    def test_save_and_load_roundtrip(self) -> None:
        config = {
            "auth_url": "https://example.com/auth",
            "token_url": "https://example.com/token",
        }
        _save_oauth2_config("_test_oauth2_app", config)
        loaded = _load_oauth2_config("_test_oauth2_app")
        assert loaded == config

    def test_load_corrupt_json(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("not json")
        assert _load_oauth2_config("_test_oauth2_app") is None

    def test_load_non_dict_json(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("[1, 2, 3]")
        assert _load_oauth2_config("_test_oauth2_app") is None


# ---------------------------------------------------------------------------
# Dynamic app creation (no registry)
# ---------------------------------------------------------------------------


class TestDynamicAppCreation:
    """Tests that any app name is accepted — no hardcoded registry."""

    def test_accepts_any_app_name(self) -> None:
        ch = ChannelAgent("github")
        assert ch.app_name == "github"

    def test_accepts_unknown_app(self) -> None:
        ch = ChannelAgent("jira")
        assert ch.app_name == "jira"

    def test_accepts_custom_app(self) -> None:
        ch = ChannelAgent("my_custom_app")
        assert ch.app_name == "my_custom_app"

    def test_display_name_title_case(self) -> None:
        ch = ChannelAgent("github")
        assert ch.display_name == "Github"

    def test_display_name_with_underscores(self) -> None:
        ch = ChannelAgent("my_custom_app")
        assert ch.display_name == "My Custom App"

    def test_display_name_with_hyphens(self) -> None:
        ch = ChannelAgent("my-custom-app")
        assert ch.display_name == "My Custom App"

    def test_case_insensitive(self) -> None:
        ch = ChannelAgent("  GitHub  ")
        assert ch.app_name == "github"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ChannelAgent("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ChannelAgent("   ")


# ---------------------------------------------------------------------------
# ChannelAgent
# ---------------------------------------------------------------------------


class TestChannelAgent:
    """Tests for ChannelAgent lifecycle."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear(_TEST_APP)

    def teardown_method(self) -> None:
        _restore(_TEST_APP, self._backup)

    def test_not_authenticated_initially(self) -> None:
        ch = ChannelAgent("github")
        assert not ch.is_authenticated

    def test_authenticate_with_token(self) -> None:
        ch = ChannelAgent("github")
        result = ch.authenticate_with_token("ghp_test123")
        assert "Token saved" in result
        assert ch.is_authenticated
        assert ch.token_data == {"access_token": "ghp_test123"}
        assert _load_token("github") == {"access_token": "ghp_test123"}

    def test_authenticate_strips_whitespace(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("  tok  ")
        assert ch.token_data == {"access_token": "tok"}

    def test_get_tools_unauthenticated(self) -> None:
        ch = ChannelAgent("github")
        assert ch.get_tools() == []

    def test_get_tools_authenticated(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("ghp_test")
        tools = ch.get_tools()
        assert len(tools) == 1
        assert tools[0].__name__ == "api_call"

    def test_tools_cached(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        t1 = ch.get_tools()
        t2 = ch.get_tools()
        assert t1 is t2

    def test_tools_reset_on_reauth(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok1")
        t1 = ch.get_tools()
        ch.authenticate_with_token("tok2")
        t2 = ch.get_tools()
        assert t1 is not t2

    def test_clear_auth(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        assert ch.is_authenticated
        result = ch.clear_auth()
        assert "cleared" in result
        assert not ch.is_authenticated
        assert _load_token("github") is None

    def test_is_authenticated_false_for_empty_token(self) -> None:
        ch = ChannelAgent("github")
        ch.token_data = {"access_token": ""}
        assert not ch.is_authenticated

    def test_is_authenticated_false_for_no_key(self) -> None:
        ch = ChannelAgent("github")
        ch.token_data = {"other_key": "val"}
        assert not ch.is_authenticated

    def test_loads_existing_token(self) -> None:
        _save_token("github", {"access_token": "existing"})
        ch = ChannelAgent("github")
        assert ch.is_authenticated
        assert ch.token_data["access_token"] == "existing"  # type: ignore[index]

    def test_refresh_no_refresh_token(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        result = ch.refresh_access_token()
        assert "No refresh token" in result

    def test_refresh_no_oauth2_config(self) -> None:
        ch = ChannelAgent("github")
        ch.token_data = {
            "access_token": "tok",
            "refresh_token": "ref",
        }
        result = ch.refresh_access_token()
        assert "No OAuth2 config" in result

    def test_create_app_agent(self) -> None:
        ch = ChannelAgent("github")
        path = ch.create_app_agent()
        try:
            assert isinstance(path, Path)
            assert path.exists()
            assert path.name == "GithubAgent.py"
            content = path.read_text()
            ast.parse(content)
            assert "class GithubAgent" in content
            assert 'ChannelAgent("github")' in content
        finally:
            if path.exists():
                path.unlink()

    def test_create_channel_helper(self) -> None:
        ch = create_channel("github")
        assert isinstance(ch, ChannelAgent)
        assert ch.app_name == "github"

    def test_create_channel_any_app(self) -> None:
        ch = create_channel("notion")
        assert isinstance(ch, ChannelAgent)
        assert ch.app_name == "notion"


# ---------------------------------------------------------------------------
# OAuth2 configuration
# ---------------------------------------------------------------------------


class TestOAuth2Configuration:
    """Tests for OAuth2 config and auth flow on ChannelAgent."""

    def setup_method(self) -> None:
        self._app = "_test_oauth2_ch"
        self._backup = _backup_and_clear(self._app)
        self._oauth2_path = _oauth2_config_path(self._app)
        self._oauth2_backup = None
        if self._oauth2_path.exists():
            self._oauth2_backup = self._oauth2_path.read_text()

    def teardown_method(self) -> None:
        _restore(self._app, self._backup)
        if self._oauth2_backup is not None:
            self._oauth2_path.write_text(self._oauth2_backup)
        elif self._oauth2_path.exists():
            self._oauth2_path.unlink()
        parent = self._oauth2_path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    def test_configure_oauth2(self) -> None:
        ch = ChannelAgent(self._app)
        result = ch.configure_oauth2(
            auth_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            client_id_env="TEST_CLIENT_ID",
            client_secret_env="TEST_CLIENT_SECRET",
            scopes="read write",
        )
        assert "OAuth2 configured" in result
        assert ch.oauth2_config is not None
        assert ch.oauth2_config["auth_url"] == "https://auth.example.com/authorize"

    def test_configure_oauth2_persisted(self) -> None:
        ch = ChannelAgent(self._app)
        ch.configure_oauth2(
            auth_url="https://a.com/auth",
            token_url="https://a.com/token",
            client_id_env="CID",
            client_secret_env="CSEC",
        )
        loaded = _load_oauth2_config(self._app)
        assert loaded is not None
        assert loaded["auth_url"] == "https://a.com/auth"

    def test_authenticate_oauth2_no_config(self) -> None:
        ch = ChannelAgent(self._app)
        result = ch.authenticate_oauth2("code123", "http://localhost/cb")
        assert "No OAuth2 config" in result

    def test_get_auth_url_no_config(self) -> None:
        ch = ChannelAgent(self._app)
        result = ch.get_auth_url()
        assert "No OAuth2 config" in result

    def test_get_auth_url_no_client_id(self) -> None:
        ch = ChannelAgent(self._app)
        ch.configure_oauth2(
            auth_url="https://a.com/auth",
            token_url="https://a.com/token",
            client_id_env="_TEST_MISSING_CID_ENV",
            client_secret_env="_TEST_CSEC",
        )
        old = os.environ.pop("_TEST_MISSING_CID_ENV", None)
        try:
            result = ch.get_auth_url()
            assert "_TEST_MISSING_CID_ENV" in result
        finally:
            if old is not None:
                os.environ["_TEST_MISSING_CID_ENV"] = old

    def test_get_auth_url_with_client_id(self) -> None:
        ch = ChannelAgent(self._app)
        ch.configure_oauth2(
            auth_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            client_id_env="_TEST_CID_ENV",
            client_secret_env="_TEST_CSEC_ENV",
            scopes="read write",
        )
        os.environ["_TEST_CID_ENV"] = "my_client_id"
        try:
            url = ch.get_auth_url()
            assert "auth.example.com/authorize" in url
            assert "my_client_id" in url
            assert "scope=" in url
        finally:
            os.environ.pop("_TEST_CID_ENV", None)


# ---------------------------------------------------------------------------
# Generic REST tools
# ---------------------------------------------------------------------------


class TestGenericRestTools:
    """Test that the generic tool factory creates the right tools."""

    def test_tool_count_and_names(self) -> None:
        tools = _generic_rest_tools("github", "fake_token")
        assert len(tools) == 1
        assert tools[0].__name__ == "api_call"

    def test_tool_callable_with_docstring(self) -> None:
        tools = _generic_rest_tools("myapp", "fake_token")
        assert callable(tools[0])
        assert tools[0].__doc__

    def test_tools_work_for_any_app(self) -> None:
        for app in ["github", "google", "spotify", "jira", "notion"]:
            tools = _generic_rest_tools(app, "fake_token")
            assert len(tools) == 1
            assert tools[0].__name__ == "api_call"


# ---------------------------------------------------------------------------
# AppAgent
# ---------------------------------------------------------------------------


class TestAppAgent:
    """Tests for AppAgent tool integration."""

    def setup_method(self) -> None:
        self._backup = _backup_and_clear(_TEST_APP)

    def teardown_method(self) -> None:
        _restore(_TEST_APP, self._backup)

    def test_inherits_sorcar_agent(self) -> None:
        ch = ChannelAgent("github")
        agent = AppAgent(ch)
        assert isinstance(agent, SorcarAgent)

    def test_has_auth_tools(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        assert "authenticate" in names
        assert "configure_oauth2" in names
        assert "authenticate_oauth2" in names
        assert "check_auth" in names
        assert "clear_auth" in names

    def test_has_api_call_tool_when_authenticated(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        assert "api_call" in names

    def test_no_api_call_tool_when_unauthenticated(self) -> None:
        ch = ChannelAgent("github")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        names = {t.__name__ for t in tools}
        assert "authenticate" in names
        assert "api_call" not in names

    def test_authenticate_tool_with_token(self) -> None:
        ch = ChannelAgent("github")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth_tool = next(t for t in tools if t.__name__ == "authenticate")
        result = auth_tool(token="test_token_123")
        assert "Token saved" in result
        assert ch.is_authenticated

    def test_authenticate_tool_already_authenticated(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth_tool = next(t for t in tools if t.__name__ == "authenticate")
        result = auth_tool()
        assert "Already authenticated" in result

    def test_authenticate_tool_not_authenticated(self) -> None:
        ch = ChannelAgent("github")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        auth_tool = next(t for t in tools if t.__name__ == "authenticate")
        result = auth_tool()
        assert "Not authenticated" in result

    def test_check_auth_tool(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_auth")
        assert "Authenticated" in check()

    def test_check_auth_tool_unauthenticated(self) -> None:
        ch = ChannelAgent("github")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        check = next(t for t in tools if t.__name__ == "check_auth")
        result = check()
        assert "Not authenticated" in result

    def test_clear_auth_tool(self) -> None:
        ch = ChannelAgent("github")
        ch.authenticate_with_token("tok")
        agent = AppAgent(ch)
        agent.web_use_tool = None
        tools = agent._get_tools()
        clear = next(t for t in tools if t.__name__ == "clear_auth")
        result = clear()
        assert "cleared" in result
        assert not ch.is_authenticated

    def test_agent_name(self) -> None:
        ch = ChannelAgent("github")
        agent = AppAgent(ch)
        assert "Github" in agent.name

    def test_agent_name_custom_app(self) -> None:
        ch = ChannelAgent("my_custom_app")
        agent = AppAgent(ch)
        assert "My Custom App" in agent.name

    def test_agent_with_callbacks(self) -> None:
        ch = ChannelAgent("github")

        def wait_cb(a: str, b: str) -> None:
            pass

        def ask_cb(q: str) -> str:
            return "answer"

        agent = AppAgent(
            ch,
            wait_for_user_callback=wait_cb,
            ask_user_question_callback=ask_cb,
        )
        assert agent._wait_for_user_callback is wait_cb
        assert agent._ask_user_question_callback is ask_cb


# ---------------------------------------------------------------------------
# Agent file generation
# ---------------------------------------------------------------------------


class TestAgentFileGeneration:
    """Tests for create_app_agent() file generation."""

    def test_creates_file_on_disk(self) -> None:
        ch = ChannelAgent("_test_file_gen")
        path = ch.create_app_agent()
        try:
            assert path.exists()
            assert path.name == "TestFileGenAgent.py"
            assert path.parent == _CHANNELS_DIR / "_test_file_gen"
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()

    def test_generated_file_is_valid_python(self) -> None:
        ch = ChannelAgent("_test_valid_py")
        path = ch.create_app_agent()
        try:
            ast.parse(path.read_text())
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()

    def test_generated_file_has_correct_class(self) -> None:
        ch = ChannelAgent("_test_cls_name")
        path = ch.create_app_agent()
        try:
            content = path.read_text()
            assert "class TestClsNameAgent(AppAgent):" in content
            assert 'ChannelAgent("_test_cls_name")' in content
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()

    def test_multi_word_app_name(self) -> None:
        ch = ChannelAgent("_test_my_app")
        path = ch.create_app_agent()
        try:
            assert path.name == "TestMyAppAgent.py"
            content = path.read_text()
            assert "class TestMyAppAgent" in content
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()

    def test_hyphenated_app_name(self) -> None:
        ch = ChannelAgent("_test-cool-app")
        path = ch.create_app_agent()
        try:
            assert path.name == "TestCoolAppAgent.py"
            content = path.read_text()
            assert "class TestCoolAppAgent" in content
        finally:
            path.unlink(missing_ok=True)
            # clean up both possible token.json etc
            for f in path.parent.iterdir():
                f.unlink()
            path.parent.rmdir()

    def test_overwrites_existing_file(self) -> None:
        ch = ChannelAgent("_test_overwrite")
        path = ch.create_app_agent()
        try:
            first_content = path.read_text()
            path2 = ch.create_app_agent()
            assert path == path2
            assert path.read_text() == first_content
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()

    def test_generated_file_has_main_block(self) -> None:
        ch = ChannelAgent("_test_main_block")
        path = ch.create_app_agent()
        try:
            content = path.read_text()
            assert 'if __name__ == "__main__":' in content
            assert "def main()" in content
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()

    def test_generated_file_imports(self) -> None:
        ch = ChannelAgent("_test_imports")
        path = ch.create_app_agent()
        try:
            content = path.read_text()
            assert "from kiss.channels.channel_agent import AppAgent" in content
            assert "from kiss.channels.channel_agent import" in content
        finally:
            path.unlink(missing_ok=True)
            path.parent.rmdir()


class TestRenderAgentTemplate:
    """Tests for the _render_agent_template helper."""

    def test_basic_render(self) -> None:
        content = _render_agent_template("github", "Github", "GithubAgent")
        assert "class GithubAgent" in content
        assert 'ChannelAgent("github")' in content
        ast.parse(content)

    def test_display_name_in_docstring(self) -> None:
        content = _render_agent_template("spotify", "Spotify", "SpotifyAgent")
        assert "Spotify Agent" in content
        assert "Spotify agent" in content

    def test_multi_word_display_name(self) -> None:
        content = _render_agent_template(
            "my_app", "My App", "MyAppAgent"
        )
        assert "class MyAppAgent" in content
        assert "My App" in content
        ast.parse(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the main() entry point argument handling."""

    def test_empty_app_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ChannelAgent("")

    def test_create_channel_any_name(self) -> None:
        ch = create_channel("anything")
        assert ch.app_name == "anything"

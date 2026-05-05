"""Integration tests for VS Code configuration panel backend."""

from __future__ import annotations

import io
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.vscode.vscode_config import (
    API_KEY_ENV_VARS,
    DEFAULTS,
    _get_user_shell,
    _shell_rc_path,
    apply_config_to_env,
    get_custom_model_entry,
    load_config,
    save_api_key_to_shell,
    save_config,
    source_shell_env,
)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect config and RC files to temp dir for isolation.

    Sets HOME and CONFIG_DIR/CONFIG_PATH to a temp directory so tests
    don't touch real user files.  Does NOT replace any functions — the
    real ``_shell_rc_path`` is used, reading ``Path.home()`` (which
    respects the monkeypatched HOME env var).

    Also snapshots all API key env vars so that ``save_api_key_to_shell``
    (which writes directly to ``os.environ``) does not leak test values
    into later tests.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(
        "kiss.agents.vscode.vscode_config.CONFIG_DIR", fake_home / ".kiss"
    )
    monkeypatch.setattr(
        "kiss.agents.vscode.vscode_config.CONFIG_PATH",
        fake_home / ".kiss" / "config.json",
    )
    for key in API_KEY_ENV_VARS:
        val = os.environ.get(key)
        if val is not None:
            monkeypatch.setenv(key, val)
        else:
            monkeypatch.delenv(key, raising=False)
    from kiss.core import config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.DEFAULT_CONFIG)


class TestLoadSaveConfig:
    """Test load_config / save_config round-trip."""

    def test_defaults_when_no_file(self) -> None:
        cfg = load_config()
        assert cfg == DEFAULTS

    def test_save_and_load(self) -> None:
        data = {
            "max_budget": 50,
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_api_key": "sk-test",
            "use_web_browser": False,
            "remote_password": "secret",
        }
        save_config(data)
        loaded = load_config()
        assert loaded["max_budget"] == 50
        assert loaded["custom_endpoint"] == "http://localhost:8080/v1"
        assert loaded["use_web_browser"] is False
        assert loaded["remote_password"] == "secret"

    def test_save_excludes_unknown_keys(self) -> None:
        save_config({"max_budget": 75, "secret_api_key": "should_not_save"})
        cfg_path = Path.home() / ".kiss" / "config.json"
        raw = json.loads(cfg_path.read_text())
        assert "secret_api_key" not in raw
        assert raw["max_budget"] == 75

    def test_load_survives_corrupt_json(self) -> None:
        cfg_dir = Path.home() / ".kiss"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text("{corrupt")
        assert load_config() == DEFAULTS

    def test_load_non_dict_json(self) -> None:
        cfg_dir = Path.home() / ".kiss"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text("[1, 2, 3]")
        assert load_config() == DEFAULTS

    def test_load_partial_config(self) -> None:
        """Stored config missing some keys gets defaults for the rest."""
        cfg_dir = Path.home() / ".kiss"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text('{"max_budget": 42}')
        cfg = load_config()
        assert cfg["max_budget"] == 42
        assert cfg["use_web_browser"] is True
        assert cfg["custom_endpoint"] == ""

    def test_load_with_extra_stored_keys(self) -> None:
        """Stored config with extra keys preserves them in loaded dict."""
        cfg_dir = Path.home() / ".kiss"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text('{"max_budget": 10, "extra": "val"}')
        cfg = load_config()
        assert cfg["max_budget"] == 10
        assert cfg["extra"] == "val"

    def test_save_creates_directory(self) -> None:
        """save_config creates CONFIG_DIR if it doesn't exist."""
        cfg_dir = Path.home() / ".kiss"
        assert not cfg_dir.exists()
        save_config({"max_budget": 99})
        assert cfg_dir.exists()
        assert load_config()["max_budget"] == 99

    def test_load_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError during load returns defaults."""
        cfg_dir = Path.home() / ".kiss"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "config.json"
        cfg_path.write_text('{"max_budget": 1}')
        cfg_path.unlink()
        cfg_path.mkdir()
        assert load_config() == DEFAULTS


class TestApiKeyShell:
    """Test saving API keys to shell RC files."""

    def test_save_key_to_zshrc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key_to_shell("GEMINI_API_KEY", "test-key-123")
        rc = Path.home() / ".zshrc"
        content = rc.read_text()
        # H3 fix uses shlex.quote which omits quotes for shell-safe values.
        assert (
            f"export GEMINI_API_KEY={shlex.quote('test-key-123')}" in content
        )
        assert os.environ["GEMINI_API_KEY"] == "test-key-123"

    def test_save_key_to_bashrc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/bash")
        save_api_key_to_shell("OPENAI_API_KEY", "sk-test")
        rc = Path.home() / ".bashrc"
        content = rc.read_text()
        assert f"export OPENAI_API_KEY={shlex.quote('sk-test')}" in content

    def test_save_key_to_fish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        save_api_key_to_shell("ANTHROPIC_API_KEY", "ant-key")
        rc = Path.home() / ".config" / "fish" / "config.fish"
        content = rc.read_text()
        assert "set -gx ANTHROPIC_API_KEY ant-key" in content

    def test_replace_existing_key_zsh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text('export GEMINI_API_KEY="old-key"\n# other stuff\n')
        save_api_key_to_shell("GEMINI_API_KEY", "new-key")
        content = rc.read_text()
        assert f"export GEMINI_API_KEY={shlex.quote('new-key')}" in content
        assert "old-key" not in content
        assert "# other stuff" in content

    def test_replace_existing_key_fish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        rc = Path.home() / ".config" / "fish" / "config.fish"
        rc.parent.mkdir(parents=True, exist_ok=True)
        rc.write_text("set -gx OPENAI_API_KEY old-fish-key\n# fish comment\n")
        save_api_key_to_shell("OPENAI_API_KEY", "new-fish-key")
        content = rc.read_text()
        assert "set -gx OPENAI_API_KEY new-fish-key" in content
        assert "old-fish-key" not in content
        assert "# fish comment" in content

    def test_save_key_empty_rc_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RC file exists but is empty."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text("")
        save_api_key_to_shell("OPENAI_API_KEY", "key123")
        content = rc.read_text()
        assert f"export OPENAI_API_KEY={shlex.quote('key123')}" in content

    def test_save_key_no_trailing_newline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text("# no trailing newline")
        save_api_key_to_shell("OPENAI_API_KEY", "test-key")
        content = rc.read_text()
        assert "# no trailing newline\n" in content
        assert f"export OPENAI_API_KEY={shlex.quote('test-key')}" in content

    def test_save_key_sets_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Key is set in os.environ immediately."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        save_api_key_to_shell("TOGETHER_API_KEY", "tok-val")
        assert os.environ["TOGETHER_API_KEY"] == "tok-val"

    def test_save_key_refreshes_default_config(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving a key calls _refresh_config to rebuild DEFAULT_CONFIG."""
        from kiss.core import config as config_module

        monkeypatch.setenv("SHELL", "/bin/zsh")
        old_cfg = config_module.DEFAULT_CONFIG
        save_api_key_to_shell("MINIMAX_API_KEY", "mm-key")
        assert config_module.DEFAULT_CONFIG is not old_cfg

    def test_multiple_keys_sequential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple keys saved sequentially all appear in RC file."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key_to_shell("GEMINI_API_KEY", "gem-key")
        save_api_key_to_shell("OPENAI_API_KEY", "oai-key")
        save_api_key_to_shell("ANTHROPIC_API_KEY", "ant-key")
        rc = Path.home() / ".zshrc"
        content = rc.read_text()
        assert f"export GEMINI_API_KEY={shlex.quote('gem-key')}" in content
        assert f"export OPENAI_API_KEY={shlex.quote('oai-key')}" in content
        assert f"export ANTHROPIC_API_KEY={shlex.quote('ant-key')}" in content
        assert os.environ["GEMINI_API_KEY"] == "gem-key"
        assert os.environ["OPENAI_API_KEY"] == "oai-key"
        assert os.environ["ANTHROPIC_API_KEY"] == "ant-key"


class TestApplyConfig:
    """Test config application to runtime."""

    def test_apply_budget(self) -> None:
        from kiss.core import config as config_module

        original = config_module.DEFAULT_CONFIG.max_budget
        try:
            apply_config_to_env({"max_budget": 42})
            assert config_module.DEFAULT_CONFIG.max_budget == 42.0
        finally:
            config_module.DEFAULT_CONFIG.max_budget = original

    def test_apply_default_budget_when_missing(self) -> None:
        """Missing max_budget uses DEFAULTS value."""
        from kiss.core import config as config_module

        original = config_module.DEFAULT_CONFIG.max_budget
        try:
            apply_config_to_env({})
            assert config_module.DEFAULT_CONFIG.max_budget == float(
                DEFAULTS["max_budget"]
            )
        finally:
            config_module.DEFAULT_CONFIG.max_budget = original


class TestCustomModelEntry:
    """Test custom endpoint model entry generation."""

    def test_no_endpoint_returns_none(self) -> None:
        assert get_custom_model_entry({"custom_endpoint": ""}) is None

    def test_empty_config_returns_none(self) -> None:
        assert get_custom_model_entry({}) is None

    def test_endpoint_returns_entry(self) -> None:
        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_api_key": "sk-custom",
        })
        assert entry is not None
        assert entry["name"] == "custom/v1"
        assert entry["vendor"] == "Custom"
        assert entry["endpoint"] == "http://localhost:8080/v1"
        assert entry["api_key"] == "sk-custom"

    def test_endpoint_without_key(self) -> None:
        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:1234/api",
        })
        assert entry is not None
        assert entry["api_key"] == ""

    def test_endpoint_trailing_slash(self) -> None:
        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1/",
        })
        assert entry is not None
        assert entry["name"] == "custom/v1"


class TestGetUserShell:
    """Test shell detection."""

    def test_zsh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        assert _get_user_shell() == "zsh"

    def test_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/bash")
        assert _get_user_shell() == "bash"

    def test_fish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        assert _get_user_shell() == "fish"

    def test_unknown_defaults_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/csh")
        assert _get_user_shell() == "bash"

    def test_no_shell_env_defaults_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHELL", raising=False)
        assert _get_user_shell() == "bash"


class TestShellRcPath:
    """Test RC file path resolution."""

    def test_zsh_path(self) -> None:
        assert _shell_rc_path("zsh") == Path.home() / ".zshrc"

    def test_bash_path(self) -> None:
        assert _shell_rc_path("bash") == Path.home() / ".bashrc"

    def test_fish_path(self) -> None:
        assert _shell_rc_path("fish") == Path.home() / ".config" / "fish" / "config.fish"


class TestSourceShellEnv:
    """Test sourcing shell env vars."""

    def test_source_picks_up_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = Path.home() / ".zshrc"
        rc.write_text('export GEMINI_API_KEY="sourced-key"\n')
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        source_shell_env()
        assert os.environ.get("GEMINI_API_KEY") == "sourced-key"

    def test_source_no_rc_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No RC file — should not crash."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        source_shell_env()

    def test_source_env_output_with_non_api_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Subprocess output includes lines without = and non-API-key vars.

        This covers the branch where ``"=" not in line`` and where
        ``k not in API_KEY_ENV_VARS``.
        """
        rc = Path.home() / ".zshrc"
        rc.write_text(
            'echo "no-equals-line"\n'
            'export GEMINI_API_KEY="from-source"\n'
        )
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        source_shell_env()
        assert os.environ.get("GEMINI_API_KEY") == "from-source"

    def test_source_handles_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError (e.g. shell binary not found) is caught gracefully."""
        rc = Path.home() / ".zshrc"
        rc.write_text('export GEMINI_API_KEY="key"\n')
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setenv("PATH", "")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        source_shell_env()
        assert os.environ.get("GEMINI_API_KEY") is None

    def test_source_fish_shell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fish shell sourcing doesn't crash (fish may not be installed)."""
        fish_dir = Path.home() / ".config" / "fish"
        fish_dir.mkdir(parents=True)
        rc = fish_dir / "config.fish"
        rc.write_text("set -gx OPENAI_API_KEY fish-key\n")
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        source_shell_env()


class TestCommandHandlerIntegration:
    """Integration tests for getConfig/saveConfig using real VSCodeServer."""

    def _capture_broadcasts(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[Any, io.StringIO]:
        """Create a real VSCodeServer with stdout redirected to StringIO."""
        from kiss.agents.vscode.server import VSCodeServer

        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        server = VSCodeServer()
        return server, captured

    @staticmethod
    def _parse_events(captured: io.StringIO) -> list[dict]:
        output = captured.getvalue()
        lines = [line for line in output.strip().split("\n") if line.strip()]
        return [json.loads(line) for line in lines]

    def test_get_config_broadcasts_defaults(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({"type": "getConfig"})
        events = self._parse_events(captured)
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert cfg_events[0]["config"]["max_budget"] == 100
        assert cfg_events[0]["config"]["use_web_browser"] is True

    def test_save_config_persists_and_broadcasts(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 25, "use_web_browser": False},
            "apiKeys": {},
        })
        events = self._parse_events(captured)
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert cfg_events[0]["config"]["max_budget"] == 25
        assert cfg_events[0]["config"]["use_web_browser"] is False
        assert load_config()["max_budget"] == 25

    def test_save_config_with_api_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """saveConfig with API keys writes them to RC file and env."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"OPENROUTER_API_KEY": "or-key-123"},
        })
        assert os.environ["OPENROUTER_API_KEY"] == "or-key-123"
        rc = Path.home() / ".zshrc"
        assert (
            f"export OPENROUTER_API_KEY={shlex.quote('or-key-123')}"
            in rc.read_text()
        )

    def test_save_config_skips_empty_api_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty API key values are not written to shell RC."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {},
            "apiKeys": {"GEMINI_API_KEY": ""},
        })
        rc = Path.home() / ".zshrc"
        assert not rc.exists()

    def test_save_config_refreshes_models(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """saveConfig triggers a models broadcast."""
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {"custom_endpoint": "http://localhost:8080/v1"},
            "apiKeys": {},
        })
        events = self._parse_events(captured)
        model_events = [e for e in events if e["type"] == "models"]
        assert len(model_events) == 1
        names = [m["name"] for m in model_events[0]["models"]]
        assert "custom/v1" in names

    def test_get_config_after_save(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """getConfig returns the config that was previously saved."""
        server, captured = self._capture_broadcasts(monkeypatch)
        save_config({"max_budget": 77, "remote_password": "pw123"})
        server._handle_command({"type": "getConfig"})
        events = self._parse_events(captured)
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert cfg_events[0]["config"]["max_budget"] == 77
        assert cfg_events[0]["config"]["remote_password"] == "pw123"


class TestEndToEndFlows:
    """Full integration flows across multiple functions."""

    def test_save_load_apply_budget_flow(self) -> None:
        """Save budget → load → apply → verify runtime value."""
        from kiss.core import config as config_module

        original = config_module.DEFAULT_CONFIG.max_budget
        try:
            save_config({"max_budget": 33})
            cfg = load_config()
            apply_config_to_env(cfg)
            assert config_module.DEFAULT_CONFIG.max_budget == 33.0
        finally:
            config_module.DEFAULT_CONFIG.max_budget = original

    def test_api_key_save_then_source_flow(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save API key → clear env → source env → key is back."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key_to_shell("GEMINI_API_KEY", "flow-key")
        assert os.environ["GEMINI_API_KEY"] == "flow-key"

        monkeypatch.delenv("GEMINI_API_KEY")
        assert os.environ.get("GEMINI_API_KEY") is None

        source_shell_env()
        assert os.environ.get("GEMINI_API_KEY") == "flow-key"

    def test_custom_model_in_models_list(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Custom endpoint config appears in VSCodeServer._get_models output."""
        save_config({
            "custom_endpoint": "http://localhost:9999/completions",
            "custom_api_key": "ck-test",
        })

        from kiss.agents.vscode.server import VSCodeServer

        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        server = VSCodeServer()
        server._get_models()

        output = captured.getvalue()
        lines = [line for line in output.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines]
        model_events = [e for e in events if e["type"] == "models"]
        assert len(model_events) == 1
        custom_models = [
            m for m in model_events[0]["models"] if m["vendor"] == "Custom"
        ]
        assert len(custom_models) == 1
        assert custom_models[0]["name"] == "custom/completions"
        assert custom_models[0]["api_key"] == "ck-test"

    def test_no_custom_model_without_endpoint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No custom model in list when endpoint is empty."""
        save_config({"custom_endpoint": ""})

        from kiss.agents.vscode.server import VSCodeServer

        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        server = VSCodeServer()
        server._get_models()

        output = captured.getvalue()
        lines = [line for line in output.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines]
        model_events = [e for e in events if e["type"] == "models"]
        assert len(model_events) == 1
        custom_models = [
            m for m in model_events[0]["models"] if m.get("vendor") == "Custom"
        ]
        assert len(custom_models) == 0


class TestGetCurrentApiKeys:
    """Test get_current_api_keys reads from environment / DEFAULT_CONFIG."""

    def test_returns_keys_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keys present in os.environ are returned."""
        from kiss.agents.vscode.vscode_config import get_current_api_keys

        monkeypatch.setenv("GEMINI_API_KEY", "gem-from-env")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-from-env")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        keys = get_current_api_keys()
        assert keys["GEMINI_API_KEY"] == "gem-from-env"
        assert keys["OPENAI_API_KEY"] == "oai-from-env"
        assert keys["ANTHROPIC_API_KEY"] == ""

    def test_returns_empty_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All keys empty when none are set."""
        from kiss.agents.vscode.vscode_config import get_current_api_keys

        for k in API_KEY_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        keys = get_current_api_keys()
        assert all(v == "" for v in keys.values())
        assert set(keys.keys()) == API_KEY_ENV_VARS

    def test_all_keys_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All expected API key names are included in the result."""
        from kiss.agents.vscode.vscode_config import get_current_api_keys

        keys = get_current_api_keys()
        assert set(keys.keys()) == API_KEY_ENV_VARS


class TestGetConfigIncludesApiKeys:
    """Test that getConfig command includes current API keys in the response."""

    def test_get_config_includes_api_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """getConfig broadcast includes apiKeys with current env values."""
        from kiss.agents.vscode.server import VSCodeServer

        monkeypatch.setenv("GEMINI_API_KEY", "gem-test-val")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-test-val")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        server = VSCodeServer()
        server._handle_command({"type": "getConfig"})
        events = [
            json.loads(line)
            for line in captured.getvalue().strip().split("\n")
            if line.strip()
        ]
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert "apiKeys" in cfg_events[0]
        assert cfg_events[0]["apiKeys"]["GEMINI_API_KEY"] == "gem-test-val"
        assert cfg_events[0]["apiKeys"]["OPENAI_API_KEY"] == "oai-test-val"
        assert cfg_events[0]["apiKeys"]["ANTHROPIC_API_KEY"] == ""

    def test_get_config_api_keys_after_save(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After saving an API key, getConfig returns the updated value."""
        from kiss.agents.vscode.server import VSCodeServer

        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        server = VSCodeServer()
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"TOGETHER_API_KEY": "tog-key-saved"},
        })
        captured.truncate(0)
        captured.seek(0)
        server._handle_command({"type": "getConfig"})
        events = [
            json.loads(line)
            for line in captured.getvalue().strip().split("\n")
            if line.strip()
        ]
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert cfg_events[0]["apiKeys"]["TOGETHER_API_KEY"] == "tog-key-saved"


class TestApiKeyEnvVarsConstant:
    """Verify the API_KEY_ENV_VARS frozenset is correct."""

    def test_is_frozenset(self) -> None:
        assert isinstance(API_KEY_ENV_VARS, frozenset)

    def test_expected_providers_present(self) -> None:
        expected = {
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
            "MINIMAX_API_KEY",
        }
        assert API_KEY_ENV_VARS == expected

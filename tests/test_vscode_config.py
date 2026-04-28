"""Integration tests for VS Code configuration panel backend."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kiss.agents.vscode.vscode_config import (
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
    """Redirect config and RC files to temp dir for isolation."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(
        "kiss.agents.vscode.vscode_config.CONFIG_DIR", fake_home / ".kiss"
    )
    monkeypatch.setattr(
        "kiss.agents.vscode.vscode_config.CONFIG_PATH", fake_home / ".kiss" / "config.json"
    )
    # Redirect shell RC to temp
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(
        "kiss.agents.vscode.vscode_config._shell_rc_path",
        lambda shell: fake_home / (".zshrc" if shell == "zsh" else (
            ".bashrc" if shell == "bash" else ".config/fish/config.fish"
        )),
    )
    yield


class TestLoadSaveConfig:
    """Test load_config / save_config round-trip."""

    def test_defaults_when_no_file(self) -> None:
        cfg = load_config()
        assert cfg["max_budget"] == 100
        assert cfg["custom_endpoint"] == ""
        assert cfg["custom_api_key"] == ""
        assert cfg["use_web_browser"] is True
        assert cfg["remote_password"] == ""

    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_path = tmp_path / "home" / ".kiss" / "config.json"
        monkeypatch.setattr("kiss.agents.vscode.vscode_config.CONFIG_PATH", cfg_path)
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config.CONFIG_DIR", cfg_path.parent
        )

        data = {
            "max_budget": 50,
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_api_key": "sk-test",
            "use_web_browser": False,
            "remote_password": "secret",
        }
        save_config(data)
        assert cfg_path.exists()
        loaded = load_config()
        assert loaded["max_budget"] == 50
        assert loaded["custom_endpoint"] == "http://localhost:8080/v1"
        assert loaded["use_web_browser"] is False
        assert loaded["remote_password"] == "secret"

    def test_save_excludes_unknown_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg_path = tmp_path / "home" / ".kiss" / "config.json"
        monkeypatch.setattr("kiss.agents.vscode.vscode_config.CONFIG_PATH", cfg_path)
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config.CONFIG_DIR", cfg_path.parent
        )

        save_config({"max_budget": 75, "secret_api_key": "should_not_save"})
        raw = json.loads(cfg_path.read_text())
        assert "secret_api_key" not in raw
        assert raw["max_budget"] == 75

    def test_load_survives_corrupt_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg_path = tmp_path / "home" / ".kiss" / "config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("{corrupt")
        monkeypatch.setattr("kiss.agents.vscode.vscode_config.CONFIG_PATH", cfg_path)
        cfg = load_config()
        assert cfg == DEFAULTS


class TestApiKeyShell:
    """Test saving API keys to shell RC files."""

    def test_save_key_to_zshrc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        rc_path = fake_home / ".zshrc"
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        save_api_key_to_shell("GEMINI_API_KEY", "test-key-123")
        content = rc_path.read_text()
        assert 'export GEMINI_API_KEY="test-key-123"' in content
        assert os.environ["GEMINI_API_KEY"] == "test-key-123"

    def test_save_key_to_bashrc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        rc_path = fake_home / ".bashrc"
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        save_api_key_to_shell("OPENAI_API_KEY", "sk-test")
        content = rc_path.read_text()
        assert 'export OPENAI_API_KEY="sk-test"' in content

    def test_save_key_to_fish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        rc_path = fake_home / ".config" / "fish" / "config.fish"
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        save_api_key_to_shell("ANTHROPIC_API_KEY", "ant-key")
        content = rc_path.read_text()
        assert "set -gx ANTHROPIC_API_KEY ant-key" in content

    def test_replace_existing_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        rc_path = fake_home / ".zshrc"
        rc_path.write_text('export GEMINI_API_KEY="old-key"\n# other stuff\n')
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        save_api_key_to_shell("GEMINI_API_KEY", "new-key")
        content = rc_path.read_text()
        assert 'export GEMINI_API_KEY="new-key"' in content
        assert "old-key" not in content
        assert "# other stuff" in content


class TestApplyConfig:
    """Test config application to runtime."""

    def test_apply_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kiss.core import config as config_module

        original = config_module.DEFAULT_CONFIG.max_budget
        try:
            apply_config_to_env({"max_budget": 42})
            assert config_module.DEFAULT_CONFIG.max_budget == 42.0
        finally:
            config_module.DEFAULT_CONFIG.max_budget = original


class TestCustomModelEntry:
    """Test custom endpoint model entry generation."""

    def test_no_endpoint_returns_none(self) -> None:
        assert get_custom_model_entry({"custom_endpoint": ""}) is None

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

    def test_source_picks_up_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        rc_path = fake_home / ".zshrc"
        rc_path.write_text('export GEMINI_API_KEY="sourced-key"\n')
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        source_shell_env()
        assert os.environ.get("GEMINI_API_KEY") == "sourced-key"

    def test_source_no_rc_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        rc_path = fake_home / ".zshrc_missing"
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        # Should not raise
        source_shell_env()

    def test_source_fish_shell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fish uses different source syntax."""
        fake_home = tmp_path / "home"
        fish_dir = fake_home / ".config" / "fish"
        fish_dir.mkdir(parents=True)
        rc_path = fish_dir / "config.fish"
        rc_path.write_text("set -gx OPENAI_API_KEY fish-key\n")
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        # Fish may not be installed; just verify no crash
        source_shell_env()


class TestLoadNonDictJson:
    """Test load_config when JSON is valid but not a dict."""

    def test_load_array_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg_path = tmp_path / "home" / ".kiss" / "config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("[1, 2, 3]")
        monkeypatch.setattr("kiss.agents.vscode.vscode_config.CONFIG_PATH", cfg_path)
        cfg = load_config()
        assert cfg == DEFAULTS


class TestSaveKeyNewlineHandling:
    """Test saving key when RC file doesn't end with newline."""

    def test_appends_newline_before_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        rc_path = fake_home / ".zshrc"
        rc_path.write_text("# no trailing newline")
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setattr(
            "kiss.agents.vscode.vscode_config._shell_rc_path",
            lambda shell: rc_path,
        )
        save_api_key_to_shell("OPENAI_API_KEY", "test-key")
        content = rc_path.read_text()
        assert "# no trailing newline\n" in content
        assert 'export OPENAI_API_KEY="test-key"' in content


class TestCommandHandlers:
    """Integration tests for the getConfig/saveConfig command handlers."""

    def test_get_config_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify _cmd_get_config broadcasts configData."""
        from unittest.mock import MagicMock

        from kiss.agents.vscode.commands import _CommandsMixin

        # Create a minimal mixin instance
        mixin = object.__new__(_CommandsMixin)
        broadcasts: list[dict] = []
        mixin.printer = MagicMock()  # type: ignore[attr-defined]
        mixin.printer.broadcast = lambda ev: broadcasts.append(ev)

        mixin._cmd_get_config({"type": "getConfig"})
        assert len(broadcasts) == 1
        assert broadcasts[0]["type"] == "configData"
        assert "config" in broadcasts[0]
        assert broadcasts[0]["config"]["max_budget"] == 100

    def test_save_config_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify _cmd_save_config persists and broadcasts."""
        from unittest.mock import MagicMock

        from kiss.agents.vscode.commands import _CommandsMixin

        mixin = object.__new__(_CommandsMixin)
        broadcasts: list[dict] = []
        mixin.printer = MagicMock()  # type: ignore[attr-defined]
        mixin.printer.broadcast = lambda ev: broadcasts.append(ev)
        mixin._default_model = "gemini-2.5-pro"  # type: ignore[attr-defined]
        mixin._state_lock = __import__("threading").Lock()  # type: ignore[attr-defined]

        # Mock _get_models to avoid full model loading
        mixin._get_models = lambda: broadcasts.append(  # type: ignore[attr-defined]
            {"type": "models", "models": []}
        )

        cmd = {
            "type": "saveConfig",
            "config": {"max_budget": 25, "use_web_browser": False},
            "apiKeys": {},
        }
        mixin._cmd_save_config(cmd)

        # Check that configData was broadcast
        config_events = [b for b in broadcasts if b["type"] == "configData"]
        assert len(config_events) == 1
        assert config_events[0]["config"]["max_budget"] == 25
        assert config_events[0]["config"]["use_web_browser"] is False

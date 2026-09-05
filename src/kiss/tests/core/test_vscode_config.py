# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for VS Code configuration panel backend.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_vscode_config``; the non-core tests remain there.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.vscode_config as vscode_config
from kiss.core.vscode_config import (
    API_KEY_ENV_VARS,
    DEFAULTS,
    RC_HOOK_BEGIN,
    _get_user_shell,
    _resolve_shell_path,
    _shell_rc_path,
    api_keys_env_path,
    apply_config_to_env,
    get_custom_model_entry,
    load_api_keys,
    load_config,
    save_api_key,
    save_config,
)


@pytest.fixture(autouse=True)
def _isolate_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Redirect config and RC files to temp dir for isolation.

    Sets HOME and CONFIG_DIR/CONFIG_PATH to a temp directory so tests
    don't touch real user files.  Does NOT replace any functions — the
    real ``_shell_rc_path`` is used, reading ``Path.home()`` (which
    respects the monkeypatched HOME env var).

    Also snapshots all API key env vars so that ``save_api_key``
    (which writes directly to ``os.environ``) does not leak test values
    into later tests.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    import kiss.core.vscode_config as _vc

    monkeypatch.setitem(vars(_vc), "CONFIG_DIR", fake_home / ".kiss")
    monkeypatch.setitem(
        vars(_vc), "CONFIG_PATH", fake_home / ".kiss" / "config.json",
    )
    for key in API_KEY_ENV_VARS:
        val = os.environ.get(key)
        if val is not None:
            monkeypatch.setenv(key, val)
        else:
            monkeypatch.delenv(key, raising=False)
    from kiss.core import config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.DEFAULT_CONFIG)
    saved = config_module.DEFAULT_CONFIG
    snapshot = dict(saved.model_copy(deep=True).__dict__)
    yield
    for _k, _v in snapshot.items():
        setattr(saved, _k, _v)


def _installed_posix_shell() -> str:
    """Return the first POSIX shell installed on this machine.

    Prefers ``zsh`` to keep exercising the historical default, but
    falls back to ``bash`` (present on every Linux/macOS CI box) so the
    end-to-end sourcing tests still run for real — instead of failing —
    on machines without zsh.  The legacy-key migration treats both
    shells identically (same ``source rc; env`` pipeline).
    """
    for shell in ("zsh", "bash"):
        if _resolve_shell_path(shell) is not None:
            return shell
    pytest.skip("no zsh or bash binary available on this system")


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

    def test_save_excludes_api_keys_but_keeps_extension_keys(self) -> None:
        """API keys never reach the file; other keys passed in are written.

        This used to assert that *every* key outside ``DEFAULTS`` was
        dropped, which was the bug: ``tunnel_token``,
        ``skill_permissions``, ``mcp_permissions`` and ``email`` are all
        read at runtime, so accepting and discarding them lost the value
        silently until the next daemon restart.  Only real API keys are
        excluded — they belong in the shell RC.
        """
        save_config({
            "max_budget": 75,
            "ANTHROPIC_API_KEY": "should_not_save",
            "tunnel_token": "tok-xyz",
        })
        cfg_path = Path.home() / ".kiss" / "config.json"
        raw = json.loads(cfg_path.read_text())
        assert "ANTHROPIC_API_KEY" not in raw
        assert "should_not_save" not in cfg_path.read_text()
        assert raw["tunnel_token"] == "tok-xyz"
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


class TestApiKeySave:
    """Saving stores the key in the canonical file — and nowhere else."""

    def test_save_key_writes_canonical_store(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key("GEMINI_API_KEY", "test-key-123")
        content = api_keys_env_path().read_text()
        assert (
            f"export GEMINI_API_KEY={shlex.quote('test-key-123')}" in content
        )
        assert os.environ["GEMINI_API_KEY"] == "test-key-123"

    def test_save_key_installs_rc_hook_not_a_copy(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The RC gets the sourcing hook, never a second copy of the key."""
        monkeypatch.setenv("SHELL", "/bin/bash")
        save_api_key("OPENAI_API_KEY", "sk-test")
        rc = Path.home() / ".bashrc"
        content = rc.read_text()
        assert "sk-test" not in content
        assert "OPENAI_API_KEY" not in content
        assert RC_HOOK_BEGIN in content
        assert '. "$HOME/.kiss/api_keys.env"' in content

    def test_save_key_rc_hook_installed_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated saves append exactly one hook block."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key("GEMINI_API_KEY", "one")
        save_api_key("OPENAI_API_KEY", "two")
        rc = Path.home() / ".zshrc"
        assert rc.read_text().count(RC_HOOK_BEGIN) == 1

    def test_save_key_fish_gets_no_hook(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """fish cannot source a bash-syntax file, so no hook is written."""
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        save_api_key("ANTHROPIC_API_KEY", "ant-key")
        content = api_keys_env_path().read_text()
        assert "export ANTHROPIC_API_KEY=ant-key" in content
        rc = Path.home() / ".config" / "fish" / "config.fish"
        assert not rc.exists()
        assert os.environ["ANTHROPIC_API_KEY"] == "ant-key"

    def test_replace_existing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A key already in the canonical store is replaced in place."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            'export GEMINI_API_KEY="old-key"\n# other stuff\n',
        )
        save_api_key("GEMINI_API_KEY", "new-key")
        content = env_file.read_text()
        assert f"export GEMINI_API_KEY={shlex.quote('new-key')}" in content
        assert "old-key" not in content
        assert "# other stuff" in content
        assert content.count("GEMINI_API_KEY") == 1

    def test_save_scrubs_legacy_rc_assignment(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An ``export KEY=…`` line a previous release wrote is removed.

        Left in place, an RC line sourced after the hook would shadow
        the canonical value in every interactive shell.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text('export GEMINI_API_KEY="old-key"\n# other stuff\n')
        save_api_key("GEMINI_API_KEY", "new-key")
        content = rc.read_text()
        assert "old-key" not in content
        assert "# other stuff" in content
        assert RC_HOOK_BEGIN in content

    def test_save_scrubs_legacy_fish_assignment(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        rc = Path.home() / ".config" / "fish" / "config.fish"
        rc.parent.mkdir(parents=True, exist_ok=True)
        rc.write_text("set -gx OPENAI_API_KEY old-fish-key\n# fish comment\n")
        save_api_key("OPENAI_API_KEY", "new-fish-key")
        content = rc.read_text()
        assert "old-fish-key" not in content
        assert "# fish comment" in content
        assert (
            "export OPENAI_API_KEY=new-fish-key"
            in api_keys_env_path().read_text()
        )

    def test_save_key_no_trailing_newline(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A store lacking a final newline is not corrupted by an append."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("# no trailing newline")
        save_api_key("OPENAI_API_KEY", "test-key")
        content = env_file.read_text()
        assert "# no trailing newline\n" in content
        assert f"export OPENAI_API_KEY={shlex.quote('test-key')}" in content

    def test_save_key_sets_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Key is set in os.environ immediately."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        save_api_key("TOGETHER_API_KEY", "tok-val")
        assert os.environ["TOGETHER_API_KEY"] == "tok-val"

    def test_store_file_is_owner_only(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The canonical store holds every key, so it must be mode 0600."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key("GEMINI_API_KEY", "secret")
        mode = api_keys_env_path().stat().st_mode & 0o777
        assert mode == 0o600

    def test_save_key_refreshes_default_config(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving a key refreshes DEFAULT_CONFIG without losing other settings.

        This used to assert that ``DEFAULT_CONFIG`` was replaced by a
        brand-new instance, which was the bug: rebuilding re-reads only
        the environment-backed fields, so it reset ``max_budget`` (which
        is not environment-backed) and discarded whatever
        ``apply_config_to_env`` had just applied.  The singleton is now
        updated in place; what matters is that the new key is visible.
        """
        from kiss.core import config as config_module

        monkeypatch.setenv("SHELL", "/bin/zsh")
        previous_budget = config_module.DEFAULT_CONFIG.max_budget
        config_module.DEFAULT_CONFIG.max_budget = 5.0
        try:
            save_api_key("ZAI_API_KEY", "z-key")
            assert config_module.DEFAULT_CONFIG.ZAI_API_KEY == "z-key"
            assert config_module.DEFAULT_CONFIG.max_budget == 5.0
        finally:
            config_module.DEFAULT_CONFIG.max_budget = previous_budget

    def test_multiple_keys_sequential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple keys saved sequentially all appear in the store."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key("GEMINI_API_KEY", "gem-key")
        save_api_key("OPENAI_API_KEY", "oai-key")
        save_api_key("ANTHROPIC_API_KEY", "ant-key")
        content = api_keys_env_path().read_text()
        assert f"export GEMINI_API_KEY={shlex.quote('gem-key')}" in content
        assert f"export OPENAI_API_KEY={shlex.quote('oai-key')}" in content
        assert f"export ANTHROPIC_API_KEY={shlex.quote('ant-key')}" in content
        assert os.environ["GEMINI_API_KEY"] == "gem-key"
        assert os.environ["OPENAI_API_KEY"] == "oai-key"
        assert os.environ["ANTHROPIC_API_KEY"] == "ant-key"


class TestApiKeyDelete:
    """An empty value passed to ``save_api_key`` deletes the key."""

    def test_delete_removes_store_line_env_and_config(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Delete removes the export line, os.environ, and DEFAULT_CONFIG."""
        from kiss.core import config as config_module

        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key("GEMINI_API_KEY", "gem-key-del")
        env_file = api_keys_env_path()
        assert "gem-key-del" in env_file.read_text()

        save_api_key("GEMINI_API_KEY", "")
        content = env_file.read_text()
        assert "gem-key-del" not in content
        assert "GEMINI_API_KEY" not in content
        assert "GEMINI_API_KEY" not in os.environ
        assert config_module.DEFAULT_CONFIG.GEMINI_API_KEY == ""

    def test_delete_survives_reload(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deleted key stays deleted across the next daemon start.

        This is the settings-panel delete bug: the old scheme removed
        the key only from the shell RC, so the copy in the deploy-time
        env files resurrected it at the next (service) restart.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key("GEMINI_API_KEY", "doomed")
        save_api_key("GEMINI_API_KEY", "")
        load_api_keys()
        assert "GEMINI_API_KEY" not in os.environ

    def test_delete_removes_legacy_systemd_mirror(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stale ``api_keys.systemd.env`` of an old deploy is deleted.

        The old unit's ``EnvironmentFile=-`` tolerates the missing file,
        and nothing is left that could re-inject the deleted key.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        mirror = env_file.with_name("api_keys.systemd.env")
        mirror.write_text('OPENAI_API_KEY="stale"\n')
        save_api_key("OPENAI_API_KEY", "")
        assert not mirror.exists()

    def test_delete_removes_legacy_fish_line(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        rc = Path.home() / ".config" / "fish" / "config.fish"
        rc.parent.mkdir(parents=True, exist_ok=True)
        rc.write_text("set -gx OPENAI_API_KEY fish-key-del\n")
        monkeypatch.setenv("OPENAI_API_KEY", "fish-key-del")
        save_api_key("OPENAI_API_KEY", "")
        assert "OPENAI_API_KEY" not in rc.read_text()
        assert "OPENAI_API_KEY" not in os.environ

    def test_delete_matches_whitespace_variant_lines(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hand-written ``export<TAB>KEY=...`` line is also removed.

        A literal-prefix match would leave the line behind, so a fresh
        shell would silently restore the key the panel just deleted.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text("export\tOPENAI_API_KEY=tab-separated\n# keep me\n")
        save_api_key("OPENAI_API_KEY", "")
        content = rc.read_text()
        assert "OPENAI_API_KEY" not in content
        assert "# keep me" in content

    def test_save_scrubs_whitespace_variant_rc_lines(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save also scrubs a hand-written ``export<TAB>KEY=...`` RC line."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text("export\tOPENAI_API_KEY=tab-old\n")
        save_api_key("OPENAI_API_KEY", "new-val")
        content = rc.read_text()
        assert "tab-old" not in content
        assert "OPENAI_API_KEY" not in content
        assert (
            f"export OPENAI_API_KEY={shlex.quote('new-val')}"
            in api_keys_env_path().read_text()
        )

    def test_delete_preserves_prefixed_key_names(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Deleting KEY never touches KEY_EXTRA (POSIX and fish)."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text(
            "export GEMINI_API_KEY=short\n"
            "export GEMINI_API_KEY_EXTRA=longer\n",
        )
        save_api_key("GEMINI_API_KEY", "")
        assert rc.read_text() == "export GEMINI_API_KEY_EXTRA=longer\n"

        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        fish_rc = Path.home() / ".config" / "fish" / "config.fish"
        fish_rc.parent.mkdir(parents=True, exist_ok=True)
        fish_rc.write_text(
            "set -gx GEMINI_API_KEY short\n"
            "set -gx GEMINI_API_KEY_EXTRA longer\n",
        )
        save_api_key("GEMINI_API_KEY", "")
        assert fish_rc.read_text() == "set -gx GEMINI_API_KEY_EXTRA longer\n"

    def test_delete_missing_rc_creates_no_rc(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Deleting with no RC file present must not create one."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setenv("TOGETHER_API_KEY", "from-elsewhere")
        rc = Path.home() / ".zshrc"
        assert not rc.exists()
        save_api_key("TOGETHER_API_KEY", "")
        assert not rc.exists()
        assert "TOGETHER_API_KEY" not in os.environ

    def test_delete_no_matching_line_keeps_rc_intact(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No-match delete leaves the RC byte-identical, env still popped."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setenv("ZAI_API_KEY", "env-only")
        rc = Path.home() / ".zshrc"
        rc.write_text("# unrelated\nexport OTHER_VAR=1\n")
        save_api_key("ZAI_API_KEY", "")
        assert rc.read_text() == "# unrelated\nexport OTHER_VAR=1\n"
        assert "ZAI_API_KEY" not in os.environ

    def test_multiline_value_refused(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value with an embedded newline is refused outright.

        ``shlex.quote`` would write a valid multiline assignment, but
        the line-oriented replace/delete would later remove only its
        first physical line and corrupt the RC with an unterminated
        quote — so the save never happens.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        rc = Path.home() / ".zshrc"
        rc.write_text("# untouched\n")
        save_api_key("MOONSHOT_API_KEY", "line1\nline2")
        assert rc.read_text() == "# untouched\n"
        assert "MOONSHOT_API_KEY" not in os.environ


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


class TestResolveShellPath:
    """Test absolute shell binary resolution."""

    def test_resolve_via_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``PATH`` is populated, returns the ``shutil.which`` result."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        resolved = _resolve_shell_path("sh") or _resolve_shell_path("bash")
        assert resolved is not None
        assert os.path.isabs(resolved)

    def test_resolve_falls_back_when_path_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With empty ``PATH``, falls back to known absolute locations."""
        monkeypatch.setenv("PATH", "")
        resolved = _resolve_shell_path("bash")
        if resolved is not None:
            assert os.path.isabs(resolved)
            assert Path(resolved).is_file()

    def test_resolve_missing_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Unknown shell name with no fallback yields None."""
        monkeypatch.setenv("PATH", str(tmp_path))
        assert _resolve_shell_path("no-such-shell-binary") is None


class TestShellRcPath:
    """Test RC file path resolution."""

    def test_zsh_path(self) -> None:
        assert _shell_rc_path("zsh") == Path.home() / ".zshrc"

    def test_bash_path(self) -> None:
        assert _shell_rc_path("bash") == Path.home() / ".bashrc"

    def test_fish_path(self) -> None:
        assert _shell_rc_path("fish") == Path.home() / ".config" / "fish" / "config.fish"


class TestLoadApiKeys:
    """The deterministic loader plus the one-way legacy-RC migration."""

    def test_load_from_canonical_store(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Loading is a pure parse of the file — no shell involved."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text('export GEMINI_API_KEY="stored-key"\n')
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "stored-key"

    def test_load_no_file_no_rc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Neither a store nor an RC — should not crash."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        load_api_keys()

    def test_load_imports_non_model_tokens(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every assignment loads, not only the settings-panel key names.

        Deploys ship channel tokens (``GH_TOKEN``-style) in the same
        file; on the remote these used to reach the daemon only through
        the systemd ``EnvironmentFile=``, which the loader replaces.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("KISS_TEST_CHANNEL_TOKEN", raising=False)
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "export KISS_TEST_CHANNEL_TOKEN=tok-1\n"
            "# a comment\n"
            "not an assignment\n",
        )
        load_api_keys()
        assert os.environ.get("KISS_TEST_CHANNEL_TOKEN") == "tok-1"
        del os.environ["KISS_TEST_CHANNEL_TOKEN"]

    def test_load_skips_shell_only_and_expansion_lines(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RC-distilled ``PATH=…:$PATH``-style lines never load.

        The file may come from an ``./rsorcar`` distillation of a shell
        RC; importing an unexpanded ``$PATH`` verbatim would corrupt the
        daemon environment.  A fully single-quoted ``$`` value (what
        ``shlex.quote`` writes) is a literal and does load.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("KISS_TEST_DOLLAR", raising=False)
        monkeypatch.delenv("KISS_TEST_EXPAND", raising=False)
        old_path = os.environ["PATH"]
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "export PATH=/evil:$PATH\n"
            "export KISS_TEST_EXPAND=$OTHER\n"
            "export KISS_TEST_DOLLAR='lit-$eral'\n",
        )
        load_api_keys()
        assert os.environ["PATH"] == old_path
        assert "KISS_TEST_EXPAND" not in os.environ
        assert os.environ.get("KISS_TEST_DOLLAR") == "lit-$eral"
        del os.environ["KISS_TEST_DOLLAR"]

    def test_migrates_legacy_rc_key_into_store(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A key only in the RC (old scheme) is imported and persisted."""
        shell = _installed_posix_shell()
        rc = _shell_rc_path(shell)
        rc.write_text('export GEMINI_API_KEY="legacy-key"\n')
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "legacy-key"
        assert "GEMINI_API_KEY=legacy-key" in api_keys_env_path().read_text()

    def test_migration_never_overwrites_store(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A key present in the store wins over a stale RC line."""
        shell = _installed_posix_shell()
        rc = _shell_rc_path(shell)
        rc.write_text('export GEMINI_API_KEY="stale-rc"\n')
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("export GEMINI_API_KEY=canonical\n")
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "canonical"
        assert "stale-rc" not in env_file.read_text()

    def test_migration_ignores_daemon_environment(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only RC-authored values migrate — never ad-hoc process env.

        A key injected as ``KEY=x kiss-web`` must stay ephemeral, so the
        migration shell runs with a clean environment.
        """
        shell = _installed_posix_shell()
        rc = _shell_rc_path(shell)
        rc.write_text("# no keys here\n")
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.setenv("GEMINI_API_KEY", "ad-hoc-value")
        load_api_keys()
        env_file = api_keys_env_path()
        if env_file.exists():
            assert "ad-hoc-value" not in env_file.read_text()
        assert os.environ["GEMINI_API_KEY"] == "ad-hoc-value"

    def test_migration_works_with_empty_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Migration succeeds even when ``PATH`` is empty (e.g. cron env).

        The shell binary is resolved via :func:`_resolve_shell_path`
        fallback locations, and the migration subshell gets a fixed
        system ``PATH``.
        """
        shell = _installed_posix_shell()
        rc = _shell_rc_path(shell)
        rc.write_text('export GEMINI_API_KEY="empty-path-key"\n')
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.setenv("PATH", "")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "empty-path-key"

    def test_migration_scans_rc_without_a_shell_binary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """A directly-assigned RC key migrates even with no shell at all.

        The textual scan reads the exact line format the old saver
        wrote, so a missing shell binary costs nothing for that case.
        """
        rc = Path.home() / ".zshrc"
        rc.write_text('export GEMINI_API_KEY="text-scanned"\n')
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from kiss.core import vscode_config as vc
        monkeypatch.setattr(
            vc, "_SHELL_FALLBACK_PATHS",
            {"zsh": (str(tmp_path / "no-such-zsh"),)},
        )
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "text-scanned"

    def test_migration_handles_missing_shell_binary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When neither ``PATH`` nor fallback paths contain the shell,
        the sourcing fallback logs a warning and the loader still runs.

        The key is assigned in a file the RC *sources*, so the textual
        scan cannot see it and only the (unavailable) shell could.
        """
        side = Path.home() / ".zshrc_keys"
        side.write_text('export GEMINI_API_KEY="never-set"\n')
        rc = Path.home() / ".zshrc"
        rc.write_text(f'. "{side}"\n')
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from kiss.core import vscode_config as vc
        monkeypatch.setattr(
            vc, "_SHELL_FALLBACK_PATHS",
            {"zsh": (str(tmp_path / "no-such-zsh"),)},
        )
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") is None

    def test_load_parses_quoting_edge_cases(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unbalanced quotes, empty values, and multi-word raws load honestly.

        A line ``shlex`` cannot parse (unbalanced quote) keeps its raw
        text; a multi-word remainder loads its first word (what a shell
        assigns for ``KEY=two words``); a bare ``KEY=`` loads as empty.
        None of these may crash the loader or corrupt the rest of the
        file's imports.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        for name in (
            "KISS_TEST_UNBALANCED", "KISS_TEST_EMPTY",
            "KISS_TEST_WORDS", "KISS_TEST_OK",
        ):
            monkeypatch.delenv(name, raising=False)
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "export KISS_TEST_UNBALANCED=\"oops\n"
            "export KISS_TEST_EMPTY=\n"
            "export KISS_TEST_WORDS=two words\n"
            "export KISS_TEST_OK='fine'\n",
        )
        load_api_keys()
        assert os.environ.get("KISS_TEST_UNBALANCED") == '"oops'
        assert os.environ.get("KISS_TEST_EMPTY") == ""
        assert os.environ.get("KISS_TEST_WORDS") == "two"
        assert os.environ.get("KISS_TEST_OK") == "fine"
        for name in (
            "KISS_TEST_UNBALANCED", "KISS_TEST_EMPTY",
            "KISS_TEST_WORDS", "KISS_TEST_OK",
        ):
            del os.environ[name]

    def test_save_collapses_duplicate_store_lines(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A store holding two lines for one key ends up with exactly one.

        Duplicates arise when an old deploy distilled an RC that
        assigned the same variable twice; replacing both with two copies
        of the new line would keep the file growing forever.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "export GEMINI_API_KEY=first\n"
            "GEMINI_API_KEY=second\n",
        )
        save_api_key("GEMINI_API_KEY", "final")
        content = env_file.read_text()
        assert content.count("GEMINI_API_KEY") == 1
        assert "export GEMINI_API_KEY=final" in content

    def test_load_fish_shell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fish RC migration doesn't crash (fish may not be installed)."""
        fish_dir = Path.home() / ".config" / "fish"
        fish_dir.mkdir(parents=True)
        rc = fish_dir / "config.fish"
        rc.write_text("set -gx OPENAI_API_KEY fish-key\n")
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        load_api_keys()


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

    def test_api_key_save_then_load_flow(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save API key → clear env → load (as a daemon start would) → back."""
        monkeypatch.setenv("SHELL", f"/bin/{_installed_posix_shell()}")
        save_api_key("GEMINI_API_KEY", "flow-key")
        assert os.environ["GEMINI_API_KEY"] == "flow-key"

        monkeypatch.delenv("GEMINI_API_KEY")
        assert os.environ.get("GEMINI_API_KEY") is None

        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "flow-key"


class TestGetCurrentApiKeys:
    """Test get_current_api_keys reads from environment / DEFAULT_CONFIG."""

    def test_returns_keys_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keys present in os.environ are returned."""
        from kiss.core.vscode_config import get_current_api_keys

        monkeypatch.setenv("GEMINI_API_KEY", "gem-from-env")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-from-env")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        keys = get_current_api_keys()
        assert keys["GEMINI_API_KEY"] == "gem-from-env"
        assert keys["OPENAI_API_KEY"] == "oai-from-env"
        assert keys["ANTHROPIC_API_KEY"] == ""

    def test_returns_empty_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All keys empty when none are set."""
        from kiss.core.vscode_config import get_current_api_keys

        for k in API_KEY_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        keys = get_current_api_keys()
        assert all(v == "" for v in keys.values())
        assert set(keys.keys()) == API_KEY_ENV_VARS

    def test_all_keys_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All expected API key names are included in the result."""
        from kiss.core.vscode_config import get_current_api_keys

        keys = get_current_api_keys()
        assert set(keys.keys()) == API_KEY_ENV_VARS


class TestApiKeyEnvVarsConstant:
    """Verify the API_KEY_ENV_VARS frozenset is correct."""

    def test_is_frozenset(self) -> None:
        assert isinstance(API_KEY_ENV_VARS, frozenset)

    def test_expected_providers_present(self) -> None:
        expected = {
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_WORKSPACE_ID",
            "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
            "ZAI_API_KEY",
            "MOONSHOT_API_KEY",
        }
        assert API_KEY_ENV_VARS == expected


class TestSaveConfigAtomicity:
    """``save_config`` must replace ``config.json`` atomically.

    The VS Code extension's ``readKissConfig`` reads ``config.json``
    while the Python ``save_config`` may be writing it.  If the writer
    truncates the file before populating it (i.e. uses ``open(path, "w")``
    + incremental ``json.dump``), concurrent readers see an empty or
    half-written file and silently fall back to ``{}``, which then makes
    the extension prompt the user for a ``remote_password`` that is
    actually already set.
    """

    def test_concurrent_reader_never_sees_empty_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reader hammering the file during ``save_config`` calls
        must always observe a valid JSON object with ``remote_password``
        set — never an empty file, partial bytes, or ``JSONDecodeError``.
        """
        import threading

        from kiss.core import vscode_config as vc

        save_config({"remote_password": "secret-1", "max_budget": 1})
        assert vc.CONFIG_PATH.exists()

        stop = threading.Event()
        bad_reads: list[str] = []

        def reader() -> None:
            while not stop.is_set():
                try:
                    raw = vc.CONFIG_PATH.read_bytes()
                except FileNotFoundError:
                    bad_reads.append("FileNotFoundError")
                    continue
                if not raw.strip():
                    bad_reads.append("empty")
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    bad_reads.append(f"parse:{e.msg}")
                    continue
                if not isinstance(parsed, dict):
                    bad_reads.append("not-dict")
                    continue
                pw = parsed.get("remote_password")
                if pw not in ("secret-1", "secret-2"):
                    bad_reads.append(f"bad-pw:{pw!r}")

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        try:
            for i in range(200):
                save_config({
                    "remote_password": "secret-2" if i % 2 else "secret-1",
                    "max_budget": i,
                })
        finally:
            stop.set()
            t.join(timeout=5)

        assert not bad_reads, (
            f"Concurrent reader observed invalid states during save_config: "
            f"{bad_reads[:10]} (total {len(bad_reads)})"
        )

    def test_remote_password_preserved_when_other_fields_saved(
        self,
    ) -> None:
        """Saving a config dict that omits ``remote_password`` must
        preserve the existing ``remote_password`` value (regression
        guard for the ``save_config`` merge behaviour relied on by
        the install flow).
        """
        save_config({"remote_password": "sensca95", "max_budget": 10})
        save_config({"max_budget": 25})
        cfg = load_config()
        assert cfg["remote_password"] == "sensca95"
        assert cfg["max_budget"] == 25


class TestRetiredKeys:
    """A setting that no longer exists must be forgotten, not preserved.

    ``config.json`` is written by every previous release, so removing a
    key from :data:`DEFAULTS` cannot be the whole job: ``load_config``
    overlays whatever the file holds, ``sanitize_config`` deliberately
    lets unknown keys through so genuine extension-owned keys survive,
    and ``save_config`` rewrites the file from its own former contents.
    """

    def _write_legacy_config(self) -> Path:
        """Write a config file as an older release would have left it."""
        path = Path(vscode_config.CONFIG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "demo_mode": True,
                "max_budget": 42,
                "tunnel_token": "keep-me",
                "email": "someone@example.com",
            }),
            encoding="utf-8",
        )
        return path

    def test_demo_mode_is_retired(self) -> None:
        assert "demo_mode" in vscode_config.RETIRED_KEYS
        assert "demo_mode" not in DEFAULTS

    def test_load_config_drops_a_retired_key(self) -> None:
        self._write_legacy_config()
        cfg = load_config()
        assert "demo_mode" not in cfg
        assert cfg["max_budget"] == 42
        assert cfg["tunnel_token"] == "keep-me"

    def test_save_config_purges_a_retired_key_from_disk(self) -> None:
        path = self._write_legacy_config()
        save_config({"max_budget": 7})
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert "demo_mode" not in stored, (
            "a retired setting must be purged from config.json, not "
            f"rewritten forever: {stored}"
        )
        assert stored["max_budget"] == 7
        assert stored["tunnel_token"] == "keep-me"
        assert stored["email"] == "someone@example.com"

    def test_save_config_ignores_an_incoming_retired_key(self) -> None:
        save_config({"demo_mode": True, "max_budget": 9})
        path = Path(vscode_config.CONFIG_PATH)
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert "demo_mode" not in stored
        assert "demo_mode" not in load_config()


class TestReviewRegressions:
    """Regressions for the reviewed failure scenarios of the key rework."""

    def test_migration_sees_past_debian_interactivity_guard(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An old key appended after the stock non-interactive guard migrates.

        The previous saver appended ``export KEY=…`` to the END of
        ``.bashrc``; a stock Debian/Ubuntu RC returns near the top for
        non-interactive shells, so sourcing alone never reaches the key.
        The textual scan must find it anyway.
        """
        rc = Path.home() / ".bashrc"
        rc.write_text(
            "case $- in\n"
            "    *i*) ;;\n"
            "      *) return;;\n"
            "esac\n"
            "export GEMINI_API_KEY=legacy-after-guard\n",
        )
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "legacy-after-guard"
        assert "legacy-after-guard" in api_keys_env_path().read_text()

    def test_delete_scrubs_every_shell_rc(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A key deleted under zsh cannot resurrect from a bash RC copy.

        Old releases wrote to whichever shell was current at the time,
        and users switch shells — deletion must reach all of them or a
        later daemon start under the other shell re-imports the key.
        """
        (Path.home() / ".zshrc").write_text("export GEMINI_API_KEY=zsh-copy\n")
        (Path.home() / ".bashrc").write_text("export GEMINI_API_KEY=bash-copy\n")
        fish_rc = Path.home() / ".config" / "fish" / "config.fish"
        fish_rc.parent.mkdir(parents=True, exist_ok=True)
        fish_rc.write_text("set -gx GEMINI_API_KEY fish-copy\n")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("export GEMINI_API_KEY=canonical\n")
        monkeypatch.setenv("GEMINI_API_KEY", "canonical")

        monkeypatch.setenv("SHELL", "/bin/zsh")
        save_api_key("GEMINI_API_KEY", "")
        for rc in (Path.home() / ".zshrc", Path.home() / ".bashrc", fish_rc):
            assert "GEMINI_API_KEY" not in rc.read_text(), rc

        monkeypatch.setenv("SHELL", "/bin/bash")
        load_api_keys()
        assert "GEMINI_API_KEY" not in os.environ
        assert "GEMINI_API_KEY" not in env_file.read_text()

    def test_symlinked_rc_edited_in_place(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Editing a dotfiles-managed RC keeps the symlink and the target.

        Atomically replacing the link path itself would disconnect the
        maintained file — and leave the deleted key inside it.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        real = Path.home() / "dotfiles" / "zshrc"
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text("export GEMINI_API_KEY=dotfiles-copy\n# mine\n")
        rc = Path.home() / ".zshrc"
        rc.symlink_to(real)
        monkeypatch.setenv("GEMINI_API_KEY", "dotfiles-copy")

        save_api_key("GEMINI_API_KEY", "")

        assert rc.is_symlink()
        assert rc.resolve() == real.resolve()
        assert "GEMINI_API_KEY" not in real.read_text()
        assert "# mine" in real.read_text()

    def test_saved_value_with_apostrophe_and_dollar_round_trips(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``shlex.quote``'s concatenated quoting must load back exactly.

        A value holding both an apostrophe and ``$`` is stored as
        ``'abc'"'"'def$ghi'``; the loader must read it as the literal it
        is instead of rejecting it as an expansion line.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        value = "abc'def$ghi"
        save_api_key("OPENAI_API_KEY", value)
        monkeypatch.delenv("OPENAI_API_KEY")
        load_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == value

    def test_nul_value_refused_and_poisoned_store_still_loads(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A NUL never reaches the store; a poisoned line never kills startup.

        ``os.environ`` raises on NUL, so persisting one first would break
        the save *and* every later daemon start.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        save_api_key("OPENAI_API_KEY", "abc\x00def")
        env_file = api_keys_env_path()
        if env_file.exists():
            assert "abc" not in env_file.read_text()
        assert "OPENAI_API_KEY" not in os.environ

        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "export OPENAI_API_KEY='poi\x00soned'\n"
            "export GEMINI_API_KEY=healthy\n",
        )
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "healthy"
        assert "OPENAI_API_KEY" not in os.environ

    def test_code_only_upgrade_retires_systemd_mirror(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bare daemon restart removes the old deploy's second key file.

        With every model key already canonical, migration edits nothing —
        the loader itself must retire the mirror, or a git-pull upgrade
        keeps two key stores (and the old unit keeps injecting the stale
        one) indefinitely.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "".join(f"export {k}=v-{k}\n" for k in sorted(API_KEY_ENV_VARS)),
        )
        mirror = env_file.with_name("api_keys.systemd.env")
        mirror.write_text("OPENAI_API_KEY=stale\n")
        load_api_keys()
        assert not mirror.exists()
        for k in API_KEY_ENV_VARS:
            assert os.environ.get(k) == f"v-{k}"

    def test_rc_background_child_cannot_stall_startup(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A background process started by the RC must not block loading.

        The sourcing shell's descendants inherit the output pipe; the
        migration must kill the whole session on timeout instead of
        waiting for them.
        """
        import time as _time

        shell = _installed_posix_shell()
        rc = _shell_rc_path(shell)
        # The key is defined indirectly so the textual scan cannot skip
        # the sourcing fallback, and a child outlives the shell.
        side = Path.home() / ".rc_keys"
        side.write_text("export GEMINI_API_KEY=indirect\n")
        rc.write_text(f'. "{side}"\nsleep 30 &\n')
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(vscode_config, "_MIGRATION_TIMEOUT_S", 0.5)
        start = _time.monotonic()
        load_api_keys()
        elapsed = _time.monotonic() - start
        assert elapsed < 5.0, f"load_api_keys blocked for {elapsed:.1f}s"

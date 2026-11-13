# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the chat page's settings panel backend.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_user_config``; the non-core tests remain there.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.user_config as user_config
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.processes import bash_version, pid_alive
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.core.user_config import (
    _RC_HOOK_LINE,
    API_KEY_ENV_VARS,
    DEFAULTS,
    RC_HOOK_BEGIN,
    RC_HOOK_END,
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
from kiss.core.utils import atomic_write_text
from kiss.tests.conftest import (
    AMBIENT_AWS_ENV_VARS,
    HOT_READER_PAUSE,
    IS_WINDOWS,
    is_root,
    posix_only,
)

_BASH_VERSION = bash_version()


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
    monkeypatch.setenv("USERPROFILE", str(fake_home))  # what Path.home() reads on Windows
    import kiss.core.user_config as _vc

    monkeypatch.setitem(vars(_vc), "CONFIG_DIR", fake_home / ".kiss")
    monkeypatch.setitem(
        vars(_vc), "CONFIG_PATH", fake_home / ".kiss" / "config.json",
    )
    # Protect API-key env vars against DIRECT os.environ writes made by
    # save_api_key/load_api_keys during the tests.
    #
    # Present keys: ``monkeypatch.setenv(key, current)`` records the
    # original value as the EARLIEST undo entry; monkeypatch undoes in
    # reverse order, so even when a test's own ``monkeypatch.delenv``
    # captures a mid-test mutated value, the setup-time record is
    # applied last and the original wins.
    #
    # Absent keys: ``monkeypatch.delenv(key, raising=False)`` records
    # NOTHING, so a key the tests then set directly (e.g. a fake
    # ANTHROPIC_WORKSPACE_ID written by load_api_keys()) leaked into
    # every later test file in the process and poisoned live Anthropic
    # calls with a 400 "must be a valid workspace ID".  The explicit
    # snapshot restore after ``yield`` removes such keys.
    env_snapshot = {key: os.environ.get(key) for key in API_KEY_ENV_VARS}
    for key, val in env_snapshot.items():
        if val is not None:
            monkeypatch.setenv(key, val)
        else:
            # Record the key's ABSENCE as the earliest undo entries: the
            # setenv records "was absent" (undo deletes), the delenv
            # keeps it absent during the test.  Without this pair, a
            # test that writes the key directly (save_api_key) and then
            # calls its own monkeypatch.delenv leaves that delenv's
            # captured mid-test value as the LAST undo applied,
            # resurrecting the leak after the explicit restore below.
            monkeypatch.setenv(key, "")
            monkeypatch.delenv(key)
    from kiss.core import config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.DEFAULT_CONFIG)
    saved = config_module.DEFAULT_CONFIG
    snapshot = dict(saved.model_copy(deep=True).__dict__)
    yield
    for _k, _v in snapshot.items():
        setattr(saved, _k, _v)
    for key, val in env_snapshot.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


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


def _sourced(shell: str, rc: Path) -> str:
    """What the real *shell* prints sourcing *rc*, then ``key=`` and the key it holds."""
    if shell == "fish":
        show = "set -q OPENAI_API_KEY; and echo key=$OPENAI_API_KEY; or echo key=unset"
        command = ["--no-config", "-c", "source $argv[1]; " + show, str(rc)]
    else:
        show = 'source "$1"; printf "key=%s\\n" "${OPENAI_API_KEY-unset}"'
        flags = ["-f"] if shell == "zsh" else ["--noprofile", "--norc"]
        command = [*flags, "-c", show, "probe", str(rc)]
    return subprocess.run(
        [str(_resolve_shell_path(shell)), *command],
        capture_output=True, text=True, timeout=30,
        env={"HOME": str(Path.home()), "PATH": os.defpath},
    ).stdout


def _load_api_keys_into(raised: list[BaseException]) -> None:
    """Run ``load_api_keys`` and record what it raises (a thread's target)."""
    try:
        load_api_keys()
    except BaseException as error:  # noqa: BLE001 - the Stop under test is no Exception
        raised.append(error)


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

    @posix_only("creating a symlink needs no privilege")
    def test_a_symlinked_config_is_saved_where_it_lives(self, tmp_path: Path) -> None:
        """The saved file was renamed over the link: the config the link named (one kept
        in a dotfiles checkout) was disconnected and kept its old settings."""
        real = tmp_path / "dotfiles" / "config.json"
        real.parent.mkdir()
        real.write_text('{"max_budget": 1}')
        link = user_config._config_path()
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
        save_config({"max_budget": 99})
        assert link.is_symlink() and json.loads(real.read_text())["max_budget"] == 99
        atomic_write_text(link, "{}")
        assert link.is_symlink() and real.read_text() == "{}"
        # A dangling link names no file to keep: it is replaced, as before.
        real.unlink()
        save_config({"max_budget": 5})
        assert not link.is_symlink() and load_config()["max_budget"] == 5
        # So is the link of a caller whose writes must stay in its directory.
        real.write_text("kept")
        link.unlink()
        link.symlink_to(real)
        atomic_write_text(link, "{}", follow_symlink=False)
        assert not link.is_symlink() and real.read_text() == "kept"

    def test_record_recent_work_dir_stores_extra_entries_in_one_write(
        self, tmp_path: Path,
    ) -> None:
        """Adopting a work dir persists the row and the extra entries together.

        A folder that is no directory (not created yet, or removed
        since the caller looked) gets no row, but its extra entries are
        stored all the same: they were dropped with the row, so the
        daemon adopted a work dir that a restart forgot.
        """
        work = tmp_path / "proj"
        work.mkdir()
        save_config({"max_budget": 7, "custom_endpoint": "http://x/v1"})
        user_config.record_recent_work_dir(str(work), {"work_dir": str(work), "max_budget": "9"})
        cfg = load_config()
        assert cfg["work_dir"] == str(work)
        assert cfg["max_budget"] == 9  # sanitized like save_config
        assert cfg["custom_endpoint"] == "http://x/v1"
        assert [e["path"] for e in user_config.recent_work_dirs()] == [str(work)]
        gone = str(tmp_path / "gone")
        user_config.record_recent_work_dir(gone, {"work_dir": gone})
        assert load_config()["work_dir"] == gone
        assert [e["path"] for e in user_config.recent_work_dirs()] == [str(work)]

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

    def test_save_key_joins_another_brands_hook_block(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A block another brand's install left (same markers, its own source line) gains
        this brand's source line instead of blocking it or opening a second block."""
        monkeypatch.setenv("SHELL", "/bin/bash")
        other = '[ -f "$HOME/.other/api_keys.env" ] && . "$HOME/.other/api_keys.env"'
        rc = Path.home() / ".bashrc"
        rc.write_text(f"# mine\n{RC_HOOK_BEGIN}\n{other}\n{RC_HOOK_END}\n")
        save_api_key("OPENAI_API_KEY", "sk-test")
        save_api_key("GEMINI_API_KEY", "g-test")
        lines = rc.read_text().splitlines()
        assert lines == ["# mine", RC_HOOK_BEGIN, other, _RC_HOOK_LINE, RC_HOOK_END]

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
        # No RC will be written, so no lock file (nor its directory) is made for it.
        assert not (Path.home() / ".config").exists()
        assert os.environ["ANTHROPIC_API_KEY"] == "ant-key"

    @posix_only("an unreadable RC needs chmod-based permission denial")
    def test_a_failed_rc_scrub_leaves_the_store_and_the_process_agreeing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A save whose RC scrub fails (the error reaches the caller) changes
        nothing: the store, the process environment and the config singleton
        all keep the old key.  Editing the store first left it holding the new
        key while this daemon kept using, and showing, the old one until a
        restart loaded the new."""
        if is_root():
            pytest.skip("root reads a mode-000 file")
        from kiss.core import config as config_module

        monkeypatch.setenv("SHELL", "/bin/zsh")
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("export OPENAI_API_KEY=old\n")
        load_api_keys()
        assert os.environ["OPENAI_API_KEY"] == "old"
        rc = Path.home() / ".zshrc"
        rc.write_text("export OPENAI_API_KEY=old\n")
        rc.chmod(0)
        try:
            with pytest.raises(PermissionError):
                save_api_key("OPENAI_API_KEY", "new")
        finally:
            rc.chmod(0o600)
        assert env_file.read_text() == "export OPENAI_API_KEY=old\n"
        assert os.environ["OPENAI_API_KEY"] == "old"
        assert config_module.DEFAULT_CONFIG.OPENAI_API_KEY == "old"

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

    @posix_only("Windows chmod carries no owner-only mode bits")
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
        """Fish has no here-documents: a ``<<DOC`` in its RC hides no line from the deletion."""
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        rc = Path.home() / ".config" / "fish" / "config.fish"
        rc.parent.mkdir(parents=True, exist_ok=True)
        rc.write_text("cat <<DOC\nset -gx OPENAI_API_KEY fish-key-del\nDOC\n")
        monkeypatch.setenv("OPENAI_API_KEY", "fish-key-del")
        save_api_key("OPENAI_API_KEY", "")
        assert rc.read_text() == "cat <<DOC\nDOC\n"
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

    def test_delete_keeps_the_other_commands_of_a_compound_rc_line(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only the assignment of the key goes; the commands after the ``;`` stay.

        The scrub dropped the whole physical line, so deleting one key
        also deleted an unrelated assignment sharing its line, and a
        line assigning the key after another command was left behind.
        A ``;`` inside quotes is a value character, not a separator.

        Then it dropped everything up to the next ``;``: the ``&& export
        GEMINI_API_KEY=…`` of an and-list, the other operands of one
        ``export``, and the ``)`` of a command substitution holding a
        ``;`` -- a deletion that lost unrelated credentials or left the
        RC unparseable.  Only the key's own ``KEY=…`` word goes; an
        export emptied inside an and/or-list or a compound command is
        replaced by the ``true`` no-op so the structure stays valid, and
        an export inside ``$(…)`` runs in a subshell that never reaches
        the sourcing shell, so that line is left alone.

        A redirection is not an operand: ``export KEY=old 2>/dev/null``
        left ``export 2>/dev/null``, and a bare ``export`` prints every
        exported variable -- the other keys -- to whatever sources the
        RC.  The export goes whole when the key was its only operand;
        the redirection stays with the operands it keeps.

        The ``&`` of ``2>&1`` (``&>/dev/null``) is part of the
        redirection, not a background separator: split there, the scrub
        left ``true &1 GEMINI_API_KEY=keep``, which runs the command
        ``1`` and loses the other key.  An escaped ``\\>`` is a value
        character, though, and the ``&`` after it a background
        separator again: taken for a redirection's, it joined the next
        command into the export, whose operands that command became
        once the key went (``export printf …``).  A here-document
        (``<<``, ``<<-``) is a redirection too, and its lines follow:
        the emptied export becomes ``true`` with the document rather
        than leaving a bare ``export << EOF`` (every variable printed)
        or the document's lines to run as commands.  The operator needs
        no blank before it: the shell reads ``KEY=old<<EOF`` as the word
        ``KEY=old`` and the operator ``<<``; read as one word, the
        document went unseen and the introducer was deleted, leaving
        its lines to run as commands.  A process substitution glued to
        the value (``KEY=old<(:)``, ``KEY=old>(:)``) is part of that
        word, though, not a redirection: taken for ``<``, the word
        ended before it and the scrub left ``export<(:) …``, a command
        the shell does not find (exit 127), losing the other key of
        the line; ``< <(:)`` is still the redirection ``<`` and its
        target.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        rc = Path.home() / ".zshrc"
        rc.write_text(
            "export OPENAI_API_KEY=old; export ANTHROPIC_API_KEY=keep-me\n"
            "export KEEP_FIRST=1; export OPENAI_API_KEY='a;b'; # note\n"
            "export OPENAI_API_KEY='x;y'\n"
            "export OPENAI_API_KEY=old && export GEMINI_API_KEY=keep-and\n"
            "export GEMINI_API_KEY=keep-first OPENAI_API_KEY=old TOGETHER_API_KEY=keep-last\n"
            'printf "%s\\n" $(printf token; export OPENAI_API_KEY=local)\n'
            "true || export OPENAI_API_KEY=old\n"
            "if true; then export OPENAI_API_KEY=old; fi\n"
            "export OPENAI_API_KEY=old 2>/dev/null\n"
            "export OPENAI_API_KEY=old GEMINI_API_KEY=keep-redirected 2> /dev/null\n"
            "export OPENAI_API_KEY=old 2>&1\n"
            "export OPENAI_API_KEY=old &>/dev/null\n"
            "export OPENAI_API_KEY=old 2>&1 GEMINI_API_KEY=keep-duplicated\n"
            "export OPENAI_API_KEY=old<(:) ANTHROPIC_API_KEY=keep-substituted-in\n"
            "export OPENAI_API_KEY=old>(:) TOGETHER_API_KEY=keep-substituted-out\n"
            "export OPENAI_API_KEY=old OPENROUTER_API_KEY=keep-redirected-in < <(:)\n"
            "export OPENAI_API_KEY=old << EOF\n"
            "printf leaked-document\n"
            "EOF\n"
            "export OPENAI_API_KEY=old <<- EOF\n"
            "\tprintf leaked-document\n"
            "\tEOF\n"
            "export OPENAI_API_KEY=old<<EOF\n"
            "printf leaked-document\n"
            "EOF\n"
            "export OPENAI_API_KEY=old\\>& printf 'retained-background\\n'\n"
            "wait\n"
            'printf "%s\\n" "$GEMINI_API_KEY" "$ANTHROPIC_API_KEY" "$TOGETHER_API_KEY" '
            '"$OPENROUTER_API_KEY"\n'
        )
        save_api_key("OPENAI_API_KEY", "")
        assert rc.read_text() == (
            "export ANTHROPIC_API_KEY=keep-me\nexport KEEP_FIRST=1; # note\n"
            "true && export GEMINI_API_KEY=keep-and\n"
            "export GEMINI_API_KEY=keep-first TOGETHER_API_KEY=keep-last\n"
            'printf "%s\\n" $(printf token; export OPENAI_API_KEY=local)\n'
            "true || true\n"
            "if true; then true; fi\n"
            "export GEMINI_API_KEY=keep-redirected 2> /dev/null\n"
            "export 2>&1 GEMINI_API_KEY=keep-duplicated\n"
            "export ANTHROPIC_API_KEY=keep-substituted-in\n"
            "export TOGETHER_API_KEY=keep-substituted-out\n"
            "export OPENROUTER_API_KEY=keep-redirected-in < <(:)\n"
            "true << EOF\n"
            "printf leaked-document\n"
            "EOF\n"
            "true <<- EOF\n"
            "\tprintf leaked-document\n"
            "\tEOF\n"
            "true<<EOF\n"
            "printf leaked-document\n"
            "EOF\n"
            "true & printf 'retained-background\\n'\n"
            "wait\n"
            'printf "%s\\n" "$GEMINI_API_KEY" "$ANTHROPIC_API_KEY" "$TOGETHER_API_KEY" '
            '"$OPENROUTER_API_KEY"\n'
        )
        bash = _resolve_shell_path("bash")
        if bash is not None:
            syntax = subprocess.run([bash, "-n", str(rc)], capture_output=True, text=True)
            assert syntax.returncode == 0, syntax.stderr
            # Sourced for real, the RC prints its one token, the output
            # of the command backgrounded after the escaped ``\>`` and
            # the keys kept beside the descriptor duplication, the two
            # process substitutions and the redirection from one,
            # nothing of the environment (no bare ``export`` was left
            # behind) and no here-document line run as a command.
            sourced = subprocess.run(
                [bash, "--noprofile", "--norc", "-c", 'source "$1"', "probe", str(rc)],
                capture_output=True,
                text=True,
                env={"HOME": str(Path.home()), "UNRELATED_SECRET": "dummy-secret"},
            )
            assert sourced.returncode == 0, sourced.stderr
            assert sourced.stdout == (
                "token\nretained-background\nkeep-duplicated\nkeep-substituted-in\n"
                "keep-substituted-out\nkeep-redirected-in\n"
            )

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            pytest.param(
                "if true; then\n  export OPENAI_API_KEY=old\nfi\n",
                "if true; then\n  true\nfi\n",
                id="then",
            ),
            pytest.param(
                "if false; then\n  :\nelse\n\n  # the key\n"
                "  export OPENAI_API_KEY=old # note\nfi\n",
                "if false; then\n  :\nelse\n\n  # the key\n  true # note\nfi\n",
                id="else, past a blank and a comment line",
            ),
            pytest.param(
                "setup() {\n\texport OPENAI_API_KEY=old; export OPENAI_API_KEY=older\n}\nsetup\n",
                "setup() {\n\ttrue\n}\nsetup\n",
                id="function body",
            ),
            pytest.param(
                "for x in 1; do\n  export OPENAI_API_KEY=old\n  export OPENAI_API_KEY=new\ndone\n",
                "for x in 1; do\n  true\ndone\n",
                id="do, both lines of the body",
            ),
            pytest.param(
                "true &&\n  export OPENAI_API_KEY=old\n",
                "true &&\n  true\n",
                id="and-list over two lines",
            ),
            pytest.param(
                "if true && \\\n   true; then\n  export OPENAI_API_KEY=old\nfi\n",
                "if true && \\\n   true; then\n  true\nfi\n",
                id="then of a condition of two lines",
            ),
            pytest.param(
                "cat <<EOF >/dev/null ||\ntext\nEOF\nexport OPENAI_API_KEY=old\n",
                "cat <<EOF >/dev/null ||\ntext\nEOF\ntrue\n",
                id="or-list across a here-document",
            ),
            pytest.param(
                "false && export OPENAI_API_KEY=old # note\n",
                "false && true # note\n",
                id="and-list ended by a comment",
            ),
            pytest.param(
                "false |\n  export OPENAI_API_KEY=old; echo after-the-pipeline\n",
                "false |\n  true; echo after-the-pipeline\n",
                id="pipeline over two lines, then a command",
            ),
            pytest.param(
                "if true; then\n  export OPENAI_API_KEY=old\n  export KEPT=1\nfi\n",
                "if true; then\n  true\n  export KEPT=1\nfi\n",
                id="the first command of a body",
            ),
            pytest.param(
                "if true; then\n  export KEPT=1\n  export OPENAI_API_KEY=old\nfi\n",
                "if true; then\n  export KEPT=1\nfi\n",
                id="a later command of a body",
            ),
        ],
    )
    def test_delete_leaves_a_command_where_the_shell_needs_one(
        self, before: str, after: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The only command of a body is replaced by ``true``, not removed.

        The export on a line of its own was deleted with its line.  As
        the one command between ``then`` and ``fi`` (``do`` and
        ``done``, the braces of a function, after an ``&&`` ending the
        line before) that left ``then`` followed by ``fi``: a syntax
        error for bash, which stops reading the RC there, so saving or
        deleting a key in the settings panel cost the user every line
        of their ``.bashrc`` after it.  Where the command before the
        line still awaits one, the emptied line keeps a ``true`` (and
        its comment); after another command of the body the line just
        goes.
        """
        monkeypatch.setenv("SHELL", f"/bin/{_installed_posix_shell()}")
        rcs = {shell: _shell_rc_path(shell) for shell in ("bash", "zsh")}
        for rc in rcs.values():
            rc.write_text(before + "echo rest-of-the-rc\n")
        save_api_key("OPENAI_API_KEY", "")
        # The real shell still runs every command the RC kept, and holds no key.
        printed = "after-the-pipeline\n" if "after-the-pipeline" in before else ""
        for shell, rc in rcs.items():
            assert rc.read_text() == after + "echo rest-of-the-rc\n"
            if _resolve_shell_path(shell) is not None:
                assert _sourced(shell, rc) == printed + "rest-of-the-rc\nkey=unset\n"

    def test_delete_refuses_an_appended_value_of_several_lines(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``KEY+=`` in a command of several lines is refused as ``KEY=`` is.

        Only ``KEY=`` counted as an assignment no edit is safe for: the
        deletion was reported as done and the RC went on appending to
        the key in every new shell.
        """
        rc = _shell_rc_path("bash")
        text = 'export OPENAI_API_KEY+="two\nlines"\n'
        rc.write_text(text)
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setenv("OPENAI_API_KEY", "kept")
        with pytest.raises(OSError, match="OPENAI_API_KEY"):
            save_api_key("OPENAI_API_KEY", "")
        assert rc.read_text() == text
        kept_key = os.environ["OPENAI_API_KEY"]
        assert kept_key == "kept"

    @pytest.mark.parametrize(
        ("text", "line"),
        [
            pytest.param(
                "set -gx OPENAI_API_KEY 'one\ntwo'\necho remainder\n", 1,
                id="quoted value of two lines",
            ),
            pytest.param(
                "echo first\nset -gx OPENAI_API_KEY \\\n  continued\necho remainder\n", 2,
                id="continued line",
            ),
            pytest.param(
                "echo 'To set the key run\nset -gx OPENAI_API_KEY value\nand restart'\n", 1,
                id="quoted help text",
            ),
            pytest.param(
                "set -gx OPENAI_API_KEY (\n  cat /dev/null\n)\n", 1,
                id="command substitution of several lines",
            ),
            pytest.param(
                "true &&\n\nset -gx OPENAI_API_KEY value\n", 1,
                id="list continued past a blank line",
            ),
            pytest.param(
                "set -gx OPENAI_API_KEY 'it\\'s'; echo kept\n", 1,
                id="escaped quote in a quoted value",
            ),
            pytest.param(
                "set -gx OPENAI_API_KEY a&b; echo kept\n", 1, id="ampersand inside a word"
            ),
            pytest.param(
                "echo a \\\n  # note\n  b\nset -gx OPENAI_API_KEY value\n", 1,
                id="after a comment inside a continued command",
            ),
            pytest.param(
                "echo )\nset -gx OPENAI_API_KEY value\n", 1,
                id="after a bracket that closes nothing",
            ),
            pytest.param("set -gx OPENAI_API_KEY 'open\n", 1, id="quote left open"),
        ],
    )
    def test_delete_refuses_a_fish_command_no_line_edit_is_safe_for(
        self, text: str, line: int, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fish command of several lines that names the key is refused, not cut in two.

        fish's RC was edited line by line, with no regard for what a
        quote, a bracket, a backslash or an ``&&`` keeps open across
        the line's end: deleting the key removed the first line of its
        command and left the rest (``two'``, an unmatched quote fish
        rejects the whole file for; ``continued``, run as a command),
        and cut a line that only looked like the assignment out of a
        quoted text.  A line holding what fish reads differently from
        the POSIX shells (``\\'`` inside quotes, an ``&`` inside a
        word) lost the commands after it.  Each is refused as the same
        command is for bash and zsh, and nothing is written.  So is a
        key named after a text this reading does not follow fish
        through (a comment line inside a continued command) or that
        fish rejects (a bracket that closes nothing, a quote left open).
        """
        rc = _shell_rc_path("fish")
        rc.parent.mkdir(parents=True)
        rc.write_text(text, newline="")
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        monkeypatch.setenv("OPENAI_API_KEY", "kept")
        with pytest.raises(OSError, match=rf"config\.fish: line {line} may set OPENAI_API_KEY"):
            save_api_key("OPENAI_API_KEY", "")
        assert rc.read_text() == text
        kept_key = os.environ["OPENAI_API_KEY"]
        assert kept_key == "kept"

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            pytest.param(
                "echo 'two\nlines'\nset -gx OPENAI_API_KEY 'old key' # note\necho rest\n",
                "echo 'two\nlines'\n# note\necho rest\n",
                id="after a quoted text of two lines",
            ),
            pytest.param(
                "echo it\\'s # don't\nset -gx OPENAI_API_KEY old\n",
                "echo it\\'s # don't\n",
                id="after an escaped quote and one in a comment",
            ),
            pytest.param(
                "if true\n    set -gx OPENAI_API_KEY old\nend\n",
                "if true\nend\n",
                id="only command of a block",
            ),
            pytest.param(
                "set -q OPENAI_API_KEY; or set -gx OPENAI_API_KEY old\n",
                "set -q OPENAI_API_KEY; or true\n",
                id="after a query",
            ),
            pytest.param(
                "echo \"$(echo 'a\"b')\" (echo ')')\nset -gx OPENAI_API_KEY (cat /dev/null)\n",
                "echo \"$(echo 'a\"b')\" (echo ')')\n",
                id="command substitutions closed on their line",
            ),
            pytest.param(
                "echo $PATH[1] >/dev/null\nset -gx OPENAI_API_KEY old",
                "echo $PATH[1] >/dev/null\n",
                id="after an index, on a last line with no line feed",
            ),
            pytest.param(
                "function run_model\n    command docker run -e OPENAI_API_KEY \\\n"
                "        --rm example\nend\nset -gx OPENAI_API_KEY old\n",
                "function run_model\n    command docker run -e OPENAI_API_KEY \\\n"
                "        --rm example\nend\n",
                id="beside a command of two lines that passes the name on",
            ),
        ],
    )
    def test_delete_edits_the_fish_lines_that_are_commands_of_their_own(
        self, before: str, after: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """What fish closes on the line it opened it on keeps no later line from the edit.

        Nor does a command of two lines that names the key without
        assigning it (``docker run -e KEY \\``): it was refused.
        """
        rc = _shell_rc_path("fish")
        rc.parent.mkdir(parents=True)
        rc.write_text(before)
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        save_api_key("OPENAI_API_KEY", "")
        assert rc.read_text() == after
        fish = _resolve_shell_path("fish")
        if fish is not None:
            # The real fish still reads the whole RC (it runs none of a file
            # it cannot parse) and holds no key.
            parsed = subprocess.run([fish, "--no-config", "--no-execute", str(rc)], timeout=30)
            assert parsed.returncode == 0
            assert _sourced("fish", rc).endswith("key=unset\n")

    @pytest.mark.parametrize(
        ("shells", "assignment"),
        [
            pytest.param("bash zsh", "typeset -x OPENAI_API_KEY=legacy", id="typeset"),
            pytest.param("bash zsh", "declare -x OPENAI_API_KEY=legacy", id="declare"),
            pytest.param("bash zsh", "readonly OPENAI_API_KEY=legacy", id="readonly"),
            pytest.param("bash zsh", "builtin export OPENAI_API_KEY=legacy", id="builtin"),
            pytest.param("bash zsh", "\\export OPENAI_API_KEY=legacy", id="escaped command"),
            pytest.param("bash zsh", 'export "OPENAI_API_KEY=legacy"', id="quoted operand"),
            pytest.param("bash zsh", "eval 'export OPENAI_API_KEY=legacy'", id="eval"),
            pytest.param("bash zsh", "read -r OPENAI_API_KEY <<< legacy", id="read"),
            pytest.param("bash zsh", ': "${OPENAI_API_KEY:=legacy}"', id="default expansion"),
            pytest.param(
                "bash zsh", "if true; then typeset -x OPENAI_API_KEY=legacy; fi", id="under then"
            ),
            pytest.param("fish", "set -x OPENAI_API_KEY legacy", id="fish, set -x"),
            pytest.param("fish", "set --global --export OPENAI_API_KEY legacy", id="fish, long"),
            pytest.param("fish", "echo legacy | read -gx OPENAI_API_KEY", id="fish, read"),
            pytest.param(
                "bash zsh", "for OPENAI_API_KEY in legacy; do :; done", id="loop variable"
            ),
            pytest.param(
                "zsh", "if [[ -n 1 ]] export OPENAI_API_KEY=legacy", id="zsh, after a test"
            ),
            pytest.param(
                "zsh", "{ : } always { typeset -x OPENAI_API_KEY=legacy }", id="zsh, after a }"
            ),
            pytest.param(
                "fish", "echo (set -gx OPENAI_API_KEY legacy) >/dev/null",
                id="fish, command substitution",
            ),
            pytest.param(
                "bash zsh", "echo $'it\\'s' >/dev/null; typeset -x OPENAI_API_KEY=legacy",
                id="after an escaped quote in $'…'",
            ),
            pytest.param(
                "bash", "echo ${x:-(} >/dev/null; typeset -x OPENAI_API_KEY=legacy",
                id="bash, after a parenthesis in ${…}",
            ),
            pytest.param(
                "bash zsh", "x=$(\n  true # it's\n); read -r OPENAI_API_KEY <<< legacy",
                id="after a comment in a command substitution of several lines",
            ),
            pytest.param(
                "bash", "mapfile -t OPENAI_API_KEY <<< legacy", id="bash, mapfile",
                marks=pytest.mark.skipif(
                    _BASH_VERSION < (4, 0), reason="mapfile was added in Bash 4",
                ),
            ),
            pytest.param("zsh", "print -v OPENAI_API_KEY legacy", id="zsh, print -v"),
            pytest.param("bash zsh", "(( OPENAI_API_KEY = 5 ))", id="arithmetic"),
            pytest.param(
                "bash zsh", "x=export; $x OPENAI_API_KEY=legacy", id="command named by a variable"
            ),
            pytest.param(
                "bash zsh", 'export "$unset"OPENAI_API_KEY=legacy',
                id="operand after an empty expansion",
            ),
            pytest.param(
                "bash zsh", '"$unset": ${OPENAI_API_KEY:=legacy}',
                id="default expansion, command named by an expansion",
            ),
            pytest.param(
                "bash zsh", "export OPENAI_API_KEY=legacy $unset",
                id="export beside a word that may expand to nothing",
            ),
            pytest.param(
                "zsh", "{read -r OPENAI_API_KEY <<< legacy\n}", id="zsh, { glued to the command"
            ),
            pytest.param(  # zsh does not split the expansion into words: it runs nothing
                "bash", "${unset:-export OPENAI_API_KEY=legacy}",
                id="bash, command a default expansion yields",
            ),
            pytest.param(
                "bash", "echo $${ >/dev/null & read -r OPENAI_API_KEY <<< legacy",
                id="bash, after the process id before a {",
            ),
            pytest.param(
                "bash", "shopt -s nullglob; export OPENAI_API_KEY=legacy no_such_file_*",
                id="bash, export beside a glob that may match nothing",
            ),
            pytest.param(
                "zsh", "setopt nullglob; export OPENAI_API_KEY=legacy no_such_file_*",
                id="zsh, export beside a glob that may match nothing",
            ),
            pytest.param(
                "bash", "export OPENAI_API_KEY=legacy {,}", id="bash, export beside empty braces"
            ),
            pytest.param(
                "zsh", "export OPENAI_API_KEY=legacy no_such_file(N)",
                id="zsh, export beside a null glob qualifier",
            ),
            pytest.param(
                "bash", "OPENAI_API_KEY=legacy export OPENAI_API_KEY",
                id="bash, assignment before export",
            ),
            pytest.param(
                "bash", "OPENAI_API_KEY=legacy command declare -x OPENAI_API_KEY",
                id="bash, assignment before a declaration",
            ),
            pytest.param(
                "zsh", "export OPENAI_API_KEY; OPENAI_API_KEY=legacy exec",
                id="zsh, assignment before exec",
            ),
            pytest.param(
                "bash", "export OPENAI_API_KEY; printf -vOPENAI_API_KEY %s legacy",
                id="bash, printf -v glued to the name",
            ),
            pytest.param(
                "zsh", "print -vOPENAI_API_KEY legacy", id="zsh, print -v glued to the name"
            ),
            pytest.param(
                "bash", "read -raOPENAI_API_KEY <<< legacy", id="bash, read -a glued to the name"
            ),
            pytest.param(
                "zsh", "read 'OPENAI_API_KEY?key: ' <<< legacy", id="zsh, read with a prompt"
            ),
            pytest.param("bash", "OPENAI_API_KEY[0]=legacy", id="bash, an element of the key"),
            pytest.param("zsh", "OPENAI_API_KEY[1]=legacy", id="zsh, an element of the key"),
            pytest.param(
                "zsh", "OPENAI_API_KEY[1]=legacy true", id="zsh, an element before a command"
            ),
            pytest.param(
                "bash",
                'OPENAI_API_KEY[0]=legacy x="a\nb"',
                id="bash, an element beside a quote of two lines",
            ),
            pytest.param(
                "bash",
                '"$unset"printf -vOPENAI_API_KEY %s legacy',
                id="bash, a writer after an empty expansion",
            ),
            pytest.param(
                "bash zsh",
                "setopt nullglob 2>/dev/null || shopt -s nullglob;"
                " OPENAI_API_KEY=legacy no_such_command_*",
                id="before a command word that expands to none",
            ),
            pytest.param(
                "bash zsh",
                "export KEPT=1 OPENAI_API_KEY=old {,}OPENAI_API_KEY=legacy",
                id="an operand braces put the assignment in",
            ),
            pytest.param(
                "zsh", "true &!read -r OPENAI_API_KEY <<< legacy", id="zsh, after the &! operator"
            ),
            pytest.param(
                "bash zsh",
                "export KEPT=1 OPENAI_API_KEY=legacy $unset",
                id="export beside a kept operand and an expansion",
            ),
        ],
    )
    def test_delete_refuses_an_assignment_it_cannot_remove(
        self, shells: str, assignment: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A key some other command of the RC assigns is not reported as deleted.

        Only ``export KEY=…`` and a plain ``KEY=…`` (fish: ``set -gx
        KEY …``) are edited.  With ``typeset -x KEY=legacy`` in the RC
        the deletion cleared the store and reported success while every
        new shell went on exporting the key.  A command that assigns the
        key and that no edit reaches is refused before anything is
        written, naming the file and the line to remove by hand.
        So is one on a line the edit read otherwise than the shell
        (after ``$'it\\'s'``, a ``(`` inside ``${…}``, a comment inside a
        command substitution of several lines): the ``typeset`` there
        went unseen.  An ``export`` the edit would leave only words to
        that may expand to none (a glob under ``nullglob``, ``{,}``,
        zsh's ``(N)``) is refused too: bare, it prints every exported
        variable at each shell start.  One such operand is enough, also
        beside a kept one: zsh reports a glob of it that fails only
        once no assignment stands before it, whatever the command
        redirects.  So is the assignment before a
        declaration that names the key (bash keeps and exports it) or
        before zsh's ``exec``, the name glued to the option of a
        builtin that assigns (``printf -vKEY``, ``read -aKEY``) and the
        assignment of an element (``KEY[0]=…``, which zsh keeps before
        a command too).  An assignment also stays when the command word
        after it expands to none, and a writer runs when the expansion
        glued before its name is empty.
        """
        text = f"echo first >/dev/null\n{assignment}\n"
        set_to = "5" if assignment.startswith("((") else "legacy"
        store = api_keys_env_path()
        store.parent.mkdir(parents=True, exist_ok=True)
        for shell in shells.split():
            rc = _shell_rc_path(shell)
            rc.parent.mkdir(parents=True, exist_ok=True)
            rc.write_text(text, newline="")
            store.write_text("export OPENAI_API_KEY=stored\n")
            monkeypatch.setenv("SHELL", f"/bin/{shell}")
            monkeypatch.setenv("OPENAI_API_KEY", "stored")
            refusal = rf"{re.escape(str(rc))}: line 2 may set OPENAI_API_KEY .*by hand"
            for value in ("", "replacement"):
                with pytest.raises(OSError, match=refusal):
                    save_api_key("OPENAI_API_KEY", value)
                assert rc.read_text() == text
                assert store.read_text() == "export OPENAI_API_KEY=stored\n"
                stored_key = os.environ["OPENAI_API_KEY"]
                assert stored_key == "stored"
            if _resolve_shell_path(shell) is not None:
                # The refusal is no false alarm: the real shell does set the key.
                assert _sourced(shell, rc) == f"key={set_to}\n"
            rc.unlink()

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            pytest.param("export -- OPENAI_API_KEY=old\n", "", id="option, only operand"),
            pytest.param(
                "export -- OPENAI_API_KEY=old KEPT=1\n", "export -- KEPT=1\n", id="option"
            ),
            pytest.param(
                "OPENAI_API_KEY=old\nexport OPENAI_API_KEY\n",
                "export OPENAI_API_KEY\n",
                id="assigned, then exported",
            ),
            pytest.param("KEPT=1 OPENAI_API_KEY=old\n", "KEPT=1\n", id="second assignment"),
            pytest.param(
                "KEPT[1]=1 OPENAI_API_KEY=old\n", "KEPT[1]=1\n", id="after an element's assignment"
            ),
            pytest.param(
                'ai() { OPENAI_API_KEY="$(cat k)" "$HOME/bin/ai" -e OPENAI_API_KEY "$@"; }\n'
                "export OPENAI_API_KEY=old\n",
                'ai() { OPENAI_API_KEY="$(cat k)" "$HOME/bin/ai" -e OPENAI_API_KEY "$@"; }\n',
                id="beside a wrapper that runs a program by its path",
            ),
            pytest.param(
                'with_key() { OPENAI_API_KEY="$(cat k)" "$@"; }\nexport OPENAI_API_KEY=old\n',
                'with_key() { OPENAI_API_KEY="$(cat k)" "$@"; }\n',
                id="beside a wrapper that gives the key to its arguments",
            ),
            pytest.param(
                "export AZURE_OPENAI_API_KEY=kept OPENAI_API_KEY=old\n",
                "export AZURE_OPENAI_API_KEY=kept\n",
                id="beside a longer name that ends in the key's",
            ),
            pytest.param(
                "OPENAI_API_KEY=old KEPT=1; OPENAI_API_KEY=older\n",
                "KEPT=1\n",
                id="first assignment",
            ),
            pytest.param("export OPENAI_API_KEY+=more\n", "", id="appended"),
            pytest.param(
                "case x in\n  x) export OPENAI_API_KEY=old ;;\nesac\n",
                "case x in\n  x) true ;;\nesac\n",
                id="case clause",
            ),
            pytest.param(
                "case x in x) export OPENAI_API_KEY=old;; esac\n",
                "case x in x) true;; esac\n",
                id="case on one line",
            ),
            pytest.param(
                "case x in\n  (x) export OPENAI_API_KEY=old ;;\nesac\n",
                "case x in\n  (x) true ;;\nesac\n",
                id="case clause, pattern in parentheses",
            ),
            pytest.param(
                "setup() { export OPENAI_API_KEY=old; }; setup\n",
                "setup() { true; }; setup\n",
                id="function on one line",
            ),
            pytest.param(
                "function setup { export OPENAI_API_KEY=old; }; setup\n",
                "function setup { true; }; setup\n",
                id="function keyword on one line",
            ),
            pytest.param(">/dev/null export OPENAI_API_KEY=old\n", "", id="redirection first"),
            pytest.param(
                "OPENAI_API_KEY=old <<EOF\ntext\nEOF\n",
                "true <<EOF\ntext\nEOF\n",
                id="assignment with a here-document",
            ),
            pytest.param(
                "OPENAI_API_KEY=only-for printf ''\n",
                "OPENAI_API_KEY=only-for printf ''\n",
                id="the environment of one command",
            ),
            pytest.param(
                "env OPENAI_API_KEY=only-for true\n",
                "env OPENAI_API_KEY=only-for true\n",
                id="the environment env gives one command",
            ),
            pytest.param(
                "alias with_key='OPENAI_API_KEY=only-for true'\n",
                "alias with_key='OPENAI_API_KEY=only-for true'\n",
                id="the text of an alias",
            ),
            pytest.param(
                "echo 'run: export OPENAI_API_KEY=sk-1 yourself' >/dev/null # or typeset -x\n",
                "echo 'run: export OPENAI_API_KEY=sk-1 yourself' >/dev/null # or typeset -x\n",
                id="the text of a message",
            ),
            pytest.param(
                "echo $'it\\'s' >/dev/null; export OPENAI_API_KEY=old\n",
                "echo $'it\\'s' >/dev/null\n",
                id="after an escaped quote in $'…'",
            ),
            pytest.param(
                "echo $'\\';export OPENAI_API_KEY=sk-1;\\'' >/dev/null\n",
                "echo $'\\';export OPENAI_API_KEY=sk-1;\\'' >/dev/null\n",
                id="the text of a message in $'…'",
            ),
            pytest.param(
                "(true)#; export OPENAI_API_KEY=sk-1\n",
                "(true)#; export OPENAI_API_KEY=sk-1\n",
                id="a comment after a closing parenthesis",
            ),
            pytest.param(
                "echo hi >/dev/null |\n;export OPENAI_API_KEY=old\n",
                "echo hi >/dev/null |\ntrue\n",
                id="after a ; that opens the line a pipe awaits",
            ),
            pytest.param(
                "echo hi >/dev/null |; export OPENAI_API_KEY=old\n",
                "echo hi >/dev/null |; true\n",
                id="after a ; that ends a pipe",
            ),
            pytest.param(
                "!<<<x OPENAI_API_KEY=old\n", "! true\n", id="after a ! glued to a redirection"
            ),
            pytest.param("OPENAI_API_KEY=old$()\n", "", id="value ending in an empty $()"),
            pytest.param(
                "export OPENAI_API_KEY=old \\\n", "\\\n", id="before a backslash that ends the text"
            ),
            pytest.param(
                "if false; then { :; } else export OPENAI_API_KEY=old; fi\n",
                "if false; then { :; } else true; fi\n",
                id="after a closing brace and else",
            ),
            pytest.param(
                "run_model() {\n  command docker run -e OPENAI_API_KEY \\\n    --rm example\n}\n"
                "export OPENAI_API_KEY=old\n",
                "run_model() {\n  command docker run -e OPENAI_API_KEY \\\n    --rm example\n}\n",
                id="beside a command of two lines that passes the name on",
            ),
            pytest.param(
                "for name in OPENAI_API_KEY OTHER_KEY; do [[ -v $name ]]; done\n"
                "export OPENAI_API_KEY=old\n",
                "for name in OPENAI_API_KEY OTHER_KEY; do [[ -v $name ]]; done\n",
                id="beside a loop over the names",
            ),
        ],
    )
    def test_delete_reaches_every_spelling_of_the_assignment(
        self, before: str, after: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An assignment goes however it is spelled, and nothing else does.

        Only ``export KEY=…`` as the first words of a command went.  An
        ``export --`` lost its operand and was left bare, printing every
        exported variable at each shell start; a plain ``KEY=…``, an
        ``export KEY+=…``, the export of a ``case`` clause or of a
        function written on one line (``name()`` or ``function name``)
        and one after a redirection stayed in the RC of a key reported
        as deleted.  The assignment before
        a command (``KEY=x command``) is that command's environment
        alone and stays, as does what a command that assigns nothing
        is given (``env``, the text of an alias or a message): none of
        them is refused either, nor is a command that only passes the
        name on, also over two lines (``docker run -e KEY \\``), or a
        loop over the names.  The export after ``} else`` was left
        where bash runs it.
        """
        monkeypatch.setenv("SHELL", f"/bin/{_installed_posix_shell()}")
        rcs = {shell: _shell_rc_path(shell) for shell in ("bash", "zsh")}
        for rc in rcs.values():
            rc.write_text(before)
        save_api_key("OPENAI_API_KEY", "")
        for shell, rc in rcs.items():
            assert rc.read_text() == after
            if _resolve_shell_path(shell) is not None:
                # Nothing printed, and no key (zsh's ``export KEY`` sets an empty one).
                assert _sourced(shell, rc) in ("key=unset\n", "key=\n")

    @pytest.mark.parametrize(
        "assignment",
        [
            pytest.param("GEMINI_API_KEY=from-the-rc\n", id="plain assignment"),
            pytest.param(
                "GEMINI_API_KEY=from-the-rc\nexport GEMINI_API_KEY\n", id="assigned, then exported"
            ),
            pytest.param(
                'set_key() { export "$@"; }\nset_key GEMINI_API_KEY=from-the-rc\n',
                id="a function of the RC's own",
            ),
        ],
    )
    def test_delete_survives_reload_whatever_the_rc_holds(
        self, assignment: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No line of the RC brings a deleted key back at the next daemon start.

        The migration imports a plain ``KEY=…`` line as it imports an
        ``export``, but the deletion scrubbed exports only: the key the
        panel had just deleted was read from the RC again at the next
        start.  A spelling no edit reaches and no refusal knows (a
        function of the RC's own that exports its arguments) came back
        through the one-time sourcing of the RC, which a store that
        never needed it had not marked as done.
        """
        shell = _installed_posix_shell()
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        _shell_rc_path(shell).write_text(assignment)
        save_api_key("GEMINI_API_KEY", "from-the-panel")
        save_api_key("GEMINI_API_KEY", "")
        load_api_keys()
        assert "GEMINI_API_KEY" not in os.environ
        store = api_keys_env_path()
        assert "GEMINI_API_KEY=" not in store.read_text()
        # The "sourced" marker is written once, also when a hand edit
        # left it as a last line without its newline.
        store.write_text(store.read_text().rstrip("\n"))
        save_api_key("GEMINI_API_KEY", "")
        assert store.read_text().count(user_config._RC_SOURCED_MARKER) <= 1

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

    @pytest.mark.parametrize(
        ("path", "shell"),
        [
            ("/home/fisher/bin/bash", "bash"),
            ("/home/fisher/.nix-profile/bin/zsh", "zsh"),
            ("/opt/zsh-tools/bin/fish", "fish"),
            ("/nix/store/abc-zsh-5.9/bin/zsh-5.9", "zsh"),
        ],
    )
    def test_the_binary_names_the_shell_not_its_directories(
        self, path: str, shell: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A directory of ``$SHELL`` that spells a shell's name names no shell.

        The whole path was searched, so the bash of a user called
        ``fisher`` was taken for fish: no hook was written to the
        ``.bashrc`` (fish gets none) and the migration sourced a fish
        the user does not run.
        """
        monkeypatch.setenv("SHELL", path)
        assert _get_user_shell() == shell


class TestResolveShellPath:
    """Test absolute shell binary resolution."""

    @posix_only("looks up sh/bash in /usr/bin:/bin")
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
            "export KISS_TEST_DOLLAR='lit-$eral'\n"
            "export KISS_TEST_TILDE=~/keys/token.json\n"
            "export KISS_TEST_TILDES=first:~/second\n"
            "export KISS_TEST_BRACES=a{b,c}\n"
            "export KISS_TEST_ARRAY=(one)\n"
            "export KISS_TEST_COMMAND==ls\n"
            'export KISS_TEST_TILDE_QUOTED=""~/token\n'
            "export KISS_TEST_CONTINUED=joined-with-the-next-line\\\n"
            "export KISS_TEST_BACKGROUND=https://host/path?first=1&second=2\n"
            "export KISS_TEST_PIPED=one|two\n"
            "export KISS_TEST_LITERAL=a~b:'~/c':\\~d{e}'{f,g}'\n"
            "export KISS_TEST_BLANK=ends-in-a-blank\\ \t \n"
            "export KISS_TEST_LISTED=listed && true\n"
            "export KISS_TEST_REDIRECTED=redirected &>/dev/null\n",
        )
        load_api_keys()
        assert os.environ["PATH"] == old_path
        assert "KISS_TEST_EXPAND" not in os.environ
        assert os.environ.get("KISS_TEST_DOLLAR") == "lit-$eral"
        del os.environ["KISS_TEST_DOLLAR"]
        # A tilde the shell expands (the value's start, after a ``:``) is
        # no literal either (``~/keys/token.json`` names no file), nor
        # are the braces bash expands, an array (bash assigns ``one``),
        # zsh's tilde after empty quotes, or a line the shell joins
        # with the next; an unquoted ``&`` or ``|`` runs the assignment
        # where the shell keeps none (half a URL is no value).  A blank
        # a backslash escapes is the value's, not a blank that ends the
        # line, a value that opens with ``=`` (unquoted in the store of
        # a previous release) loads as bash reads it, and ``&&`` and a
        # redirection leave the assignment its value.
        skipped = (
            "KISS_TEST_TILDE", "KISS_TEST_TILDES", "KISS_TEST_BRACES", "KISS_TEST_ARRAY",
            "KISS_TEST_TILDE_QUOTED", "KISS_TEST_CONTINUED", "KISS_TEST_BACKGROUND",
            "KISS_TEST_PIPED",
        )
        loaded = {name: os.environ.pop(name, None) for name in skipped}
        assert loaded == dict.fromkeys(skipped)
        kept = {
            "KISS_TEST_LITERAL": "a~b:~/c:~d{e}{f,g}",
            "KISS_TEST_BLANK": "ends-in-a-blank ",
            "KISS_TEST_COMMAND": "=ls",
            "KISS_TEST_LISTED": "listed",
            "KISS_TEST_REDIRECTED": "redirected",
        }
        loaded = {name: os.environ.pop(name, None) for name in kept}
        assert loaded == kept

    @pytest.mark.parametrize(
        ("line", "value"),
        [
            ("OPENAI_API_KEY=value >/dev/null &", None),
            ("export OPENAI_API_KEY=value &>/dev/null &", None),
            ("OPENAI_API_KEY=value >/dev/null & true", None),
            ("export OPENAI_API_KEY=value && true &", None),
            ("OPENAI_API_KEY=value >/dev/null || true &", None),
            ("OPENAI_API_KEY=value >/dev/null | cat", None),
            ("export OPENAI_API_KEY=value 2>&1 |& cat", None),
            ("OPENAI_API_KEY=value >/dev/null; true &", "value"),
            ("export OPENAI_API_KEY=value 2>/dev/null && true '&' | cat", "value"),
            ("OPENAI_API_KEY=value >/dev/null # note &", "value"),
            ("export OPENAI_API_KEY=value || true | cat", "value"),
            ("OPENAI_API_KEY=value >/dev/null true", None),
            ("OPENAI_API_KEY=value true", None),
            ("OPENAI_API_KEY=value OTHER=1 true", None),
            ("OPENAI_API_KEY=value OTHER=1", "value"),
        ],
    )
    def test_load_skips_an_assignment_run_in_the_background_or_a_pipeline(
        self, line: str, value: str | None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A store line the shell runs in the background or in a pipeline loads nothing.

        The parser stopped reading at a redirection and at ``&&``: the
        ``&`` after ``KEY=value >/dev/null`` or ending ``KEY=value &&
        true`` went unseen and the value loaded as a credential, while
        bash runs such a list in a subshell and keeps no assignment (so
        does a pipeline).  A ``;`` ends the list first, and a quoted
        or commented ``&`` is no operator.  A command word after the
        redirection went unseen too: the assignment is that command's
        environment.  So it is with no redirection between them (``KEY=value
        command`` loaded ``value``), while more assignments keep it.
        """
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        store = api_keys_env_path()
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(line + "\n")
        load_api_keys()
        loaded = os.environ.get("OPENAI_API_KEY")  # a local: a failure prints no environment
        assert loaded == value
        # The store is bash syntax: bash decides.  zsh agrees on what loads
        # (of a list in the background it keeps all but the last pipeline).
        for shell in ("bash", "zsh"):
            if _resolve_shell_path(shell) is not None and (value or shell == "bash"):
                assert _sourced(shell, store) == f"key={value or 'unset'}\n"

    def test_migrates_legacy_rc_key_into_store(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A key only in the RC (old scheme) is imported and persisted, from a CRLF line too."""
        shell = _installed_posix_shell()
        rc = _shell_rc_path(shell)
        rc.write_bytes(b'export GEMINI_API_KEY="legacy-key"\nexport OPENAI_API_KEY=crlf-key\r\n')
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "legacy-key"
        assert "GEMINI_API_KEY=legacy-key" in api_keys_env_path().read_text()
        crlf_key = os.environ.get("OPENAI_API_KEY")
        assert crlf_key == "crlf-key"

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
        from kiss.core import user_config as vc
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

        The RC names the key but assembles its value from a variable, so
        the textual scan cannot import it and only the (unavailable)
        shell could.
        """
        rc = Path.home() / ".zshrc"
        rc.write_text('V=never\nexport GEMINI_API_KEY="$V-set"\n')
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from kiss.core import user_config as vc
        monkeypatch.setattr(
            vc, "_SHELL_FALLBACK_PATHS",
            {"zsh": (str(tmp_path / "no-such-zsh"),)},
        )
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") is None

    def test_the_users_shell_rc_wins_over_a_stale_key_in_another_shells_rc(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bash user's current ``.bashrc`` key migrates, not the ``.zshrc`` leftover.

        The migration scanned bash, zsh and fish in that fixed order and
        let each later RC overwrite what the earlier ones found, so a
        stale key in an abandoned shell's RC beat the current one of
        ``$SHELL``.  The user's shell is read first and the other RCs
        only fill what it does not set -- including a key whose
        assignment the text cannot import: a value assembled from a
        variable is resolved by sourcing that RC rather than replaced
        by the leftover, and an explicit ``KEY=""`` leaves the key
        unset instead of resurrecting the leftover into the store.
        """
        shell = _installed_posix_shell()
        other = "zsh" if shell == "bash" else "bash"
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        _shell_rc_path(shell).write_text(
            "export OPENAI_API_KEY=current-key\n"
            'V=current\nexport ANTHROPIC_API_KEY="$V-expanded"\n'
            'export TOGETHER_API_KEY=""\n'
        )
        _shell_rc_path(other).write_text(
            "export OPENAI_API_KEY=obsolete-key\nexport GEMINI_API_KEY=other-only\n"
            "export ANTHROPIC_API_KEY=obsolete-key\nexport TOGETHER_API_KEY=obsolete-key\n"
        )
        load_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == "current-key"
        assert os.environ.get("GEMINI_API_KEY") == "other-only"
        assert os.environ.get("ANTHROPIC_API_KEY") == "current-expanded"
        assert os.environ.get("TOGETHER_API_KEY") is None
        assert "obsolete-key" not in api_keys_env_path().read_text()

    def test_a_later_rc_assignment_wins_even_when_it_cannot_be_imported(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``export KEY=old`` followed by ``export KEY=""`` leaves the key unset in
        a shell, so the migration must not resurrect ``old`` into the store; the
        same holds for a later assignment that needs the shell to expand it."""
        shell = _installed_posix_shell()
        rc = _shell_rc_path(shell)
        rc.write_text(
            'export OPENAI_API_KEY=old\nexport OPENAI_API_KEY=""\n'
            'export GEMINI_API_KEY=stale\nG=later\nexport GEMINI_API_KEY="$G-key"\n'
        )
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        load_api_keys()
        store = api_keys_env_path().read_text()
        assert "old" not in store and os.environ.get("OPENAI_API_KEY") is None
        assert "stale" not in store and os.environ.get("GEMINI_API_KEY") == "later-key"

    def test_load_parses_quoting_edge_cases(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unbalanced quotes, empty values, and multi-word raws load honestly.

        A line with an unbalanced quote keeps its raw text; a multi-word
        remainder loads its first word (what a shell assigns for
        ``KEY=two words``); a bare ``KEY=`` loads as empty, and so does
        ``KEY= # removed`` (the value ends at the blank, the rest is a
        comment); ``\\$`` and ``\\``` inside double quotes lose their
        backslash as in a shell, ``#`` inside a word is no comment, and
        neither is one opening the value (``KEY=#literal`` continues the
        assignment's word: the shell assigns ``#literal``).  None of
        these may crash the loader or corrupt the rest of the file's
        imports.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        names = (
            "KISS_TEST_UNBALANCED", "KISS_TEST_EMPTY", "KISS_TEST_WORDS", "KISS_TEST_OK",
            "KISS_TEST_BLANK", "KISS_TEST_ESCAPED", "KISS_TEST_HASH", "KISS_TEST_NOTE",
            "KISS_TEST_LEADING_HASH", "KISS_TEST_SEMICOLON", "KISS_TEST_QUOTED_SEMICOLON",
            "KISS_TEST_REDIRECTED",
        )
        for name in names:
            monkeypatch.delenv(name, raising=False)
        env_file = api_keys_env_path()
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "export KISS_TEST_UNBALANCED=\"oops\n"
            "export KISS_TEST_EMPTY=\n"
            "export KISS_TEST_WORDS=two words\n"
            "export KISS_TEST_OK='fine'\n"
            "export KISS_TEST_BLANK= # removed\n"
            'export KISS_TEST_ESCAPED="sk-\\$lit\\`eral\\n"\n'
            "export KISS_TEST_HASH=abc#def\n"
            "export KISS_TEST_NOTE=abc # note\n"
            "export KISS_TEST_LEADING_HASH=#literal # note\n"
            "export KISS_TEST_SEMICOLON='sk-audit'; export OTHER=x\n"
            "export KISS_TEST_QUOTED_SEMICOLON='a;b'\n"
            "export KISS_TEST_REDIRECTED=sk-kept>/dev/null\n"
            "export KISS_TEST_TWICE=first KISS_TEST_TWICE=second\n",
        )
        load_api_keys()
        # A name a later word assigns again is the shell's: ``first`` is not what it holds.
        assert "KISS_TEST_TWICE" not in os.environ
        assert os.environ.get("KISS_TEST_UNBALANCED") == '"oops'
        assert os.environ.get("KISS_TEST_EMPTY") == ""
        assert os.environ.get("KISS_TEST_WORDS") == "two"
        assert os.environ.get("KISS_TEST_OK") == "fine"
        assert os.environ.get("KISS_TEST_BLANK") == ""
        assert os.environ.get("KISS_TEST_ESCAPED") == "sk-$lit`eral\\n"
        assert os.environ.get("KISS_TEST_HASH") == "abc#def"
        assert os.environ.get("KISS_TEST_NOTE") == "abc"
        assert os.environ.get("KISS_TEST_LEADING_HASH") == "#literal"
        # An unquoted ``;`` ends the assignment (the rest is another command).
        assert os.environ.get("KISS_TEST_SEMICOLON") == "sk-audit"
        assert os.environ.get("KISS_TEST_QUOTED_SEMICOLON") == "a;b"
        # So does an unquoted redirection: the hook sourcing the store assigns ``sk-kept``.
        assert os.environ.get("KISS_TEST_REDIRECTED") == "sk-kept"
        for name in names:
            del os.environ[name]
        # In fish the value is a word of its own, so a ``#`` opening it is a
        # comment: the RC assigns the key, but to nothing the store can hold.
        assert user_config._scan_rc_text_for_keys(
            "set -gx OPENAI_API_KEY #literal\nset -gx GEMINI_API_KEY g#1\n",
            "fish",
            {"OPENAI_API_KEY", "GEMINI_API_KEY"},
        ) == {"OPENAI_API_KEY": None, "GEMINI_API_KEY": "g#1"}

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
        from kiss.core.user_config import get_current_api_keys

        monkeypatch.setenv("GEMINI_API_KEY", "gem-from-env")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-from-env")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        keys = get_current_api_keys()
        assert keys["GEMINI_API_KEY"] == "gem-from-env"
        assert keys["OPENAI_API_KEY"] == "oai-from-env"
        assert keys["ANTHROPIC_API_KEY"] == ""

    def test_returns_empty_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All keys empty when none are set."""
        from kiss.core.user_config import get_current_api_keys

        for k in API_KEY_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        keys = get_current_api_keys()
        assert all(v == "" for v in keys.values())
        assert set(keys.keys()) == API_KEY_ENV_VARS

    def test_all_keys_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All expected API key names are included in the result."""
        from kiss.core.user_config import get_current_api_keys

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
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CODEX_ACCESS_TOKEN",
        }
        assert API_KEY_ENV_VARS == expected


class TestSaveConfigAtomicity:
    """``save_config`` must replace ``config.json`` atomically.

    Other processes (a second daemon, the CLI) read ``config.json``
    while the Python ``save_config`` may be writing it.  If the writer
    truncates the file before populating it (i.e. uses ``open(path, "w")``
    + incremental ``json.dump``), concurrent readers see an empty or
    half-written file and silently fall back to ``{}``, which then makes
    them prompt the user for a ``remote_password`` that is actually
    already set.
    """

    def test_concurrent_reader_never_sees_empty_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reader hammering the file during ``save_config`` calls
        must always observe a valid JSON object with ``remote_password``
        set — never an empty file, partial bytes, or ``JSONDecodeError``.
        """
        import threading
        import time

        from kiss.core import user_config as vc

        passwords = ("secret-1", "secret-2")
        save_config({"remote_password": passwords[0], "max_budget": 1})
        assert vc.CONFIG_PATH.exists()

        stop = threading.Event()
        bad_reads: list[str] = []

        def reader() -> None:
            while not stop.is_set():
                time.sleep(HOT_READER_PAUSE)
                try:
                    raw = vc.CONFIG_PATH.read_bytes()
                except FileNotFoundError:
                    bad_reads.append("FileNotFoundError")
                    continue
                except PermissionError:
                    # Windows makes the target briefly inaccessible while
                    # os.replace swaps it; that is neither empty nor partial.
                    if not IS_WINDOWS:
                        raise
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
                if pw not in passwords:
                    bad_reads.append(f"bad-pw:{pw!r}")

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        try:
            for i in range(200):
                save_config({
                    "remote_password": passwords[i % 2],
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
        stored = "sensca95"
        save_config({"remote_password": stored, "max_budget": 10})
        save_config({"max_budget": 25})
        cfg = load_config()
        assert cfg["remote_password"] == stored
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
        path = Path(user_config.CONFIG_PATH)
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
        assert "demo_mode" in user_config.RETIRED_KEYS
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
        path = Path(user_config.CONFIG_PATH)
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

        The scan reads every command of a line, not only its first:
        ``export A=old; export A=current; export B=…`` migrates
        ``current`` and ``B`` -- a first-command-only scan stored the
        obsolete ``old`` and lost ``B`` to the sourcing fallback, which
        the guard turns back before it reaches the line.

        A command that only names a key (a message, an existence test,
        ``env -u KEY`` in a function, ``docker run -e KEY`` over two
        lines) assigns nothing: each one left the key to that sourcing,
        and the literal above it was never imported.
        """
        rc = Path.home() / ".bashrc"
        rc.write_text(
            "case $- in\n"
            "    *i*) ;;\n"
            "      *) return;;\n"
            "esac\n"
            "export GEMINI_API_KEY=legacy-after-guard\n"
            "export OPENAI_API_KEY=old; export OPENAI_API_KEY=current;"
            " export ANTHROPIC_API_KEY=same-line\n"
            'printf "%s\\n" "GEMINI_API_KEY configured" >/dev/null\n'
            "without_key() { env -u OPENAI_API_KEY true; }\n"
            "[[ -v ANTHROPIC_API_KEY ]] && true\n"
            "run_model() {\n  command docker run -e GEMINI_API_KEY \\\n    --rm example\n}\n",
        )
        monkeypatch.setenv("SHELL", "/bin/bash")
        for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "legacy-after-guard"
        assert os.environ.get("OPENAI_API_KEY") == "current"
        assert os.environ.get("ANTHROPIC_API_KEY") == "same-line"
        store = api_keys_env_path().read_text()
        assert "legacy-after-guard" in store
        assert "OPENAI_API_KEY=old" not in store
        # Everything was read from the text: the shell was never spawned.
        assert user_config._RC_SOURCED_MARKER not in store

    def test_migration_leaves_conditional_assignments_to_the_shell(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An assignment the shell may skip, or run in a subshell, migrates as the shell says.

        The scan read every command of a line as an unconditional
        assignment: ``export A=current || export A=obsolete`` imported
        ``obsolete``, which the shell never assigns (the ``||`` branch
        of a succeeding export), ``false && export B=never`` imported
        ``never``, and an export piped into ``cat`` runs in a subshell
        that never reaches the sourcing shell.  Only a command bounded
        by ``;`` (an ``&`` before it, an ``&&`` or ``||`` after it) is
        the text's to read; the others are left to the one-time
        sourcing pass, which sees what the shell assigns.

        So is an assignment under a reserved word or in a ``case``
        clause (the scan did not see it, and imported the earlier value
        the shell had replaced) and a value the shell expands a tilde
        in, imported as the path ``~/…`` no program opens.  The
        assignment before a command is that command's environment: it
        replaced the value the RC exports when a redirection stood
        between the two.
        """
        shell = _installed_posix_shell()
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        names = (
            "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY", "ZAI_API_KEY", "MOONSHOT_API_KEY", "AWS_REGION",
        )
        for name in names:
            monkeypatch.delenv(name, raising=False)
        _shell_rc_path(shell).write_text(
            "export OPENAI_API_KEY=current || export OPENAI_API_KEY=obsolete\n"
            "false && export GEMINI_API_KEY=never-assigned\n"
            "export ANTHROPIC_API_KEY=pipeline-only | cat\n"
            "export TOGETHER_API_KEY=replaced\n"
            "if true; then export TOGETHER_API_KEY=under-then; fi\n"
            "export OPENROUTER_API_KEY=replaced\n"
            "case x in x) export OPENROUTER_API_KEY=in-a-clause;; esac\n"
            "export ZAI_API_KEY=~/zai-key\n"
            "export MOONSHOT_API_KEY=a:~/b\n"
            "export AWS_REGION=exported\n"
            "AWS_REGION=for-one-command >/dev/null printf ''\n"
        )
        load_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == "current"
        assert os.environ.get("GEMINI_API_KEY") is None
        assert os.environ.get("ANTHROPIC_API_KEY") is None
        assert os.environ.get("TOGETHER_API_KEY") == "under-then"
        assert os.environ.get("OPENROUTER_API_KEY") == "in-a-clause"
        assert os.environ.get("ZAI_API_KEY") == f"{Path.home()}/zai-key"
        assert os.environ.get("MOONSHOT_API_KEY") == f"a:{Path.home()}/b"
        assert os.environ.get("AWS_REGION") == "exported"
        store = api_keys_env_path().read_text()
        assert "OPENAI_API_KEY=current" in store
        assert "obsolete" not in store
        assert "never-assigned" not in store
        assert "pipeline-only" not in store
        # The text left them to the shell, which was sourced once.
        assert user_config._RC_SOURCED_MARKER in store

    @pytest.mark.parametrize(
        ("shell", "rest", "value", "from_text"),
        [
            pytest.param("sh", "if false; then\n  export K=bad\nfi\n", "good", False, id="if"),
            pytest.param("sh", "if true; then\n  export K=new\nfi\n", "new", False, id="if, run"),
            pytest.param(
                "sh", "set_key () {\n  export K=bad\n}\n", "good", False, id="function body"
            ),
            pytest.param(
                "sh", "for x in; do\n  export K=bad\ndone\n", "good", False, id="for"
            ),
            pytest.param(
                "sh", "case x in\n  y)\n    export K=bad\n    ;;\nesac\n", "good", False, id="case"
            ),
            pytest.param(
                "sh", "false && {\n  export K=bad\n}\n", "good", False, id="brace group"
            ),
            pytest.param(
                "sh", "if false; then :; export K=bad; fi\n", "good", False,
                id="on one line, after another command",
            ),
            pytest.param(
                "sh", "false &&\n  export K=bad\n", "good", False, id="after && on the line before"
            ),
            pytest.param(
                "sh", "typeset -x K=new\n", "new", False, id="a command no edit removes"
            ),
            pytest.param("sh", "unset K\n", "unset", False, id="unset"),
            pytest.param(
                "sh", "set_key () {\n  export K=late\n}\nexport K=early\nset_key\n", "late", False,
                id="function run after a later assignment",
            ),
            pytest.param(
                "sh",
                "if true; then\n  while false; do\n    export O=1\n  done\nfi\nexport K=last\n",
                "last", True, id="after compound commands that leave it alone",
            ),
            pytest.param(
                "fish", "if false\n    set -gx K bad\nend\n", "good", False, id="fish, if"
            ),
            pytest.param(
                "fish", "function set_other\n    set -gx O 1\nend\nset -gx K last\n", "last", True,
                id="fish, after a function that leaves it alone",
            ),
            pytest.param("sh", 'echo "$K" >/dev/null\n', "good", True, id="read, not assigned"),
            pytest.param(
                "bash", "echo } >/dev/null\nexport K=last\n", "last", True,
                id="bash, a closing brace that is an argument",
            ),
            pytest.param(
                "zsh", "repeat 0; do\n  export K=bad\ndone\n", "good", False,
                id="zsh, a loop the reading knows no opener of",
            ),
            pytest.param(
                "sh", "echo $'\\';export K=bad;\\'' >/dev/null\n", "good", True,
                id="the text of a message in $'…'",
            ),
            pytest.param(
                "sh", "without_key () {\n  env -u K true\n}\n", "good", True,
                id="function body that passes the name on",
            ),
            pytest.param("fish", "set -q K; or echo missing\n", "good", True, id="fish, a query"),
            pytest.param(
                "sh", "export K=last\nexport K\n", "last", True, id="export of the name alone"
            ),
            pytest.param(
                "sh",
                'ai() { K="$(cat k)" "$HOME/bin/ai" -e K "$@"; }\nexport K=last\n',
                "last",
                True,
                id="a wrapper that runs a program by its path",
            ),
            pytest.param(
                "sh",
                'with_key() { K="$(cat k)" "$@"; }\nexport K=last\n',
                "last",
                False,
                id="a wrapper that gives the key to its arguments, or keeps it without any",
            ),
            pytest.param(
                "sh", "readonly K\nexport K=bad\n", "good", False,
                id="read-only before a later assignment",
            ),
            pytest.param(
                "bash", "declare -i K\nexport K=7+1\n", "8", False,
                id="bash, an attribute that changes what is assigned",
            ),
            pytest.param(
                "sh", "(true)#; export K=bad\n", "good", True,
                id="a comment after a closing parenthesis",
            ),
            pytest.param(
                "sh", "export K=first >/dev/null K=second\n", "second", False,
                id="a second operand after a redirection",
            ),
            pytest.param(  # zsh takes it for the operand of the ||; bash rejects the line
                "sh", "true ||; export K=bad\n", "good", False,
                id="after a ; that follows no command",
            ),
        ],
    )
    def test_migration_leaves_the_body_of_a_compound_command_to_the_shell(
        self, shell: str, rest: str, value: str, from_text: bool, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An assignment inside a compound command of several lines is not read as unconditional.

        The scan read each line by itself: after ``export KEY=good`` an
        ``export KEY=bad`` in the body of ``if false; then … fi`` (a
        function, a loop, a ``case`` clause, a brace group) was
        imported as the last assignment, where the shell keeps
        ``good``.  A key that a compound command names is the shell's
        to resolve, like the one-line forms (the RC is sourced once),
        whatever follows: a function may run after a later assignment.
        So is one a command sets or unsets in a way of its own.  After
        compound commands that do not assign the key (they may name it:
        a message, ``env -u KEY``, a query) the text is read again,
        with no shell.
        """
        text = ("set -gx K good\n" if shell == "fish" else "export K=good\n") + rest
        text = text.replace("K", "OPENAI_API_KEY").replace("O=", "OTHER_KEY=")
        if shell == "sh":
            shell = _installed_posix_shell()
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rc = _shell_rc_path(shell)
        rc.parent.mkdir(parents=True, exist_ok=True)
        rc.write_text(text)
        scanned = user_config._scan_rc_text_for_keys(text, shell, {"OPENAI_API_KEY"})
        assert scanned == {"OPENAI_API_KEY": value if from_text else None}
        if _resolve_shell_path(shell) is None:
            return  # that shell is not here: the text left it to a shell there is none of
        load_api_keys()
        assert _sourced(shell, rc) == f"key={value}\n"
        assert os.environ.get("OPENAI_API_KEY", "unset") == value
        sourced_once = user_config._RC_SOURCED_MARKER in api_keys_env_path().read_text()
        assert sourced_once is not from_text

    def test_migration_reads_a_redirection_as_the_shell_does(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A redirection ends the value before it.

        ``export KEY=real-key>/dev/null`` migrated ``real-key>/dev/null``,
        a credential no provider accepts: the shell assigns ``real-key``
        and redirects the export.  The descriptor number of ``2>`` is no
        second word, a quoted or escaped angle bracket is a character of
        the value, and a process substitution is the shell's to expand.
        """
        names = (
            "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
        )
        rc = Path.home() / ".bashrc"
        rc.write_text(
            "export OPENAI_API_KEY=real-key>/dev/null\n"
            "export GEMINI_API_KEY=input-key</dev/null\n"
            "export ANTHROPIC_API_KEY=descriptor-key 2>/dev/null\n"
            "export TOGETHER_API_KEY='quoted>key'\\<escaped\n"
            "export OPENROUTER_API_KEY=substituted<(:)\n"
        )
        monkeypatch.setenv("SHELL", "/bin/bash")
        for name in names:
            monkeypatch.delenv(name, raising=False)
        assert user_config._scan_rc_text_for_keys(rc.read_text(), "bash", set(names)) == {
            "OPENAI_API_KEY": "real-key",
            "GEMINI_API_KEY": "input-key",
            "ANTHROPIC_API_KEY": "descriptor-key",
            "TOGETHER_API_KEY": "quoted>key<escaped",
            "OPENROUTER_API_KEY": None,
        }
        load_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == "real-key"
        bash = _resolve_shell_path("bash")
        if bash is not None:
            # What the real shell assigns, sourcing the same RC (bounded, clean environment).
            show = 'source "$1"; printf "%s\\n" ' + " ".join(f'"${name}"' for name in names)
            sourced = subprocess.run(
                [bash, "--noprofile", "--norc", "-c", show, "probe", str(rc)],
                capture_output=True, text=True, timeout=30, env={"HOME": str(Path.home())},
            )
            assert [os.environ.get(name, "") for name in names] == sourced.stdout.splitlines()

    @pytest.mark.parametrize("shell", ["bash", "zsh"])
    def test_plain_here_document_text_is_neither_migrated_nor_deleted(
        self, shell: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The lines of a plain here-document are text: the shell runs none of them.

        One that looks like an export of the key was migrated as the key
        (the last assignment of an RC decides) and deleted with it, so
        the document lost its example.  The four spellings, the quoted
        ones with a line ending in a backslash, which those do not join
        to the next; the export after the second closing line is real.
        """
        if _resolve_shell_path(shell) is None:
            pytest.skip(f"{shell} is not installed")
        text = "export OPENAI_API_KEY=example-only\n"
        quoted = "a quoted document joins no lines \\\n" + text
        documents = [
            f"cat <<DOC\n{text}DOC\n",
            f"cat <<-DOC\n\t{text}\tDOC\n",
            f"cat <<'DOC'\n{quoted}DOC\n",
            f'cat <<"DOC"\n{quoted}DOC\n',
        ]
        rc = _shell_rc_path(shell)
        export = "export OPENAI_API_KEY=made-up-value\n"
        rc.write_text("".join([*documents[:2], export, *documents[2:]]))
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _sourced(shell, rc) == text * 2 + quoted * 2 + "key=made-up-value\n"
        load_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == "made-up-value"
        save_api_key("OPENAI_API_KEY", "")
        assert rc.read_text() == "".join(documents)
        assert _sourced(shell, rc) == text * 2 + quoted * 2 + "key=unset\n"
        # The rule reads the text it is given and no file: what the RC holds by now is nothing
        # to it (a second read of the RC decided, so a link retargeted in between misled it).
        rc.write_bytes(b"# a carriage return\r\n")
        scanned = user_config._scan_rc_text_for_keys(documents[0], shell, {"OPENAI_API_KEY"})
        assert scanned == {}

    @pytest.mark.parametrize(
        ("before", "runs_in"),
        [
            pytest.param("NOTE='usage:\ncat <<true\n'\n", "bash zsh", id="string over lines"),
            pytest.param(": << OUTER\ncat <<true\nOUTER\n", "bash zsh", id="other document"),
            pytest.param("true=2\na[1 <<true ]=x\n", None, id="shift in a subscript"),
            pytest.param("NOTE=$'it\\'s\ncat <<true\na'\n", "bash zsh", id="dollar-quoted string"),
            pytest.param("NOTE=$(echo a # )\ncat <<true )\n", None, id="closes a $( it is in"),
            pytest.param("NOTE=$(echo a # )\ncat <<true\ntrue)\n", None, id="ends at true)"),
            pytest.param("cat <<true >/dev/null\ntr\\\nue\n", "bash zsh", id="joined lines end it"),
            pytest.param(
                "cat <<true >/dev/null\ntr\0ue\n",
                "bash" if _BASH_VERSION >= (4, 0) else "",
                id="NUL dropped by bash",
            ),
            pytest.param("# note \x0c cat <<true\n", "bash zsh", id="form feed ends no line"),
            pytest.param("# note \r cat <<true\n", "bash zsh", id="lone CR ends no line"),
            pytest.param("cat <<DOC\r\ntext\nDOC\ncat <<true\nDOC\r\n", "bash zsh", id="CR in DOC"),
            pytest.param("cat <<never >/dev/null\n", "", id="no closing line"),
            pytest.param("X=a \\\n", None, id="second line of a command"),
        ],
    )
    def test_export_is_deleted_where_no_plain_here_document_holds_it(
        self, before: str, runs_in: str | None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The export after a ``<<true`` is deleted in the shells that run it, and only there.

        Each RC has a ``<<true`` and a line ``true`` after the export.
        In the shells of *runs_in* the export is a command (the real
        shell, where installed, agrees): there it is deleted and every
        other byte the shell reads, a carriage return too, stays.  In
        the others it is the text of a document (zsh keeps the NUL that
        bash drops; a document with no closing line runs to the end) and
        stays as well.  ``None``: bash and zsh read the RC differently,
        or the export is one line of a longer command (deleted alone, it
        left ``X=a true``): the deletion is refused and nothing changes.
        """
        rcs = {shell: _shell_rc_path(shell) for shell in ("bash", "zsh")}
        installed = [shell for shell in rcs if _resolve_shell_path(shell) is not None]
        export = "export OPENAI_API_KEY=made-up-value\n"
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setenv("OPENAI_API_KEY", "made-up-value")
        for rc in rcs.values():
            rc.write_bytes((before + export + "true\n").encode())
        if runs_in is None:
            with pytest.raises(OSError, match="OPENAI_API_KEY"):
                save_api_key("OPENAI_API_KEY", "")
            kept_key = os.environ["OPENAI_API_KEY"]
            assert kept_key == "made-up-value"
        else:
            for shell in installed:
                ran = _sourced(shell, rcs[shell]).endswith("key=made-up-value\n")
                assert ran == (shell in runs_in.split())
            save_api_key("OPENAI_API_KEY", "")
            for shell in installed:
                assert _sourced(shell, rcs[shell]).endswith("key=unset\n")
        for shell, rc in rcs.items():
            deleted = runs_in is not None and shell in runs_in.split()
            kept = before + ("" if deleted else export) + "true\n"
            if deleted and shell == "bash":
                kept = kept.replace("\0", "")  # the edit is of the text bash reads
            assert rc.read_bytes() == kept.encode()

    @pytest.mark.parametrize(
        "document",
        [
            pytest.param(": <<'EOF'\r\n{text}\r\nEOF\r\n", id="CRLF"),
            pytest.param(": <<'EOF'\n{text}\nEOF\n#\r\n", id="lone-CR"),
            pytest.param(": <<$'EOF'\n{text}\nEOF\n", id="dollar-quote"),
            pytest.param(": <<E'OF'\n{text}\nEOF\n", id="inner-quote"),
            pytest.param(": <<EO\\\nF\n{text}\nEOF\n", id="line-continuation"),
            pytest.param("X=$(cat <<'EOF'\n{text}\nEOF\n)\n", id="substitution"),
            pytest.param(": <<'EOF'\n{text}\n", id="unterminated"),
            pytest.param("\r{text}\n", id="carriage return opens the line"),
            pytest.param(": <<'EOF'\n{text}\nEOF\nX=1\\", id="backslash ends the RC"),
            pytest.param(": <<'EOF'\n{text}\nEOF\nX=1\\\n", id="backslash continues the last line"),
        ],
    )
    def test_here_document_text_is_no_export_however_it_is_spelled(
        self, document: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The text of a here-document is never migrated as a key, nor deleted with one.

        Seven spellings the reader gave up on, reading every line as a
        command: start-up imported the example as the key and a save
        deleted it from the document.  Real bash and zsh set nothing;
        the saved key reaches a new bash through the hook, which goes
        before a document that runs to the end of the RC.  Nor is a
        carriage return a blank: the line it opens runs no ``export``.
        A backslash ending the RC joined the hook block to the last
        line, a syntax error that kept the hook from running.
        """
        before = document.replace("{text}", "export OPENAI_API_KEY=example-only").encode()
        rcs = {shell: _shell_rc_path(shell) for shell in ("bash", "zsh")}
        installed = [shell for shell in rcs if _resolve_shell_path(shell) is not None]
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        for rc in rcs.values():
            rc.write_bytes(before)
        for shell in installed:
            assert _sourced(shell, rcs[shell]).endswith("key=unset\n")
        load_api_keys()
        # A local: pytest prints ``os.environ``, real keys included, when an assert naming it fails.
        migrated = os.environ.get("OPENAI_API_KEY")
        assert migrated is None
        save_api_key("OPENAI_API_KEY", "made-up-value")
        assert rcs["zsh"].read_bytes() == before
        assert before in rcs["bash"].read_bytes()
        if "bash" in installed:
            assert _sourced("bash", rcs["bash"]).endswith("key=made-up-value\n")
        save_api_key("OPENAI_API_KEY", "")
        assert rcs["zsh"].read_bytes() == before
        assert before in rcs["bash"].read_bytes()

    @pytest.mark.parametrize("opener", ["<<'DOC'", "<<DOC"])
    @pytest.mark.parametrize("example", [_RC_HOOK_LINE, RC_HOOK_END])
    def test_hook_text_in_a_here_document_is_no_hook(
        self, opener: str, example: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hook line or end marker quoted in a here-document installs nothing.

        The hook line counted as the installed hook and the marker drew
        the new hook into the document: the save deleted the old export
        and no new shell loaded the saved key.
        """
        if _resolve_shell_path("bash") is None:
            pytest.skip("bash is not installed")
        rc = _shell_rc_path("bash")
        document = f": {opener}\n{example}\nDOC\n"
        rc.write_text(document + "export OPENAI_API_KEY=old-value\n")
        monkeypatch.setenv("SHELL", "/bin/bash")
        save_api_key("OPENAI_API_KEY", "new-value")
        assert rc.read_text().startswith(document)
        assert "old-value" not in rc.read_text()
        assert _sourced("bash", rc).endswith("key=new-value\n")

    @posix_only("an unreadable RC needs chmod-based permission denial")
    @pytest.mark.parametrize("failure", ["an RC cannot be read", "the store cannot be written"])
    def test_failed_save_leaves_every_rc_as_it_was(
        self, failure: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A save that fails changes no RC, whichever step fails.

        The first RC was scrubbed and replaced before the second was
        read, and every RC before the store was written: the failed save
        had already taken the old key out of new shells.
        """
        text = "export OPENAI_API_KEY=old-value\n# mine\n"
        rcs = [_shell_rc_path(shell) for shell in ("bash", "zsh")]
        for rc in rcs:
            rc.write_text(text)
            rc.chmod(0o644)
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setenv("OPENAI_API_KEY", "old-value")
        if failure == "the store cannot be written":
            api_keys_env_path().mkdir(parents=True)
        elif is_root():
            pytest.skip("root reads a file of mode 000")
        else:
            rcs[1].chmod(0o000)
        with pytest.raises(OSError):
            save_api_key("OPENAI_API_KEY", "new-value")
        rcs[1].chmod(0o644)
        assert [rc.read_text() for rc in rcs] == [text, text]
        assert rcs[0].stat().st_mode & 0o777 == 0o644
        kept_key = os.environ["OPENAI_API_KEY"]
        assert kept_key == "old-value"

    @posix_only("creating a symlink needs no privilege")
    def test_rc_behind_two_links_is_edited_under_one_lock(self, tmp_path: Path) -> None:
        """Two homes whose RC is a link to one file write it one after the other.

        Each writer locked a file beside its own link, so both read the
        shared RC and the second replace brought back the export the
        first had deleted.  Here two processes each delete one key while
        the lock beside the RC itself is held: both wait for it, and
        once it is free both exports are gone and the links are links.
        """
        keys = ("OPENAI_API_KEY", "GEMINI_API_KEY")
        target = tmp_path / "shared" / "bashrc"
        target.parent.mkdir()
        target.write_text("".join(f"export {key}=made-up\n" for key in keys) + "# mine\n")
        # Both writers start saving at ``go``: neither waits for the lock while the other loads.
        script = "\n".join([
            "import pathlib, sys, time",
            "from kiss.core.user_config import save_api_key",
            "pathlib.Path(sys.argv[2]).write_text('ready')",
            "deadline = time.monotonic() + 120",
            "while not pathlib.Path(sys.argv[3]).exists() and time.monotonic() < deadline:",
            "    time.sleep(0.02)",
            "save_api_key(sys.argv[1], '')",
        ])
        writers: list[subprocess.Popen[bytes]] = []
        try:
            with exclusive_file_lock(target.with_name("bashrc.kiss.lock"), timeout=30):
                for key in keys:
                    home = tmp_path / key
                    home.mkdir()
                    (home / ".bashrc").symlink_to(target)
                    env = {**os.environ, "HOME": str(home), "KISS_HOME": str(home / ".kiss")}
                    ready, go = str(home / "ready"), str(tmp_path / "go")
                    command = [sys.executable, "-c", script, key, ready, go]
                    writers.append(subprocess.Popen(command, env={**env, "SHELL": "/bin/bash"}))
                deadline = time.monotonic() + 120
                while not all((tmp_path / key / "ready").exists() for key in keys):
                    assert time.monotonic() < deadline, "a writer never started"
                    time.sleep(0.05)
                (tmp_path / "go").write_text("go")
                time.sleep(1.0)  # a writer that does not wait is done within milliseconds
                assert [writer.poll() for writer in writers] == [None, None]
                assert "OPENAI_API_KEY" in target.read_text()
            assert [writer.wait(60) for writer in writers] == [0, 0]
        finally:
            for writer in writers:
                if writer.poll() is None:
                    writer.kill()
                    writer.wait(30)
        assert target.read_text() == "# mine\n"
        assert all((tmp_path / key / ".bashrc").is_symlink() for key in keys)

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

    def test_saved_value_that_opens_with_an_equals_sign_reads_back_in_every_shell(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value that begins with ``=`` is stored quoted, for zsh.

        ``shlex.quote`` leaves ``=`` alone, so the store held ``export
        KEY==value``.  zsh looks such a word up as a command (``=ls``)
        and, finding none, stopped sourcing the store there: every key
        after that line was unset in the user's terminals.
        """
        monkeypatch.setenv("SHELL", f"/bin/{_installed_posix_shell()}")
        save_api_key("GEMINI_API_KEY", "=opens-with-equals")
        save_api_key("OPENAI_API_KEY", "the-key-after-it")
        for shell in ("bash", "zsh"):
            if _resolve_shell_path(shell) is not None:
                assert _sourced(shell, api_keys_env_path()) == "key=the-key-after-it\n"
        monkeypatch.delenv("GEMINI_API_KEY")
        load_api_keys()
        assert os.environ.get("GEMINI_API_KEY") == "=opens-with-equals"

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
        # The RC names the key but assembles its value indirectly, so
        # the textual scan cannot import it and the migration reaches
        # the sourcing fallback; a child outlives the shell.
        rc.write_text('V=indirect\nexport GEMINI_API_KEY="$V"\nsleep 30 &\n')
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(user_config, "_MIGRATION_TIMEOUT_S", 0.5)
        start = _time.monotonic()
        load_api_keys()
        elapsed = _time.monotonic() - start
        assert elapsed < 5.0, f"load_api_keys blocked for {elapsed:.1f}s"

    @posix_only("the RC is sourced by a POSIX shell")
    @pytest.mark.parametrize("stop", [KeyboardInterrupt, ToolCallInterrupted])
    def test_stop_while_sourcing_the_rc_ends_its_shell_and_frees_the_locks(
        self, stop: type[BaseException], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Stop raised in the thread that sources the RC ends the shell and its children.

        ``KeyboardInterrupt`` left the shell's whole group running, and
        any other exception waited for it without limit while holding
        the config lock and the store lock every later load and save
        needs.  The RC prints a line every 0.1 s, which wakes the reader
        the Stop is injected into.
        """
        shell = _installed_posix_shell()
        pids = tmp_path / "pids"
        _shell_rc_path(shell).write_text(
            'V=indirect\nexport GEMINI_API_KEY="$V"\nsleep 600 &\n'
            f'echo "$$ $!" > {shlex.quote(str(pids))}\nwhile :; do echo tick; sleep 0.1; done\n'
        )
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # Only the Stop ends the sourcing: under load the 5 s limit could come first.
        monkeypatch.setattr(user_config, "_MIGRATION_TIMEOUT_S", 120.0)
        raised: list[BaseException] = []
        group: list[int] = []
        loader = threading.Thread(target=_load_api_keys_into, args=(raised,), daemon=True)
        loader.start()
        try:
            deadline = time.monotonic() + 60
            while not pids.exists() or len(pids.read_text().split()) < 2:
                assert time.monotonic() < deadline, "the RC never ran"
                time.sleep(0.05)
            group += [int(pid) for pid in pids.read_text().split()]
            assert loader.ident is not None
            assert inject_async_exception(loader.ident, stop) == 1
            loader.join(20)
            assert not loader.is_alive(), "load_api_keys still waits for the RC's shell"
            assert [type(error) for error in raised] == [stop]
            deadline = time.monotonic() + 20
            while any(pid_alive(pid) for pid in group):
                assert time.monotonic() < deadline, f"the RC's processes still run: {group}"
                time.sleep(0.05)
            _shell_rc_path(shell).write_text("export GEMINI_API_KEY=plain\n")
            again = threading.Thread(target=_load_api_keys_into, args=(raised,), daemon=True)
            again.start()
            again.join(60)
            assert not again.is_alive(), "a lock of the stopped load is still held"
            loaded = os.environ.get("GEMINI_API_KEY")
            assert loaded == "plain"
        finally:
            for pid in group:
                if pid_alive(pid):
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGKILL)
            loader.join(30)


# Environment snapshot taken when pytest imports this module (before any
# of its tests run): the leak guard below compares against it.  The
# ambient AWS credential variables and ``OPENAI_API_KEY`` are left out:
# the autouse ``no_ambient_aws_credentials`` and
# ``_offline_memory_embeddings`` fixtures in ``conftest.py`` hide them
# for the duration of every test by design, so on a machine where the
# developer has exported real keys they differ from the import-time
# value inside the guard itself without any test having leaked.
_ENV_AT_IMPORT = {
    k: os.environ.get(k)
    for k in API_KEY_ENV_VARS
    if k not in AMBIENT_AWS_ENV_VARS and k != "OPENAI_API_KEY"
}


class TestNoApiKeyEnvLeak:
    """This file's tests must not leak API-key env vars to later files.

    ``_isolate_config`` used ``monkeypatch.delenv(key, raising=False)``
    for keys ABSENT from the environment — which records no restoration
    action — while tests such as
    ``test_code_only_upgrade_retires_systemd_mirror`` make
    ``load_api_keys()`` write fake values (``v-ANTHROPIC_WORKSPACE_ID``)
    straight into ``os.environ``.  The fake workspace id then survived
    into later test FILES in the same pytest process, where a live
    Anthropic call sent it as the ``anthropic-workspace-id`` header and
    the API rejected the run with 400 "must be a valid workspace ID"
    (full-suite run 2026-09-17, split py_08).

    Defined last in this module so, in pytest's definition order, it
    runs after every test above has had the chance to leak.
    """

    def test_env_restored_after_all_prior_tests(self) -> None:
        # Names and set/unset only: the values are live credentials and
        # must never reach a test report.
        leaked = {
            k: ("set" if k in os.environ else "unset", "set" if at_import is not None else "unset")
            for k, at_import in sorted(_ENV_AT_IMPORT.items())
            if os.environ.get(k) != at_import
        }
        assert not leaked, (
            "API-key env vars changed by earlier tests in this file were "
            f"not restored (current, at-import): {leaked}"
        )

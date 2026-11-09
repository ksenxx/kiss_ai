# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the chat page's settings panel backend.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_user_config``; the non-core tests remain there.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.user_config as user_config
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
from kiss.tests.conftest import (
    AMBIENT_AWS_ENV_VARS,
    HOT_READER_PAUSE,
    IS_WINDOWS,
    is_root,
    posix_only,
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

    def test_record_recent_work_dir_stores_extra_entries_in_one_write(
        self, tmp_path: Path,
    ) -> None:
        """Adopting a work dir persists the row and the extra entries together."""
        work = tmp_path / "proj"
        work.mkdir()
        save_config({"max_budget": 7, "custom_endpoint": "http://x/v1"})
        user_config.record_recent_work_dir(str(work), {"work_dir": str(work), "max_budget": "9"})
        cfg = load_config()
        assert cfg["work_dir"] == str(work)
        assert cfg["max_budget"] == 9  # sanitized like save_config
        assert cfg["custom_endpoint"] == "http://x/v1"
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
            "export KISS_TEST_QUOTED_SEMICOLON='a;b'\n",
        )
        load_api_keys()
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
        """
        rc = Path.home() / ".bashrc"
        rc.write_text(
            "case $- in\n"
            "    *i*) ;;\n"
            "      *) return;;\n"
            "esac\n"
            "export GEMINI_API_KEY=legacy-after-guard\n"
            "export OPENAI_API_KEY=old; export OPENAI_API_KEY=current;"
            " export ANTHROPIC_API_KEY=same-line\n",
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
        """
        shell = _installed_posix_shell()
        monkeypatch.setenv("SHELL", f"/bin/{shell}")
        for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        _shell_rc_path(shell).write_text(
            "export OPENAI_API_KEY=current || export OPENAI_API_KEY=obsolete\n"
            "false && export GEMINI_API_KEY=never-assigned\n"
            "export ANTHROPIC_API_KEY=pipeline-only | cat\n"
        )
        load_api_keys()
        assert os.environ.get("OPENAI_API_KEY") == "current"
        assert os.environ.get("GEMINI_API_KEY") is None
        assert os.environ.get("ANTHROPIC_API_KEY") is None
        store = api_keys_env_path().read_text()
        assert "OPENAI_API_KEY=current" in store
        assert "obsolete" not in store
        assert "never-assigned" not in store
        assert "pipeline-only" not in store
        # The text left them to the shell, which was sourced once.
        assert user_config._RC_SOURCED_MARKER in store

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


# Environment snapshot taken when pytest imports this module (before any
# of its tests run): the leak guard below compares against it.  The
# ambient AWS credential variables are left out: the autouse
# ``no_ambient_aws_credentials`` fixture in ``conftest.py`` hides them
# for the duration of every test by design, so on a machine where the
# developer has exported real keys they differ from the import-time
# value inside the guard itself without any test having leaked.
_ENV_AT_IMPORT = {
    k: os.environ.get(k) for k in API_KEY_ENV_VARS if k not in AMBIENT_AWS_ENV_VARS
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
        leaked = {
            k: (os.environ.get(k), at_import)
            for k, at_import in sorted(_ENV_AT_IMPORT.items())
            if os.environ.get(k) != at_import
        }
        assert not leaked, (
            "API-key env vars changed by earlier tests in this file were "
            f"not restored (current, at-import): {leaked}"
        )

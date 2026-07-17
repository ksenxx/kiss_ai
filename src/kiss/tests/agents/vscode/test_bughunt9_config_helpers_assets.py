# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt round 9 — e2e repros for three real defects.

1. ``helpers.clean_llm_output`` stripped *unpaired* quote characters:
   ``str.strip('"')`` removes leading and trailing quotes
   independently, so a commit message that legitimately **ends** with
   a quoted word — e.g. ``feat: rename "foo"`` — was corrupted to
   ``feat: rename "foo`` (dangling opening quote).  Only *paired*
   surrounding quotes (LLM decoration) must be stripped.

2. ``vscode_config.apply_config_to_env`` coerced a boolean
   ``max_budget`` to ``1.0``/``0.0`` via ``float()`` (``bool`` is an
   ``int`` subclass), silently shrinking the live budget to $1 or $0.
   ``sanitize_config`` in the very same module explicitly rejects
   booleans for numeric keys, and the function's own contract says a
   non-numeric value falls back to ``DEFAULTS['max_budget']``.

3. ``user_assets.ensure_user_asset_from_default`` seeded the asset
   with a plain ``Path.write_text`` — ``open('w')`` truncates first,
   so a concurrent reader (e.g. the autocomplete worker thread calling
   ``read_tricks`` while a command-handler thread seeds
   ``MY_INJECTION.md``) could observe an empty or partially-written
   file.  The seed must be atomic: a reader sees either no file or
   the full default content, never a torn write.

4. ``vscode_config.save_api_key_to_shell`` shell-quoted the key
   *value* (the H3 fix) but interpolated the key *name* verbatim into
   the ``export`` line — and ``_cmd_save_config`` forwards **any**
   string name from an untrusted client payload.  A name containing a
   newline (``"X\\nrm -rf ~ #"``) or shell metacharacters writes
   arbitrary commands into the user's RC file, executed the next time
   a shell starts.  A name containing ``=`` also raises ``ValueError``
   out of ``os.environ[key_name] = …``, killing the client connection.
   Only valid environment-variable identifiers may be written.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core.vscode_config import (
    DEFAULTS,
    apply_config_to_env,
    save_api_key_to_shell,
)
from kiss.server.helpers import clean_llm_output
from kiss.server.user_assets import ensure_user_asset_from_default


class TestCleanLlmOutputPairedQuotesOnly:
    """Only *paired* surrounding quotes are decoration to strip."""

    def test_trailing_quoted_word_not_corrupted(self) -> None:
        """A message ending in a quoted word must keep its quotes."""
        msg = 'feat: rename "foo"'
        assert clean_llm_output(msg) == msg

    def test_leading_quoted_word_not_corrupted(self) -> None:
        """A message starting with a quoted word must keep its quotes."""
        msg = '"foo" was renamed'
        assert clean_llm_output(msg) == msg

    def test_trailing_single_quoted_word_not_corrupted(self) -> None:
        msg = "fix: escape 'bar'"
        assert clean_llm_output(msg) == msg

    def test_paired_quotes_still_stripped(self) -> None:
        """The documented decoration-stripping behaviour is preserved."""
        assert clean_llm_output('"fix the bug"\n') == "fix the bug"
        assert clean_llm_output("'msg'\n") == "msg"
        assert clean_llm_output('  "hello"  ') == "hello"

    def test_nested_paired_quotes_stripped_repeatedly(self) -> None:
        assert clean_llm_output('""double wrapped""') == "double wrapped"
        assert clean_llm_output("'\"both kinds\"'") == "both kinds"

    def test_inner_quotes_preserved(self) -> None:
        assert clean_llm_output('say "hi" now') == 'say "hi" now'

    def test_plain_text_unchanged(self) -> None:
        assert clean_llm_output("feat: add widget\n") == "feat: add widget"
        assert clean_llm_output("") == ""
        assert clean_llm_output("   \n  ") == ""


class TestApplyConfigBooleanBudget:
    """A boolean ``max_budget`` must fall back to the default budget."""

    @pytest.fixture(autouse=True)
    def _restore_budget(self):  # type: ignore[no-untyped-def]
        saved = config_module.DEFAULT_CONFIG.max_budget
        yield
        config_module.DEFAULT_CONFIG.max_budget = saved

    def test_true_budget_falls_back_to_default(self) -> None:
        """``True`` must not become a $1.00 live budget."""
        config_module.DEFAULT_CONFIG.max_budget = 55.0
        apply_config_to_env({"max_budget": True})
        assert config_module.DEFAULT_CONFIG.max_budget == float(
            DEFAULTS["max_budget"],
        )

    def test_false_budget_falls_back_to_default(self) -> None:
        """``False`` must not become a $0.00 live budget."""
        config_module.DEFAULT_CONFIG.max_budget = 55.0
        apply_config_to_env({"max_budget": False})
        assert config_module.DEFAULT_CONFIG.max_budget == float(
            DEFAULTS["max_budget"],
        )

    def test_genuine_numbers_still_apply(self) -> None:
        apply_config_to_env({"max_budget": 42})
        assert config_module.DEFAULT_CONFIG.max_budget == 42.0
        apply_config_to_env({"max_budget": 7.5})
        assert config_module.DEFAULT_CONFIG.max_budget == 7.5


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``KISS_HOME`` to a fresh ``tmp_path`` for each test."""
    home = tmp_path / ".kiss"
    monkeypatch.setenv("KISS_HOME", str(home))
    return home


class TestUserAssetSeedIsAtomic:
    """Concurrent readers must never observe a torn/empty seed write."""

    def test_reader_never_sees_partial_content(self, kiss_home: Path) -> None:
        """Race a spinning reader against repeated first-read seeding.

        With the non-atomic ``write_text`` seed, ``open('w')``
        truncates the file before the content lands, so the reader
        observes an empty (or partial) file within a few hundred
        iterations.  With an atomic seed the reader only ever sees
        ``FileNotFoundError`` or the complete default content.
        """
        content = "## Trick\n\n" + ("trick body line\n" * 4000)
        name = "RACE_ASSET.md"
        target = kiss_home / name
        kiss_home.mkdir(parents=True, exist_ok=True)
        bad: list[int] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    text = target.read_text()
                except FileNotFoundError:
                    continue
                if text != content:
                    bad.append(len(text))
                    stop.set()
                    return

        t = threading.Thread(target=reader)
        t.start()
        try:
            for _ in range(400):
                if stop.is_set():
                    break
                target.unlink(missing_ok=True)
                result = ensure_user_asset_from_default(name, content)
                assert result == target
        finally:
            stop.set()
            t.join()
        assert not bad, (
            f"reader observed torn seed content of length {bad[0]} "
            f"(expected {len(content)})"
        )

    def test_seed_and_preserve_contract_unchanged(self, kiss_home: Path) -> None:
        """Seeding + never-overwrite semantics survive the atomicity fix."""
        result = ensure_user_asset_from_default("A.md", "## Trick\n\nseed\n")
        assert result == kiss_home / "A.md"
        assert result is not None
        assert result.read_text() == "## Trick\n\nseed\n"
        result.write_text("user edit")
        again = ensure_user_asset_from_default("A.md", "## Trick\n\nseed\n")
        assert again == result
        assert result.read_text() == "user edit"

    def test_no_temp_file_litter_after_seed(self, kiss_home: Path) -> None:
        """The atomic seed must not leave staging files behind."""
        ensure_user_asset_from_default("B.md", "body\n")
        leftovers = [
            p.name for p in kiss_home.iterdir() if p.name != "B.md"
        ]
        assert leftovers == []


class TestSaveApiKeyNameValidation:
    """Only valid env-var identifiers may reach the shell RC file."""

    @pytest.fixture(autouse=True)
    def fake_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> Generator[Path]:
        """Point ``HOME``/``SHELL`` at a scratch dir so the real RC is safe."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("SHELL", "/bin/bash")
        # Pre-set via monkeypatch so any value written by the function
        # under test is restored on teardown.
        monkeypatch.setenv("OPENAI_API_KEY", "sentinel-original")
        # ``save_api_key_to_shell`` rebuilds the ``DEFAULT_CONFIG``
        # singleton via ``_refresh_config()``, capturing the fake key
        # from the patched environment.  monkeypatch restores the env
        # on teardown but not the singleton, so later tests (e.g. live
        # model tests) would see the stale fake key.  Restore the
        # original singleton object after the test.
        saved_config = config_module.DEFAULT_CONFIG
        yield tmp_path
        config_module.DEFAULT_CONFIG = saved_config

    def test_newline_name_cannot_inject_rc_commands(
        self, fake_home: Path,
    ) -> None:
        """A name embedding a newline must not write commands to the RC."""
        evil = "OPENAI_API_KEY\ntouch " + str(fake_home / "pwned") + "\n#"
        save_api_key_to_shell(evil, "sk-x")
        rc = fake_home / ".bashrc"
        assert not rc.exists(), rc.read_text()
        assert evil not in os.environ

    def test_metacharacter_name_rejected(self, fake_home: Path) -> None:
        """Shell metacharacters in the name must never reach the RC."""
        for evil in ("FOO$(touch pwned)", "FOO; rm -rf ~", "FOO BAR", ""):
            save_api_key_to_shell(evil, "sk-x")
        assert not (fake_home / ".bashrc").exists()

    def test_equals_name_does_not_raise(self, fake_home: Path) -> None:
        """A name containing ``=`` must not raise out of the handler."""
        save_api_key_to_shell("A=B", "sk-x")
        assert not (fake_home / ".bashrc").exists()

    def test_valid_name_still_written_and_exported(
        self, fake_home: Path,
    ) -> None:
        """The legitimate save path keeps working after the fix."""
        save_api_key_to_shell("OPENAI_API_KEY", "sk-valid")
        rc = fake_home / ".bashrc"
        assert rc.exists()
        assert "export OPENAI_API_KEY=sk-valid" in rc.read_text()
        assert os.environ["OPENAI_API_KEY"] == "sk-valid"

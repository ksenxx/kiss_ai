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

import threading
from pathlib import Path

import pytest

from kiss.agents.sorcar.commit_message import clean_llm_output
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

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for INJECTIONS.md trick fast-complete in the CLI REPL.

The browser/VS Code ghost text and the readline-style CLI completer
must both surface user "Inject instruction" tricks (loaded from
``~/.kiss/INJECTIONS.md``) as fast-complete suggestions at the
beginning of every sentence.  These tests exercise the CLI side via
``CliCompleter._predictive_matches``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.cli_repl import CliCompleter

_LONG_TRICK = (
    "Use claude-opus-4-7 model for all tasks including coding, "
    "bug fixing, and test creation. Use gpt-5.5 model (not codex) "
    "for thorough review of the work done by the other model. "
    "Check if the other model has missed some code."
)

_FAKE_INJECTIONS = (
    "## Trick\n"
    "\n"
    "Reproduce the issue by writing end-to-end test. Then fix the issue.\n"
    "\n"
    "## Trick\n"
    "\n"
    f"{_LONG_TRICK}\n"
    "\n"
    "## Trick\n"
    "\n"
    "Use internet search extensively.\n"
)


@pytest.fixture
def kiss_db(tmp_path: Path):
    """Redirect the history DB AND ``$KISS_HOME`` to an isolated temp dir.

    Pins the bundled INJECTIONS.md tricks via ``KISS_INJECTIONS_PATH``
    so the test is independent of the package's real bundled tricks.
    A zero-byte ``MY_INJECTION.md`` is created in the temp ``~/.kiss/``
    so the auto-seeded default trick does not contribute extra entries
    to the prefix-match dictionary.
    """
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    fake_path = kiss_dir / "fake_INJECTIONS.md"
    fake_path.write_text(_FAKE_INJECTIONS)
    (kiss_dir / "MY_INJECTION.md").write_text("")
    saved_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    saved_home = os.environ.get("KISS_HOME")
    saved_inj = os.environ.get("KISS_INJECTIONS_PATH")
    os.environ["KISS_HOME"] = str(kiss_dir)
    os.environ["KISS_INJECTIONS_PATH"] = str(fake_path)
    yield kiss_dir
    if th._db_conn is not None:
        th._db_conn.close()
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved_db
    if saved_home is None:
        os.environ.pop("KISS_HOME", None)
    else:
        os.environ["KISS_HOME"] = saved_home
    if saved_inj is None:
        os.environ.pop("KISS_INJECTIONS_PATH", None)
    else:
        os.environ["KISS_INJECTIONS_PATH"] = saved_inj


def test_cli_trick_suggested_at_start_of_line(tmp_path: Path, kiss_db) -> None:
    """Typing the start of a trick at position 0 yields a whole-line completion."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("Reproduce")
    assert matches == [
        "Reproduce the issue by writing end-to-end test. "
        "Then fix the issue."
    ], f"Got {matches!r}"


def test_cli_trick_suggested_after_sentence_boundary(
    tmp_path: Path, kiss_db,
) -> None:
    """A trick is offered after ``. `` in the middle of the line.

    The suggestion is the raw trick body — no head-splicing onto the
    preamble.  Accepting the completion replaces the input with just
    the trick body.
    """
    completer = CliCompleter(str(tmp_path))
    line = "Some preamble text. Reproduce"
    matches = completer._build_matches(line)
    assert matches == [
        "Reproduce the issue by writing end-to-end test. "
        "Then fix the issue."
    ], f"Got {matches!r}"


def test_cli_no_trick_mid_sentence(tmp_path: Path, kiss_db) -> None:
    """No trick is suggested when the matching partial sits mid-sentence."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("please Reproduce")
    assert matches == [], (
        f"Expected no trick suggestion mid-sentence, got {matches!r}"
    )


def test_cli_history_wins_over_trick(tmp_path: Path, kiss_db) -> None:
    """A history match takes precedence over an INJECTIONS.md trick."""
    th._add_task("Reproduce my custom historical task", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("Reproduce")
    # History match must be present and must NOT be the trick.
    assert "Reproduce my custom historical task" in matches
    assert "Reproduce the issue by writing end-to-end test. " \
           "Then fix the issue." not in matches


def test_cli_trick_falls_back_to_active_file_when_no_match(
    tmp_path: Path, kiss_db,
) -> None:
    """Identifier completion still works when no trick / history matches."""
    active = tmp_path / "code.py"
    active.write_text("def calculate_total(items):\n    return sum(items)\n")
    completer = CliCompleter(str(tmp_path), active_file=str(active))
    # ``calculate_t`` is not at a sentence start (no period before "call")
    # so no trick match — but also no trick prefixes "calculate_t", so
    # the trick branch correctly skips and the active-file branch runs.
    matches = completer._build_matches("call calculate_t")
    assert matches == ["calculate_total"]


def test_cli_case_sensitive_trick_match(tmp_path: Path, kiss_db) -> None:
    """Lowercase ``reproduce`` does not match capitalised ``Reproduce`` trick."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("reproduce")
    assert matches == [], f"Expected case-sensitive miss, got {matches!r}"

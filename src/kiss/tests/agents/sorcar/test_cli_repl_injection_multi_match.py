# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproducer for Round-1 review bug: multi-matching INJECTIONS.md tricks.

When two or more ``## Trick`` sections in ``~/.kiss/INJECTIONS.md`` share
a common prefix (e.g. both real tricks
``Reproduce the issue by writing integration tests. ...`` and
``Reproduce the issue by writing end-to-end test. ...`` begin with
``Reproduce``), the prompt_toolkit dropdown / readline cycle must surface
ALL matching tricks — not just the first one in file order — so the user
can pick the variant they actually want.

The prior commit (754704ab) had ``_predictive_matches`` return a single
trick-completed line, dropping the alternates on the floor.  The CLI
history pathway already returns up to 8 alternatives via
``_prefix_match_tasks``; the trick pathway must mirror that.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.cli_repl import CliCompleter

_TRICKS = (
    "## Trick\n"
    "\n"
    "Reproduce the issue by writing integration tests. Then fix the issue.\n"
    "\n"
    "## Trick\n"
    "\n"
    "Reproduce the issue by writing end-to-end test. Then fix the issue.\n"
    "\n"
    "## Trick\n"
    "\n"
    "Reproduce the bug step by step.\n"
    "\n"
    "## Trick\n"
    "\n"
    "Use internet search extensively.\n"
)


@pytest.fixture
def kiss_db(tmp_path: Path):
    """Redirect ``$KISS_HOME`` (and the history DB) to an isolated temp dir.

    Pins the bundled INJECTIONS.md tricks via ``KISS_INJECTIONS_PATH``
    so the test is independent of the package's real bundled tricks.
    A zero-byte ``MY_INJECTION.md`` is created in the temp ``~/.kiss/``
    so the auto-seeded default trick does not contribute extra entries
    to the prefix-match dictionary.
    """
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    fake_path = kiss_dir / "fake_INJECTIONS.md"
    fake_path.write_text(_TRICKS)
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


def test_cli_returns_all_matching_tricks_at_sentence_start(
    tmp_path: Path, kiss_db,
) -> None:
    """Typing the common prefix of multiple tricks must surface all of them."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("Reproduce")
    expected = {
        "Reproduce the issue by writing integration tests. Then fix the issue.",
        "Reproduce the issue by writing end-to-end test. Then fix the issue.",
        "Reproduce the bug step by step.",
    }
    assert set(matches) >= expected, (
        f"Expected ALL three tricks starting with 'Reproduce' to appear in "
        f"the dropdown, but got {matches!r}"
    )
    # The unrelated trick must NOT appear.
    assert "Use internet search extensively." not in matches


def test_cli_returns_all_matching_tricks_after_sentence_boundary(
    tmp_path: Path, kiss_db,
) -> None:
    """Multiple tricks must be offered after a mid-line ``. `` boundary too."""
    completer = CliCompleter(str(tmp_path))
    line = "Some preamble text. Reproduce"
    matches = completer._build_matches(line)
    expected = {
        "Reproduce the issue by writing integration tests. Then fix the issue.",
        "Reproduce the issue by writing end-to-end test. Then fix the issue.",
        "Reproduce the bug step by step.",
    }
    assert set(matches) >= expected, (
        f"Expected three trick completions after the sentence boundary, "
        f"got {matches!r}"
    )


def test_cli_returns_single_trick_when_only_one_matches(
    tmp_path: Path, kiss_db,
) -> None:
    """A unique-prefix partial must still produce exactly one match."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("Use intern")
    assert matches == ["Use internet search extensively."], (
        f"Expected exactly one trick match for unique prefix, got {matches!r}"
    )

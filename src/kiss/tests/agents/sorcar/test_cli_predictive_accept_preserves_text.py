# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: accepting a CLI fast-complete must keep typed text.

Mirrors the VS Code webview bug where accepting a fast-autocomplete
suggestion replaced the whole chat input with the raw suggestion,
erasing the text the user had already typed.  The Sorcar CLI REPL had
the identical bug in ``CliCompleter._predictive_matches``: trick and
active-file-identifier candidates were returned as raw suggestions
(starting only at the current sentence / trailing token) while BOTH
frontends — the readline completer (word-break delims cleared, so the
candidate replaces the entire line) and the prompt_toolkit dropdown
(``Completion(..., start_position=-len(line))``) — substitute the
candidate for the whole line.  Accepting such a candidate therefore
dropped everything before the matched partial.

These end-to-end tests drive the real completer backends and apply the
completions exactly the way readline / prompt_toolkit do, asserting the
previously-typed head survives the accept.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import kiss.agents.sorcar.persistence as th
from kiss.ui.cli.cli_prompt import PtkCompleter
from kiss.ui.cli.cli_repl import CliCompleter

_FAKE_INJECTIONS = (
    "## Trick\n"
    "\n"
    "Reproduce the issue by writing end-to-end test. Then fix the issue.\n"
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


def _apply_readline(completer: CliCompleter, line: str) -> str:
    """Accept the first readline candidate exactly as readline would.

    ``_setup_readline`` clears the completer word-break delimiters, so
    the completion "word" is the entire line and readline substitutes
    the returned candidate for all of it.
    """
    cand = completer.complete(line, 0)
    assert cand is not None, f"no completion offered for {line!r}"
    return cand


def _apply_ptk(cli: CliCompleter, line: str) -> list[str]:
    """Accept every prompt_toolkit dropdown candidate for *line*.

    Applies each :class:`~prompt_toolkit.completion.Completion` the way
    prompt_toolkit's ``Buffer.apply_completion`` does: delete
    ``-start_position`` characters before the cursor, insert ``text``.
    """
    doc = Document(text=line, cursor_position=len(line))
    comps = list(PtkCompleter(cli).get_completions(doc, CompleteEvent()))
    assert comps, f"no dropdown completions offered for {line!r}"
    return [
        line[: len(line) + c.start_position] + c.text for c in comps
    ]


def test_readline_identifier_accept_preserves_existing_text(
    tmp_path: Path, kiss_db,
) -> None:
    """Tab-accepting an identifier completion must keep the typed head."""
    active = tmp_path / "code.py"
    active.write_text("def parse_arguments(argv):\n    return argv\n")
    completer = CliCompleter(str(tmp_path), active_file=str(active))
    line = "please fix the bug in parse_arg"
    result = _apply_readline(completer, line)
    assert result.startswith("please fix the bug in "), (
        f"typed head was erased on accept: {result!r}"
    )
    assert result == "please fix the bug in parse_arguments"


def test_readline_trick_accept_preserves_earlier_sentences(
    tmp_path: Path, kiss_db,
) -> None:
    """Accepting a trick after ``. `` must keep the earlier sentence."""
    completer = CliCompleter(str(tmp_path))
    line = "Fix the crash in cli_repl. Reproduce"
    result = _apply_readline(completer, line)
    assert result == (
        "Fix the crash in cli_repl. Reproduce the issue by writing "
        "end-to-end test. Then fix the issue."
    ), f"earlier sentence was erased on accept: {result!r}"


def test_ptk_identifier_accept_preserves_existing_text(
    tmp_path: Path, kiss_db,
) -> None:
    """The prompt_toolkit dropdown accept must keep the typed head."""
    active = tmp_path / "code.py"
    active.write_text("def parse_arguments(argv):\n    return argv\n")
    cli = CliCompleter(str(tmp_path), active_file=str(active))
    line = "please fix the bug in parse_arg"
    results = _apply_ptk(cli, line)
    assert results == ["please fix the bug in parse_arguments"], (
        f"typed head was erased on accept: {results!r}"
    )


def test_ptk_trick_accept_preserves_earlier_sentences(
    tmp_path: Path, kiss_db,
) -> None:
    """The dropdown trick accept must keep every earlier sentence."""
    cli = CliCompleter(str(tmp_path))
    line = "Fix the crash in cli_repl. Reproduce"
    results = _apply_ptk(cli, line)
    assert results == [
        "Fix the crash in cli_repl. Reproduce the issue by writing "
        "end-to-end test. Then fix the issue."
    ], f"earlier sentence was erased on accept: {results!r}"


def test_ptk_history_accept_still_replaces_whole_line(
    tmp_path: Path, kiss_db,
) -> None:
    """History candidates already start with the line — accept unchanged."""
    th._add_task("refactor the authentication module thoroughly", chat_id="c1")
    cli = CliCompleter(str(tmp_path))
    line = "refactor the auth"
    results = _apply_ptk(cli, line)
    assert results == ["refactor the authentication module thoroughly"]

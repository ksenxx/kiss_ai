# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the sorcar CLI ``@``-mention picker.

Covers the prompt_toolkit dropdown added in
:mod:`kiss.agents.sorcar.cli_prompt`: pressing ``@`` must immediately
offer the project's files and folders, Up/Down must navigate the menu,
and Tab or Enter must insert the highlighted ``./<path>`` mention
without submitting the line.  Outside the ``@``-mention file picker
(predictive history, slash commands, ``/model``) Enter must never
accept a completion candidate — it submits the typed text as-is.
"""

from __future__ import annotations

import threading
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.cli_prompt import PtkCompleter, PtkLineReader
from kiss.agents.sorcar.cli_repl import SLASH_COMMANDS, CliCompleter


def _redirect(tmp_path: Path):
    """Redirect persistence to a per-test SQLite DB.

    Mirrors the helper used in ``test_persistence.py`` so the predictive
    completion tests get a clean task-history table they can seed.
    """
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved) -> None:
    """Restore the persistence singletons saved by ``_redirect``."""
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_project(tmp_path: Path) -> Path:
    """Create a small project tree with files and a folder."""
    (tmp_path / "alpha.py").write_text("def alpha_one(): pass\n")
    (tmp_path / "beta.md").write_text("# beta\n")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "gamma.txt").write_text("gamma\n")
    return tmp_path


def _typing_event() -> CompleteEvent:
    """A completion event as produced while the user is typing."""
    return CompleteEvent(text_inserted=True)


def test_at_press_immediately_lists_files_and_folders(tmp_path: Path) -> None:
    """Typing a bare ``@`` pops candidates for every file and folder."""
    project = _make_project(tmp_path)
    completer = PtkCompleter(CliCompleter(str(project)))
    completions = list(
        completer.get_completions(Document("@"), _typing_event())
    )
    shown = {str(c.display_text) for c in completions}
    assert "alpha.py" in shown
    assert "beta.md" in shown
    assert "subdir/" in shown  # folders are offered too
    assert "subdir/gamma.txt" in shown
    for c in completions:
        assert c.text.startswith("./")
        assert c.text.endswith(" ")
        assert c.start_position == -1  # replaces just the "@"


def test_at_query_filters_and_replaces_whole_token(tmp_path: Path) -> None:
    """``@alp`` filters to matching paths and replaces the full token."""
    project = _make_project(tmp_path)
    completer = PtkCompleter(CliCompleter(str(project)))
    completions = list(
        completer.get_completions(Document("fix @alp"), _typing_event())
    )
    assert completions, "expected at least one match for @alp"
    for c in completions:
        assert "alp" in str(c.display_text)
        assert c.start_position == -len("@alp")
    assert any(c.text == "./alpha.py " for c in completions)


def test_folder_completion_meta_marks_folders(tmp_path: Path) -> None:
    """Folder candidates are labelled as folders in the menu."""
    project = _make_project(tmp_path)
    completer = PtkCompleter(CliCompleter(str(project)))
    completions = list(
        completer.get_completions(Document("@subdir"), _typing_event())
    )
    by_display = {str(c.display_text): c for c in completions}
    folder = by_display["subdir/"]
    assert "folder" in str(folder.display_meta_text)
    assert folder.text == "./subdir/ "


def test_slash_menu_lists_commands_with_help(tmp_path: Path) -> None:
    """Typing ``/he`` offers ``/help`` with its description."""
    completer = PtkCompleter(CliCompleter(str(tmp_path)))
    completions = list(
        completer.get_completions(Document("/he"), _typing_event())
    )
    displays = {str(c.display_text) for c in completions}
    assert "/help" in displays
    help_completion = next(
        c for c in completions if str(c.display_text) == "/help"
    )
    assert "command" in str(help_completion.display_meta_text).lower()
    assert help_completion.start_position == -len("/he")


def test_bare_slash_menu_visible_room_for_every_command(tmp_path: Path) -> None:
    """Pressing ``/`` must show *every* built-in slash command at once.

    Regression test for a bug where the dropdown clipped to the first 8
    entries — prompt_toolkit's :class:`CompletionsMenu` only renders as
    many rows as the prompt reserves below the input line via
    ``reserve_space_for_menu``, so an under-sized reservation hides the
    rest of the commands behind a scrollbar that the user has to
    discover with Up/Down.  The reservation must therefore be at least
    as large as the number of slash commands so every entry is visible
    on a single ``/`` press.
    """
    # 1. The completer itself must offer every built-in slash command.
    completer = PtkCompleter(CliCompleter(str(tmp_path)))
    completions = list(
        completer.get_completions(Document("/"), _typing_event())
    )
    displays = {str(c.display_text) for c in completions}
    for cmd in SLASH_COMMANDS:
        assert cmd in displays, f"{cmd} missing from /-menu completions"
    # 2. The prompt session must reserve enough rows to render them all.
    hist = tmp_path / "hist"
    reader = PtkLineReader(CliCompleter(str(tmp_path)), hist)
    assert reader.session.reserve_space_for_menu >= len(SLASH_COMMANDS), (
        "reserve_space_for_menu must fit every built-in slash command so "
        "they are all visible on a single ``/`` press"
    )


def test_model_partial_completes_model_names(tmp_path: Path) -> None:
    """``/model gpt`` offers real model names containing the partial."""
    completer = PtkCompleter(CliCompleter(str(tmp_path)))
    completions = list(
        completer.get_completions(Document("/model gpt"), _typing_event())
    )
    assert completions, "expected model suggestions for 'gpt'"
    for c in completions:
        assert "gpt" in c.text.lower()
        assert c.start_position == -len("gpt")


def test_predictive_completion_pops_while_typing(tmp_path: Path) -> None:
    """A typed prefix matching prior tasks pops the dropdown immediately.

    Three distinct tasks share the prefix ``rewrite the`` so the menu
    must list all three (most-recent-first), each Completion replacing
    the typed line and tagged with the ``history`` meta so the menu
    shows where the suggestion came from.
    """
    saved = _redirect(tmp_path)
    try:
        th._add_task("rewrite the README intro")
        th._add_task("rewrite the docstring of `run_repl`")
        th._add_task("rewrite the help banner")
        completer = PtkCompleter(CliCompleter(str(tmp_path)))
        completions = list(
            completer.get_completions(Document("rewrite the"), _typing_event())
        )
        texts = [c.text for c in completions]
        assert "rewrite the help banner" in texts
        assert "rewrite the docstring of `run_repl`" in texts
        assert "rewrite the README intro" in texts
        # Most recently inserted task is offered first.
        assert texts[0] == "rewrite the help banner"
        # Every completion replaces the full typed line and is tagged
        # so the menu shows where the suggestion came from.
        for c in completions:
            assert c.start_position == -len("rewrite the")
            assert "history" in str(c.display_meta_text).lower()
    finally:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(saved)


def test_predictive_completion_empty_when_no_match(tmp_path: Path) -> None:
    """No menu is offered when the typed prefix matches nothing."""
    saved = _redirect(tmp_path)
    try:
        th._add_task("entirely unrelated task")
        completer = PtkCompleter(CliCompleter(str(tmp_path)))
        doc = Document("zq nonexistent prefix 8327462")
        assert list(completer.get_completions(doc, _typing_event())) == []
    finally:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(saved)


def test_predictive_arrow_down_then_tab_accepts(tmp_path: Path) -> None:
    """End-to-end: typing + Down + Tab picks a history suggestion.

    Drives a real PromptSession through a pipe: typing ``ref`` pops the
    predictive dropdown, Down highlights the first candidate, Tab
    confirms it (without submitting), and Enter submits the line.
    """
    saved = _redirect(tmp_path)
    try:
        th._add_task("refactor the cli_prompt module")
        completer = CliCompleter(str(tmp_path))
        hist = tmp_path / "hist"
        with create_pipe_input() as pipe:
            def _send_keys() -> None:
                pipe.send_text("\x1b[B")  # Down: highlight the suggestion
                pipe.send_text("\t")  # Tab: confirm without submitting
                pipe.send_text("\r")  # Enter: submit the now-completed line

            timer = threading.Timer(0.5, _send_keys)
            pipe.send_text("ref")
            timer.start()
            try:
                with create_app_session(input=pipe, output=DummyOutput()):
                    reader = PtkLineReader(completer, hist)
                    line = reader.read("> ")
            finally:
                timer.cancel()
        assert line == "refactor the cli_prompt module"
    finally:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(saved)


def test_predictive_arrow_down_then_enter_submits_typed_text(
    tmp_path: Path,
) -> None:
    """End-to-end: typing + Down + Enter submits the typed prefix.

    Enter must never accept the highlighted predictive suggestion —
    only Tab does.  The submitted line is exactly what the user typed.
    """
    saved = _redirect(tmp_path)
    try:
        th._add_task("refactor the cli_prompt module")
        completer = CliCompleter(str(tmp_path))
        hist = tmp_path / "hist"
        with create_pipe_input() as pipe:
            def _send_keys() -> None:
                pipe.send_text("\x1b[B")  # Down: highlight the suggestion
                pipe.send_text("\r")  # Enter: submit typed text as-is

            timer = threading.Timer(0.5, _send_keys)
            pipe.send_text("ref")
            timer.start()
            try:
                with create_app_session(input=pipe, output=DummyOutput()):
                    reader = PtkLineReader(completer, hist)
                    line = reader.read("> ")
            finally:
                timer.cancel()
        assert line == "ref", (
            f"Enter must submit the typed prefix, not the suggestion: {line!r}"
        )
    finally:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(saved)


def test_arrow_down_then_enter_selects_mention(tmp_path: Path) -> None:
    """End-to-end: ``@`` + Down + Enter inserts the mention, Enter submits.

    The ``@``-mention file picker is the one menu where Enter still
    accepts: the first Enter lands while a completion is highlighted,
    so it must *insert* the ``./<path>`` mention (not submit); the
    second Enter submits the line.
    """
    project = _make_project(tmp_path)
    completer = CliCompleter(str(project))
    hist = tmp_path / "hist"
    with create_pipe_input() as pipe:
        # Let complete_while_typing populate the menu before navigating.
        def _send_keys() -> None:
            pipe.send_text("\x1b[B")  # Down: highlight first candidate
            pipe.send_text("\r")  # Enter: accept highlighted completion
            pipe.send_text(" please\r")  # Enter: submit the line

        timer = threading.Timer(0.5, _send_keys)
        pipe.send_text("look at @")
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    assert line.startswith("look at ./")
    assert line.endswith(" please")
    mentioned = line[len("look at ./"):-len(" please")].strip()
    assert (project / mentioned).exists()


def test_tab_confirms_highlighted_completion(tmp_path: Path) -> None:
    """End-to-end: ``@alp`` + Down + Tab confirms without submitting."""
    project = _make_project(tmp_path)
    completer = CliCompleter(str(project))
    hist = tmp_path / "hist"
    with create_pipe_input() as pipe:
        def _send_keys() -> None:
            pipe.send_text("\x1b[B")  # Down: highlight alpha.py
            pipe.send_text("\t")  # Tab: confirm it (stay editing)
            pipe.send_text("now\r")  # Enter: submit

        timer = threading.Timer(0.5, _send_keys)
        pipe.send_text("@alp")
        timer.start()
        try:
            with create_app_session(input=pipe, output=DummyOutput()):
                reader = PtkLineReader(completer, hist)
                line = reader.read("> ")
        finally:
            timer.cancel()
    assert line == "./alpha.py now"


def test_readline_history_is_migrated_once(tmp_path: Path) -> None:
    """Old readline history seeds the prompt_toolkit history file."""
    hist = tmp_path / "hist"
    hist.write_text("first task\nsecond task\n")
    PtkLineReader(CliCompleter(str(tmp_path)), hist)
    ptk_file = tmp_path / "hist.ptk"
    assert ptk_file.exists()
    loaded = list(FileHistory(str(ptk_file)).load_history_strings())
    assert "first task" in loaded
    assert "second task" in loaded
    # A second reader must not duplicate the migrated entries.
    PtkLineReader(CliCompleter(str(tmp_path)), hist)
    again = list(FileHistory(str(ptk_file)).load_history_strings())
    assert len(again) == len(loaded)

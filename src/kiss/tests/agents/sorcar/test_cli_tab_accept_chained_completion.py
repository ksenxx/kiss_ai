# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a SINGLE Tab accepts a completion candidate and,
on a slash-command line, IMMEDIATELY pops the next-level menu.

The user-reported bug: pressing Tab once on an open menu only
*selected* the candidate (the text appeared in the buffer, so from the
user's point of view the command WAS autocompleted) but the
argument-candidate menu never popped — a second, non-obvious Tab press
was required.  One Tab per level must therefore accept the highlighted
candidate (or the first one when none is highlighted yet) and chain:

* ``/resu`` → Tab accepts ``/resume `` → the argument-option menu
  (``--task`` / ``--limit``) pops by itself;
* Tab accepts ``--task `` → the recent-task-id menu pops by itself,
  each candidate DISPLAYED as ``<id>: <one-line description>`` while
  accepting inserts only the bare id;
* Tab accepts the id → the remaining-option menu (``--limit``) pops;
  Enter then submits the typed line unchanged.

Up/Down still navigate the menu, and Tab then accepts the navigated
candidate instead of the first one.

Everything runs through a real :class:`PtkLineReader` prompt session
driven over a pipe input against a real isolated history database — no
mocks, no direct completer calls.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.ui.cli.cli_prompt import PtkLineReader
from kiss.ui.cli.cli_repl import CliCompleter

_TIMEOUT = 10.0


@pytest.fixture
def kiss_db(tmp_path: Path):
    """Redirect the history DB and config dir to an isolated temp dir."""
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR, vc.CONFIG_DIR)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    vc.CONFIG_DIR = kiss_dir
    yield kiss_dir
    if th._db_conn is not None:
        th._db_conn.close()
    th._DB_PATH, th._db_conn, th._KISS_DIR, vc.CONFIG_DIR = saved


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    """Poll *condition* until true or fail after :data:`_TIMEOUT` seconds."""
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _menu(buf) -> list:
    """Return the open completion menu's candidates ([] when closed)."""
    state = buf.complete_state
    return list(state.completions) if state is not None else []


def _drive_session(
    tmp_path: Path, driver: Callable[[object, object], None],
) -> str:
    """Run one real prompt read, feeding keys from *driver* (pipe, buf).

    The driver runs on a background thread while the main thread blocks
    inside :meth:`PtkLineReader.read`; it must eventually send ``\\r``
    so the read returns.  Returns the submitted line.
    """
    completer = CliCompleter(str(tmp_path))
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            reader = PtkLineReader(completer, tmp_path / "hist")
            buf = reader.session.default_buffer
            errors: list[BaseException] = []

            def run_driver() -> None:
                try:
                    driver(pipe, buf)
                except BaseException as exc:  # propagated after join
                    errors.append(exc)
                    pipe.send_text("\r")

            thread = threading.Thread(target=run_driver)
            thread.start()
            try:
                line = reader.read("> ")
            finally:
                thread.join(timeout=_TIMEOUT)
            if errors:
                raise errors[0]
    return line


def test_single_tab_accepts_and_chains_all_menu_levels(
    tmp_path: Path, kiss_db,
) -> None:
    """ONE Tab per level: command → flag → task-id → remaining options.

    This is the user-reported bug reproduction: a single Tab press must
    both insert the candidate AND pop the next-level menu — no second
    Tab required at any level.
    """
    task_id, _ = th._add_task("fix the parser bug", chat_id="c1")
    seen: dict[str, list[tuple[str, str, str]]] = {}

    def snapshot(buf) -> list[tuple[str, str, str]]:
        return [
            (c.text, c.display_text, c.display_meta_text)
            for c in _menu(buf)
        ]

    def driver(pipe, buf) -> None:
        pipe.send_text("/resu")
        _wait_for(
            lambda: buf.text == "/resu" and bool(_menu(buf)),
            "the /resume command menu",
        )
        pipe.send_text("\t")  # ONE Tab: accept -> option menu pops
        _wait_for(
            lambda: buf.text == "/resume "
            and bool(_menu(buf))
            and _menu(buf)[0].display_text == "--task",
            "the argument-option menu after one Tab on /resume",
        )
        seen["options"] = snapshot(buf)
        pipe.send_text("\t")  # ONE Tab: accept --task -> task-id menu
        _wait_for(
            lambda: buf.text == "/resume --task "
            and bool(_menu(buf))
            and _menu(buf)[0].display_text.startswith(f"{task_id}:"),
            "the task-id menu after one Tab on --task",
        )
        seen["task_ids"] = snapshot(buf)
        pipe.send_text("\t")  # ONE Tab: accept the id -> remaining menu
        _wait_for(
            lambda: buf.text == f"/resume --task {task_id} "
            and bool(_menu(buf))
            and _menu(buf)[0].display_text == "--limit",
            "the remaining-option menu after one Tab on the task id",
        )
        seen["remaining"] = snapshot(buf)
        pipe.send_text("\r")  # Enter submits the typed line unchanged

    line = _drive_session(tmp_path, driver)

    # Level 1: one Tab on "/resume" popped its argument options.
    assert [d for _, d, _ in seen["options"]] == ["--task", "--limit"]
    assert seen["options"][0][0] == "/resume --task "
    # Level 2: one Tab on "--task" popped the task ids, DISPLAYED as
    # "<id>: <description>" but inserting only the bare id.
    assert seen["task_ids"] == [
        (
            f"/resume --task {task_id} ",
            f"{task_id}: fix the parser bug",
            "task",
        ),
    ]
    # Level 3: one Tab on the id popped the not-yet-used options.
    assert [d for _, d, _ in seen["remaining"]] == ["--limit"]
    # Enter submitted the typed text — never a completion candidate.
    assert line == f"/resume --task {task_id} "


def test_down_navigation_then_tab_accepts_navigated_candidate(
    tmp_path: Path, kiss_db,
) -> None:
    """Down highlights a later candidate; Tab accepts THAT one and chains."""
    th._add_task("fix the parser bug", chat_id="c1")

    def driver(pipe, buf) -> None:
        pipe.send_text("/resume ")
        _wait_for(
            lambda: bool(_menu(buf))
            and [c.display_text for c in _menu(buf)] == ["--task", "--limit"],
            "the argument-option menu",
        )
        pipe.send_text("\x1b[B\x1b[B")  # Down Down: highlight --limit
        _wait_for(
            lambda: buf.complete_state is not None
            and buf.complete_state.current_completion is not None
            and buf.complete_state.current_completion.display_text
            == "--limit",
            "--limit to be highlighted",
        )
        pipe.send_text("\t")  # Tab accepts the NAVIGATED candidate
        _wait_for(
            lambda: buf.text == "/resume --limit ",
            "--limit to be accepted",
        )
        pipe.send_text("\r")

    assert _drive_session(tmp_path, driver) == "/resume --limit "


def test_tab_accept_model_command_pops_model_names(
    tmp_path: Path, kiss_db,
) -> None:
    """One Tab on ``/model`` immediately pops ``list`` + the model names."""
    seen: dict[str, list[str]] = {}

    def driver(pipe, buf) -> None:
        pipe.send_text("/mode")
        _wait_for(
            lambda: bool(_menu(buf))
            and _menu(buf)[0].display_text == "/model",
            "the /model command candidate",
        )
        pipe.send_text("\t")  # ONE Tab: accept -> model-name menu pops
        _wait_for(
            lambda: buf.text == "/model "
            and bool(_menu(buf))
            and _menu(buf)[0].display_text == "list",
            "the model-name menu after one Tab on /model",
        )
        seen["models"] = [c.display_text for c in _menu(buf)]
        pipe.send_text("\r")

    line = _drive_session(tmp_path, driver)
    assert seen["models"][0] == "list"
    assert len(seen["models"]) > 1, "model names must follow the list entry"
    assert line == "/model "


def test_tab_accept_without_arguments_pops_no_menu(
    tmp_path: Path, kiss_db,
) -> None:
    """Accepting a command with no completable arguments opens no menu."""

    def driver(pipe, buf) -> None:
        pipe.send_text("/hel")
        _wait_for(
            lambda: bool(_menu(buf))
            and _menu(buf)[0].display_text == "/help",
            "the /help command menu",
        )
        pipe.send_text("\t")  # ONE Tab: accept; no follow-up candidates
        _wait_for(lambda: buf.text == "/help ", "/help to be inserted")
        # The restarted completion finds nothing; give it a moment and
        # assert no menu is (or becomes) visible.
        time.sleep(0.3)
        assert not _menu(buf), "no argument menu expected after /help"
        pipe.send_text("\r")

    assert _drive_session(tmp_path, driver) == "/help "


def test_tab_accept_predictive_history_does_not_restart(
    tmp_path: Path, kiss_db,
) -> None:
    """Accepting a non-slash predictive candidate keeps the menu closed."""
    th._add_task("fix the parser bug", chat_id="c1")

    def driver(pipe, buf) -> None:
        pipe.send_text("fix")
        _wait_for(
            lambda: bool(_menu(buf))
            and _menu(buf)[0].text == "fix the parser bug",
            "the predictive history menu",
        )
        pipe.send_text("\t")  # ONE Tab: accept; non-slash line, no restart
        _wait_for(
            lambda: buf.text == "fix the parser bug",
            "the history candidate to be inserted",
        )
        time.sleep(0.3)
        assert not _menu(buf), "no menu expected after a non-slash accept"
        pipe.send_text("\r")

    assert _drive_session(tmp_path, driver) == "fix the parser bug"

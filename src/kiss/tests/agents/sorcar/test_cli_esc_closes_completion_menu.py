# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: pressing ESC closes an open autocomplete menu.

When the CLI REPL shows a list of completion candidates, a single ESC
key press must dismiss the list without touching the typed buffer.
Both interactive input stacks are covered:

* the anchored bottom input box (:class:`_InputBox` in
  ``cli_steering.py``) — what a real terminal session of ``sorcar``
  uses for ALL interactive input.  A bare ESC arrives as a lone
  ``\\x1b`` byte that is deferred (it could be the first byte of a
  split escape sequence), so the select pump's idle tick resolves it
  via :meth:`_InputBox.flush_pending_escape` — the same
  ttimeout-style disambiguation vim uses;
* the prompt_toolkit fallback (:class:`PtkLineReader` in
  ``cli_prompt.py``) — ESC must cancel the completion dropdown,
  restoring the text the user actually typed even after Up/Down
  navigation placed a candidate's text in the buffer.

The tests drive a real :class:`_InputBox` through its byte-level
:meth:`feed` API and a real :class:`PtkLineReader` prompt session over
a pipe input, both against a real (isolated) history database — no
mocks.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import kiss.agents.sorcar.persistence as th
import kiss.server.vscode_config as vc
from kiss.ui.cli.cli_prompt import PtkLineReader
from kiss.ui.cli.cli_repl import CliCompleter
from kiss.ui.cli.cli_steering import _InputBox

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


# ---------------------------------------------------------------------------
# Anchored steering box (_InputBox) — the real interactive input path
# ---------------------------------------------------------------------------


def _make_box(work_dir: str) -> tuple[_InputBox, list[str]]:
    """Build a real box wired to a real ``CliCompleter.build_menu``."""
    completer = CliCompleter(work_dir)
    box = _InputBox(threading.RLock(), io.StringIO())
    box.completer_fn = completer.build_menu
    return box, []


def _feed(box: _InputBox, data: bytes, submitted: list[str]) -> None:
    """Feed raw bytes into the box, collecting submitted lines."""
    box.feed(data, submitted.append, lambda: None)


def test_bare_esc_closes_menu_via_idle_flush(tmp_path: Path, kiss_db) -> None:
    """A lone ESC press dismisses the open menu, leaving the buffer intact.

    A bare ESC key arrives as a single ``\\x1b`` byte that ``feed``
    defers (it might be the first byte of a split escape sequence);
    the select pump's next idle tick calls ``flush_pending_escape``
    which must close the menu.
    """
    th._add_task("fix the parser bug", chat_id="c1")
    box, submitted = _make_box(str(tmp_path))
    _feed(box, b"/resume ", submitted)
    assert box._menu_open is True
    _feed(box, b"\x1b", submitted)  # bare ESC — deferred as pending
    assert box.flush_pending_escape() is True
    assert box._menu_open is False
    assert box._menu_items == []
    assert box.buf == "/resume "  # typed text untouched
    # The pending ESC was consumed: a subsequent flush is a no-op ...
    assert box.flush_pending_escape() is False
    # ... and the next printable keystroke types normally (no stale
    # ESC prepended that would swallow it).
    _feed(box, b"-", submitted)
    assert box.buf == "/resume -"


def test_esc_followed_by_key_in_same_chunk_closes_menu(
    tmp_path: Path, kiss_db,
) -> None:
    """ESC immediately followed by another key still dismisses the menu."""
    th._add_task("fix the parser bug", chat_id="c1")
    box, submitted = _make_box(str(tmp_path))
    _feed(box, b"/resume ", submitted)
    assert box._menu_open is True
    # ESC + 'x' arriving in one read (fast typing / Alt+x): the ESC
    # closes the menu; menus for the subsequent edit may reopen only
    # for actual candidates ('x' matches none here).
    _feed(box, b"\x1bx", submitted)
    assert box._menu_open is False
    assert box.buf == "/resume x"


def test_flush_keeps_split_escape_sequences_pending(
    tmp_path: Path, kiss_db,
) -> None:
    """Only an exact lone ESC is flushed — split CSI stays deferred.

    A chunk ending in ``\\x1b[`` is a split escape sequence whose
    remainder arrives in the next read; the idle flush must not
    misread it as a bare ESC press.
    """
    th._add_task("fix the parser bug", chat_id="c1")
    th._add_task("write release notes", chat_id="c2")
    box, submitted = _make_box(str(tmp_path))
    _feed(box, b"/resume --task ", submitted)
    assert box._menu_open is True
    assert box._menu_sel == 0
    _feed(box, b"\x1b[", submitted)  # split Down-arrow, first half
    assert box.flush_pending_escape() is False
    assert box._menu_open is True  # menu untouched
    _feed(box, b"B", submitted)  # second half completes ESC[B (Down)
    assert box._menu_open is True
    assert box._menu_sel == 1


def test_bare_esc_without_menu_is_a_noop(tmp_path: Path) -> None:
    """ESC with no menu open neither crashes nor edits the buffer."""
    box = _InputBox(threading.RLock(), io.StringIO())
    submitted: list[str] = []
    _feed(box, b"hello", submitted)
    assert box._menu_open is False
    _feed(box, b"\x1b", submitted)
    assert box.flush_pending_escape() is False
    assert box.buf == "hello"
    assert box._menu_open is False


# ---------------------------------------------------------------------------
# prompt_toolkit fallback (PtkLineReader)
# ---------------------------------------------------------------------------


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    """Poll *condition* until true or fail after :data:`_TIMEOUT` seconds."""
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


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


def test_ptk_esc_closes_menu_and_enter_submits_typed_text(
    tmp_path: Path, kiss_db,
) -> None:
    """ESC dismisses the ptk dropdown; Enter then submits the typed line."""
    th._add_task("fix the parser bug", chat_id="c1")

    def driver(pipe, buf) -> None:
        pipe.send_text("/resume ")
        _wait_for(
            lambda: buf.complete_state is not None
            and bool(buf.complete_state.completions),
            "the /resume argument-option menu",
        )
        pipe.send_text("\x1b")  # bare ESC — must close the menu
        _wait_for(
            lambda: buf.complete_state is None,
            "the menu to close after ESC",
        )
        assert buf.text == "/resume "
        pipe.send_text("\r")

    assert _drive_session(tmp_path, driver) == "/resume "


def test_ptk_esc_restores_typed_text_after_navigation(
    tmp_path: Path, kiss_db,
) -> None:
    """ESC after Down-navigation restores the text the user typed."""
    th._add_task("fix the parser bug", chat_id="c1")

    def driver(pipe, buf) -> None:
        pipe.send_text("/resume ")
        _wait_for(
            lambda: buf.complete_state is not None
            and bool(buf.complete_state.completions),
            "the /resume argument-option menu",
        )
        pipe.send_text("\x1b[B")  # Down: highlight + insert a candidate
        _wait_for(
            lambda: buf.text != "/resume ",
            "navigation to insert the candidate text",
        )
        pipe.send_text("\x1b")  # ESC — close menu, restore typed text
        _wait_for(
            lambda: buf.complete_state is None and buf.text == "/resume ",
            "ESC to close the menu and restore the typed text",
        )
        pipe.send_text("\r")

    assert _drive_session(tmp_path, driver) == "/resume "

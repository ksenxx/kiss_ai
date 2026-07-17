# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for chained completion in the anchored steering box.

The anchored bottom input box (:class:`_InputBox`) is what a real
terminal session of ``sorcar`` uses for ALL interactive input (the
prompt_toolkit dropdown is only the off-TTY / tiny-terminal fallback),
so the "one Tab per level" chained-completion behaviour must hold
there:

* Tab-accepting a slash command (``/resume``) must immediately pop the
  argument-option menu (``--task`` / ``--limit``) with help text;
* Tab-accepting ``--task`` must immediately pop the recent task ids,
  each row displayed as ``<id>: <one-line task description>`` while
  accepting inserts only the bare id;
* Tab-accepting a task id must immediately pop the remaining options;
* Enter always submits the typed buffer, never a completion;
* non-slash accepts (predictive history) must NOT chain a menu.

The tests drive a real :class:`_InputBox` through its byte-level
:meth:`feed` API with a real :class:`CliCompleter` reading a real
(isolated) history database — no mocks.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
import kiss.server.vscode_config as vc
from kiss.ui.cli.cli_repl import CliCompleter
from kiss.ui.cli.cli_steering import _InputBox


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


def _make_box(work_dir: str) -> tuple[_InputBox, list[str]]:
    """Build a real box wired to a real ``CliCompleter.build_menu``."""
    completer = CliCompleter(work_dir)
    box = _InputBox(threading.RLock(), io.StringIO())
    box.completer_fn = completer.build_menu
    return box, []


def _feed(box: _InputBox, data: bytes, submitted: list[str]) -> None:
    """Feed raw bytes into the box, collecting submitted lines."""
    box.feed(data, submitted.append, lambda: None)


def test_single_tab_chains_all_menu_levels(tmp_path: Path, kiss_db) -> None:
    """One Tab per level: command -> options -> task ids -> options."""
    old_id, _ = th._add_task("fix the parser bug", chat_id="c1")
    new_id, _ = th._add_task("write release notes", chat_id="c2")
    box, submitted = _make_box(str(tmp_path))

    # Typing ``/resu`` previews the single matching command.
    _feed(box, b"/resu", submitted)
    assert box._menu_open is True
    assert box._menu_repls == ["/resume "]

    # Tab 1: accept ``/resume`` -> buffer completed AND the argument
    # option menu pops by itself, showing each option with its help.
    _feed(box, b"\t", submitted)
    assert box.buf == "/resume "
    assert box._menu_open is True
    assert box._menu_repls == ["/resume --task ", "/resume --limit "]
    assert box._menu_items[0].startswith("--task — ")
    assert box._menu_items[1].startswith("--limit — ")

    # Tab 2: accept ``--task`` -> the recent task-id menu pops by
    # itself, newest first, each row showing ``<id>: <description>``
    # while the replacement inserts only the bare id.
    _feed(box, b"\t", submitted)
    assert box.buf == "/resume --task "
    assert box._menu_open is True
    assert box._menu_items == [
        f"{new_id}: write release notes",
        f"{old_id}: fix the parser bug",
    ]
    assert box._menu_repls == [
        f"/resume --task {new_id} ",
        f"/resume --task {old_id} ",
    ]

    # Tab 3: accept the highlighted (newest) task id -> only the bare
    # id lands in the buffer (no colon, no description) AND the
    # remaining unused option menu pops by itself.
    _feed(box, b"\t", submitted)
    assert box.buf == f"/resume --task {new_id} "
    assert box._menu_open is True
    assert box._menu_repls == [f"/resume --task {new_id} --limit "]
    assert box._menu_items[0].startswith("--limit — ")

    # Enter submits the typed buffer as-is (never a completion).
    _feed(box, b"\r", submitted)
    assert submitted == [f"/resume --task {new_id} "]
    assert box.buf == ""
    assert box._menu_open is False


def test_down_arrow_then_tab_accepts_navigated_candidate(
    tmp_path: Path, kiss_db,
) -> None:
    """Down highlights the next candidate; Tab accepts THAT one and chains."""
    old_id, _ = th._add_task("fix the parser bug", chat_id="c1")
    new_id, _ = th._add_task("write release notes", chat_id="c2")
    box, submitted = _make_box(str(tmp_path))
    _feed(box, b"/resume --task ", submitted)
    assert box._menu_open is True
    assert box._menu_sel == 0
    _feed(box, b"\x1b[B", submitted)  # Down arrow
    assert box._menu_sel == 1
    _feed(box, b"\t", submitted)
    assert box.buf == f"/resume --task {old_id} "
    # Chained menu with the remaining option pops immediately.
    assert box._menu_open is True
    assert box._menu_repls == [f"/resume --task {old_id} --limit "]


def test_predictive_accept_does_not_chain(tmp_path: Path, kiss_db) -> None:
    """Accepting a non-slash predictive candidate keeps the menu closed."""
    th._add_task("fix the parser bug", chat_id="c1")
    box, submitted = _make_box(str(tmp_path))
    _feed(box, b"fix", submitted)
    assert box._menu_open is True
    assert box._menu_repls == ["fix the parser bug"]
    _feed(box, b"\t", submitted)
    assert box.buf == "fix the parser bug"
    assert box._menu_open is False


def test_plain_string_completer_still_works(tmp_path: Path) -> None:
    """Back-compat: a ``list[str]`` completer keeps display == replacement."""
    box = _InputBox(threading.RLock(), io.StringIO())
    box.completer_fn = lambda _buf: ["alpha", "beta"]
    submitted: list[str] = []
    _feed(box, b"a", submitted)
    assert box._menu_open is True
    assert box._menu_items == ["alpha", "beta"]
    assert box._menu_repls == ["alpha", "beta"]
    _feed(box, b"\t", submitted)
    assert box.buf == "alpha"
    assert box._menu_open is False  # non-slash: no chained menu


def test_model_flag_value_rows_show_picker_group(
    tmp_path: Path, kiss_db,
) -> None:
    """``--model`` value rows display ``<name> — <group>``, insert the name."""
    box, submitted = _make_box(str(tmp_path))
    _feed(box, b"/parallel --model ", submitted)
    if not box._menu_open:  # no models completable in this environment
        pytest.skip("no completable models in this environment")
    first_repl = box._menu_repls[0]
    first_disp = box._menu_items[0]
    name = first_repl[len("/parallel --model "):].strip()
    assert first_repl.endswith(f"--model {name} ")
    assert first_disp.startswith(f"{name} — ")

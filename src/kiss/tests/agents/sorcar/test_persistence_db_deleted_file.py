# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: persistence survives deletion of the DB file.

Before the fix, ``_get_db()``'s fast path returned the cached
per-thread connection without checking that the database file still
existed on disk (or was still the SAME file).  If the file was deleted
while a cached connection was open (log rotation, a test cleaning up
``$KISS_HOME``, a user removing ``~/.kiss/history.db`` while the daemon
runs) — and possibly recreated at the same pathname by an independent
``sqlite3.connect`` — every subsequent write went into the orphaned
inode and silently disappeared, while any NEW reader of ``_DB_PATH``
connected to a fresh empty database and failed with
``sqlite3.OperationalError: no such table: task_history``.

This was the shared root cause behind the order-dependent failures of
the wave-2/wave-3 runner tests (``test_f2_*`` / ``test_b2_*``) when
they ran in a large split after a test that removed the redirected DB
file without invalidating the cached connection.

No mocks/patches/fakes: the real persistence layer, a real sqlite
database file, and a real ``os.unlink``.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import CodeType

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _add_task
from kiss.tests.conftest import posix_only
from kiss.tests.persistence_helpers import close_db

_TOOL = 4  # a ``sys.monitoring`` tool id of this module's own (``kiss.tests.server.pause``: 3)


@contextlib.contextmanager
def _watching(code: CodeType, event: int, callback: Callable[..., None]) -> Iterator[None]:
    """Call *callback* on each ``sys.monitoring`` *event* of *code* while the block runs.

    Unlike a trace function it leaves the one in place (coverage's) at work, so the
    product lines a test reaches this way are still measured; *callback* must not raise.
    """
    mon = sys.monitoring
    mon.use_tool_id(_TOOL, "kiss-tests")
    mon.register_callback(_TOOL, event, callback)
    mon.set_local_events(_TOOL, code, event)
    try:
        yield
    finally:
        mon.set_local_events(_TOOL, code, 0)
        mon.register_callback(_TOOL, event, None)
        mon.free_tool_id(_TOOL)


def _on_thread(
    thread: threading.Thread, at: int | str, left: list[float], action: Callable[[], object],
    _code: CodeType, where: int, *_retval: object,
) -> None:
    """Monitoring callback (the first four arguments bound): run *action* as *thread* reaches
    line *at* (``"start"`` and ``"return"``: every event), ``left[0]`` times at most."""
    if threading.current_thread() is thread and at in ("start", "return", where) and left[0]:
        left[0] -= 1
        action()


def park_new_connection_in_setup(opener: threading.Thread) -> sqlite3.Connection:
    """Start *opener* and hold its first ``_get_db()`` between the open and the return.

    A claimed legacy journal snapshot makes the new connection's setup write its
    marker, and the returned connection owns the database's write lock, so that
    write waits in SQLite's busy handler until the returned connection rolls back.
    """
    legacy = th._failed_events_path(th._LEGACY_DB_NAME) + th._JOURNAL_CONSUMED_SUFFIX + "1"
    (th._DB_PATH.parent / legacy).write_text("")
    blocker = sqlite3.connect(str(th._DB_PATH), isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    parked = threading.Event()
    adoption = th._adopt_legacy_journal_snapshots.__code__
    entered = functools.partial(_on_thread, opener, "start", [1], parked.set)
    with _watching(adoption, sys.monitoring.events.PY_START, entered):
        opener.start()
        if parked.wait(30):
            return blocker
    blocker.close()
    raise AssertionError("the new connection never reached its legacy-journal adoption")


def get_db_at(
    code: CodeType, at: int | str, times: float, action: Callable[[], object]
) -> sqlite3.Connection:
    """``_get_db()`` on the calling thread, with *action* run the first *times* times *code*
    reaches line *at* (``"return"``: returns) during the call."""
    events = sys.monitoring.events
    reached = functools.partial(_on_thread, threading.current_thread(), at, [times], action)
    with _watching(code, events.PY_RETURN if at == "return" else events.LINE, reached):
        return th._get_db()


def _add_task_on_thread(task: str) -> None:
    """Add *task* on a thread that has no connection yet, then close the one it opened."""
    try:
        _add_task(task, chat_id="")
    finally:
        th._close_thread_db()


@posix_only("unlinking a SQLite file another connection holds open")
class TestDbFileDeletedUnderCachedConnection:
    """Writes after an external DB-file deletion must be durable."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss_dbdel_")
        self.saved = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"

    def teardown_method(self) -> None:
        close_db()
        th._DB_PATH, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _delete_db_files(self) -> None:
        """Remove the DB file and its WAL/SHM side files, keeping the
        cached per-thread connection open (no ``close_db()``)."""
        for suffix in ("", "-wal", "-shm"):
            path = th._DB_PATH.with_name(th._DB_PATH.name + suffix)
            path.unlink(missing_ok=True)

    def test_write_after_delete_recreates_schema_and_is_visible(self) -> None:
        _add_task("dbdel first", chat_id="")
        assert th._DB_PATH.exists()

        self._delete_db_files()
        assert not th._DB_PATH.exists()

        _add_task("dbdel second", chat_id="")
        assert th._DB_PATH.exists()

        conn = sqlite3.connect(str(th._DB_PATH))
        try:
            rows = conn.execute(
                "SELECT task FROM task_history ORDER BY timestamp",
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == ["dbdel second"]

    def test_reads_also_heal_after_delete(self) -> None:
        _add_task("dbdel read-heal", chat_id="")
        self._delete_db_files()
        entries = th._load_history()
        assert entries == []

    def test_same_path_recreated_file_is_detected(self) -> None:
        _add_task("dbdel recreate first", chat_id="")
        self._delete_db_files()
        sqlite3.connect(str(th._DB_PATH)).close()
        assert th._DB_PATH.exists()

        _add_task("dbdel recreate second", chat_id="")

        conn = sqlite3.connect(str(th._DB_PATH))
        try:
            rows = conn.execute(
                "SELECT task FROM task_history ORDER BY timestamp",
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == ["dbdel recreate second"]

    def _replace_db_file(self) -> None:
        """Delete the database file and its sidecars and put an empty file in its place."""
        self._delete_db_files()
        sqlite3.connect(str(th._DB_PATH)).close()

    def _tasks_in(self, path: Path) -> list[str]:
        """The tasks in the database file *path*, read by a connection of its own."""
        conn = sqlite3.connect(str(path))
        try:
            return sorted(r[0] for r in conn.execute("SELECT task FROM task_history"))
        finally:
            conn.close()

    def test_replacement_during_connection_setup_is_detected(self) -> None:
        """The file is replaced while a thread's new connection is being set up (it waits
        on another writer): the connection is on the old file, so the thread reconnects
        and its row is durable."""
        _add_task("dbdel setup first", chat_id="")
        th._close_thread_db()
        opener = threading.Thread(
            target=_add_task_on_thread, args=("dbdel setup second",), daemon=True)
        blocker = park_new_connection_in_setup(opener)
        self._replace_db_file()
        blocker.rollback()
        blocker.close()
        opener.join(60)
        assert not opener.is_alive()
        assert self._tasks_in(th._DB_PATH) == ["dbdel setup second"]

    @pytest.mark.parametrize(
        ("absent", "times"), [(False, 1), (True, 1), (False, 2), (False, math.inf)])
    def test_replacement_at_every_line_of_the_open_is_detected(
        self, absent: bool, times: float
    ) -> None:
        """The file is replaced as each line of the open is reached, with a file or none at
        the pathname before the call: once, again on the retry, or in every setup of the
        call.  The connection returned is on the file then at the pathname (a row written
        through it is there); replaced in every setup, the call fails and leaves its thread
        no connection rather than return one to a replaced file."""
        _add_task("dbdel sweep first", chat_id="")
        code = th._open_db_connection.__code__
        refused = []
        for lineno in sorted({line for _s, _e, line in code.co_lines() if line}):
            th._close_thread_db()
            if absent:
                self._delete_db_files()
            task = f"dbdel line {lineno}"
            try:
                conn = get_db_at(code, lineno, times, self._replace_db_file)
            except sqlite3.OperationalError:
                me = threading.current_thread()
                left = [c for c, o, _p in th._open_conns.values() if o is me]
                assert times == math.inf and not left and th._thread_local.conn is None, task
                refused.append(task)
                continue
            conn.execute(
                "INSERT INTO task_history (id, task, timestamp) VALUES (?, ?, 0)", (task, task))
            assert task in self._tasks_in(th._DB_PATH), task
        assert bool(refused) == (times == math.inf)

    def test_a_database_put_in_place_as_the_absent_file_is_created_keeps_its_rows(self) -> None:
        """Nothing is at the pathname, and a database is put there the instant SQLite has
        created the file: the connection returned is on that database, whose rows are kept."""
        _add_task("dbdel kept", chat_id="")
        th._close_thread_db()
        aside = th._DB_PATH.with_name("aside.db")
        os.replace(th._DB_PATH, aside)
        conn = get_db_at(
            th._LockedConnection.__init__.__code__, "return", 1,
            functools.partial(os.replace, aside, th._DB_PATH))
        conn.execute("INSERT INTO task_history (id, task, timestamp) VALUES ('new', 'new', 0)")
        assert self._tasks_in(th._DB_PATH) == ["dbdel kept", "new"]

    def test_reads_heal_after_same_path_recreation(self) -> None:
        _add_task("dbdel recreate read-heal", chat_id="")
        self._delete_db_files()
        sqlite3.connect(str(th._DB_PATH)).close()
        entries = th._load_history()
        assert entries == []

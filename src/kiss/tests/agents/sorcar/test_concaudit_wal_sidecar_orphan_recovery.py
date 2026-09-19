# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_get_db`` must recover when the live ``-wal``/``-shm`` are unlinked.

Reproduces the recurring production ``sqlite3.OperationalError: disk
I/O error`` (``SQLITE_IOERR_SHORT_READ``) that made every NEW task fail
within 100 ms of starting, and the "WAL loss" that followed a daemon
restart.  The daemon at PID 2514112 was found holding sixteen file
descriptors to ``sorcar.db-wal (deleted)`` and ``sorcar.db-shm
(deleted)``: something outside the daemon had unlinked the sidecars.

SQLite maps ONE ``-shm`` per process per database inode (shared by every
connection in the process) but opens the ``-wal`` by *name* for each
connection.  Once the sidecars are unlinked:

* connections that were already open keep working — against a deleted
  inode, so every frame they commit is lost on a hard kill;
* every NEW connection in the same process (a new task thread, an
  executor thread, the event writer after a restart) inherits the old
  ``-shm`` mapping but opens a fresh, empty ``-wal`` — the two disagree
  and the connection fails with ``disk I/O error``;
* other processes see a database missing the frames still only in the
  deleted WAL.

The fix makes ``_get_db`` notice that the ``-shm`` on disk is not the one
this process is attached to (or that a fresh connection failed with
``SQLITE_IOERR``), fold the process's committed frames into the main
file with a checkpoint while the old mapping is still valid, close EVERY
open connection so SQLite drops the stale mapping, and reconnect.

No mocks: real SQLite files, real threads, real ``os.unlink``.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.tests.conftest import posix_only


def _redirect(tmpdir: str):
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _unlink_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        os.unlink(str(db_path) + suffix)


def _external_count(db_path: Path) -> int:
    """Count ``task_history`` rows the way a separate process sees them."""
    code = (
        "import sqlite3, sys;"
        f"c = sqlite3.connect({str(db_path)!r});"
        "print(c.execute('SELECT count(*) FROM task_history').fetchone()[0]);"
        "c.close()"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        timeout=60, check=True,
    )
    return int(out.stdout.strip())


@posix_only("unlinking WAL sidecars another connection holds open")
class TestWalSidecarOrphanRecovery:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        th._flush_chat_events()
        th._close_db()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        th._flush_chat_events()
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _add_in_thread(self, task: str) -> tuple[str | None, BaseException | None]:
        """Run ``_add_task`` on a brand-new thread (a fresh connection)."""
        result: list = [None, None]

        def run() -> None:
            try:
                result[0] = th._add_task(task)[0]
            except BaseException as exc:  # noqa: BLE001 — recorded for the assertion
                result[1] = exc
            finally:
                th._close_thread_db()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=60)
        assert not t.is_alive(), "task thread hung"
        return result[0], result[1]

    def test_new_thread_connection_survives_sidecar_unlink(self):
        """A fresh connection after the unlink must work and lose nothing.

        Without the fix the new thread's ``_get_db`` raises
        ``sqlite3.OperationalError: disk I/O error`` (SQLITE_IOERR_SHORT_READ).
        """
        first_id, _chat = th._add_task("before the unlink")
        th._append_chat_event({"type": "response", "text": "in the WAL"},
                              task_id=first_id)
        th._flush_chat_events()
        db_path = th._DB_PATH
        assert os.path.exists(str(db_path) + "-shm")

        _unlink_sidecars(db_path)
        # The main thread's connection is now on a deleted inode; keep
        # committing so there are frames only the deleted WAL holds.
        second_id, _chat = th._add_task("committed into the deleted WAL")

        third_id, exc = self._add_in_thread("after the unlink, new thread")
        assert exc is None, f"new connection failed: {exc!r}"
        assert third_id is not None

        # Every commit — including the ones made against the deleted
        # WAL — is visible to this process and to a separate process.
        tasks = {h["task"] for h in th._load_history()}
        assert tasks == {
            "before the unlink",
            "committed into the deleted WAL",
            "after the unlink, new thread",
        }
        assert _external_count(db_path) == 3
        loaded = th._load_chat_events_by_task_id(first_id)
        assert loaded is not None
        events = loaded["events"]
        assert isinstance(events, list)
        assert any(
            isinstance(e, dict) and e.get("text") == "in the WAL"
            for e in events
        )
        # The process is attached to the sidecars that exist on disk again.
        assert os.path.exists(str(db_path) + "-shm")
        assert second_id != third_id

    def test_main_thread_connection_is_reconnected_too(self):
        """The thread that detects the orphaning gets a healthy connection.

        After recovery the detecting thread's own connection was closed;
        ``_get_db`` must hand it a fresh one and writes must keep landing.
        """
        th._add_task("seed")
        db_path = th._DB_PATH
        _unlink_sidecars(db_path)
        # Same thread, next _get_db(): the -shm on disk is gone while the
        # process is still attached to the deleted one.
        db = th._get_db()
        db.execute("SELECT count(*) FROM task_history").fetchone()
        th._add_task("after recovery on the same thread")
        assert _external_count(db_path) == 2

    def test_replaced_sidecars_from_another_process_are_adopted(self):
        """Sidecars unlinked AND recreated by another process are detected.

        A separate process opening the database after the unlink creates
        fresh ``-wal``/``-shm`` files; their identity differs from the
        one this process is mapped to, so the same recovery applies.
        """
        th._add_task("seed")
        db_path = th._DB_PATH
        _unlink_sidecars(db_path)
        # Another process opens the database: it recreates -wal/-shm
        # (and, being cut off from this process's WAL, may not even see
        # the schema yet — exactly the stale view the bug produced).
        subprocess.run(
            [sys.executable, "-c",
             f"import sqlite3; sqlite3.connect({str(db_path)!r})"
             ".execute('PRAGMA journal_mode').fetchone()"],
            check=True, timeout=60,
        )
        assert os.path.exists(str(db_path) + "-shm")
        _task_id, exc = self._add_in_thread("new thread after replacement")
        assert exc is None, f"new connection failed: {exc!r}"
        assert _external_count(db_path) == 2

    def test_ioerr_on_fresh_connection_triggers_recovery(self):
        """The identity check can lose a race; the open error must not.

        Sequence in production: ``_get_db`` stats an intact ``-shm``,
        another process unlinks and recreates the sidecars, then the
        ``sqlite3.connect`` that follows fails with ``SQLITE_IOERR``
        because the process is still mapped to the old ``-shm``.  The
        window cannot be hit deterministically from outside, so the
        resulting state is set up directly: the process is mapped to the
        deleted ``-shm`` while the recorded identity already names the
        new one on disk, so the identity check passes and only the
        ``SQLITE_IOERR`` path can recover.
        """
        th._add_task("seed")
        db_path = th._DB_PATH
        _unlink_sidecars(db_path)
        subprocess.run(
            [sys.executable, "-c",
             f"import sqlite3; sqlite3.connect({str(db_path)!r})"
             ".execute('PRAGMA journal_mode').fetchone()"],
            check=True, timeout=60,
        )
        attached = th._attached_shm.get(str(db_path))
        assert attached is not None
        db_id = attached[0]
        new_shm_id = th._db_file_identity(str(db_path) + "-shm")
        assert new_shm_id is not None
        th._attached_shm[str(db_path)] = (db_id, new_shm_id)
        assert not th._sidecars_orphaned(str(db_path), db_id)
        _task_id, exc = self._add_in_thread("new thread, IOERR path")
        assert exc is None, f"new connection failed: {exc!r}"
        assert _external_count(db_path) == 2

    def test_event_writer_thread_recovers(self):
        """The background event writer (a long-lived thread) recovers too."""
        task_id, _chat = th._add_task("events task")
        th._append_chat_event({"type": "response", "text": "one"},
                              task_id=task_id)
        th._flush_chat_events()
        _unlink_sidecars(th._DB_PATH)
        th._append_chat_event({"type": "response", "text": "two"},
                              task_id=task_id)
        th._flush_chat_events()
        # A new thread must be able to read both events.
        got: list = []

        def read() -> None:
            try:
                loaded = th._load_chat_events_by_task_id(task_id)
                got.append(loaded)
            finally:
                th._close_thread_db()

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout=60)
        assert got and got[0] is not None
        texts = [e.get("text") for e in got[0]["events"] if isinstance(e, dict)]
        assert texts == ["one", "two"]

    def test_plain_sqlite_reproduces_the_failure_mode(self):
        """Document the raw SQLite behaviour the fix is built around."""
        path = os.path.join(self.tmpdir, "raw.db")
        c1 = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        c1.execute("PRAGMA journal_mode=WAL")
        c1.execute("CREATE TABLE t(x)")
        c1.execute("INSERT INTO t VALUES (1)")
        os.unlink(path + "-wal")
        os.unlink(path + "-shm")
        c1.execute("INSERT INTO t VALUES (2)")  # old connection still works
        c2 = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        try:
            c2.execute("PRAGMA journal_mode=WAL")
            c2.execute("SELECT count(*) FROM t").fetchone()
        except sqlite3.OperationalError as exc:
            assert "disk I/O error" in str(exc)
            assert exc.sqlite_errorcode & 0xFF == sqlite3.SQLITE_IOERR
        else:
            raise AssertionError("expected SQLITE_IOERR on the new connection")
        finally:
            c2.close()
            c1.close()

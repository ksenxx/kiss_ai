# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Orphaned-sidecar recovery must never close a handle another thread is using.

``_recover_orphaned_sidecars`` has to close every connection of the
process to drop a dead ``-shm`` mapping (see
``test_concaudit_wal_sidecar_orphan_recovery.py``).  It used to close
other threads' handles outright, and closing a ``sqlite3.Connection``
while its owner is inside ``sqlite3_step`` crashes CPython.  Now every
statement step holds the connection's lock (``_LockedConnection``) and
recovery closes a handle only once it can take that lock; a handle kept
busy beyond ``_RECOVERY_IDLE_WAIT_S`` stays with its owner, who retires
it on its next ``_get_db()``.

Real threads, real SQLite files, real ``os.unlink``; no mocks.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.tests.conftest import posix_only

_SLOW_COUNT = (
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 500000000) "
    "SELECT count(*) FROM c"
)
_THOUSAND_ROWS = (
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 1000) SELECT x FROM c"
)


def _unlink_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        os.unlink(str(db_path) + suffix)


def _open_file_links() -> list[str]:
    """Targets of this process's open file descriptors (Linux ``/proc``)."""
    links = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            links.append(os.readlink(f"/proc/self/fd/{fd}"))
        except FileNotFoundError:  # the listing's own descriptor, already closed
            pass
    return links


def _sleep_two_seconds() -> int:
    """A SQL function whose single step outlasts a short recovery wait."""
    time.sleep(2.0)
    return 1


@posix_only("unlinking WAL sidecars another connection holds open")
class TestRecoveryWithLiveThreads:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        th._flush_chat_events()
        th._close_db()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        th._flush_chat_events()
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_handle_mid_statement_is_interrupted_not_closed_underneath(self) -> None:
        """Recovery interrupts a running statement, then closes the idle handle.

        Thread B is inside ``sqlite3_step`` of a multi-second query when the
        main thread recovers.  B's statement must fail with
        ``OperationalError: interrupted`` (never a crash), B's next
        ``_get_db()`` must hand it a working connection, and both threads'
        later writes must land.
        """
        th._add_task("seed")
        stepping = threading.Event()
        outcome: dict[str, object] = {}

        def run_b() -> None:
            try:
                db = th._get_db()
                stepping.set()
                try:
                    db.execute(_SLOW_COUNT).fetchone()
                except sqlite3.OperationalError as exc:
                    outcome["error"] = exc
                outcome["after"] = th._add_task("B after recovery")[0]
            except BaseException as exc:  # noqa: BLE001 — recorded for the assertion
                outcome["fatal"] = exc
            finally:
                th._close_thread_db()

        b = threading.Thread(target=run_b, daemon=True)
        b.start()
        assert stepping.wait(timeout=30)
        time.sleep(0.3)  # B is now inside the step
        _unlink_sidecars(th._DB_PATH)
        started = time.monotonic()
        th._get_db()  # recovery: interrupt B, take its lock, close, reconnect
        assert time.monotonic() - started < th._RECOVERY_IDLE_WAIT_S
        th._add_task("main after recovery")
        b.join(timeout=60)
        assert not b.is_alive(), "thread B hung"
        assert "fatal" not in outcome, repr(outcome.get("fatal"))
        assert "interrupted" in str(outcome["error"])
        assert outcome["after"] is not None
        tasks = {h["task"] for h in th._load_history()}
        assert tasks == {"seed", "B after recovery", "main after recovery"}

    def test_handle_idle_between_rows_is_closed_and_owner_reconnects(self) -> None:
        """A handle paused between two rows is idle: recovery closes it.

        Thread B's next row then raises ``ProgrammingError`` (the documented
        fail-once), and B's next ``_get_db()`` reconnects.
        """
        th._add_task("seed")
        first_row = threading.Event()
        resume = threading.Event()
        outcome: dict[str, object] = {}

        def run_b() -> None:
            try:
                rows: list[int] = []
                cursor = th._get_db().execute(_THOUSAND_ROWS)
                try:
                    for row in cursor:
                        rows.append(row[0])
                        if len(rows) == 1:
                            first_row.set()
                            assert resume.wait(timeout=30)
                except sqlite3.ProgrammingError as exc:
                    outcome["error"] = exc
                outcome["rows"] = rows
                outcome["after"] = th._add_task("B after recovery")[0]
            except BaseException as exc:  # noqa: BLE001 — recorded for the assertion
                outcome["fatal"] = exc
            finally:
                th._close_thread_db()

        b = threading.Thread(target=run_b, daemon=True)
        b.start()
        assert first_row.wait(timeout=30)
        _unlink_sidecars(th._DB_PATH)
        th._get_db()
        resume.set()
        th._add_task("main after recovery")
        b.join(timeout=60)
        assert not b.is_alive(), "thread B hung"
        assert "fatal" not in outcome, repr(outcome.get("fatal"))
        assert "closed" in str(outcome["error"])
        assert outcome["rows"] == [1]
        assert outcome["after"] is not None
        tasks = {h["task"] for h in th._load_history()}
        assert tasks == {"seed", "B after recovery", "main after recovery"}

    def test_handle_busy_past_the_wait_is_left_to_its_owner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A step that outlasts the wait keeps the handle; its owner retires it.

        While the busy handle pins the dead mapping, nothing may be
        reopened (a new connection would join that mapping for good), so
        the recovering thread's ``_get_db()`` raises and leaves the handle
        registered under B.  B's statement ends with ``interrupted``, B's
        own next ``_get_db()`` completes the recovery, and from then on
        every thread connects to a live mapping again.
        """
        monkeypatch.setattr(th, "_RECOVERY_IDLE_WAIT_S", 0.3)
        th._add_task("seed")
        stepping = threading.Event()
        outcome: dict[str, object] = {}

        def run_b() -> None:
            try:
                db = th._get_db()
                db.create_function("sleep_two_seconds", 0, _sleep_two_seconds)
                stepping.set()
                try:
                    db.execute("SELECT sleep_two_seconds()").fetchone()
                except sqlite3.OperationalError as exc:
                    outcome["error"] = exc
                outcome["after"] = th._add_task("B after recovery")[0]
            except BaseException as exc:  # noqa: BLE001 — recorded for the assertion
                outcome["fatal"] = exc
            finally:
                th._close_thread_db()

        b = threading.Thread(target=run_b, daemon=True)
        b.start()
        assert stepping.wait(timeout=30)
        time.sleep(0.3)  # B is now inside the sleeping step
        _unlink_sidecars(th._DB_PATH)
        with pytest.raises(sqlite3.OperationalError, match="still busy"):
            th._get_db()
        assert b.is_alive(), "B's statement must still be running"
        assert any(owner is b for _conn, owner, _path in th._open_conns.values()), (
            "B's busy handle must stay registered for B to retire"
        )
        assert str(th._DB_PATH) in th._attached_shm, "the recovery must stay pending"
        b.join(timeout=60)
        assert not b.is_alive(), "thread B hung"
        assert "fatal" not in outcome, repr(outcome.get("fatal"))
        assert "interrupted" in str(outcome["error"])
        assert outcome["after"] is not None
        assert not any(owner is b for _conn, owner, _path in th._open_conns.values())
        th._add_task("main after B reconnected")
        tasks = {h["task"] for h in th._load_history()}
        assert tasks == {"seed", "B after recovery", "main after B reconnected"}
        if os.path.isdir("/proc/self/fd"):  # the dead mapping is really gone
            assert f"{th._DB_PATH}-shm (deleted)" not in _open_file_links(), _open_file_links()

    def test_context_manager_and_scripts_run_under_the_lock(self) -> None:
        """``with conn:`` and ``executescript`` go through the locked wrappers."""
        db = th._get_db()
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS r2_probe (x INTEGER)")
        db.cursor().executescript(
            "INSERT INTO r2_probe VALUES (1); INSERT INTO r2_probe VALUES (2);"
        )
        assert [row[0] for row in db.execute("SELECT x FROM r2_probe ORDER BY x")] == [1, 2]

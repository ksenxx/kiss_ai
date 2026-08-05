# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The startup orphan sweep must finish before a test closes the DB.

Every :class:`~kiss.server.server.VSCodeServer` constructor starts a
daemon thread named ``orphan-task-sweep`` that rewrites the
``"Agent Failed Abruptly"`` sentinel of tasks killed with their owning
process (``persistence._recover_orphaned_tasks``).  The sweep runs on
its own per-thread SQLite connection, and ``persistence._get_db()``
also publishes that connection in the module-global ``_db_conn``.

Roughly 150 test files across ``tests/agents/sorcar`` and
``tests/agents/vscode`` construct a server in ``setUp`` and then, in
``tearDown``, close ``persistence._db_conn`` and delete the temporary
KISS_HOME.  When the sweep is still inside ``db.execute(...)`` at that
moment, the C-level ``pysqlite_connection_execute`` call dereferences a
freed connection and the whole interpreter dies with SIGSEGV — a crash
that takes down the entire pytest process, not just one test.  It was
observed in the wild as::

    Fatal Python error: Segmentation fault
    Current thread [orphan-task-sweep] (most recent call first):
      File "src/kiss/agents/sorcar/persistence.py", line 1193
        in _recover_orphaned_tasks
    Thread 0x1f5861d80 (most recent call first):
      File "src/kiss/tests/.../test_bughunt6_ghost_quote_suffix.py",
        line 72 in tearDown

The root ``tests/conftest.py`` therefore joins every live sweep thread
BEFORE each ``unittest.TestCase.tearDown`` body runs, while the
connection is still valid.  This test pins that guarantee.

The interleaving is made deterministic without any mock: the test body
holds the real writer lock the sweep needs
(``persistence._rw_lock.write_lock()``), so the sweep parks *before*
its first statement; releasing the lock at the end of the test body
leaves the sweep runnable but unfinished, exactly the window in which
``tearDown`` used to pull the connection out from under it.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server.server import VSCodeServer

SENTINEL = "Agent Failed Abruptly"
RECOVERED = "Task terminated unexpectedly (process killed)"


def _live_sweep_threads() -> list[threading.Thread]:
    """Return every still-running ``orphan-task-sweep`` thread."""
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "orphan-task-sweep" and thread.is_alive()
    ]


class TestOrphanSweepJoinedBeforeTearDown(unittest.TestCase):
    """A test's ``tearDown`` never races the startup orphan sweep."""

    # Large enough that the sweep (one forensics log line plus one
    # UPDATE row per sentinel) still has work left when the lock is
    # released, small enough to stay a few milliseconds.
    BACKLOG = 300

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved: tuple[Any, Any, Any] = (
            th._DB_PATH, th._db_conn, th._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        stale = time.time() - 600
        th._get_db().executemany(
            "INSERT INTO task_history (task, timestamp, result) "
            "VALUES (?, ?, ?)",
            [(f"orphan {i}", stale, SENTINEL) for i in range(self.BACKLOG)],
        )
        th._get_db().commit()

    def tearDown(self) -> None:
        # Recorded before any cleanup: the harness must already have
        # joined the sweep by the time this body runs.
        still_sweeping = [thread.name for thread in _live_sweep_threads()]
        try:
            recovered = th._get_db().execute(
                "SELECT COUNT(*) FROM task_history WHERE result = ?",
                (RECOVERED,),
            ).fetchone()[0]
        finally:
            if th._db_conn is not None:
                th._db_conn.close()
            th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.assertEqual(
            still_sweeping, [],
            "the orphan-task sweep was still running SQL when tearDown "
            "closed the database connection — that is the SIGSEGV race",
        )
        self.assertEqual(
            recovered, self.BACKLOG,
            "the joined sweep must have rewritten every sentinel row "
            f"before tearDown closed the connection; got {recovered}",
        )

    def test_startup_sweep_finishes_before_the_test_closes_the_db(
        self,
    ) -> None:
        """The sweep is asynchronous, yet complete by ``tearDown``."""
        with th._rw_lock.write_lock():
            self.server = VSCodeServer()
            self.assertTrue(
                _live_sweep_threads(),
                "VSCodeServer must sweep orphaned tasks on a background "
                "thread so startup does not block on the database",
            )
            self.assertEqual(
                th._get_db().execute(
                    "SELECT COUNT(*) FROM task_history WHERE result = ?",
                    (SENTINEL,),
                ).fetchone()[0],
                self.BACKLOG,
                "the parked sweep must not have rewritten anything yet",
            )
        # The sweep is now runnable but unfinished; tearDown follows
        # immediately.
        self.assertTrue(
            _live_sweep_threads(),
            "the released sweep must still be in flight, otherwise this "
            "test no longer exercises the tearDown race",
        )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: ``_get_db()`` must never unlink a LIVE database's
``-wal``/``-shm`` side files.

Before the fix, ``_get_db()``'s stale-side-file cleanup re-read the
``_DB_PATH`` global between the "main file missing?" existence check
and the unlink of the ``-wal``/``-shm`` targets — a TOCTOU race.  Any
test (or embedder) that redirects ``_DB_PATH`` to a scratch directory,
deletes that scratch database, and restores the original path could
interleave so that this thread observed "missing" for the SCRATCH
file but computed the unlink targets from the freshly RESTORED shared
path — deleting the live shared database's ``-shm`` out from under
every open connection.  SQLite then fails every NEW connection to the
shared database with a permanent ``sqlite3.OperationalError: disk I/O
error`` for as long as any old connection keeps the unlinked ``-shm``
mapped (e.g. the long-lived ``kiss-event-writer`` thread).  This was
the shared root cause of the order-dependent ``disk I/O error``
failures across the vscode-server test suites in large parallel runs.

Runs against real SQLite databases and real threads — no mocks,
patches, fakes, or test doubles.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _add_task


class TestWalUnlinkToctou:
    """Hammer the redirect/delete/restore pattern against ``_get_db()``."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss_toctou_")
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved

    def test_shared_db_survives_concurrent_scratch_redirects(self) -> None:
        """The shared DB's ``-shm`` must never be unlinked (and new
        connections must never see ``disk I/O error``) while other
        threads redirect ``_DB_PATH`` to scratch dirs, delete the
        scratch DB files, and restore the shared path — the exact
        pattern used by persistence tests like
        ``test_persistence_db_deleted_file``."""
        shared_path = th._DB_PATH
        shared_kiss_dir = th._KISS_DIR
        _add_task("toctou seed", chat_id="")
        shm = Path(str(shared_path) + "-shm")
        assert shm.exists()
        shm_ino = os.stat(shm).st_ino

        stop = threading.Event()
        problems: list[str] = []

        def redirect_delete_restore() -> None:
            """Emulate a test that redirects, deletes, and restores."""
            while not stop.is_set():
                scratch = Path(tempfile.mkdtemp(prefix="kiss_toctou_s_"))
                kd = scratch / ".kiss"
                kd.mkdir()
                th._DB_PATH = kd / "sorcar.db"
                th._KISS_DIR = kd
                try:
                    _add_task("scratch row", chat_id="")
                except Exception:  # noqa: BLE001 — path swap races are expected here
                    pass
                for suffix in ("", "-wal", "-shm"):
                    Path(str(kd / "sorcar.db") + suffix).unlink(missing_ok=True)
                time.sleep(0.0005)
                th._DB_PATH = shared_path
                th._KISS_DIR = shared_kiss_dir

        def background_get_db() -> None:
            """Emulate background threads (event writer, orphan sweep)
            repeatedly opening fresh per-thread connections."""
            while not stop.is_set():
                try:
                    th._get_db()
                except Exception:  # noqa: BLE001 — path swap races are expected here
                    pass
                th._close_thread_db()
                time.sleep(0.0002)

        workers = [
            threading.Thread(target=redirect_delete_restore),
            threading.Thread(target=background_get_db),
            threading.Thread(target=background_get_db),
        ]
        for w in workers:
            w.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not problems:
                try:
                    if os.stat(shm).st_ino != shm_ino:
                        problems.append("shared -shm was unlinked and recreated")
                except FileNotFoundError:
                    problems.append("shared -shm was unlinked")
                time.sleep(0.005)
        finally:
            stop.set()
            for w in workers:
                w.join()

        th._DB_PATH = shared_path
        th._KISS_DIR = shared_kiss_dir
        assert not problems, problems[0]

        # A brand-new thread (fresh per-thread connection) must still be
        # able to use the shared database — before the fix this raised
        # ``sqlite3.OperationalError: disk I/O error`` permanently.
        health: list[str] = []

        def probe() -> None:
            try:
                _add_task("toctou health check", chat_id="")
                health.append("ok")
            except Exception as exc:  # noqa: BLE001 — the failure IS the regression
                health.append(f"failed: {exc}")

        p = threading.Thread(target=probe)
        p.start()
        p.join()
        assert health == ["ok"]

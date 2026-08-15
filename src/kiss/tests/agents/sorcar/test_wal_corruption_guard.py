# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The live WAL must never be destroyed by ``_get_db``'s cleanup path.

Reproduces the 2026-08-15 production corruption of ``sorcar.db``: under
heavy load a transient ``os.stat`` failure on the database path made
``_db_file_identity`` report "file does not exist", so every thread
tore down its healthy connection and the reconnect path deleted the
LIVE ``-wal``/``-shm`` sidecars out from under the remaining
connections — corrupting the database ("file is not a database",
"database disk image is malformed") and losing every commit still in
the WAL.

The fix has two parts, each tested here end-to-end (no mocks):

* ``_db_file_identity`` treats only a confirmed ``FileNotFoundError``
  as "absent"; any other ``OSError`` (EACCES from an unreadable parent
  directory in these tests) returns a distinct "unknown" sentinel and
  the cached connection keeps serving.
* ``_get_db`` no longer unlinks ``-wal``/``-shm`` sidecars AT ALL —
  check-then-unlink can never be made atomic against other threads
  and processes opening the same database, and SQLite itself heals a
  stale sidecar of a deleted-and-recreated database safely.
"""

import gc
import os
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str):
    """Redirect the DB to a temp dir and reset the singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestWalCorruptionGuard:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        th._flush_chat_events()
        th._close_db()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        os.chmod(Path(self.tmpdir) / ".kiss", 0o700)
        th._flush_chat_events()
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed_task(self) -> str:
        task_id, _chat_id = th._add_task("wal guard probe task")
        th._append_chat_event(
            {"type": "response", "text": "committed via WAL"},
            task_id=task_id,
        )
        th._flush_chat_events()
        return task_id

    def test_transient_stat_failure_keeps_cached_connection(self):
        """EACCES on the db directory must not tear down healthy conns.

        Before the fix, ``_db_file_identity`` mapped the EACCES to
        "file does not exist": the cached connection was closed and the
        reconnect raised ``sqlite3.OperationalError`` (and, in
        production, deleted the live WAL).  After the fix the cached
        connection keeps serving reads and writes throughout the
        outage.
        """
        if os.geteuid() == 0:
            pytest.skip("permission checks are bypassed for root")
        task_id = self._seed_task()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        os.chmod(kiss_dir, 0)
        try:
            # Reads served by the cached connection despite stat EACCES.
            history = th._load_history()
            assert [h["task"] for h in history] == ["wal guard probe task"]
            loaded = th._load_chat_events_by_task_id(task_id)
            assert loaded is not None
            events = loaded["events"]
            assert isinstance(events, list)
            assert any(
                isinstance(e, dict) and e.get("text") == "committed via WAL"
                for e in events
            )
            # Writes too: the result lands in the same database.
            th._save_task_result("survived the outage", task_id=task_id)
        finally:
            os.chmod(kiss_dir, 0o700)
        history = th._load_history()
        assert history[0]["result"] == "survived the outage"
        # The sidecars of the healthy database were never unlinked.
        assert os.path.exists(str(th._DB_PATH) + "-wal")

    def test_wal_not_unlinked_while_other_thread_holds_connection(self):
        """A confirmed-missing db file must not cost another thread its WAL.

        Thread A holds an open connection (its WAL may contain
        committed, uncheckpointed pages).  The main db file is renamed
        away and replaced with a DANGLING SYMLINK, so a reconnecting
        thread B sees a confirmed ENOENT — the exact trigger of the
        old unlink — while B's own SQLite sidecars resolve to the
        symlink target's name, leaving A's ``sorcar.db-wal`` for
        application code alone to touch.  Before the fix, B unlinked
        it even though A still had it open; after the fix (which
        removes application-level sidecar deletion entirely) it
        survives untouched (same inode).
        """
        self._seed_task()
        wal_path = str(th._DB_PATH) + "-wal"
        assert os.path.exists(wal_path)
        wal_inode_before = os.stat(wal_path).st_ino

        hold = threading.Event()
        release = threading.Event()

        def thread_a():
            th._get_db()
            hold.set()
            release.wait(timeout=30)
            th._close_thread_db()

        def thread_b():
            # B connects to the symlink target (a fresh database); what
            # the fix guarantees is that its reconnect attempt never
            # unlinks the sidecars thread A still holds.
            try:
                th._get_db()
            except Exception:
                pass
            finally:
                th._close_thread_db()

        a = threading.Thread(target=thread_a)
        a.start()
        try:
            assert hold.wait(timeout=30)
            db_path = str(th._DB_PATH)
            hidden = db_path + ".hidden"
            target = db_path + ".gone"
            os.rename(db_path, hidden)
            os.symlink(target, db_path)
            try:
                b = threading.Thread(target=thread_b)
                b.start()
                b.join(timeout=30)
                assert not b.is_alive()
                assert os.path.exists(wal_path), (
                    "live WAL was unlinked while another thread still "
                    "held an open connection"
                )
                assert os.stat(wal_path).st_ino == wal_inode_before
            finally:
                # Put the original database back for teardown.
                os.unlink(db_path)
                for leftover in (target, target + "-wal", target + "-shm"):
                    if os.path.exists(leftover):
                        os.unlink(leftover)
                os.rename(hidden, db_path)
        finally:
            release.set()
            a.join(timeout=30)
        assert not a.is_alive()

    def test_sqlite_heals_stale_sidecars_without_manual_unlink(self):
        """A deleted db with leftover sidecars must heal without unlink.

        ``_get_db`` deliberately contains NO application-level sidecar
        cleanup any more — check-then-unlink can never be made safe
        against other threads and processes.  This test pins the
        property that makes that removal correct: SQLite itself
        ignores/resets a stale ``-wal``/``-shm`` (garbage here) when a
        fresh database is created at the same path, so persistence
        keeps working with no manual deletion.
        """
        self._seed_task()
        db_path = str(th._DB_PATH)
        th._close_db()
        gc.collect()
        os.unlink(db_path)
        Path(db_path + "-wal").write_bytes(b"stale garbage, not a real WAL")
        Path(db_path + "-shm").write_bytes(b"stale garbage")
        task_id, _chat_id = th._add_task("post-stale-sidecar task")
        th._flush_chat_events()
        history = th._load_history()
        assert [h["task"] for h in history] == ["post-stale-sidecar task"]
        assert history[0]["id"] == task_id

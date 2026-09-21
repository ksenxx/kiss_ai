# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A-C1: a failing background file-cache refresh must not strand the
``@``-mention picker or leak the connection's request token.

``_refresh_file_cache._do_refresh`` runs on a daemon thread and used
to call ``_load_file_usage()`` (a raw SQLite read) unguarded: a
database failure — here made real by replacing the redirected
``sorcar.db`` path with a **directory**, so ``sqlite3`` cannot open it
— killed the thread through the silent default excepthook.  The
populated ``files`` reply was never emitted (picker stuck on its
``loading`` placeholder) and the connection's ``_files_latest_request``
token was never popped, violating the map's short-lived contract.
``_refresh_files_after_task._do_refresh`` had the same unguarded
shape (its scan raises ``UnicodeDecodeError`` on a ``.gitignore``
holding invalid UTF-8 — ``_load_gitignore_dirs`` catches only
``OSError``).  ``VSCodeServer.drop_connection_state`` also never
cleaned the token map, so a connection departing with a scan still in
flight leaked its entry forever.

All failures here are real on-disk faults (no mocks, no patches of the
code under test): a directory where the database file belongs, and a
non-UTF-8 ``.gitignore``.  Thread death is observed through a
recording ``threading.excepthook`` installed by the test around the
refresh — pre-fix the hook captures the escaped exception; post-fix
nothing escapes.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as _persistence
from kiss.server.server import VSCodeServer
from kiss.tests.conftest import (
    nproc_limit_lowered_to_one,
    thread_start_can_be_starved,
)
from kiss.tests.server._memory_printer import MemoryPrinter


class _LogWaiter(logging.Handler):
    """A real logging handler that signals when a message arrives.

    The guarded refresh workers report failure through
    ``logger.exception`` — observing that record is a deterministic
    completion signal for the failure path (no sleeps, no vacuous
    pass when the worker never starts).
    """

    def __init__(self, needle: str) -> None:
        super().__init__()
        self.needle = needle
        self.seen = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        if self.needle in record.getMessage():
            self.seen.set()


class _RecordingExcepthook:
    """Capture unhandled thread exceptions raised during the test."""

    def __init__(self) -> None:
        self.records: list[BaseException] = []
        self._orig = threading.excepthook

    def __enter__(self) -> _RecordingExcepthook:
        def hook(args: threading.ExceptHookArgs) -> None:
            if args.exc_value is not None:
                self.records.append(args.exc_value)

        threading.excepthook = hook
        return self

    def __exit__(self, *exc: object) -> None:
        threading.excepthook = self._orig


class TestFilesRefreshFailure(unittest.TestCase):
    """Real server, real work dir, real (broken) persistence database."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-conc2026-ac1-")
        self.work_dir = str(Path(self.tmpdir) / "work")
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.work_dir, "hello.py").write_text("print('hi')\n")

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self.printer = MemoryPrinter()
        self.server = VSCodeServer(self.printer)
        self.server.work_dir = self.work_dir

    def tearDown(self) -> None:
        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- helpers ---------------------------------------------------

    def _break_database(self) -> None:
        """Make every fresh SQLite open of the redirected DB fail.

        A directory where the database file belongs is a real,
        durable open failure (``sqlite3.OperationalError: unable to
        open database file``) that needs no mocking.
        """
        db = Path(str(_persistence._DB_PATH))
        if db.is_file():
            db.unlink()
        db.mkdir(parents=True, exist_ok=True)

    def _heal_database(self) -> None:
        db = Path(str(_persistence._DB_PATH))
        if db.is_dir():
            shutil.rmtree(db)

    def _token_map(self) -> dict[str, object]:
        with self.server._state_lock:
            return dict(self.server._files_request_map())

    def _wait_token_released(self, conn_id: str, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if conn_id not in self._token_map():
                return
            time.sleep(0.01)
        raise AssertionError(
            f"_files_latest_request still holds {conn_id!r}: "
            f"{self._token_map()!r}",
        )

    def _files_events(self, conn_id: str) -> list[dict[str, Any]]:
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("type") == "files" and ev.get("connId") == conn_id
        ]

    # -- tests -----------------------------------------------------

    def test_failed_refresh_releases_token_and_next_request_completes(
        self,
    ) -> None:
        """The scan thread survives a real database failure, releases
        the connection's request token, and a later ``getFiles`` on
        the same connection is answered normally."""
        conn_id = "conn-ac1"
        self._break_database()
        with _RecordingExcepthook() as hook:
            self.server._handle_command({
                "type": "getFiles",
                "prefix": "hel",
                "workDir": self.work_dir,
                "connId": conn_id,
                "tabId": "tab-1",
            })
            # Pre-fix the daemon thread dies here (the hook records
            # the escaped sqlite error) and the token is never popped.
            self._wait_token_released(conn_id)
        self.assertEqual(
            hook.records, [],
            f"refresh thread died with {hook.records!r}",
        )
        # The loading placeholder was emitted; the populated reply was
        # legitimately skipped (the refresh failed), not wedged.
        events = self._files_events(conn_id)
        self.assertTrue(events and events[0].get("loading") is True)

        # With the database healthy again the SAME connection's next
        # request completes end-to-end (the failed scan still
        # published the directory listing to the cache, so this is
        # the cache-hit path exercising a fresh _load_file_usage).
        self._heal_database()
        self.server._handle_command({
            "type": "getFiles",
            "prefix": "hel",
            "workDir": self.work_dir,
            "connId": conn_id,
            "tabId": "tab-1",
        })
        self._wait_token_released(conn_id)
        populated = [
            ev for ev in self._files_events(conn_id)
            if not ev.get("loading")
        ]
        self.assertTrue(populated, "no populated files reply arrived")
        names = [
            f.get("path") or f.get("label") or str(f)
            for f in populated[-1].get("files", [])
        ]
        self.assertTrue(
            any("hello.py" in str(n) for n in names),
            f"hello.py missing from {names!r}",
        )

    def test_post_task_refresh_survives_a_raising_scan(self) -> None:
        """``_refresh_files_after_task``'s daemon thread must survive a
        real scan failure (non-UTF-8 ``.gitignore``) and leave the
        existing cache usable.

        Review finding 10 rewrite: the worker is synchronized on its
        own deterministic completion signals — the failure path's
        ``logger.exception`` record and the success path's cache
        update — so the test neither sleeps ten seconds on success nor
        passes vacuously when the worker never starts.
        """
        conn_id = "conn-ac1b"
        # Populate the cache for the work dir first (healthy paths).
        self.server._handle_command({
            "type": "getFiles",
            "prefix": "",
            "workDir": self.work_dir,
            "connId": conn_id,
            "tabId": "tab-1",
        })
        self._wait_token_released(conn_id)
        wd = self.server._resolve_work_dir(self.work_dir)
        with self.server._state_lock:
            cached_before = self.server._file_cache.get(wd)
        self.assertIsNotNone(cached_before)

        # A .gitignore holding invalid UTF-8 makes the next scan raise
        # UnicodeDecodeError (only OSError is swallowed downstream).
        Path(self.work_dir, ".gitignore").write_bytes(b"\xff\xfe\xff\n")
        waiter = _LogWaiter("post-task file-cache refresh failed")
        auto_logger = logging.getLogger("kiss.server.autocomplete")
        auto_logger.addHandler(waiter)
        self.addCleanup(auto_logger.removeHandler, waiter)
        with _RecordingExcepthook() as hook:
            self.server._refresh_files_after_task(self.work_dir)
            self.assertTrue(
                waiter.seen.wait(15),
                "the refresh worker never ran (or never completed its "
                "guarded failure path)",
            )
        self.assertEqual(
            hook.records, [],
            f"post-task refresh thread died with {hook.records!r}",
        )
        with self.server._state_lock:
            self.assertIs(
                self.server._file_cache.get(wd), cached_before,
                "a failed rescan must leave the existing cache alone",
            )

        # The cached listing still answers pickers.
        self.server._handle_command({
            "type": "getFiles",
            "prefix": "hel",
            "workDir": self.work_dir,
            "connId": conn_id,
            "tabId": "tab-1",
        })
        self._wait_token_released(conn_id)
        populated = [
            ev for ev in self._files_events(conn_id)
            if not ev.get("loading")
        ]
        self.assertTrue(populated, "picker wedged after failed refresh")

        # Success path, synchronized on the real completion condition
        # (the published cache update): heal the scan input, add a new
        # file, and the refresh worker must swap the cache entry.
        Path(self.work_dir, ".gitignore").unlink()
        Path(self.work_dir, "fresh_file.py").write_text("print('new')\n")
        self.server._refresh_files_after_task(self.work_dir)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with self.server._state_lock:
                current = self.server._file_cache.get(wd)
            if current is not cached_before and current is not None and any(
                "fresh_file.py" in str(entry) for entry in current
            ):
                break
            time.sleep(0.01)
        else:
            with self.server._state_lock:
                current = self.server._file_cache.get(wd)
            raise AssertionError(
                f"post-task refresh never published the rescan: {current!r}",
            )

    def test_failed_get_files_refresh_with_broken_scan_and_database(
        self,
    ) -> None:
        """The cache-miss refresh also survives when the scan itself
        raises (non-UTF-8 ``.gitignore``), covering the guard before
        ``_load_file_usage`` is even reached."""
        conn_id = "conn-ac1c"
        other = str(Path(self.tmpdir) / "work2")
        Path(other).mkdir(parents=True, exist_ok=True)
        Path(other, ".gitignore").write_bytes(b"\xff\xfe\xff\n")
        with _RecordingExcepthook() as hook:
            self.server._handle_command({
                "type": "getFiles",
                "prefix": "x",
                "workDir": other,
                "connId": conn_id,
                "tabId": "tab-1",
            })
            self._wait_token_released(conn_id)
        self.assertEqual(
            hook.records, [],
            f"refresh thread died with {hook.records!r}",
        )

    def test_superseded_failing_refresh_leaves_the_newer_token_alone(
        self,
    ) -> None:
        """A failing scan whose request was superseded must NOT pop the
        newer request's token (identity-checked error cleanup).

        The persistence writer lock (a real lock in the production
        read path of ``_load_file_usage``) holds the FIRST refresh
        thread at its database read until the same connection's second
        request has installed its own token and the database has been
        broken; both refreshes then fail — the stale one must leave
        the map alone, the latest one pops its own entry.
        """
        conn_id = "conn-ac1d"
        other = str(Path(self.tmpdir) / "work3")
        Path(other).mkdir(parents=True, exist_ok=True)
        with _RecordingExcepthook() as hook:
            with _persistence._rw_lock.write_lock():
                # First request: its refresh thread blocks at the
                # database read behind the held writer lock.
                self.server._handle_command({
                    "type": "getFiles",
                    "prefix": "a",
                    "workDir": self.work_dir,
                    "connId": conn_id,
                    "tabId": "tab-1",
                })
                # Second request (different work dir → cache miss)
                # installs the connection's NEW token synchronously
                # before its own scan thread starts.
                self.server._handle_command({
                    "type": "getFiles",
                    "prefix": "a",
                    "workDir": other,
                    "connId": conn_id,
                    "tabId": "tab-1",
                })
                self.assertIn(conn_id, self._token_map())
                self._break_database()
            # Lock released: both refresh threads now fail their
            # database read.  The stale one sees a foreign token and
            # stands down; the latest one pops its own.
            self._wait_token_released(conn_id)
        self.assertEqual(
            hook.records, [],
            f"refresh thread died with {hook.records!r}",
        )

    def test_cache_hit_database_failure_releases_the_token(self) -> None:
        """Review finding 9: the synchronous cache-hit path installs
        the request token, then reads the database
        (``_load_file_usage``) — a real open failure used to propagate
        BEFORE the token removal, stranding the token and wedging the
        connection's picker.  The identity-checked ``finally`` must
        release it even when the request fails."""
        conn_id = "conn-ac1e"
        # Populate the cache through the real scan (healthy database).
        self.server._get_files("", self.work_dir, conn_id, "tab-1")
        self._wait_token_released(conn_id)
        wd = self.server._resolve_work_dir(self.work_dir)
        with self.server._state_lock:
            self.assertIsNotNone(self.server._file_cache.get(wd))

        self._break_database()
        with self.assertRaises(Exception):
            self.server._get_files("hel", self.work_dir, conn_id, "tab-1")
        self.assertNotIn(
            conn_id, self._token_map(),
            "a failing cache-hit request stranded its token",
        )

        # With the database healthy again the SAME connection is
        # served normally (nothing wedged).
        self._heal_database()
        before = len(self._files_events(conn_id))
        self.server._get_files("hel", self.work_dir, conn_id, "tab-1")
        self._wait_token_released(conn_id)
        populated = [
            ev for ev in self._files_events(conn_id)[before:]
            if not ev.get("loading")
        ]
        self.assertTrue(populated, "picker wedged after the failure")

    def test_refresh_thread_launch_failure_releases_the_token(self) -> None:
        """Review finding 9 (second path): when the cache-miss refresh
        thread cannot start (real thread exhaustion via
        ``RLIMIT_NPROC``), the request token must be released — no
        worker will ever run the deferred cleanup — and the post-task
        refresh hook must not propagate the spawn failure into the
        task runner's cleanup."""
        if not thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        conn_id = "conn-ac1f"
        # Populate a cache entry so the post-task hook below has one.
        self.server._get_files("", self.work_dir, conn_id, "tab-1")
        self._wait_token_released(conn_id)
        other = str(Path(self.tmpdir) / "work-nproc")
        Path(other).mkdir(parents=True, exist_ok=True)

        raised: list[BaseException] = []
        with nproc_limit_lowered_to_one():
            try:
                # Cache miss for a new work dir: the refresh thread's
                # start() genuinely fails.
                self.server._get_files("x", other, conn_id, "tab-1")
            except BaseException as exc:  # noqa: BLE001 — the bug propagated here
                raised.append(exc)
            try:
                self.server._refresh_files_after_task(self.work_dir)
            except BaseException as exc:  # noqa: BLE001 — the bug propagated here
                raised.append(exc)

        self.assertEqual(
            raised, [],
            "a refresh-thread spawn failure escaped to the caller",
        )
        self.assertNotIn(
            conn_id, self._token_map(),
            "a failed refresh-thread launch stranded the request token",
        )
        # With threads available again the same connection completes.
        before = len(self._files_events(conn_id))
        self.server._get_files("x", other, conn_id, "tab-1")
        self._wait_token_released(conn_id)
        populated = [
            ev for ev in self._files_events(conn_id)[before:]
            if not ev.get("loading")
        ]
        self.assertTrue(populated, "picker wedged after the spawn failure")

    def test_drop_connection_state_clears_the_request_token(self) -> None:
        """A connection departing with a scan still in flight must not
        leave its ``_files_latest_request`` entry behind — and the
        empty ``conn_id`` shared by direct callers must survive."""
        with self.server._state_lock:
            reqs = self.server._files_request_map()
            reqs["conn-gone"] = object()
            reqs[""] = object()
        self.server.drop_connection_state("conn-gone")
        self.assertNotIn("conn-gone", self._token_map())
        self.assertIn("", self._token_map())
        # An empty id is ignored entirely.
        self.server.drop_connection_state("")
        self.assertIn("", self._token_map())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

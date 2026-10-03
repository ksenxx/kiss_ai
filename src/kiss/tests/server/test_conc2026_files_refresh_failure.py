# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A-C1: a failing ``@``-mention picker reply must not strand the picker
or leak the connection's request token.

The picker is served by :class:`kiss.server.file_index.FileIndexRegistry`.
A ``getFiles`` that finds the index of its work_dir ready is answered
synchronously by ``_get_files`` (the *warm* path); one that finds no
index emits a ``loading`` placeholder and has the registry's single
worker thread build the index and then run ``_emit_indexed_files`` (the
*cold* path).  Both paths read ``_load_file_usage()`` — a raw SQLite
read — between installing the connection's ``_files_latest_request``
token and emitting the reply.  A database failure — here made real by
redirecting the ``history.db`` path to a **directory**, so ``sqlite3``
cannot open it — used to escape from the reply path, leaving the token
in the map forever (violating its short-lived contract) and, on the
cold path, killing the thread through the silent default excepthook so
the picker stayed on its placeholder.

``_refresh_files_after_task`` (the post-task rescan) runs on the same
worker; a scan that trips over a corrupt ``.gitignore`` (invalid UTF-8
used to raise ``UnicodeDecodeError`` out of the scan) must neither
raise into the task runner's cleanup nor kill the worker.

All failures here are real on-disk faults (no mocks, no patches of the
code under test): a directory where the database file belongs, and a
non-UTF-8 ``.gitignore``.  Thread death is observed through a
recording ``threading.excepthook`` installed by the test around the
refresh — pre-fix the hook captures the escaped exception; post-fix
nothing escapes.  Every test uses a private registry rooted in the test's
temp dir so the real home directory is never scanned.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as _persistence
from kiss.server.file_index import FileIndex, FileIndexRegistry
from kiss.server.server import VSCodeServer
from kiss.tests.conftest import (
    nproc_limit_lowered_to_one,
    thread_start_can_be_starved,
)
from kiss.tests.server._memory_printer import MemoryPrinter

_REPLY_FAILED = "indexed file-picker reply failed"


class _LogWaiter(logging.Handler):
    """A real logging handler that signals when a message arrives.

    The guarded reply path reports failure through
    ``logger.exception`` — observing that record is a deterministic
    completion signal for the failure path (no sleeps, no vacuous
    pass when the worker never runs).
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
        self.db_path = kiss_dir / "history.db"
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = self.db_path
        _persistence._db_conn = None

        self.printer = MemoryPrinter()
        self.server = VSCodeServer(self.printer)
        self.server.work_dir = self.work_dir
        self.server._file_index.stop()
        self.registry = FileIndexRegistry(
            home=str(Path(self.tmpdir) / "home"),
            cache_dir=Path(self.tmpdir) / "cache",
        )
        self.server._file_index = self.registry

    def tearDown(self) -> None:
        self.registry.stop()
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
        open database file``) that needs no mocking.  The database
        path is redirected to a sibling directory rather than the
        healthy ``history.db`` being replaced in place: ``_get_db``
        treats the path change exactly like an on-disk replacement
        (every thread's cached connection is stale and the reconnect
        fails), while unlinking the file is refused on Windows
        (``PermissionError`` WinError 32) for as long as any thread's
        SQLite connection still holds it open.
        """
        broken = self.db_path.with_name("broken-history.db")
        broken.mkdir(exist_ok=True)
        _persistence._DB_PATH = broken

    def _heal_database(self) -> None:
        _persistence._DB_PATH = self.db_path

    def _token_map(self) -> dict[str, object]:
        with self.server._state_lock:
            return dict(self.server._files_request_map())

    def _wait(self, predicate: Callable[[], bool], timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def _wait_token_released(self, conn_id: str, timeout: float = 15.0) -> None:
        if not self._wait(lambda: conn_id not in self._token_map(), timeout):
            raise AssertionError(
                f"_files_latest_request still holds {conn_id!r}: "
                f"{self._token_map()!r}",
            )

    def _index_of(self, work_dir: str) -> FileIndex | None:
        root, _ = self.registry.root_for(work_dir)
        with self.registry._lock:
            return self.registry._indexes.get(root)

    def _build_index(self, work_dir: str) -> FileIndex:
        """Index *work_dir* through the registry and return the index."""
        done = threading.Event()
        self.registry.ensure(work_dir, done.set)
        self.assertTrue(done.wait(15), "the index build never finished")
        index = self._index_of(work_dir)
        self.assertIsNotNone(index)
        assert index is not None
        return index

    def _files_events(self, conn_id: str) -> list[dict[str, Any]]:
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("type") == "files" and ev.get("connId") == conn_id
        ]

    def _populated(self, conn_id: str, since: int = 0) -> list[dict[str, Any]]:
        return [
            ev for ev in self._files_events(conn_id)[since:]
            if not ev.get("loading")
        ]

    def _reply_failure_waiter(self) -> _LogWaiter:
        waiter = _LogWaiter(_REPLY_FAILED)
        auto_logger = logging.getLogger("kiss.server.autocomplete")
        auto_logger.addHandler(waiter)
        self.addCleanup(auto_logger.removeHandler, waiter)
        return waiter

    # -- tests -----------------------------------------------------

    def test_failed_refresh_releases_token_and_next_request_completes(
        self,
    ) -> None:
        """The cold path survives a real database failure on the worker
        thread, releases the connection's request token, and a later
        ``getFiles`` on the same connection is answered normally."""
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
            # Pre-fix the worker died here (the hook records the
            # escaped sqlite error) and the token was never popped.
            self._wait_token_released(conn_id)
        self.assertEqual(
            hook.records, [],
            f"index worker died with {hook.records!r}",
        )
        # The loading placeholder was emitted; the populated reply was
        # legitimately skipped (the usage read failed), not wedged.
        events = self._files_events(conn_id)
        self.assertTrue(events and events[0].get("loading") is True)
        self.assertEqual(self._populated(conn_id), [])

        # With the database healthy again the SAME connection's next
        # request completes end-to-end.  The failed reply still left
        # the built index in place, so this is the warm synchronous
        # path exercising a fresh _load_file_usage.
        self._heal_database()
        self.assertIsNotNone(self.registry.view_for(self.work_dir))
        self.server._handle_command({
            "type": "getFiles",
            "prefix": "hel",
            "workDir": self.work_dir,
            "connId": conn_id,
            "tabId": "tab-1",
        })
        self._wait_token_released(conn_id)
        populated = self._populated(conn_id)
        self.assertTrue(populated, "no populated files reply arrived")
        names = [f.get("text", str(f)) for f in populated[-1].get("files", [])]
        self.assertIn("./hello.py", names)

    def test_post_task_refresh_survives_a_corrupt_gitignore(self) -> None:
        """``_refresh_files_after_task`` must never raise into the task
        runner's cleanup, and the worker must survive a rescan of a
        tree whose ``.gitignore`` holds invalid UTF-8 (the input that
        used to raise ``UnicodeDecodeError`` out of the scan).

        The worker is synchronized on its own deterministic completion
        signal — the rescan publishing a NEW index object for the root
        — so the test neither sleeps on success nor passes vacuously
        when the worker never runs.
        """
        conn_id = "conn-ac1b"
        index_before = self._build_index(self.work_dir)

        Path(self.work_dir, ".gitignore").write_bytes(b"\xff\xfe\xff\n")
        with _RecordingExcepthook() as hook:
            self.server._refresh_files_after_task(self.work_dir)
            self.assertTrue(
                self._wait(
                    lambda: self._index_of(self.work_dir) is not index_before,
                ),
                "the post-task rescan never published a new index",
            )
        self.assertEqual(
            hook.records, [],
            f"index worker died with {hook.records!r}",
        )
        # The corrupt ignore file is tolerated: the tree is still indexed.
        view = self.registry.view_for(self.work_dir)
        self.assertIsNotNone(view)
        assert view is not None
        self.assertIn("hello.py", view.paths)

        # The rebuilt index still answers pickers (warm path).
        self.server._handle_command({
            "type": "getFiles",
            "prefix": "hel",
            "workDir": self.work_dir,
            "connId": conn_id,
            "tabId": "tab-1",
        })
        self._wait_token_released(conn_id)
        self.assertTrue(
            self._populated(conn_id), "picker wedged after the rescan",
        )

        # Success path: heal the scan input, add a new file, and the
        # refresh worker must publish an index that lists it.
        index_mid = self._index_of(self.work_dir)
        Path(self.work_dir, ".gitignore").unlink()
        Path(self.work_dir, "fresh_file.py").write_text("print('new')\n")
        self.server._refresh_files_after_task(self.work_dir)
        self.assertTrue(
            self._wait(
                lambda: self._index_of(self.work_dir) is not index_mid
                and "fresh_file.py" in (self.registry.view_for(self.work_dir) or view).paths,
            ),
            f"post-task refresh never published the rescan: "
            f"{(self.registry.view_for(self.work_dir) or view).paths!r}",
        )

    def test_cold_request_with_corrupt_gitignore_and_broken_database(
        self,
    ) -> None:
        """The cold path also completes when the scanned tree carries a
        corrupt ``.gitignore``: the scan tolerates it, the index is
        published, and a database failure in the reply still releases
        the token."""
        conn_id = "conn-ac1c"
        other = str(Path(self.tmpdir) / "work2")
        Path(other).mkdir(parents=True, exist_ok=True)
        Path(other, ".gitignore").write_bytes(b"\xff\xfe\xff\n")
        Path(other, "x_marker.py").write_text("x = 1\n")
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
            f"index worker died with {hook.records!r}",
        )
        populated = self._populated(conn_id)
        self.assertTrue(populated, "no populated files reply arrived")
        names = [f.get("text", str(f)) for f in populated[-1]["files"]]
        self.assertIn("./x_marker.py", names)

        # Same connection, a fresh cold root, broken database: the
        # reply fails on the worker but the token is still released.
        third = str(Path(self.tmpdir) / "work2b")
        Path(third).mkdir(parents=True, exist_ok=True)
        self._break_database()
        with _RecordingExcepthook() as hook:
            self.server._handle_command({
                "type": "getFiles",
                "prefix": "x",
                "workDir": third,
                "connId": conn_id,
                "tabId": "tab-1",
            })
            self._wait_token_released(conn_id)
        self.assertEqual(hook.records, [])
        self.assertIsNotNone(self.registry.view_for(third))

    def test_superseded_failing_refresh_leaves_the_newer_token_alone(
        self,
    ) -> None:
        """A failing reply whose request was superseded must NOT pop the
        newer request's token (identity-checked error cleanup).

        The persistence writer lock (a real lock in the production
        read path of ``_load_file_usage``) holds the FIRST request's
        reply at its database read on the worker until the same
        connection's second request has installed its own token and
        the database has been broken.  A gate job queued between the
        two requests parks the single worker after the first reply
        fails, so the map can be inspected while the stale failure has
        run and the newer request has not: the newer token must still
        be there.  Once the gate opens the newer reply fails too and
        pops its own entry.
        """
        conn_id = "conn-ac1d"
        other = str(Path(self.tmpdir) / "work3")
        Path(other).mkdir(parents=True, exist_ok=True)
        gate = threading.Event()
        failed = self._reply_failure_waiter()
        with _RecordingExcepthook() as hook:
            with _persistence._rw_lock.write_lock():
                # First request: the worker builds the index and its
                # reply blocks at the database read behind the held
                # writer lock.
                self.server._handle_command({
                    "type": "getFiles",
                    "prefix": "a",
                    "workDir": self.work_dir,
                    "connId": conn_id,
                    "tabId": "tab-1",
                })
                self.assertTrue(
                    self._wait(lambda: self._index_of(self.work_dir) is not None),
                    "the first request's index was never built",
                )
                # Park the worker between the two replies.
                self.registry.ensure(str(Path(self.tmpdir) / "gate"), gate.wait)
                # Second request (different work dir → cold path)
                # installs the connection's NEW token synchronously.
                self.server._handle_command({
                    "type": "getFiles",
                    "prefix": "a",
                    "workDir": other,
                    "connId": conn_id,
                    "tabId": "tab-1",
                })
                newer = self._token_map().get(conn_id)
                self.assertIsNotNone(newer)
                self._break_database()
            # Lock released: the stale reply fails its database read,
            # sees a foreign token and stands down; the worker then
            # parks on the gate.
            self.assertTrue(failed.seen.wait(15), "the stale reply never failed")
            self.assertIs(
                self._token_map().get(conn_id), newer,
                "the superseded request's failure cleanup popped the "
                "newer request's token",
            )
            gate.set()
            # The newer reply fails too and pops its own token.
            self._wait_token_released(conn_id)
        self.assertEqual(
            hook.records, [],
            f"index worker died with {hook.records!r}",
        )
        self.assertEqual(self._populated(conn_id), [])

    def test_warm_path_database_failure_releases_the_token(self) -> None:
        """Review finding 9: the synchronous warm path installs the
        request token, then reads the database (``_load_file_usage``)
        — a real open failure used to propagate BEFORE the token
        removal, stranding the token and wedging the connection's
        picker.  The identity-checked ``finally`` must release it even
        when the request fails."""
        conn_id = "conn-ac1e"
        self._build_index(self.work_dir)
        self.assertIsNotNone(self.registry.view_for(self.work_dir))

        self._break_database()
        with self.assertRaises(Exception):
            self.server._get_files("hel", self.work_dir, conn_id, "tab-1")
        self.assertNotIn(
            conn_id, self._token_map(),
            "a failing warm request stranded its token",
        )

        # With the database healthy again the SAME connection is
        # served normally (nothing wedged).
        self._heal_database()
        before = len(self._files_events(conn_id))
        self.server._get_files("hel", self.work_dir, conn_id, "tab-1")
        self._wait_token_released(conn_id)
        self.assertTrue(
            self._populated(conn_id, before), "picker wedged after the failure",
        )

    def test_thread_exhaustion_does_not_strand_the_token(self) -> None:
        """Review finding 9 (second path): when the registry's worker
        thread cannot be started (real ``RLIMIT_NPROC`` exhaustion on a
        registry that has never run a job), ``_get_files`` must neither
        propagate the spawn failure nor strand the request token, and
        the picker must recover once threads are available again."""
        if not thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        conn_id = "conn-ac1f"
        other = str(Path(self.tmpdir) / "work-nproc")
        Path(other).mkdir(parents=True, exist_ok=True)
        Path(other, "x_new.py").write_text("x = 1\n")
        self.assertIsNone(self.server._file_index._worker, "fresh registry: no worker yet")

        raised: list[BaseException] = []
        with nproc_limit_lowered_to_one():
            try:
                self.server._get_files("x", other, conn_id, "tab-1")
            except BaseException as exc:  # noqa: BLE001 — the bug propagated here
                raised.append(exc)
            try:
                self.server._refresh_files_after_task(self.work_dir)
            except BaseException as exc:  # noqa: BLE001 — the bug propagated here
                raised.append(exc)
            self.assertEqual(raised, [], "a thread spawn failure escaped to the caller")
            self.assertIsNone(self.server._file_index._worker, "the worker did not start")
            self._wait_token_released(conn_id)
        self.assertFalse(self._populated(conn_id), "nobody could have answered")

        # Threads are available again: the next request starts the worker
        # and is answered.
        self.server._get_files("x", other, conn_id, "tab-1")
        self._wait_token_released(conn_id)
        populated = self._populated(conn_id)
        self.assertTrue(populated, "picker wedged after thread exhaustion")

    def test_drop_connection_state_clears_the_request_token(self) -> None:
        """A connection departing with a build still in flight must not
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

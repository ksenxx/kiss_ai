# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for persistence.py concurrency findings.

Covers Fixer-1 findings:

- A1: ``_RWLock.write_lock`` leaked ``_pending_writers`` when an
  exception (e.g. an injected ``KeyboardInterrupt``) was delivered while
  blocked in ``Condition.wait()`` — starving every future reader.
- A2: invariant guard — repeated ``_stop_event_writer`` cycles under a
  concurrent producer must never run two writer threads at once and must
  preserve per-task FIFO event ordering with no event loss.
- A3: ``_maybe_reset_caches`` cleared ``_next_seq_cache`` under
  ``_caches_lock`` while ``_write_event_batch`` mutated it under
  ``_rw_lock`` only — a mid-batch clear silently dropped events whose
  batch had already passed the origin-path filter.
- A6: event-writer batch failures were logged at DEBUG, silently
  discarding events; they must surface at WARNING or higher.
- A9: ``_add_task`` stored ``str(bytes)`` reprs (``"b'...'"``) for
  model/work_dir/version instead of routing through ``_safe_str``.

All tests use real SQLite databases under a temp dir and real threads —
no mocks, patches, or fakes.
"""

from __future__ import annotations

import ctypes
import json
import logging
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _queue_chat_event,
)


class _DBSandbox:
    """Redirect the persistence DB to a fresh temp dir per test."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._close_db()

    def teardown_method(self) -> None:
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestA1WriteLockInterruptLeak:
    """A1: interrupted write_lock waiter must not starve readers forever."""

    def test_interrupted_writer_does_not_starve_readers(self) -> None:
        for attempt in range(4):
            lock = th._RWLock()
            release_reader = threading.Event()
            reader_in = threading.Event()

            def hold_read(
                lk: th._RWLock = lock,
                rin: threading.Event = reader_in,
                rel: threading.Event = release_reader,
            ) -> None:
                with lk.read_lock():
                    rin.set()
                    rel.wait(10)

            interrupted: list[bool] = []

            def try_write(
                lk: th._RWLock = lock, out: list[bool] = interrupted
            ) -> None:
                try:
                    with lk.write_lock():
                        pass
                except KeyboardInterrupt:
                    out.append(True)

            tr = threading.Thread(target=hold_read, daemon=True)
            tr.start()
            assert reader_in.wait(5)
            tw = threading.Thread(target=try_write, daemon=True)
            tw.start()
            # Wait until the writer is registered as pending (it is about
            # to block — or already blocked — in cond.wait()).
            deadline = time.monotonic() + 10
            while lock._pending_writers != 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert lock._pending_writers == 1
            time.sleep(0.1)  # let it enter cond.wait()
            assert tw.ident is not None
            # Pass the raw int thread id with an explicit C signature.
            # Thread ids are unsigned longs; wrapping in ctypes.c_long
            # breaks on Python 3.14 (and whenever another module has
            # already set argtypes=[c_ulong, py_object] on this cached
            # function object, as task_runner.py does at import time).
            set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
            set_async_exc.argtypes = [ctypes.c_ulong, ctypes.py_object]
            set_async_exc.restype = ctypes.c_int
            res = set_async_exc(tw.ident, ctypes.py_object(KeyboardInterrupt))
            assert res == 1
            release_reader.set()
            tw.join(15)
            tr.join(15)
            assert not tw.is_alive()

            acquired = threading.Event()

            def probe(
                lk: th._RWLock = lock, acq: threading.Event = acquired
            ) -> None:
                with lk.read_lock():
                    acq.set()

            tp = threading.Thread(target=probe, daemon=True)
            tp.start()
            assert acquired.wait(5), (
                f"attempt {attempt}: read_lock starved after interrupted "
                "write_lock — _pending_writers leaked"
            )


class TestA2SingleWriterInvariant(_DBSandbox):
    """A2 guard: stop/start churn keeps one writer, FIFO order, no loss."""

    def test_stop_start_cycles_keep_single_writer_and_fifo_order(self) -> None:
        tid, _ = _add_task("a2 task")
        produced = 300
        alive_samples: list[int] = []
        stop_sampler = threading.Event()

        def sampler() -> None:
            while not stop_sampler.is_set():
                n = sum(
                    1
                    for t in threading.enumerate()
                    if t.name == "kiss-event-writer" and t.is_alive()
                )
                alive_samples.append(n)
                time.sleep(0.001)

        ts = threading.Thread(target=sampler, daemon=True)
        ts.start()

        def producer() -> None:
            for i in range(produced):
                _queue_chat_event({"type": "e", "i": i}, tid)
                time.sleep(0.0005)

        tp = threading.Thread(target=producer, daemon=True)
        tp.start()
        for _ in range(15):
            th._stop_event_writer()
            time.sleep(0.005)
        tp.join(60)
        assert not tp.is_alive()
        _flush_chat_events()
        stop_sampler.set()
        ts.join(5)

        assert max(alive_samples or [0]) <= 1, (
            "two concurrent kiss-event-writer threads observed"
        )
        db = th._get_db()
        rows = db.execute(
            "SELECT seq, event_json FROM events WHERE task_id = ? "
            "ORDER BY seq",
            (tid,),
        ).fetchall()
        idxs = [json.loads(r["event_json"])["i"] for r in rows]
        assert idxs == list(range(produced)), (
            "event loss or FIFO order violation across writer restarts"
        )


class TestA3SeqCacheClearRace(_DBSandbox):
    """A3: DB-path flip mid-batch must not drop origin-matched events."""

    def test_db_path_flip_does_not_drop_committed_origin_events(self) -> None:
        n_tasks = 200
        tids = [_add_task(f"a3 task {i}")[0] for i in range(n_tasks)]
        _flush_chat_events()
        db1 = str(th._DB_PATH)
        # A second (never-created) DB path used purely to drive
        # ``_maybe_reset_caches`` — the real cache-clear trigger that a
        # concurrent ``_get_db()`` fires after a ``_DB_PATH`` swap.
        # ``_DB_PATH`` itself is never changed, so no event can ever be
        # legitimately dropped by the origin-path filter: every missing
        # event is a genuine bug drop.
        db2 = str(Path(self.tmpdir) / ".kiss2" / "sorcar.db")
        expected = 0

        for rnd in range(12):
            release = threading.Event()
            reader_in = threading.Event()

            def hold_read(
                rin: threading.Event = reader_in,
                rel: threading.Event = release,
            ) -> None:
                with th._rw_lock.read_lock():
                    rin.set()
                    rel.wait(30)

            tr = threading.Thread(target=hold_read, daemon=True)
            tr.start()
            assert reader_in.wait(5)
            # Enqueue one event per task while _DB_PATH is stable (db1) so
            # the whole batch passes the origin-path filter, then wedges on
            # the write lock held (shared) by the reader.
            for i, tid in enumerate(tids):
                _queue_chat_event({"type": "e", "r": rnd, "i": i}, tid, db1)
            expected += n_tasks
            time.sleep(0.08)  # batch collected; writer blocked at write_lock

            stop_flip = threading.Event()

            def flipper(stop: threading.Event = stop_flip) -> None:
                while not stop.is_set():
                    th._maybe_reset_caches(db2)
                    th._maybe_reset_caches(db1)

            tf = threading.Thread(target=flipper, daemon=True)
            tf.start()
            time.sleep(0.02)
            release.set()  # writer seeds/consumes seq cache while flips run
            time.sleep(0.15)
            stop_flip.set()
            tf.join(10)
            tr.join(10)
            th._maybe_reset_caches(db1)
            _flush_chat_events()

        raw = sqlite3.connect(db1)
        try:
            placeholders = ",".join("?" * len(tids))
            count = raw.execute(
                f"SELECT COUNT(*) FROM events WHERE task_id IN ({placeholders})",
                tids,
            ).fetchone()[0]
        finally:
            raw.close()
        assert count == expected, (
            f"{expected - count} origin-matched events silently dropped by "
            "seq-cache clear racing the event-writer batch"
        )


class TestA6BatchFailureLoggedAtWarning(_DBSandbox):
    """A6: batch write failures must be logged at WARNING, not DEBUG."""

    def test_event_batch_failure_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tid, _ = _add_task("a6 task")
        _queue_chat_event({"type": "x", "i": 0}, tid)
        _flush_chat_events()
        raw = sqlite3.connect(str(th._DB_PATH))
        try:
            raw.execute("ALTER TABLE events RENAME TO events_broken")
            raw.commit()
            with caplog.at_level(
                logging.DEBUG, logger="kiss.agents.sorcar.persistence"
            ):
                _queue_chat_event({"type": "x", "i": 1}, tid)
                _flush_chat_events()
            failures = [
                r
                for r in caplog.records
                if "event writer batch failed" in r.getMessage()
            ]
            assert failures, "expected a batch-failure log record"
            assert all(r.levelno >= logging.WARNING for r in failures), (
                "batch failure logged below WARNING — silent event loss"
            )
        finally:
            raw.execute("ALTER TABLE events_broken RENAME TO events")
            raw.commit()
            raw.close()


class TestA9AddTaskSafeStr(_DBSandbox):
    """A9: bytes payload fields must decode via _safe_str, not repr."""

    def test_bytes_payload_fields_are_decoded_not_reprd(self) -> None:
        model = b"gpt-4\xff"
        work_dir = b"/tmp/w\xff"
        version = b"1.2\xff"
        tid, _ = _add_task(
            "a9 task",
            extra={"model": model, "work_dir": work_dir, "version": version},
        )
        db = th._get_db()
        row = db.execute(
            "SELECT model, work_dir, version FROM task_history WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["model"] == model.decode("utf-8", errors="replace")
        assert row["work_dir"] == work_dir.decode("utf-8", errors="replace")
        assert row["version"] == version.decode("utf-8", errors="replace")
        assert not str(row["model"]).startswith("b'")

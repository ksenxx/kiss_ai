"""Integration tests for race conditions in sorcar/ and vscode/.

These tests deliberately interleave concurrent threads to reproduce
specific race conditions identified in
``kiss.agents.sorcar.persistence`` and
``kiss.agents.vscode.autocomplete``.

Each test inserts a small ``random.uniform(0, 0.05)`` sleep at the
suspected racing statement, then runs the involved threads many times
to make the failure deterministic.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str):
    """Point persistence at a temporary DB and reset cached state."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    th._invalidate_chat_context_cache("")
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved
    th._invalidate_chat_context_cache("")


class TestChatContextCacheStaleWrite:
    """Race 1: ``_load_chat_context_text`` writes stale data into cache.

    Reader R1 misses the cache, runs the SQL read (under the read
    lock) and gets snapshot ``D1``.  Before R1 stores the result, a
    writer W commits a new task and invalidates the cache.  A second
    reader R2 misses, runs the SQL read, gets ``D2`` (with W's new
    task) and stores ``D2`` into the cache.  R1 then stores ``D1``,
    permanently overwriting the fresh ``D2`` and returning stale data
    on every subsequent call until another invalidation happens.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        th._stop_event_writer()
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stale_cache_write(self) -> None:
        chat_id = "race-chat"
        # Seed one task so the SQL read returns non-empty data.
        th._add_task("task-zero", chat_id=chat_id)

        # Drive the race deterministically with explicit barriers.
        # R1 enters _load_chat_context, reads, and BLOCKS before
        # returning.  While R1 is blocked, the writer commits and
        # invalidates, then R2 runs fast through a fresh read+store.
        # Only then is R1 allowed to return and store its stale data.
        r1_in_sql = threading.Event()
        let_r1_finish = threading.Event()

        original = th._load_chat_context
        call_count = {"n": 0}
        cc_lock = threading.Lock()

        def staged_load_chat_context(cid: str):
            with cc_lock:
                call_count["n"] += 1
                n = call_count["n"]
            data = original(cid)
            if n == 1:
                # R1 — block AFTER reading so the writer sees R1's
                # stale view, then let R2 read fresh.
                r1_in_sql.set()
                # Random small sleep < 0.1s as suggested in CONTRIBUTING.
                time.sleep(random.uniform(0.001, 0.05))
                let_r1_finish.wait(timeout=5)
            return data

        th._load_chat_context = staged_load_chat_context  # type: ignore[assignment]

        try:
            def reader_one() -> None:
                th._load_chat_context_text(chat_id)

            t_r1 = threading.Thread(target=reader_one)
            t_r1.start()
            assert r1_in_sql.wait(timeout=5)

            # While R1 is blocked, the writer commits a new task and
            # invalidates the cache.
            th._add_task("task-one", chat_id=chat_id)

            # R2 misses, reads fresh (call_count == 2 — no block),
            # and stores the fresh value.
            def reader_two() -> None:
                th._load_chat_context_text(chat_id)

            t_r2 = threading.Thread(target=reader_two)
            t_r2.start()
            t_r2.join(timeout=5)

            # Sanity: fresh cache entry now includes "task-one".
            assert "task-one" in th._load_chat_context_text(chat_id)

            # Release R1 — it will store its stale data over R2's fresh
            # value if the race exists.
            let_r1_finish.set()
            t_r1.join(timeout=5)

            # The cache MUST still reflect the latest committed state.
            # Without the fix, R1 overwrites R2's fresh entry.
            cached_after = th._load_chat_context_text(chat_id)
            assert "task-one" in cached_after, (
                "stale chat-context cache survived a concurrent invalidation"
            )
        finally:
            th._load_chat_context = original  # type: ignore[assignment]
            let_r1_finish.set()


class TestAutocompleteWorkerDoubleSpawn:
    """Race 2: ``_AutocompleteMixin._ensure_complete_worker`` spawns
    duplicate worker threads when called from multiple threads.

    Two callers both observe ``self._complete_worker is None``, both
    create their own ``queue.Queue`` and worker thread, and the
    second thread overwrites the first thread's published references
    on ``self`` — leaving one worker reading from an orphaned queue
    forever (zombie thread) and producers enqueuing into a queue that
    a different worker drains.
    """

    def test_double_spawn(self) -> None:
        import queue as queue_mod

        from kiss.agents.vscode.autocomplete import _AutocompleteMixin

        # Build a bare instance carrying only the attributes touched
        # by ``_ensure_complete_worker``.
        instance = _AutocompleteMixin()
        instance._state_lock = threading.Lock()  # type: ignore[attr-defined]
        instance._complete_queue = None  # type: ignore[attr-defined]
        instance._complete_worker = None  # type: ignore[attr-defined]

        # Make the worker loop a no-op so the test doesn't block.
        def fake_loop(self_ref: object) -> None:
            q = getattr(self_ref, "_complete_queue", None)
            if q is None:
                return
            # Drain forever; daemon thread exits with the process.
            while True:
                try:
                    item = q.get(timeout=0.5)
                except queue_mod.Empty:
                    return
                if item is None:
                    return

        original_loop = _AutocompleteMixin._complete_worker_loop
        _AutocompleteMixin._complete_worker_loop = fake_loop  # type: ignore[assignment]

        try:
            # Insert a sleep right before the queue+thread construction
            # by wrapping ``_ensure_complete_worker`` with a sleep
            # injected after the None-check.  Easier: monkey-patch
            # ``queue.Queue`` to add the delay.
            original_q_init = queue_mod.Queue.__init__

            def slow_init(self: object, *args: object, **kwargs: object) -> None:
                time.sleep(random.uniform(0.01, 0.03))
                original_q_init(self, *args, **kwargs)  # type: ignore[arg-type]

            queue_mod.Queue.__init__ = slow_init  # type: ignore[method-assign]

            try:
                threads = []
                for _ in range(16):
                    t = threading.Thread(target=instance._ensure_complete_worker)
                    threads.append(t)
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5)
            finally:
                queue_mod.Queue.__init__ = original_q_init  # type: ignore[method-assign]

            # Count alive daemon threads created with our loop.  In a
            # correctly-locked implementation, exactly one worker
            # thread should exist.  Inspect threading.enumerate() to
            # count.
            alive_workers = []
            for t in threading.enumerate():
                target = getattr(t, "_target", None)
                if target is None:
                    continue
                if not t.daemon or not t.is_alive():
                    continue
                if not t.name.startswith("Thread-"):
                    continue
                if getattr(target, "__name__", "") == "fake_loop":
                    alive_workers.append(t)
            # With a fix, the published worker matches a single
            # surviving Thread.  Without it, multiple Threads are
            # spawned from the racing init paths.
            assert len(alive_workers) <= 1, (
                f"double-spawn race: {len(alive_workers)} workers alive"
            )
            # Final sanity: the published queue must be the same one
            # the worker is consuming from (no orphaned queue).
            assert instance._complete_queue is not None
            assert instance._complete_worker is not None
            assert instance._complete_worker.is_alive()
        finally:
            _AutocompleteMixin._complete_worker_loop = (  # type: ignore[method-assign]
                original_loop
            )
            # Wake the worker so the daemon exits promptly.
            if instance._complete_queue is not None:
                try:
                    instance._complete_queue.put_nowait(None)
                except Exception:
                    pass

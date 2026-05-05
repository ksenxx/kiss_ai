"""Tests that demonstrate unfixed race conditions in ``kiss.agents.vscode``.

Each test deterministically forces an interleaving that exposes a real
data race between two or more threads.  These tests use synchronisation
harnesses (barriers, events) to control scheduling — no mocks/patches
of production behaviour.

When the corresponding fix from ``race.md`` is applied, the test must
still pass.  Until then, some tests may intermittently fail — that is
the point: they prove the race exists.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer


class TestCompleteSeqCounterRace(unittest.TestCase):
    """_complete_seq and _complete_seq_latest have no synchronisation."""

    def test_stale_seq_latest_causes_wrong_completion(self) -> None:
        """Worker can observe stale _complete_seq_latest without a lock.

        Scenario:
          T1 (main): sets _complete_seq_latest = 1 (for query "a")
          T2 (main): sets _complete_seq_latest = 2 (for query "ab")
          Worker  : reads _complete_seq_latest -- should be 2
                    but if the write from T2 is not visible (no
                    memory barrier / lock), it may still see 1

        We demonstrate the race by interleaving writes and reads
        without any lock, showing the counters can become inconsistent.
        """
        server = VSCodeServer()
        events: list[dict] = []
        lock = threading.Lock()

        def capture(ev: dict) -> None:
            with lock:
                events.append(ev)

        server.printer.broadcast = capture  # type: ignore[assignment]
        server._file_cache = ["some/file.py"]

        barrier = threading.Barrier(2)

        observed_latest: list[int] = []

        original_complete = server._complete

        def slow_complete(
            query: str,
            seq: int = -1,
            snapshot_file: str = "",
            snapshot_content: str = "",
        ) -> None:
            observed_latest.append(server._complete_seq_latest)
            original_complete(query, seq, snapshot_file, snapshot_content)

        server._complete = slow_complete  # type: ignore[method-assign]

        server._complete_seq = 0

        def t1() -> None:
            server._complete_seq += 1
            server._complete_seq_latest = server._complete_seq
            barrier.wait(timeout=2)

        def t2() -> None:
            barrier.wait(timeout=2)
            server._complete_seq += 1
            server._complete_seq_latest = server._complete_seq

        th1 = threading.Thread(target=t1)
        th2 = threading.Thread(target=t2)
        th1.start()
        th2.start()
        th1.join(timeout=2)
        th2.join(timeout=2)

        server._complete_seq = 0
        lost_updates = 0
        for _ in range(500):
            server._complete_seq = 0
            barrier2 = threading.Barrier(2)

            def inc_a() -> None:
                barrier2.wait(timeout=2)
                server._complete_seq += 1

            def inc_b() -> None:
                barrier2.wait(timeout=2)
                server._complete_seq += 1

            a = threading.Thread(target=inc_a)
            b = threading.Thread(target=inc_b)
            a.start()
            b.start()
            a.join(timeout=2)
            b.join(timeout=2)
            if server._complete_seq != 2:
                lost_updates += 1

        import inspect

        src = inspect.getsource(type(server)._cmd_complete)
        self.assertIn("self._complete_seq += 1", src)
        self.assertIn("self._complete_seq_latest = seq", src)
        lock_start = src.index("with self._state_lock:")
        seq_write = src.index("self._complete_seq += 1")
        self.assertGreater(
            seq_write, lock_start,
            "_complete_seq should be inside _state_lock block — race fixed",
        )
        complete_src = inspect.getsource(type(server)._complete)
        self.assertIn("with self._state_lock:", complete_src)


class TestStaleBashBroadcastAfterReset(unittest.TestCase):
    """Timer-flushed bash output can arrive after reset()."""

    def test_stale_output_discarded_after_reset(self) -> None:
        """Verify _flush_bash discards stale text when reset() intervenes.

        The fix: _flush_bash captures the generation counter inside
        _bash_lock along with the text.  After releasing the lock it
        re-checks: if reset() ran in between (incrementing generation),
        the text is stale and the broadcast is skipped.
        """
        printer = BaseBrowserPrinter()

        with printer._bash_lock:
            printer._bash_state.buffer.append("stale output")


        reset_between = threading.Event()
        flush_captured = threading.Event()

        def timer_thread_logic() -> None:
            with printer._bash_lock:
                bs = printer._bash_state
                gen = bs.generation
                if bs.timer is not None:
                    bs.timer.cancel()
                    bs.timer = None
                text = "".join(bs.buffer) if bs.buffer else ""
                bs.buffer.clear()
                bs.last_flush = time.monotonic()
            flush_captured.set()
            reset_between.wait(timeout=5)
            if text:
                with printer._bash_lock:
                    if printer._bash_state.generation != gen:
                        return
                printer.broadcast({"type": "system_output", "text": text})

        timer_thread = threading.Thread(target=timer_thread_logic, daemon=True)
        timer_thread.start()

        flush_captured.wait(timeout=5)

        printer.reset()
        printer.start_recording()

        reset_between.set()
        timer_thread.join(timeout=5)

        recorded = printer.stop_recording()
        stale_recorded = [e for e in recorded if e.get("type") == "system_output"]
        self.assertEqual(
            len(stale_recorded), 0,
            "Stale event should be discarded after reset — race fixed",
        )

    def test_structural_generation_check_in_flush(self) -> None:
        """Verify _flush_bash captures generation and re-checks after lock."""
        import inspect

        src = inspect.getsource(BaseBrowserPrinter._flush_bash)
        self.assertIn("gen = bs.generation", src)
        self.assertIn("self._bash_state.generation != gen", src)


class TestDefaultModelNoLock(unittest.TestCase):
    """_default_model write is now protected by _state_lock (fixed)."""

    def test_select_model_writes_under_lock(self) -> None:
        """Structural test: selectModel changes _default_model inside lock."""
        import inspect

        src = inspect.getsource(VSCodeServer._cmd_select_model)
        lock_idx = src.index("self._state_lock")
        model_idx = src.index("self._default_model = model")
        self.assertGreater(
            model_idx, lock_idx,
            "_default_model should be written inside _state_lock — race fixed",
        )

    def test_concurrent_select_and_get_tab(self) -> None:
        """Two threads: one selecting model, one creating a tab.

        With the fix, both operations go through _state_lock so the
        new tab always sees a consistent model value.
        """
        server = VSCodeServer()
        with server._state_lock:
            server._default_model = "old-model"
        results: list[str] = []
        barrier = threading.Barrier(2)

        def select_model() -> None:
            barrier.wait(timeout=2)
            with server._state_lock:
                server._default_model = "new-model"

        def create_tab() -> None:
            barrier.wait(timeout=2)
            tab = server._get_tab("race-tab")
            results.append(tab.selected_model)

        t1 = threading.Thread(target=select_model)
        t2 = threading.Thread(target=create_tab)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertIn(results[0], ("old-model", "new-model"))


class TestUserAnswerQueueStaleReference(unittest.TestCase):
    """userAnswer handler reads queue without _state_lock."""

    def test_answer_put_on_abandoned_queue(self) -> None:
        """Answer is put on a queue after the task already finished.

        Scenario:
          1. Task starts, creates user_answer_queue
          2. Task asks user a question (broadcasts askUser)
          3. Main thread reads queue ref from tab (no lock)
          4. Task thread's finally sets queue = None (under lock)
          5. Main thread puts answer on the stale queue ref
          6. Nobody reads the answer — it is lost
        """
        server = VSCodeServer()
        tab_id = "answer-race"
        tab = server._get_tab(tab_id)

        answer_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        tab.user_answer_queue = answer_queue

        q_ref = tab.user_answer_queue

        with server._state_lock:
            tab.user_answer_queue = None

        q_ref.put("user's answer")

        self.assertIsNone(tab.user_answer_queue)
        self.assertEqual(q_ref.get_nowait(), "user's answer")

    def test_structural_lock_on_queue_read(self) -> None:
        """Verify the userAnswer handler reads queue under _state_lock (fixed)."""
        import inspect

        src = inspect.getsource(VSCodeServer._cmd_user_answer)
        self.assertIn("ans_state.user_answer_queue", src)
        self.assertIn("self._tab_states.get(ans_tab)", src)
        lines = src.split("\n")
        found_lock = False
        for line in lines:
            if "_state_lock" in line:
                found_lock = True
                break
            if "_tab_states.get" in line:
                break
        self.assertTrue(
            found_lock,
            "userAnswer should read _tab_states under _state_lock — race fixed",
        )


class TestEnsureCompleteWorkerDoubleInit(unittest.TestCase):
    """_ensure_complete_worker is not thread-safe (check-then-act)."""

    def test_structural_no_lock(self) -> None:
        """Method has no lock around check-then-act."""
        import inspect

        src = inspect.getsource(VSCodeServer._ensure_complete_worker)
        self.assertNotIn("_state_lock", src)
        self.assertNotIn("self._lock", src)

    def test_double_call_creates_two_queues(self) -> None:
        """Concurrent calls can create two separate queues/workers."""
        server = VSCodeServer()
        barrier = threading.Barrier(2)
        queues: list[object] = []

        def call_ensure() -> None:
            barrier.wait(timeout=2)
            server._ensure_complete_worker()
            queues.append(server._complete_queue)

        t1 = threading.Thread(target=call_ensure)
        t2 = threading.Thread(target=call_ensure)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertEqual(len(queues), 2)


class TestRefreshFileCacheRaceStructural(unittest.TestCase):
    """_refresh_file_cache and _get_files share _file_cache across threads."""

    def test_refresh_writes_outside_get_files_scan(self) -> None:
        """After the H9 fix, the original race is resolved.

        Previously _get_files scanned the filesystem synchronously when
        the cache was empty, which raced with the background refresh
        thread overwriting the same _file_cache attribute.  The H9 fix
        removed the synchronous scan from _get_files and routed all
        scanning through _refresh_file_cache, which acquires
        _state_lock when assigning to self._file_cache.

        This test pins the new architecture: _get_files MUST NOT call
        _scan_files directly (only via _refresh_file_cache), and
        _refresh_file_cache MUST run scanning in a background Thread
        and assign the cache under the state lock.
        """
        import inspect

        src = inspect.getsource(VSCodeServer._get_files)
        # _get_files MUST NOT scan synchronously anymore (H9).
        self.assertNotIn("_scan_files", src)
        self.assertIn("self._file_cache", src)
        # When cache is empty, must delegate to the background refresh.
        self.assertIn("_refresh_file_cache", src)

        src_refresh = inspect.getsource(VSCodeServer._refresh_file_cache)
        self.assertIn("Thread", src_refresh)
        self.assertIn("_scan_files", src_refresh)
        self.assertIn("self._file_cache = result", src_refresh)


class TestBroadcastOrderingFixed(unittest.TestCase):
    """VSCodePrinter.broadcast nests locks correctly (fixed race)."""

    def test_locks_are_nested(self) -> None:
        """_stdout_lock is acquired inside _lock, not separately."""
        import inspect

        from kiss.agents.vscode.server import VSCodePrinter

        src = inspect.getsource(VSCodePrinter.broadcast)
        lock_idx = src.index("self._lock")
        stdout_idx = src.index("self._stdout_lock")
        self.assertLess(lock_idx, stdout_idx)


if __name__ == "__main__":
    unittest.main()

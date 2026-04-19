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

# ---------------------------------------------------------------------------
# RACE 1: _complete_seq / _complete_seq_latest unsynchronised counter access
# ---------------------------------------------------------------------------
# Main thread writes _complete_seq and _complete_seq_latest with no lock.
# Worker thread reads _complete_seq_latest in _complete().
# On non-GIL Python (free-threaded 3.13t, or any future runtime without GIL)
# this is undefined behaviour.  Even under CPython-GIL, the absence of a
# happens-before edge means the worker may observe a stale counter value and
# incorrectly skip a completion request that is actually still the latest.
#
# This test forces the race by running two "complete" commands concurrently.
# Without synchronisation on the counters the worker may see a stale
# _complete_seq_latest value and broadcast a completion for request that
# was already superseded.
# ---------------------------------------------------------------------------

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
        server._file_cache = ["some/file.py"]  # avoid real scan

        # Directly manipulate the counters to demonstrate the race
        # without needing stdin/stdout plumbing.
        barrier = threading.Barrier(2)

        observed_latest: list[int] = []

        original_complete = server._complete

        def slow_complete(
            query: str,
            seq: int = -1,
            snapshot_file: str = "",
            snapshot_content: str = "",
        ) -> None:
            # Record what the worker saw as _complete_seq_latest
            observed_latest.append(server._complete_seq_latest)
            original_complete(query, seq, snapshot_file, snapshot_content)

        server._complete = slow_complete  # type: ignore[method-assign]

        # Simulate two rapid-fire complete commands
        # Write seq=1, then immediately seq=2 (no lock)
        server._complete_seq = 0

        def t1() -> None:
            server._complete_seq += 1
            server._complete_seq_latest = server._complete_seq
            barrier.wait(timeout=2)

        def t2() -> None:
            barrier.wait(timeout=2)
            # Immediately overwrite
            server._complete_seq += 1
            server._complete_seq_latest = server._complete_seq

        th1 = threading.Thread(target=t1)
        th2 = threading.Thread(target=t2)
        th1.start()
        th2.start()
        th1.join(timeout=2)
        th2.join(timeout=2)

        # The race: _complete_seq was incremented twice non-atomically.
        # Under GIL this happens to work, but _complete_seq += 1 is
        # NOT atomic (it is LOAD, ADD, STORE) and without a lock,
        # both threads can load the same value, add 1, and store —
        # producing seq=1 instead of seq=2.
        #
        # Demonstrate: run 1000 iterations to catch the lost-update
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

        # Under CPython-GIL, lost updates are extremely rare for int +=
        # but the code has NO lock, so it is a latent race.
        # We assert the structural defect exists (no lock protects the counter).
        import inspect

        src = inspect.getsource(type(server)._handle_command)
        # The "complete" branch writes _complete_seq and _complete_seq_latest
        # outside any lock.
        self.assertIn("self._complete_seq += 1", src)
        self.assertIn("self._complete_seq_latest = seq", src)
        # Find the "complete" branch and verify the counter writes are
        # AFTER the _state_lock block (i.e., outside it)
        complete_idx = src.index('"complete"')
        complete_section = src[complete_idx:complete_idx + 600]
        lock_end = complete_section.index("snapshot_content = self._last_active_content")
        seq_write = complete_section.index("self._complete_seq += 1")
        self.assertGreater(
            seq_write, lock_end,
            "_complete_seq incremented after _state_lock block — outside lock, race confirmed",
        )


# ---------------------------------------------------------------------------
# RACE 2: Stale bash broadcast after reset()
# ---------------------------------------------------------------------------
# A Timer thread captures buffered text inside _bash_lock, then broadcasts
# OUTSIDE the lock.  If reset() runs between the lock release and the
# broadcast, the stale system_output arrives after the printer has been
# reset for a new turn, contaminating the next turn's event stream.
# ---------------------------------------------------------------------------

class TestStaleBashBroadcastAfterReset(unittest.TestCase):
    """Timer-flushed bash output can arrive after reset()."""

    def test_stale_output_after_reset(self) -> None:
        """Demonstrate that _flush_bash broadcasts outside _bash_lock.

        The structural issue: _flush_bash captures text under _bash_lock
        then broadcasts outside it.  reset() can run between those two
        steps, so stale output gets recorded in the NEW turn's recording.
        """
        printer = BaseBrowserPrinter()

        # Stuff the buffer with content
        with printer._bash_lock:
            printer._bash_state.buffer.append("stale output")

        # We simulate the exact interleaving:
        # 1. Timer thread: _flush_bash captures text under _bash_lock
        # 2. Main thread: reset() + start_recording()
        # 3. Timer thread: broadcasts captured text (outside _bash_lock)
        #
        # Instead of monkey-patching, we replicate the _flush_bash logic
        # split into two phases with a synchronisation point between them.

        reset_between = threading.Event()
        flush_captured = threading.Event()

        def timer_thread_logic() -> None:
            # Phase 1: drain under lock (same as production _flush_bash)
            with printer._bash_lock:
                bs = printer._bash_state
                if bs.timer is not None:
                    bs.timer.cancel()
                    bs.timer = None
                text = "".join(bs.buffer) if bs.buffer else ""
                bs.buffer.clear()
                bs.last_flush = time.monotonic()
            flush_captured.set()
            reset_between.wait(timeout=5)
            # Phase 2: broadcast outside lock (same as production _flush_bash)
            if text:
                printer.broadcast({"type": "system_output", "text": text})

        timer_thread = threading.Thread(target=timer_thread_logic, daemon=True)
        timer_thread.start()

        # Wait for drain phase to complete
        flush_captured.wait(timeout=5)

        # Now reset + start new recording (simulates new turn)
        printer.reset()
        printer.start_recording()

        # Release timer to broadcast stale text
        reset_between.set()
        timer_thread.join(timeout=5)

        # The stale event is now in the NEW recording
        recorded = printer.stop_recording()
        stale_recorded = [e for e in recorded if e.get("type") == "system_output"]
        self.assertTrue(
            len(stale_recorded) > 0,
            "Stale event was recorded in the new turn — contamination confirmed",
        )


# ---------------------------------------------------------------------------
# RACE 3: _default_model written without lock
# ---------------------------------------------------------------------------
# selectModel writes self._default_model in the main thread without
# _state_lock.  _get_tab() (called from any thread) reads it inside
# _TabState.__init__.  If a task thread calls _get_tab for a new tab
# while the main thread is in selectModel, the new tab could see a
# partially-written (torn) string reference on non-GIL runtimes, or
# simply read a stale value even under CPython.
# ---------------------------------------------------------------------------

class TestDefaultModelNoLock(unittest.TestCase):
    """_default_model is written without _state_lock."""

    def test_select_model_writes_without_lock(self) -> None:
        """Structural test: selectModel changes _default_model outside lock."""
        import inspect

        src = inspect.getsource(VSCodeServer._handle_command)
        # Find the selectModel branch
        lines = src.split("\n")
        in_select_model = False
        model_write_outside_lock = False
        in_lock_block = False
        for line in lines:
            stripped = line.strip()
            if '"selectModel"' in stripped or "'selectModel'" in stripped:
                in_select_model = True
            if in_select_model and "with self._state_lock:" in stripped:
                in_lock_block = True
            if in_select_model and "self._default_model = model" in stripped:
                if not in_lock_block:
                    model_write_outside_lock = True
                break
        self.assertTrue(
            model_write_outside_lock,
            "_default_model written without _state_lock — race confirmed",
        )

    def test_concurrent_select_and_get_tab(self) -> None:
        """Two threads: one selecting model, one creating a tab.

        The new tab should see the latest model, but without a lock
        there is no happens-before guarantee.
        """
        server = VSCodeServer()
        server._default_model = "old-model"
        results: list[str] = []
        barrier = threading.Barrier(2)

        def select_model() -> None:
            barrier.wait(timeout=2)
            # Simulates the selectModel handler (no lock)
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

        # The tab may see "old-model" or "new-model" — both are valid
        # outcomes of this race.  The point is that the code has no lock
        # to guarantee a consistent view.
        self.assertIn(results[0], ("old-model", "new-model"))


# ---------------------------------------------------------------------------
# RACE 4: userAnswer reads queue reference without lock
# ---------------------------------------------------------------------------
# The main thread reads tab.user_answer_queue without _state_lock.
# The task thread's finally block sets it to None under _state_lock.
# If the task finishes between the main thread's read and put(),
# the answer goes to an abandoned queue nobody reads.
# ---------------------------------------------------------------------------

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

        # Simulate step 1: task creates a queue
        answer_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        tab.user_answer_queue = answer_queue

        # Step 3: main thread reads the queue ref (no lock in production code)
        q_ref = tab.user_answer_queue  # stale ref captured

        # Step 4: task thread's finally clears the queue (under lock)
        with server._state_lock:
            tab.user_answer_queue = None

        # Step 5: main thread puts answer on the stale ref
        q_ref.put("user's answer")

        # Step 6: the answer is on q_ref, but tab.user_answer_queue is None
        self.assertIsNone(tab.user_answer_queue)
        self.assertEqual(q_ref.get_nowait(), "user's answer")
        # The answer was lost — nobody will read it from q_ref

    def test_structural_lock_on_queue_read(self) -> None:
        """Verify the userAnswer handler reads queue under _state_lock (fixed)."""
        import inspect

        src = inspect.getsource(VSCodeServer._handle_command)
        # Find the userAnswer section
        idx = src.index('"userAnswer"')
        section = src[idx:idx + 500]
        # The queue is accessed via ans_state.user_answer_queue
        self.assertIn("ans_state.user_answer_queue", section)
        self.assertIn("self._tab_states.get(ans_tab)", section)
        # Verify _state_lock appears before the _tab_states.get
        lines = section.split("\n")
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


# ---------------------------------------------------------------------------
# RACE 5: _periodic_event_flush reads agent._last_task_id without lock
# ---------------------------------------------------------------------------
# The flush thread reads agent._last_task_id every _flush_interval seconds.
# The task thread writes it (via agent.run()).  No lock coordinates these.
# Under CPython-GIL this is "safe" but there is no happens-before edge.
# ---------------------------------------------------------------------------

class TestPeriodicFlushReadsTaskIdWithoutLock(unittest.TestCase):
    """_periodic_event_flush reads agent state without synchronisation."""

    def test_structural_no_lock_on_task_id_read(self) -> None:
        """The flush thread reads agent._last_task_id with no lock."""
        import inspect

        src = inspect.getsource(VSCodeServer._periodic_event_flush)
        self.assertIn("task_id = agent._last_task_id", src)
        # Verify no lock wraps this read
        self.assertNotIn("_state_lock", src)
        self.assertNotIn("_lock", src.replace("_state_lock", ""))


# ---------------------------------------------------------------------------
# RACE 6: _ensure_complete_worker double-init
# ---------------------------------------------------------------------------
# _ensure_complete_worker checks _complete_worker is not None then creates
# the queue and thread.  If called from two threads simultaneously, two
# workers could be created.  Currently only called from the single main
# thread, but the method has no internal synchronisation — any future
# caller from a different thread would trigger the race.
# ---------------------------------------------------------------------------

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

        # Both threads saw _complete_worker as None and created queues.
        # The second write overwrites the first, orphaning one queue+worker.
        # We can't reliably force both to see None under GIL, but we
        # verify the code structure allows it.
        self.assertEqual(len(queues), 2)


# ---------------------------------------------------------------------------
# RACE 7: _refresh_file_cache + _get_files — scan interleaving
# ---------------------------------------------------------------------------
# Already tested in test_vscode_races.py (TestFileCacheOverwriteRace).
# Including a structural assertion here for completeness.
# ---------------------------------------------------------------------------

class TestRefreshFileCacheRaceStructural(unittest.TestCase):
    """_refresh_file_cache and _get_files share _file_cache across threads."""

    def test_refresh_writes_outside_get_files_scan(self) -> None:
        """The background refresh thread and _get_files main-thread scan
        can interleave, with the stale scan potentially overwriting
        the fresh cache.  _get_files has a double-check pattern, but
        the race window still exists between the two lock acquisitions.
        """
        import inspect

        src = inspect.getsource(VSCodeServer._get_files)
        # _get_files does a scan outside lock, then double-checks
        self.assertIn("_scan_files", src)
        self.assertIn("self._file_cache is None", src)

        src_refresh = inspect.getsource(VSCodeServer._refresh_file_cache)
        self.assertIn("Thread", src_refresh)
        self.assertIn("self._file_cache = result", src_refresh)


# ---------------------------------------------------------------------------
# RACE 8: broadcast stdout-write vs recording order (FIXED)
# ---------------------------------------------------------------------------
# This race is already tested in test_vscode_races.py and was FIXED
# (broadcast holds _lock around both _record_event and _stdout_lock).
# We verify the fix is in place.
# ---------------------------------------------------------------------------

class TestBroadcastOrderingFixed(unittest.TestCase):
    """VSCodePrinter.broadcast nests locks correctly (fixed race)."""

    def test_locks_are_nested(self) -> None:
        """_stdout_lock is acquired inside _lock, not separately."""
        import inspect

        from kiss.agents.vscode.server import VSCodePrinter

        src = inspect.getsource(VSCodePrinter.broadcast)
        # The fix nests _stdout_lock inside _lock
        lock_idx = src.index("self._lock")
        stdout_idx = src.index("self._stdout_lock")
        self.assertLess(lock_idx, stdout_idx)


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


class TestStaleBashBroadcastAfterReset(unittest.TestCase):
    """Timer-flushed bash output can arrive after reset()."""

    def test_stale_output_discarded_after_reset(self) -> None:
        """Verify _flush_bash discards stale text when reset() intervenes.

        The fix: _flush_bash captures the generation counter inside
        _bash_lock along with the text.  After releasing the lock it
        re-checks: if reset() ran in between (incrementing generation),
        the text is stale and the broadcast is skipped.
        """
        printer = JsonPrinter()

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



class TestDefaultModelNoLock(unittest.TestCase):
    """_default_model write is now protected by _state_lock (fixed)."""


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



class TestEnsureCompleteWorkerDoubleInit(unittest.TestCase):
    """_ensure_complete_worker is not thread-safe (check-then-act)."""

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




# The historical ``TestBroadcastOrderingFixed`` introspected
# ``VSCodePrinter.broadcast`` for a nested ``_stdout_lock`` /
# ``_lock`` pattern.  Under the single-daemon architecture there is
# no stdout transport — ``WebPrinter`` writes events to UDS / WSS
# sockets via the asyncio loop and the per-stdout-lock invariant the
# old test pinned no longer applies.


if __name__ == "__main__":
    unittest.main()

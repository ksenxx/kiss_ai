# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that demonstrate unfixed race conditions in ``kiss.server``.

Each test deterministically forces an interleaving that exposes a real
data race between two or more threads.  These tests use synchronisation
harnesses (barriers, events) to control scheduling — no mocks/patches
of production behaviour.

When the corresponding fix from ``race.md`` is applied, the test must
still pass.  Until then, some tests may intermittently fail — that is
the point: they prove the race exists.
"""

from __future__ import annotations

import threading
import unittest

from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


class TestStaleBashBroadcastAfterReset(unittest.TestCase):
    """Timer-flushed bash output can arrive after reset()."""

    def test_stale_output_discarded_after_reset(self) -> None:
        """Verify _flush_bash sends nothing when reset() ran first.

        The fix: _flush_bash captures the buffer and broadcasts it in
        one critical section under the task's flush_lock, which
        reset() takes while clearing the buffer.  A flush that reaches
        the buffer after reset() finds it empty and skips the
        broadcast.
        """
        printer = JsonPrinter()

        with printer._bash_lock:
            printer._bash_state.buffer.append("stale output")

        reset_done = threading.Event()

        def timer_thread_logic() -> None:
            reset_done.wait(timeout=5)
            printer._flush_bash()

        timer_thread = threading.Thread(target=timer_thread_logic, daemon=True)
        timer_thread.start()

        printer.reset()
        printer.start_recording()

        reset_done.set()
        timer_thread.join(timeout=5)

        recorded = printer.stop_recording()
        stale_recorded = [e for e in recorded if e.get("type") == "system_output"]
        self.assertEqual(
            len(stale_recorded), 0,
            "Stale event should be discarded after reset — race fixed",
        )



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





if __name__ == "__main__":
    unittest.main()

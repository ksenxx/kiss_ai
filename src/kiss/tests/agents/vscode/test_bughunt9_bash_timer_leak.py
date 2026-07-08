# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9: straggler bash-flush resurrects per-task state after cleanup.

``JsonPrinter.cleanup_task`` pops the task's ``_BashState`` and cancels
any pending flush timer — but ``threading.Timer.cancel()`` cannot stop a
callback that has already fired.  A straggler
``_timer_flush_for_task(task_id)`` (the exact function the timer thread
runs) then called ``_flush_bash``, whose ``_bash_state`` property
RE-CREATED the just-popped entry in ``_bash_states``.  Every task that
ended while a flush timer was in flight therefore leaked a permanent
``_BashState`` keyed by a dead task id, and could broadcast stale
``system_output`` attributed to the finished task.

No mocks, patches, fakes, or test doubles: a real :class:`JsonPrinter`
(via the shared :class:`MemoryPrinter` transport used by all vscode
tests) and the real production timer callback.
"""

from __future__ import annotations

import threading
import time
import unittest

from kiss.tests.agents.vscode._memory_printer import MemoryPrinter


class TestBashTimerStateResurrection(unittest.TestCase):
    """A straggler flush after ``cleanup_task`` must not resurrect state."""

    def test_straggler_timer_flush_does_not_recreate_state(self) -> None:
        p = MemoryPrinter()
        p._thread_local.task_id = "task-9"
        p.subscribe_tab("task-9", "tab-9")
        # First bash_stream print flushes immediately (last_flush == 0).
        p.print("first chunk\n", type="bash_stream")
        # Second print inside the 0.1s window arms the flush timer.
        p.print("second chunk\n", type="bash_stream")
        self.assertIn("task-9", p._bash_states)
        # Task ends: the runner frees all per-task state.
        p.cleanup_task("task-9")
        self.assertNotIn("task-9", p._bash_states)
        emitted_before = len(p.emitted)
        # A timer that already fired before cancel() runs exactly this
        # callback on its worker thread.
        worker = threading.Thread(
            target=p._timer_flush_for_task, args=("task-9",),
        )
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        # The popped per-task state must NOT be resurrected (leak), and
        # no stale system_output may be broadcast for the dead task.
        self.assertNotIn("task-9", p._bash_states)
        self.assertEqual(len(p.emitted), emitted_before)

    def test_timer_race_with_cleanup_leaves_no_state(self) -> None:
        """Timing variant: cleanup racing the real 0.1s timer thread."""
        p = MemoryPrinter()
        p._thread_local.task_id = "task-10"
        p.print("a\n", type="bash_stream")  # immediate flush
        p.print("b\n", type="bash_stream")  # arms the real timer
        # Sleep close to the timer deadline so cancel() races the fire.
        time.sleep(0.09)
        p.cleanup_task("task-10")
        time.sleep(0.3)  # let any straggler callback finish
        self.assertNotIn("task-10", p._bash_states)

    def test_active_task_bash_streaming_still_works(self) -> None:
        """Regression guard: normal buffering + timer flush still emits."""
        p = MemoryPrinter()
        p._thread_local.task_id = "task-11"
        p.subscribe_tab("task-11", "tab-11")
        p.print("hello ", type="bash_stream")  # immediate flush
        p.print("world\n", type="bash_stream")  # buffered, timer armed
        time.sleep(0.3)  # timer fires and flushes the buffer
        texts = [
            e.get("text", "")
            for e in p.emitted
            if e.get("type") == "system_output"
        ]
        self.assertIn("hello ", "".join(texts))
        self.assertIn("world", "".join(texts))
        p.cleanup_task("task-11")
        self.assertNotIn("task-11", p._bash_states)


if __name__ == "__main__":
    unittest.main()

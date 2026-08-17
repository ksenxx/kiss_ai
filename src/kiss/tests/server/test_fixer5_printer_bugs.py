# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixer-5 JsonPrinter lifecycle bugs (findings F5-05, F5-06).

F5-05 — ``cleanup_task`` removed a task's ``_BashState`` without
invalidating its generation, so a timer flush that had already copied
buffered text (but not yet broadcast) still emitted stale
``system_output`` AFTER the task was cleaned up.  A second
interleaving let ``print(type="bash_stream")``'s post-broadcast tail
touch the *creating* ``_bash_state`` property and permanently
resurrect the freed state under a dead task id.

F5-06 — completed-task cleanup never pruned ``_subscribers[task_id]``,
so a long-lived tab running many sequential tasks accumulated one
stale subscriber entry per completed task forever.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from kiss.server.json_printer import JsonPrinter

from ._memory_printer import MemoryPrinter


class _GatedPrinter(JsonPrinter):
    """Printer whose ``system_output`` transport blocks on a gate.

    Models a slow transport (e.g. ``WebPrinter``'s socket sends) so a
    ``cleanup_task`` can run while a broadcast is still in flight.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.emitted: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        super().broadcast(event)
        self.emitted.append(event)
        if event.get("type") == "system_output":
            self.gate.wait(timeout=10)


class TestStaleFlushAfterCleanup(unittest.TestCase):
    """F5-05a: a copied-but-unbroadcast flush dies with the task."""

    def test_timer_flush_after_cleanup_is_dropped(self) -> None:
        p = MemoryPrinter()
        # Keep the subscriber alive across cleanup (long linger) so a
        # stale post-cleanup broadcast WOULD be visible in ``emitted``.
        p.subscribe_tab("task-55", "tab-X")
        p._thread_local.task_id = "task-55"
        # First fragment flushes immediately (last_flush == 0.0) and
        # stamps last_flush; the second buffers and arms the 0.1s timer.
        p.print("early output", type="bash_stream")
        p.print("late output", type="bash_stream")
        with p._bash_lock:
            bs = p._bash_states["task-55"]
        # Park the timer between its copy step and its broadcast step:
        # it copies the text under _bash_lock, then blocks acquiring
        # flush_lock (held here) before the generation re-check.
        bs.flush_lock.acquire()
        cleaner = threading.Thread(
            # cleanup_task itself waits on flush_lock (to let an
            # authorized in-flight broadcast finish), so it must run on
            # its own thread while this test thread holds the lock.  It
            # bumps the generation BEFORE that wait, so the parked
            # timer's re-check fails regardless of who gets the lock
            # first once it is released.
            target=p.cleanup_task,
            args=("task-55",),
            kwargs={"subscriber_linger_seconds": 30},
            daemon=True,
        )
        try:
            time.sleep(0.35)
            self.assertEqual(
                bs.buffer, [],
                "timer should have copied and cleared the buffer by now",
            )
            cleaner.start()
            time.sleep(0.1)
        finally:
            bs.flush_lock.release()
        cleaner.join(timeout=10)
        self.assertFalse(cleaner.is_alive())
        time.sleep(0.3)
        late = [
            e for e in p.emitted
            if e.get("type") == "system_output"
            and "late output" in e.get("text", "")
        ]
        self.assertEqual(
            late, [],
            "stale bash output was broadcast after cleanup_task",
        )

    def test_flush_before_cleanup_still_emits(self) -> None:
        """Regression guard: normal timer flushes are unaffected."""
        p = MemoryPrinter()
        p.subscribe_tab("task-56", "tab-Y")
        p._thread_local.task_id = "task-56"
        p.print("early output", type="bash_stream")
        p.print("late output", type="bash_stream")
        time.sleep(0.35)
        texts = [
            e.get("text", "") for e in p.emitted
            if e.get("type") == "system_output"
        ]
        self.assertTrue(
            any("late output" in t for t in texts),
            f"buffered bash output never flushed: {p.emitted!r}",
        )


class TestNoBashStateResurrection(unittest.TestCase):
    """F5-05b: cleanup during a slow broadcast must stay permanent."""

    def test_print_tail_does_not_recreate_freed_state(self) -> None:
        p = _GatedPrinter()
        done = threading.Event()

        def _bash_print() -> None:
            p._thread_local.task_id = "task-77"
            p.print("streamed text", type="bash_stream")
            done.set()

        worker = threading.Thread(target=_bash_print, daemon=True)
        worker.start()
        # Wait until the broadcast is in flight (blocked on the gate).
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            e.get("type") == "system_output" for e in p.emitted
        ):
            time.sleep(0.01)
        self.assertTrue(
            any(e.get("type") == "system_output" for e in p.emitted),
            "bash broadcast never started",
        )
        # cleanup_task waits for the in-flight broadcast (R5-01), so it
        # runs on its own thread; the gate is released while it waits.
        cleaner = threading.Thread(
            target=p.cleanup_task,
            args=("task-77",),
            kwargs={"subscriber_linger_seconds": 0},
            daemon=True,
        )
        cleaner.start()
        time.sleep(0.1)
        p.gate.set()
        cleaner.join(timeout=10)
        self.assertFalse(cleaner.is_alive())
        self.assertTrue(done.wait(timeout=10))
        with p._bash_lock:
            self.assertNotIn(
                "task-77", p._bash_states,
                "print()'s post-broadcast tail resurrected the freed "
                "bash state under a dead task id",
            )

    def test_cleanup_waits_for_inflight_broadcast(self) -> None:
        """R5-01: cleanup_task returns only after an authorized
        in-flight bash broadcast (already past its generation check,
        holding ``flush_lock``) has finished."""
        p = _GatedPrinter()
        done = threading.Event()

        def _bash_print() -> None:
            p._thread_local.task_id = "task-79"
            p.print("streamed text", type="bash_stream")
            done.set()

        worker = threading.Thread(target=_bash_print, daemon=True)
        worker.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            e.get("type") == "system_output" for e in p.emitted
        ):
            time.sleep(0.01)
        cleaner = threading.Thread(
            target=p.cleanup_task,
            args=("task-79",),
            kwargs={"subscriber_linger_seconds": 0},
            daemon=True,
        )
        cleaner.start()
        time.sleep(0.2)
        self.assertTrue(
            cleaner.is_alive(),
            "cleanup_task must wait for the in-flight broadcast",
        )
        p.gate.set()
        cleaner.join(timeout=10)
        self.assertFalse(cleaner.is_alive())
        self.assertTrue(done.wait(timeout=10))

    def test_tool_result_after_cleanup_does_not_recreate_state(self) -> None:
        p = MemoryPrinter()
        p._thread_local.task_id = "task-78"
        p.print("some output", type="bash_stream")
        p.cleanup_task("task-78", subscriber_linger_seconds=0)
        p.print("result", type="tool_result", tool_name="Bash")
        p.print("Bash", type="tool_call", tool_input={"command": "ls"})
        with p._bash_lock:
            self.assertNotIn("task-78", p._bash_states)


class TestSubscriberPruning(unittest.TestCase):
    """F5-06: subscriber sets are pruned after the post-task linger."""

    def test_subscribers_survive_linger_then_prune(self) -> None:
        p = MemoryPrinter()
        p.subscribe_tab("task-91", "tab-A")
        p.cleanup_task("task-91", subscriber_linger_seconds=0.1)
        # Within the linger window post-task broadcasts still fan out.
        self.assertEqual(p._fanout_targets("task-91"), ["tab-A"])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and p._fanout_targets("task-91"):
            time.sleep(0.02)
        self.assertEqual(
            p._fanout_targets("task-91"), [],
            "subscriber entry was never pruned after the linger window",
        )

    def test_many_sequential_tasks_do_not_accumulate(self) -> None:
        p = MemoryPrinter()
        threads_before = threading.active_count()
        for i in range(200):
            p.subscribe_tab(f"task-{i}", "tab-B")
            p.cleanup_task(f"task-{i}", subscriber_linger_seconds=0.05)
        # Pruning is opportunistic (no timer thread per task): any
        # subscriber-map operation after the linger sweeps them out.
        self.assertLess(
            threading.active_count(), threads_before + 5,
            "cleanup_task must not spawn one thread per completed task",
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            p._fanout_targets("task-0")  # sweeps expired entries
            with p._lock:
                if not p._subscribers:
                    break
            time.sleep(0.05)
        with p._lock:
            self.assertEqual(
                dict(p._subscribers), {},
                "completed-task subscriber entries leaked",
            )
            self.assertEqual(dict(p._subscriber_expiry), {})

    def test_synchronous_prune_with_zero_linger(self) -> None:
        p = MemoryPrinter()
        p.subscribe_tab("task-92", "tab-C")
        p.cleanup_task("task-92", subscriber_linger_seconds=0)
        self.assertEqual(p._fanout_targets("task-92"), [])


if __name__ == "__main__":
    unittest.main()

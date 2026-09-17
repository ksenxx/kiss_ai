# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: straggler ``bash_stream`` prints cannot resurrect freed state.

Concurrency-audit fix C1 (tmp/audit-server-scout-printer.md): the
``bash_stream`` branch of :meth:`JsonPrinter.print` was the ONE
straggler path that used the CREATING ``_bash_state`` property with no
``_closed_tasks`` guard.  A bash fragment arriving after
``cleanup_task`` re-created a fresh ``_BashState`` under the dead task
id — a permanent leak (no future cleanup pops that key again) — and,
because the fresh state's ``last_flush`` is 0.0, immediately broadcast
a ``system_output`` attributed to the finished task, which the
subscriber set (kept alive for 300 s after cleanup) delivered to a
watching tab.

The fix checks ``_closed_tasks`` and creates the state atomically
under ``_lock`` (nested around ``_bash_lock``), and ``cleanup_task``
now marks ``_closed_tasks`` BEFORE popping the bash state so no
interleaving escapes the guard.

Real printers only (``MemoryPrinter`` is the repo's in-memory
transport subclass); no mocks.
"""

from __future__ import annotations

import threading
import time
import unittest

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.tests.server._memory_printer import MemoryPrinter


def _system_outputs(printer: MemoryPrinter) -> list[dict[str, object]]:
    """Return every ``system_output`` event the printer emitted."""
    return [e for e in printer.emitted if e.get("type") == "system_output"]


class StragglerBashStreamTest(unittest.TestCase):
    """A ``bash_stream`` print after ``cleanup_task`` is dropped."""

    def test_straggler_does_not_resurrect_state_or_broadcast(self) -> None:
        """Post-cleanup fragments create no state and reach no tab.

        The subscriber set deliberately lingers after ``cleanup_task``
        (default 300 s), so a resurrected flush WOULD reach the tab —
        this asserts it does not.
        """
        printer = MemoryPrinter()
        printer._thread_local.task_id = "conc1-task"
        printer.subscribe_tab("conc1-task", "tab-1")

        printer.print("live output\n", type="bash_stream")
        live = _system_outputs(printer)
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].get("text"), "live output\n")
        self.assertEqual(live[0].get("tabId"), "tab-1")

        printer.cleanup_task("conc1-task")
        self.assertNotIn("conc1-task", printer._bash_states)
        emitted_before = len(printer.emitted)

        # The straggler: same thread, same (now dead) task id.
        printer.print("post-task straggler\n", type="bash_stream")
        # Give a wrongly-scheduled 0.1 s flush timer time to fire.
        time.sleep(0.3)

        self.assertNotIn(
            "conc1-task",
            printer._bash_states,
            "straggler bash_stream resurrected the freed _BashState",
        )
        self.assertEqual(
            printer.emitted[emitted_before:],
            [],
            "straggler bash output was broadcast after cleanup_task",
        )

    def test_taskless_stream_still_flushes(self) -> None:
        """The guard never blocks the task-less ``""`` key.

        ``cleanup_task`` early-returns for an empty key, so ``""`` is
        never in ``_closed_tasks`` and thread-without-task streaming
        (unit-test / pre-task lifecycle paths) keeps working.
        """
        printer = MemoryPrinter()
        printer.start_recording()  # no task id: no-op, key "" unrecorded
        printer.print("no task bound\n", type="bash_stream")
        with printer._bash_lock:
            state = printer._bash_states.get("")
            self.assertIsNotNone(state)
            assert state is not None
            # The first fragment flushed inline (last_flush starts at
            # 0.0, so the 0.1 s throttle passes immediately): the
            # buffer was consumed and the flush time stamped.
            self.assertEqual(state.buffer, [])
            self.assertGreater(state.last_flush, 0.0)

    def test_hammer_stream_vs_cleanup_never_resurrects(self) -> None:
        """Concurrent streaming and cleanup never leak a dead key.

        A streamer thread prints fragments (exercising both the
        inline-flush and the 0.1 s timer paths) while the main thread
        runs ``cleanup_task``; whatever the interleaving, the dead key
        must be absent once the streamer stops.
        """
        printer = MemoryPrinter()

        def stream(key: str, stop: threading.Event) -> None:
            printer._thread_local.task_id = key
            while not stop.is_set():
                printer.print("x", type="bash_stream")

        for i in range(150):
            key = f"conc1-hammer-{i}"
            stop = threading.Event()
            worker = threading.Thread(target=stream, args=(key, stop))
            worker.start()
            time.sleep(0.0005)
            printer.cleanup_task(key, subscriber_linger_seconds=0)
            stop.set()
            worker.join()
            self.assertNotIn(
                key,
                printer._bash_states,
                f"iteration {i}: cleanup_task raced bash_stream and "
                "the freed state was resurrected",
            )


class TombstoneEvictionTest(unittest.TestCase):
    """Review finding 5: the no-resurrection guarantee cannot expire.

    ``_closed_tasks`` is bounded (256 entries), so enough later task
    cleanups evict an old task's tombstone — a sufficiently delayed
    ``bash_stream`` fragment then passed the membership-only guard,
    recreated the freed ``_BashState`` under the dead id, and
    broadcast stale ``system_output`` to the lingering subscriber
    set.  After the first eviction, creating a NEW bash state now also
    requires the task to be POSITIVELY live in the agent-state
    registry — a check that never forgets — while the tombstone set
    stays bounded (no memory leak traded back in).
    """

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def _evict(self, printer: MemoryPrinter, count: int = 256) -> None:
        """Close *count* fresh tasks, evicting the oldest tombstones."""
        for i in range(count):
            printer.cleanup_task(
                f"conc5-filler-{i}", subscriber_linger_seconds=0,
            )

    def test_straggler_after_eviction_is_still_dropped(self) -> None:
        printer = MemoryPrinter()
        printer._thread_local.task_id = "conc5-old"
        printer.subscribe_tab("conc5-old", "tab-old")
        printer.print("live\n", type="bash_stream")
        printer.cleanup_task("conc5-old")
        self._evict(printer)
        self.assertNotIn(
            "conc5-old", printer._closed_tasks,
            "eviction never happened; the scenario is not exercised",
        )
        emitted_before = len(printer.emitted)
        printer._thread_local.task_id = "conc5-old"
        printer.print("very late straggler\n", type="bash_stream")
        time.sleep(0.3)
        self.assertNotIn(
            "conc5-old", printer._bash_states,
            "an evicted tombstone let a straggler resurrect bash state",
        )
        self.assertEqual(
            printer.emitted[emitted_before:], [],
            "stale bash output was broadcast after tombstone eviction",
        )

    def test_finished_but_registered_task_is_still_dropped(self) -> None:
        """A retired state kept registered (e.g. a pending worktree)
        is idle with a dead thread — not a licence to stream."""
        printer = MemoryPrinter()
        printer.cleanup_task("conc5-done")
        self._evict(printer)
        state = AgentState("conc5-done", tab_id="tab-done")
        state.is_task_active = False
        with agent_state.STATE_LOCK:
            agent_state.register(state)
        printer._thread_local.task_id = "conc5-done"
        printer.print("late\n", type="bash_stream")
        self.assertNotIn("conc5-done", printer._bash_states)

    def test_completed_task_with_live_cleanup_thread_is_dropped(self) -> None:
        """The production finalization tail is not a live producer.

        gpt-5.6-sol round-2 review, finding 2: normal task
        finalization clears ``is_task_active`` and then calls
        ``cleanup_task`` on the SAME still-running runner thread, so
        right after cleanup the task's state is registered, INACTIVE,
        with a LIVE ``task_thread``.  The old gate accepted
        ``is_task_active or thread_alive()``, so once 256 newer
        cleanups evicted the tombstone, a delayed ``bash_stream``
        under that key recreated the freed state and broadcast stale
        ``system_output`` to the lingering subscriber.  This drives
        that exact schedule — the "runner thread" is this test's own
        (alive) thread — and asserts the straggler is dropped.
        """
        printer = MemoryPrinter()
        key = "conc5-finalizing"
        state = AgentState(
            key,
            tab_id="tab-finalizing",
            server_owned=True,
            task_thread=threading.current_thread(),
            is_task_active=False,
        )
        with agent_state.STATE_LOCK:
            agent_state.register(state)
        printer.subscribe_tab(key, "tab-finalizing")
        printer._thread_local.task_id = key
        printer.cleanup_task(key)
        # The exact post-finalization state: cleaned up, inactive,
        # runner thread still alive, state still registered.
        self.assertFalse(state.is_task_active)
        self.assertTrue(state.thread_alive())
        self._evict(printer)
        self.assertNotIn(
            key, printer._closed_tasks,
            "eviction never happened; the scenario is not exercised",
        )
        emitted_before = len(printer.emitted)
        printer._thread_local.task_id = key
        printer.print("stale after cleanup\n", type="bash_stream")
        time.sleep(0.3)
        self.assertNotIn(
            key, printer._bash_states,
            "a completed task's live cleanup thread was mistaken for a "
            "live producer and resurrected the freed bash state",
        )
        self.assertEqual(
            printer.emitted[emitted_before:], [],
            "a completed task's stale bash output was broadcast after "
            "tombstone eviction",
        )

    def test_live_task_still_streams_after_eviction(self) -> None:
        """The positive liveness check never drops a live task's
        output: a registered active task streams normally even after
        the tombstone map has cycled."""
        printer = MemoryPrinter()
        self._evict(printer, 300)
        state = AgentState("conc5-live", tab_id="tab-live", is_task_active=True)
        with agent_state.STATE_LOCK:
            agent_state.register(state)
        printer.subscribe_tab("conc5-live", "tab-live")
        printer._thread_local.task_id = "conc5-live"
        printer.print("fresh live output\n", type="bash_stream")
        self.assertIn("conc5-live", printer._bash_states)
        outputs = _system_outputs(printer)
        self.assertTrue(
            any(e.get("text") == "fresh live output\n" for e in outputs),
            "a live task's bash output was wrongly dropped",
        )

    def test_taskless_stream_survives_eviction(self) -> None:
        """The ``""`` key needs no registry entry, before or after."""
        printer = MemoryPrinter()
        self._evict(printer, 300)
        printer.print("no task bound\n", type="bash_stream")
        with printer._bash_lock:
            self.assertIsNotNone(printer._bash_states.get(""))


if __name__ == "__main__":
    unittest.main()

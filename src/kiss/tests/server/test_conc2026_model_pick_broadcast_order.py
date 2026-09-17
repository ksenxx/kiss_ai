# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: modelPick wire order always matches override membership.

Concurrency-audit fix C2 (tmp/audit-server-scout-printer.md):
``subscribe_tab``'s catch-up and ``broadcast_agent_model_pick``
updated ``_model_override_tabs`` under ``_lock`` but broadcast the
``modelPick`` AFTER releasing it, so a racing ``restore_model_pick``
could interleave between the state write and the broadcast: the
restore discarded the tab's membership and broadcast the user's model
first, then the resumed agent broadcast repainted the DEAD agent's
model — with the tab no longer in ``_model_override_tabs``, no later
restore could repair the label (restore early-returns for
non-members).

The fix serialises {override-state update + broadcast} in the three
methods with ``_model_pick_lock``, making the last ``modelPick`` on
the wire always agree with the final membership state.

Real printers only: ``SlowWirePrinter`` is a ``MemoryPrinter`` whose
transport (the ``broadcast`` hook every production printer overrides)
models a slow socket send with a tiny sleep — that is the very window
the production ``WebPrinter`` has; nothing under test is stubbed.
"""

from __future__ import annotations

import random
import threading
import time
import unittest
from typing import Any

from kiss.tests.server._memory_printer import MemoryPrinter


class SlowWirePrinter(MemoryPrinter):
    """MemoryPrinter with a slow transport, like a real socket send."""

    def broadcast(self, event: dict[str, Any]) -> None:
        """Append *event* after a short, jittery transport delay.

        The jitter (0-5 ms, like a real socket send under load) is
        what lets the pre-fix interleaving — restore's wire copy
        landing between the agent pick's state write and its own wire
        copy — actually occur within a bounded number of iterations.

        Args:
            event: The event dictionary to emit.
        """
        time.sleep(random.uniform(0.0, 0.005))
        super().broadcast(event)


def _model_picks(printer: MemoryPrinter, tab_id: str) -> list[dict[str, Any]]:
    """Return the ``modelPick`` events emitted for *tab_id*, in order."""
    return [
        e
        for e in printer.emitted
        if e.get("type") == "modelPick" and e.get("tabId") == tab_id
    ]


class ModelPickBroadcastOrderTest(unittest.TestCase):
    """Wire order of agent/restore picks matches the override state."""

    def _assert_consistent(
        self, printer: MemoryPrinter, tab: str, iteration: int,
    ) -> None:
        """The last modelPick for *tab* agrees with its membership."""
        picks = _model_picks(printer, tab)
        self.assertTrue(picks, f"iteration {iteration}: no modelPick at all")
        last = picks[-1]
        member = tab in printer._model_override_tabs
        if member:
            self.assertEqual(
                last.get("source"),
                "agent",
                f"iteration {iteration}: tab still overridden but the "
                f"wire ends with {last!r}",
            )
        else:
            self.assertEqual(
                last.get("source"),
                "restore",
                f"iteration {iteration}: tab restored but the wire ends "
                f"with the dead agent's pick {last!r}",
            )

    def test_serial_catch_up_then_restore(self) -> None:
        """Baseline: catch-up then restore ends on the user's model."""
        printer = SlowWirePrinter()
        # Seed the task's override the way a running agent does (no
        # watcher yet, so nothing is broadcast).
        printer.broadcast_agent_model_pick("agent-model", "", "conc2-task")
        printer.subscribe_tab("conc2-task", "tab-serial")
        printer.restore_model_pick("user-model", "tab-serial")
        picks = _model_picks(printer, "tab-serial")
        self.assertEqual(
            [(p.get("source"), p.get("model")) for p in picks],
            [("agent", "agent-model"), ("restore", "user-model")],
        )
        self.assertNotIn("tab-serial", printer._model_override_tabs)

    def test_hammer_subscribe_catch_up_vs_restore(self) -> None:
        """subscribe_tab's catch-up racing restore stays consistent."""
        printer = SlowWirePrinter()
        for i in range(120):
            key, tab = f"conc2-sub-{i}", f"tab-sub-{i}"
            printer.broadcast_agent_model_pick("agent-model", "", key)
            barrier = threading.Barrier(2)

            def subscribe(k: str = key, t: str = tab) -> None:
                barrier.wait()
                printer.subscribe_tab(k, t)

            def restore(t: str = tab) -> None:
                barrier.wait()
                time.sleep(0.001)
                printer.restore_model_pick("user-model", t)

            t1 = threading.Thread(target=subscribe)
            t2 = threading.Thread(target=restore)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self._assert_consistent(printer, tab, i)

    def test_hammer_agent_pick_vs_restore(self) -> None:
        """broadcast_agent_model_pick racing restore stays consistent."""
        printer = SlowWirePrinter()
        for i in range(120):
            key, tab = f"conc2-agent-{i}", f"tab-agent-{i}"
            printer.subscribe_tab(key, tab)
            barrier = threading.Barrier(2)

            def agent_pick(k: str = key, t: str = tab) -> None:
                barrier.wait()
                printer.broadcast_agent_model_pick("agent-model", t, k)

            def restore(t: str = tab) -> None:
                barrier.wait()
                time.sleep(0.001)
                printer.restore_model_pick("user-model", t)

            t1 = threading.Thread(target=agent_pick)
            t2 = threading.Thread(target=restore)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self._assert_consistent(printer, tab, i)


class _ProbePauser:
    """Seam: pause a named thread inside ``_fanout_targets``.

    Delegates to the real (bound) method — nothing under test is
    stubbed; the pause models a thread preemption between the pick's
    target snapshot and its state write, the exact window review
    finding 4 demonstrated.
    """

    def __init__(self, printer: MemoryPrinter, thread_name: str) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()
        self._name = thread_name
        self._real = printer._fanout_targets

    def __call__(self, task_id: Any) -> list[str]:
        result = self._real(task_id)
        if threading.current_thread().name == self._name:
            self.reached.set()
            assert self.release.wait(10)
        return result


class ModelPickCleanupRaceTest(unittest.TestCase):
    """Review finding 4: cleanup writers are serialized with picks.

    ``cleanup_tab`` / ``cleanup_task`` used to mutate the model-
    override lifecycle state under only ``_lock``, so an agent pick
    paused between its target snapshot and its state write could
    resume AFTER a cleanup and resurrect the cleaned state: the closed
    tab reappeared in ``_model_override_tabs`` (with a trailing stale
    ``modelPick`` on the wire), and a completed task's
    ``_task_model_override`` entry was recreated, feeding the dead
    agent's model to every later subscriber.  With both cleanups
    taking ``_model_pick_lock`` the pick-vs-cleanup pair serializes,
    and the pick refuses a task already in ``_closed_tasks``.
    """

    def _paused_pick(
        self, printer: MemoryPrinter, seam: _ProbePauser,
    ) -> tuple[threading.Thread, list[BaseException]]:
        """Start an agent pick that pauses at the seam; return it."""
        errors: list[BaseException] = []

        def pick() -> None:
            try:
                printer.broadcast_agent_model_pick("agent-model", "", "task")
            except BaseException as exc:  # pragma: no cover — fail loudly
                errors.append(exc)

        thread = threading.Thread(target=pick, name="conc2-pick")
        thread.start()
        assert seam.reached.wait(5)
        return thread, errors

    def test_cleanup_tab_is_not_resurrected_by_a_paused_pick(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("task", "tab")
        seam = _ProbePauser(printer, "conc2-pick")
        printer._fanout_targets = seam  # type: ignore[method-assign]
        try:
            thread, errors = self._paused_pick(printer, seam)
            cleanup_done = threading.Event()

            def run_tab_cleanup() -> None:
                printer.cleanup_tab("tab")
                cleanup_done.set()

            cleanup = threading.Thread(target=run_tab_cleanup)
            cleanup.start()
            # The cleanup writer must serialize behind the in-flight
            # pick instead of interleaving into its window.
            self.assertFalse(
                cleanup_done.wait(0.3),
                "cleanup_tab ran inside the pick's snapshot/write window",
            )
            seam.release.set()
            thread.join(10)
            cleanup.join(10)
            self.assertEqual(errors, [])
            self.assertFalse(cleanup.is_alive())
            # Serial order pick → cleanup: the cleaned tab stays gone.
            self.assertNotIn("tab", printer._model_override_tabs)
            self.assertEqual(printer._fanout_targets("task"), [])
        finally:
            seam.release.set()
            printer._fanout_targets = seam._real  # type: ignore[method-assign]

    def test_cleanup_task_override_is_not_resurrected(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("task", "tab")
        seam = _ProbePauser(printer, "conc2-pick")
        printer._fanout_targets = seam  # type: ignore[method-assign]
        try:
            thread, errors = self._paused_pick(printer, seam)
            cleanup_done = threading.Event()

            def run_task_cleanup() -> None:
                printer.cleanup_task("task")
                cleanup_done.set()

            cleanup = threading.Thread(target=run_task_cleanup)
            cleanup.start()
            self.assertFalse(
                cleanup_done.wait(0.3),
                "cleanup_task ran inside the pick's snapshot/write window",
            )
            seam.release.set()
            thread.join(10)
            cleanup.join(10)
            self.assertEqual(errors, [])
            self.assertFalse(cleanup.is_alive())
            # Serial order pick → cleanup: the completed task's
            # override stays gone, and a later subscriber must NOT
            # catch up to the dead agent's model.
            self.assertNotIn("task", printer._task_model_override)
            before = len(printer.emitted)
            printer.subscribe_tab("task", "later-tab")
            stale = [
                e for e in printer.emitted[before:]
                if e.get("type") == "modelPick"
                and e.get("tabId") == "later-tab"
            ]
            self.assertEqual(
                stale, [],
                "a later subscriber caught up to a dead agent's model",
            )
        finally:
            seam.release.set()
            printer._fanout_targets = seam._real  # type: ignore[method-assign]

    def test_pick_after_cleanup_task_never_stores_the_override(self) -> None:
        """A pick that starts AFTER its task was cleaned up must not
        resurrect ``_task_model_override`` — nothing would ever pop it
        again and later subscribers would catch up to the dead agent's
        model.  The transient broadcast itself still reaches the
        lingering subscribers (the linger window exists precisely for
        post-task broadcasts, see ``_transient_targets``)."""
        printer = MemoryPrinter()
        printer.subscribe_tab("task", "tab")
        printer.cleanup_task("task")
        printer.broadcast_agent_model_pick("agent-model", "", "task")
        self.assertNotIn(
            "task", printer._task_model_override,
            "a closed task's model override was resurrected",
        )
        # A later subscriber must not catch up to the dead model.
        before = len(printer.emitted)
        printer.subscribe_tab("task", "later-tab")
        stale = [
            e for e in printer.emitted[before:]
            if e.get("type") == "modelPick" and e.get("tabId") == "later-tab"
        ]
        self.assertEqual(
            stale, [],
            "a later subscriber caught up to a dead agent's model",
        )


if __name__ == "__main__":
    unittest.main()

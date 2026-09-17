# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Concurrency regressions for :mod:`kiss.server.voice_wake_control`.

Reproduces two event-ordering races in the wake-word listener
controller's self-exit reporting (audit candidates C1 and C2), driving
the REAL :class:`VoiceWakeController` with REAL child processes: tiny
listener stand-in scripts written to a temp dir that speak the exact
stdout protocol of ``kiss.server.voice_wake`` (no mocks, patches, or
fakes of the code under test).

C1 — the stdout pump used to deregister the listener from
``_listeners`` *before* awaiting the final ``voiceWakeState`` send, so
while that send was suspended (a) a fresh ``start()`` spawned and
registered a new listener that the stale ``listening: false`` report
then clobbered, and (b) a concurrent ``stop()`` popped ``None`` and
returned without setting ``stopped`` or joining the pumps, voiding its
"no controller coroutine touches the endpoint after this returns"
guarantee.

C2 (CONFIRMED by gpt-5.6-sol round-2 review, finding 4) — the exit
diagnostic used to read ``listener.stderr_tail`` right after
``proc.wait()``, relying on the stderr pump's read future resolving
AND its task running before the stdout pump resumed.  The transport
does wake ``proc.wait()`` waiters only once all pipe transports have
disconnected, but nothing guarantees the stderr TASK is scheduled
first: on CPython 3.14.3 the diagnostic intermittently (~13 % of
isolated runs) omitted the child's final stderr line.  The fix joins
the sibling pump explicitly — ``_pump_stderr`` sets
``listener.stderr_done`` in a ``finally`` and ``_pump_stdout`` awaits
it (bounded) before composing the diagnostic.
``test_exit_diagnostic_includes_late_final_stderr_line`` is the
regression, with a grandchild that delivers the traceback 0.4 s after
the child exited.  The bounded wait's timeout branch is unreachable
without test doubles: ``proc.wait()`` returns only after every pipe
transport has disconnected, so stderr is at EOF and its pump — which
cannot be blocked elsewhere — sets the event as soon as it is
scheduled; the bound is belt-and-braces against future transport
semantics.

Branch-coverage note for the fixed code (branch unreachable without
test doubles, documented here instead of mocked): ``start()``'s
dead-listener guard computes ``pending`` from ``existing.pumps``; an
*empty* ``pending`` with the dead listener still registered is
unreachable, because the stdout pump deregisters the listener before
its task completes — a registered listener with ``returncode`` set
always has a live stdout pump.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from typing import Any

from kiss.server.voice_wake_control import VoiceWakeController

_EXIT_ONCE_SCRIPT = r"""
import os, sys, time
sentinel = sys.argv[1]
if not os.path.exists(sentinel):
    open(sentinel, "w").close()
    print("READY", flush=True)
    print("RuntimeError: mic watchdog gave up", file=sys.stderr, flush=True)
    sys.exit(1)
print("READY", flush=True)
time.sleep(60)
"""
"""Crashes (code 1, stderr traceback) on its first run; sleeps after."""

_CLEAN_EXIT_SCRIPT = r"""
print("READY", flush=True)
"""
"""Exits 0 immediately after READY (a clean self-exit)."""

_GRANDCHILD_STDERR_SCRIPT = r"""
import subprocess, sys
print("READY", flush=True)
subprocess.Popen(
    [sys.executable, "-c",
     "import sys, time; time.sleep(0.4); "
     "sys.stderr.write('RuntimeError: mic exploded in the grandchild\\n'); "
     "sys.stderr.flush()"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=sys.stderr,
)
sys.exit(7)
"""
"""Exits 7 at once; a grandchild holding the inherited stderr fd writes
the traceback line 0.4 s later, so the stderr data reaches the daemon
only AFTER ``proc.wait()`` has completed — the deterministic C2 race."""


def _is_final_state(event: dict[str, Any]) -> bool:
    """Return whether *event* is a ``listening: false`` state report."""
    return (
        event.get("type") == "voiceWakeState"
        and event.get("listening") is False
    )


class _GatedSender:
    """A real send callback that suspends on the final exit report.

    Delivery of every event is recorded in order; the first
    ``listening: false`` state report additionally snapshots which
    listener the controller has registered at delivery-decision time
    and then suspends until the test releases :attr:`gate` — widening
    the natural suspension window of a slow endpoint send so the test
    can interleave ``start()`` / ``stop()`` deterministically.
    """

    def __init__(self, controller: VoiceWakeController, conn_id: str) -> None:
        self.controller = controller
        self.conn_id = conn_id
        self.gate = asyncio.Event()
        self.events: list[dict[str, Any]] = []
        self.snapshots: list[Any] = []

    async def send(self, event: dict[str, Any]) -> None:
        if _is_final_state(event):
            self.snapshots.append(
                self.controller._listeners.get(self.conn_id)
            )
            await self.gate.wait()
        self.events.append(event)

    async def wait_until_report_in_flight(self, timeout: float = 15.0) -> None:
        """Block until the exit report's send has begun (and is gated)."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not self.snapshots:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("exit report never went in flight")
            await asyncio.sleep(0.01)


class TestVoiceWakeExitRaces(unittest.TestCase):
    """C1/C2: self-exit reporting races in ``VoiceWakeController``."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _script(self, name: str, body: str) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        return path

    def _run(self, coro: Any) -> Any:
        asyncio.run(coro)

    def test_exit_report_never_clobbers_a_fresh_start(self) -> None:
        """C1(a): a fresh ``start()`` racing the in-flight exit report.

        The self-exited listener's final ``listening: false`` state
        must be attributed to the exited listener (delivered while it
        is still the registered one) and must be delivered BEFORE a
        concurrent fresh ``start()`` registers its new listener — the
        client must never see the stale report after the restart.
        """

        async def _scenario() -> None:
            script = self._script("exit_once.py", _EXIT_ONCE_SCRIPT)
            sentinel = os.path.join(self.tmp, "ran-once")
            controller = VoiceWakeController(
                [sys.executable, "-u", script, sentinel]
            )
            sender = _GatedSender(controller, "c1")
            try:
                await controller.start("c1", None, sender.send)
                first = controller._listeners["c1"]
                await sender.wait_until_report_in_flight()
                start_task = asyncio.ensure_future(
                    controller.start("c1", None, sender.send)
                )
                await asyncio.sleep(0.05)
                sender.gate.set()
                await asyncio.wait_for(start_task, 15.0)
                idx = len(sender.events)
                # The second listener runs the sleeping branch of the
                # script; wait for its READY-derived listening: true.
                deadline = asyncio.get_running_loop().time() + 15.0
                while not any(
                    e.get("type") == "voiceWakeState"
                    and e.get("listening") is True
                    for e in sender.events[idx:]
                ):
                    if asyncio.get_running_loop().time() > deadline:
                        raise AssertionError(
                            f"no fresh listening state in "
                            f"{sender.events[idx:]!r}"
                        )
                    await asyncio.sleep(0.02)
                self.assertFalse(
                    [e for e in sender.events[idx:] if _is_final_state(e)],
                    "stale exit report delivered after the fresh start",
                )
                self.assertTrue(controller.running("c1"))
                self.assertIsNot(controller._listeners["c1"], first)
                # The report was decided while the exited listener was
                # still the registered one (report first, deregister
                # after), and it was delivered exactly once.
                self.assertIs(sender.snapshots[0], first)
                self.assertEqual(
                    len([e for e in sender.events if _is_final_state(e)]), 1,
                )
                error = next(
                    e for e in sender.events if _is_final_state(e)
                ).get("error", "")
                self.assertIn("code 1", error)
                self.assertIn("mic watchdog gave up", error)
            finally:
                sender.gate.set()
                await controller.stop_all()

        self._run(_scenario())

    def test_stop_racing_the_exit_report_joins_the_pump(self) -> None:
        """C1(b): ``stop()`` racing the in-flight exit report.

        ``stop()`` must find the still-registered exited listener,
        join its pumps, and only return once no controller coroutine
        can touch the connection's endpoint any more — it must not pop
        ``None`` and return while the report send is still live.
        """

        async def _scenario() -> None:
            script = self._script("exit_once.py", _EXIT_ONCE_SCRIPT)
            sentinel = os.path.join(self.tmp, "ran-once")
            controller = VoiceWakeController(
                [sys.executable, "-u", script, sentinel]
            )
            sender = _GatedSender(controller, "c1")
            try:
                await controller.start("c1", None, sender.send)
                first = controller._listeners["c1"]
                await sender.wait_until_report_in_flight()
                stop_task = asyncio.ensure_future(controller.stop("c1"))
                await asyncio.sleep(0.05)
                if stop_task.done():
                    self.assertTrue(
                        all(t.done() for t in first.pumps),
                        "stop() returned while the exit-report pump "
                        "was still live",
                    )
                sender.gate.set()
                await asyncio.wait_for(stop_task, 15.0)
                self.assertTrue(all(t.done() for t in first.pumps))
                self.assertFalse(controller.running("c1"))
                delivered = len(sender.events)
                await asyncio.sleep(0.1)
                self.assertEqual(
                    len(sender.events), delivered,
                    "an event reached the endpoint after stop() returned",
                )
            finally:
                sender.gate.set()
                await controller.stop_all()

        self._run(_scenario())

    def test_start_survives_a_wedged_exit_report_send(self) -> None:
        """C1 fallback: the exit report's send never completes.

        When the in-flight report is wedged on a stuck endpoint, a
        fresh ``start()`` must not be swallowed forever: after the
        bounded join it force-deregisters the dead listener and spawns
        a new one.  (Covers the timed-out-join branch of ``start()``'s
        dead-listener guard; takes ~5 s by design.)
        """

        async def _scenario() -> None:
            script = self._script("exit_once.py", _EXIT_ONCE_SCRIPT)
            sentinel = os.path.join(self.tmp, "ran-once")
            controller = VoiceWakeController(
                [sys.executable, "-u", script, sentinel]
            )
            sender = _GatedSender(controller, "c1")
            try:
                await controller.start("c1", None, sender.send)
                first = controller._listeners["c1"]
                await sender.wait_until_report_in_flight()
                # Never release the gate: the report send stays wedged.
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 15.0
                )
                self.assertTrue(controller.running("c1"))
                self.assertIsNot(controller._listeners["c1"], first)
            finally:
                sender.gate.set()
                await controller.stop_all()

        self._run(_scenario())

    def test_timed_out_restart_retires_the_old_pump_and_its_report(
        self,
    ) -> None:
        """Review finding 7: the timed-out dead-listener join must
        RETIRE the old listener, not merely deregister it.

        Pre-fix, ``start()``'s timeout branch deleted the old listener
        from ``_listeners`` but left its wedged stdout pump alive and
        unfindable by ``stop``/``stop_all``; when the wedged send later
        resumed, the old ``listening: false`` landed AFTER the
        replacement's ``listening: true``, telling the client the new
        listener was dead.  Post-fix the old pumps are cancelled and
        awaited (and the listener marked stopped) before the
        replacement spawns, so releasing the endpoint later delivers
        nothing.
        """

        async def _scenario() -> None:
            script = self._script("exit_once.py", _EXIT_ONCE_SCRIPT)
            sentinel = os.path.join(self.tmp, "ran-once")
            controller = VoiceWakeController(
                [sys.executable, "-u", script, sentinel]
            )
            sender = _GatedSender(controller, "c1")
            try:
                await controller.start("c1", None, sender.send)
                old = controller._listeners["c1"]
                await sender.wait_until_report_in_flight()
                # Never release the gate: the join times out (~5 s).
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 15.0
                )
                self.assertIsNot(controller._listeners["c1"], old)
                self.assertTrue(
                    old.stopped,
                    "the timed-out restart left the old listener "
                    "unretired",
                )
                self.assertTrue(
                    all(t.done() for t in old.pumps),
                    "the timed-out restart abandoned a live old pump",
                )
                # The replacement runs the sleeping branch: wait for
                # its listening: true.
                deadline = asyncio.get_running_loop().time() + 15.0
                while not any(
                    e.get("type") == "voiceWakeState"
                    and e.get("listening") is True
                    for e in sender.events
                ):
                    if asyncio.get_running_loop().time() > deadline:
                        raise AssertionError(
                            f"no fresh listening state in {sender.events!r}"
                        )
                    await asyncio.sleep(0.02)
                # Un-wedge the endpoint: the cancelled report must NOT
                # arrive now (pre-fix it did, after listening: true).
                sender.gate.set()
                await asyncio.sleep(0.2)
                self.assertFalse(
                    [e for e in sender.events if _is_final_state(e)],
                    "the old listener's stale exit report was delivered "
                    "after the replacement started listening",
                )
                self.assertTrue(controller.running("c1"))
            finally:
                sender.gate.set()
                await controller.stop_all()

        self._run(_scenario())

    def test_clean_self_exit_reports_no_error(self) -> None:
        """A code-0 self-exit reports ``listening: false`` without error."""

        async def _scenario() -> None:
            script = self._script("clean.py", _CLEAN_EXIT_SCRIPT)
            controller = VoiceWakeController([sys.executable, "-u", script])
            sender = _GatedSender(controller, "c1")
            try:
                await controller.start("c1", None, sender.send)
                await sender.wait_until_report_in_flight()
                sender.gate.set()
                deadline = asyncio.get_running_loop().time() + 15.0
                while not any(_is_final_state(e) for e in sender.events):
                    if asyncio.get_running_loop().time() > deadline:
                        raise AssertionError(
                            f"no final state in {sender.events!r}"
                        )
                    await asyncio.sleep(0.01)
                final = next(
                    e for e in sender.events if _is_final_state(e)
                )
                self.assertNotIn("error", final)
                self.assertFalse(controller.running("c1"))
            finally:
                sender.gate.set()
                await controller.stop_all()

        self._run(_scenario())

    def test_stop_while_reaping_is_held_by_a_grandchild(self) -> None:
        """A ``stop()`` landing before the exit report reports nothing.

        The grandchild holds the inherited stderr fd for ~0.4 s after
        the child's exit, keeping the stdout pump suspended in
        ``proc.wait()`` (exit reaping is pipe-closure driven); a
        ``stop()`` issued in that window owns the shutdown — it kills
        the process group (grandchild included), joins the pumps, and
        no final ``listening: false`` report may be delivered.
        """

        async def _scenario() -> None:
            script = self._script(
                "grandchild.py", _GRANDCHILD_STDERR_SCRIPT
            )
            controller = VoiceWakeController([sys.executable, "-u", script])
            sender = _GatedSender(controller, "c1")
            try:
                await controller.start("c1", None, sender.send)
                deadline = asyncio.get_running_loop().time() + 15.0
                while not sender.events:
                    if asyncio.get_running_loop().time() > deadline:
                        raise AssertionError("listener never became ready")
                    await asyncio.sleep(0.01)
                # READY seen; the child exits at once but reaping is
                # held by the grandchild (~0.4 s).  Stop inside that
                # window.
                await asyncio.sleep(0.1)
                first = controller._listeners["c1"]
                await asyncio.wait_for(controller.stop("c1"), 15.0)
                self.assertTrue(all(t.done() for t in first.pumps))
                self.assertFalse(controller.running("c1"))
                self.assertFalse(
                    [e for e in sender.events if _is_final_state(e)],
                    "a stopped listener must not report a final state",
                )
            finally:
                sender.gate.set()
                await controller.stop_all()

        self._run(_scenario())

    def test_exit_diagnostic_includes_late_final_stderr_line(self) -> None:
        """C2 regression: the exit error carries the last stderr line.

        The grandchild delivers the traceback line 0.4 s after the
        child exited.  Pre-fix the stdout pump read ``stderr_tail``
        straight after ``proc.wait()`` and intermittently lost this
        line to task-scheduling order (round-2 review, finding 4);
        post-fix it joins the stderr pump via ``stderr_done`` first,
        so the diagnostic deterministically contains it.
        """

        async def _scenario() -> None:
            script = self._script(
                "grandchild.py", _GRANDCHILD_STDERR_SCRIPT
            )
            controller = VoiceWakeController([sys.executable, "-u", script])
            sender = _GatedSender(controller, "c1")
            try:
                await controller.start("c1", None, sender.send)
                await sender.wait_until_report_in_flight()
                sender.gate.set()
                final: dict[str, Any] | None = None
                deadline = asyncio.get_running_loop().time() + 15.0
                while final is None:
                    if asyncio.get_running_loop().time() > deadline:
                        raise AssertionError(
                            f"no final state in {sender.events!r}"
                        )
                    await asyncio.sleep(0.01)
                    final = next(
                        (e for e in sender.events if _is_final_state(e)),
                        None,
                    )
                error = final.get("error", "")
                self.assertIn("code 7", error)
                self.assertIn("mic exploded in the grandchild", error)
                self.assertFalse(controller.running("c1"))
            finally:
                sender.gate.set()
                await controller.stop_all()

        self._run(_scenario())


if __name__ == "__main__":
    unittest.main()

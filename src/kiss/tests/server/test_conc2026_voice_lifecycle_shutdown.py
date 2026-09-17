# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: voice-wake lifecycle is send-independent, cancel- and shutdown-safe.

gpt-5.6-sol round-4 review, findings 2-5 and 7, each demonstrated with
a real listener child process:

2.  The per-connection lifecycle lock was held across client sends
    (the duplicate-start re-report and the spawn-failure report).  A
    send callback is not time-bounded by the ``SendCallback``
    contract, so one blocked report held the lock forever, stalling
    every later ``stop()``/``stop_all()`` and keeping the child alive.
    Fixed: the duplicate-start re-report runs as a pump task and the
    failure report is delivered by ``start()`` after the lock is
    released — no send ever runs under the lock.

3.  Cancelling an active ``stop()`` orphaned the child: the listener
    left ``_listeners`` before the first cancellable teardown await,
    and nothing else retained the process.  The cancellation is
    production-reachable (the UDS handler drain cancels straggling
    disconnect cleanups).  Fixed: termination runs as an owned reap
    task registered in ``_reap_tasks`` in the same no-await block as
    the deregistration; the stop shield-awaits it, and ``stop_all()``
    joins outstanding reap tasks.

4.  The post-SIGKILL ``proc.wait()`` was unbounded while the lifecycle
    lock was held: a detached descendant holding the inherited
    stdout/stderr pipes keeps asyncio's subprocess transport pending
    even after the parent is dead.  Fixed: the wait is bounded by
    ``_KILL_REAP_SECONDS`` and abandoned with a log line (SIGKILL was
    already delivered to the process group).

5.  ``stop_all()`` snapshotted only ``_listeners``, so a ``start()``
    suspended in its spawn (holding the lifecycle lock, listener not
    yet registered) survived it and published a live child after
    ``stop_all()`` returned.  Fixed: the snapshot also covers every
    connection with lifecycle-lock activity (the queued ``stop()``
    then reaps the late registration), and a ``_closing`` barrier
    refuses starts that had not yet entered lifecycle processing.

7.  A failed dead-listener restart returned without collecting the
    connection's ``_generations`` entry once the retired pumps had
    already completed — an ownerless entry leaked per failed restart.
    Fixed: every spawn-refusal/failure path drops the entry when
    nothing owns it any more.

Real ``VoiceWakeController``, real child processes speaking the real
stdout protocol, real cancellations; the only seam is a delegating
wrapper around ``asyncio.create_subprocess_exec`` that gates (then
performs) the real spawn, adopted from the reviewer's reproduction.
No mocks.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.server.voice_wake_control import (
    _KILL_REAP_SECONDS,
    _TERM_GRACE_SECONDS,
    VoiceWakeController,
)

_SLEEP_SCRIPT = """\
print("READY", flush=True)
import time
time.sleep(120)
"""

_IGNORE_TERM_SCRIPT = """\
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("READY", flush=True)
time.sleep(120)
"""

# The child ignores SIGTERM and hands its inherited stdout/stderr to a
# detached grandchild in its OWN session: SIGKILL to the child's
# process group kills the child but not the grandchild, whose open
# pipe copies keep asyncio's subprocess transport (and so
# ``Process.wait()``) pending after the parent is dead.
_ESCAPED_PIPE_HOLDER_SCRIPT = """\
import os, signal, subprocess, sys, time
piddir = sys.argv[1]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
grand = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)"],
    stdin=subprocess.DEVNULL,
    stdout=sys.stdout,
    stderr=sys.stderr,
    start_new_session=True,
)
with open(os.path.join(piddir, "grandchild"), "w") as fh:
    fh.write(str(grand.pid))
print("READY", flush=True)
time.sleep(120)
"""

_EXIT_SCRIPT = 'print("READY", flush=True)\n'


def _alive(pid: int) -> bool:
    """Return whether a process identifier still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_group(pid: int) -> None:
    """Best-effort SIGKILL of a test child's process group."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def _poll(predicate: Any, timeout: float = 15.0) -> None:
    """Wait until a synchronous predicate becomes true."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true in time")
        await asyncio.sleep(0.01)


class VoiceLifecycleShutdownTest(unittest.TestCase):
    """Round-4 findings 2-5 and 7 against real listener children."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _controller(self, script: str, *extra: str) -> VoiceWakeController:
        path = os.path.join(self.tmp, "listener.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        return VoiceWakeController(
            [sys.executable, "-u", path, *extra],
        )

    def test_duplicate_start_send_does_not_hold_the_lifecycle_lock(
        self,
    ) -> None:
        """Finding 2 (duplicate-start path): the reviewer's schedule.

        The duplicate start's ``listening: true`` re-report blocks
        forever inside the client callback; the lifecycle lock must
        already be free, and a concurrent ``stop()`` must complete
        within its bounded joins and reap the child.
        """

        async def scenario() -> None:
            controller = self._controller(_SLEEP_SCRIPT)
            ready = asyncio.Event()

            async def collect(event: dict[str, Any]) -> None:
                if event.get("listening") is True:
                    ready.set()

            await controller.start("c1", None, collect)
            await asyncio.wait_for(ready.wait(), 15)
            proc = controller._listeners["c1"].proc
            entered = asyncio.Event()
            release = asyncio.Event()

            async def blocked_send(event: dict[str, Any]) -> None:
                entered.set()
                await release.wait()

            try:
                # Pre-fix, this call itself hung inside the blocked
                # re-report while holding the lifecycle lock.
                await asyncio.wait_for(
                    controller.start("c1", None, blocked_send), 5,
                )
                await asyncio.wait_for(entered.wait(), 5)
                self.assertNotIn(
                    "c1", controller._lifecycle_holds,
                    "the duplicate-start re-report holds the lifecycle "
                    "lock across the client send",
                )
                # Pre-fix, stop() queued on the lock forever and the
                # child stayed alive; the send stays blocked the whole
                # time, so completion proves independence from it.
                await asyncio.wait_for(controller.stop("c1"), 20)
                self.assertFalse(controller.running("c1"))
                self.assertIsNotNone(proc.returncode)
            finally:
                release.set()
                await controller.stop_all()

        asyncio.run(scenario())

    def test_spawn_failure_report_is_delivered_off_the_lock(self) -> None:
        """Finding 2 (spawn-failure path): the report cannot wedge stop.

        The failure report's send blocks forever; ``start()`` may stay
        pending inside it, but the lifecycle lock must be free and a
        concurrent ``stop()`` must return at once.
        """

        async def scenario() -> None:
            controller = VoiceWakeController(
                ["/nonexistent-binary-for-kiss-test"],
            )
            entered = asyncio.Event()
            release = asyncio.Event()
            events: list[dict[str, Any]] = []

            async def blocked_send(event: dict[str, Any]) -> None:
                entered.set()
                await release.wait()
                events.append(dict(event))

            start = asyncio.ensure_future(
                controller.start("c1", None, blocked_send),
            )
            try:
                await asyncio.wait_for(entered.wait(), 5)
                self.assertFalse(start.done())
                self.assertNotIn(
                    "c1", controller._lifecycle_holds,
                    "the spawn-failure report holds the lifecycle lock "
                    "across the client send",
                )
                await asyncio.wait_for(controller.stop("c1"), 5)
                release.set()
                await asyncio.wait_for(start, 5)
                self.assertTrue(
                    any(
                        "failed to start" in event.get("error", "")
                        for event in events
                    ),
                    events,
                )
            finally:
                release.set()
                await asyncio.gather(start, return_exceptions=True)
                await controller.stop_all()

        asyncio.run(scenario())

    def test_cancelled_stop_cannot_orphan_the_child(self) -> None:
        """Finding 3: the reviewer's schedule with a SIGTERM-proof child.

        ``stop()`` is cancelled right after deregistration.  Ownership
        must survive in a reap task: a replacement child may start
        meanwhile, and ``stop_all()`` must reap BOTH the replacement
        and the cancelled stop's child before returning.
        """

        async def scenario() -> None:
            controller = self._controller(_IGNORE_TERM_SCRIPT)
            ready = asyncio.Event()

            async def collect(event: dict[str, Any]) -> None:
                if event.get("event") == "ready":
                    ready.set()

            await controller.start("c1", None, collect)
            old = controller._listeners["c1"]
            # READY proves the child installed its SIGTERM handler:
            # a SIGTERM landing before that would kill it at once and
            # skip the retained-ownership window under test.
            await asyncio.wait_for(ready.wait(), 15)
            stopping = asyncio.ensure_future(controller.stop("c1"))
            try:
                await _poll(lambda: not controller.running("c1"))
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
                # Ownership retained: a reap task holds the child.
                self.assertTrue(
                    any(not t.done() for t in controller._reap_tasks),
                    "no reap task owns the cancelled stop's child",
                )
                # A replacement spawns while the old child is reaped.
                await controller.start("c1", None, collect)
                replacement = controller._listeners["c1"]
                self.assertNotEqual(replacement.proc.pid, old.proc.pid)
                # Pre-fix, stop_all() returned with the old child
                # alive forever (it ignores SIGTERM; only the retained
                # reap task escalates to SIGKILL).
                await asyncio.wait_for(controller.stop_all(), 30)
                self.assertFalse(controller.running("c1"))
                self.assertFalse(_alive(old.proc.pid))
                self.assertIsNotNone(replacement.proc.returncode)
            finally:
                _kill_group(old.proc.pid)
                await asyncio.wait_for(controller.stop_all(), 30)

        asyncio.run(scenario())

    def test_post_sigkill_reap_wait_is_bounded(self) -> None:
        """Finding 4: the reviewer's escaped-pipe-holder schedule.

        The grandchild keeps the child's pipes open forever; the
        reviewer's probe showed ``stop()`` still pending (holding the
        lifecycle lock) long after the parent died with ``-SIGKILL``.
        Post-fix the wait is abandoned at ``_KILL_REAP_SECONDS`` and
        ``stop()`` completes while the escaped process still runs.
        """

        async def scenario() -> None:
            controller = self._controller(
                _ESCAPED_PIPE_HOLDER_SCRIPT, self.tmp,
            )

            async def discard(event: dict[str, Any]) -> None:
                return

            await controller.start("c1", None, discard)
            listener = controller._listeners["c1"]
            grand_path = Path(self.tmp) / "grandchild"
            await _poll(grand_path.exists)
            grand_pid = int(grand_path.read_text())
            try:
                # SIGTERM grace + bounded SIGKILL reap + bounded pump
                # joins; pre-fix this wait_for never returned.
                await asyncio.wait_for(
                    controller.stop("c1"),
                    _TERM_GRACE_SECONDS + _KILL_REAP_SECONDS + 12.0,
                )
                self.assertFalse(controller.running("c1"))
                self.assertNotIn("c1", controller._lifecycle_holds)
                self.assertEqual(
                    listener.proc.returncode, -signal.SIGKILL,
                )
                self.assertTrue(
                    _alive(grand_pid),
                    "the escaped holder died early; the abandoned-wait "
                    "path was not exercised",
                )
            finally:
                _kill_group(grand_pid)
                try:
                    os.kill(grand_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await controller.stop_all()

        asyncio.run(scenario())

    def test_stop_all_covers_an_in_flight_start(self) -> None:
        """Finding 5: the reviewer's gated-spawn schedule.

        A start suspended inside the subprocess spawn holds its
        lifecycle lock but has not registered a listener.  Pre-fix,
        ``stop_all()`` returned immediately and the released start
        published a live child afterwards.  Post-fix, ``stop_all()``
        queues behind the in-flight start and reaps its registration,
        and a brand-new start during the shutdown is refused outright.
        """

        async def scenario() -> None:
            controller = self._controller(_SLEEP_SCRIPT)
            real_spawn = asyncio.create_subprocess_exec
            entered = asyncio.Event()
            release = asyncio.Event()
            spawned: list[asyncio.subprocess.Process] = []

            async def gated_spawn(*args: Any, **kwargs: Any) -> Any:
                entered.set()
                await release.wait()
                proc = await real_spawn(*args, **kwargs)
                spawned.append(proc)
                return proc

            async def discard(event: dict[str, Any]) -> None:
                return

            setattr(asyncio, "create_subprocess_exec", gated_spawn)
            start = asyncio.ensure_future(
                controller.start("c1", None, discard),
            )
            stop_all: asyncio.Task[None] | None = None
            try:
                await asyncio.wait_for(entered.wait(), 5)
                stop_all = asyncio.ensure_future(controller.stop_all())
                await _poll(lambda: controller._closing > 0)
                await asyncio.sleep(0.05)
                # Pre-fix, stop_all() had already returned here.
                self.assertFalse(
                    stop_all.done(),
                    "stop_all() ignored the in-flight start",
                )
                # A start that had not entered lifecycle processing is
                # refused by the closing barrier — no child, and an
                # explicit failure report.
                refusals: list[dict[str, Any]] = []

                async def collect(event: dict[str, Any]) -> None:
                    refusals.append(dict(event))

                await asyncio.wait_for(
                    controller.start("c2", None, collect), 5,
                )
                self.assertFalse(controller.running("c2"))
                self.assertTrue(
                    any(
                        event.get("listening") is False
                        and "not started" in event.get("error", "")
                        for event in refusals
                    ),
                    refusals,
                )
                release.set()
                await asyncio.wait_for(start, 15)
                await asyncio.wait_for(stop_all, 15)
                stop_all = None
                # The late registration was reaped before stop_all
                # returned — no listener survives it.
                self.assertFalse(controller.running("c1"))
                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(spawned[0].returncode)
            finally:
                setattr(asyncio, "create_subprocess_exec", real_spawn)
                release.set()
                await asyncio.gather(start, return_exceptions=True)
                if stop_all is not None:
                    await asyncio.gather(stop_all, return_exceptions=True)
                await controller.stop_all()

        asyncio.run(scenario())

    def test_failed_restart_frees_the_generation_entry(self) -> None:
        """Finding 7: the reviewer's finite failed-restart schedule.

        The listener self-exits; its exit report wedges until the
        restart's cancellation collects the pump DURING the bounded
        joins (while the old listener is still registered, so the
        done-callback keeps the generation).  The replacement spawn
        then fails.  Pre-fix, ``_generations['c1']`` survived with no
        owner; post-fix every spawn-failure path collects it.
        """

        async def scenario() -> None:
            controller = self._controller(_EXIT_SCRIPT)
            first_false = asyncio.Event()
            never = asyncio.Event()
            false_count = 0
            events: list[dict[str, Any]] = []

            async def send(event: dict[str, Any]) -> None:
                nonlocal false_count
                if event.get("listening") is False:
                    false_count += 1
                    if false_count == 1:
                        first_false.set()
                        await never.wait()
                events.append(dict(event))

            await controller.start("c1", None, send)
            await asyncio.wait_for(first_false.wait(), 15)
            controller._listener_args = [
                "/definitely/missing/kiss-voice-listener",
            ]
            try:
                await asyncio.wait_for(
                    controller.start("c1", None, send), 20,
                )
                self.assertFalse(controller.running("c1"))
                self.assertFalse(controller._retiring.get("c1"))
                self.assertEqual(
                    controller._generations, {},
                    "a failed restart leaked an ownerless generation "
                    "entry",
                )
                self.assertTrue(
                    any(
                        "failed to start" in event.get("error", "")
                        for event in events
                    ),
                    events,
                )
            finally:
                never.set()
                await controller.stop_all()

        asyncio.run(scenario())


class VoiceRound6ShutdownTest(unittest.TestCase):
    """Round-6 findings 2-4 against real listener children.

    2.  A ``start()`` that queued behind ``stop_all()``'s OWN per-
        connection ``stop()`` took its lifecycle reference after the
        one-time snapshot, so ``stop_all()`` lowered ``_closing`` and
        returned before the woken waiter ran — the queued start then
        spawned a listener after shutdown completed.  Fixed:
        ``stop_all()`` loops until no listeners, lifecycle
        holders/waiters, or unfinished reap tasks remain; the final
        emptiness check runs with no await before returning.

    3.  The stop's pump join/cancel ran in the cancellable caller
        AFTER the shielded reap await, and ``stop_all()`` joined only
        reap tasks: a stop cancelled during the pump join left live
        endpoint-touching pump tasks (and their generation) behind
        shutdown.  Fixed: the owned reap task performs the ENTIRE
        teardown (bounded termination, pump join/cancel, generation
        GC), and ``stop_all()`` additionally settles any still-pending
        retired pumps — bounded — before returning.

    4.  Cancelling a dead-listener restart at the replacement
        ``create_subprocess_exec`` bypassed the ``OSError``-only
        generation GC, leaving an ownerless ``_generations`` entry —
        and could orphan a child the loop had already forked.  Fixed:
        the spawn is shielded and, on cancellation, handed to an owned
        reap-registered disposal task that terminates any materialized
        child and repeats the generation GC.

    Real ``VoiceWakeController``, real children, real cancellations;
    the only seam is the reviewer's delegating gate around
    ``asyncio.create_subprocess_exec``.  No mocks.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _controller(self, script: str, *extra: str) -> VoiceWakeController:
        path = os.path.join(self.tmp, "listener.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        return VoiceWakeController(
            [sys.executable, "-u", path, *extra],
        )

    def test_stop_all_drains_a_start_queued_behind_its_own_stop(
        self,
    ) -> None:
        """Finding 2: the reviewer's SIGTERM-immune queued-start schedule.

        The start queues on c1's lifecycle lock while ``stop_all()``'s
        own stop reaps the SIGTERM-immune child.  Pre-fix, the one-time
        snapshot missed the waiter and it spawned a listener after
        ``stop_all()`` returned; post-fix the sweep loops until the
        waiter has been admitted and refused under the still-raised
        barrier.
        """

        async def scenario() -> None:
            controller = self._controller(_IGNORE_TERM_SCRIPT)
            ready = asyncio.Event()

            async def send(event: dict[str, Any]) -> None:
                if event.get("event") == "ready":
                    ready.set()

            await controller.start("c1", None, send)
            await asyncio.wait_for(ready.wait(), 15)
            first_pid = controller._listeners["c1"].proc.pid
            stopping_all = asyncio.ensure_future(controller.stop_all())
            try:
                # stop_all's own stop has deregistered c1 but is still
                # reaping the SIGTERM-immune child under c1's lock.
                await _poll(
                    lambda: controller._closing == 1
                    and not controller.running("c1")
                )
                queued = asyncio.ensure_future(
                    controller.start("c1", None, send)
                )
                try:
                    await _poll(
                        lambda: controller._lifecycle_holds.get("c1", 0) >= 2
                    )
                    await asyncio.wait_for(stopping_all, 45)
                    await asyncio.wait_for(queued, 15)
                finally:
                    await asyncio.gather(queued, return_exceptions=True)
                self.assertFalse(
                    controller.running("c1"),
                    "a start queued behind stop_all's own stop spawned "
                    "a listener after shutdown returned",
                )
                self.assertEqual(controller._listeners, {})
                self.assertEqual(controller._lifecycle_holds, {})
                self.assertEqual(controller._generations, {})
                self.assertFalse(_alive(first_pid))
            finally:
                _kill_group(first_pid)
                await asyncio.wait_for(controller.stop_all(), 30)

        asyncio.run(scenario())

    def test_stop_all_settles_pumps_of_a_cancelled_stop(self) -> None:
        """Finding 3: the reviewer's blocked-duplicate-report schedule.

        The duplicate-start re-report pump wedges in its send; the
        stop is cancelled while the owned reap task joins the pumps.
        ``stop_all()`` must not return while a cancellation-responsive
        endpoint-touching pump is still pending, and the generation
        must be collected.
        """

        async def scenario() -> None:
            controller = self._controller(_SLEEP_SCRIPT)
            ready = asyncio.Event()

            async def initial_send(event: dict[str, Any]) -> None:
                if event.get("event") == "ready":
                    ready.set()

            await controller.start("c1", None, initial_send)
            await asyncio.wait_for(ready.wait(), 15)
            pid = controller._listeners["c1"].proc.pid
            blocked = asyncio.Event()
            release = asyncio.Event()

            async def blocked_duplicate_send(
                event: dict[str, Any],
            ) -> None:
                blocked.set()
                await release.wait()

            await controller.start("c1", None, blocked_duplicate_send)
            await asyncio.wait_for(blocked.wait(), 15)
            stopping = asyncio.ensure_future(controller.stop("c1"))
            try:
                # Deregistered; the owned reap task now runs the full
                # teardown, including the pump join the cancellation
                # abandons in the caller.
                await _poll(
                    lambda: not controller.running("c1")
                    and any(not t.done() for t in controller._reap_tasks)
                )
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
                await asyncio.wait_for(controller.stop_all(), 30)
                self.assertEqual(
                    controller._retiring, {},
                    "stop_all returned with an uncancelled endpoint-"
                    "touching pump",
                )
                self.assertEqual(
                    controller._generations, {},
                    "stop_all returned with a retained generation",
                )
                self.assertFalse(_alive(pid))
            finally:
                release.set()
                _kill_group(pid)
                await asyncio.wait_for(controller.stop_all(), 30)

        asyncio.run(scenario())

    def test_stop_all_settles_escaped_pipe_pumps_of_a_cancelled_stop(
        self,
    ) -> None:
        """Finding 3, bounded-SIGKILL composition: escaped pipe holder.

        The SIGTERM-immune child hands its pipes to a detached
        grandchild, so the reap abandons the post-SIGKILL wait and the
        pumps pend on the held-open pipes.  A cancelled stop must
        still lead to a ``stop_all()`` that cancels/joins those pumps
        and collects the generation before returning.
        """

        async def scenario() -> None:
            grand_path = Path(self.tmp) / "grandchild"
            controller = self._controller(
                _ESCAPED_PIPE_HOLDER_SCRIPT, self.tmp,
            )
            ready = asyncio.Event()

            async def send(event: dict[str, Any]) -> None:
                if event.get("event") == "ready":
                    ready.set()

            await controller.start("c1", None, send)
            await asyncio.wait_for(ready.wait(), 15)
            await _poll(grand_path.exists)
            grand_pid = int(grand_path.read_text(encoding="utf-8"))
            pid = controller._listeners["c1"].proc.pid
            stopping = asyncio.ensure_future(controller.stop("c1"))
            try:
                await _poll(lambda: not controller.running("c1"))
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
                # Bounded: SIGTERM grace + abandoned post-SIGKILL wait
                # + pump join/cancel.
                deadline = (
                    _TERM_GRACE_SECONDS + _KILL_REAP_SECONDS + 6.0 + 20.0
                )
                await asyncio.wait_for(controller.stop_all(), deadline)
                self.assertEqual(
                    controller._retiring, {},
                    "stop_all returned with pending retired pumps "
                    "despite their sends being cancellation-responsive",
                )
                self.assertEqual(
                    controller._generations, {},
                    "stop_all returned with a retained generation",
                )
                self.assertFalse(_alive(pid))
            finally:
                _kill_group(grand_pid)
                try:
                    os.kill(grand_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _kill_group(pid)
                await asyncio.wait_for(controller.stop_all(), 30)

        asyncio.run(scenario())

    def test_cancelled_restart_spawn_frees_generation_and_child(
        self,
    ) -> None:
        """Finding 4: the reviewer's gated-replacement-spawn schedule.

        The first listener self-exits and its exit report wedges; the
        restart collects it, deletes the listener, and is cancelled
        while awaiting the replacement spawn.  The generation entry
        must be collected at once, and any child the event loop
        already forked must be terminated by an owned task that
        ``stop_all()`` joins.
        """

        async def scenario() -> None:
            controller = self._controller(_EXIT_SCRIPT)
            false_entered = asyncio.Event()
            never = asyncio.Event()

            async def blocked_final(event: dict[str, Any]) -> None:
                if event.get("listening") is False:
                    false_entered.set()
                    await never.wait()

            await controller.start("c1", None, blocked_final)
            await asyncio.wait_for(false_entered.wait(), 15)
            # The replacement must be long-lived so an orphan would be
            # observable: only the owned disposal task may reap it.
            sleeper = os.path.join(self.tmp, "sleeper.py")
            with open(sleeper, "w", encoding="utf-8") as fh:
                fh.write(_SLEEP_SCRIPT)
            controller._listener_args = [sys.executable, "-u", sleeper]

            real_spawn = asyncio.create_subprocess_exec
            spawn_entered = asyncio.Event()
            release_spawn = asyncio.Event()
            spawned: list[Any] = []

            async def gated_spawn(*args: Any, **kwargs: Any) -> Any:
                spawn_entered.set()
                await release_spawn.wait()
                proc = await real_spawn(*args, **kwargs)
                spawned.append(proc)
                return proc

            setattr(asyncio, "create_subprocess_exec", gated_spawn)
            restart = asyncio.ensure_future(
                controller.start("c1", None, blocked_final)
            )
            try:
                await asyncio.wait_for(spawn_entered.wait(), 20)
                restart.cancel()
                await asyncio.gather(restart, return_exceptions=True)
                self.assertFalse(controller.running("c1"))
                self.assertFalse(controller._retiring.get("c1"))
                self.assertEqual(
                    controller._generations, {},
                    "a cancelled restart left an ownerless generation "
                    "entry",
                )
                self.assertTrue(
                    any(not t.done() for t in controller._reap_tasks),
                    "no owned task holds the abandoned spawn",
                )
                release_spawn.set()
                # stop_all joins the disposal task: the materialized
                # child must be terminated, nothing retained.
                await asyncio.wait_for(controller.stop_all(), 30)
                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(
                    spawned[0].returncode,
                    "the abandoned replacement child was orphaned",
                )
                self.assertEqual(controller._generations, {})
                self.assertFalse(controller.running("c1"))
            finally:
                setattr(asyncio, "create_subprocess_exec", real_spawn)
                release_spawn.set()
                never.set()
                await asyncio.gather(restart, return_exceptions=True)
                for proc in spawned:
                    _kill_group(proc.pid)
                await controller.stop_all()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

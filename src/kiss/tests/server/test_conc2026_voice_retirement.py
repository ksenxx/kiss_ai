# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a cancellation-resistant stale voice report cannot outrun a restart.

gpt-5.6-sol round-2 review, finding 3: the timed-out-restart path of
:meth:`VoiceWakeController.start` cancelled the dead listener's pumps,
waited ONE second, and deregistered them even when a pump was still
pending.  Cancellation is cooperative — the ``SendCallback`` contract
does not require an implementation to honour ``CancelledError``
immediately — so a send that finishes endpoint cleanup before yielding
to cancellation could deliver the old ``listening: false`` AFTER the
replacement's ``listening: true``, telling the client the new listener
was dead.  The same one-second abandonment existed in :meth:`stop`.

The fix retains such pumps in a per-connection ``_retiring`` set: a
successor listener defers its deliveries (its stdout pump and the
duplicate-start ``listening: true`` re-report both wait for
retirement), so in every schedule where the retired pump completes,
the stale report — which cannot be retracted once its send is in
flight — is delivered BEFORE the successor's first report.

gpt-5.6-sol round-3 review, findings 2-3, hardened the mechanism:

* ``stop()`` (and the timed-out dead-listener restart) publishes the
  retiring pumps in ``_retiring`` in the same no-await block that
  deregisters the listener, so a concurrent ``start()`` can never
  observe both empty and cross the gate early;
* the gate is BOUNDED (``_RETIREMENT_GATE_SECONDS``) so one
  never-completing retired send cannot suppress every successor's
  delivery forever; past the bound, ordering is preserved by the
  per-listener generation tag (``voiceGen``) every event carries —
  the delivery boundary (``VoiceWakeController.accepts``, applied by
  the daemon's per-connection send wrapper) discards a retired
  generation's report.

Real :class:`VoiceWakeController`, real child processes speaking the
listener's stdout protocol, a real (deliberately cancellation-
resistant) send callback; no mocks.  The bounded joins make each test
take ~6-12 s by design.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from typing import Any

from kiss.server.voice_wake_control import (
    _RETIREMENT_GATE_SECONDS,
    VoiceWakeController,
)

_EXIT_ONCE_SCRIPT = r"""
import os, sys, time
sentinel = sys.argv[1]
if not os.path.exists(sentinel):
    open(sentinel, "w").close()
    print("READY", flush=True)
    sys.exit(0)
print("READY", flush=True)
time.sleep(60)
"""
"""Exits cleanly right after READY on its first run; sleeps after."""


def _is_true(event: dict[str, Any]) -> bool:
    """Return whether *event* reports ``listening: true``."""
    return (
        event.get("type") == "voiceWakeState"
        and event.get("listening") is True
    )


def _is_false(event: dict[str, Any]) -> bool:
    """Return whether *event* reports ``listening: false``."""
    return (
        event.get("type") == "voiceWakeState"
        and event.get("listening") is False
    )


class _ResistantSender:
    """A send callback whose final report resists cancellation.

    The first ``listening: false`` report suspends until
    :attr:`release` is set and swallows every ``CancelledError``
    meanwhile — the strongest behaviour the ``SendCallback`` contract
    permits (a transport may finish endpoint cleanup before honouring
    cancellation), and exactly what the round-2 review used to defeat
    the old one-second abandonment.
    """

    def __init__(self) -> None:
        self.final_started = asyncio.Event()
        self.release = asyncio.Event()
        self.events: list[dict[str, Any]] = []

    async def send(self, event: dict[str, Any]) -> None:
        if _is_false(event):
            self.final_started.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue
        self.events.append(dict(event))

    async def assert_no_true_after(
        self, idx: int, settle: float = 0.3,
    ) -> None:
        """Assert no ``listening: true`` lands in ``events[idx:]``."""
        await asyncio.sleep(settle)
        stale = [e for e in self.events[idx:] if _is_true(e)]
        assert not stale, f"listening: true delivered while gated: {stale!r}"

    async def wait_for_true_after(
        self, idx: int, count: int = 1, timeout: float = 15.0,
    ) -> None:
        """Wait until ``events[idx:]`` holds *count* true reports."""
        deadline = asyncio.get_running_loop().time() + timeout
        while (
            len([e for e in self.events[idx:] if _is_true(e)]) < count
        ):
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(
                    f"expected {count} listening: true after index {idx}, "
                    f"got {self.events[idx:]!r}"
                )
            await asyncio.sleep(0.02)


class VoiceRetirementTest(unittest.TestCase):
    """Retirement of cancellation-resistant pumps is airtight."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _controller(self) -> VoiceWakeController:
        path = os.path.join(self.tmp, "exit_once.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_EXIT_ONCE_SCRIPT)
        sentinel = os.path.join(self.tmp, "ran-once")
        return VoiceWakeController([sys.executable, "-u", path, sentinel])

    def test_timed_out_restart_gates_the_replacement(self) -> None:
        """The review's schedule, plus the deferred duplicate re-report.

        After the timed-out restart the resistant old pump is still
        pending; the replacement must deliver nothing (neither its
        READY-derived ``listening: true`` nor a duplicate-start
        re-report) until the old pump actually completes, so the stale
        ``listening: false`` always precedes every fresh
        ``listening: true``.
        """

        async def scenario() -> None:
            controller = self._controller()
            sender = _ResistantSender()
            await controller.start("c1", None, sender.send)
            old = controller._listeners["c1"]
            await asyncio.wait_for(sender.final_started.wait(), 15)
            try:
                # Timed-out restart: 5 s join + 1 s cancellation wait.
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 20,
                )
                replacement = controller._listeners["c1"]
                self.assertIsNot(replacement, old)
                self.assertTrue(old.stopped)
                self.assertTrue(
                    any(not t.done() for t in old.pumps),
                    "the resistant old pump unexpectedly completed; the "
                    "retirement window is not exercised",
                )
                idx = len(sender.events)
                # The replacement is gated on the retired pump.
                await sender.assert_no_true_after(idx)
                # A duplicate start during retirement defers its
                # re-report instead of delivering it immediately.
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 5,
                )
                await sender.assert_no_true_after(idx, settle=0.2)
                # Un-wedge the endpoint: the stale report completes,
                # then (and only then) the replacement's reports flow —
                # the pump's READY-derived true AND the deferred
                # duplicate re-report.
                sender.release.set()
                await asyncio.wait_for(
                    asyncio.gather(*old.pumps, return_exceptions=True), 10,
                )
                await sender.wait_for_true_after(idx, count=2)
                falses = [
                    i for i, e in enumerate(sender.events) if _is_false(e)
                ]
                trues = [
                    i
                    for i, e in enumerate(sender.events[idx:], start=idx)
                    if _is_true(e)
                ]
                self.assertTrue(falses and trues)
                self.assertLess(
                    max(falses), min(trues),
                    "a stale listening: false landed after the "
                    "replacement's listening: true",
                )
                self.assertTrue(controller.running("c1"))
            finally:
                sender.release.set()
                await controller.stop_all()
                await asyncio.gather(*old.pumps, return_exceptions=True)

        asyncio.run(scenario())

    def test_stop_retires_a_resistant_pump_and_gates_the_successor(
        self,
    ) -> None:
        """``stop()``'s bounded join also retains a resistant pump.

        A fresh ``start()`` on the same connection after such a stop
        must not report before the retired pump has completed.
        """

        async def scenario() -> None:
            controller = self._controller()
            sender = _ResistantSender()
            await controller.start("c1", None, sender.send)
            old = controller._listeners["c1"]
            await asyncio.wait_for(sender.final_started.wait(), 15)
            try:
                # Bounded stop: 5 s join + 1 s cancellation wait.
                await asyncio.wait_for(controller.stop("c1"), 20)
                self.assertFalse(controller.running("c1"))
                self.assertTrue(
                    any(not t.done() for t in old.pumps),
                    "the resistant old pump unexpectedly completed; the "
                    "retirement window is not exercised",
                )
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 5,
                )
                idx = len(sender.events)
                await sender.assert_no_true_after(idx)
                sender.release.set()
                await asyncio.wait_for(
                    asyncio.gather(*old.pumps, return_exceptions=True), 10,
                )
                await sender.wait_for_true_after(idx)
                falses = [
                    i for i, e in enumerate(sender.events) if _is_false(e)
                ]
                trues = [
                    i
                    for i, e in enumerate(sender.events[idx:], start=idx)
                    if _is_true(e)
                ]
                self.assertLess(max(falses), min(trues))
            finally:
                sender.release.set()
                await controller.stop_all()
                await asyncio.gather(*old.pumps, return_exceptions=True)

        asyncio.run(scenario())

    def test_concurrent_start_during_stop_sees_the_retirement_gate(
        self,
    ) -> None:
        """Round-3 finding 2: ``start()`` racing ``stop()``'s joins.

        Pre-fix, ``stop()`` removed the listener from ``_listeners``
        BEFORE its bounded joins and only added the unfinished pumps
        to ``_retiring`` afterwards, so a ``start()`` issued in that
        window observed both empty: its replacement crossed the
        delivery gate and reported ``listening: true`` before the old
        pump's cancellation-resistant ``listening: false`` landed.
        Post-fix the retirement is published in the same no-await
        block that deregisters the listener, so the racing
        replacement defers — the stale report (released within the
        gate bound) lands strictly before the fresh
        ``listening: true``.
        """

        async def scenario() -> None:
            controller = self._controller()
            sender = _ResistantSender()
            await controller.start("c1", None, sender.send)
            old = controller._listeners["c1"]
            await asyncio.wait_for(sender.final_started.wait(), 15)
            try:
                stop_task = asyncio.ensure_future(controller.stop("c1"))
                deadline = asyncio.get_running_loop().time() + 5.0
                while controller.running("c1"):
                    if asyncio.get_running_loop().time() > deadline:
                        raise AssertionError(
                            "stop never deregistered the listener"
                        )
                    await asyncio.sleep(0)
                # The review's window: the listener is gone but the
                # stop's joins are still running.  Deregistration and
                # retirement must be atomic — the wedged pump is
                # already visible to a concurrent start().
                self.assertFalse(stop_task.done())
                self.assertTrue(
                    controller._retiring.get("c1"),
                    "stop() deregistered the listener before "
                    "publishing its unfinished pumps in _retiring",
                )
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 10,
                )
                replacement = controller._listeners["c1"]
                self.assertIsNot(replacement, old)
                idx = len(sender.events)
                # The replacement is gated on the retired pump.
                await sender.assert_no_true_after(idx)
                # Un-wedge within the gate bound: the stale report
                # lands first, then the replacement's reports flow.
                sender.release.set()
                await asyncio.wait_for(
                    asyncio.gather(*old.pumps, return_exceptions=True), 10,
                )
                await asyncio.wait_for(stop_task, 15)
                await sender.wait_for_true_after(idx)
                falses = [
                    i for i, e in enumerate(sender.events) if _is_false(e)
                ]
                trues = [
                    i
                    for i, e in enumerate(sender.events[idx:], start=idx)
                    if _is_true(e)
                ]
                self.assertTrue(falses and trues)
                self.assertLess(
                    max(falses), min(trues),
                    "a stale listening: false outran the racing "
                    "replacement's listening: true",
                )
                self.assertTrue(controller.running("c1"))
            finally:
                sender.release.set()
                await controller.stop_all()
                await asyncio.gather(*old.pumps, return_exceptions=True)

        asyncio.run(scenario())

    def test_wedged_retired_pump_cannot_suppress_successors_forever(
        self,
    ) -> None:
        """Round-3 finding 3: the retirement gate is bounded.

        The old pump's final send ignores cancellation until an
        explicit release that never comes; the ``SendCallback``
        contract permits an event-specific endpoint that would accept
        every successor send immediately.  Pre-fix the replacement's
        stdout pump waited in ``_await_retirement`` with no deadline,
        so no replacement report was ever delivered.  Post-fix the
        gate expires after ``_RETIREMENT_GATE_SECONDS`` and the
        replacement's ``listening: true`` arrives WITHOUT any release;
        the wedged stale report can then only land with a RETIRED
        generation tag, which the delivery boundary
        (:meth:`VoiceWakeController.accepts`, applied by the daemon's
        per-connection send wrapper) discards — it can neither reorder
        reports at the endpoint nor wedge successors.
        """

        async def scenario() -> None:
            controller = self._controller()
            sender = _ResistantSender()
            await controller.start("c1", None, sender.send)
            old = controller._listeners["c1"]
            await asyncio.wait_for(sender.final_started.wait(), 15)
            try:
                # Timed-out restart: 5 s join + 1 s cancellation wait.
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 20,
                )
                replacement = controller._listeners["c1"]
                self.assertIsNot(replacement, old)
                self.assertTrue(
                    any(not t.done() for t in old.pumps),
                    "the resistant old pump unexpectedly completed; "
                    "the bounded-gate window is not exercised",
                )
                idx = len(sender.events)
                # LIVENESS: no release — the replacement's report must
                # arrive on its own once the bounded gate expires.
                await sender.wait_for_true_after(
                    idx, timeout=_RETIREMENT_GATE_SECONDS + 10.0,
                )
                fresh_true = next(
                    e for e in sender.events[idx:] if _is_true(e)
                )
                self.assertTrue(
                    controller.accepts("c1", fresh_true["voiceGen"]),
                    "the replacement's report does not carry the "
                    "connection's current generation",
                )
                self.assertTrue(
                    any(not t.done() for t in old.pumps),
                    "the replacement report was not delivered past a "
                    "still-wedged retired pump",
                )
                # The stale report lands only after the release, and
                # only with a retired generation: the delivery
                # boundary discards it.
                sender.release.set()
                await asyncio.wait_for(
                    asyncio.gather(*old.pumps, return_exceptions=True), 10,
                )
                stale_falses = [
                    e for e in sender.events if _is_false(e)
                ]
                self.assertTrue(stale_falses)
                for event in stale_falses:
                    self.assertIn("voiceGen", event)
                    self.assertFalse(
                        controller.accepts("c1", event["voiceGen"]),
                        "a wedged stale report kept a live generation "
                        "and would pass the delivery boundary",
                    )
                self.assertTrue(controller.running("c1"))
            finally:
                sender.release.set()
                await controller.stop_all()
                await asyncio.gather(*old.pumps, return_exceptions=True)

        asyncio.run(scenario())

    def test_stopping_the_gated_replacement_reports_nothing(self) -> None:
        """A deferred re-report checks liveness after retirement.

        The replacement (and its deferred duplicate re-report) is
        stopped while still gated; the endpoint is released DURING the
        stop's bounded join, so retirement completes with the deferred
        task still alive — it must then observe ``stopped`` and skip
        its report.  No ``listening: true`` may ever surface for the
        stopped replacement.
        """

        async def scenario() -> None:
            controller = self._controller()
            sender = _ResistantSender()
            await controller.start("c1", None, sender.send)
            old = controller._listeners["c1"]
            await asyncio.wait_for(sender.final_started.wait(), 15)
            try:
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 20,
                )
                self.assertTrue(any(not t.done() for t in old.pumps))
                await asyncio.wait_for(
                    controller.start("c1", None, sender.send), 5,
                )
                idx = len(sender.events)
                stop_task = asyncio.ensure_future(controller.stop("c1"))
                await asyncio.sleep(0.2)
                sender.release.set()
                await asyncio.wait_for(stop_task, 20)
                self.assertFalse(controller.running("c1"))
                await asyncio.wait_for(
                    asyncio.gather(*old.pumps, return_exceptions=True), 10,
                )
                await asyncio.sleep(0.3)
                self.assertFalse(
                    [e for e in sender.events[idx:] if _is_true(e)],
                    "a stopped replacement still reported listening: true",
                )
            finally:
                sender.release.set()
                await controller.stop_all()
                await asyncio.gather(*old.pumps, return_exceptions=True)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: voice-wake generation is re-validated at the wire admission point.

gpt-5.6-sol round-4 review, finding 6: the delivery boundary of the
controller's ``voiceGen`` tag (the ``_send`` wrapper installed by
``RemoteAccessServer._handle_voice_wake_start``) validated the
generation and only THEN queued the payload on the per-endpoint FIFO
send lock (``WebPrinter._locked_send``).  A report that passed the
check could wait behind an already-held lock while a concurrent
``stop()`` retired its generation; when the lock opened, no check ran
inside it and the stale ``listening: false`` payload was written to
the wire — observable after the connection had moved on.

The fix re-validates the generation INSIDE the send lock: the wrapper
passes ``partial(controller.accepts, conn_id, gen)`` as the
``admit`` argument of ``_locked_send``, which evaluates it after
acquiring the lock and drops the payload on a stale reading.

The test drives the REAL production path — a real
:class:`VoiceWakeController` child speaking the protocol, the real
``_handle_voice_wake_start`` wrapper, the real
``WebPrinter._locked_send`` FIFO lock — against a recording endpoint,
using the reviewer's reproduction schedule: hold the endpoint's send
lock, let the child self-exit so its final report queues behind it,
retire the generation with ``stop()``, then open the lock.  No mocks.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.server.voice_wake_control import VoiceWakeController
from kiss.server.web_server import RemoteAccessServer, WebPrinter

_CONTROLLED_EXIT_SCRIPT = """\
import os, sys, time
sentinel = sys.argv[1]
print("READY", flush=True)
while not os.path.exists(sentinel):
    time.sleep(0.01)
"""
"""Emits READY, then exits cleanly once the sentinel file appears."""


class _Endpoint:
    """Endpoint recording the JSON frames admitted to the wire."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send(self, data: str) -> None:
        self.frames.append(json.loads(data))


async def _poll(predicate: Any, timeout: float = 15.0) -> None:
    """Wait until a synchronous predicate becomes true."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true in time")
        await asyncio.sleep(0.01)


class VoiceWireAdmissionTest(unittest.TestCase):
    """A retired report queued behind the send lock never hits the wire."""

    def test_retired_report_queued_before_the_lock_is_dropped(self) -> None:
        """The reviewer's schedule, asserting the FIXED outcome.

        The gen-1 final exit report passes the pre-check, queues
        behind the held endpoint lock, is retired by ``stop()``, and
        must be dropped by the in-lock re-validation once the lock
        opens — no ``listening: false`` frame ever reaches the
        endpoint, and no frame carries a ``voiceGen`` tag.
        """

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                sentinel = Path(tmp) / "exit"
                script = Path(tmp) / "listener.py"
                script.write_text(_CONTROLLED_EXIT_SCRIPT)
                controller = VoiceWakeController(
                    [sys.executable, "-u", str(script), str(sentinel)],
                )
                server = RemoteAccessServer.__new__(RemoteAccessServer)
                server._printer = WebPrinter()
                server._voice_wake = controller
                endpoint = _Endpoint()
                server._printer.add_client(endpoint)  # type: ignore[arg-type]

                await server._handle_voice_wake_start({}, endpoint, "c1")
                listener = controller._listeners["c1"]
                await _poll(lambda: len(endpoint.frames) >= 2)
                old_gen = listener.gen
                lock = server._printer.send_lock(endpoint)
                await lock.acquire()
                stop: asyncio.Task[None] | None = None
                try:
                    # The child self-exits; its final report passes the
                    # pre-lock check and queues behind the held lock.
                    sentinel.touch()
                    await _poll(
                        lambda: bool(getattr(lock, "_waiters", ())),
                    )
                    # Retire the generation while the report waits.
                    stop = asyncio.ensure_future(controller.stop("c1"))
                    await _poll(
                        lambda: not controller.accepts("c1", old_gen),
                    )
                    self.assertFalse(controller.running("c1"))
                    self.assertTrue(lock.locked())
                    lock.release()
                    await asyncio.wait_for(stop, 15)
                    stop = None
                    # Give any (wrongly) admitted late write a moment.
                    await asyncio.sleep(0.2)
                    stale = [
                        frame for frame in endpoint.frames
                        if frame.get("type") == "voiceWakeState"
                        and frame.get("listening") is False
                    ]
                    self.assertEqual(
                        stale, [],
                        "a retired-generation report reached the wire "
                        "after its retirement",
                    )
                    self.assertTrue(
                        any(
                            frame.get("listening") is True
                            for frame in endpoint.frames
                        ),
                        endpoint.frames,
                    )
                    self.assertTrue(
                        all(
                            "voiceGen" not in frame
                            for frame in endpoint.frames
                        ),
                        endpoint.frames,
                    )
                finally:
                    if lock.locked():
                        lock.release()
                    if stop is not None:
                        await asyncio.gather(stop, return_exceptions=True)
                    await controller.stop_all()
                    server._printer.remove_client(endpoint)  # type: ignore[arg-type]

        asyncio.run(scenario())

    def test_current_generation_report_still_passes_under_the_lock(
        self,
    ) -> None:
        """The admission re-check must not drop CURRENT reports.

        The same queued-behind-the-lock schedule without a retirement:
        once the lock opens, the still-current final exit report is
        admitted exactly as before the fix.
        """

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                sentinel = Path(tmp) / "exit"
                script = Path(tmp) / "listener.py"
                script.write_text(_CONTROLLED_EXIT_SCRIPT)
                controller = VoiceWakeController(
                    [sys.executable, "-u", str(script), str(sentinel)],
                )
                server = RemoteAccessServer.__new__(RemoteAccessServer)
                server._printer = WebPrinter()
                server._voice_wake = controller
                endpoint = _Endpoint()
                server._printer.add_client(endpoint)  # type: ignore[arg-type]

                await server._handle_voice_wake_start({}, endpoint, "c1")
                await _poll(lambda: len(endpoint.frames) >= 2)
                lock = server._printer.send_lock(endpoint)
                await lock.acquire()
                try:
                    sentinel.touch()
                    await _poll(
                        lambda: bool(getattr(lock, "_waiters", ())),
                    )
                finally:
                    lock.release()
                await _poll(lambda: any(
                    frame.get("type") == "voiceWakeState"
                    and frame.get("listening") is False
                    for frame in endpoint.frames
                ))
                self.assertTrue(
                    all(
                        "voiceGen" not in frame
                        for frame in endpoint.frames
                    ),
                    endpoint.frames,
                )
                await controller.stop_all()
                server._printer.remove_client(endpoint)  # type: ignore[arg-type]

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

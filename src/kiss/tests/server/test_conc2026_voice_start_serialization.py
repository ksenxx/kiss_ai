# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: two fully concurrent voice-wake starts spawn exactly one child.

Round-4 fix.  :meth:`VoiceWakeController.start` spans an await (the
``create_subprocess_exec`` spawn) while the connection's listener slot
is still empty: two fully concurrent ``start()`` calls could both pass
the duplicate-start check and both spawn a listener, the second
registration overwriting the first — the first child then had no owner
and leaked until self-exit (adjacent observation of the round-3 fix
report, ``tmp/fixes3-server.md``).

The fix serializes each connection's start/stop lifecycle with a
refcounted per-connection ``asyncio.Lock``: the second start now waits,
observes the first's registration, and takes the duplicate-start
re-report path.  Both lifecycle bodies are bounded, so the lock cannot
stall a stop (or a later start) indefinitely, and the lock/refcount
maps free their entries at zero references, so the bookkeeping stays
bounded across many short-lived connections.

Real :class:`VoiceWakeController`, real child processes that record
their own pid on disk and speak the listener's stdout protocol; no
mocks.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from typing import Any

from kiss.core.processes import pid_alive
from kiss.server.voice_wake_control import VoiceWakeController

_PID_READY_SCRIPT = r"""
import os, sys, time
piddir = sys.argv[1]
with open(os.path.join(piddir, f"{os.getpid()}-{os.getppid()}"), "w"):
    pass
print("READY", flush=True)
time.sleep(120)
"""
"""Records ``<pid>-<ppid>`` in ``piddir``, reports READY, then idles.

The parent pid is recorded too because a Windows venv ``python.exe``
is a launcher that runs the real interpreter as its child: there the
spawned ``Process.pid`` is the launcher, i.e. the script's parent.
"""


class _Collector:
    """A send callback that records every delivered event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))

    def listening_true_count(self) -> int:
        """Return how many ``listening: true`` reports were delivered."""
        return len([
            e for e in self.events
            if e.get("type") == "voiceWakeState"
            and e.get("listening") is True
        ])


class VoiceStartSerializationTest(unittest.TestCase):
    """Concurrent starts of one connection never leak a listener."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.piddir = os.path.join(self._tmp.name, "pids")
        os.mkdir(self.piddir)

    def _controller(self) -> VoiceWakeController:
        path = os.path.join(self._tmp.name, "pid_ready.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_PID_READY_SCRIPT)
        return VoiceWakeController([sys.executable, "-u", path, self.piddir])

    def _spawned(self) -> list[tuple[int, int]]:
        """Return ``(pid, ppid)`` of every child that ever ran the script."""
        return [
            (int(pid), int(ppid))
            for pid, ppid in (name.split("-") for name in os.listdir(self.piddir))
        ]

    @staticmethod
    def _owns(proc_pid: int, entry: tuple[int, int]) -> bool:
        """Whether the spawned ``Process.pid`` is *entry*'s pid or launcher."""
        return proc_pid in entry

    def test_two_concurrent_starts_spawn_exactly_one_child(self) -> None:
        """The leak schedule: both starts pass the empty-slot check.

        Unserialized, both calls cross the spawn await while
        ``_listeners`` has no entry for the connection, so two
        children run and the first is orphaned by the second's
        registration.  Serialized, the second start must find the
        first's registration and only re-report ``listening: true``.
        """

        async def scenario() -> None:
            controller = self._controller()
            sender = _Collector()
            await asyncio.wait_for(
                asyncio.gather(
                    controller.start("c1", None, sender.send),
                    controller.start("c1", None, sender.send),
                ),
                timeout=30,
            )
            self.assertTrue(controller.running("c1"))
            listener = controller._listeners["c1"]
            # Wait for the registered child to record its pid.
            deadline = asyncio.get_running_loop().time() + 15
            while not any(
                self._owns(listener.proc.pid, e) for e in self._spawned()
            ):
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError(
                        "registered listener never recorded its pid"
                    )
                await asyncio.sleep(0.02)
            # Settle so a hypothetically leaked second child (which is
            # spawned before either start() returns) surfaces too.
            await asyncio.sleep(1.0)
            spawned = self._spawned()
            live = [entry for entry in spawned if pid_alive(entry[0])]
            self.assertEqual(
                len(live), 1,
                f"expected exactly the registered child alive, got "
                f"(pid, ppid) entries {spawned!r} (live: {live!r})",
            )
            self.assertTrue(
                self._owns(listener.proc.pid, live[0]),
                f"the live child {live[0]!r} is not the registered "
                f"listener {listener.proc.pid}",
            )
            self.assertEqual(
                len(spawned), 1,
                f"a second listener child was spawned: {spawned!r}",
            )
            # The duplicate start re-reported listening: true; the
            # READY-derived report may or may not have landed yet.
            self.assertGreaterEqual(sender.listening_true_count(), 1)
            await asyncio.wait_for(controller.stop_all(), 30)
            self.assertFalse(controller.running("c1"))
            self.assertFalse(pid_alive(listener.proc.pid))
            # The interpreter behind a launcher must be gone as well.
            deadline = asyncio.get_running_loop().time() + 15
            while pid_alive(live[0][0]):
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError(
                        f"listener interpreter {live[0][0]} outlived stop_all()"
                    )
                await asyncio.sleep(0.02)
            # The refcounted lock bookkeeping freed its entries.
            self.assertEqual(controller._lifecycle_locks, {})
            self.assertEqual(controller._lifecycle_holds, {})

        asyncio.run(scenario())

    def test_cancelled_lock_waiter_leaves_no_bookkeeping(self) -> None:
        """A cancelled queued lifecycle call drops only its own ref.

        Cancelling a stop() that is still waiting for the connection's
        lifecycle lock must not release the holder's lock nor strand a
        refcount entry, and the connection must remain fully usable
        (start/stop) afterwards.
        """

        async def scenario() -> None:
            controller = self._controller()
            sender = _Collector()
            await controller._acquire_lifecycle("c1")
            waiter = asyncio.ensure_future(controller.stop("c1"))
            await asyncio.sleep(0.05)
            self.assertFalse(waiter.done())
            self.assertEqual(controller._lifecycle_holds["c1"], 2)
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            self.assertEqual(controller._lifecycle_holds["c1"], 1)
            controller._release_lifecycle("c1")
            self.assertEqual(controller._lifecycle_locks, {})
            self.assertEqual(controller._lifecycle_holds, {})
            await asyncio.wait_for(
                controller.start("c1", None, sender.send), 30,
            )
            self.assertTrue(controller.running("c1"))
            await asyncio.wait_for(controller.stop_all(), 30)
            self.assertFalse(controller.running("c1"))
            self.assertEqual(controller._lifecycle_locks, {})

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

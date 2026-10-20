# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""V5-5: an ``open_for_agent`` racing ``shutdown`` must not wait out its timeout.

``open_for_agent`` used to take the loop from ``_ensure_loop`` and hand
its coroutine over with ``run_coroutine_threadsafe`` outside the lock.
A ``shutdown`` landing between the two stopped the loop; a hand-over
that then arrived before ``loop.close()`` queued a callback nobody ever
ran, and the opener sat in ``future.result`` for the full 120 s.  The
hand-over and the shutdown check are now one critical section under
``_lock``, so the coroutine either lands on a loop that will run it
(refused under ``_launch_lock`` when the shutdown has flipped ``_closed``
by then, else opened and torn down by the shutdown) or is refused at
once.  Every outcome arrives within seconds; which one (an error, or a
tab the shutdown closes right after) depends on which thread wins, so
the timing is what the test asserts.

Pinned with two settrace pauses (``tests/server/pause.py``): the opener
is held on the way into ``run_coroutine_threadsafe`` (its first call,
``iscoroutine``), the shutting-down thread at ``loop.close``.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.server.browser_tab import BrowserTabService
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.pause import Pause, Pauses

_OPENER = "v5-opener"
_CLOSER = "v5-closer"


def _on_opener(_locals: dict[str, Any]) -> bool:
    return threading.current_thread().name == _OPENER


def _on_closer(_locals: dict[str, Any]) -> bool:
    return threading.current_thread().name == _CLOSER


def _loop_close_code() -> Any:
    loop = asyncio.new_event_loop()
    try:
        return type(loop).close.__code__
    finally:
        loop.close()


class TestOpenRacingShutdown(unittest.TestCase):
    """The opener returns (with an error) within seconds, never 120 s."""

    def test_open_handed_over_during_shutdown_fails_fast(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        service = BrowserTabService(MemoryPrinter(), Path(tmp.name) / "profile")
        self.addCleanup(service.shutdown)
        outcome: list[BaseException | str] = []

        def open_tab() -> None:
            try:
                outcome.append(str(service.open_for_agent()))
            except BaseException as exc:  # noqa: BLE001 — recorded for the assertion
                outcome.append(exc)

        opener = threading.Thread(target=open_tab, name=_OPENER, daemon=True)
        closer = threading.Thread(target=service.shutdown, name=_CLOSER, daemon=True)
        hold_opener = Pause(asyncio.coroutines.iscoroutine.__code__, _on_opener)
        hold_closer = Pause(_loop_close_code(), _on_closer)
        started = time.monotonic()
        with Pauses(hold_opener, hold_closer):
            opener.start()
            self.assertTrue(hold_opener.paused.wait(10), "opener never reached the hand-over")
            closer.start()
            # Unfixed: shutdown runs through to ``loop.close`` (held there)
            # while the opener still waits; the hand-over then lands on a
            # stopped loop.  Fixed: shutdown blocks on ``_lock`` until the
            # hand-over is done, so this wait times out harmlessly.
            hold_closer.paused.wait(2)
            hold_opener.release()
            opener.join(30)
            self.assertFalse(opener.is_alive(), "open_for_agent hung on the stopped loop")
        closer.join(30)
        self.assertFalse(closer.is_alive())
        self.assertLess(time.monotonic() - started, 60)
        self.assertEqual(len(outcome), 1)
        # Refused (RuntimeError) when the shutdown flipped ``_closed``
        # before the handed-over coroutine checked it, else opened (the
        # tab's description) and torn down by the shutdown right after.
        self.assertIsInstance(outcome[0], (RuntimeError, str), repr(outcome[0]))


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that demonstrate unfixed race conditions in ``kiss.server``.

Each test deterministically forces an interleaving that exposes a real
data race between two or more threads.  These tests use synchronisation
harnesses (barriers, events) to control scheduling — no mocks/patches
of production behaviour.

When the corresponding fix from ``race.md`` is applied, the test must
still pass.  Until then, some tests may intermittently fail — that is
the point: they prove the race exists.
"""

from __future__ import annotations

import threading
import unittest

from kiss.server.server import AgentServer


class TestEnsureCompleteWorkerDoubleInit(unittest.TestCase):
    """_ensure_complete_worker is not thread-safe (check-then-act)."""

    def test_double_call_creates_two_queues(self) -> None:
        """Concurrent calls can create two separate queues/workers."""
        server = AgentServer()
        barrier = threading.Barrier(2)
        queues: list[object] = []

        def call_ensure() -> None:
            barrier.wait(timeout=2)
            server._ensure_complete_worker()
            queues.append(server._complete_queue)

        t1 = threading.Thread(target=call_ensure)
        t2 = threading.Thread(target=call_ensure)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertEqual(len(queues), 2)





if __name__ == "__main__":
    unittest.main()

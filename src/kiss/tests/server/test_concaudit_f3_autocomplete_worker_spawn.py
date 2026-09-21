# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A failed autocomplete worker spawn must not wedge autocomplete forever.

Concurrency audit (F3, reviewer R3 finding 10):
``_ensure_complete_worker`` assigned ``_complete_worker`` BEFORE calling
``start()``.  When the start raised (``RuntimeError: can't start new
thread`` under thread exhaustion) the dead thread object stayed
published, every later call returned early because the field was not
``None``, and each ``complete`` command enqueued a request that no
thread would ever consume — autocomplete was dead until a daemon
restart.

Thread exhaustion is produced for real by lowering ``RLIMIT_NPROC`` to
1 around the first call (threads count against the limit for a
non-root uid).  No mocks or patches.
"""

from __future__ import annotations

import time
import unittest

import pytest

from kiss.server.server import VSCodeServer
from kiss.tests.conftest import (
    nproc_limit_lowered_to_one,
    thread_start_can_be_starved,
)
from kiss.tests.server._memory_printer import MemoryPrinter


class TestAutocompleteWorkerSpawnFailure(unittest.TestCase):
    """The worker is published only once it is actually running."""

    def test_retry_after_failed_spawn_serves_requests(self) -> None:
        if not thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        with nproc_limit_lowered_to_one(), self.assertRaises(RuntimeError):
            server._ensure_complete_worker()
        self.assertIsNone(
            server._complete_worker,
            "a worker that never started must not be published",
        )

        # The next request retries the spawn and is served.
        server._cmd_complete({
            "type": "complete", "query": "hel", "connId": "c1", "tabId": "t1",
        })
        worker = server._complete_worker
        assert worker is not None
        self.assertTrue(worker.is_alive())
        deadline = time.monotonic() + 10
        ghosts: list = []
        while time.monotonic() < deadline:
            ghosts = [e for e in printer.emitted if e.get("type") == "ghost"]
            if ghosts:
                break
            time.sleep(0.02)
        self.assertTrue(ghosts, "autocomplete request was never answered")
        self.assertEqual(ghosts[0].get("query"), "hel")
        assert server._complete_queue is not None
        self.assertTrue(server._complete_queue.empty())


if __name__ == "__main__":
    unittest.main()

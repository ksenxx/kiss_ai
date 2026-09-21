# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""SIGTERM must still shut the daemon down when no thread can be spawned.

Concurrency audit (F3, reviewer R3 finding 12): the SIGTERM handler set
the ``_shutdown_initiated`` latch and then started the graceful
shutdown thread unguarded.  Under thread exhaustion — precisely the
state in which an operator reaches for SIGTERM — ``Thread.start()``
raised out of the signal handler: no cleanup ran, and because the
latch was already set every later SIGTERM was logged as "cleanup
already in progress" and ignored.  The handler now falls back to
unwinding the event loop directly (``_request_loop_shutdown`` via
``call_soon_threadsafe``), so ``asyncio.run`` returns and ``start()``'s
``finally`` performs the cleanup.

The event loop is real (running in a background thread) and thread
exhaustion is produced for real by lowering ``RLIMIT_NPROC`` to 1
around the handler call.  The handler is invoked directly, exactly as
``signal.signal`` would invoke it on the main thread.  No mocks.
"""

from __future__ import annotations

import asyncio
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pytest

from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import (
    nproc_limit_lowered_to_one,
    thread_start_can_be_starved,
)


class TestSigtermFallbackWithoutThreads(unittest.TestCase):
    """A failed shutdown-thread spawn still resolves the shutdown future."""

    def test_loop_unwinds_when_shutdown_thread_cannot_start(self) -> None:
        if not thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_dir = Path(tmp.name)
        server = RemoteAccessServer(
            use_tunnel=False,
            url_file=tmp_dir / "remote-url.json",
            uds_path=tmp_dir / "kiss.sock",
        )
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        self.addCleanup(loop_thread.join, 5)
        self.addCleanup(loop.call_soon_threadsafe, loop.stop)
        server._loop = loop
        server._shutdown_future = asyncio.run_coroutine_threadsafe(
            self._make_future(), loop,
        ).result(timeout=5)

        with nproc_limit_lowered_to_one():
            # Must not raise out of the signal handler.
            server._handle_shutdown_signal(signal.SIGTERM)

        self.assertTrue(server._shutdown_initiated)
        deadline = time.monotonic() + 5
        while not server._shutdown_future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(
            server._shutdown_future.done(),
            "the serve loop was never asked to unwind after the thread "
            "spawn failed",
        )
        self.assertFalse(
            any(t.name == "kiss-sigterm-shutdown" for t in threading.enumerate()),
        )

    @staticmethod
    async def _make_future() -> asyncio.Future[None]:
        return asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    unittest.main()

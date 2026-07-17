# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: SIGINT-initiated shutdown must arm the SIGTERM guard.

``_handle_shutdown_signal`` only raises ``KeyboardInterrupt`` for a
SIGTERM when ``_shutdown_initiated`` is still False.  The flag used to
be set ONLY inside the SIGTERM branch of the handler itself — so a
shutdown that began via SIGINT / ``KeyboardInterrupt`` (Ctrl-C, or an
IDE stop button) never set it.  A first SIGTERM arriving DURING
``start()``'s ``finally`` cleanup (which can block ~12s in
``_stop_active_agent_tasks``) then raised ``KeyboardInterrupt`` inside
the cleanup, skipping ``_detach_tunnel()`` — the spawned cloudflared
later died of SIGPIPE and the public tunnel URL was lost.

The fix sets ``_shutdown_initiated = True`` as the first statement of
``start()``'s ``finally``.  This test drives the REAL ``start()`` on a
real listening server, interrupts it with a genuine SIGINT, and then
verifies (a) the flag was armed by the cleanup path and (b) a SIGTERM
delivered afterwards is absorbed instead of raising.
"""

from __future__ import annotations

import os
import signal
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kiss.server.web_server import RemoteAccessServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _sigint_when_listening(port: int, pid: int) -> None:
    """Wait for the server to accept TCP, then deliver a real SIGINT."""
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.05)
    os.kill(pid, signal.SIGINT)


class TestSigtermDuringSigintCleanup(unittest.TestCase):
    """A SIGTERM landing after a SIGINT-initiated shutdown must not raise."""

    def test_sigint_shutdown_arms_sigterm_guard(self) -> None:
        """start()'s cleanup sets _shutdown_initiated; SIGTERM only logs."""
        old_handlers = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        # When pytest itself runs as a *background job of a
        # non-job-control shell* (``sh -c 'pytest … &'`` — exactly how
        # CI wrappers and parallel test runners launch workers), POSIX
        # requires that shell to start the job with SIGINT set to
        # ``SIG_IGN``.  CPython then never installs its default
        # KeyboardInterrupt handler, so the genuine SIGINT this test
        # delivers to itself is silently discarded, ``srv.start()``
        # never unwinds, and the test hangs forever.  Restore the
        # default handler for the duration of the test so it sees the
        # same signal environment as a real terminal; the ``finally``
        # below puts the original disposition back.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        port = _free_port()
        with tempfile.TemporaryDirectory() as tmp:
            srv = RemoteAccessServer(
                host="127.0.0.1",
                port=port,
                use_tunnel=False,
                url_file=Path(tmp) / "remote-url.json",
                uds_path=Path(tmp) / "sorcar.sock",
            )
            interrupter = threading.Thread(
                target=_sigint_when_listening,
                args=(port, os.getpid()),
                daemon=True,
            )
            interrupter.start()
            try:
                srv.start()  # returns after the SIGINT unwinds cleanup
                self.assertTrue(
                    srv._shutdown_initiated,
                    "start()'s cleanup did not arm _shutdown_initiated "
                    "after a SIGINT-initiated shutdown — a SIGTERM "
                    "landing during cleanup would raise "
                    "KeyboardInterrupt and skip _detach_tunnel()",
                )
                # The production consequence: a SIGTERM delivered while
                # (or after) the cleanup runs must be absorbed.
                try:
                    srv._handle_shutdown_signal(int(signal.SIGTERM))
                except KeyboardInterrupt:
                    self.fail(
                        "SIGTERM during/after SIGINT-initiated cleanup "
                        "raised KeyboardInterrupt",
                    )
            finally:
                interrupter.join(timeout=25.0)
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)


if __name__ == "__main__":
    unittest.main()

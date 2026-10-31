# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A cancelled startup stops the ``services.d`` helpers without freezing the event loop.

``_bind_listeners`` starts the brand's ``services.d`` helpers right after
the local endpoint file is written, before the up-to-30 s wait for a
``remote_password``.  A startup cancelled during that wait rolls back in
``_setup_server``'s ``except`` branch, and stopping a helper that ignores
SIGTERM takes ``STOP_GRACE_SECONDS`` (5 s) before the SIGKILL.  That stop
used to run inline on the event loop (``_close_partial_setup``), so every
other coroutine of an embedder's loop stood still for those 5 s.

The server runs in a child process with its own empty ``KISS_HOME`` (no
password, so the startup waits) holding one helper that ignores SIGTERM.
A ticker coroutine on the same loop records how long the loop stalls
while the cancelled startup unwinds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kiss.tests.conftest import posix_only

STUBBORN = """#!/bin/sh
trap '' TERM
echo $$ > "$KISS_HOME/stubborn.tmp"
mv "$KISS_HOME/stubborn.tmp" "$KISS_HOME/stubborn.pid"
while :; do sleep 0.1; done
"""

CHILD = """
import asyncio, json, os, sys, time
from pathlib import Path
from kiss.server.web_server import RemoteAccessServer

tmp, home = Path(sys.argv[1]), Path(sys.argv[2])


async def tick(stamps):
    while True:
        stamps.append(time.monotonic())
        await asyncio.sleep(0.02)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def main():
    server = RemoteAccessServer(
        host="127.0.0.1", port=0, use_tunnel=True,
        url_file=tmp / "remote-url.json", local_endpoint_file=tmp / "sorcar-local.json",
        scheduler=False,
    )
    setup = asyncio.create_task(server.start_async())
    marker = home / "stubborn.pid"
    deadline = time.monotonic() + 60.0
    while not marker.exists():
        assert not setup.done(), "the startup ended before its services started"
        assert time.monotonic() < deadline, "the stubborn service never started"
        await asyncio.sleep(0.02)
    pid = int(marker.read_text())
    stamps = []
    ticker = asyncio.create_task(tick(stamps))
    await asyncio.sleep(0.3)
    cancelled_at = time.monotonic()
    setup.cancel()
    try:
        await setup
    except asyncio.CancelledError:
        pass
    took = time.monotonic() - cancelled_at
    await asyncio.sleep(0.1)
    ticker.cancel()
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b >= cancelled_at]
    print(json.dumps({
        "max_gap": max(gaps), "took": took, "service_alive": alive(pid),
    }), flush=True)
    os._exit(0)


asyncio.run(main())
"""


@posix_only("a services.d helper that ignores SIGTERM")
class TestCancelledStartupStopsServicesOffTheLoop(unittest.TestCase):
    """The loop keeps running while a cancelled startup stops its services."""

    def test_cancelled_startup_does_not_block_the_loop_on_services_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            services = home / "services.d"
            services.mkdir(parents=True)
            helper = services / "stubborn"
            helper.write_text(STUBBORN)
            helper.chmod(0o755)
            env = dict(os.environ, KISS_HOME=str(home), HOME=tmp)
            result = subprocess.run(
                [sys.executable, "-c", CHILD, tmp, str(home)],
                env=env, capture_output=True, text=True, timeout=120,
            )
            lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
            self.assertTrue(lines, result.stdout + result.stderr)
            report = json.loads(lines[-1])
            # The helper ignores SIGTERM: its stop waits STOP_GRACE_SECONDS
            # (5 s), so the cancelled startup takes that long to unwind...
            self.assertGreater(report["took"], 4.0, report)
            # ...but off the loop: the ticker never stalls for long.
            self.assertLess(report["max_gap"], 1.0, report)
            # And the helper is gone once the startup has unwound.
            self.assertFalse(report["service_alive"], report)


if __name__ == "__main__":
    unittest.main()

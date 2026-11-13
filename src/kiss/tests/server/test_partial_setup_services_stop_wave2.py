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

A22-4 (audit 2026-11-07): the same rollback after a LATE failure.
``_bind_listeners`` arms the watchdog, version-check, machine-registry
and deferred-restart loops and subscribes to the SEA registry, then
claims the server-reset flag; a failure there used to roll back the
listeners and services only, leaving four loops and the subscriber
running in an embedder that caught the failed start.  The claim raises
``PermissionError`` in a server subclass: Python 3.14's ``Path.is_file``
no longer re-raises the ``EACCES`` of a flag in an unreadable directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from kiss.server import agent_state, web_server
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import find_free_port

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


class _FlagClaimFailingServer(RemoteAccessServer):
    """A server whose reset-flag claim, the last step of setup, raises.

    The claim runs after every background loop is armed.  No filesystem
    state makes it raise on Python 3.14, whose ``Path.is_file`` returns
    False instead of re-raising ``EACCES``, so the failure is injected here.
    """

    def _maybe_schedule_server_reset_complete(self) -> None:
        raise PermissionError("reset flag unreadable")


class TestLateSetupFailureRollsBackTheLoops(unittest.IsolatedAsyncioTestCase):
    """A setup failing after its loops are armed cancels them and unsubscribes.

    It also stops the agent tasks: the listeners take commands from the
    moment the endpoint file is written, long before the setup ends, and
    a task started meanwhile used to keep running on a server that is
    done for (a failed ``start_async`` cannot be started again).
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-late-setup-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # A registered agent task that runs until it is told to stop.
        self.stop = threading.Event()
        state = agent_state.AgentState(
            "late-setup-task", chat_id="late-setup-chat", tab_id="late-setup-tab",
            server_owned=True, stop_event=self.stop, is_task_active=True,
        )
        state.task_thread = self.worker = threading.Thread(target=self.stop.wait, daemon=True)
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, "late-setup-task", state)
        self.addCleanup(self.stop.set)
        self.worker.start()

    def _server(self, cls: type[RemoteAccessServer], port: int) -> RemoteAccessServer:
        return cls(
            host="127.0.0.1", port=port, work_dir=str(self.tmp),
            url_file=self.tmp / "remote-url.json",
            local_endpoint_file=self.tmp / "sorcar-local.json",
        )

    async def test_start_that_never_listened_leaves_the_tasks_alone(self) -> None:
        # The task registry is the process's: a server whose port is taken
        # admitted no task, and those in it are the serving server's.
        busy = socket.socket()
        self.addCleanup(busy.close)
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        self.addCleanup(
            setattr, web_server, "_BIND_RETRY_ATTEMPTS", web_server._BIND_RETRY_ATTEMPTS,
        )
        web_server._BIND_RETRY_ATTEMPTS = 1
        server = self._server(RemoteAccessServer, busy.getsockname()[1])
        with self.assertRaises(SystemExit):
            await server.start_async()
        self.assertFalse(self.stop.is_set(), "a server that never listened stopped a task")

    async def test_failed_startup_leaves_no_loop_or_subscriber_behind(self) -> None:
        server = self._server(_FlagClaimFailingServer, find_free_port())
        with self.assertRaises(PermissionError):
            await server.start_async()
        self.assertTrue(self.stop.is_set(), "the failed start left an agent task running")
        self.worker.join(5)
        self.assertFalse(self.worker.is_alive())
        await asyncio.sleep(0.1)
        self.assertIsNone(server._ws_server)
        self.assertEqual(
            [
                server._watchdog_task, server._version_check_task,
                server._machines_watch_task, server._deferred_restart_task,
            ],
            [None] * 4,
        )
        live = sorted(
            getattr(t.get_coro(), "__qualname__", "?") for t in asyncio.all_tasks()
            if not t.done() and t is not asyncio.current_task()
        )
        self.assertEqual(live, [], "background loops survived the failed start")
        self.assertFalse(server._sea_command_subscribed, "the SEA subscriber was left behind")


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: a healthy cloudflared must survive install-triggered restarts.

``./install.sh`` makes the VS Code extension restart the kiss-web
daemon (fingerprint change).  Two timing-dependent bugs rotated the
public tunnel URL on such restarts even though the running cloudflared
was perfectly recoverable:

1. **Late detach.**  ``_detach_tunnel`` (which spawns the stderr drain
   shim keeping the detached cloudflared alive) ran only at the END of
   the shutdown path.  The extension escalates its SIGTERM to SIGKILL
   after only a few seconds; when kiss-web's cleanup (agent-task
   joins, MCP disconnects, a wedged event loop) outlived that grace,
   the shim was never spawned, cloudflared died of SIGPIPE on its next
   stderr write, and the next kiss-web logged "cloudflared pidfile
   points to dead pid" and minted a fresh URL.  The fix detaches the
   tunnel FIRST, before any slow cleanup.

2. **Shutdown-window tunnel spawn.**  A watchdog tick landing after
   the early detach but before the loop unwound saw "no tunnel" and
   would spawn a fresh cloudflared mid-shutdown — orphaning it or
   rotating the URL.  The fix makes ``_check_and_restart_tunnel`` and
   ``_restart_tunnel_url`` no-ops once ``_shutdown_initiated`` is set.

These tests use real subprocesses, real signals, real pipes, and a
real (deliberately wedged) asyncio event loop.  No mocks.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import web_server as ws
from kiss.server.web_server import RemoteAccessServer


def _pid_alive(pid: int) -> bool:
    """Return True iff *pid* is a live process."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


# Driver process: stands in for a kiss-web daemon whose event loop is
# wedged (a callback blocking the loop thread — the exact scenario the
# 30s ``_SHUTDOWN_EXIT_FAILSAFE`` exists for).  It wires a noisy
# stderr=PIPE child into ``_tunnel_proc`` (standing in for cloudflared,
# which logs to stderr continuously), installs the REAL signal
# handlers, and waits.  On SIGTERM the real ``_shutdown_on_sigterm``
# thread runs; the test then SIGKILLs the driver shortly after —
# exactly like the VS Code extension's SIGTERM->SIGKILL escalation —
# and asserts the child survived.  Without the early detach the drain
# shim is not spawned within that window (the failsafe would only
# reach ``_detach_tunnel`` after 30s), the SIGKILL closes the pipe's
# only read end, and the child dies of EPIPE on its next stderr write.
_DRIVER = r"""
import asyncio, subprocess, sys, threading, time

from kiss.server.web_server import RemoteAccessServer

child_pid_file = sys.argv[1]

server = RemoteAccessServer(use_tunnel=False)

child = subprocess.Popen(
    [
        sys.executable, "-u", "-c",
        "import sys, time\n"
        "while True:\n"
        "    sys.stderr.write('x' * 200 + '\\n')\n"
        "    sys.stderr.flush()\n"
        "    time.sleep(0.005)\n",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True,
)
with open(child_pid_file, "w") as f:
    f.write(str(child.pid))

server._tunnel_proc = child
server._tunnel_metrics_port = 19999
server._tunnel_started_at = time.monotonic()
server._active_url = "https://example.trycloudflare.com"

# A real event loop, wedged by a blocking callback: loop.is_running()
# is True (so the SIGTERM handler takes the _shutdown_on_sigterm
# thread path), but call_soon_threadsafe callbacks never run.
loop = asyncio.new_event_loop()
server._loop = loop
loop.call_soon(time.sleep, 600)
threading.Thread(target=loop.run_forever, daemon=True).start()
for _ in range(200):
    if loop.is_running():
        break
    time.sleep(0.01)

server._install_signal_handlers()
print("READY", flush=True)
while True:
    time.sleep(3600)
"""


class TestEarlyDetachOnSigterm(unittest.TestCase):
    """cloudflared survives SIGTERM->SIGKILL even with a wedged loop."""

    def test_child_survives_sigterm_then_sigkill(self) -> None:
        """The drain shim must exist BEFORE slow cleanup, not after."""
        child_pid: int | None = None
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            driver_path = Path(tmp) / "driver.py"
            driver_path.write_text(_DRIVER)

            proc = subprocess.Popen(
                [sys.executable, str(driver_path), str(pid_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
            try:
                self.assertTrue(
                    self._wait_for_line(proc, "READY", timeout=30.0),
                    "driver never reached READY",
                )
                deadline = time.monotonic() + 10.0
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                child_pid = int(pid_file.read_text().strip())
                self.assertTrue(_pid_alive(child_pid))

                # The extension's escalation, compressed: SIGTERM, a
                # short grace (far below the 30s loop-unwind failsafe),
                # then SIGKILL.
                proc.send_signal(signal.SIGTERM)
                time.sleep(2.0)
                proc.kill()
                proc.wait(timeout=10.0)

                # The child writes stderr every 5ms; without the drain
                # shim its pipe write returns EPIPE within moments of
                # the driver's death.  Give it ample time to prove it.
                time.sleep(3.0)
                self.assertTrue(
                    _pid_alive(child_pid),
                    "cloudflared stand-in died after SIGTERM->SIGKILL: "
                    "the tunnel was NOT detached before slow cleanup",
                )
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if child_pid is not None and _pid_alive(child_pid):
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass

    @staticmethod
    def _wait_for_line(
        proc: subprocess.Popen[str], needle: str, timeout: float,
    ) -> bool:
        """Block until *proc* prints a stdout line containing *needle*."""
        deadline = time.monotonic() + timeout
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    return False
                continue
            if needle in line:
                return True
        return False


class TestNoTunnelSpawnDuringShutdown(unittest.TestCase):
    """Once shutdown starts, the watchdog must never (re)start a tunnel."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._config_snapshot = (
            CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        )
        # A remote password is what normally authorizes tunnel spawns.
        save_config({"remote_password": "secret"})
        # Point PATH at an empty dir so that even a regression cannot
        # spawn a real cloudflared on the machine running the tests.
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self._tmp.name

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        if self._config_snapshot is None:
            CONFIG_PATH.unlink(missing_ok=True)
        else:
            CONFIG_PATH.write_text(self._config_snapshot)
        self._tmp.cleanup()

    def _make_server(self) -> RemoteAccessServer:
        server = RemoteAccessServer(use_tunnel=True)
        server._shutdown_initiated = True
        server._url_file = Path(self._tmp.name) / "remote-url.json"
        return server

    def test_check_and_restart_tunnel_noop_during_shutdown(self) -> None:
        """The watchdog tick is a no-op once shutdown has started."""
        server = self._make_server()

        async def run() -> None:
            server._loop = asyncio.get_running_loop()
            await server._check_and_restart_tunnel()

        asyncio.run(run())
        self.assertIsNone(server._tunnel_proc)
        self.assertFalse(
            server._url_file.exists(),
            "watchdog tick during shutdown attempted a tunnel restart "
            "(URL file was rewritten)",
        )

    def test_restart_tunnel_url_noop_during_shutdown(self) -> None:
        """The tunnel (re)starter itself refuses to run during shutdown."""
        server = self._make_server()

        async def run() -> None:
            server._loop = asyncio.get_running_loop()
            await server._restart_tunnel_url()

        asyncio.run(run())
        self.assertIsNone(server._tunnel_proc)
        self.assertFalse(
            server._url_file.exists(),
            "tunnel restart during shutdown rewrote the URL file",
        )


class TestAdoptionPreservesUrlAcrossRestart(unittest.TestCase):
    """The full pidfile round-trip preserves one URL across two 'daemons'."""

    def test_url_identical_across_two_adoptions(self) -> None:
        """Two successive adoptions of one healthy cloudflared agree."""
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                if self.path.startswith("/ready"):
                    body = json.dumps({"readyConnections": 2})
                elif self.path.startswith("/quicktunnel"):
                    body = json.dumps(
                        {"hostname": "stable.trycloudflare.com"},
                    )
                else:
                    body = ""
                data = body.encode()
                self.send_response(200 if body else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:
                pass

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]

        old_pidfile = ws._CLOUDFLARED_PIDFILE
        tmp = tempfile.TemporaryDirectory()
        ws._CLOUDFLARED_PIDFILE = Path(tmp.name) / "cloudflared.pid"
        link = Path(tmp.name) / "cloudflared"
        os.symlink(sys.executable, link)
        proc = subprocess.Popen(
            [str(link), "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            ws._save_cloudflared_pidfile(proc.pid, port, None)

            first = ws._try_adopt_existing_cloudflared()
            self.assertIsNotNone(first)
            pid1, port1, url1 = first  # type: ignore[misc]
            # Mirror what _setup_server does after a successful adoption.
            ws._save_cloudflared_pidfile(pid1, port1, url1)

            second = ws._try_adopt_existing_cloudflared()
            self.assertEqual(
                second, first,
                "a second kiss-web adoption changed the public URL",
            )
            self.assertEqual(url1, "https://stable.trycloudflare.com")
            self.assertIsNone(proc.poll())
        finally:
            ws._CLOUDFLARED_PIDFILE = old_pidfile
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            httpd.shutdown()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()

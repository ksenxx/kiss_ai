# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``cloudflared`` must survive a ``kiss-web`` shutdown.

The spawned ``cloudflared`` subprocess is launched with
``start_new_session=True`` and its pid + metrics port are persisted to
``~/.kiss/cloudflared.pid`` so the next ``kiss-web`` instance can
re-adopt it via :func:`_try_adopt_existing_cloudflared`.  That adoption
is what preserves the public ``*.trycloudflare.com`` (or named-tunnel)
URL across ``kiss-web`` restarts: the installer's ``pkill kiss-web`` /
launchd respawn / "Server reset" must not mint a brand-new hostname
every time.

For this design to work, the ``kiss-web`` shutdown path must NOT
terminate the spawned ``cloudflared``: it should reset only its own
in-memory bookkeeping and let the detached child keep running.  These
tests enforce that property in two layers:

* A focused unit test on :meth:`RemoteAccessServer._detach_tunnel`
  that wires a real long-running child subprocess into
  ``_tunnel_proc`` and asserts the child is still alive afterwards.
* An end-to-end driver that goes through the full ``start()`` cleanup
  ``finally`` block (delivered via SIGTERM exactly like ``pkill
  kiss-web``) and verifies the child still lives once the driver
  exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.server.web_server as ws_mod
from kiss.core.processes import pid_alive
from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer, _save_url_file
from kiss.tests.conftest import posix_only
from kiss.tests.server.pause import Pause


class TestDetachTunnelLeavesProcessAlive(unittest.TestCase):
    """``_detach_tunnel`` resets state without killing the child."""

    def test_spawned_proc_keeps_running(self) -> None:
        """A child wired into ``_tunnel_proc`` survives ``_detach_tunnel``."""
        server = RemoteAccessServer(use_tunnel=False)
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            server._tunnel_proc = proc  # type: ignore[assignment]
            server._tunnel_metrics_port = 12345
            server._tunnel_started_at = time.monotonic()
            server._tunnel_failure_count = 7
            server._active_url = "https://example.trycloudflare.com"

            server._detach_tunnel()

            self.assertIsNone(server._tunnel_proc)
            self.assertIsNone(server._tunnel_metrics_port)
            self.assertIsNone(server._tunnel_started_at)
            self.assertEqual(server._tunnel_failure_count, 0)
            self.assertIsNone(server._active_url)

            time.sleep(0.2)
            self.assertIsNone(proc.poll())
            self.assertTrue(pid_alive(proc.pid))
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait()

    def test_adopted_pid_keeps_running(self) -> None:
        """An adopted cloudflared survives ``_detach_tunnel``."""
        server = RemoteAccessServer(use_tunnel=False)
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            server._tunnel_adopted_pid = proc.pid
            server._tunnel_metrics_port = 12346
            server._tunnel_started_at = time.monotonic()
            server._active_url = "https://example.trycloudflare.com"

            server._detach_tunnel()

            self.assertIsNone(server._tunnel_adopted_pid)
            self.assertIsNone(server._tunnel_metrics_port)
            self.assertIsNone(server._active_url)

            time.sleep(0.2)
            self.assertTrue(pid_alive(proc.pid))
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait()


_DRIVER = r"""
import os, signal, sys, time
from pathlib import Path

from kiss.server.web_server import RemoteAccessServer

cloudflared_pid_file = sys.argv[1]

server = RemoteAccessServer(use_tunnel=False)

# Stand in for cloudflared: a detached child that ignores SIGTERM (so
# we can prove kiss-web's shutdown is the ONLY thing that could kill
# it, not an accidental signal propagation).
import subprocess
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(120)",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
Path(cloudflared_pid_file).write_text(str(child.pid))
server._tunnel_proc = child
server._tunnel_metrics_port = 19999
server._tunnel_started_at = time.monotonic()
server._active_url = "https://example.trycloudflare.com"

server._install_signal_handlers()

pid = os.getpid()
try:
    # READY is printed *inside* the try: the test sends SIGTERM as soon as
    # it reads this line, and under load the signal can land while this
    # process is still returning from print().  Outside the try that
    # KeyboardInterrupt would escape as an unclean exit (rc=-2).
    print("READY", flush=True)
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("SIGTERM_CAUGHT", flush=True)
finally:
    # Mirror the production shutdown path in RemoteAccessServer.start.
    server._stop_active_agent_tasks()
    server._detach_tunnel()
print("CLEAN_EXIT", flush=True)
sys.exit(0)
"""


@posix_only("SIGTERM delivery and start_new_session detachment")
class TestCloudflaredSurvivesKissWebShutdown(unittest.TestCase):
    """A SIGTERM to ``kiss-web`` leaves the spawned ``cloudflared`` alive."""

    def test_cloudflared_child_still_alive_after_sigterm(self) -> None:
        """End-to-end: ``cloudflared`` PID is alive after the driver exits."""
        child_pid: int | None = None
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            driver_path = Path(tmp) / "driver.py"
            driver_path.write_text(_DRIVER)

            proc = subprocess.Popen(
                [sys.executable, str(driver_path), str(pid_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                ready = self._wait_for_line(proc, "READY", timeout=30.0)
                self.assertTrue(ready, "driver never reached READY")

                deadline = time.monotonic() + 10.0
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists())
                child_pid = int(pid_file.read_text().strip())
                self.assertTrue(pid_alive(child_pid))

                proc.send_signal(signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=30.0)

                self.assertEqual(
                    proc.returncode, 0,
                    f"driver exited uncleanly (rc={proc.returncode})\n"
                    f"stdout={stdout!r}\nstderr={stderr!r}",
                )
                self.assertIn("CLEAN_EXIT", stdout)

                self.assertTrue(
                    pid_alive(child_pid),
                    "cloudflared stand-in was killed by kiss-web shutdown",
                )
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if child_pid is not None and pid_alive(child_pid):
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


_FAKE_CLOUDFLARED = (
    f"#!{sys.executable}\n"
    "import sys, time\n"
    "sys.stderr.write('INF Registered tunnel connection\\n')\n"
    "sys.stderr.flush()\n"
    "time.sleep(300)\n"
)


class TestCancelledAsyncStartupStopsItsTunnel(IsolatedAsyncioTestCase):
    """The embedder path is the opposite of ``start()``: a cancelled ``start_async``
    must not leave the ``cloudflared`` it spawned running with no listener and no
    watchdog, since nothing will ever adopt or supervise it.  A real named-tunnel
    stand-in on ``PATH``; the startup is held inside the real URL-file write
    (:class:`Pause`) so the cancel lands after the spawn.
    """

    async def asyncSetUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-cancel-start-"))
        fake = self.tmp / "cloudflared"
        fake.write_text(_FAKE_CLOUDFLARED)
        fake.chmod(0o755)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.tmp}{os.pathsep}{self._old_path}"
        self._orig_pidfile = ws_mod._CLOUDFLARED_PIDFILE
        ws_mod._CLOUDFLARED_PIDFILE = self.tmp / "cloudflared.pid"
        self._orig_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None
        save_config({"remote_password": "cancel-start-only"})

    async def asyncTearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        ws_mod._CLOUDFLARED_PIDFILE = self._orig_pidfile
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        else:
            CONFIG_PATH.unlink(missing_ok=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    @posix_only("the fake cloudflared is a script")
    async def test_cancelled_start_async_terminates_the_spawned_tunnel(self) -> None:
        url_file = self.tmp / "remote-url.json"
        server = RemoteAccessServer(
            host="127.0.0.1", port=0, use_tunnel=True, scheduler=False,
            tunnel_token="fake-token", tunnel_url="https://cancel.example.invalid",
            work_dir=str(self.tmp), url_file=url_file,
            local_endpoint_file=self.tmp / "local.json",
        )
        proc = None
        try:
            with Pause(_save_url_file.__code__, lambda loc: loc["url_file"] == url_file) as pause:
                setup = asyncio.ensure_future(server.start_async())
                self.assertTrue(await asyncio.to_thread(pause.paused.wait, 60))
                proc = server._tunnel_proc
                assert proc is not None
                self.assertIsNone(proc.poll(), "the stand-in tunnel is up before the cancel")
                setup.cancel()
                # The rollback waits for the held write to finish.
                pause.release()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(setup, 30)
            self.assertIsNone(server._ws_server)
            self.assertIsNotNone(
                await asyncio.to_thread(_wait_exit, proc, 5),
                "the cancelled startup leaked its spawned cloudflared",
            )
            self.assertTrue(server._tunnel_stopped, "a late spawn would install its child")
            self.assertFalse(url_file.exists(), "the URL file advertises a closed listener")
        finally:
            await server.stop_async()
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    async def test_cancelled_start_async_cannot_publish_the_url_file_after_its_rollback(
        self,
    ) -> None:
        """A URL writer the startup queued but that is not at its lock yet when the cancel
        lands finds publication closed afterwards: the rollback's removal found no file,
        and the writer must not advertise the closed listener once released.
        """
        url_file = self.tmp / "remote-url.json"
        server = RemoteAccessServer(
            host="127.0.0.1", port=0, scheduler=False, work_dir=str(self.tmp),
            url_file=url_file, local_endpoint_file=self.tmp / "local.json",
        )
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        try:
            with Pause(
                RemoteAccessServer._write_url_file_sync.__code__,
                lambda loc: loc["self"] is server,
            ) as pause:
                setup = asyncio.ensure_future(server.start_async())
                self.assertTrue(await asyncio.to_thread(pause.paused.wait, 60))
                setup.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(setup, 30)
                self.assertIsNone(server._ws_server)
                self.assertFalse(url_file.exists())
                # The other worker is kept busy, so the sentinel queued
                # after the release runs on the writer's worker: when it
                # has run, the writer is done.
                gate = threading.Event()
                busy = loop.run_in_executor(None, gate.wait)
                pause.release()
                await loop.run_in_executor(None, _nothing)
                gate.set()
                await busy
            self.assertFalse(url_file.exists(), "the released writer published the closed listener")
        finally:
            await server.stop_async()


def _nothing() -> None:
    """Executor sentinel: runs once a worker is free."""


def _wait_exit(proc: subprocess.Popen[str], timeout: float) -> int | None:
    """Return *proc*'s exit status, ``None`` when it is still running after *timeout* s."""
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


class TestStderrDrainShim(unittest.TestCase):
    """``_spawn_stderr_drain_shim`` keeps the stderr pipe open after exit.

    The whole point of the drain shim is to prevent ``cloudflared``
    from receiving ``SIGPIPE`` on its next stderr write once
    ``kiss-web`` has exited.  These tests exercise the shim against a
    real long-running child whose stderr is a real pipe.
    """

    @posix_only("the SIGPIPE drain shim is a detached cat; Windows has no SIGPIPE")
    def test_drain_shim_keeps_proc_alive_under_stderr_load(self) -> None:
        """A noisy child stays alive after its parent reader is closed.

        Stands in for ``cloudflared``: a Python child that writes a
        line to stderr every 50 ms (well over the 64 KiB pipe buffer
        in a few seconds).  Without the drain shim, this would either
        block on write (if the parent's reader stops draining) or
        crash with ``SIGPIPE`` (once the parent's read end is closed
        and the child writes again).  With the drain shim, neither
        happens and the child stays alive.
        """
        shim: subprocess.Popen[bytes] | None = None
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
        try:
            shim = RemoteAccessServer._spawn_stderr_drain_shim(child)
            self.assertIsNotNone(shim)

            assert child.stderr is not None
            child.stderr.close()

            time.sleep(3.0)

            self.assertIsNone(child.poll(), "child died despite drain shim")
            assert shim is not None
            self.assertIsNone(shim.poll(), "drain shim exited prematurely")
        finally:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            if shim is not None:
                try:
                    shim.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    shim.kill()
                    shim.wait()

    def test_drain_shim_none_when_no_stderr(self) -> None:
        """Returns ``None`` (no-op) when the proc has no stderr pipe."""
        proc: subprocess.Popen[str] = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            stdout=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            shim = RemoteAccessServer._spawn_stderr_drain_shim(proc)
            self.assertIsNone(shim)
        finally:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    unittest.main()

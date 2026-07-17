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
URL across ``kiss-web`` restarts: VS Code's ``pkill kiss-web`` /
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

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from kiss.server.web_server import RemoteAccessServer


def _child_is_alive(pid: int) -> bool:
    """Return True iff *pid* is a live process."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


class TestDetachTunnelLeavesProcessAlive(unittest.TestCase):
    """``_detach_tunnel`` resets state without killing the child."""

    def test_spawned_proc_keeps_running(self) -> None:
        """A child wired into ``_tunnel_proc`` survives ``_detach_tunnel``."""
        server = RemoteAccessServer(use_tunnel=False)
        proc = subprocess.Popen(
            ["sleep", "30"],
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

            # State has been cleared so the next adoption attempt
            # starts from a clean slate.
            self.assertIsNone(server._tunnel_proc)
            self.assertIsNone(server._tunnel_metrics_port)
            self.assertIsNone(server._tunnel_started_at)
            self.assertEqual(server._tunnel_failure_count, 0)
            self.assertIsNone(server._active_url)

            # Critical property: the spawned cloudflared (here a
            # stand-in ``sleep``) is still running.  Without this,
            # every kiss-web restart would mint a new public URL.
            time.sleep(0.2)
            self.assertIsNone(proc.poll())
            self.assertTrue(_child_is_alive(proc.pid))
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
            ["sleep", "30"],
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
            self.assertTrue(_child_is_alive(proc.pid))
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait()


# ---------------------------------------------------------------------------
# End-to-end: ``kiss-web`` SIGTERM finally does not kill cloudflared
# ---------------------------------------------------------------------------

# A driver subprocess that mirrors the exact shutdown structure of
# :meth:`RemoteAccessServer.start` -- ``try / except KeyboardInterrupt /
# finally`` with the same cleanup calls -- and wires a long-running
# ``sleep`` child in as the "spawned cloudflared".  The parent sends
# SIGTERM, the driver unwinds through the finally, and the parent then
# verifies the child is still alive.
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

print("READY", flush=True)
pid = os.getpid()
try:
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
                # New session so our SIGTERM hits only the driver, not
                # the pytest runner that spawned us.
                start_new_session=True,
            )
            try:
                ready = self._wait_for_line(proc, "READY", timeout=30.0)
                self.assertTrue(ready, "driver never reached READY")

                # Read the cloudflared stand-in pid the driver recorded.
                deadline = time.monotonic() + 10.0
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_file.exists())
                child_pid = int(pid_file.read_text().strip())
                self.assertTrue(_child_is_alive(child_pid))

                # Deliver the same signal VS Code's ``pkill kiss-web``
                # delivers.  The driver's signal handler raises
                # KeyboardInterrupt, the finally runs, and the driver
                # exits 0 -- WITHOUT killing the cloudflared stand-in.
                proc.send_signal(signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=30.0)

                self.assertEqual(
                    proc.returncode, 0,
                    f"driver exited uncleanly (rc={proc.returncode})\n"
                    f"stdout={stdout!r}\nstderr={stderr!r}",
                )
                self.assertIn("CLEAN_EXIT", stdout)

                # The point of the whole exercise: the spawned
                # cloudflared stand-in must still be running after the
                # kiss-web driver has fully exited.  Pre-fix this
                # assertion failed because ``_stop_tunnel`` called
                # ``proc.terminate()`` on the spawned child as part of
                # the shutdown ``finally``.
                self.assertTrue(
                    _child_is_alive(child_pid),
                    "cloudflared stand-in was killed by kiss-web shutdown",
                )
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if child_pid is not None and _child_is_alive(child_pid):
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


class TestStderrDrainShim(unittest.TestCase):
    """``_spawn_stderr_drain_shim`` keeps the stderr pipe open after exit.

    The whole point of the drain shim is to prevent ``cloudflared``
    from receiving ``SIGPIPE`` on its next stderr write once
    ``kiss-web`` has exited.  These tests exercise the shim against a
    real long-running child whose stderr is a real pipe.
    """

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
            # Hand off the stderr pipe to the drain shim.
            shim = RemoteAccessServer._spawn_stderr_drain_shim(child)
            self.assertIsNotNone(shim)

            # Simulate kiss-web closing its end: drop the parent's
            # reference to the pipe's read end.  The shim still
            # holds a dup'd copy of the fd, so the kernel keeps the
            # read end open and the child does not get SIGPIPE.
            assert child.stderr is not None
            child.stderr.close()

            # Give the child plenty of time to overflow the pipe
            # buffer.  Without the shim, ~64 KiB / 200 B per write =
            # ~320 writes ≈ 1.6 s and it would either block or die.
            time.sleep(3.0)

            # The child must still be running.
            self.assertIsNone(child.poll(), "child died despite drain shim")
            # And the shim must still be running, draining bytes.
            assert shim is not None
            self.assertIsNone(shim.poll(), "drain shim exited prematurely")
        finally:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            # The shim should now exit cleanly on EOF.
            if shim is not None:
                try:
                    shim.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    shim.kill()
                    shim.wait()

    def test_drain_shim_none_when_no_stderr(self) -> None:
        """Returns ``None`` (no-op) when the proc has no stderr pipe."""
        proc: subprocess.Popen[str] = subprocess.Popen(
            ["sleep", "5"],
            stdout=subprocess.DEVNULL,
            # Note: NO stderr=PIPE -- this proc has no captured stderr.
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

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20, finding C2-02: the SIGTERM failsafe stops the daemon's children.

When the event loop does not unwind within ``_SHUTDOWN_EXIT_FAILSAFE``
seconds of SIGTERM, :meth:`RemoteAccessServer._shutdown_on_sigterm`
force-exits the process.  ``start()``'s ``finally`` never runs then, and
the failsafe used to skip :meth:`RemoteAccessServer._shutdown_children`
with it: terminal shells were not hung up, the streamed Browser tab and
the remote tabs not closed, the tab registry not flushed.

A real daemon in a child process, with a loop task that swallows its
cancellation (the loop never unwinds) and a terminal tab whose shell
ignores SIGHUP: the pty closing with the process does not end it, only
the hang-up-then-kill of ``terminals.shutdown()`` does.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import find_free_port

_CHILD_SCRIPT = r"""
import asyncio
import os
import sys
import threading
import time
from pathlib import Path

tmp = Path(sys.argv[1])
os.environ.setdefault("KISS_WORKDIR", str(tmp))

import kiss.agents.sorcar.persistence as th

kiss_dir = tmp / ".kiss"
kiss_dir.mkdir(parents=True, exist_ok=True)
th._KISS_DIR = kiss_dir
th._DB_PATH = kiss_dir / "history.db"
th._db_conn = None

import kiss.server.web_server as ws
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert

ws._SHUTDOWN_EXIT_FAILSAFE = 2.0
certfile, keyfile = tmp / "cert.pem", tmp / "key.pem"
_generate_self_signed_cert(certfile, keyfile)
server = RemoteAccessServer(
    host="127.0.0.1",
    port=int(sys.argv[2]),
    certfile=str(certfile),
    keyfile=str(keyfile),
    use_tunnel=False,
    url_file=tmp / "remote-url.json",
    local_endpoint_file=tmp / "sorcar-local.json",
)


wedged = threading.Event()


async def swallow_everything_forever():
    wedged.set()
    while True:
        try:
            await asyncio.sleep(0)
        except BaseException:
            pass


def wedge_the_loop_and_open_a_terminal():
    while server._loop is None:
        time.sleep(0.05)
    server._loop.call_soon_threadsafe(
        lambda: asyncio.ensure_future(swallow_everything_forever()),
    )
    terminals = server._agent_server.terminals
    terminals.open("hup-tab", "conn-1", str(tmp), 80, 24)
    # The pid file is the test's cue to send SIGTERM: publish it only once
    # the loop is wedged (a SIGTERM before that unwinds the loop normally
    # and never reaches the failsafe), and atomically: this thread shares
    # the GIL with the wedged loop, so a plain write_text leaves the file
    # empty for long enough to be read as such.
    wedged.wait()
    partial = tmp / "shell_pid.txt.partial"
    partial.write_text(str(terminals._sessions["hup-tab"].pid))
    os.replace(partial, tmp / "shell_pid.txt")


threading.Thread(target=wedge_the_loop_and_open_a_terminal, daemon=True).start()
server.start()
"""

_HUP_PROOF_SHELL = "#!/bin/sh\ntrap '' HUP\nexec sleep 300\n"


def _alive(pid: int) -> bool:
    """Whether *pid* is a live process (a zombie counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state != "Z"


@posix_only("the daemon's terminal tabs need a pty")
class TestSigtermFailsafeStopsChildren(unittest.TestCase):
    """The force-exit path hangs up the terminal shells first."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-failsafe-e2e-"))
        self.child_script = self.tmpdir / "child_daemon.py"
        self.child_script.write_text(_CHILD_SCRIPT, encoding="utf-8")
        self.shell = self.tmpdir / "hupshell"
        self.shell.write_text(_HUP_PROOF_SHELL)
        self.shell.chmod(0o755)
        self.child_log = self.tmpdir / "child.log"
        self.proc: subprocess.Popen[bytes] | None = None
        self.shell_pid = 0

    def tearDown(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        if self.shell_pid and _alive(self.shell_pid):
            os.kill(self.shell_pid, signal.SIGKILL)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wait_for(self, cond, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return
            time.sleep(0.1)
        log = self.child_log.read_text(errors="replace") if self.child_log.exists() else ""
        raise AssertionError(f"timed out waiting for {what}\nchild log:\n{log}")

    @pytest.mark.slow
    def test_failsafe_exit_hangs_up_the_terminal_shells(self) -> None:
        with self.child_log.open("wb") as log_file:
            self.proc = subprocess.Popen(
                [sys.executable, str(self.child_script), str(self.tmpdir), str(find_free_port())],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(self.tmpdir),
                env={**os.environ, "PYTHONUNBUFFERED": "1", "SHELL": str(self.shell)},
            )
        pid_file = self.tmpdir / "shell_pid.txt"
        self._wait_for((self.tmpdir / "sorcar-local.json").exists, 30.0, "the endpoint file")
        self._wait_for(pid_file.exists, 30.0, "the terminal shell")
        self.shell_pid = int(pid_file.read_text())
        self.assertTrue(_alive(self.shell_pid))

        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            self.fail("the daemon did not exit after SIGTERM\n" + self.child_log.read_text())
        log = self.child_log.read_text(errors="replace")
        self.assertIn("Shutdown failsafe", log, "the loop unwound: the failsafe was not exercised")
        # The pty closed with the daemon: a shell that ignores SIGHUP
        # survives that, so only ``terminals.shutdown()``'s hang-up-then-kill
        # can have ended it.
        self._wait_for(lambda: not _alive(self.shell_pid), 5.0, "the shell to be killed")


if __name__ == "__main__":
    unittest.main()

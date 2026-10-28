# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20, slice RF5: ending the sign-in hook, and a late ``status`` answer.

Review R5 of the sign-in window (:mod:`kiss.server.signin`):

1. ``cancel`` closed the hook's buffered stdin under the sign-in lock
   before terminating the hook.  A paste the hook never read fills the
   pipe and leaves its writer blocked on the stream's buffer lock, which
   ``close()`` needs too: cancel, a replacing click and the daemon's
   shutdown hung behind it.  Now the pipe is non-blocking (a paste that
   does not fit is refused, never parked), so no writer ever holds the
   stream: cancel closes it and terminates the hook at once, and a
   thread kills a hook that ignores SIGTERM.
2. A ``status`` answer computed before a sign-in succeeded could be
   published after its ``signinDone``, reopening the window.  Now the
   answer is handed to the page under the status lock, in one step with
   the cache drop and the ``signinDone`` of a success.
3. The SIGKILL fallback of a hook that ignores SIGTERM was a daemon
   timer the interpreter's exit abandons.  Now the daemon's shutdown
   waits, bounded, for the ended hook to be gone.

The daemon is real; the hooks are shell stubs (the shared one of
:mod:`kiss.tests.server.test_signin_window`, and one whose ``status``
snapshots its answer before a delay and whose ``login`` records the
sign-in, as a real hook does).
"""

from __future__ import annotations

import asyncio
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kiss.agents.sorcar._concurrency import locked_call
from kiss.server import signin
from kiss.server.signin import SIGNIN_HOOK
from kiss.tests.conftest import posix_only
from kiss.tests.server.test_audit1020_c12_sigterm_failsafe import _alive
from kiss.tests.server.test_signin_window import HOOK, NOT_LOGGED_IN, SigninHookCase

if sys.platform != "win32":
    import fcntl

# Bytes a pipe holds before a write blocks.  Linux reports its size;
# a macOS pipe grows under pressure up to this same 64 KiB.
_PIPE_CAPACITY = 65536

SNAPSHOT_HOOK = """#!/bin/bash
case "$1" in
  status)
    status=$(cat "$KISS_HOME/status.json")
    touch "$KISS_HOME/status-started"
    [ -f "$KISS_HOME/status-sleep" ] && sleep "$(cat "$KISS_HOME/status-sleep")"
    echo "$status"
    ;;
  login)
    echo '{"url": "https://accounts.example/auth"}'
    read -r line || exit 2
    echo '{"logged_in": true, "email": "me@example.com"}' > "$KISS_HOME/status.json"
    touch "$KISS_HOME/pasted"
    echo '{"logged_in": true, "email": "me@example.com"}'
    ;;
esac
"""


def _writable(fd: int) -> bool:
    """Whether the pipe of *fd* has room; False once it is full and a write blocks."""
    return bool(select.select([], [fd], [], 0)[1])


def _pending_send_count(pending: dict[object, set[object]]) -> int:
    """Sends still in flight across every endpoint of a printer's ``_pending_sends``."""
    return sum(len(futures) for futures in pending.values())


def _pipe_capacity(fd: int) -> int:
    """Bytes the pipe of *fd* takes before a write blocks."""
    get_size = getattr(fcntl, "F_GETPIPE_SZ", None)  # Linux only
    if get_size is None:
        return _PIPE_CAPACITY
    return int(fcntl.fcntl(fd, get_size))


class TestCancelBehindAFullPipe(SigninHookCase):
    """Issue 1: pastes the hook never reads cannot hold up cancel, replace or shutdown."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        saved = signin.HOOK_END_TIMEOUT
        signin.HOOK_END_TIMEOUT = 0.3
        self.addCleanup(setattr, signin, "HOOK_END_TIMEOUT", saved)

    def _sends_in_flight(self) -> int:
        """How many replies the printer has scheduled but not yet written."""
        printer = self.server._printer
        return locked_call(printer._ws_lock, _pending_send_count, printer._pending_sends)

    async def _landed_signin_events(self, reader) -> list[dict[str, object]]:
        """The sign-in events every command sent so far on *reader* produced.

        A paste's refusal is a broadcast scheduled on the loop, while
        the ``pong`` answering a ping is sent inline, so a pong can
        overtake a refusal queued before the ping and the next batch
        would then see two.  The pong does guarantee the paste was
        handled and its refusal, if any, is among the printer's pending
        sends; once those have all been written, a second ping's batch
        holds whatever the first one missed.
        """
        landed = await self._pending_signin_events(reader)
        for _ in range(500):
            if not self._sends_in_flight():
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self._sends_in_flight(), 0, "a queued reply never landed")
        return landed + await self._pending_signin_events(reader)

    async def _fill_the_hooks_pipe(self, reader) -> subprocess.Popen[str]:
        """Start the ``stubborn`` hook (never reads stdin) and paste until its pipe is full.

        The paste that no longer fits is refused at once, as a
        ``signinError``: the pipe is non-blocking, so no paste is ever
        parked in it.  One paste is sent per batch and the batch holds
        every refusal produced so far, so the first non-empty batch
        holds exactly the refusal of the paste that did not fit.
        """
        await self._send(reader, {"type": "signinLogin", "provider": "stubborn"})
        await self._wait_for_event(reader, "signinUrl")
        proc = self.server._server_api.signin._proc
        assert proc is not None and proc.stdin is not None
        fd = proc.stdin.fileno()
        for _ in range(_pipe_capacity(fd) // signin.PASTE_LIMIT + 2):
            await self._send(reader, {"type": "signinPaste", "text": "x" * signin.PASTE_LIMIT})
            refused = await self._landed_signin_events(reader)
            if refused:
                break
        else:
            self.fail("no paste was refused")
        self.assertEqual([e["type"] for e in refused], ["signinError"])
        self.assertIn("No sign-in is in progress", str(refused[0]["error"]))
        self.assertFalse(_writable(fd), "the pipe is not full")
        return proc

    async def test_cancel_from_another_window_is_answered_and_the_hook_dies(self) -> None:
        self._install_hook()
        reader_a, _ = await self._ready("tab-a")
        reader_b, _ = await self._ready("tab-b")
        proc = await self._fill_the_hooks_pipe(reader_a)
        # The hook's pipe is full; B's cancel must not wait for the hook to drain it.
        started = time.monotonic()
        await self._send(reader_b, {"type": "signinCancel"})
        await self._send(reader_b, {"type": "ping"})
        await self._wait_for_event(reader_b, "pong", timeout=3.0)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertIsNone(self.server._server_api.signin._proc)
        # SIGTERM is ignored; the fallback kill ends the hook.
        self.assertEqual(await asyncio.to_thread(proc.wait, 5), -9)
        await self._send(reader_a, {"type": "signinPaste", "text": "late"})
        error = await self._wait_for_event(reader_a, "signinError")
        self.assertIn("No sign-in is in progress", str(error["error"]))
        for _ in range(500):
            if not self.server._server_api.signin._ending:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.server._server_api.signin._ending, [])
        self.assertTrue(proc.stdin is not None and proc.stdin.closed)

    async def test_a_replacing_click_is_answered_while_the_pipe_is_full(self) -> None:
        self._install_hook()
        reader_a, _ = await self._ready("tab-a")
        reader_b, _ = await self._ready("tab-b")
        proc = await self._fill_the_hooks_pipe(reader_a)
        await self._send(reader_b, {"type": "signinLogin", "provider": "google"})
        url = await self._wait_for_event(reader_b, "signinUrl", timeout=3.0)
        self.assertEqual(url["provider"], "google")
        self.assertEqual(await asyncio.to_thread(proc.wait, 5), -9)
        await self._send(reader_b, {"type": "signinPaste", "text": "ok"})
        done = await self._wait_for_event(reader_b, "signinDone")
        self.assertEqual(done["email"], "me@example.com")

    async def test_shutdown_is_bounded_while_the_pipe_is_full(self) -> None:
        self._install_hook()
        reader_a, _ = await self._ready("tab-a")
        proc = await self._fill_the_hooks_pipe(reader_a)
        started = time.monotonic()
        await self.server._server_api.cancel_background_replies()
        await asyncio.to_thread(self.server._server_api.signin.shutdown)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(proc.poll(), -9)
        self.assertEqual(self.server._server_api.signin._ending, [])


class TestLateStatusAnswer(SigninHookCase):
    """Issue 2: a ``status`` answer from before a sign-in never follows its ``signinDone``."""

    async def test_a_status_answer_in_flight_during_the_sign_in_does_not_reopen_the_window(
        self,
    ) -> None:
        hook = self._install_hook()
        hook.write_text(SNAPSHOT_HOOK)
        reader_a, events = await self._ready("tab-a")
        self.assertEqual(len(self._signin_events(events)), 1)
        await self._send(reader_a, {"type": "signinLogin", "provider": "google"})
        await self._wait_for_event(reader_a, "signinUrl")
        # B's ready starts a status query that has read "logged out" and now sleeps.
        (self.home / "status-sleep").write_text("1\n")
        reader_b = await self._connect()
        await self._send_ready(self._readers_writers[reader_b], "tab-b")
        await self._read_until(reader_b, "tabs_state")
        await self._wait_for_file(self.home / "status-started")
        # A's paste completes the sign-in while that query is in flight...
        await self._send(reader_a, {"type": "signinPaste", "text": "code"})
        await self._wait_for_file(self.home / "pasted")
        relay = next(t for t in threading.enumerate() if t.name == "signin-login")
        # ...and the loop is busy (here: until the hook's relay has announced the
        # sign-in) when the query's answer comes back to it.
        relay.join(10.0)
        self.assertFalse(relay.is_alive(), "the login never ended")
        done = await self._wait_for_event(reader_a, "signinDone")
        self.assertEqual(done["email"], "me@example.com")
        await self._wait_for_event(reader_b, "signinDone")
        # Nothing reopens B's window after that: the stale answer was either
        # delivered before the signinDone or not at all.
        self.assertEqual(await self._pending_signin_events(reader_b), [])
        # A page that asks now is told nothing: the hook says logged in.
        _reader, events = await self._ready("tab-c")
        self.assertEqual(self._signin_events(events), [])


_CHILD_SCRIPT = r"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

tmp = Path(sys.argv[1])
os.environ.setdefault("KISS_WORKDIR", str(tmp))

import kiss.agents.sorcar.persistence as th

kiss_dir = tmp / ".kiss"
kiss_dir.mkdir(parents=True, exist_ok=True)
th._KISS_DIR = kiss_dir
th._DB_PATH = kiss_dir / "history.db"
th._db_conn = None

from kiss.server import signin
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.local_ws import open_local_connection

signin.HOOK_END_TIMEOUT = float(sys.argv[2])
certfile, keyfile = tmp / "cert.pem", tmp / "key.pem"
_generate_self_signed_cert(certfile, keyfile)
server = RemoteAccessServer(
    host="127.0.0.1",
    port=0,
    certfile=str(certfile),
    keyfile=str(keyfile),
    use_tunnel=False,
    url_file=tmp / "remote-url.json",
    local_endpoint_file=tmp / "sorcar-local.json",
)


async def send(writer, cmd):
    writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
    await writer.drain()


def command_name(pid):
    # ``ps`` rather than ``/proc/<pid>/comm``: macOS has no procfs.
    out = subprocess.run(["ps", "-o", "comm=", "-p", str(pid)], capture_output=True, text=True)
    return os.path.basename(out.stdout.strip())


async def main():
    await server.start_async()
    reader, writer = await open_local_connection(server)
    await send(writer, {"type": "ready", "tabId": "tab-1", "restoredTabs": []})
    await send(writer, {"type": "signinLogin", "provider": "stubborn"})
    while json.loads(await reader.readline()).get("type") != "signinUrl":
        pass
    proc = server._server_api.signin._proc
    # The hook ignores SIGTERM once it has become ``sleep``.
    while command_name(proc.pid) != "sleep":
        await asyncio.sleep(0.02)
    (tmp / "hook_pid.txt").write_text(str(proc.pid))
    writer.close()
    await server.stop_async()
    print("stopped", flush=True)


asyncio.run(main())
"""


@posix_only("the stub hook is a bash script")
class TestShutdownLeavesNoHookBehind(unittest.TestCase):
    """Issue 3: a daemon's normal exit does not abandon the kill of a hook that ignores SIGTERM."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-signin-end-"))
        self.home = self.tmpdir / "kiss-home"
        self.home.mkdir()
        hook = self.home / SIGNIN_HOOK
        hook.write_text(HOOK)
        hook.chmod(0o755)
        (self.home / "status.json").write_text(json.dumps(NOT_LOGGED_IN) + "\n")
        self.child_script = self.tmpdir / "child_daemon.py"
        self.child_script.write_text(_CHILD_SCRIPT, encoding="utf-8")
        self.hook_pid = 0

    def tearDown(self) -> None:
        if self.hook_pid and _alive(self.hook_pid):
            os.kill(self.hook_pid, signal.SIGKILL)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_the_hook_is_dead_when_the_daemon_has_exited(self) -> None:
        end_timeout = 3.0
        started = time.monotonic()
        run = subprocess.run(
            [sys.executable, str(self.child_script), str(self.tmpdir), str(end_timeout)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=str(self.tmpdir),
            env={**os.environ, "KISS_HOME": str(self.home)},
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("stopped", run.stdout)
        self.hook_pid = int((self.tmpdir / "hook_pid.txt").read_text())
        self.assertFalse(_alive(self.hook_pid), "the hook outlived the daemon")
        # The shutdown waited for the kill, and not much longer.
        self.assertGreaterEqual(time.monotonic() - started, end_timeout)


if __name__ == "__main__":
    unittest.main()

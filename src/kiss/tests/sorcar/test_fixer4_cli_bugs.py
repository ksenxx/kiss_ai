# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for fixer-4 CLI findings (findings-3).

Covered findings:

* F1 — a task that starts AND finishes between two of the submitter's
  50 ms ``task_active`` polls used to wedge :func:`_start_task` (and
  therefore :func:`_submit_task`) for the full ``timeout_seconds``
  budget.  The daemon peer here emits ``status:running=true`` and
  ``status:running=false`` back-to-back in a single socket write so
  the set→clear edge is guaranteed to land inside one poll gap.
* F2 — ``_pump_stdin`` used to busy-spin at 100 % CPU once stdin hit
  EOF (``select`` reports the closed fd readable forever and
  ``os.read`` returns ``b""`` forever).  Measured with
  ``time.process_time`` in a dedicated subprocess whose stdin is
  ``/dev/null`` (immediate EOF).
* F3 — ``AnchoredRepl.read_idle_line`` used to return only the FIRST
  submitted line when one ``os.read`` chunk carried several
  Enter-terminated lines (piped stdin / paste / fast typing),
  silently discarding the rest.
* F9 — Ctrl+C (SIGINT) while the plain client is blocked in
  ``input()`` answering an in-task ``askUser`` question used to
  fabricate the literal answer ``"done"`` instead of forwarding a
  ``stop`` to the daemon.

No mocks, patches or test doubles: the tests drive the real client
code over real Unix-domain sockets, real pipes and real subprocesses
with real signals.  The daemon peer is a genuine wire-level protocol
server (plain ``socket`` + threads) — required because reproducing
the F1 race and the F9 signal semantics demands exact control over
the byte-level timing of the daemon's replies, which a full
``RemoteAccessServer`` running a real LLM task cannot provide
deterministically.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from typing import Any

from kiss.core.print_to_console import ConsolePrinter
from kiss.ui.cli.cli_client import CliClient, _submit_task
from kiss.ui.cli.cli_steering import AnchoredRepl


class _ScriptedDaemon:
    """A real UDS peer speaking the sorcar daemon's newline-JSON protocol.

    Listens on *sock_path*, records every inbound command, and reacts:

    * ``run`` → replies ``status:true`` (+ either an immediate
      ``status:false`` in the SAME write for the F1 fast-finish race,
      or an ``askUser`` event when ``ask_user`` is set).
    * ``stop`` / ``userAnswer`` → replies ``status:false`` so the
      client's wait loop terminates on both the fixed and the buggy
      code path.

    Everything else (``setWorkDir`` / ``ready`` / ``closeTab``) is
    recorded and ignored, exactly like a daemon that has nothing to
    fan out for them.
    """

    def __init__(self, sock_path: Path, *, ask_user: bool = False) -> None:
        self.sock_path = sock_path
        self.ask_user = ask_user
        self.received: list[dict[str, Any]] = []
        self.received_cond = threading.Condition()
        self._task: tuple[str, str] = ("", "")
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(sock_path))
        self._srv.listen(4)
        threading.Thread(
            target=self._accept_loop, daemon=True,
            name="scripted-daemon-accept",
        ).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(conn,), daemon=True,
                name="scripted-daemon-conn",
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        try:
            for raw in f:
                try:
                    cmd = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(cmd, dict):
                    continue
                with self.received_cond:
                    self.received.append(cmd)
                    self.received_cond.notify_all()
                try:
                    self._react(cmd, conn)
                except OSError:
                    return
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _send(self, conn: socket.socket, msgs: list[dict[str, Any]]) -> None:
        payload = b"".join(
            json.dumps(m).encode("utf-8") + b"\n" for m in msgs
        )
        conn.sendall(payload)

    def _react(self, cmd: dict[str, Any], conn: socket.socket) -> None:
        ctype = cmd.get("type", "")
        if ctype == "run":
            tab = str(cmd.get("tabId", ""))
            task = str(cmd.get("taskId", ""))
            self._task = (tab, task)
            if self.ask_user:
                self._send(conn, [
                    {"type": "status", "running": True,
                     "tabId": tab, "taskId": task},
                    {"type": "askUser", "question": "pick a colour",
                     "tabId": tab, "taskId": task},
                ])
            else:
                # F1 reproduction: start AND end land in ONE socket
                # write, so both events are dispatched back-to-back on
                # the client's loop thread — microseconds apart, far
                # inside the submitter's 50 ms poll gap.
                self._send(conn, [
                    {"type": "status", "running": True,
                     "tabId": tab, "taskId": task},
                    {"type": "status", "running": False,
                     "tabId": tab, "taskId": task},
                ])
        elif ctype in ("stop", "userAnswer"):
            tab, task = self._task
            self._send(conn, [
                {"type": "status", "running": False,
                 "tabId": tab, "taskId": task},
            ])

    def wait_for(
        self, type_: str, timeout: float = 10.0,
    ) -> dict[str, Any] | None:
        """Block until a command of *type_* was received (or timeout)."""
        deadline = time.monotonic() + timeout
        with self.received_cond:
            while True:
                for cmd in self.received:
                    if cmd.get("type") == type_:
                        return cmd
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.received_cond.wait(remaining)

    def close(self) -> None:
        """Stop accepting connections and release the socket."""
        try:
            self._srv.close()
        except OSError:
            pass


class TestF1FastFinishingTaskAcknowledged(unittest.TestCase):
    """F1: a fast set→clear ``status`` pair must not wedge the submitter."""

    def test_fast_finishing_task_does_not_wedge_submit(self) -> None:
        tmp = tempfile.mkdtemp()
        sock = Path(tmp) / "d.sock"
        daemon = _ScriptedDaemon(sock)
        client = CliClient(sock, tmp, uuid.uuid4().hex, ConsolePrinter())
        client.start(timeout=5.0)
        try:
            t0 = time.monotonic()
            _submit_task(client, "instant task", timeout_seconds=4.0)
            elapsed = time.monotonic() - t0
            # Buggy code misses the set→clear edge and spins until the
            # full 4 s acknowledgement deadline; fixed code latches the
            # observed status and returns within milliseconds.
            self.assertLess(
                elapsed, 3.0,
                "submit wedged waiting for acknowledgement of a task "
                "that already started AND finished (F1 race)",
            )
            self.assertIsNotNone(daemon.wait_for("run", timeout=1.0))
        finally:
            client.close()
            daemon.close()


_F2_SCRIPT = r"""
import sys, threading, time
from kiss.ui.cli.cli_steering import _InputBox, _pump_stdin

box = _InputBox(threading.RLock(), sys.stdout)  # never start()ed: no TTY use
deadline = time.monotonic() + 0.8

def is_done():
    return time.monotonic() > deadline

def noop(*args):
    pass

cpu0 = time.process_time()
_pump_stdin(box, is_done, noop, noop)
print("CPU=%.3f" % (time.process_time() - cpu0), flush=True)
"""


class TestF2PumpStdinEofBusySpin(unittest.TestCase):
    """F2: ``_pump_stdin`` must idle, not spin, once stdin hits EOF."""

    def test_pump_does_not_busy_spin_after_stdin_eof(self) -> None:
        # stdin = /dev/null → the very first os.read returns b"" (EOF),
        # putting the pump in the EOF regime for its whole 0.8 s run.
        proc = subprocess.run(
            [sys.executable, "-c", _F2_SCRIPT],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        cpu_lines = [
            ln for ln in proc.stdout.splitlines() if ln.startswith("CPU=")
        ]
        self.assertTrue(cpu_lines, proc.stdout)
        cpu = float(cpu_lines[-1].split("=", 1)[1])
        # Buggy code burns ~0.8 s of CPU in the 0.8 s window (100 %
        # spin); fixed code sleeps in select and uses a few ms.
        self.assertLess(
            cpu, 0.3,
            f"_pump_stdin burned {cpu:.3f}s CPU in a 0.8s window after "
            f"stdin EOF — busy-spin (F2)",
        )


class TestF3MultiLineChunkNotDropped(unittest.TestCase):
    """F3: every line of a multi-line input chunk must be returned."""

    def test_second_line_in_one_chunk_is_returned_by_next_read(self) -> None:
        read_fd, write_fd = os.pipe()
        saved_stdin = sys.stdin
        stdin_file = os.fdopen(read_fd)
        sys.stdin = stdin_file
        write_closed = threading.Event()

        def close_writer() -> None:
            os.close(write_fd)
            write_closed.set()

        closer = threading.Timer(0.3, close_writer)
        try:
            repl = AnchoredRepl()  # box never start()ed: redraw no-ops
            # Two Enter(\r)-terminated lines in ONE chunk: feed()
            # submits both before the pump re-polls is_done().
            os.write(write_fd, b"first\rsecond\r")
            first = repl.read_idle_line()
            self.assertEqual(first, "first")
            # Unblock the buggy code path (which would otherwise wait
            # for more stdin data forever) by closing the writer soon.
            closer.start()
            second = repl.read_idle_line()
            self.assertEqual(
                second, "second",
                "second submitted line of the chunk was dropped (F3)",
            )
            self.assertIn("second", repl.box.history)
            # EOF must still surface once the surplus is drained.
            write_closed.wait(5.0)
            self.assertIsNone(repl.read_idle_line())
        finally:
            closer.cancel()
            sys.stdin = saved_stdin
            stdin_file.close()
            if not write_closed.is_set():
                os.close(write_fd)


_F9_CHILD = r"""
import sys
from pathlib import Path
from kiss.core.print_to_console import ConsolePrinter
from kiss.ui.cli.cli_client import CliClient, _submit_task

sock = Path(sys.argv[1])
work = sys.argv[2]
client = CliClient(sock, work, sys.argv[3], ConsolePrinter())
client.start(timeout=10.0)
try:
    try:
        _submit_task(client, "please ask me something", timeout_seconds=30.0)
    except KeyboardInterrupt:
        # Tolerated: SIGINT landed outside the input() prompt window.
        print("CHILD-KBINT", flush=True)
finally:
    client.close()
print("CHILD-DONE", flush=True)
"""


class TestF9CtrlCAtAskUserPrompt(unittest.TestCase):
    """F9: Ctrl+C at the askUser ``input()`` prompt must stop, not answer."""

    def test_sigint_sends_stop_and_never_fabricates_done(self) -> None:
        tmp = tempfile.mkdtemp()
        sock = Path(tmp) / "d.sock"
        daemon = _ScriptedDaemon(sock, ask_user=True)
        tab_id = uuid.uuid4().hex
        proc = subprocess.Popen(
            [sys.executable, "-c", _F9_CHILD, str(sock), tmp, tab_id],
            stdin=subprocess.PIPE,  # kept open so input() blocks
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertIsNotNone(
                daemon.wait_for("run", timeout=20.0),
                "child never submitted the task",
            )
            # Give the child time to drain the askUser event on its
            # REPL thread, print the question and block in input().
            time.sleep(1.0)
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=30)
            stop_cmd = daemon.wait_for("stop", timeout=5.0)
            answers = [
                c for c in daemon.received if c.get("type") == "userAnswer"
            ]
            self.assertFalse(
                any(a.get("answer") == "done" for a in answers),
                f"Ctrl+C fabricated the answer 'done' (F9): {answers}",
            )
            self.assertIsNotNone(
                stop_cmd,
                "Ctrl+C at the askUser prompt did not forward a stop "
                f"to the daemon (F9); child out={out!r} err={err!r}",
            )
            self.assertEqual(proc.returncode, 0, f"out={out!r} err={err!r}")
            self.assertIn("CHILD-DONE", out)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            daemon.close()


if __name__ == "__main__":
    unittest.main()

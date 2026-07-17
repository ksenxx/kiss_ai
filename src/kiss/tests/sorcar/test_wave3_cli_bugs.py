# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for wave-3 CLI findings.

Covered findings (see ./tmp/w3-findings-A.md / w3-findings-C.md):

* A-F1 — a stray UNTAGGED ``status:running=false`` broadcast landing
  between ``_submit_task``'s ``task_started.clear()`` and the daemon's
  first echoed status for the new task used to satisfy the
  acknowledgement latch with ``task_active`` clear, so the submitter
  returned immediately and the freshly submitted task kept running in
  the daemon silently orphaned.
* A-F4 — a :class:`RecordingConsolePrinter` garbage-collected mid-task
  (agent abandoned the task without a terminal ``result``) fell out of
  the old WeakSet safety net, so the process-exit ``cliTaskEnd``
  guarantee was silently dropped and the daemon's running-indicator
  entry leaked forever.
* A-F6 — a single stdin chunk carrying ``"line\\r" + Ctrl+C`` used to
  drop the submitted line entirely (neither returned nor buffered),
  while the same line arriving one chunk before the Ctrl+C survived.
* C-1 — ``cli_daemon_bridge._connect`` leaked the AF_UNIX socket fd
  (ResourceWarning at GC) whenever ``connect()`` failed, i.e. once per
  broadcast for a CLI run with no daemon listening.
* C-3 — a ``--header`` value without a ``:`` was silently dropped by
  ``_build_run_kwargs`` instead of being rejected loudly like the
  sibling ``sorcar mcp`` CLI does.
* C-5 — the CLI-side default UDS socket path was hard-coded to
  ``~/.kiss/sorcar.sock`` while the daemon binds
  ``$KISS_HOME/sorcar.sock`` — with ``KISS_HOME`` set, every CLI
  broadcast silently went nowhere.

No mocks, patches or test doubles: real Unix-domain sockets, real
pipes, real subprocesses, real environment variables.
"""

from __future__ import annotations

import json
import os
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

from kiss.agents.sorcar.cli_helpers import _build_arg_parser, _build_run_kwargs
from kiss.core.print_to_console import ConsolePrinter
from kiss.ui.cli.cli_client import CliClient, _submit_task
from kiss.ui.cli.cli_steering import AnchoredRepl


class _EnvelopeSink:
    """A real UDS server that records every newline-JSON envelope.

    Plays the daemon side of :mod:`cli_daemon_bridge`: accepts
    connections at *sock_path* and appends each decoded envelope to
    :attr:`received` (condition-notified so tests can block for one).
    """

    def __init__(self, sock_path: Path) -> None:
        self.sock_path = sock_path
        self.received: list[dict[str, Any]] = []
        self.cond = threading.Condition()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(sock_path))
        self._srv.listen(4)
        threading.Thread(
            target=self._accept_loop, daemon=True, name="envelope-sink",
        ).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(conn,), daemon=True,
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        try:
            for raw in f:
                try:
                    env = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(env, dict):
                    with self.cond:
                        self.received.append(env)
                        self.cond.notify_all()
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def wait_for(
        self, type_: str, timeout: float = 10.0,
    ) -> dict[str, Any] | None:
        """Block until an envelope of *type_* arrives (or timeout)."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                for env in self.received:
                    if env.get("type") == type_:
                        return env
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)

    def close(self) -> None:
        """Stop accepting connections and release the socket."""
        try:
            self._srv.close()
        except OSError:
            pass


class _StrayStatusDaemon:
    """A real UDS daemon peer reproducing the wave-3 F1 race.

    On ``run`` it FIRST sends an UNTAGGED ``status:running=false``
    (exactly what the real daemon emits for webview-launched tasks
    ending on the same chat and for viewer-fanout broadcasts), then —
    only after a delay — the properly ``taskId``-tagged
    ``status:true`` / ``status:false`` lifecycle of the submitted
    task.  ``lifecycle_done`` is set immediately BEFORE the final
    tagged ``status:false`` is written, so a submitter that returns
    while it is still clear provably abandoned a task that was still
    running on the daemon.
    """

    def __init__(self, sock_path: Path) -> None:
        self.sock_path = sock_path
        self.lifecycle_done = threading.Event()
        self.got_run = threading.Event()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(sock_path))
        self._srv.listen(4)
        threading.Thread(
            target=self._accept_loop, daemon=True, name="stray-daemon",
        ).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(conn,), daemon=True,
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        try:
            for raw in f:
                try:
                    cmd = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(cmd, dict) or cmd.get("type") != "run":
                    continue
                tab = str(cmd.get("tabId", ""))
                task = str(cmd.get("taskId", ""))
                self.got_run.set()
                # The stray untagged end — no ``taskId`` key at all.
                conn.sendall(json.dumps({
                    "type": "status", "running": False, "tabId": tab,
                }).encode() + b"\n")
                time.sleep(0.4)
                conn.sendall(json.dumps({
                    "type": "status", "running": True,
                    "tabId": tab, "taskId": task,
                }).encode() + b"\n")
                time.sleep(0.4)
                self.lifecycle_done.set()
                conn.sendall(json.dumps({
                    "type": "status", "running": False,
                    "tabId": tab, "taskId": task,
                }).encode() + b"\n")
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        """Stop accepting connections and release the socket."""
        try:
            self._srv.close()
        except OSError:
            pass


class TestW3F1UntaggedStatusDoesNotOrphanTask(unittest.TestCase):
    """A-F1: an untagged status:false must not orphan a fresh submit."""

    def test_submit_waits_out_full_lifecycle_despite_stray_status(
        self,
    ) -> None:
        tmp = tempfile.mkdtemp()
        sock = Path(tmp) / "d.sock"
        daemon = _StrayStatusDaemon(sock)
        client = CliClient(sock, tmp, uuid.uuid4().hex, ConsolePrinter())
        client.start(timeout=5.0)
        try:
            _submit_task(client, "long task", timeout_seconds=20.0)
            self.assertTrue(daemon.got_run.wait(1.0))
            self.assertTrue(
                daemon.lifecycle_done.is_set(),
                "_submit_task returned while the submitted task was "
                "still running on the daemon — a stray UNTAGGED "
                "status:false satisfied the acknowledgement latch and "
                "the task was silently orphaned (w3 F1)",
            )
        finally:
            client.close()
            daemon.close()


_F4_CHILD = r"""
import gc, sys
from kiss.ui.cli.cli_printer import RecordingConsolePrinter

tid = sys.argv[1]
printer = RecordingConsolePrinter()
printer.broadcast({"type": "text_delta", "text": "hi", "taskId": tid})
# The agent abandons the task: the printer's last reference dies
# mid-task, long before process exit, and no terminal ``result``
# event is ever broadcast.
del printer
gc.collect()
print("CHILD-EXIT", flush=True)
"""


class TestW3F4PrinterGcMidTaskStillEndsTask(unittest.TestCase):
    """A-F4: cliTaskEnd must be sent even if the printer was GC'd."""

    def test_atexit_sends_cli_task_end_for_gcd_printer(self) -> None:
        tmp = tempfile.mkdtemp()
        sock = Path(tmp) / "sink.sock"
        sink = _EnvelopeSink(sock)
        tid = uuid.uuid4().hex
        env = dict(os.environ)
        env["KISS_SORCAR_SOCK"] = str(sock)
        env["KISS_HOME"] = tmp  # isolate the chat-DB side effects
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _F4_CHILD, tid],
                env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("CHILD-EXIT", proc.stdout)
            start = sink.wait_for("cliTaskStart", timeout=5.0)
            self.assertIsNotNone(start, "cliTaskStart never arrived")
            assert start is not None
            self.assertEqual(start.get("taskId"), tid)
            end = sink.wait_for("cliTaskEnd", timeout=5.0)
            self.assertIsNotNone(
                end,
                "cliTaskEnd was never sent for a printer garbage-"
                "collected mid-task — the daemon's running-indicator "
                "entry leaks until restart (w3 F4)",
            )
            assert end is not None
            self.assertEqual(end.get("taskId"), tid)
        finally:
            sink.close()


class TestW3F6CtrlCSameChunkLineSurvives(unittest.TestCase):
    """A-F6: a line sharing its input chunk with Ctrl+C must survive."""

    def test_line_before_ctrl_c_in_one_chunk_is_buffered(self) -> None:
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
            # One chunk: an Enter-terminated line followed by Ctrl+C.
            os.write(write_fd, b"deploy\r\x03")
            with self.assertRaises(KeyboardInterrupt):
                repl.read_idle_line()
            # Unblock the buggy code path (which dropped the line and
            # would otherwise block on stdin forever) by closing the
            # writer shortly; the fixed code serves the buffered line
            # without touching stdin at all.
            closer.start()
            line = repl.read_idle_line()
            self.assertEqual(
                line, "deploy",
                "the line submitted in the same chunk as Ctrl+C was "
                "dropped (w3 F6)",
            )
        finally:
            closer.cancel()
            sys.stdin = saved_stdin
            stdin_file.close()
            if not write_closed.is_set():
                os.close(write_fd)


_C1_CHILD = r"""
import gc
from kiss.ui.cli import cli_daemon_bridge

for _ in range(20):
    cli_daemon_bridge.send_event({"type": "probe"})
gc.collect()
print("CHILD-OK", flush=True)
"""


class TestW3C1NoFdLeakOnFailedConnect(unittest.TestCase):
    """C-1: a failed daemon connect must close its socket fd."""

    def test_no_resource_warning_when_daemon_absent(self) -> None:
        tmp = tempfile.mkdtemp()
        env = dict(os.environ)
        # Point the bridge at a socket path that cannot exist.
        env["KISS_SORCAR_SOCK"] = str(Path(tmp) / "no" / "daemon.sock")
        proc = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", "-c", _C1_CHILD],
            env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("CHILD-OK", proc.stdout)
        self.assertNotIn(
            "ResourceWarning", proc.stderr,
            "cli_daemon_bridge._connect leaked the AF_UNIX socket fd "
            f"on a failed connect (w3 C-1): {proc.stderr!r}",
        )


class TestW3C3MalformedHeaderRejected(unittest.TestCase):
    """C-3: a --header without ':' must be rejected, not dropped."""

    def _parse(self, header: str) -> Any:
        parser = _build_arg_parser()
        return parser.parse_args(["-t", "x", "--header", header])

    def test_header_without_colon_raises(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            _build_run_kwargs(self._parse("Authorization Bearer x"))
        self.assertIn("Authorization Bearer x", str(ctx.exception))

    def test_header_with_empty_key_raises(self) -> None:
        with self.assertRaises(SystemExit):
            _build_run_kwargs(self._parse(":no-key"))

    def test_valid_header_still_flows_into_model_config(self) -> None:
        kwargs = _build_run_kwargs(self._parse("X-Custom: value1"))
        self.assertEqual(
            kwargs["model_config"]["extra_headers"], {"X-Custom": "value1"},
        )


_C5_CHILD = r"""
from kiss.ui.cli import cli_client, cli_daemon_bridge

print("CLIENTSOCK=" + str(cli_client._sock_path()), flush=True)
print("BRIDGESOCK=" + str(cli_daemon_bridge._sock_path()), flush=True)
cli_daemon_bridge.send_event({"type": "w3c5probe"})
print("CHILD-OK", flush=True)
"""


class TestW3C5SocketPathHonoursKissHome(unittest.TestCase):
    """C-5: CLI socket defaults must follow $KISS_HOME like the daemon."""

    def test_bridge_and_client_target_kiss_home_socket(self) -> None:
        tmp = tempfile.mkdtemp()
        expected = Path(tmp) / "sorcar.sock"
        sink = _EnvelopeSink(expected)
        env = dict(os.environ)
        env["KISS_HOME"] = tmp
        env.pop("KISS_SORCAR_SOCK", None)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _C5_CHILD],
                env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(f"CLIENTSOCK={expected}", proc.stdout)
            self.assertIn(f"BRIDGESOCK={expected}", proc.stdout)
            envelope = sink.wait_for("cliEvent", timeout=5.0)
            self.assertIsNotNone(
                envelope,
                "with KISS_HOME set the CLI bridge never reached the "
                "daemon socket at $KISS_HOME/sorcar.sock — it silently "
                "targeted ~/.kiss/sorcar.sock instead (w3 C-5)",
            )
            assert envelope is not None
            self.assertEqual(envelope["event"]["type"], "w3c5probe")
        finally:
            sink.close()


if __name__ == "__main__":
    unittest.main()

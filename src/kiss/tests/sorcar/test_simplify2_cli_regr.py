# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the simplify-2 CLI refactoring pass.

Pins the externally observable behaviour of the code paths about to be
simplified (see ./tmp/simplify-report-sorcar-cli.md), so the
refactoring can be verified not to change behaviour:

* C1 — ``_submit_task`` / ``_submit_task_anchored`` arm/reset dance
  (mint task id, clear latches, drain queues; always reset in
  ``finally``) shared via one context manager.
* C2 — ``/commands`` / ``/skills`` / ``/mcp`` / ``/cost`` / ``/usage``
  / ``/context`` info commands all print the daemon's ``cliInfo``
  reply text.
* C3 — the three poll-a-queue-with-deadline-and-disconnect-bail loops
  (``_request_cli_info``, ``_request_models``, ``/autocommit``).
* S1 — CSI (``ESC[``) and SS3 (``ESC O``) navigation keys behave
  identically in the steering box (arrows / Home / End).
* V1/V2 — voice listener spawn error handling is identical for the
  modal (``start_voice``) and anchored (``start_voice_anchored``)
  entry points.
* R1 — the idle ``_read_line`` panel frame (top / bottom borders) and
  backslash line continuation off-TTY.

No mocks, patches or test doubles: real Unix-domain sockets, real
threads, real subprocesses, real environment variables.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import cli_voice
from kiss.agents.sorcar.cli_client import (
    CliClient,
    _handle_client_slash,
    _request_cli_info,
    _request_models,
    _submit_task,
    _submit_task_anchored,
)
from kiss.agents.sorcar.cli_steering import AnchoredRepl, _InputBox
from kiss.core.print_to_console import ConsolePrinter


class _FakeDaemon:
    """A real UDS server speaking the daemon's newline-JSON protocol.

    Records every command the client sends and lets each test install
    per-command-type reply handlers (returning lists of reply events).
    """

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="sorcar-simplify2-")
        self.path = Path(self.dir) / "d.sock"
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(self.path))
        self._srv.listen(1)
        self.received: list[dict[str, Any]] = []
        self.handlers: dict[
            str, Callable[[dict[str, Any]], list[dict[str, Any]]]
        ] = {}
        self._conn: socket.socket | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        self._conn = conn
        buf = b""
        while True:
            try:
                data = conn.recv(65536)
            except OSError:
                return
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                cmd = json.loads(line)
                self.received.append(cmd)
                handler = self.handlers.get(cmd.get("type", ""))
                if handler is None:
                    continue
                for reply in handler(cmd):
                    try:
                        conn.sendall(json.dumps(reply).encode() + b"\n")
                    except OSError:
                        return

    def drop_connection(self) -> None:
        """Hard-close the accepted connection (simulates daemon death)."""
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                self._conn.close()

    def close(self) -> None:
        self.drop_connection()
        with contextlib.suppress(OSError):
            self._srv.close()
        self._thread.join(timeout=2)

    def wait_for(
        self, cmd_type: str, timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Block until a command of *cmd_type* was received."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for cmd in list(self.received):
                if cmd.get("type") == cmd_type:
                    return cmd
            time.sleep(0.02)
        raise AssertionError(f"daemon never received {cmd_type!r}")


def _connect_client(daemon: _FakeDaemon) -> CliClient:
    client = CliClient(daemon.path, daemon.dir, "tab-simplify2",
                       ConsolePrinter())
    client.start(timeout=5.0)
    return client


class TestClientRequestReplyLoops(unittest.TestCase):
    """C2 / C3 — the queue-poll loops and the info slash commands."""

    def setUp(self) -> None:
        self.daemon = _FakeDaemon()
        self.addCleanup(self.daemon.close)

    def test_request_cli_info_roundtrip_filters_request_id(self) -> None:
        def on_cli_info(cmd: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                # A stale reply (wrong requestId) must be skipped.
                {"type": "cliInfo", "subtype": cmd["subtype"],
                 "requestId": "stale", "text": "STALE"},
                {"type": "cliInfo", "subtype": cmd["subtype"],
                 "requestId": cmd["requestId"], "text": "FRESH"},
            ]

        self.daemon.handlers["cliInfo"] = on_cli_info
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        reply = _request_cli_info(client, "help", timeout=5.0)
        self.assertEqual(reply.get("text"), "FRESH")

    def test_request_cli_info_disconnect_bails_early(self) -> None:
        def on_cli_info(cmd: dict[str, Any]) -> list[dict[str, Any]]:
            del cmd
            self.daemon.drop_connection()
            return []

        self.daemon.handlers["cliInfo"] = on_cli_info
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        start = time.monotonic()
        reply = _request_cli_info(client, "help", timeout=30.0)
        elapsed = time.monotonic() - start
        self.assertTrue(reply.get("error"))
        self.assertIn("connection lost", str(reply.get("errorMessage")))
        self.assertLess(elapsed, 10.0)

    def test_request_cli_info_timeout_reply(self) -> None:
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        reply = _request_cli_info(client, "help", timeout=0.5)
        self.assertTrue(reply.get("timedOut"))
        self.assertTrue(reply.get("error"))

    def test_request_models_updates_current_model(self) -> None:
        def on_get_models(cmd: dict[str, Any]) -> list[dict[str, Any]]:
            del cmd
            return [{"type": "models",
                     "models": [{"name": "m1"}, {"name": "m2"}],
                     "selected": "m2"}]

        self.daemon.handlers["getModels"] = on_get_models
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        models = _request_models(client)
        self.assertEqual([m["name"] for m in models], ["m1", "m2"])
        self.assertEqual(client.dispatcher.current_model, "m2")

    def test_request_models_timeout_returns_empty(self) -> None:
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        self.daemon.drop_connection()
        # ``client._closed`` flips once the reader hits EOF; the wait
        # must bail out early instead of sitting out the full 10 s.
        start = time.monotonic()
        models = _request_models(client)
        self.assertEqual(models, [])
        self.assertLess(time.monotonic() - start, 10.0)

    def test_info_slash_commands_print_reply_text(self) -> None:
        def on_cli_info(cmd: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"type": "cliInfo", "subtype": cmd["subtype"],
                     "requestId": cmd["requestId"],
                     "text": f"INFO-{cmd['subtype']}"}]

        self.daemon.handlers["cliInfo"] = on_cli_info
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        for line, subtype in (
            ("/commands", "commands"),
            ("/skills", "skills"),
            ("/mcp", "mcp"),
            ("/cost", "cost"),
            ("/usage", "cost"),
            ("/context", "cost"),
        ):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                should_exit = _handle_client_slash(client, line)
            self.assertFalse(should_exit)
            self.assertIn(f"INFO-{subtype}", out.getvalue(), msg=line)

    def test_skills_command_forwards_name_argument(self) -> None:
        def on_cli_info(cmd: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"type": "cliInfo", "subtype": cmd["subtype"],
                     "requestId": cmd["requestId"],
                     "text": f"SKILL={cmd.get('name', '')}"}]

        self.daemon.handlers["cliInfo"] = on_cli_info
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _handle_client_slash(client, "/skills my-skill")
        self.assertIn("SKILL=my-skill", out.getvalue())

    def test_autocommit_timeout_and_disconnect_messages(self) -> None:
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        # Disconnect: the wait must abort with the connection-lost
        # message, well before the 30 s commit-message budget.
        self.daemon.drop_connection()
        deadline = time.monotonic() + 5.0
        while (not client._closed.is_set()
               and time.monotonic() < deadline):
            time.sleep(0.02)
        out = io.StringIO()
        start = time.monotonic()
        with contextlib.redirect_stdout(out):
            _handle_client_slash(client, "/autocommit")
        self.assertLess(time.monotonic() - start, 10.0)
        self.assertIn("daemon connection lost", out.getvalue())


class TestSubmitTaskArmReset(unittest.TestCase):
    """C1 — the arm/reset dance around every task submission."""

    def setUp(self) -> None:
        self.daemon = _FakeDaemon()
        self.addCleanup(self.daemon.close)

    def test_submit_task_full_cycle_resets_dispatcher_state(self) -> None:
        def on_run(cmd: dict[str, Any]) -> list[dict[str, Any]]:
            tid = cmd.get("taskId", "")
            tab = cmd.get("tabId", "")
            return [
                {"type": "status", "running": True,
                 "taskId": tid, "tabId": tab},
                {"type": "status", "running": False,
                 "taskId": tid, "tabId": tab},
            ]

        self.daemon.handlers["run"] = on_run
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        _submit_task(
            client, "do the thing",
            use_worktree=False, use_parallel=False, auto_commit=True,
            timeout_seconds=15.0,
        )
        run_cmd = self.daemon.wait_for("run")
        self.assertEqual(run_cmd["prompt"], "do the thing")
        self.assertFalse(run_cmd["useWorktree"])
        self.assertFalse(run_cmd["useParallel"])
        self.assertTrue(run_cmd["autoCommit"])
        self.assertTrue(run_cmd["taskId"])
        # After the submission the per-task dispatcher state is reset
        # regardless of exit path.
        self.assertEqual(client.dispatcher.current_task_id, "")
        self.assertFalse(client.dispatcher.task_active.is_set())
        self.assertFalse(client.dispatcher.task_started.is_set())

    def test_submit_task_unacknowledged_resets_state(self) -> None:
        # Daemon never acknowledges: the submitter must time out AND
        # still reset the armed task id / latches in its finally.
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _submit_task(client, "never acked", timeout_seconds=1.0)
        self.assertEqual(client.dispatcher.current_task_id, "")
        self.assertFalse(client.dispatcher.task_active.is_set())
        self.assertFalse(client.dispatcher.task_started.is_set())

    def test_anchored_timeout_starts_after_submission_is_armed(self) -> None:
        """Dispatcher-lock contention must not consume the task timeout."""
        def on_run(cmd: dict[str, Any]) -> list[dict[str, Any]]:
            return [{
                "type": "status",
                "running": True,
                "taskId": cmd.get("taskId", ""),
                "tabId": cmd.get("tabId", ""),
            }]

        class ProbeRepl(AnchoredRepl):
            def __init__(self) -> None:
                super().__init__()
                self.first_done: bool | None = None

            def run_steering_loop(
                self,
                on_submit: Callable[[str], None],
                on_abort: Callable[[], None],
                is_done: Callable[[], bool],
                on_idle: Callable[[], None] | None = None,
            ) -> None:
                del on_submit, on_abort
                if on_idle is not None:
                    on_idle()
                self.first_done = is_done()

        self.daemon.handlers["run"] = on_run
        client = _connect_client(self.daemon)
        self.addCleanup(client.close)
        repl = ProbeRepl()
        errors: list[BaseException] = []

        def submit() -> None:
            try:
                _submit_task_anchored(
                    client, "anchored", repl, timeout_seconds=0.5,
                )
            except BaseException as exc:  # surfaced in the test thread
                errors.append(exc)

        # Hold the real dispatcher lock longer than the timeout.  The old
        # ordering armed first and only then created the deadline, so the
        # first steering-loop check must still report an active task.
        client.dispatcher.task_id_lock.acquire()
        worker = threading.Thread(target=submit)
        try:
            worker.start()
            time.sleep(0.8)
            self.assertTrue(worker.is_alive(), "submitter did not block on task-id lock")
        finally:
            client.dispatcher.task_id_lock.release()
        worker.join(timeout=5.0)

        self.assertFalse(worker.is_alive(), "anchored submitter did not finish")
        self.assertEqual(errors, [])
        self.assertIs(repl.first_done, False)


class TestSteeringNavKeyParity(unittest.TestCase):
    """S1 — CSI and SS3 navigation keys behave identically."""

    @staticmethod
    def _box() -> _InputBox:
        return _InputBox(threading.RLock(), io.StringIO())

    @staticmethod
    def _feed(box: _InputBox, data: bytes) -> None:
        box.feed(data, lambda _line: None, lambda: None)

    def test_csi_and_ss3_cursor_moves_match(self) -> None:
        for left, right, home, end in (
            (b"\x1b[D", b"\x1b[C", b"\x1b[H", b"\x1b[F"),   # CSI
            (b"\x1bOD", b"\x1bOC", b"\x1bOH", b"\x1bOF"),   # SS3
            (b"\x1b[D", b"\x1b[C", b"\x1b[1~", b"\x1b[4~"),  # tilde Home/End
            (b"\x1b[D", b"\x1b[C", b"\x1b[7~", b"\x1b[8~"),  # rxvt Home/End
        ):
            box = self._box()
            box.buf = "hello"
            self.assertEqual(box.cursor, 5)
            self._feed(box, left)
            self.assertEqual(box.cursor, 4, msg=repr(left))
            self._feed(box, right)
            self.assertEqual(box.cursor, 5, msg=repr(right))
            self._feed(box, home)
            self.assertEqual(box.cursor, 0, msg=repr(home))
            self._feed(box, end)
            self.assertEqual(box.cursor, 5, msg=repr(end))

    def test_csi_and_ss3_up_down_move_between_lines(self) -> None:
        for up, down in ((b"\x1b[A", b"\x1b[B"), (b"\x1bOA", b"\x1bOB")):
            box = self._box()
            box.buf = "ab\ncd"
            self.assertEqual(box.cursor, 5)
            self._feed(box, up)
            self.assertEqual(box.cursor, 2, msg=repr(up))
            self._feed(box, down)
            self.assertEqual(box.cursor, 5, msg=repr(down))

    def test_csi_and_ss3_up_browse_history_on_empty_buffer(self) -> None:
        for up in (b"\x1b[A", b"\x1bOA"):
            box = self._box()
            box.history = ["first", "second"]
            self._feed(box, up)
            self.assertEqual(box.buf, "second", msg=repr(up))
            self._feed(box, up)
            self.assertEqual(box.buf, "first", msg=repr(up))


class TestVoiceSpawnParity(unittest.TestCase):
    """V1 / V2 — listener spawn + failure handling in both voice modes."""

    def setUp(self) -> None:
        self._prev = os.environ.get("KISS_SORCAR_VOICE_CMD")

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("KISS_SORCAR_VOICE_CMD", None)
        else:
            os.environ["KISS_SORCAR_VOICE_CMD"] = self._prev

    def test_spawn_failure_prints_same_error_in_both_modes(self) -> None:
        os.environ["KISS_SORCAR_VOICE_CMD"] = (
            "/nonexistent-simplify2-voice-binary"
        )
        out_modal = io.StringIO()
        with contextlib.redirect_stdout(out_modal):
            session = cli_voice.start_voice(cli_voice.read_voice_line_plain)
        self.assertIsNone(session)
        box = _InputBox(threading.RLock(), io.StringIO())
        out_anchored = io.StringIO()
        with contextlib.redirect_stdout(out_anchored):
            session2 = cli_voice.start_voice_anchored(box)
        self.assertIsNone(session2)
        for text in (out_modal.getvalue(), out_anchored.getvalue()):
            self.assertIn("could not start the wake-word listener", text)

    def test_anchored_success_injects_recognised_speech(self) -> None:
        script = Path(tempfile.mkdtemp(prefix="sorcar-simplify2-")) / "l.py"
        script.write_text(
            "import json, sys, time\n"
            "print('READY', flush=True)\n"
            "print('WAKE', flush=True)\n"
            "print('TRANSCRIBING', flush=True)\n"
            "payload = {'text': 'hello world', 'speaker': 1,"
            " 'language': 'en-US'}\n"
            "print('SPEECH ' + json.dumps(payload), flush=True)\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        os.environ["KISS_SORCAR_VOICE_CMD"] = (
            f"{sys.executable} {script}"
        )
        box = _InputBox(threading.RLock(), io.StringIO())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            session = cli_voice.start_voice_anchored(box)
        self.assertIsNotNone(session)
        assert session is not None
        try:
            deadline = time.monotonic() + 10.0
            injected: list[str] = []
            while time.monotonic() < deadline and not injected:
                injected = box.drain_injected()
                time.sleep(0.02)
            self.assertTrue(injected, "pump never injected the utterance")
            self.assertIn("hello world", injected[0])
        finally:
            with contextlib.redirect_stdout(io.StringIO()):
                session.close()
        proc = session.listener.proc
        assert proc is not None
        self.assertIsNotNone(proc.poll(), "listener child was leaked")


class TestIdleReadLineFrame(unittest.TestCase):
    """R1 — the off-TTY idle panel frame and line continuation."""

    @staticmethod
    def _run_read_line(stdin_text: str) -> str:
        repo_src = str(
            Path(__file__).resolve().parents[3],
        )
        code = (
            "import sys\n"
            "from kiss.agents.sorcar.cli_repl import _read_line\n"
            "line = _read_line('> ')\n"
            "sys.stdout.write('GOT=' + repr(line) + '\\n')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(  # noqa: S603 - our own interpreter
            [sys.executable, "-c", code],
            input=stdin_text, capture_output=True, text=True,
            timeout=60, env=env, check=True,
        )
        return proc.stdout

    def test_panel_frame_and_returned_line(self) -> None:
        out = self._run_read_line("hello\n")
        self.assertIn("╭─", out)   # top border
        self.assertIn("╰", out)    # bottom border
        self.assertIn("GOT='hello'", out)

    def test_backslash_continuation_joins_lines(self) -> None:
        out = self._run_read_line("foo \\\ncontinued\n")
        self.assertIn("GOT='foo \\ncontinued'", out)


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Wave-2 e2e regression tests for CLI bugs (findings-3 F4-F23).

Covered findings (see ./tmp/findings-3.md):

* F4  — ``_InputBox`` buffer/menu state mutated without the shared
        lock while other threads read it under the lock.
* F5  — ``SteeringSession`` mutated ``box.title`` / ``box.status``
        outside the lock the redraw path holds.
* F6  — ``_interrupt_worker`` ignored ``PyThreadState_SetAsyncExc``'s
        return value and a failed ``join``.
* F13 — readline-format and plain-lines history written to the SAME
        file by different REPL modes (libedit ``_HiStOrY_V2_`` header
        leaked into the anchored history).
* F19 — importing ``cli_prompt`` permanently mutated prompt_toolkit's
        process-global ``ANSI_SEQUENCES`` table.
* F21 — every ``RecordingConsolePrinter`` registered a per-instance
        ``atexit`` handler that pinned it in memory forever.
* F22 — concurrent first broadcasts of a task could reach the daemon
        before the matching ``cliTaskStart`` envelope.
* F23 — ``-v/--verbose`` took a mandatory string value and treated
        any spelling other than exactly "true" as ``False``.

All tests are real end-to-end tests: real threads, real pipes, real
UNIX sockets, real subprocesses, and real files — no mocks.
"""

from __future__ import annotations

import gc
import io
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import weakref
from contextlib import redirect_stdout
from pathlib import Path

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.cli_helpers import _build_arg_parser
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.ui.cli import cli_daemon_bridge
from kiss.ui.cli.cli_repl import (
    _load_history_lines,
    _save_history_lines,
)
from kiss.ui.cli.cli_steering import SteeringSession, _InputBox

_REPO_SRC = str(Path(__file__).resolve().parents[3])


def _noop_submit(line: str) -> None:
    """Discard a submitted line (feed callback for tests)."""
    del line


def _noop_abort() -> None:
    """Ignore an abort (feed callback for tests)."""


class _UdsCaptureServer:
    """A real AF_UNIX newline-JSON server capturing bridge envelopes."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.envelopes: list[dict[str, object]] = []
        self.lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(8)
        self._closed = threading.Event()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(
                target=self._read_loop, args=(conn,), daemon=True
            ).start()

    def _read_loop(self, conn: socket.socket) -> None:
        buf = b""
        with conn:
            while not self._closed.is_set():
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    envelope = json.loads(line.decode())
                    with self.lock:
                        self.envelopes.append(envelope)

    def wait_for(self, count: int, timeout: float = 10.0) -> None:
        """Block until at least *count* envelopes arrived (or timeout)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if len(self.envelopes) >= count:
                    return
            time.sleep(0.01)

    def snapshot(self) -> list[dict[str, object]]:
        """Return a copy of the envelopes captured so far."""
        with self.lock:
            return list(self.envelopes)

    def close(self) -> None:
        """Stop the server and its worker threads."""
        self._closed.set()
        try:
            self._sock.close()
        except OSError:
            pass


def _reset_bridge_writer() -> None:
    """Drop cli_daemon_bridge's cached UDS writer between tests."""
    with cli_daemon_bridge._LOCK:
        writer = cli_daemon_bridge._WRITER
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass
            cli_daemon_bridge._WRITER = None
            cli_daemon_bridge._WRITER_PATH = None


class TestF4BufLockDiscipline(unittest.TestCase):
    """F4: ``buf`` must never change while another thread holds the lock.

    The pump thread used to mutate ``_InputBox.buf`` (printable chars,
    backspace, newline inserts, submit reset) WITHOUT the shared lock,
    while worker threads read ``buf`` under the lock for cursor
    parking/redraws.  Post-fix every mutation happens under the lock,
    so a locked reader must observe a stable ``buf``.
    """

    def test_buf_stable_while_lock_held(self) -> None:
        box = _InputBox(threading.RLock(), io.StringIO())
        stop = threading.Event()
        violations: list[tuple[str, str]] = []

        def reader() -> None:
            while not stop.is_set():
                with box.lock:
                    first = box.buf
                    time.sleep(random.uniform(0.0, 0.001))
                    second = box.buf
                if first != second:
                    violations.append((first, second))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            box.feed(b"hello world", _noop_submit, _noop_abort)
            box.feed(b"\x7f\x7f\x7f\x7f", _noop_submit, _noop_abort)
            box.feed(b"\n", _noop_submit, _noop_abort)
            box.feed(b"\r", _noop_submit, _noop_abort)
        stop.set()
        thread.join(timeout=5.0)
        self.assertEqual(
            violations, [],
            "buf changed while another thread held the lock",
        )

    def test_feed_still_edits_and_submits(self) -> None:
        """Locking must not change feed()'s editing semantics."""
        box = _InputBox(threading.RLock(), io.StringIO())
        lines: list[str] = []
        box.feed(b"abc", lines.append, _noop_abort)
        self.assertEqual(box.buf, "abc")
        box.feed(b"\x7f", lines.append, _noop_abort)
        self.assertEqual(box.buf, "ab")
        box.feed(b"\n", lines.append, _noop_abort)  # Ctrl+J newline insert
        self.assertEqual(box.buf, "ab\n")
        box.feed(b"c\r", lines.append, _noop_abort)  # Enter submits
        self.assertEqual(lines, ["ab\nc"])
        self.assertEqual(box.buf, "")
        box.feed(b"\x1b[A", lines.append, _noop_abort)  # Up: history
        self.assertEqual(box.buf, "")  # history list not fed here — no-op


class TestF5TitleStatusLockDiscipline(unittest.TestCase):
    """F5: ``box.title`` / ``box.status`` must only mutate under the lock."""

    def _session(self) -> SteeringSession:
        agent = SorcarAgent("w2-steer-test")
        state = _RunningAgentState("w2-chat", "", agent=None)
        return SteeringSession(agent, state, "w2-chat")

    def test_title_and_status_stable_while_lock_held(self) -> None:
        session = self._session()
        box = session.box
        stop = threading.Event()
        violations: list[tuple[tuple[str, str], tuple[str, str]]] = []

        def reader() -> None:
            while not stop.is_set():
                with session.lock:
                    before = (box.title, box.status)
                    time.sleep(random.uniform(0.0, 0.001))
                    after = (box.title, box.status)
                if before != after:
                    violations.append((before, after))

        def answerer() -> None:
            while not stop.is_set():
                if session._question_pending.is_set():
                    session._on_submit("an answer")
                time.sleep(0.0005)

        reader_t = threading.Thread(target=reader, daemon=True)
        answer_t = threading.Thread(target=answerer, daemon=True)
        reader_t.start()
        answer_t.start()
        deadline = time.monotonic() + 1.5
        with redirect_stdout(io.StringIO()):
            while time.monotonic() < deadline:
                answer = session.ask_user_question("which one?")
                self.assertEqual(answer, "an answer")
                session._on_submit("queued instruction")
        stop.set()
        reader_t.join(timeout=5.0)
        answer_t.join(timeout=5.0)
        self.assertEqual(
            violations, [],
            "box.title/status changed while another thread held the lock",
        )


class TestF6InterruptWorker(unittest.TestCase):
    """F6: async-exception injection must honour the API's return value."""

    def _session(self) -> SteeringSession:
        agent = SorcarAgent("w2-interrupt-test")
        state = _RunningAgentState("w2-chat-int", "", agent=None)
        return SteeringSession(agent, state, "w2-chat-int")

    def test_live_worker_receives_keyboard_interrupt(self) -> None:
        session = self._session()
        interrupted = threading.Event()
        started = threading.Event()

        def work() -> None:
            started.set()
            try:
                while True:
                    time.sleep(0.005)
            except KeyboardInterrupt:
                interrupted.set()

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self.assertTrue(started.wait(timeout=5.0))
        session._interrupt_worker(worker)
        worker.join(timeout=5.0)
        self.assertTrue(interrupted.is_set())
        self.assertFalse(worker.is_alive())

    def test_dead_worker_is_a_fast_noop(self) -> None:
        """A worker that already exited must not wedge the 5 s join."""
        session = self._session()
        worker = threading.Thread(target=_noop_abort, daemon=True)
        worker.start()
        worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        t0 = time.monotonic()
        session._interrupt_worker(worker)  # must not raise
        self.assertLess(time.monotonic() - t0, 1.0)


class TestF13HistoryFormats(unittest.TestCase):
    """F13: readline and anchored history must not share one file."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-w2-hist-")
        self.base = Path(self.tmpdir) / "history"

    def test_readline_uses_dot_readline_sibling(self) -> None:
        """``_save_history``/``_setup_readline`` must use ``.readline``.

        Runs in a subprocess because the ``readline`` module holds
        process-global state.
        """
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "# Use the SAME readline module cli_repl resolved (it\n"
            "# prefers the gnureadline wheel over the stdlib module).\n"
            "from kiss.ui.cli import cli_repl\n"
            "from kiss.ui.cli.cli_repl import (\n"
            "    CliCompleter, _save_history, _setup_readline,\n"
            ")\n"
            "readline = cli_repl.readline\n"
            "if readline is None:\n"
            "    print('NO_READLINE'); raise SystemExit(0)\n"
            "base = Path(sys.argv[1])\n"
            "workdir = sys.argv[2]\n"
            "readline.clear_history()\n"
            "readline.add_history('w2 history entry')\n"
            "_save_history(base)\n"
            "print('BASE_EXISTS', base.exists())\n"
            "sib = base.with_name(base.name + '.readline')\n"
            "print('SIBLING_EXISTS', sib.exists())\n"
            "readline.clear_history()\n"
            "_setup_readline(CliCompleter(workdir, ''), base)\n"
            "print('RELOADED', readline.get_history_item(1))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script, str(self.base), self.tmpdir],
            capture_output=True, text=True, timeout=60,
            cwd=_REPO_SRC,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        if "NO_READLINE" in proc.stdout:
            self.skipTest("readline module unavailable")
        self.assertIn("BASE_EXISTS False", proc.stdout)
        self.assertIn("SIBLING_EXISTS True", proc.stdout)
        self.assertIn("RELOADED w2 history entry", proc.stdout)
        # The anchored REPL's view of the base path stays empty — the
        # readline format never contaminates it.
        self.assertEqual(_load_history_lines(self.base), [])

    def test_anchored_load_skips_legacy_libedit_header(self) -> None:
        """A pre-fix contaminated file must not surface the header."""
        self.base.write_text(
            "_HiStOrY_V2_\nfix\\040the\\040bug\nplain entry\n",
            encoding="utf-8",
        )
        lines = _load_history_lines(self.base)
        self.assertNotIn("_HiStOrY_V2_", lines)
        self.assertIn("plain entry", lines)

    def test_anchored_plain_roundtrip(self) -> None:
        history = ["first", "second one", "third"]
        _save_history_lines(self.base, history)
        self.assertEqual(_load_history_lines(self.base), history)


class TestF19NoImportTimeGlobalMutation(unittest.TestCase):
    """F19: importing cli_prompt must not strip prompt_toolkit globals."""

    SEQ = "\\x1b[27;2;13~"  # Shift+Enter (modifyOtherKeys)

    def test_import_leaves_ansi_sequences_intact(self) -> None:
        script = (
            "import json\n"
            "import kiss.ui.cli.cli_client  # noqa: F401\n"
            "import kiss.ui.cli.cli_prompt  # noqa: F401\n"
            "from prompt_toolkit.input.ansi_escape_sequences import (\n"
            "    ANSI_SEQUENCES,\n"
            ")\n"
            f"print(json.dumps({{'mapped': '{self.SEQ}' in ANSI_SEQUENCES}}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120, cwd=_REPO_SRC,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(
            payload["mapped"],
            "importing sorcar modules must no longer unmap "
            "modifier+Enter for other prompt_toolkit consumers",
        )

    def test_reader_construction_unmaps_for_sorcar_prompt(self) -> None:
        """Building sorcar's own reader still performs the unmap."""
        import pty

        out_file = Path(self.tmp_out.name)
        script = (
            "import sys, tempfile\n"
            "from pathlib import Path\n"
            "from kiss.ui.cli.cli_prompt import PtkLineReader\n"
            "from kiss.ui.cli.cli_repl import CliCompleter\n"
            "from prompt_toolkit.input.ansi_escape_sequences import (\n"
            "    ANSI_SEQUENCES,\n"
            ")\n"
            "d = tempfile.mkdtemp()\n"
            "PtkLineReader(CliCompleter(d, ''), Path(d) / 'hist')\n"
            "Path(sys.argv[1]).write_text(\n"
            f"    str('{self.SEQ}' not in ANSI_SEQUENCES)\n"
            ")\n"
        )
        master, slave = pty.openpty()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, str(out_file)],
                stdin=slave, stdout=slave, stderr=subprocess.PIPE,
                timeout=120, cwd=_REPO_SRC,
            )
        finally:
            os.close(master)
            os.close(slave)
        self.assertEqual(
            proc.returncode, 0, proc.stderr.decode(errors="replace")
        )
        self.assertEqual(out_file.read_text().strip(), "True")

    def setUp(self) -> None:
        self.tmp_out = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="kiss-w2-ptk-", delete=False
        )
        self.tmp_out.close()

    def tearDown(self) -> None:
        Path(self.tmp_out.name).unlink(missing_ok=True)


class _PrinterTestBase(unittest.TestCase):
    """Shared persistence + bridge redirection for printer tests."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-w2-printer-")
        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None
        self.sock_path = str(Path(self.tmpdir) / "w2.sock")
        self._saved_sock_env = os.environ.get("KISS_SORCAR_SOCK")
        os.environ["KISS_SORCAR_SOCK"] = self.sock_path
        _reset_bridge_writer()
        self.server = _UdsCaptureServer(self.sock_path)

    def tearDown(self) -> None:
        self.server.close()
        _reset_bridge_writer()
        if self._saved_sock_env is None:
            os.environ.pop("KISS_SORCAR_SOCK", None)
        else:
            os.environ["KISS_SORCAR_SOCK"] = self._saved_sock_env
        if _persistence._db_conn is not None:
            try:
                _persistence._db_conn.close()
            except Exception:  # pragma: no cover - cleanup best-effort
                pass
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db


class TestF21PrinterAtexitLeak(_PrinterTestBase):
    """F21: printers must be collectable; the exit safety net must stay."""

    def test_printer_is_garbage_collectable(self) -> None:
        from kiss.ui.cli.cli_printer import RecordingConsolePrinter

        printer = RecordingConsolePrinter()
        ref = weakref.ref(printer)
        del printer
        gc.collect()
        self.assertIsNone(
            ref(),
            "RecordingConsolePrinter must not be pinned in memory by "
            "its atexit registration",
        )

    def test_atexit_safety_net_still_sends_cli_task_end(self) -> None:
        """A CLI process dying mid-task must still emit cliTaskEnd."""
        task_id = uuid.uuid4().hex
        script = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "os.environ['KISS_SORCAR_SOCK'] = sys.argv[1]\n"
            "import kiss.agents.sorcar.persistence as _p\n"
            "kiss_dir = Path(sys.argv[2]) / '.kiss-child'\n"
            "kiss_dir.mkdir(parents=True, exist_ok=True)\n"
            "_p._KISS_DIR = kiss_dir\n"
            "_p._DB_PATH = kiss_dir / 'sorcar.db'\n"
            "_p._db_conn = None\n"
            "from kiss.ui.cli.cli_printer import (\n"
            "    RecordingConsolePrinter,\n"
            ")\n"
            "printer = RecordingConsolePrinter()\n"
            "printer.broadcast(\n"
            "    {'type': 'text', 'text': 'hi', 'taskId': sys.argv[3]}\n"
            ")\n"
            "# exit WITHOUT a result event: atexit must send cliTaskEnd\n"
        )
        proc = subprocess.run(
            [
                sys.executable, "-c", script,
                self.sock_path, self.tmpdir, task_id,
            ],
            capture_output=True, text=True, timeout=120, cwd=_REPO_SRC,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.server.wait_for(3)
        envelopes = self.server.snapshot()
        types = [e.get("type") for e in envelopes]
        self.assertIn("cliTaskStart", types)
        self.assertIn("cliTaskEnd", types)
        self.assertLess(
            types.index("cliTaskStart"), types.index("cliTaskEnd")
        )


class TestF22StartBeforeEvents(_PrinterTestBase):
    """F22: no task event may reach the daemon before its cliTaskStart."""

    def test_concurrent_first_broadcasts_ordered_after_start(self) -> None:
        from kiss.ui.cli.cli_printer import RecordingConsolePrinter

        printer = RecordingConsolePrinter()
        task_ids = [uuid.uuid4().hex for _ in range(15)]
        for task_id in task_ids:

            def blast(tid: str = task_id) -> None:
                time.sleep(random.uniform(0.0, 0.01))
                printer.broadcast(
                    {"type": "text", "text": "x", "taskId": tid}
                )

            threads = [
                threading.Thread(target=blast, daemon=True)
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
            printer.broadcast(
                {
                    "type": "result",
                    "text": "done",
                    "taskId": task_id,
                }
            )
        # start + 4 events + result event + end = 7 envelopes per task.
        self.server.wait_for(7 * len(task_ids))
        envelopes = self.server.snapshot()
        for task_id in task_ids:
            first_kind = None
            saw_start = False
            saw_end_before_start = False
            for env in envelopes:
                etype = env.get("type")
                inner = env.get("event")
                inner_tid = (
                    inner.get("taskId")
                    if isinstance(inner, dict)
                    else None
                )
                if etype == "cliTaskStart" and env.get("taskId") == task_id:
                    saw_start = True
                    if first_kind is None:
                        first_kind = "start"
                elif etype == "cliTaskEnd" and env.get("taskId") == task_id:
                    if not saw_start:
                        saw_end_before_start = True
                    if first_kind is None:
                        first_kind = "end"
                elif etype == "cliEvent" and inner_tid == task_id:
                    if first_kind is None:
                        first_kind = "event"
            self.assertEqual(
                first_kind, "start",
                f"task {task_id}: daemon observed a {first_kind!r} "
                "envelope before cliTaskStart",
            )
            self.assertFalse(saw_end_before_start)


class TestF23VerboseFlag(unittest.TestCase):
    """F23: ``-v`` works as a flag; explicit values are validated."""

    def test_bare_flag_and_default(self) -> None:
        parser = _build_arg_parser()
        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["-v"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_explicit_values(self) -> None:
        parser = _build_arg_parser()
        self.assertFalse(parser.parse_args(["--verbose", "false"]).verbose)
        self.assertFalse(parser.parse_args(["-v", "0"]).verbose)
        self.assertFalse(parser.parse_args(["--verbose", "No"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose", "true"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose", "YES"]).verbose)
        self.assertTrue(parser.parse_args(["-v", "on"]).verbose)

    def test_invalid_value_is_rejected_loudly(self) -> None:
        parser = _build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--verbose", "maybe"])

    def test_bare_flag_composes_with_other_args(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["-v", "-t", "do things"])
        self.assertTrue(args.verbose)
        self.assertEqual(args.task, "do things")


if __name__ == "__main__":
    unittest.main()

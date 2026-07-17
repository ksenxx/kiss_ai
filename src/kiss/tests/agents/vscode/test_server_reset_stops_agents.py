# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproduction: "Reset Server" + OK must stop running agents.

User-visible flow under test: the settings-panel "Reset Server" button
posts ``{type:'serverReset'}`` (after the OK confirmation), the daemon's
``RemoteAccessServer._handle_server_reset`` schedules a self-``SIGTERM``,
and the daemon's shutdown path (``start()``'s ``finally`` →
``_stop_active_agent_tasks``) must cooperatively stop every in-flight
agent worker thread before the process exits — persisting
``"Task interrupted by server restart/shutdown"`` rather than abandoning
the task.

Unlike the in-process tests in ``test_server_reset.py`` (which replace
the self-``SIGTERM`` with a recorder), this test spawns a **real**
``RemoteAccessServer.start()`` daemon in a child process, starts a real
in-flight agent worker thread inside it, drives the very same
``serverReset`` command a webview OK click produces over the real UDS
socket, and lets the real ``SIGTERM`` → graceful loop-shutdown →
shutdown-``finally`` machinery run to completion.

Assertions:
  1. the daemon process actually exits (the restart half of the reset);
  2. the agent worker STOPS running (heartbeat file goes quiet);
  3. the task row is persisted as interrupted-by-restart, not left
     running or at the "Agent Failed Abruptly" sentinel.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase

_CHILD_SCRIPT = r"""
import os
import queue
import sys
import threading
import time
from pathlib import Path

tmp = Path(sys.argv[1])
os.environ.setdefault("KISS_WORKDIR", "/tmp")
# "busy": also run an event-loop task that swallows every exception —
# the production failure mode (see kiss-web-stderr.log pid=71720): with
# many agents active the main thread spends its time inside coroutine /
# callback code, and the KeyboardInterrupt raised by the SIGTERM signal
# handler lands inside frames whose broad except/finally clauses swallow
# it, so the shutdown never begins and later SIGTERMs are ignored.
BUSY_LOOP = len(sys.argv) > 3 and sys.argv[3] == "busy"

import kiss.agents.sorcar.persistence as th

kiss_dir = tmp / ".kiss"
kiss_dir.mkdir(parents=True, exist_ok=True)
th._KISS_DIR = kiss_dir
th._DB_PATH = kiss_dir / "sorcar.db"
th._db_conn = None

from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

certfile = tmp / "cert.pem"
keyfile = tmp / "key.pem"
_generate_self_signed_cert(certfile, keyfile)

server = RemoteAccessServer(
    host="127.0.0.1",
    port=int(sys.argv[2]),
    certfile=str(certfile),
    keyfile=str(keyfile),
    use_tunnel=False,
    url_file=tmp / "remote-url.json",
    uds_path=tmp / "sorcar.sock",
)

vscode = server._vscode_server
tab_id = "reset-e2e-tab"
tab = vscode._get_tab(tab_id)
agent = WorktreeSorcarAgent("Sorcar VS Code")
tab.agent = agent
tab.chat_id = ""

heartbeat = tmp / "heartbeat.txt"
marker = tmp / "agent_exit.txt"
task_id_file = tmp / "task_id.txt"


def fake_run(**kwargs):
    # Register a real task_history row exactly like ChatSorcarAgent.run
    # does early on, then block WITHOUT polling the cooperative stop
    # event — mimicking an agent wedged inside a blocking LLM API call.
    agent.total_tokens_used = 1
    agent.budget_used = 0.0
    agent.step_count = 1
    agent._chat_id = agent._chat_id or "reset-e2e-chat"
    task_id, _ = th._add_task(
        kwargs.get("prompt_template", ""),
        chat_id=agent._chat_id,
        extra={
            "model": kwargs.get("model_name", ""),
            "work_dir": kwargs.get("work_dir", ""),
            "version": "test",
            "is_parallel": False,
            "is_worktree": False,
        },
    )
    agent._last_task_id = task_id
    task_id_file.write_text(str(task_id), encoding="utf-8")
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            heartbeat.write_text(repr(time.time()), encoding="utf-8")
            time.sleep(0.05)
        marker.write_text("TIMEOUT", encoding="utf-8")
    except BaseException as exc:  # noqa: BLE001 — record how we were stopped
        marker.write_text("INTERRUPTED:" + type(exc).__name__, encoding="utf-8")
        raise


agent.run = fake_run

tab.stop_event = threading.Event()
tab.user_answer_queue = queue.Queue()
worker = threading.Thread(
    target=vscode._run_task,
    args=({
        "type": "run",
        "prompt": "reset-server-e2e",
        "tabId": tab_id,
        "workDir": "/tmp",
        "useParallel": False,
        "useWorktree": False,
        "autoCommit": False,
    },),
    daemon=True,
)
tab.task_thread = worker
worker.start()

if BUSY_LOOP:
    import asyncio

    import kiss.server.web_server as web_server_mod

    # Keep the shutdown failsafe short so the test does not wait the
    # full production window when the loop is wedged.
    if hasattr(web_server_mod, "_SHUTDOWN_EXIT_FAILSAFE"):
        web_server_mod._SHUTDOWN_EXIT_FAILSAFE = 5.0

    async def _swallow_everything_forever():
        # Stand-in for the busy-daemon coroutine mix that ate the
        # signal handler's KeyboardInterrupt in production; also
        # swallows CancelledError so asyncio.run's task-cancellation
        # phase can never finish.
        while True:
            try:
                await asyncio.sleep(0)
            except BaseException:
                pass

    def _inject_busy_task():
        while server._loop is None:
            time.sleep(0.05)
        server._loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(_swallow_everything_forever()),
        )

    threading.Thread(target=_inject_busy_task, daemon=True).start()

# Blocks until the serverReset-triggered SIGTERM unwinds asyncio.run;
# the shutdown finally must stop the worker before this returns.
server.start()
"""


def _find_free_port() -> int:
    """Find an available TCP port for the child daemon's WSS listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


class TestServerResetStopsRunningAgents(TestCase):
    """The reset must stop in-flight agents before the daemon exits."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-reset-e2e-"))
        self.child_script = self.tmpdir / "child_daemon.py"
        self.child_script.write_text(_CHILD_SCRIPT, encoding="utf-8")
        self.child_log = self.tmpdir / "child.log"
        self.proc: subprocess.Popen[bytes] | None = None

    def tearDown(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wait_for(self, cond, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return
            time.sleep(0.1)
        log = ""
        if self.child_log.exists():
            log = self.child_log.read_text(encoding="utf-8", errors="replace")
        raise AssertionError(f"timed out waiting for {what}\nchild log:\n{log}")

    def _launch_child(self, busy_loop: bool) -> None:
        """Start the child daemon and wait for the agent to be in flight.

        Args:
            busy_loop: When True the child also schedules an event-loop
                task that swallows every exception (including
                CancelledError), reproducing the busy-daemon state in
                which the production reset failed to stop agents.
        """
        port = _find_free_port()
        argv = [
            sys.executable, str(self.child_script), str(self.tmpdir), str(port),
        ]
        if busy_loop:
            argv.append("busy")
        with self.child_log.open("wb") as log:
            self.proc = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(self.tmpdir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

        uds_path = self.tmpdir / "sorcar.sock"
        heartbeat = self.tmpdir / "heartbeat.txt"
        task_id_file = self.tmpdir / "task_id.txt"

        # The daemon is up (UDS bound) and the agent worker is running
        # (heartbeat file being rewritten, task row registered).
        self._wait_for(uds_path.exists, 30.0, "UDS socket to appear")
        self._wait_for(heartbeat.exists, 30.0, "agent heartbeat to start")
        self._wait_for(task_id_file.exists, 30.0, "task row to be registered")

    def _send_server_reset(self) -> None:
        """Drive EXACTLY what the webview OK button produces.

        Sends a ``serverReset`` command on the daemon's UDS socket and
        waits for the ``server-reset-restarting`` acknowledgement.
        """
        uds_path = self.tmpdir / "sorcar.sock"
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(10.0)
        client.connect(str(uds_path))
        try:
            client.sendall(json.dumps({"type": "serverReset"}).encode() + b"\n")
            # Read until the acknowledgement notification arrives so we
            # know the command was dispatched (not just buffered).
            buf = b""
            deadline = time.monotonic() + 10.0
            acked = False
            while time.monotonic() < deadline and not acked:
                try:
                    chunk = client.recv(65536)
                except TimeoutError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if (
                        msg.get("type") == "notification"
                        and msg.get("id") == "server-reset-restarting"
                    ):
                        acked = True
                        break
            self.assertTrue(acked, "serverReset was never acknowledged")
        finally:
            client.close()

    def _assert_agent_stopped_and_daemon_exited(self) -> None:
        """Assert the reset stopped the agent and exited the daemon."""
        heartbeat = self.tmpdir / "heartbeat.txt"
        task_id_file = self.tmpdir / "task_id.txt"

        # 1. The daemon process must exit (the restart half of a reset).
        assert self.proc is not None
        try:
            self.proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            log = self.child_log.read_text(encoding="utf-8", errors="replace")
            self.fail(
                "regression: daemon process did not exit after serverReset; "
                "the reset never actually restarts the server\n"
                f"child log:\n{log}"
            )

        # 2. The agent worker must have STOPPED: once the daemon is dead
        # the heartbeat file must go quiet.  (If agent threads survived
        # as another process, or the daemon lingered, this keeps ticking.)
        mtime_after_exit = heartbeat.stat().st_mtime
        time.sleep(1.0)
        self.assertEqual(
            heartbeat.stat().st_mtime,
            mtime_after_exit,
            "regression: agent heartbeat still ticking after server reset — "
            "the running agent was not stopped",
        )

        # 3. The task must be persisted as gracefully interrupted by the
        # restart — NOT left at the abrupt-kill sentinel and NOT still
        # marked running.  This is what distinguishes "the reset stopped
        # the agent" from "the process death happened to kill it".
        task_id = task_id_file.read_text(encoding="utf-8").strip()
        db = sqlite3.connect(str(self.tmpdir / ".kiss" / "sorcar.db"))
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT result FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
        db.close()
        self.assertIsNotNone(row, "task row vanished")
        log = ""
        if self.child_log.exists():
            log = self.child_log.read_text(encoding="utf-8", errors="replace")
        self.assertEqual(
            row["result"],
            "Task interrupted by server restart/shutdown",
            "regression: serverReset did not stop the running agent "
            f"gracefully; persisted result={row['result']!r}\nchild log:\n{log}",
        )

    def test_server_reset_stops_running_agent(self) -> None:
        """serverReset (the OK click) stops the agent and exits the daemon."""
        self._launch_child(busy_loop=False)
        self._send_server_reset()
        self._assert_agent_stopped_and_daemon_exited()

    def test_server_reset_stops_agent_when_loop_swallows_interrupts(
        self,
    ) -> None:
        """The reset must work even when the event loop is busy/wedged.

        Production repro (``kiss-web-stderr.log`` pid=71720, pid=1194):
        with many agents active, the ``KeyboardInterrupt`` raised by the
        SIGTERM signal handler landed inside coroutine/callback frames
        whose broad ``except``/``finally`` clauses swallowed it — the
        shutdown never began, every later SIGTERM (i.e. every further
        "Reset Server" click) was ignored because
        ``_shutdown_initiated`` had latched, and the running agents
        kept going until the daemon was SIGKILLed.

        The child daemon runs an event-loop task that swallows every
        exception (including ``CancelledError``).  The reset must STILL
        stop the in-flight agent, persist the interrupted result, and
        exit the process.
        """
        self._launch_child(busy_loop=True)
        self._send_server_reset()
        self._assert_agent_stopped_and_daemon_exited()

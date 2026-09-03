# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a stop that lands during task SETUP is acknowledged and labelled.

``_run_task`` wraps the whole run in an outer ``try``; a
``KeyboardInterrupt`` injected before the run reaches
``_run_task_inner``'s own ``try`` (agent-script overrides, state
resolution, the ``status`` broadcast, the inner prologue up to the
``Task started`` log line) is caught by that OUTER catch.  It used to
hard-code ``"Task stopped by user"`` there without calling
``_cancel_outcome``:

* the stop was never ACKNOWLEDGED (``AgentState.stop_acknowledged``
  stayed ``False``), so the Stop watchdog (``_force_stop_thread``)
  re-injected a second ``KeyboardInterrupt`` five seconds later into
  the cancellation handling itself, aborting the ``result`` broadcast;
* ``AgentState.interrupted_by_shutdown`` was ignored, so a graceful
  daemon shutdown was reported to the user as a user stop.

Everything here is real: a ``RemoteAccessServer`` serving its Unix
domain socket, ``run`` / ``stop`` commands sent as a client would, the
real worker thread, the real watchdog, the real shutdown routine
(``_stop_active_agent_tasks``).  The interrupt is made to land in the
prologue deterministically with a real :class:`logging.Handler` on the
runner's logger that parks the worker inside the production
``Task started`` log call; the same handler then parks the worker
inside the outer catch's own log call, holding the cancellation
handling past the watchdog's retry moment (6 s after Stop).

Why the interrupt cannot be aimed at ``apply_agent_overrides`` itself:
``execute_python_file`` and every getter call wrap ``BaseException``
into ``AgentFileError``, so an interrupt landing inside user script
code never reaches the outer catch as a ``KeyboardInterrupt``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.task_runner import _state_owns_thread
from kiss.server.web_server import RemoteAccessServer

# Watchdog schedule: first injection after 1 s, retry after 5 more.
_RETRY_MOMENT = 6.0
_USER_STOP_LABEL = "Task stopped by user"
_SHUTDOWN_LABEL = "Task interrupted by server restart/shutdown"
_PARK_LIMIT = 60.0


class _ParkingHandler(logging.Handler):
    """Real logging handler that parks the emitting thread on two records.

    1. The prologue's ``Task started: tab_id=<tab> ...`` record: the
       worker is parked here (with no release) so the stop's
       ``KeyboardInterrupt`` lands in the prologue, i.e. in
       ``_run_task``'s outer catch.
    2. The first record the outer catch logs for the tab after that:
       the worker is parked until :attr:`release_cancel` is set, which
       holds the cancellation handling past the watchdog's retry.

    Every ``KeyboardInterrupt`` that lands while parked is recorded in
    :attr:`interrupts` and re-raised, so the test can count injections.
    """

    def __init__(self, tab_id: str) -> None:
        super().__init__(level=logging.DEBUG)
        self.tab_id = tab_id
        self.started = threading.Event()
        self.cancel_logged = threading.Event()
        self.release_cancel = threading.Event()
        self.cancel_message = ""
        self.interrupts: list[str] = []
        self.messages: list[str] = []

    def handle(self, record: logging.LogRecord) -> bool:
        # No per-handler lock around emit(): the dispatch loop logs
        # "Stop requested" while the worker is parked inside emit(), and
        # the base class's lock would deadlock it.
        self.emit(record)
        return True

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if f"tab_id={self.tab_id}" not in msg:
            return
        self.messages.append(msg)
        if not self.started.is_set():
            if msg.startswith("Task started:"):
                self._park("Task started", self.started, threading.Event())
            return
        if not self.cancel_logged.is_set():
            self.cancel_message = msg
            self._park("cancellation handling", self.cancel_logged, self.release_cancel)

    def _park(
        self, where: str, arrived: threading.Event, release: threading.Event,
    ) -> None:
        arrived.set()
        deadline = time.monotonic() + _PARK_LIMIT
        try:
            while not release.wait(0.05):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"parked too long in {where}")
        except KeyboardInterrupt:
            self.interrupts.append(where)
            raise


class _UdsClient:
    """Newline-delimited JSON client of the server's Unix domain socket."""

    def __init__(self, sock_path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_path)
        self.events: list[dict[str, Any]] = []
        self._cond = threading.Condition()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        buf = b""
        while True:
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    with self._cond:
                        self.events.append(json.loads(line))
                        self._cond.notify_all()

    def send(self, cmd: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(cmd) + "\n").encode())

    def wait_for(
        self,
        event_type: str,
        tab_id: str,
        timeout: float = 30.0,
        **fields: Any,
    ) -> dict[str, Any]:
        """Return the first *event_type* event for *tab_id* matching *fields*."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for ev in self.events:
                    if (
                        ev.get("type") == event_type
                        and ev.get("tabId") == tab_id
                        and all(ev.get(k) == v for k, v in fields.items())
                    ):
                        return ev
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"no {event_type!r} for {tab_id!r}; got "
                        f"{[e.get('type') for e in self.events]}",
                    )
                self._cond.wait(remaining)

    def close(self) -> None:
        self.sock.close()


class TestSetupStopIsAcknowledgedAndLabelled(TestCase):
    """One injection, correct label, for a stop landing in setup."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-setup-stop-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.sock_path = str(self.tmp / "sorcar.sock")
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.remote = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=str(self.work_dir),
        )
        self.remote._printer._loop = self.loop
        self.remote._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.remote._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=10)
        self.client = _UdsClient(self.sock_path)
        self.logger = logging.getLogger("kiss.server.task_runner")
        self._saved_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.handler: _ParkingHandler | None = None

    def tearDown(self) -> None:
        if self.handler is not None:
            self.handler.release_cancel.set()
            self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._saved_level)
        self.client.close()
        self.loop.call_soon_threadsafe(self.uds_server.close)
        self.loop.call_soon_threadsafe(self.loop.stop)
        agent_state.agent_states.clear()

    def _start_parked_run(
        self, tab_id: str,
    ) -> tuple[_ParkingHandler, AgentState, threading.Thread]:
        """Send a real ``run`` and park its worker in the setup prologue."""
        handler = _ParkingHandler(tab_id)
        self.handler = handler
        self.logger.addHandler(handler)
        self.client.send({
            "type": "run",
            "prompt": f"setup-stop {tab_id}",
            "tabId": tab_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        })
        self.assertTrue(
            handler.started.wait(timeout=30), "worker never reached the prologue",
        )
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            self.assertIsNotNone(state)
            assert state is not None
            thread = state.task_thread
            self.assertIsNotNone(thread)
            assert thread is not None
        self.assertTrue(state.is_task_active)
        return handler, state, thread

    def _assert_run_ended(
        self,
        handler: _ParkingHandler,
        state: AgentState,
        thread: threading.Thread,
        tab_id: str,
        label: str,
    ) -> None:
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive(), "worker never finished")
        result = self.client.wait_for("result", tab_id)
        self.assertEqual(result["text"], label)
        self.assertFalse(result["success"])
        self.client.wait_for("status", tab_id, running=True)
        self.client.wait_for("status", tab_id, running=False)
        self.assertTrue(handler.cancel_message.startswith(label), handler.cancel_message)
        self.assertNotIn("Task setup failed", "\n".join(handler.messages))
        self.assertEqual(handler.interrupts, ["Task started"], "exactly one injection")
        self.assertFalse(state.stop_acknowledged, "flag is per run and reset")
        self.assertIsNone(state.task_thread)

    def test_user_stop_in_setup_is_acknowledged_once(self) -> None:
        tab_id = "setup-stop-user"
        handler, state, thread = self._start_parked_run(tab_id)

        self.client.send({"type": "stop", "tabId": tab_id})
        ack = self.client.wait_for("stop_ack", tab_id)
        t0 = time.monotonic()
        self.assertTrue(ack["accepted"])
        # The watchdog's first injection (1 s) lands in the prologue;
        # the outer catch then logs and is parked by the handler.
        self.assertTrue(
            handler.cancel_logged.wait(timeout=15),
            "the outer catch never logged the cancellation",
        )
        self.assertEqual(handler.interrupts, ["Task started"])
        # Just past the watchdog's retry moment the worker must still be
        # in its (parked) cancellation handling, un-reinterrupted.
        time.sleep(max(0.0, _RETRY_MOMENT + 0.5 - (time.monotonic() - t0)))
        self.assertTrue(thread.is_alive(), "cancellation handling was aborted")
        self.assertTrue(
            state.stop_acknowledged,
            "the outer catch must acknowledge the stop before anything "
            "that can block",
        )
        with agent_state.STATE_LOCK:
            self.assertFalse(_state_owns_thread(state, thread))
        self.assertEqual(
            handler.interrupts, ["Task started"],
            "BUG: the watchdog re-injected into the cancellation handling",
        )
        handler.release_cancel.set()
        self._assert_run_ended(handler, state, thread, tab_id, _USER_STOP_LABEL)

    def test_setup_error_is_still_reported_as_failure(self) -> None:
        """A non-interrupt setup failure keeps its diagnostic label."""
        tab_id = "setup-error"
        handler = _ParkingHandler(tab_id)
        self.handler = handler
        self.logger.addHandler(handler)
        self.client.send({
            "type": "run",
            "prompt": "broken agent script",
            "tabId": tab_id,
            "workDir": str(self.work_dir),
            "agentPath": str(self.tmp / "missing_agent.py"),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        })
        result = self.client.wait_for("result", tab_id)
        self.assertTrue(
            result["text"].startswith("Task failed: AgentFileError: "), result["text"],
        )
        self.client.wait_for("status", tab_id, running=False)
        self.assertIn("Task setup failed", "\n".join(handler.messages))
        self.assertEqual(handler.interrupts, [])

    def test_shutdown_in_setup_is_labelled_as_shutdown(self) -> None:
        tab_id = "setup-stop-shutdown"
        handler, state, thread = self._start_parked_run(tab_id)

        # The real graceful-shutdown routine: flags the state, sets the
        # stop event, injects once and joins the worker.
        shutdown = threading.Thread(
            target=self.remote._stop_active_agent_tasks,
            kwargs={"timeout": 40.0},
            daemon=True,
        )
        shutdown.start()
        self.assertTrue(
            handler.cancel_logged.wait(timeout=15),
            "the outer catch never logged the cancellation",
        )
        self.assertTrue(state.interrupted_by_shutdown)
        self.assertTrue(state.stop_acknowledged)
        handler.release_cancel.set()
        shutdown.join(timeout=40)
        self.assertFalse(shutdown.is_alive(), "shutdown routine did not return")
        self._assert_run_ended(handler, state, thread, tab_id, _SHUTDOWN_LABEL)

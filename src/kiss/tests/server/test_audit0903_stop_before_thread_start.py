# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (server-core): a Stop landing before ``thread.start()``.

``_cmd_run`` registers the run's :class:`AgentState` — worker thread
installed on ``state.task_thread`` — under ``_state_lock``, releases
the lock, updates the shared tab registry (a ``tabs.json`` disk write)
and broadcasts the ``clear`` event, and only THEN calls
``thread.start()``.  A ``stop`` processed in that window (e.g. the
abort cascade / stop-on-timeout frames of ``daemon_client.run``
arriving on another connection) found the thread with
``is_alive() == False``:

* ``_stop_task`` decided watchdog-arming with a raw
  ``task_thread.is_alive()`` — which had DRIFTED from the ident-aware
  :meth:`AgentState.thread_alive` the rest of the codebase uses for
  exactly this window (S3-05 / C-R4) — so no force-stop watchdog was
  ever started;
* only the cooperative stop event was set, which nothing in the run's
  untrusted setup code (agent-script getters, tools files) ever
  checks, so the run kept executing that code indefinitely and the
  client's stop confirmation wait starved.

(The fixed ``_force_stop_thread`` must also survive the unstarted
thread itself: ``Thread.join`` raises ``RuntimeError`` on a thread
that has not started.)

The register→start interleaving cannot be timed reliably through the
wire, so this test constructs it directly with the REAL objects: it
registers the state exactly as ``_cmd_run`` does (same fields, same
lock), calls the real ``_stop_task`` while the real worker thread is
created but not yet started — the wire-visible window — and only then
starts the thread on the real ``_run_task``.  The run parks inside a
real agent-script getter (untrusted setup code with no cooperative
checks); a run the stop fails to kill is released via a flag file in
``tearDown``, where the getter raises — so no LLM is ever invoked.

Branch-coverage notes:

* ``_force_stop_thread``'s 30-second give-up on a thread that NEVER
  starts while ownership is still held is unreachable without
  stalling ``thread.start()`` itself for 30 s; it stays uncovered by
  design (the ownership-dropped exit is covered below).
* The start-wait loop's ``still_owns is None`` arm is likewise
  uncovered: production always passes the ownership guard, and a
  guard-less direct call with an unstarted thread could only be
  produced by a caller that does not exist.
"""

from __future__ import annotations

import os
import queue
import tempfile
import textwrap
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter

_STOP_LABEL = "Task stopped by user"

_BLOCKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    _DIR = pathlib.Path(__file__).resolve().parent


    def get_prompt():
        \"\"\"Block until interrupted; raise if released or timed out.\"\"\"
        (_DIR / "entered").write_text("1", encoding="utf-8")
        deadline = time.time() + 60
        while time.time() < deadline:
            if (_DIR / "release").exists():
                raise RuntimeError("released before the stop landed")
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for the stop")
    """
)


class TestStopBeforeThreadStart(TestCase):
    """A stop in the register→start window must still kill the run."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit0903-prestart-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.work_dir)
        self.thread: threading.Thread | None = None

    def tearDown(self) -> None:
        (self.tmp / "release").write_text("1", encoding="utf-8")
        if self.thread is not None:
            self.thread.join(timeout=90)
        agent_state.agent_states.clear()

    def _event(
        self, event_type: str, tab_id: str, **fields: Any,
    ) -> dict[str, Any] | None:
        for ev in list(self.printer.emitted):
            if (
                ev.get("type") == event_type
                and ev.get("tabId") == tab_id
                and all(ev.get(k) == v for k, v in fields.items())
            ):
                return ev
        return None

    def _wait_event(
        self,
        event_type: str,
        tab_id: str,
        timeout: float,
        **fields: Any,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ev = self._event(event_type, tab_id, **fields)
            if ev is not None:
                return ev
            time.sleep(0.02)
        return None

    def test_stop_in_prestart_window_still_stops_the_run(self) -> None:
        tab_id = "prestart-tab"
        cmd: dict[str, Any] = {
            "type": "run",
            "prompt": "prestart stop",
            "tabId": tab_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
            "agentPath": str(self.script),
        }
        # Register the run EXACTLY as ``_cmd_run`` does, up to (and
        # excluding) ``thread.start()`` — the wire-visible window the
        # registry write and the ``clear`` broadcast keep open.
        state_key = uuid.uuid4().hex
        cmd["_state_key"] = state_key
        state = AgentState(
            state_key,
            chat_id=uuid.uuid4().hex,
            tab_id=tab_id,
            conn_id="",
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        thread = threading.Thread(
            target=self.server._run_task, args=(cmd,), daemon=True,
        )
        self.thread = thread
        state.task_thread = thread
        with self.server._state_lock:
            agent_state.register(state)

        # The stop lands NOW — thread created, not started.
        self.server._stop_task(tab_id)
        ack = self._event("stop_ack", tab_id)
        self.assertIsNotNone(ack, "stop was not acknowledged at all")
        assert ack is not None
        self.assertTrue(ack["accepted"])
        assert state.stop_event is not None
        self.assertTrue(state.stop_event.is_set())

        # ``_cmd_run`` starts the thread; the run enters the agent
        # script's blocking getter, which never checks the cooperative
        # stop event — only the watchdog's injection can kill it.
        thread.start()
        result = self._wait_event("result", tab_id, timeout=12.0)
        self.assertIsNotNone(
            result,
            "BUG: the stop armed no watchdog for the not-yet-started "
            "thread, so the run kept executing untrusted setup code",
        )
        assert result is not None
        self.assertEqual(result["text"], _STOP_LABEL)
        self.assertFalse(result["success"])
        status_end = self._wait_event(
            "status", tab_id, timeout=10.0, running=False,
        )
        self.assertIsNotNone(status_end, "no terminal status broadcast")
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())

    def test_watchdog_exits_when_ownership_drops_before_start(self) -> None:
        """``_force_stop_thread`` on a never-started, disowned thread.

        ``thread.start()`` can raise in ``_cmd_run``, whose except
        clears ``state.task_thread`` — dropping ownership.  The
        watchdog must then exit instead of joining (``Thread.join``
        raises on an unstarted thread) or waiting out its grace.
        """
        state = AgentState(
            uuid.uuid4().hex,
            tab_id="disowned-tab",
            server_owned=True,
            stop_event=threading.Event(),
        )
        never_started = threading.Thread(target=lambda: None, daemon=True)
        state.task_thread = None  # ownership already dropped
        with self.server._state_lock:
            agent_state.register(state)

        def _still_owns() -> bool:
            return (
                agent_state.agent_states.get(state.task_id) is state
                and state.task_thread is never_started
                and not state.stop_acknowledged
            )

        t0 = time.monotonic()
        self.server._force_stop_thread(never_started, _still_owns)
        self.assertLess(
            time.monotonic() - t0,
            5.0,
            "watchdog neither exited on dropped ownership nor "
            "survived the unstarted thread",
        )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""C-RC2: ``_run_task``'s try/finally must cover the ENTIRE body.

Before the fix, the agent-script override, ``_resolve_run_state``,
the registry re-pin, and the ``clear`` re-announcement all executed
BEFORE the ``try:`` in ``_run_task``.  ``_stop_task``'s watchdog
injects a ``KeyboardInterrupt`` into the worker thread after a 1s
join, so a KI landing in that pre-try window unwound the worker
without broadcasting ``running=False`` and without clearing
``state.task_thread``.  ``_cmd_run`` gates on ``task_thread is not
None`` (S3-05 prompt queueing), so the tab was bricked — every later
submit was queued as steering for a task that no longer existed —
until a daemon restart.

The test makes the race deterministic with a delay hook (the
suite-standard KISS_RACE_DELAY style): ``_registry_update_tab`` — a
step that ran pre-try before the fix — is wrapped so the worker
signals arrival and then sleeps in small interruptible increments,
and the test injects the KeyboardInterrupt exactly there via
``PyThreadState_SetAsyncExc`` (what the stop watchdog does).

No mocks: a real ``VSCodeServer``, a real agent script file, a real
worker thread, real KI injection.
"""

from __future__ import annotations

import ctypes
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer

ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
    ctypes.c_ulong,
    ctypes.py_object,
]


class TestRunTaskGuardCoversWholeBody(unittest.TestCase):
    """A KI during run setup must still clear task_thread + broadcast."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-rr-c-rc2-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def recording_broadcast(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = recording_broadcast  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        def stub_run(self_agent: object, **kwargs: object) -> str:
            return "success: true\nsummary: ok\n"

        self._parent_class.run = stub_run

        # Delay hook: the worker signals arrival in the run-setup
        # region (pre-try before the fix) and sleeps interruptibly so
        # the injected KI lands exactly there.
        self.in_setup_region = threading.Event()
        self._orig_update_tab = self.server._registry_update_tab

        def slow_registry_update_tab(*args: Any, **kwargs: Any) -> Any:
            self.in_setup_region.set()
            for _ in range(100):
                time.sleep(0.05)
            return self._orig_update_tab(*args, **kwargs)

        self.server._registry_update_tab = slow_registry_update_tab  # type: ignore[assignment]

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _events_of_type(self, etype: str) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("type") == etype]

    def test_ki_during_setup_unbricks_tab(self) -> None:
        """KI in the setup region → running=False + task_thread cleared."""
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        # A real agent script whose get_prompt() override forces the
        # registry re-pin (the instrumented setup step) to run.
        script = Path(self.tmpdir) / "agent_script.py"
        script.write_text(
            "def get_prompt():\n"
            "    return 'overridden prompt'\n",
            encoding="utf-8",
        )
        tab_id = "rc2-tab"
        cmd = {
            "type": "run",
            "prompt": "original prompt",
            "tabId": tab_id,
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
            "agentPath": str(script),
        }
        self.server._cmd_run(dict(cmd))
        assert self.in_setup_region.wait(timeout=30), (
            "worker never reached the run-setup region"
        )
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.task_thread is not None
        worker = state.task_thread
        tid = worker.ident
        assert tid is not None
        # What _stop_task's watchdog does after its 1s join.
        rc = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(KeyboardInterrupt),
        )
        assert rc == 1, f"KI injection modified {rc} thread states"
        worker.join(timeout=30)
        assert not worker.is_alive(), "worker thread did not unwind"

        # The tab must NOT be bricked: task_thread cleared ...
        state_after = agent_state.find_by_tab(tab_id)
        assert state_after is None or state_after.task_thread is None, (
            "BUG C-RC2: KeyboardInterrupt during run setup left "
            "state.task_thread installed — the tab is bricked until "
            "daemon restart"
        )
        # ... and running=False broadcast exactly once for the tab.
        end_events = [
            e
            for e in self._events_of_type("status")
            if e.get("running") is False and e.get("tabId") == tab_id
        ]
        assert len(end_events) == 1, (
            f"BUG C-RC2: expected exactly one running=False status for "
            f"the tab, got {len(end_events)}"
        )
        # The interrupted setup must surface as a stopped-task result.
        results = [
            e
            for e in self._events_of_type("result")
            if e.get("tabId") == tab_id
        ]
        assert len(results) == 1 and results[0].get("success") is False

        # A follow-up submit on the same tab must START (not queue):
        # restore the fast registry update first.
        self.server._registry_update_tab = self._orig_update_tab  # type: ignore[assignment]
        cmd2 = dict(cmd)
        cmd2.pop("agentPath")
        self.server._cmd_run(cmd2)
        state2 = agent_state.find_by_tab(tab_id)
        assert state2 is not None, "second submit created no state"
        assert not state2.pending_user_messages, (
            "BUG C-RC2: second submit was queued as steering for a "
            "dead task instead of starting a fresh run"
        )
        thread2 = state2.task_thread
        assert thread2 is not None, "second submit started no worker"
        thread2.join(timeout=30)
        assert not thread2.is_alive(), "second run never finished"


if __name__ == "__main__":
    unittest.main()

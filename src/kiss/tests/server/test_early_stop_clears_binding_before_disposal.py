# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing before ``_run_task_inner``'s try must not leak the stop binding.

``_run_task_inner`` binds the run's stop event to the worker thread
(``printer._thread_local.stop_event``) and only then classifies the
task and retires a worktree an earlier run on the tab left pending.
Both steps precede the inner ``try`` whose cleanup clears that
binding, and both can be cut short by a Stop: the retire step raises
``KeyboardInterrupt("Stopped during worktree retire")`` itself.
``_run_task``'s own cleanup then runs the deferred disposal of a tab
the user closed meanwhile (``_dispose_if_closed``), on the SAME
thread, so the disposal's model-backed worktree retirement used to
run with the cancelled run's set stop signal still bound and could be
interrupted by it.  The outer clear right before ``_dispose_if_closed``
is what protects that path; this test pins it.

Real objects throughout: a real :class:`AgentServer`, the real
``_run_task`` on a real worker thread, a real :class:`ChatSorcarAgent`
subclass whose retire hook presses Stop and closes the tab the way a
user would mid-prologue, and whose disposal hook records the stop
binding the worker thread carries at that moment.  Classification is
switched off on the wire (``classifyTasks: False``) so no model is
ever called.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.tests.server._memory_printer import MemoryPrinter


class _StopDuringRetireAgent(ChatSorcarAgent):
    """Agent with a pending worktree whose pre-run retire is stopped and closed."""

    def __init__(self, server: AgentServer, tab_id: str) -> None:
        super().__init__("Early-stop agent")
        self._server = server
        self._tab = tab_id
        self._pending = True
        self.disposal_thread: threading.Thread | None = None
        self.disposal_stop_binding: threading.Event | None = None
        self.disposed = threading.Event()

    @property
    def _wt_pending(self) -> bool:
        return self._pending

    def _retire_after_task_saved(self) -> None:
        """The pre-run retire of the pending worktree: the user presses Stop and closes the tab."""
        self._server._stop_task(self._tab)
        self._server._close_tab(self._tab)

    def retire_for_disposal(self) -> bool:
        """The deferred disposal's retire: record the calling thread's stop binding."""
        self.disposal_thread = threading.current_thread()
        self.disposal_stop_binding = self._server.printer._thread_local.stop_event
        self._pending = False
        self.disposed.set()
        return True


class TestEarlyStopClearsBindingBeforeDisposal(unittest.TestCase):
    """Deferred disposal after an early Stop runs with no stop signal bound."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-early-stop-binding-"))
        self.printer = MemoryPrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = str(self.tmp)
        self.thread: threading.Thread | None = None

    def tearDown(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=30)
        self.printer._thread_local.stop_event = None
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wait_event(self, event_type: str, tab_id: str, **fields: Any) -> dict[str, Any] | None:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            for ev in list(self.printer.emitted):
                if (
                    ev.get("type") == event_type
                    and ev.get("tabId") == tab_id
                    and all(ev.get(k) == v for k, v in fields.items())
                ):
                    return ev
            time.sleep(0.02)
        return None

    def test_stop_during_worktree_retire_clears_binding_before_disposal(self) -> None:
        tab_id = "early-stop-tab"
        agent = _StopDuringRetireAgent(self.server, tab_id)
        cmd: dict[str, Any] = {
            "type": "run",
            "prompt": "stopped before the inner try",
            "tabId": tab_id,
            "workDir": str(self.tmp),
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
            "classifyTasks": False,
        }
        # Registered exactly as ``_cmd_run`` does, with the tab's
        # previous agent (pending worktree) carried over on the state.
        state_key = uuid.uuid4().hex
        cmd["_state_key"] = state_key
        state = AgentState(
            state_key,
            chat_id=uuid.uuid4().hex,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
            agent=agent,
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        thread = threading.Thread(target=self.server._run_task, args=(cmd,), daemon=True)
        self.thread = thread
        state.task_thread = thread
        with self.server._state_lock:
            agent_state.register(state)
        self.server.tab_registry.open_tab(tab_id, "early stop")
        run_stop_event = state.stop_event

        thread.start()
        result = self._wait_event("result", tab_id)
        self.assertIsNotNone(result, "the stopped run never reported a result")
        assert result is not None
        self.assertEqual(result["text"], "Task stopped by user")
        self.assertIsNotNone(self._wait_event("status", tab_id, running=False))
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())

        # The tab was closed while the run was busy, so its disposal
        # was deferred to the run's cleanup, on the worker thread.
        self.assertTrue(agent.disposed.is_set(), "the closed tab was never disposed")
        self.assertIs(agent.disposal_thread, thread)
        self.assertIsNotNone(run_stop_event)
        assert run_stop_event is not None
        self.assertTrue(run_stop_event.is_set())
        self.assertIsNone(
            agent.disposal_stop_binding,
            "BUG: the deferred disposal ran with the cancelled run's stop "
            "signal still bound to the worker thread, so its model-backed "
            "worktree retirement could be interrupted by that stale stop",
        )
        self.assertIsNone(agent_state.find_by_tab(tab_id), "disposed state still registered")


if __name__ == "__main__":
    unittest.main()

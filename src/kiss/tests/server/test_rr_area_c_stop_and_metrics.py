# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""C-R5 / C-R4 / C-R3: shared stop helper, liveness checks, one failure event.

* C-R5 — ``inject_keyboard_interrupt`` is the single shared
  ``PyThreadState_SetAsyncExc`` wrapper used by both the stop
  watchdog and shutdown; exercised here against a real thread.
* C-R4 — ``server.py``'s three inline ``task_thread is not None and
  is_alive()`` checks now use ``AgentState.thread_alive()``, which
  also counts the run-startup window (created-but-unstarted thread);
  exercised via ``_get_running_task_ids`` and
  ``_reattach_running_chat`` with real registry states.
* C-R3 — the two duplicated failure-``result`` payload constructions
  were merged into ``_broadcast_failure_result``; a real failing run
  must still produce exactly ONE failure result event.

The rc>1 branch of ``inject_keyboard_interrupt`` (exception installed
in multiple interpreter states) cannot be reached without corrupting
interpreter internals, so it is intentionally untested.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.server.server as _server_module
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.kiss_error import KISSError
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import inject_keyboard_interrupt


class _ExplodingRecordingPrinter(JsonPrinter):
    """Real JsonPrinter whose post-allocation recording setup raises.

    Same pattern as the F14 suite: the raise propagates out of the
    real ``ChatSorcarAgent.run`` (after the task id is allocated),
    driving the task runner's per-subtask failure path without
    touching the agent stack.  A ``KISSError`` because that is the
    one exception type ``WorktreeSorcarAgent.run`` re-raises instead
    of stringifying.
    """

    def start_recording(self) -> None:
        raise KISSError("area-C recording backend down")


def _noop() -> None:
    """Target for never-started placeholder worker threads."""


class TestInjectKeyboardInterrupt(unittest.TestCase):
    """C-R5: the shared injection helper against real threads."""

    def test_injects_into_live_thread(self) -> None:
        """A live thread receives KeyboardInterrupt; rc == 1."""
        started = threading.Event()
        caught: list[BaseException] = []

        def victim() -> None:
            started.set()
            try:
                for _ in range(600):
                    time.sleep(0.05)
            except KeyboardInterrupt as ki:
                caught.append(ki)

        thread = threading.Thread(target=victim, daemon=True)
        thread.start()
        assert started.wait(timeout=10)
        tid = thread.ident
        assert tid is not None
        rc = inject_keyboard_interrupt(tid)
        assert rc == 1, f"expected one modified thread state, got {rc}"
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert len(caught) == 1
        assert isinstance(caught[0], KeyboardInterrupt)

    def test_dead_thread_returns_zero(self) -> None:
        """A finished thread's ident yields rc == 0 (nothing injected)."""
        thread = threading.Thread(target=_noop, daemon=True)
        thread.start()
        thread.join(timeout=10)
        tid = thread.ident
        assert tid is not None
        assert inject_keyboard_interrupt(tid) == 0


class TestThreadAliveSites(unittest.TestCase):
    """C-R4: the three server.py liveness sites via thread_alive()."""

    def setUp(self) -> None:
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_get_running_task_ids_counts_startup_window(self) -> None:
        """A created-but-unstarted worker counts as running."""
        starting = AgentState(
            "starting-task",
            tab_id="tab-a",
            server_owned=True,
            task_thread=threading.Thread(target=_noop, daemon=True),
        )
        finished = AgentState(
            "finished-task", tab_id="tab-b", server_owned=True,
        )
        agent_state.register(starting)
        agent_state.register(finished)
        running = self.server._get_running_task_ids()
        assert "starting-task" in running, (
            "BUG C-R4: a task in the run-startup window (thread "
            "installed, not yet started) was not counted as running"
        )
        assert "finished-task" not in running

    def test_reattach_finds_task_in_startup_window(self) -> None:
        """A history-resume click during the startup window attaches."""
        state = AgentState(
            "startup-task",
            chat_id="chat-r4",
            tab_id="tab-c",
            server_owned=True,
            task_thread=threading.Thread(target=_noop, daemon=True),
        )
        agent_state.register(state)
        # By task id.
        assert self.server._reattach_running_chat(
            "", "viewer-1", task_id="startup-task",
        ), "BUG C-R4: startup-window task not reattachable by task id"
        # By chat id.
        assert self.server._reattach_running_chat(
            "chat-r4", "viewer-2",
        ), "BUG C-R4: startup-window task not reattachable by chat id"
        with self.server.printer._lock:
            viewers = self.server.printer._subscribers.get(
                "startup-task", set(),
            )
        assert {"viewer-1", "viewer-2"} <= viewers

    def test_reattach_ignores_finished_task(self) -> None:
        """A state with no thread and no active flag is not live."""
        state = AgentState(
            "done-task", chat_id="chat-done", tab_id="tab-d",
            server_owned=True,
        )
        agent_state.register(state)
        assert not self.server._reattach_running_chat(
            "", "viewer-3", task_id="done-task",
        )
        assert not self.server._reattach_running_chat(
            "chat-done", "viewer-4",
        )


class TestSingleFailureResultBroadcast(unittest.TestCase):
    """C-R3: a failing run emits exactly one failure result event."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-rr-c-r3-")
        self.server = VSCodeServer(printer=_ExplodingRecordingPrinter())
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def recording_broadcast(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = recording_broadcast  # type: ignore[assignment]

        self._orig_followup = _server_module.generate_followup_text

        def fake_followup(task: str, result: str, model: str) -> str:
            return ""

        _server_module.generate_followup_text = fake_followup  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        def stub_run(self_agent: object, **kwargs: object) -> str:
            return "success: true\nsummary: ok\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _server_module.generate_followup_text = self._orig_followup
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_and_collect_results(
        self, tab_id: str, extra_cmd: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        work_dir = str(Path(self.tmpdir) / f"wd-{tab_id}")
        Path(work_dir).mkdir()
        cmd: dict[str, Any] = {
            "type": "run",
            "prompt": "area-C failure-path task",
            "tabId": tab_id,
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        }
        cmd.update(extra_cmd or {})
        self.server._cmd_run(cmd)
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.task_thread is not None
        state.task_thread.join(timeout=60)
        with self._events_lock:
            return [e for e in self.events if e.get("type") == "result"]

    def test_subtask_failure_emits_one_result(self) -> None:
        """Per-subtask failure path: one result event, full payload."""
        results = self._run_and_collect_results("r3-tab")
        assert len(results) == 1, (
            f"expected exactly one terminal result event, got {results}"
        )
        result = results[0]
        assert result.get("success") is False
        assert "area-C recording backend down" in str(result.get("text"))
        # taskId-vs-tabId fallback: whichever key is present must
        # address this run.
        assert result.get("taskId") or result.get("tabId") == "r3-tab"
        for key in ("total_tokens", "cost", "step_count"):
            assert key in result, f"failure result lost field {key!r}"

    def test_outer_failure_emits_one_result(self) -> None:
        """Outer catch-all path (broken tools file, pre-loop failure)."""
        broken_tools = Path(self.tmpdir) / "broken_tools.py"
        broken_tools.write_text(
            "raise RuntimeError('broken tools import')\n",
            encoding="utf-8",
        )
        results = self._run_and_collect_results(
            "r3-outer-tab", {"toolsFile": str(broken_tools)},
        )
        assert len(results) == 1, (
            f"expected exactly one terminal result event, got {results}"
        )
        result = results[0]
        assert result.get("success") is False
        assert "Task failed" in str(result.get("text"))
        # No task id was ever allocated on this path, so the event
        # must fall back to the launcher tab id.
        assert result.get("tabId") == "r3-outer-tab"
        for key in ("total_tokens", "cost", "step_count"):
            assert key in result, f"failure result lost field {key!r}"


if __name__ == "__main__":
    unittest.main()

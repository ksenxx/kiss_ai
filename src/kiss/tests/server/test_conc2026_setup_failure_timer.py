# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A setup failure still broadcasts its result when no timer thread can start.

Concurrency audit 2026 (C3): ``_TaskRunnerMixin._run_task``'s
setup-failure handler records the early failure
(``ensure_recording_for_task``) and arms a 300 s ``threading.Timer``
to release that recording — but the ``Timer.start()`` was unguarded.
Under thread exhaustion (``RuntimeError: can't start new thread``,
the regime the sibling run/stop paths already survive — see
``test_concaudit_f2_stop_watchdog_spawn_failure.py`` and
``test_concaudit_f3_run_spawn_failure_status.py``) the raise escaped
the ``except`` block, so ``self.printer.broadcast(setup_result)``
never ran: the tab's spinner stopped (the ``finally`` still emitted
``status running=False``) but NO failure ``result`` was ever shown,
and the recording leaked permanently (the timer was its only
releaser).

Reproduced for real: a run whose ``agentPath`` names a broken agent
script (so the REAL ``_run_task`` takes the setup-failure path), with
``RLIMIT_NPROC`` lowered to 1 so ``Timer.start()`` genuinely fails.
No mocks or patches.
"""

from __future__ import annotations

import resource
import tempfile
import unittest
from pathlib import Path

import pytest

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.test_concaudit_w6_commit_msg_claim import (
    _thread_start_can_be_starved,
)


class TestSetupFailureTimerSpawnFailure(unittest.TestCase):
    """The failure ``result`` broadcast is unconditional."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.addCleanup(agent_state.agent_states.clear)

    def test_setup_failure_result_survives_timer_spawn_failure(self) -> None:
        if not _thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        broken_script = Path(tmp.name) / "broken_agent.py"
        broken_script.write_text("def prompt(:\n", encoding="utf-8")
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        server.use_private_tab_registry(Path(tmp.name) / "tabs.json")
        tab_id = "TAB-C3-TIMER"
        state = AgentState(
            "c3-timer-task",
            tab_id=tab_id,
            server_owned=True,
        )
        agent_state.register(state)
        cmd = {
            "type": "run",
            "tabId": tab_id,
            "prompt": "hello",
            "workDir": tmp.name,
            "agentPath": str(broken_script),
            "_state_key": state.task_id,
        }
        # ``_run_task`` normally runs on a worker thread; calling it
        # inline is equivalent here and lets the NPROC limit apply to
        # exactly the failure-path ``Timer.start()``.
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
        raised: BaseException | None = None
        try:
            try:
                server._run_task(cmd)
            except BaseException as exc:  # noqa: BLE001 — the bug re-raised here
                raised = exc
        finally:
            resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))

        self.assertIsNone(
            raised,
            f"BUG: the timer spawn failure escaped the handler: {raised!r}",
        )
        results = [
            e for e in printer.emitted
            if e.get("type") == "result" and e.get("success") is False
        ]
        self.assertEqual(len(results), 1, printer.emitted)
        self.assertIn("Task failed", results[0]["text"])
        status_ends = [
            e for e in printer.emitted
            if e.get("type") == "status" and e.get("running") is False
        ]
        self.assertEqual(len(status_ends), 1, printer.emitted)
        # The tab is idle again.
        self.assertIsNone(state.task_thread)
        self.assertFalse(state.busy())

    def test_setup_failure_result_and_timer_on_the_normal_path(self) -> None:
        """Without exhaustion the same path still broadcasts one result."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        broken_script = Path(tmp.name) / "broken_agent.py"
        broken_script.write_text("def prompt(:\n", encoding="utf-8")
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        server.use_private_tab_registry(Path(tmp.name) / "tabs.json")
        tab_id = "TAB-C3-TIMER-OK"
        state = AgentState("c3-timer-task-ok", tab_id=tab_id, server_owned=True)
        agent_state.register(state)
        server._run_task({
            "type": "run",
            "tabId": tab_id,
            "prompt": "hello",
            "workDir": tmp.name,
            "agentPath": str(broken_script),
            "_state_key": state.task_id,
        })
        results = [
            e for e in printer.emitted
            if e.get("type") == "result" and e.get("success") is False
        ]
        self.assertEqual(len(results), 1, printer.emitted)
        self.assertIn("Task failed", results[0]["text"])


if __name__ == "__main__":
    unittest.main()

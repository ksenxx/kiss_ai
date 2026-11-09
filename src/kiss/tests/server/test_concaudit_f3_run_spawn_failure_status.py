# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``run`` whose worker thread cannot start must still end the tab's run.

Concurrency audit (F3, reviewer R3 finding 9): both frontends raise a
tab's "running" state the moment the user hits Enter and only a
terminal ``result`` + ``status running:false`` ever lower it — normally
emitted by ``_run_task``.  When ``Thread.start()`` itself raised
(``RuntimeError: can't start new thread`` under thread exhaustion),
``_cmd_run`` rolled the state back and re-raised without emitting
either event, so the tab stayed "running" until a daemon restart.

Thread exhaustion is produced for real by lowering ``RLIMIT_NPROC``
to 1 around the command (threads count against the limit for a
non-root uid).  No mocks or patches.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.conftest import (
    nproc_limit_lowered_to_one,
    thread_start_can_be_starved,
)
from kiss.tests.server._memory_printer import MemoryPrinter


class TestRunSpawnFailureEndsRun(unittest.TestCase):
    """The spawn failure emits the terminal result/status pair."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.addCleanup(agent_state.agent_states.clear)

    def test_failed_thread_start_emits_result_and_status_end(self) -> None:
        if not thread_start_can_be_starved():
            pytest.skip("RLIMIT_NPROC cannot starve Thread.start on this host")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        printer = MemoryPrinter()
        server = AgentServer(printer=printer)
        server.use_private_tab_registry(Path(tmp.name) / "tabs.json")
        tab_id = "TAB-F3-SPAWN"
        cmd = {
            "type": "run", "tabId": tab_id, "prompt": "hello",
            "workDir": tmp.name, "taskId": "client-run-token-1",
        }
        reporting = threading.Event()
        disposing = threading.Event()
        probed = threading.Event()
        disposal_probed = threading.Event()
        admission_open: list[bool] = []

        def try_admission() -> None:
            for entering, checked in ((reporting, probed), (disposing, disposal_probed)):
                if entering.wait(10):
                    acquired = server._state_lock.acquire(blocking=False)
                    admission_open.append(acquired)
                    if acquired:
                        server._state_lock.release()
                    checked.set()

        def at_failure_report(frame: FrameType, event: str, arg: Any) -> None:
            if event == "call" and frame.f_code is server._report_run_start_failure.__code__:
                reporting.set()
                assert probed.wait(10)
            elif event == "call" and frame.f_code is server._dispose_if_closed.__code__:
                disposing.set()
                assert disposal_probed.wait(10)

        probe = threading.Thread(target=try_admission)
        probe.start()
        sys.setprofile(at_failure_report)
        try:
            with nproc_limit_lowered_to_one(), self.assertRaises(RuntimeError):
                server._cmd_run(cmd)
        finally:
            sys.setprofile(None)
            reporting.set()
            disposing.set()
            probe.join(10)
        # Publish before admitting a successor, but release STATE_LOCK
        # before disposal can acquire the worktree's repo_lock.
        assert admission_open == [False, True], admission_open

        results = [
            e for e in printer.emitted
            if e.get("type") == "result" and e.get("tabId") == tab_id
        ]
        self.assertEqual(len(results), 1, printer.emitted)
        self.assertFalse(results[0]["success"])
        self.assertIn("can't start new thread", results[0]["text"])
        status_ends = [
            e for e in printer.emitted
            if e.get("type") == "status" and e.get("tabId") == tab_id
            and e.get("running") is False
        ]
        self.assertEqual(len(status_ends), 1, printer.emitted)
        self.assertEqual(status_ends[0].get("taskId"), "client-run-token-1")
        # The tab is idle again, so a later run is not queued behind a
        # thread that never existed.
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        self.assertIsNone(state.task_thread)
        self.assertFalse(state.busy())


if __name__ == "__main__":
    unittest.main()

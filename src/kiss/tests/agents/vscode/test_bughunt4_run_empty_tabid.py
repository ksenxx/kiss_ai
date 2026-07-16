# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: ``_cmd_run`` with an empty/missing tabId mints a phantom task.

``_cmd_run`` did ``tab_id = cmd.get("tabId", "")`` and then
unconditionally created and registered a ``_RunningAgentState`` keyed
by the empty string, broadcast a ``clear`` event for it, and started a
real task thread.  But every other code path treats an empty tab id as
"no tab": ``_stop_task`` returns immediately for an empty id,
``_cmd_close_tab`` guards ``if tab_id:``, and ``_dispose_if_closed``
returns for an empty id — so the phantom ``""`` entry can never be
stopped, closed, or disposed, and the spawned task is unstoppable.

Iteration 3 fixed this class of bug for ``newChat`` and
``selectModel``; ``run`` was left unguarded.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.server.server as _server_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.server import VSCodeServer


class TestRunEmptyTabId(unittest.TestCase):
    """A ``run`` command without a tabId must be dropped, not started."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt4-emptytab-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture_broadcast(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

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
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_cmd(self, cmd: dict[str, Any]) -> None:
        self.server._cmd_run(cmd)
        # Give any (buggy) spawned task thread time to register state.
        deadline = time.time() + 10
        while time.time() < deadline:
            tab = _RunningAgentState.running_agent_states.get("")
            if tab is None or tab.task_thread is None:
                break
            time.sleep(0.02)

    def test_missing_tab_id_is_dropped(self) -> None:
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        self._run_cmd({
            "type": "run",
            "prompt": "bughunt4 empty tab id task",
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })
        assert "" not in _RunningAgentState.running_agent_states, (
            "BUG: a run command without a tabId minted a phantom "
            "_RunningAgentState keyed by the empty string; _stop_task, "
            "_cmd_close_tab and _dispose_if_closed all ignore empty tab "
            "ids, so this entry (and its task) can never be stopped or "
            "disposed"
        )
        clears = [e for e in self.events if e.get("type") == "clear"]
        assert not clears, (
            "BUG: a run command without a tabId broadcast a clear event "
            "and started a task thread for a phantom tab"
        )

    def test_explicit_empty_tab_id_is_dropped(self) -> None:
        work_dir = str(Path(self.tmpdir) / "plain2")
        Path(work_dir).mkdir()
        self._run_cmd({
            "type": "run",
            "prompt": "bughunt4 explicit empty tab id task",
            "tabId": "",
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })
        assert "" not in _RunningAgentState.running_agent_states
        assert not [e for e in self.events if e.get("type") == "clear"]


if __name__ == "__main__":
    unittest.main()

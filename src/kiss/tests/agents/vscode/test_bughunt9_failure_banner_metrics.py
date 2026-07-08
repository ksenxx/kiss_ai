# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9: failure ``result`` banner reports cumulative agent counters.

``_run_task_inner`` persists per-task metrics as DELTAS against the
baselines captured just before each ``tab.agent.run`` call
(``sub_tokens_base`` / ``sub_cost_base`` / ``sub_steps_base`` — the
W2-F2 fix), because ``tab.agent`` can be REUSED across tasks on the
same tab: a task that ends with a pending worktree (merge/discard not
yet chosen) preserves the agent — and its cumulative
``total_tokens_used`` / ``budget_used`` / ``total_steps`` counters —
for the next task.

The per-subtask FAILURE broadcast inside the loop (and the outer
``except BaseException`` result broadcast) however reported the RAW
cumulative counters.  For a reused agent this leaked the PREVIOUS
task's tokens / cost / steps into the failure banner of the next task,
disagreeing with the (correct, delta-based) metrics persisted into the
same task's ``task_history.extra`` row and rendered in the history
sidebar.

This test drives the real ``VSCodeServer._run_task`` end-to-end with a
real ``WorktreeSorcarAgent`` whose counters start non-zero (simulating
the preserved-agent reuse) and whose ``run`` consumes a small known
delta before failing.  The broadcast ``result`` event must carry the
delta, not the cumulative total.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.vscode.server as _server_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.models.model_info import get_available_models


class TestFailureBannerMetricsAreDeltas(unittest.TestCase):
    """Failure ``result`` events must report per-task metric deltas."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt9-metrics-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._orig_followup = _server_module.generate_followup_text

        def fake_followup(task: str, result: str, model: str) -> str:
            return ""

        _server_module.generate_followup_text = fake_followup  # type: ignore[assignment]

    def tearDown(self) -> None:
        _server_module.generate_followup_text = self._orig_followup
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _result_events(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == "result"]

    def test_reused_agent_failure_banner_uses_deltas(self) -> None:
        available = get_available_models()
        if not available:
            self.skipTest("no models available in this environment")
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()

        tab = self.server._get_tab("metrics-tab")
        agent = tab.agent
        assert agent is not None
        # Simulate an agent preserved from a PREVIOUS task on this tab
        # (pending-worktree reuse): its cumulative counters are non-zero
        # before this task's run even starts.
        agent.total_tokens_used = 1000
        agent.budget_used = 1.0
        agent.total_steps = 50

        def failing_run(**kwargs: object) -> str:
            # This task consumes a small, known delta, then fails.
            assert agent is not None
            agent.total_tokens_used += 100
            agent.budget_used += 0.01
            agent.total_steps += 3
            raise RuntimeError("boom-metrics")

        agent.run = failing_run  # type: ignore[method-assign, assignment]

        self.server._run_task({
            "type": "run",
            "prompt": "bughunt9 metrics task",
            "tabId": "metrics-tab",
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": available[0],
        })

        results = self._result_events()
        assert results, "missing failure result broadcast"
        banner = results[-1]
        assert banner.get("success") is False
        assert "boom-metrics" in str(banner.get("text", ""))
        # The banner must report THIS task's consumption (the delta),
        # not the agent's cumulative lifetime counters.
        assert banner.get("total_tokens") == 100, (
            f"expected per-task delta 100 tokens, got "
            f"{banner.get('total_tokens')!r} (cumulative leak)"
        )
        assert banner.get("cost") == "$0.0100", (
            f"expected per-task delta cost $0.0100, got "
            f"{banner.get('cost')!r} (cumulative leak)"
        )
        assert banner.get("step_count") == 3, (
            f"expected per-task delta 3 steps, got "
            f"{banner.get('step_count')!r} (cumulative leak)"
        )


if __name__ == "__main__":
    unittest.main()

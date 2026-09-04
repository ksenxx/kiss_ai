# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Persisted task metrics must apply the same ``step_count`` fallback
as the failure ``result`` banner.

``_subtask_metrics`` (used by the per-subtask failure broadcasts)
reads the run's raw per-run counters and falls back to the agent's
``step_count`` when ``total_steps`` is 0 — RelentlessAgent-derived
agents accumulate completed steps into ``total_steps`` and leave
``step_count`` at 0, while plain agents do the opposite.  The
PERSISTED metrics paths (the task-level cleanup ``finally`` in
``_run_task_inner`` and ``_persist_subtask_row``) must use the same
fallback, so a failed task run by a plain agent records the same step
count in its ``task_history`` row as the failure banner shows.

This test drives the real ``VSCodeServer._run_task`` end-to-end with a
real agent whose ``run`` bumps only ``step_count`` (leaving
``total_steps`` untouched) before failing, then asserts the persisted
history row's ``steps`` column equals the ``step_count`` the failure
banner broadcast.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.persistence import _add_task, _load_history
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.server import VSCodeServer

_TASK_PROMPT = "steps-fallback persisted metrics task"


class TestPersistedStepsFallback(unittest.TestCase):
    """History row ``steps`` must match the failure banner's count."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-steps-fallback-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]


    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_history_row_steps_match_failure_banner(self) -> None:
        available = get_available_models()
        if not available:
            self.skipTest("no models available in this environment")
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()

        agent = WorktreeSorcarAgent("Sorcar VS Code")
        state = agent_state.AgentState(
            "steps-fallback-state",
            agent=agent,
            tab_id="steps-fallback-tab",
            server_owned=True,
        )
        agent_state.register(state)

        def failing_run(**kwargs: object) -> str:
            task_id, chat_id = _add_task(_TASK_PROMPT)
            agent._last_task_id = task_id
            callback = kwargs.get("_on_task_id_allocated")
            assert callable(callback)
            callback(task_id, chat_id)
            agent.step_count = 3
            raise RuntimeError("boom-steps-fallback")

        agent.run = failing_run  # type: ignore[method-assign, assignment]

        self.server._run_task({
            "type": "run",
            "prompt": _TASK_PROMPT,
            "tabId": "steps-fallback-tab",
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": available[0],
        })

        banners = [e for e in self.events if e.get("type") == "result"]
        assert banners, "missing failure result broadcast"
        banner = banners[-1]
        assert banner.get("success") is False
        assert banner.get("step_count") == 3, (
            f"expected banner step_count 3 via step_count fallback, "
            f"got {banner.get('step_count')!r}"
        )

        rows = [e for e in _load_history() if e.get("task") == _TASK_PROMPT]
        assert rows, "failed task's history row was not persisted"
        row = rows[0]
        assert row.get("steps") == banner.get("step_count"), (
            f"persisted history row records {row.get('steps')!r} steps "
            f"but the failure banner showed {banner.get('step_count')!r} "
            f"— the persisted metrics path is missing the step_count "
            f"fallback"
        )


if __name__ == "__main__":
    unittest.main()

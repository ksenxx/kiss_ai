# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``WorktreeSorcarAgent.run`` resolves the run's model exactly once.

A run launched without an explicit model falls back to the user's
last-selected model, which the model picker rewrites at any time.
``WorktreeSorcarAgent.run`` classified the task with one read of that
preference and ``ChatSorcarAgent.run`` then read it AGAIN for the run
itself (the history row, the ``task_settings`` event and the agent's
model), so a pick landing between the two — the worktree setup sits in
between — made the classifier and the run disagree about the model.
The worktree agent now resolves once and hands the resolved name down.

Real agent, real classifier endpoint (a local stand-in model server),
real config file; no mocks.
"""

from __future__ import annotations

from typing import Any

import yaml

from kiss.agents.sorcar.persistence import (
    _load_history,
    _load_last_model,
    _save_last_model,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.agents.sorcar.test_task_classifier import (
    _DEV_TASK,
    _CountingClassifierEndpoint,
    _RaisingPrinter,
    env,  # noqa: F401 — pytest fixture
)
from kiss.tests.server.parallel_agent_harness import STANDIN_MODEL, IsolatedKissHome

_PICKED_LATER = "picked-after-classification"


class _PickerSwitchingPrinter(_RaisingPrinter):
    """Rewrites the last-selected model once the worktree exists.

    That moment lies between the worktree agent's classification and
    ``ChatSorcarAgent.run``, exactly where a model-picker write can land.
    """

    def broadcast(self, event: dict[str, Any]) -> None:
        super().broadcast(event)
        if event.get("type") == "worktree_created":
            _save_last_model(_PICKED_LATER)


def test_run_without_a_model_keeps_its_first_resolution(
    env: IsolatedKissHome,  # noqa: F811 — pytest fixture
) -> None:
    _save_last_model(STANDIN_MODEL)
    standin = _CountingClassifierEndpoint(is_simple=False, is_development=True)
    agent = WorktreeSorcarAgent("r2-model-resolved-once")
    printer = _PickerSwitchingPrinter(RuntimeError("stop-after-decision"))
    try:
        result = agent.run(
            prompt_template=_DEV_TASK,
            model_name=None,
            model_config=standin.model_config,
            work_dir=str(env.repo),
            printer=printer,
        )
        assert yaml.safe_load(result)["success"] is False, result
        assert len(printer.events_of_type("worktree_created")) == 1
        # The picker's write did land between the two phases...
        assert _load_last_model() == _PICKED_LATER
        # ...yet the classifier and the run agree on the first resolution.
        assert [request["model"] for request in standin.requests] == [STANDIN_MODEL]
        (row,) = _load_history()
        assert row["model"] == STANDIN_MODEL
    finally:
        agent.discard()
        standin.stop()

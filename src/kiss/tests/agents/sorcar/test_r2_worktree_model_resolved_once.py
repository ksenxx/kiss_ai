# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``ChatSorcarAgent.run`` resolves the run's model exactly once.

A run launched without an explicit model falls back to the user's
last-selected model, which the model picker rewrites at any time.
``ChatSorcarAgent.run`` classified the task with one read of that
preference and ``ChatSorcarAgent.run`` then read it AGAIN for the run
itself (the history row, the ``task_settings`` event and the agent's
model), so a pick landing between the two — while the classifier's
model call is in flight — made the classifier and the run disagree
about the model.  ``ChatSorcarAgent.run`` now resolves once and hands
the resolved name down to ``SorcarAgent.run``, which classifies with it.

Real agent, real classifier endpoint (a local stand-in model server),
real config file; no mocks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _load_history,
    _load_last_model,
    _save_last_model,
)
from kiss.tests.agents.sorcar.test_task_classifier import (
    _DEV_TASK,
    _CountingClassifierEndpoint,
    env,  # noqa: F401 — pytest fixture
)
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
)

_PICKED_LATER = "picked-after-classification"


class _RaisingToolsHook:
    """A SEA-style ``tools`` hook that raises instead of returning the toolset.

    Stops the run right after classification (the toolset is built
    after the verdict), so the test spends exactly one model call (the
    classifier's) instead of a whole run.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __call__(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        raise self._exc


class _PickerSwitchingEndpoint(_CountingClassifierEndpoint):
    """Rewrites the last-selected model while answering the classifier.

    That moment lies between ``ChatSorcarAgent.run``'s model resolution
    and the run proper, exactly where a model-picker write can land.
    """

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        body = super()._respond(request)
        _save_last_model(_PICKED_LATER)
        return body


def test_run_without_a_model_keeps_its_first_resolution(
    env: IsolatedKissHome,  # noqa: F811 — pytest fixture
) -> None:
    _save_last_model(STANDIN_MODEL)
    standin = _PickerSwitchingEndpoint(is_simple=False)
    agent = ChatSorcarAgent("r2-model-resolved-once")
    try:
        result = agent.run(
            prompt_template=_DEV_TASK,
            model_name=None,
            model_config=standin.model_config,
            work_dir=str(env.repo),
            printer=CapturePrinter(),
            tools_hook=_RaisingToolsHook(RuntimeError("stop-after-decision")),
        )
        assert yaml.safe_load(result)["success"] is False, result
        # The picker's write did land between the two phases...
        assert _load_last_model() == _PICKED_LATER
        # ...yet the classifier and the run agree on the first resolution.
        assert [request["model"] for request in standin.requests] == [STANDIN_MODEL]
        (row,) = _load_history()
        assert row["model"] == STANDIN_MODEL
    finally:
        standin.stop()
    # ``run`` creates no worktree by itself: that is the agent's
    # ``create_worktree`` tool, which the stopped run never reached.
    assert agent.worktree is None

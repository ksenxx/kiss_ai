# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The merge-conflict resolver bills the parent's epoch that launched it.

``run_merge_sea`` attributes the resolver's spend to the parent agent
when the child finishes.  An interactive merge can be resolved after
the parent has already started its NEXT task (``reset_usage`` swapped
the ledger epoch); like every other side channel (the task-update
child, dispatch fan-out) the attribution must be bound to the epoch
captured when the resolver started, so the old task's spend settles
in the discarded ledger instead of inflating the new task's budget.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server.merge_conflict_resolver import run_merge_sea
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    """Isolated ``KISS_HOME`` with the task classifier off."""
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "1"
    isolated = IsolatedKissHome("kiss-merge-epoch-")
    isolated.write_config()
    try:
        yield isolated
    finally:
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


def _parent_at(standin: StandInModelServer, repo: Path) -> ChatSorcarAgent:
    parent = ChatSorcarAgent("merge-parent")
    parent.work_dir = str(repo)
    parent.model_name = STANDIN_MODEL
    parent.model_config = standin.model_config
    return parent


def test_spend_lands_in_the_launching_epoch(home: IsolatedKissHome) -> None:
    """The parent resets for its next task mid-merge: that task is not billed."""
    parent: ChatSorcarAgent | None = None

    def respond(_request: dict[str, Any]) -> dict[str, Any]:
        # The resolver's one model call: meanwhile the parent starts
        # its next task, swapping its usage epoch.
        assert parent is not None
        parent.reset_usage()
        return finish_response("conflict resolved")

    standin = StandInModelServer(respond)
    try:
        parent = _parent_at(standin, home.repo)
        run_merge_sea(parent, "resolve f.txt", home.repo)
    finally:
        standin.stop()
    assert parent.usage_snapshot() == (0.0, 0, 0)


def test_spend_is_attributed_without_a_reset(home: IsolatedKissHome) -> None:
    """Without an intervening reset the resolver's steps reach the parent."""
    standin = StandInModelServer(lambda _req: finish_response("conflict resolved"))
    try:
        parent = _parent_at(standin, home.repo)
        run_merge_sea(parent, "resolve f.txt", home.repo)
    finally:
        standin.stop()
    _budget, tokens, steps = parent.usage_snapshot()
    assert steps >= 1 and tokens > 0

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 0903: WorktreeSorcarAgent.run's exception wrapping, both paths.

``WorktreeSorcarAgent.run`` carried two byte-identical
``try/except KISSError/except Exception`` blocks around its
``super().run`` delegation — one for the direct-execution fallback and
one for the worktree path.  Duplicated wrapping logic drifts (a future
edit fixing one copy silently leaves the other), so the two were merged
into a single block.  A pure duplication that has not drifted cannot be
reproduced by a failing behavior test; this test PINS the wrapping
contract on every path instead, so the de-duplication (and any future
change) is verified against real behavior:

* a non-``KISSError`` exception from the underlying run is wrapped into
  a ``success: false`` YAML result — on the direct path and on the
  worktree path alike;
* a ``KISSError`` propagates unwrapped on both paths;
* the worktree path still broadcasts ``worktree_created`` before
  delegating.

End-to-end: real agents, a real scratch git repo, a real server
printer.  The underlying run is made to fail by a printer whose
``agent_task_allocated`` hook raises — a real caller-owned collaborator
(the same channel VS Code's server uses), not a mock of the code under
test.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)


class _RaisingPrinter(CapturePrinter):
    """A real server printer whose task-allocation hook raises."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    def agent_task_allocated(self, agent: Any, task_id: Any, chat_id: str = "") -> None:
        super().agent_task_allocated(agent, task_id, chat_id)
        raise self._exc


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0903-run-wrapping-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _failure_summary(result: str) -> str:
    """Parse *result* as YAML and return its failure summary."""
    payload = yaml.safe_load(result)
    assert payload["success"] is False
    summary: str = payload["summary"]
    return summary


def test_direct_path_wraps_exception_into_failure_yaml(
    env: IsolatedKissHome,
) -> None:
    """Non-repo work_dir → direct execution; a crash becomes YAML."""
    agent = WorktreeSorcarAgent("audit0903-wrap-direct")
    with tempfile.TemporaryDirectory() as plain_dir:
        result = agent.run(
            prompt_template="crash please",
            work_dir=plain_dir,
            printer=_RaisingPrinter(RuntimeError("boom-direct")),
        )
    assert agent._wt is None
    assert "Task failed with error: boom-direct" in _failure_summary(result)


def test_direct_path_propagates_kiss_error(env: IsolatedKissHome) -> None:
    """A KISSError from the underlying run is never wrapped."""
    agent = WorktreeSorcarAgent("audit0903-wrap-kisserror")
    with tempfile.TemporaryDirectory() as plain_dir:
        with pytest.raises(KISSError, match="kiss-boom"):
            agent.run(
                prompt_template="crash please",
                use_worktree=False,
                work_dir=plain_dir,
                printer=_RaisingPrinter(KISSError("kiss-boom")),
            )


def test_worktree_path_wraps_exception_and_broadcasts_creation(
    env: IsolatedKissHome,
) -> None:
    """Repo work_dir → worktree path; same wrapping, after the
    worktree_created broadcast."""
    agent = WorktreeSorcarAgent("audit0903-wrap-worktree")
    printer = _RaisingPrinter(RuntimeError("boom-worktree"))
    result = agent.run(
        prompt_template="crash please",
        work_dir=str(env.repo),
        printer=printer,
    )
    assert "Task failed with error: boom-worktree" in _failure_summary(result)
    created = printer.events_of_type("worktree_created")
    assert len(created) == 1
    assert agent._wt is not None
    wt_dir = Path(created[0]["worktreeDir"])
    assert wt_dir == agent._wt.wt_dir
    # Clean up the pending worktree so the scratch repo tears down.
    agent.discard()
    assert agent._wt is None

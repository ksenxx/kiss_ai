# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``is_worktree`` must describe reality.

``ChatSorcarAgent.run`` had an ``explicit_worktree`` branch that
recorded ``is_worktree = bool(kwargs["use_worktree"])`` — i.e. it
believed the caller's *request* rather than what actually happened.
Nothing in the repository could reach it (``WorktreeSorcarAgent.run``
consumes the kwarg first, and ``SorcarAgent.run`` has no such
parameter), but a plain ``ChatSorcarAgent`` given the kwarg would
persist ``is_worktree: True`` for a run that never created a worktree,
so the history badge lied.  The containment check in the ``else``
branch is the ground truth for every reachable case.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.sorcar.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    history_rows,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f6-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


@pytest.mark.parametrize("use_worktree", [True, False, None])
def test_plain_agent_never_records_a_worktree_it_did_not_create(
    env: IsolatedKissHome, use_worktree: bool | None,
) -> None:
    """A worktree-less run is persisted as ``is_worktree = False``.

    Whatever the caller passes for ``use_worktree``, a plain
    ``ChatSorcarAgent`` creates no worktree, so the history row must
    say so.
    """

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        return finish_response("f6 done")

    server = StandInModelServer(responder)
    kwargs: dict[str, Any] = {}
    if use_worktree is not None:
        kwargs["use_worktree"] = use_worktree
    try:
        agent = ChatSorcarAgent("f6-plain")
        result = agent.run(
            prompt_template="F6 TASK",
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            **kwargs,
        )
    finally:
        server.stop()

    assert "success: true" in result, result
    rows = history_rows()
    assert len(rows) == 1
    assert rows[0]["is_worktree"] is False, (
        "the history row claims a worktree the run never created"
    )

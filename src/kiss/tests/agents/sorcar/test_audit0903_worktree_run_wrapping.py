# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 0903: SorcarAgent.run's exception wrapping.

The worktree agent's ``run`` once carried two byte-identical
``try/except KISSError/except Exception`` blocks around its
``super().run`` delegation — one for the direct-execution fallback and
one for the worktree path.  The worktree refactor left a single path
(the agent's ``create_worktree`` tool makes the worktree on demand, so
``run`` never creates one up front), now ``SorcarAgent.run`` itself;
this test PINS the wrapping contract of that path against real behavior:

* a non-``KISSError`` exception from the underlying run is wrapped into
  a ``success: false`` YAML result — in a plain directory and in a git
  repository alike;
* a ``KISSError`` propagates unwrapped;
* a repository ``work_dir`` does not make ``run`` create a worktree or
  broadcast ``worktree_created``.

End-to-end: real agents, a real scratch git repo, a real server
printer.  The underlying run is made to fail by a ``tools_hook`` that
raises — a real caller-owned collaborator (the channel a SEA's
``tools()`` uses), called inside ``SorcarAgent.run`` once the toolset
is built — not a mock of the code under test.  (A failure in
``ChatSorcarAgent.run``'s own setup, before the inner run, propagates:
the daemon's task runner reports it as ``Task failed: ...``.)
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core.kiss_error import KISSError
from kiss.tests.server.parallel_agent_harness import (
    CapturePrinter,
    IsolatedKissHome,
)


class _RaisingToolsHook:
    """A SEA-style ``tools`` hook that raises instead of returning the toolset."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __call__(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
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
    agent = ChatSorcarAgent("audit0903-wrap-direct")
    with tempfile.TemporaryDirectory() as plain_dir:
        result = agent.run(
            prompt_template="crash please",
            work_dir=plain_dir,
            printer=CapturePrinter(),
            tools_hook=_RaisingToolsHook(RuntimeError("boom-direct")),
        )
    assert agent._wt is None
    assert "Task failed with error: boom-direct" in _failure_summary(result)


def test_direct_path_propagates_kiss_error(env: IsolatedKissHome) -> None:
    """A KISSError from the underlying run is never wrapped."""
    agent = ChatSorcarAgent("audit0903-wrap-kisserror")
    with tempfile.TemporaryDirectory() as plain_dir:
        with pytest.raises(KISSError, match="kiss-boom"):
            agent.run(
                prompt_template="crash please",
                work_dir=plain_dir,
                printer=CapturePrinter(),
                tools_hook=_RaisingToolsHook(KISSError("kiss-boom")),
            )


def test_repo_work_dir_wraps_exception_without_a_worktree(
    env: IsolatedKissHome,
) -> None:
    """Repo work_dir → same wrapping; no worktree is created up front."""
    agent = ChatSorcarAgent("audit0903-wrap-worktree")
    printer = CapturePrinter()
    result = agent.run(
        prompt_template="crash please",
        work_dir=str(env.repo),
        printer=printer,
        tools_hook=_RaisingToolsHook(RuntimeError("boom-worktree")),
    )
    assert "Task failed with error: boom-worktree" in _failure_summary(result)
    assert printer.events_of_type("worktree_created") == []
    assert agent._wt is None
    assert not (env.repo / ".kiss-worktrees").exists()

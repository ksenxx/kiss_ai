# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (sorcar-agents): preserve-outcome warnings must match the cause.

``ChatSorcarAgent._commit_and_clean_worktree`` reports WHY a
worktree was preserved (``PRESERVED_COMMIT_FAILED``: the commit did
not go through), and ``_retire_previous_worktree`` and ``merge()``
turn that outcome into a matching message: a rejecting pre-commit hook
is the case that blames the hook.

These tests use real git repos and a real rejecting pre-commit hook.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0902-preserve-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo* and return stdout."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def _agent_with_worktree(env: IsolatedKissHome) -> ChatSorcarAgent:
    """Create an agent with a live worktree holding one uncommitted file."""
    agent = ChatSorcarAgent("audit0902-preserve")
    wt_work_dir = agent._try_setup_worktree(env.repo, str(env.repo))
    assert wt_work_dir is not None
    assert agent._wt is not None
    (agent._wt.wt_dir / "agent.txt").write_text("agent work\n", encoding="utf-8")
    return agent


def _install_rejecting_pre_commit_hook(repo: Path) -> Path:
    """Install a pre-commit hook that rejects every commit; return its path."""
    hooks = Path(_git(repo, "rev-parse", "--git-common-dir").strip())
    if not hooks.is_absolute():
        hooks = repo / hooks
    hook = hooks / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    return hook


def test_preserve_hook_wording_is_kept(
    env: IsolatedKissHome,
) -> None:
    """A rejecting pre-commit hook is the one case that SHOULD blame the hook."""
    agent = _agent_with_worktree(env)
    wt = agent._wt
    assert wt is not None
    hook = _install_rejecting_pre_commit_hook(env.repo)
    try:
        assert agent._retire_previous_worktree() is True
    finally:
        hook.unlink()
    assert wt.wt_dir.exists(), "worktree deleted with the only copy of the work"
    assert "pre-commit hook" in (agent._merge_conflict_warning or "")


def test_merge_still_blames_hook_when_commit_was_rejected(
    env: IsolatedKissHome,
) -> None:
    """``merge()`` keeps the pre-commit wording for a real hook rejection."""
    agent = _agent_with_worktree(env)
    wt = agent._wt
    assert wt is not None
    hook = _install_rejecting_pre_commit_hook(env.repo)
    try:
        message = agent.merge()
    finally:
        hook.unlink()
    assert agent._wt is wt
    assert "pre-commit hook" in message, message

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (sorcar-agents): preserve-outcome warnings must match the cause.

``WorktreeSorcarAgent._commit_and_clean_worktree`` reports WHY a
worktree was preserved (``PRESERVED_SUBAGENT_ACTIVE``,
``PRESERVED_RESCUE_FAILED``, ``PRESERVED_COMMIT_FAILED``,
``PRESERVED_NO_AUTOCOMMIT``).  ``_preserve_pending_worktree_for_review``
turns each outcome into a matching user warning, but the two other
callers collapsed the outcome to a bool:

* ``_release_worktree`` (auto-merge on new task / new chat) blamed a
  pre-commit hook for EVERY preserved outcome with Auto-commit on;
* ``merge()`` returned "auto-commit ... failed (a pre-commit hook may
  have rejected the commit)" for every preserved outcome.

So a user whose worktree was kept because an abandoned sub-agent was
still writing into it, or because a git-ignored output file could not
be rescued into the main repo, was told to go fix a pre-commit hook.

These tests use real git repos, a real ``ThreadPoolExecutor`` thread
standing in for the abandoned sub-agent, and a real symlinked directory
in the main repo to make the ignored-file rescue fail closed.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kiss.agents.sorcar import worktree_sorcar_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import _AbandonedSubagent
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _WorktreeCleanupOutcome,
)
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0902-preserve-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


@pytest.fixture
def short_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the abandoned-sub-agent grace wait so the tests stay fast."""
    monkeypatch.setattr(
        worktree_sorcar_agent, "_ABANDONED_SUBAGENT_WAIT_SECONDS", 0.2,
    )


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo* and return stdout."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def _agent_with_worktree(env: IsolatedKissHome) -> WorktreeSorcarAgent:
    """Create an agent with a live worktree holding one uncommitted file."""
    agent = WorktreeSorcarAgent("audit0902-preserve")
    agent.auto_commit_enabled = True
    wt_work_dir = agent._try_setup_worktree(env.repo, str(env.repo))
    assert wt_work_dir is not None
    assert agent._wt is not None
    (agent._wt.wt_dir / "agent.txt").write_text("agent work\n", encoding="utf-8")
    return agent


class _LiveChild:
    """A real thread that stands in for an abandoned sub-agent.

    ``_AbandonedSubagent`` holds a ``Future`` and the child's agent;
    the reclaim path only asks the future whether it is done and reads
    the agent's usage counters, so a real pool thread parked on an
    event plus a never-run ``ChatSorcarAgent`` reproduce the exact
    state a wedged child leaves behind.
    """

    def __init__(self, parent: WorktreeSorcarAgent) -> None:
        self.release = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(self._wait_for_release)
        self.agent = ChatSorcarAgent("audit0902-child")
        with parent._abandoned_lock:
            parent._abandoned_subagents.append(
                _AbandonedSubagent(self.future, self.agent, (0.0, 0, 0)),
            )

    def _wait_for_release(self) -> str:
        """Block like a running sub-agent until :meth:`finish` releases it."""
        self.release.wait(60)
        return ""

    def finish(self) -> None:
        """Let the thread exit and reclaim the pool."""
        self.release.set()
        self.pool.shutdown(wait=True)


def _make_rescue_fail(env: IsolatedKissHome, agent: WorktreeSorcarAgent) -> None:
    """Arrange a git-ignored task output that cannot be rescued.

    The main repo carries ``out`` as a symlink pointing OUTSIDE the
    repository, while the worktree creates a real ``out/`` directory
    with an ignored file in it.  ``rescue_ignored_files`` refuses to
    write through the escaping symlink and fails closed.
    """
    assert agent._wt is not None
    (env.repo / ".gitignore").write_text("out/\n", encoding="utf-8")
    _git(env.repo, "add", ".gitignore")
    _git(env.repo, "commit", "-q", "-m", "ignore out")
    outside = env.tmpdir / "outside"
    outside.mkdir()
    os.symlink(outside, env.repo / "out")
    wt_out = agent._wt.wt_dir / "out"
    wt_out.mkdir()
    (wt_out / "data.txt").write_text("ignored output\n", encoding="utf-8")
    # Make the worktree's .gitignore match the main repo's so the
    # file really is ignored there too.
    (agent._wt.wt_dir / ".gitignore").write_text("out/\n", encoding="utf-8")


def test_release_warning_names_live_subagent_not_hook(
    env: IsolatedKissHome, short_wait: None,
) -> None:
    """Auto-release preserved for a live sub-agent must say so."""
    agent = _agent_with_worktree(env)
    wt = agent._wt
    assert wt is not None
    child = _LiveChild(agent)
    try:
        released = agent._release_worktree()
    finally:
        child.finish()
    assert released is None
    assert wt.wt_dir.exists(), "worktree deleted under a live sub-agent"
    assert agent._last_preserve_outcome is (
        _WorktreeCleanupOutcome.PRESERVED_SUBAGENT_ACTIVE
    )
    warning = agent._merge_conflict_warning or ""
    assert "sub-agent" in warning, warning
    assert "pre-commit" not in warning, warning
    assert str(wt.wt_dir) in warning, warning


def test_release_warning_names_failed_rescue_not_hook(
    env: IsolatedKissHome,
) -> None:
    """Auto-release preserved for an unrescuable ignored file must say so."""
    agent = _agent_with_worktree(env)
    wt = agent._wt
    assert wt is not None
    _make_rescue_fail(env, agent)
    released = agent._release_worktree()
    assert released is None
    assert wt.wt_dir.exists(), "worktree deleted with the only copy of out/"
    assert agent._last_preserve_outcome is (
        _WorktreeCleanupOutcome.PRESERVED_RESCUE_FAILED
    )
    warning = agent._merge_conflict_warning or ""
    assert "ignored" in warning, warning
    assert "pre-commit" not in warning, warning
    assert str(wt.wt_dir) in warning, warning


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


def test_release_hook_and_autocommit_off_wordings_are_kept(
    env: IsolatedKissHome,
) -> None:
    """The two historical wordings survive for their own outcomes."""
    agent = _agent_with_worktree(env)
    wt = agent._wt
    assert wt is not None
    agent.auto_commit_enabled = False
    assert agent._release_worktree() is None
    assert agent._last_preserve_outcome is (
        _WorktreeCleanupOutcome.PRESERVED_NO_AUTOCOMMIT
    )
    assert "Auto-commit is turned off" in (agent._merge_conflict_warning or "")

    # A rejecting pre-commit hook is the one case that SHOULD blame the hook.
    agent2 = _agent_with_worktree(env)
    wt2 = agent2._wt
    assert wt2 is not None
    hook = _install_rejecting_pre_commit_hook(env.repo)
    try:
        assert agent2._release_worktree() is None
    finally:
        hook.unlink()
    assert agent2._last_preserve_outcome is (
        _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED
    )
    assert "pre-commit hook" in (agent2._merge_conflict_warning or "")


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
    assert agent._last_preserve_outcome is (
        _WorktreeCleanupOutcome.PRESERVED_COMMIT_FAILED
    )
    assert "pre-commit hook" in message, message
    assert "sub-agent" not in message and "ignored" not in message


def test_merge_reports_live_subagent_and_failed_rescue(
    env: IsolatedKissHome, short_wait: None,
) -> None:
    """``merge()`` must not blame a hook for the other preserve causes."""
    agent = _agent_with_worktree(env)
    wt = agent._wt
    assert wt is not None
    child = _LiveChild(agent)
    try:
        message = agent.merge()
    finally:
        child.finish()
    assert agent._wt is wt, "merge dropped the pending worktree"
    assert "sub-agent" in message, message
    assert "pre-commit" not in message, message

    _make_rescue_fail(env, agent)
    message = agent.merge()
    assert agent._wt is wt
    assert "ignored" in message, message
    assert "pre-commit" not in message, message

    # Once the cause is gone, the same merge succeeds.
    os.unlink(env.repo / "out")
    message = agent.merge()
    assert message.startswith("Successfully merged"), message
    assert agent._wt is None
    assert (env.repo / "agent.txt").read_text(encoding="utf-8") == "agent work\n"
    assert (env.repo / "out" / "data.txt").exists()

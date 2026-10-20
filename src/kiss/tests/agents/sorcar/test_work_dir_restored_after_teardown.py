"""``ChatSorcarAgent.work_dir`` leaves the worktree with the worktree.

The agent's ``create_worktree`` tool redirects ``work_dir`` into the
task's ``.kiss-worktrees/kiss_wt-*`` checkout.  Once the worktree is
torn down (retired for review on the next run, ``merge``, ``discard``,
or a server tab teardown) the attribute used to keep naming the
removed directory, and every later reader — the task-update side
channel, ``run_agent`` sub-tasks, the printer's image resolver — was
handed a path that no longer existed.

These tests drive the real teardown paths against real git
repositories and assert that ``work_dir`` points back at the parent
repository (keeping the sub-directory offset the task was launched
from) as soon as the worktree directory is gone, and is left alone on
every path that preserves the directory.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import sorcar_agent as wta_mod
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    tool_call_response,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _make_repo_with_worktree(root: Path) -> tuple[Path, Path, str]:
    """A ``main`` repo with a ``pkg/`` subdirectory and one ``kiss/wt-*`` worktree."""
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "mod.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    branch = "kiss/wt-1700000000-deadbeef"
    wt_dir = repo / ".kiss-worktrees" / "kiss_wt-1700000000-deadbeef"
    wt_dir.parent.mkdir()
    _git(repo, "worktree", "add", "-q", "-b", branch, str(wt_dir), "main")
    return repo.resolve(), wt_dir.resolve(), branch


def _agent_in_worktree(
    repo: Path, wt_dir: Path, branch: str, *, offset: str = "pkg",
) -> ChatSorcarAgent:
    """An agent whose last run lived in ``wt_dir/<offset>``."""
    agent = ChatSorcarAgent("test")
    wt_work_dir = wt_dir / offset if offset else wt_dir
    agent._wt = GitWorktree(
        repo_root=repo, branch=branch, original_branch="main",
        wt_dir=wt_dir, baseline_commit=None, work_dir=wt_work_dir,
    )
    agent.work_dir = str(wt_work_dir)
    return agent


@pytest.fixture
def no_llm_commit_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wta_mod, "_generate_commit_message",
        lambda commit_dir, user_prompt, task_result=None, bill_to=None: "test: work",
    )


class TestWorkDirRestoredWhenWorktreeRemoved:
    """Every path that deletes the worktree directory remaps ``work_dir``."""

    def test_finalize_restores_parent_repo_subdir(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        (wt_dir / "pkg" / "new.py").write_text("y = 2\n")
        agent = _agent_in_worktree(repo, wt_dir, branch)

        assert agent._finalize_worktree() is True

        assert not wt_dir.exists()
        assert agent.work_dir == str(repo / "pkg")
        assert Path(agent.work_dir).is_dir()

    def test_finalize_at_repo_root_restores_repo_root(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        agent = _agent_in_worktree(repo, wt_dir, branch, offset="")

        assert agent._finalize_worktree() is True

        assert agent.work_dir == str(repo)

    def test_retire_previous_worktree_restores_work_dir(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        """The retire-on-next-run path (``_retire_previous_worktree``).

        The pending work is committed to its own branch and kept for
        review, never merged; the directory is gone, so ``work_dir``
        goes back to the parent repository.
        """
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        (wt_dir / "pkg" / "new.py").write_text("y = 2\n")
        agent = _agent_in_worktree(repo, wt_dir, branch)

        agent._retire_previous_worktree()

        assert agent._wt is None
        assert not wt_dir.exists()
        assert agent.work_dir == str(repo / "pkg")
        assert not (repo / "pkg" / "new.py").exists()
        assert _git(repo, "show", f"{branch}:pkg/new.py") == "y = 2"

    def test_merge_restores_work_dir(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        (wt_dir / "pkg" / "new.py").write_text("y = 2\n")
        agent = _agent_in_worktree(repo, wt_dir, branch)

        agent.merge()

        assert agent._wt is None
        assert agent.work_dir == str(repo / "pkg")

    def test_discard_restores_work_dir(self, tmp_path: Path) -> None:
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        (wt_dir / "pkg" / "junk.py").write_text("z = 3\n")
        agent = _agent_in_worktree(repo, wt_dir, branch)

        message = agent.discard()

        assert message.startswith("Discarded branch")
        assert not wt_dir.exists()
        assert agent.work_dir == str(repo / "pkg")

    def test_vanished_worktree_dir_still_restores_work_dir(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        """The directory was already deleted out from under the agent."""
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        agent = _agent_in_worktree(repo, wt_dir, branch)
        shutil.rmtree(wt_dir)

        assert agent._finalize_worktree() is True

        assert agent.work_dir == str(repo / "pkg")
        assert "kiss_wt-" not in _git(repo, "worktree", "list")


class TestWorkDirUntouchedWhenWorktreeKept:
    """Preserved worktrees still exist, so ``work_dir`` must keep naming them."""

    def test_rejected_commit_preserve_keeps_work_dir(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        """A pre-commit hook refusing the commit keeps the only copy of the work."""
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        hook = repo / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        (wt_dir / "pkg" / "new.py").write_text("y = 2\n")
        agent = _agent_in_worktree(repo, wt_dir, branch)

        assert agent._finalize_worktree() is False

        assert wt_dir.exists()
        assert agent.work_dir == str(wt_dir / "pkg")

    def test_work_dir_outside_this_worktree_is_left_alone(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        """An agent that never ran (``work_dir == ""``) or whose run lived elsewhere."""
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        agent = _agent_in_worktree(repo, wt_dir, branch)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        agent.work_dir = str(elsewhere)

        assert agent._finalize_worktree() is True

        assert agent.work_dir == str(elsewhere)

        agent2 = _agent_in_worktree(repo, wt_dir, branch)
        agent2.work_dir = ""
        assert agent2._finalize_worktree() is True
        assert agent2.work_dir == ""


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-workdir-restore-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _create_worktree_then_finish(request: dict[str, Any]) -> dict[str, Any]:
    """Stand-in model: ``create_worktree`` first, then ``finish`` until it passes.

    The worktree guard refuses the first ``finish`` while the worktree
    is unmerged and lets the second through, so the run ends with the
    worktree pending — the state a tab teardown or discard acts on.
    """
    if any(m.get("role") == "tool" for m in request["messages"]):
        return finish_response("left the worktree pending")
    return tool_call_response("create_worktree", {})


def _run_in_worktree(agent: ChatSorcarAgent, repo: Path) -> Path:
    """Drive a real ``run`` on *repo*; return the worktree the agent made.

    ``RelentlessAgent._reset`` runs for real, so ``agent.work_dir`` is
    the value production code hands to later readers.  The stand-in
    model calls ``create_worktree`` and then finishes, leaving the
    worktree pending.
    """
    standin = StandInModelServer(_create_worktree_then_finish)
    try:
        result = agent.run(
            prompt_template="x",
            work_dir=str(repo),
            printer=CapturePrinter(),
            model_name=STANDIN_MODEL,
            model_config=standin.model_config,
            web_tools=False,
            is_parallel=False,
            verbose=False,
            max_steps=5,
            max_sub_sessions=1,
        )
    finally:
        standin.stop()
    assert "left the worktree pending" in result, result
    assert agent._wt is not None, "the agent did not set up a worktree"
    assert agent.work_dir == str(agent._wt.wt_dir)
    return agent._wt.wt_dir


class TestRunThenTeardownEndToEnd:
    """The full path: ``run`` → ``_reset`` → teardown → ``work_dir`` restored."""

    def test_discard_after_run(self, env: IsolatedKissHome) -> None:
        agent = ChatSorcarAgent("workdir-restore-discard")
        wt_dir = _run_in_worktree(agent, env.repo)

        agent.discard()

        assert not wt_dir.exists()
        assert agent.work_dir == str(env.repo.resolve())
        assert Path(agent.work_dir).is_dir()

    def test_retire_for_disposal_after_run(self, env: IsolatedKissHome) -> None:
        """The server's tab-teardown path publishes the work and drops the claim."""
        agent = ChatSorcarAgent("workdir-restore-retire")
        wt_dir = _run_in_worktree(agent, env.repo)

        assert agent.retire_for_disposal() is True

        assert agent._wt is None
        assert not wt_dir.exists()
        assert agent.work_dir == str(env.repo.resolve())

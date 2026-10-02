"""``WorktreeSorcarAgent.work_dir`` leaves the worktree with the worktree.

``WorktreeSorcarAgent.run`` redirects ``work_dir`` into the task's
``.kiss-worktrees/kiss_wt-*`` checkout and ``RelentlessAgent._reset``
stores that path in ``self.work_dir``.  Once the worktree is torn down
(auto-commit + squash-merge on the next run, ``merge``, ``discard``,
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

import subprocess
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import worktree_sorcar_agent as wta_mod
from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _AbandonedSubagent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.tests.server.parallel_agent_harness import CapturePrinter, IsolatedKissHome


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
) -> WorktreeSorcarAgent:
    """An agent whose last run lived in ``wt_dir/<offset>``."""
    agent = WorktreeSorcarAgent("test")
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
        lambda commit_dir, user_prompt, task_result=None: "test: work",
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

    def test_release_worktree_restores_work_dir(
        self, tmp_path: Path, no_llm_commit_message: None,
    ) -> None:
        """The automatic retire-on-next-run path (``_release_worktree``)."""
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        (wt_dir / "pkg" / "new.py").write_text("y = 2\n")
        agent = _agent_in_worktree(repo, wt_dir, branch)
        agent.auto_commit_enabled = True

        released = agent._release_worktree()

        assert released == "main"
        assert agent._wt is None
        assert agent.work_dir == str(repo / "pkg")
        assert (repo / "pkg" / "new.py").read_text() == "y = 2\n"

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
        subprocess.run(["rm", "-rf", str(wt_dir)], check=True)

        assert agent._finalize_worktree() is True

        assert agent.work_dir == str(repo / "pkg")
        assert "kiss_wt-" not in _git(repo, "worktree", "list")


class TestWorkDirUntouchedWhenWorktreeKept:
    """Preserved worktrees still exist, so ``work_dir`` must keep naming them."""

    def test_no_autocommit_preserve_keeps_work_dir(self, tmp_path: Path) -> None:
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        (wt_dir / "pkg" / "new.py").write_text("y = 2\n")
        agent = _agent_in_worktree(repo, wt_dir, branch)
        agent.auto_commit_enabled = False

        assert agent._finalize_worktree() is False

        assert wt_dir.exists()
        assert agent.work_dir == str(wt_dir / "pkg")

    def test_deferred_discard_keeps_work_dir_until_subagent_stops(
        self, tmp_path: Path,
    ) -> None:
        """A real abandoned sub-agent thread wedges the discard (5 s wait)."""
        repo, wt_dir, branch = _make_repo_with_worktree(tmp_path)
        agent = _agent_in_worktree(repo, wt_dir, branch)
        release = threading.Event()

        def wedged_child() -> str:
            release.wait(120)
            return "done"

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(wedged_child)
            with agent._abandoned_lock:
                agent._abandoned_subagents.append(
                    _AbandonedSubagent(future, SorcarAgent("child"), (0.0, 0, 0)),
                )
            try:
                message = agent.discard()
                assert message.startswith("Discard deferred")
                assert wt_dir.exists()
                assert agent.work_dir == str(wt_dir / "pkg")
            finally:
                release.set()
            assert agent.reclaim_abandoned_subagents(timeout=120)

        message = agent.discard()

        assert message.startswith("Discarded branch")
        assert not wt_dir.exists()
        assert agent.work_dir == str(repo / "pkg")

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


def _abort_before_llm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``llm_call_hook`` that ends the run before any network call."""
    raise RuntimeError("stop-before-llm")


def _run_in_worktree(agent: WorktreeSorcarAgent, repo: Path) -> Path:
    """Drive a real ``run`` on *repo*; return the worktree it set up.

    ``RelentlessAgent._reset`` runs for real, so ``agent.work_dir`` is
    the value production code hands to later readers.  The run is
    ended by the hook before the first LLM call.
    """
    agent.run(
        prompt_template="x",
        work_dir=str(repo),
        printer=CapturePrinter(),
        model_name="claude-haiku-4-5",
        llm_call_hook=_abort_before_llm,
        web_tools=False,
        is_parallel=False,
        verbose=False,
        max_steps=3,
    )
    assert agent._wt is not None, "run() did not set up a worktree"
    assert agent.work_dir == str(agent._wt.wt_dir)
    return agent._wt.wt_dir


class TestRunThenTeardownEndToEnd:
    """The full path: ``run`` → ``_reset`` → teardown → ``work_dir`` restored."""

    def test_discard_after_run(self, env: IsolatedKissHome) -> None:
        agent = WorktreeSorcarAgent("workdir-restore-discard")
        wt_dir = _run_in_worktree(agent, env.repo)

        agent.discard()

        assert not wt_dir.exists()
        assert agent.work_dir == str(env.repo.resolve())
        assert Path(agent.work_dir).is_dir()

    def test_retire_for_disposal_after_run(self, env: IsolatedKissHome) -> None:
        """The server's tab-teardown path publishes the work and drops the claim."""
        agent = WorktreeSorcarAgent("workdir-restore-retire")
        wt_dir = _run_in_worktree(agent, env.repo)
        agent.auto_commit_enabled = True

        assert agent.retire_for_disposal() is True

        assert agent._wt is None
        assert not wt_dir.exists()
        assert agent.work_dir == str(env.repo.resolve())

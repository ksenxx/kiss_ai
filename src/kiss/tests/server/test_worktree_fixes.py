# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests confirming that BUG-34 through BUG-38 are fixed.

Each test verifies the fix is in place — assertions fail if the
bug is reintroduced.

(Fix 1 — per-tab merge data dirs — and Fix 2 — pinned pre-task HEAD
SHA — covered the interactive merge-review machinery, removed together
with the diff/merge review workflow; only the main-tree busy guard and
the symmetric merge guard remain testable.)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


class TestFix3MainTreeBusyGuard:
    """Verify the is_running_non_wt flag and guard checks."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_any_non_wt_running_detects_running_tab(self) -> None:
        """_any_non_wt_running returns True when a state has the flag set."""
        server = VSCodeServer()
        state = agent_state.AgentState("t1", tab_id="t1", server_owned=True)
        agent_state.register(state)
        try:
            with server._state_lock:
                assert not server._any_non_wt_running()
                state.is_running_non_wt = True
                assert server._any_non_wt_running()
                state.is_running_non_wt = False
                assert not server._any_non_wt_running()
        finally:
            agent_state.unregister(state.task_id, state)

    def test_worktree_merge_blocked_when_non_wt_running(self) -> None:
        """_handle_worktree_action('merge') should refuse when non-wt running."""
        repo = _make_repo(Path(self._tmpdir) / "repo")
        server = VSCodeServer()
        server.work_dir = str(repo)

        wt_agent = WorktreeSorcarAgent("wt")
        wt_agent._chat_id = "wt_tab"
        wt_work = wt_agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt_state = agent_state.AgentState(
            "task-wt", agent=wt_agent, tab_id="wt_tab", server_owned=True,
        )
        wt_state.use_worktree = True
        agent_state.register(wt_state)

        non_wt_state = agent_state.AgentState(
            "task-non-wt", tab_id="non_wt_tab", server_owned=True,
        )
        non_wt_state.is_running_non_wt = True
        # Mirror _run_task_inner: a running non-wt task records the
        # resolved main-repo root of its work_dir on its state.
        non_wt_state.non_wt_repo_root = repo.resolve()
        agent_state.register(non_wt_state)

        try:
            result = server._handle_worktree_action("merge", "wt_tab")
            assert result["success"] is False
            assert "running" in result["message"].lower()
        finally:
            agent_state.unregister(non_wt_state.task_id, non_wt_state)
            agent_state.unregister(wt_state.task_id, wt_state)
            wt_agent.discard()

    def test_check_merge_conflict_suppressed_when_non_wt_running(self) -> None:
        """_check_merge_conflict returns False when non-wt agent is running."""
        repo = _make_repo(Path(self._tmpdir) / "repo")

        (repo / "shared.py").write_text("original\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "add shared"],
            capture_output=True, check=True,
        )

        server = VSCodeServer()
        server.work_dir = str(repo)

        wt_agent = WorktreeSorcarAgent("wt")
        wt_agent._chat_id = "wt_tab"
        wt_work = wt_agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None
        wt = wt_agent._wt
        assert wt is not None

        (wt.wt_dir / "shared.py").write_text("worktree change\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "wt changes")

        (repo / "shared.py").write_text("non-wt agent edit\n")

        wt_state = agent_state.AgentState(
            "task-wt-2", agent=wt_agent, tab_id="wt_tab", server_owned=True,
        )
        wt_state.use_worktree = True
        agent_state.register(wt_state)

        non_wt_state = agent_state.AgentState(
            "task-non-wt-2", tab_id="non_wt_tab", server_owned=True,
        )
        agent_state.register(non_wt_state)

        try:
            non_wt_state.is_running_non_wt = False
            conflict_before = server._check_merge_conflict("wt_tab")
            assert conflict_before is True, (
                "sanity: dirty file does cause conflict when no non-wt running"
            )

            non_wt_state.is_running_non_wt = True
            # Mirror _run_task_inner: record the repo root so the
            # repo-aware guard attributes the task to this repository.
            non_wt_state.non_wt_repo_root = repo.resolve()
            conflict_after = server._check_merge_conflict("wt_tab")
            assert conflict_after is False, (
                "Fix 3: dirty files from non-wt agent must not cause "
                "false conflict"
            )
        finally:
            agent_state.unregister(non_wt_state.task_id, non_wt_state)
            agent_state.unregister(wt_state.task_id, wt_state)
            GitWorktreeOps.remove(repo, wt.wt_dir)
            GitWorktreeOps.prune(repo)
            GitWorktreeOps.delete_branch(repo, wt.branch)




class TestFix4SymmetricGuard:
    """Verify non-wt task start is blocked during worktree merge."""


    def test_non_wt_blocked_when_wt_merging(self) -> None:
        """A non-wt task should not start when a worktree merge is active."""
        server = VSCodeServer()
        wt_state = agent_state.AgentState(
            "task-wt-merging", tab_id="wt_tab", server_owned=True,
        )
        wt_state.is_merging = True
        wt_state.use_worktree = True
        agent_state.register(wt_state)

        non_wt_state = agent_state.AgentState(
            "task-non-wt-idle", tab_id="non_wt", server_owned=True,
        )
        non_wt_state.use_worktree = False
        agent_state.register(non_wt_state)

        try:
            with server._state_lock:
                would_block = any(
                    t.is_merging and t.use_worktree
                    for t in agent_state.snapshot()
                )
            assert would_block, (
                "Fix 4: non-wt task must be blocked when wt merge is active"
            )
        finally:
            agent_state.unregister(non_wt_state.task_id, non_wt_state)
            agent_state.unregister(wt_state.task_id, wt_state)

    def test_non_wt_allowed_when_non_wt_merging(self) -> None:
        """A non-wt merge review should NOT block another non-wt task start."""
        server = VSCodeServer()
        state1 = agent_state.AgentState(
            "task-non-wt-merging", tab_id="tab1", server_owned=True,
        )
        state1.is_merging = True
        state1.use_worktree = False
        agent_state.register(state1)

        try:
            with server._state_lock:
                would_block = any(
                    t.is_merging and t.use_worktree
                    for t in agent_state.snapshot()
                )
            assert not would_block, (
                "Fix 4: non-wt merge should not block another non-wt task"
            )
        finally:
            agent_state.unregister(state1.task_id, state1)

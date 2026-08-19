# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests confirming bugs in the non-worktree workflow that make worktree mode unsafe.

Each test CONFIRMS the bug exists (assertions pass when buggy behaviour
is present).

BUG-37: Non-worktree agent's dirty files in the main repo cause
        _check_merge_conflict to report false-positive conflicts for
        worktree merges — the main-repo dirty-file listing (at the
        time, GitWorktreeOps.unstaged_files — since removed) counts the
        agent's in-progress writes as "user dirty state", and the
        overlap check triggers even though the dirty files are not
        the user's edits.

(BUG-34, BUG-35, BUG-36 and BUG-38 covered the pre-task snapshot and
interactive merge-review machinery — _parse_diff_hunks,
_prepare_merge_view, _snapshot_files, _merge_data_dir — all removed
together with the diff/merge review workflow.)
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
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


class TestBug37FalseConflictFromNonWorktreeAgent:
    """BUG-37: _check_merge_conflict lists the main repo's dirty files
    (historically via GitWorktreeOps.unstaged_files, since removed)
    and checks overlap with worktree changes.  If a non-worktree
    agent has edited files that the worktree also changed, the overlap
    check reports a conflict even though the "dirty" files are another
    agent's work, not the user's manual edits.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


    def test_false_conflict_from_non_worktree_agent_dirty_file(self) -> None:
        """BUG-37 functional: A file dirtied by a non-worktree agent causes
        _check_merge_conflict to report a false conflict for a worktree merge.
        """
        repo = _make_repo(Path(self._tmpdir) / "repo")

        (repo / "shared.py").write_text("original content\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."], capture_output=True, check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "add shared.py"],
            capture_output=True,
            check=True,
        )

        server = VSCodeServer()
        server.work_dir = str(repo)

        wt_agent = WorktreeSorcarAgent("wt_agent")
        wt_agent._chat_id = "wt_tab"
        wt_work = wt_agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None
        wt = wt_agent._wt
        assert wt is not None

        (wt.wt_dir / "shared.py").write_text("worktree modified content\n")
        GitWorktreeOps.commit_all(wt.wt_dir, "wt changes shared.py")

        (repo / "shared.py").write_text("non-wt agent modified content\n")

        state = agent_state.AgentState(
            "task-wt-audit8", agent=wt_agent, tab_id="wt_tab",
            server_owned=True,
        )
        state.use_worktree = True
        agent_state.register(state)

        try:
            has_conflict = server._check_merge_conflict("wt_tab")

            assert has_conflict is True, (
                "BUG-37 confirmed: non-worktree agent's dirty file causes "
                "false conflict detection for worktree merge"
            )
        finally:
            agent_state.unregister(state.task_id, state)
            GitWorktreeOps.remove(repo, wt.wt_dir)
            GitWorktreeOps.prune(repo)
            GitWorktreeOps.delete_branch(repo, wt.branch)

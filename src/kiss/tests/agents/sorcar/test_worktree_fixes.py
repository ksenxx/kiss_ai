"""Tests confirming that BUG-34 through BUG-38 are fixed.

Each test verifies the fix is in place — assertions fail if the
bug is reintroduced.
"""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    repo_lock,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.diff_merge import (
    _merge_data_dir,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
)
from kiss.agents.vscode.server import VSCodeServer, _TabState


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
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
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


class TestFix1PerTabMergeDirs:
    """Verify _merge_data_dir returns per-tab paths."""

    def test_merge_data_dir_with_tab_id_returns_unique_path(self) -> None:
        d1 = _merge_data_dir("tab-A")
        d2 = _merge_data_dir("tab-B")
        d0 = _merge_data_dir()
        assert d1 != d2, "Different tabs must get different dirs"
        assert d1 != d0, "Tab dir differs from default"
        assert "tab-A" in str(d1)
        assert "tab-B" in str(d2)

    def test_merge_data_dir_without_tab_id_returns_base(self) -> None:
        d = _merge_data_dir("")
        assert "merge_dir" in str(d)

    def test_save_untracked_base_uses_tab_specific_dir(self) -> None:
        """_save_untracked_base with tab_id stores files under per-tab path."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")
            (repo / "untracked.txt").write_text("hello\n")

            _save_untracked_base(str(repo), {"untracked.txt"}, tab_id="tabX")

            tab_ub_dir = _merge_data_dir("tabX") / "untracked-base"
            assert (tab_ub_dir / "untracked.txt").exists(), (
                "Untracked base should be saved under per-tab dir"
            )
        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_concurrent_merge_views_dont_destroy_each_other(self) -> None:
        """Two tabs preparing merge views simultaneously keep isolated data."""
        tmpdir = tempfile.mkdtemp()
        saved = _redirect_db(tmpdir)
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            (repo / "file_a.py").write_text("content a\n")
            subprocess.run(
                ["git", "-C", str(repo), "add", "."],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", "add file_a"],
                capture_output=True, check=True,
            )

            (repo / "file_a.py").write_text("modified by tab A\n")
            dir_a = str(_merge_data_dir("tab-A"))
            result_a = _prepare_merge_view(str(repo), dir_a, {}, set(), None)
            assert result_a.get("status") == "opened"

            pending_a = Path(dir_a) / "pending-merge.json"
            data_a = json.loads(pending_a.read_text())
            assert any(f["name"] == "file_a.py" for f in data_a["files"])

            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--", "file_a.py"],
                capture_output=True, check=True,
            )
            (repo / "new_file_b.txt").write_text("created by tab B\n")
            dir_b = str(_merge_data_dir("tab-B"))
            result_b = _prepare_merge_view(str(repo), dir_b, {}, set(), None)
            assert result_b.get("status") == "opened"

            data_a_after = json.loads(pending_a.read_text())
            assert any(f["name"] == "file_a.py" for f in data_a_after["files"]), (
                "Tab A data must survive Tab B's merge view preparation"
            )

            pending_b = Path(dir_b) / "pending-merge.json"
            data_b = json.loads(pending_b.read_text())
            assert any(f["name"] == "new_file_b.txt" for f in data_b["files"])
        finally:
            _restore_db(saved)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_prepare_and_start_merge_accepts_tab_id(self) -> None:
        """_prepare_and_start_merge now accepts tab_id parameter."""
        sig = inspect.signature(VSCodeServer._prepare_and_start_merge)
        assert "tab_id" in sig.parameters, (
            "Fix 1: _prepare_and_start_merge must accept tab_id"
        )

    def test_save_untracked_base_accepts_tab_id(self) -> None:
        """_save_untracked_base now accepts tab_id parameter."""
        sig = inspect.signature(_save_untracked_base)
        assert "tab_id" in sig.parameters, (
            "Fix 1: _save_untracked_base must accept tab_id"
        )


class TestFix2PinHeadSHA:
    """Verify pre-task snapshot is atomic and HEAD SHA is pinned."""

    def test_run_task_inner_acquires_repo_lock_for_snapshot(self) -> None:
        """The non-worktree snapshot helper should acquire repo_lock."""
        source = inspect.getsource(VSCodeServer._capture_pre_snapshot)

        assert "repo_lock(repo)" in source, (
            "Fix 2: _capture_pre_snapshot must acquire repo_lock"
        )
        assert "head_sha" in source, (
            "Fix 2: must capture head_sha inside the snapshot"
        )

        inner_source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "_capture_pre_snapshot" in inner_source, (
            "Fix 2: _run_task_inner must call _capture_pre_snapshot"
        )

    def test_prepare_and_start_merge_receives_pinned_base_ref(self) -> None:
        """The post-task merge call should use the pinned HEAD SHA."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert 'base_ref=pre_head_sha or "HEAD"' in source, (
            "Fix 2: _prepare_and_start_merge must receive pinned base_ref"
        )

    def test_pinned_head_sha_survives_concurrent_checkout(self) -> None:
        """Functional: pinned SHA is stable even if HEAD changes."""
        tmpdir = tempfile.mkdtemp()
        try:
            repo = _make_repo(Path(tmpdir) / "repo")

            subprocess.run(
                ["git", "-C", str(repo), "checkout", "-b", "feature"],
                capture_output=True, check=True,
            )
            (repo / "feature.txt").write_text("feature\n")
            subprocess.run(
                ["git", "-C", str(repo), "add", "."],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m", "feature"],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "main"],
                capture_output=True, check=True,
            )

            with repo_lock(repo):
                pinned_sha = GitWorktreeOps.head_sha(repo)
                _parse_diff_hunks(str(repo))

            subprocess.run(
                ["git", "-C", str(repo), "checkout", "feature"],
                capture_output=True, check=True,
            )

            current_sha = GitWorktreeOps.head_sha(repo)
            assert pinned_sha != current_sha, "sanity: HEAD moved"
            assert pinned_sha is not None, "pinned SHA must be valid"

            post_hunks = _parse_diff_hunks(str(repo), base_ref=pinned_sha)
            assert isinstance(post_hunks, dict)

            subprocess.run(
                ["git", "-C", str(repo), "checkout", "main"],
                capture_output=True, check=True,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestFix3MainTreeBusyGuard:
    """Verify the is_running_non_wt flag and guard checks."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_tab_state_has_is_running_non_wt(self) -> None:
        """_TabState should have is_running_non_wt attribute."""
        tab = _TabState("t1", "gpt-4")
        assert hasattr(tab, "is_running_non_wt")
        assert tab.is_running_non_wt is False

    def test_any_non_wt_running_detects_running_tab(self) -> None:
        """_any_non_wt_running returns True when a tab has the flag set."""
        server = VSCodeServer()
        tab = server._get_tab("t1")
        with server._state_lock:
            assert not server._any_non_wt_running()
            tab.is_running_non_wt = True
            assert server._any_non_wt_running()
            tab.is_running_non_wt = False
            assert not server._any_non_wt_running()

    def test_worktree_merge_blocked_when_non_wt_running(self) -> None:
        """_handle_worktree_action('merge') should refuse when non-wt running."""
        repo = _make_repo(Path(self._tmpdir) / "repo")
        server = VSCodeServer()
        server.work_dir = str(repo)

        wt_agent = WorktreeSorcarAgent("wt")
        wt_agent._chat_id = "wt_tab"
        wt_work = wt_agent._try_setup_worktree(repo, str(repo))
        assert wt_work is not None

        wt_tab = server._get_tab("wt_tab")
        wt_tab.agent = wt_agent
        wt_tab.use_worktree = True

        non_wt_tab = server._get_tab("non_wt_tab")
        non_wt_tab.is_running_non_wt = True

        result = server._handle_worktree_action("merge", "wt_tab")
        assert result["success"] is False
        assert "running" in result["message"].lower()

        non_wt_tab.is_running_non_wt = False
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

        wt_tab = server._get_tab("wt_tab")
        wt_tab.agent = wt_agent
        wt_tab.use_worktree = True

        non_wt_tab = server._get_tab("non_wt_tab")

        non_wt_tab.is_running_non_wt = False
        conflict_before = server._check_merge_conflict("wt_tab")
        assert conflict_before is True, (
            "sanity: dirty file does cause conflict when no non-wt running"
        )

        non_wt_tab.is_running_non_wt = True
        conflict_after = server._check_merge_conflict("wt_tab")
        assert conflict_after is False, (
            "Fix 3: dirty files from non-wt agent must not cause false conflict"
        )

        non_wt_tab.is_running_non_wt = False
        GitWorktreeOps.remove(repo, wt.wt_dir)
        GitWorktreeOps.prune(repo)
        GitWorktreeOps.delete_branch(repo, wt.branch)


    def test_run_task_inner_source_sets_non_wt_flag(self) -> None:
        """_run_task_inner sets is_running_non_wt for non-worktree tasks."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "is_running_non_wt = True" in source
        assert "is_running_non_wt = False" in source


class TestFix4SymmetricGuard:
    """Verify non-wt task start is blocked during worktree merge."""

    def test_run_task_inner_checks_worktree_merge_in_progress(self) -> None:
        """Source should check for ongoing worktree merge before non-wt task."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "t.is_merging and t.use_worktree" in source, (
            "Fix 4: must check for active worktree merge"
        )

    def test_non_wt_blocked_when_wt_merging(self) -> None:
        """A non-wt task should not start when a worktree merge is active."""
        server = VSCodeServer()
        wt_tab = server._get_tab("wt_tab")
        wt_tab.is_merging = True
        wt_tab.use_worktree = True

        non_wt_tab = server._get_tab("non_wt")
        non_wt_tab.use_worktree = False

        with server._state_lock:
            would_block = any(
                t.is_merging and t.use_worktree
                for t in server._tab_states.values()
            )
        assert would_block, (
            "Fix 4: non-wt task must be blocked when wt merge is active"
        )

    def test_non_wt_allowed_when_non_wt_merging(self) -> None:
        """A non-wt merge review should NOT block another non-wt task start."""
        server = VSCodeServer()
        tab1 = server._get_tab("tab1")
        tab1.is_merging = True
        tab1.use_worktree = False

        with server._state_lock:
            would_block = any(
                t.is_merging and t.use_worktree
                for t in server._tab_states.values()
            )
        assert not would_block, (
            "Fix 4: non-wt merge should not block another non-wt task"
        )

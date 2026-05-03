"""Integration tests for workflow bugs in worktree and non-worktree modes.

Confirms four bugs, each demonstrated by a test that fails before the fix
and passes after:

BUG 1 — ``is_task_active`` leaks True on early return
    When ``_run_task_inner`` rejects a non-worktree task because a
    worktree merge is in progress on another tab, ``is_task_active``
    is set to True but the early return bypasses the finally block
    that resets it.  The tab is then permanently stuck: merge/discard
    is blocked by ``_check_worktree_busy``.

BUG 2 — ``stash_pop`` loses staging state
    ``stash_pop`` uses plain ``git stash pop`` without ``--index``,
    so user's carefully staged changes lose their staged/unstaged
    distinction after the auto-stash → merge → auto-pop cycle.

BUG 3 — ``_auto_commit_worktree`` crashes when LLM is unavailable
    ``_generate_commit_message`` calls the LLM with no fallback.  If
    the LLM API is unreachable, the exception propagates uncaught,
    preventing worktree finalization and blocking all subsequent tasks
    on that agent.

BUG 4 — ``_close_tab`` orphans pending worktrees
    Closing a tab with a pending worktree drops the in-memory
    reference without auto-merging.  The worktree directory and branch
    persist in git, and ``cleanup_orphans`` skips them because they
    have ``kiss-original`` config.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps, _git
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer


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
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "checkout", "-b", "main"],
        capture_output=True, check=True,
    )
    (path / "init.txt").write_text("init\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True, check=True,
    )
    return path


def _make_wt_with_commit(
    repo: Path, branch: str, agent: WorktreeSorcarAgent,
) -> GitWorktree:
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    (wt_dir / "agent.txt").write_text("agent work\n")
    subprocess.run(
        ["git", "-C", str(wt_dir), "add", "."],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(wt_dir), "commit", "-m", "agent"],
        capture_output=True, check=True,
    )
    wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
    )
    agent._wt = wt
    return wt


class _RecordingPrinter:
    """Concrete printer that records broadcast calls (not a mock)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._thread_local = threading.local()
        self._persist_agents: dict[str, Any] = {}
        self._bash_buffers: dict[str, Any] = {}

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def cleanup_tab(self, tab_id: str) -> None:
        self._persist_agents.pop(tab_id, None)
        self._bash_buffers.pop(tab_id, None)

    def start_recording(self) -> None:
        pass

    def stop_recording(self) -> None:
        pass

    def peek_recording(self) -> list[dict[str, Any]]:
        return []

    def reset(self) -> None:
        pass


def _server(repo: Path) -> VSCodeServer:
    server = VSCodeServer()
    server.work_dir = str(repo)
    server.printer = cast(Any, _RecordingPrinter())
    return server


class TestBug1IsTaskActiveLeaks:
    """Non-wt task rejected by worktree-merge guard leaks is_task_active."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_is_task_active_reset_on_wt_merge_block(self) -> None:
        """is_task_active must be False after a non-wt task is rejected."""
        server = _server(self.repo)

        tab_a = server._get_tab("tab-a")
        tab_a.use_worktree = True
        tab_a.is_merging = True

        tab_b = server._get_tab("tab-b")
        tab_b.stop_event = threading.Event()
        tab_b.user_answer_queue = queue.Queue(maxsize=1)
        tab_b.task_thread = threading.Thread(target=lambda: None)

        server._run_task_inner({
            "tabId": "tab-b",
            "prompt": "do something",
            "useWorktree": False,
        })

        assert tab_b.is_task_active is False, (
            "BUG 1: is_task_active leaked True after non-wt task was "
            "rejected by worktree-merge guard"
        )


class TestBug2StashPopLosesStagingState:
    """stash_pop should preserve staged vs unstaged distinction."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stash_pop_preserves_index(self) -> None:
        """After stash → pop, staged modifications should remain staged."""
        repo = self.repo

        (repo / "f.txt").write_text("line1\n")
        (repo / "g.txt").write_text("line1\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "."],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "two files"],
            capture_output=True, check=True,
        )

        (repo / "f.txt").write_text("line1\nline2\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", "f.txt"],
            capture_output=True, check=True,
        )
        (repo / "g.txt").write_text("line1\nline2\n")

        cached = _git("diff", "--cached", "--name-only", cwd=repo)
        assert "f.txt" in cached.stdout
        unstaged = _git("diff", "--name-only", cwd=repo)
        assert "g.txt" in unstaged.stdout

        did_stash = GitWorktreeOps.stash_if_dirty(repo)
        assert did_stash

        ok = GitWorktreeOps.stash_pop(repo)
        assert ok

        cached_after = _git("diff", "--cached", "--name-only", cwd=repo)
        assert "f.txt" in cached_after.stdout, (
            "BUG 2: stash_pop lost staging state — f.txt is no longer "
            "in the index after stash → pop"
        )
        unstaged_after = _git("diff", "--name-only", cwd=repo)
        assert "g.txt" in unstaged_after.stdout


class TestBug3AutoCommitNoLLMFallback:
    """_auto_commit_worktree must not crash when LLM is unavailable."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_auto_commit_fallback_on_llm_failure(self) -> None:
        """_auto_commit_worktree should commit with fallback message on LLM error."""
        agent = WorktreeSorcarAgent("test")
        branch = "kiss/wt-test-llm-fail"
        wt = _make_wt_with_commit(self.repo, branch, agent)

        (wt.wt_dir / "extra.txt").write_text("extra work\n")

        import kiss.agents.sorcar.worktree_sorcar_agent as wsa_mod

        original_fn = wsa_mod._generate_commit_message

        def _failing_commit_msg(wt_dir: Path) -> str:
            raise RuntimeError("LLM API unavailable")

        wsa_mod._generate_commit_message = _failing_commit_msg  # type: ignore[assignment]
        try:
            result = agent._auto_commit_worktree()
            assert result is True, (
                "BUG 3: _auto_commit_worktree should commit with fallback "
                "message when LLM fails, but returned False"
            )

            assert not GitWorktreeOps.has_uncommitted_changes(wt.wt_dir), (
                "BUG 3: worktree still has uncommitted changes after "
                "_auto_commit_worktree with LLM failure"
            )
        finally:
            wsa_mod._generate_commit_message = original_fn  # type: ignore[assignment]

            GitWorktreeOps.remove(self.repo, wt.wt_dir)
            GitWorktreeOps.prune(self.repo)
            GitWorktreeOps.delete_branch(self.repo, branch)


class TestBug4CloseTabOrphansWorktree:
    """Closing a tab with a pending worktree must release it."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_close_tab_releases_pending_worktree(self) -> None:
        """Closing a tab must auto-merge the pending worktree."""
        server = _server(self.repo)
        tab_id = "tab-close-test"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        branch = "kiss/wt-close-test"
        wt = _make_wt_with_commit(self.repo, branch, tab.agent)

        assert tab.agent._wt_pending

        server._close_tab(tab_id)

        assert not GitWorktreeOps.branch_exists(self.repo, branch), (
            "BUG 4: _close_tab left orphaned branch after closing tab "
            "with pending worktree"
        )
        assert not wt.wt_dir.exists(), (
            "BUG 4: _close_tab left orphaned worktree directory"
        )

    def test_close_tab_no_changes_discards(self) -> None:
        """Closing a tab with a no-change worktree should discard it."""
        server = _server(self.repo)
        tab_id = "tab-close-empty"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        branch = "kiss/wt-close-empty"
        slug = branch.replace("/", "_")
        wt_dir = self.repo / ".kiss-worktrees" / slug
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(self.repo, branch, "main")
        tab.agent._wt = GitWorktree(
            repo_root=self.repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        assert tab.agent._wt_pending

        server._close_tab(tab_id)

        assert not GitWorktreeOps.branch_exists(self.repo, branch), (
            "BUG 4: empty worktree branch not deleted on tab close"
        )

    def test_close_tab_active_task_no_cleanup(self) -> None:
        """Closing a tab with an active task must NOT clean up worktree."""
        server = _server(self.repo)
        tab_id = "tab-active"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True
        tab.is_task_active = True

        branch = "kiss/wt-active-test"
        _make_wt_with_commit(self.repo, branch, tab.agent)

        server._close_tab(tab_id)

        assert tab_id in server._tab_states
        assert GitWorktreeOps.branch_exists(self.repo, branch)

        tab.is_task_active = False
        GitWorktreeOps.remove(self.repo, tab.agent._wt.wt_dir)  # type: ignore[union-attr]
        GitWorktreeOps.prune(self.repo)
        GitWorktreeOps.delete_branch(self.repo, branch)

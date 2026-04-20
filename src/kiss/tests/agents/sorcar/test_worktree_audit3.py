"""Tests confirming bugs found in worktree audit round 3.

Each test confirms a specific bug exists in the current code, labeled
BUG-8 through BUG-11.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import _git
from kiss.agents.sorcar.sorcar_agent import SorcarAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
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
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    (path / "fileA.txt").write_text("original A\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _patch_super_run(
    return_value: str = "success: true\nsummary: test done\n",
) -> Any:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


def _make_server(repo: Path) -> tuple:
    from kiss.agents.vscode.server import VSCodeServer

    server = VSCodeServer()
    events: list[dict] = []

    def capture(event: dict) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    server.work_dir = str(repo)
    return server, events


# ---------------------------------------------------------------------------
# BUG-8: _get_worktree_changed_files false positives when original branch
#         advances (e.g. another tab merges while worktree is active)
# ---------------------------------------------------------------------------


class TestBug8ChangedFilesFalsePositives:
    """_get_worktree_changed_files reports files the agent didn't change
    when the original branch advances after the worktree was created.

    The method compares the worktree working tree against the CURRENT tip
    of the original branch (`git diff --name-only <original_branch>`),
    not the fork point.  When the original branch advances (e.g. another
    worktree merges), files changed on main but NOT by the agent appear
    in the diff.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_changed_files_includes_unrelated_files_after_main_advances(self) -> None:
        """BUG-8: Unrelated files appear as 'changed' when main branch advances.

        1. Create a worktree from main
        2. Agent modifies fileA.txt in the worktree
        3. Advance main via a temp branch (add unrelated_file.txt)
        4. _get_worktree_changed_files should only report fileA.txt,
           but it also reports unrelated_file.txt
        """

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        # Create worktree and make agent changes
        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        # Agent modifies fileA.txt in the worktree
        (wt_dir / "fileA.txt").write_text("agent modified A\n")

        # Now advance the original branch with an unrelated commit.
        # We can't use a worktree on main (already checked out), so we
        # use a temp branch, commit there, and fast-forward main.
        original_branch = tab.agent._original_branch
        assert original_branch is not None

        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_advance"
        _git("worktree", "add", "-b", "tmp-advance", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "unrelated_file.txt").write_text("unrelated content\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "advance with unrelated file", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        # Fast-forward main to include the new commit
        _git("checkout", original_branch, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-advance", cwd=self.repo)
        _git("branch", "-d", "tmp-advance", cwd=self.repo)

        # Now get changed files
        changed = server._get_worktree_changed_files("0")

        # Agent only changed fileA.txt, so that's what should be reported
        assert "fileA.txt" in changed

        # BUG: unrelated_file.txt shows up because the diff is against
        # the current tip of main (which now has unrelated_file.txt),
        # not against the fork point.  The worktree doesn't have
        # unrelated_file.txt, so git sees it as "different".
        assert "unrelated_file.txt" in changed, (
            "BUG-8 confirmed: unrelated file from main advancement "
            "appears as 'changed' in the worktree"
        )

        tab.agent.discard()


# ---------------------------------------------------------------------------
# BUG-9: _check_merge_conflict has hidden auto-commit side effect
# ---------------------------------------------------------------------------


class TestBug9CheckConflictAutoCommitSideEffect:
    """_check_merge_conflict calls _auto_commit_worktree() as a side effect,
    meaning that merely checking for conflicts commits all uncommitted
    changes in the worktree.  This happens before the user explicitly
    chooses 'Commit and Merge'.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_check_conflict_commits_worktree_changes(self) -> None:
        """BUG-9: _check_merge_conflict auto-commits as a side effect.

        1. Create a worktree task
        2. Agent writes a file (uncommitted)
        3. Call _check_merge_conflict (a "query" method)
        4. Verify the file got committed as a side effect
        """

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        branch = tab.agent._wt_branch
        original = tab.agent._original_branch
        assert wt_dir is not None and branch is not None and original is not None

        # Agent writes a file but doesn't commit
        (wt_dir / "agent_output.txt").write_text("important work\n")

        # Verify no new commits on the branch yet
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", "No commits should exist before check"

        # Call _check_merge_conflict — this is supposed to be a query
        server._check_merge_conflict("0")

        # BUG: the "query" method auto-committed the worktree changes
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() != "0", (
            "BUG-9 confirmed: _check_merge_conflict auto-committed "
            "worktree changes as a side effect"
        )

        tab.agent.discard()

    def test_broadcast_worktree_done_commits_via_check_conflict(self) -> None:
        """BUG-9: _broadcast_worktree_done triggers auto-commit through
        _check_merge_conflict, committing changes before user acts.

        This is the real-world impact: when _run_task_inner finishes and
        broadcasts worktree_done, the worktree changes are silently
        committed before the user clicks 'Commit and Merge'.
        """

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        branch = tab.agent._wt_branch
        original = tab.agent._original_branch
        assert wt_dir is not None and branch is not None and original is not None

        (wt_dir / "agent_output.txt").write_text("work\n")

        # _broadcast_worktree_done is what _run_task_inner calls
        server._broadcast_worktree_done(["agent_output.txt"], "0")

        # BUG: changes are now committed even though user hasn't acted
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() != "0", (
            "BUG-9 confirmed: _broadcast_worktree_done auto-committed "
            "changes before user chose 'Commit and Merge'"
        )

        tab.agent.discard()


# ---------------------------------------------------------------------------
# BUG-10: _replay_session doesn't restore use_worktree from persisted data
# ---------------------------------------------------------------------------


class TestBug10ReplayDoesNotRestoreUseWorktree:
    """After server restart, _replay_session calls _emit_pending_worktree
    but tab.use_worktree is still False (never restored from persisted
    'extra' data), so the pending worktree is invisible.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_replay_session_does_not_emit_worktree_done(self) -> None:
        """BUG-10: After restart, _replay_session fails to emit worktree_done
        because use_worktree is not restored from persisted data.

        1. Run a worktree task → pending worktree exists
        2. Persist task with is_worktree=True in extra data
        3. Simulate server restart (new server, fresh tab state)
        4. Call _replay_session to resume the chat
        5. Expected: worktree_done event emitted
        6. Actual: no worktree_done because use_worktree is False
        """

        # Step 1: Original server runs a worktree task
        server1, events1 = _make_server(self.repo)
        tab1 = server1._get_tab("0")
        tab1.use_worktree = True
        tab1.agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert tab1.agent._wt_pending
        chat_id = tab1.agent.chat_id
        task_id = tab1.agent._last_task_id
        assert task_id is not None

        # Persist is_worktree=True
        th._save_task_extra(
            {"is_worktree": True, "model": "test"},
            task_id=task_id,
        )

        # Step 2: Simulate server restart
        server2, events2 = _make_server(self.repo)

        # Step 3: Resume session (what the extension does on restart)
        server2._replay_session(chat_id, "0")

        # BUG: tab.use_worktree is still False after replay
        tab2 = server2._get_tab("0")
        assert tab2.use_worktree is False, (
            "BUG-10 confirmed: use_worktree not restored from persisted data"
        )

        # BUG: no worktree_done event because _emit_pending_worktree returns early
        wt_events = [e for e in events2 if e["type"] == "worktree_done"]
        assert len(wt_events) == 0, (
            "BUG-10 confirmed: worktree_done not emitted after restart "
            "because use_worktree is False"
        )

        # Clean up the original worktree
        tab1.agent.discard()


# ---------------------------------------------------------------------------
# BUG-11: test_emit_pending_worktree_after_restart masks BUG-10
# ---------------------------------------------------------------------------


class TestBug11ExistingTestMasksBug:
    """The existing regression test for emit_pending_worktree_after_restart
    manually sets use_worktree=True on the new server, bypassing the real
    restart flow where use_worktree would be False.

    This test demonstrates that the existing test in
    test_worktree_extension_workflow.py doesn't catch BUG-10.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_existing_test_manually_sets_use_worktree(self) -> None:
        """BUG-11: The existing test bypasses the bug by setting use_worktree.

        In the existing test, _make_server sets use_worktree=True on
        the new server.  In a real restart, the extension sends a
        resumeSession command — use_worktree is never set.

        Compare:
        - Existing test: server2._get_tab("0").use_worktree = True  (manual)
        - Real restart: _replay_session called → use_worktree stays False
        """

        # Original server
        server1, events1 = _make_server(self.repo)
        tab1 = server1._get_tab("0")
        tab1.use_worktree = True
        tab1.agent.run(prompt_template="task1", work_dir=str(self.repo))
        chat_id = tab1.agent.chat_id

        # Simulate restart WITH manual use_worktree=True (like existing test)
        server_with_flag, events_with = _make_server(self.repo)
        server_with_flag._get_tab("0").use_worktree = True  # <- masks bug
        server_with_flag._get_tab("0").agent.resume_chat_by_id(chat_id)
        server_with_flag._emit_pending_worktree("0")
        wt_with = [e for e in events_with if e["type"] == "worktree_done"]
        # Existing test passes because use_worktree was manually set
        assert len(wt_with) == 1, "Works with manual use_worktree=True"

        # Simulate restart WITHOUT manual flag (real restart behavior)
        server_real, events_real = _make_server(self.repo)
        # use_worktree defaults to False — just like a real restart
        server_real._get_tab("0").agent.resume_chat_by_id(chat_id)
        server_real._emit_pending_worktree("0")
        wt_real = [e for e in events_real if e["type"] == "worktree_done"]
        # BUG-11: No worktree_done because use_worktree is False
        assert len(wt_real) == 0, (
            "BUG-11 confirmed: without manual use_worktree=True, "
            "pending worktree is invisible after restart"
        )

        # Clean up
        tab1.agent.discard()

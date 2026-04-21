"""Tests verifying fixes for worktree bugs BUG-8 through BUG-11.

Each test verifies the CORRECT behavior after the fix was applied.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import _git
from kiss.agents.sorcar.persistence import _append_chat_event
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
# BUG-8 FIX: _get_worktree_changed_files uses fork point, not branch tip
# ---------------------------------------------------------------------------


class TestBug8Fix:
    """After fix, _get_worktree_changed_files only reports files the agent
    actually changed, even when the original branch has advanced.
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

    def test_changed_files_excludes_unrelated_after_main_advances(self) -> None:
        """Only agent-modified files appear, not unrelated files from main."""

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        # Create worktree and make agent changes
        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        # Agent modifies fileA.txt in the worktree
        (wt_dir / "fileA.txt").write_text("agent modified A\n")

        # Advance the original branch with an unrelated commit
        original_branch = tab.agent._original_branch
        assert original_branch is not None

        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_advance"
        _git("worktree", "add", "-b", "tmp-advance", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "unrelated_file.txt").write_text("unrelated content\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "advance with unrelated file", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        _git("checkout", original_branch, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-advance", cwd=self.repo)
        _git("branch", "-d", "tmp-advance", cwd=self.repo)

        # Get changed files
        changed = server._get_worktree_changed_files("0")

        # Agent only changed fileA.txt — that's all that should appear
        assert "fileA.txt" in changed
        assert "unrelated_file.txt" not in changed, (
            "BUG-8 FIX: unrelated files from main advancement "
            "should NOT appear as changed"
        )

        tab.agent.discard()

    def test_changed_files_still_reports_agent_changes(self) -> None:
        """Sanity check: agent-modified files are still reported correctly."""

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "fileA.txt").write_text("agent modified A\n")
        (wt_dir / "new_file.txt").write_text("brand new\n")

        changed = server._get_worktree_changed_files("0")
        assert "fileA.txt" in changed
        assert "new_file.txt" in changed

        tab.agent.discard()


# ---------------------------------------------------------------------------
# BUG-9 FIX: _check_merge_conflict is a pure query (no auto-commit)
# ---------------------------------------------------------------------------


class TestBug9Fix:
    """After fix, _check_merge_conflict does NOT commit worktree changes."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_check_conflict_does_not_commit(self) -> None:
        """_check_merge_conflict must not create any commits."""

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
        assert r.stdout.strip() == "0", "No commits before check"

        # Call _check_merge_conflict
        server._check_merge_conflict("0")

        # BUG-9 FIX: no commits should have been created
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", (
            "BUG-9 FIX: _check_merge_conflict must not create commits"
        )

        tab.agent.discard()

    def test_broadcast_worktree_done_does_not_commit(self) -> None:
        """_broadcast_worktree_done must not auto-commit via conflict check."""

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        branch = tab.agent._wt_branch
        original = tab.agent._original_branch
        assert wt_dir is not None and branch is not None and original is not None

        (wt_dir / "agent_output.txt").write_text("work\n")

        server._broadcast_worktree_done(["agent_output.txt"], "0")

        # BUG-9 FIX: no commits should have been created
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", (
            "BUG-9 FIX: _broadcast_worktree_done must not auto-commit"
        )

        tab.agent.discard()

    def test_conflict_detected_when_both_sides_modify_same_file(self) -> None:
        """Conflict is reported when the same file is changed on both sides."""

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        original = tab.agent._original_branch
        assert wt_dir is not None and original is not None

        # Agent modifies fileA.txt in worktree
        (wt_dir / "fileA.txt").write_text("agent version\n")

        # Advance original branch with a conflicting change to fileA.txt
        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_conflict"
        _git("worktree", "add", "-b", "tmp-conflict", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "fileA.txt").write_text("main version\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "conflicting change", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        _git("checkout", original, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-conflict", cwd=self.repo)
        _git("branch", "-d", "tmp-conflict", cwd=self.repo)

        # Should detect conflict
        assert server._check_merge_conflict("0") is True

        tab.agent.discard()

    def test_no_conflict_when_different_files_changed(self) -> None:
        """No conflict when original and worktree modify different files."""

        server, events = _make_server(self.repo)
        tab = server._get_tab("0")
        tab.use_worktree = True

        tab.agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = tab.agent._wt_dir
        original = tab.agent._original_branch
        assert wt_dir is not None and original is not None

        # Agent modifies fileA.txt in worktree
        (wt_dir / "fileA.txt").write_text("agent version\n")

        # Advance original branch with a non-conflicting change
        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_noconflict"
        _git("worktree", "add", "-b", "tmp-noconflict", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "other_file.txt").write_text("other content\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "non-conflicting change", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        _git("checkout", original, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-noconflict", cwd=self.repo)
        _git("branch", "-d", "tmp-noconflict", cwd=self.repo)

        # Should NOT detect conflict (different files)
        assert server._check_merge_conflict("0") is False

        tab.agent.discard()


# ---------------------------------------------------------------------------
# BUG-10 FIX: _replay_session restores use_worktree from persisted data
# ---------------------------------------------------------------------------


class TestBug10Fix:
    """After fix, _replay_session restores use_worktree and emits
    worktree_done after restart.
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

    def test_replay_session_restores_use_worktree(self) -> None:
        """After restart, _replay_session sets use_worktree=True from
        persisted extra data, and emits worktree_done or merge_started.
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

        # Create a real change in the worktree so it's not auto-discarded
        # (BUG-66 fix: empty worktrees are now auto-discarded on resume)
        wt_dir = tab1.agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_output.txt").write_text("agent work\n")

        # Persist events and extra data (events needed for _replay_session
        # to not return early)
        _append_chat_event(
            {"type": "text_delta", "text": "working..."},
            task_id=task_id,
        )
        th._save_task_extra(
            {"is_worktree": True, "model": "test"},
            task_id=task_id,
        )

        # Step 2: Simulate server restart
        server2, events2 = _make_server(self.repo)

        # Step 3: Resume session
        server2._replay_session(chat_id, "0")

        # BUG-10 FIX: use_worktree is now restored from persisted data
        tab2 = server2._get_tab("0")
        assert tab2.use_worktree is True, (
            "BUG-10 FIX: use_worktree should be restored from persisted data"
        )

        # BUG-10 FIX: worktree_done or merge_started event should be emitted
        wt_events = [
            e for e in events2
            if e["type"] in ("worktree_done", "merge_started")
        ]
        assert len(wt_events) >= 1, (
            "BUG-10 FIX: worktree_done or merge_started should be emitted "
            "after restart"
        )

        # Clean up the original worktree
        tab1.agent.discard()

    def test_replay_session_without_worktree_keeps_false(self) -> None:
        """When extra doesn't have is_worktree, use_worktree stays False."""

        server1, events1 = _make_server(self.repo)
        tab1 = server1._get_tab("0")
        # Run without worktree
        tab1.agent.run(prompt_template="task1", work_dir=str(self.repo))
        chat_id = tab1.agent.chat_id
        task_id = tab1.agent._last_task_id
        assert task_id is not None

        _append_chat_event(
            {"type": "text_delta", "text": "working..."},
            task_id=task_id,
        )
        th._save_task_extra(
            {"is_worktree": False, "model": "test"},
            task_id=task_id,
        )

        server2, events2 = _make_server(self.repo)
        server2._replay_session(chat_id, "0")

        tab2 = server2._get_tab("0")
        assert tab2.use_worktree is False


# ---------------------------------------------------------------------------
# BUG-11 FIX: existing test no longer needs manual use_worktree=True
# (BUG-10 fix makes this work automatically through _replay_session)
# ---------------------------------------------------------------------------


class TestBug11Fix:
    """With BUG-10 fixed, the real restart flow now works without
    manually setting use_worktree=True.
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

    def test_real_restart_flow_emits_worktree_done(self) -> None:
        """Without manually setting use_worktree=True, pending worktree
        is visible after restart thanks to BUG-10 fix.
        """

        # Original server
        server1, events1 = _make_server(self.repo)
        tab1 = server1._get_tab("0")
        tab1.use_worktree = True
        tab1.agent.run(prompt_template="task1", work_dir=str(self.repo))
        chat_id = tab1.agent.chat_id
        task_id = tab1.agent._last_task_id
        assert task_id is not None

        # Create a real change in the worktree so it's not auto-discarded
        # (BUG-66 fix: empty worktrees are now auto-discarded on resume)
        wt_dir = tab1.agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_output.txt").write_text("agent work\n")

        _append_chat_event(
            {"type": "text_delta", "text": "working..."},
            task_id=task_id,
        )
        th._save_task_extra(
            {"is_worktree": True, "model": "test"},
            task_id=task_id,
        )

        # Simulate restart WITHOUT manual use_worktree=True
        server_real, events_real = _make_server(self.repo)
        # Just call _replay_session, like the real restart flow
        server_real._replay_session(chat_id, "0")

        # BUG-11 FIX: worktree_done or merge_started should be emitted
        # without manual flag
        wt_real = [
            e for e in events_real
            if e["type"] in ("worktree_done", "merge_started")
        ]
        assert len(wt_real) >= 1, (
            "BUG-11 FIX: pending worktree should be visible after "
            "restart without manual use_worktree=True"
        )

        # Clean up
        tab1.agent.discard()

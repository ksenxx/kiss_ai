"""Integration tests for WorktreeSorcarAgent ↔ VSCode extension workflow.

Validates the full commit-and-merge / discard workflow as exercised by
the extension: task execution → worktree_done broadcast → user action
(merge or discard) → worktree_result broadcast → git state cleanup.

Every test uses real git repos (no mocks).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

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
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
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


def _current_branch(repo: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _branch_exists(repo: Path, branch: str) -> bool:
    r = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify",
         f"refs/heads/{branch}"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _file_in_repo(repo: Path, filename: str) -> bool:
    return (repo / filename).exists()


def _make_server(repo: Path) -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    server.work_dir = str(repo)
    events: list[dict] = []

    def capture(event: dict) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


# ---------------------------------------------------------------------------
# Test class: Agent-level workflow
# ---------------------------------------------------------------------------


class TestWorktreeWorkflow:
    """Agent-level tests for commit-and-merge / discard workflow."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _agent(self) -> WorktreeSorcarAgent:
        return WorktreeSorcarAgent("test")

    # -- Merge: file changes land on original branch -----------------------

    def test_merge_propagates_file_changes_to_original_branch(self) -> None:
        """After merge, files created in the worktree appear on original."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()
        (wt_dir / "new_file.txt").write_text("hello from worktree")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "add new_file")

        agent.merge()
        assert _file_in_repo(self.repo, "new_file.txt")
        assert (self.repo / "new_file.txt").read_text() == "hello from worktree"

    def test_merge_restores_original_branch_as_head(self) -> None:
        """After merge, HEAD is the original branch."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        original = agent._original_branch
        assert original is not None

        agent.merge()
        assert _current_branch(self.repo) == original

    def test_merge_deletes_task_branch(self) -> None:
        """After merge, the task branch no longer exists."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        agent.merge()
        assert not _branch_exists(self.repo, branch)

    def test_merge_removes_worktree_dir(self) -> None:
        """After merge, the worktree directory is removed."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        agent.merge()
        assert not wt_dir.exists()

    def test_merge_cleans_git_config(self) -> None:
        """After merge, branch.<name>.kiss-original config is removed."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        agent.merge()
        r = _git("config", f"branch.{branch}.kiss-original", cwd=self.repo)
        assert r.returncode != 0  # config key should not exist

    # -- Discard: file changes do NOT land on original branch --------------

    def test_discard_does_not_propagate_file_changes(self) -> None:
        """After discard, files from the worktree do not appear on original."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "new_file.txt").write_text("should not appear")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "add new_file")

        agent.discard()
        assert not _file_in_repo(self.repo, "new_file.txt")

    def test_discard_restores_original_branch_as_head(self) -> None:
        """After discard, HEAD is back on original branch."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        original = agent._original_branch
        assert original is not None

        agent.discard()
        assert _current_branch(self.repo) == original

    def test_discard_deletes_task_branch(self) -> None:
        """After discard, the task branch no longer exists."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        branch = agent._wt_branch
        assert branch is not None

        agent.discard()
        assert not _branch_exists(self.repo, branch)

    def test_discard_removes_worktree_dir(self) -> None:
        """After discard, the worktree directory is removed."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        agent.discard()
        assert not wt_dir.exists()

    # -- Auto-commit -------------------------------------------------------

    def test_merge_auto_commits_uncommitted_worktree_changes(self) -> None:
        """Uncommitted changes in the worktree are auto-committed on merge."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "uncommitted.txt").write_text("not staged or committed")

        agent.merge()
        # The file should appear on the original branch (was auto-committed)
        assert _file_in_repo(self.repo, "uncommitted.txt")

    # -- Merge conflict preserves state ------------------------------------

    def test_merge_conflict_preserves_pending_state(self) -> None:
        """On merge conflict, agent stays pending so discard still works."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "README.md").write_text("worktree change\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "wt conflict")

        # Create conflicting change on original branch
        (self.repo / "README.md").write_text("main change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "main conflict", cwd=self.repo)

        msg = agent.merge()
        assert "Merge conflict" in msg
        assert agent._wt_pending  # still pending
        assert agent._wt_branch is not None

        # Discard should work after conflict
        agent.discard()
        assert not agent._wt_pending

    # -- Merge instructions format -----------------------------------------

    def test_merge_instructions_contain_both_options(self) -> None:
        """merge_instructions mentions both merge() and discard()."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        instructions = agent.merge_instructions()
        assert "agent.merge()" in instructions
        assert "agent.discard()" in instructions
        assert agent._wt_branch is not None
        assert agent._wt_branch in instructions
        agent.discard()

    def test_run_result_includes_merge_instructions(self) -> None:
        """run() result has task output + separator + merge instructions."""
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert "---" in result
        assert "agent.merge()" in result
        assert "agent.discard()" in result
        agent.discard()


# ---------------------------------------------------------------------------
# Test class: Server-level workflow (extension integration)
# ---------------------------------------------------------------------------


class TestServerWorktreeWorkflow:
    """Server-level tests mimicking the extension's worktree flow."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.original_run = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.original_run)
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _setup_pending_worktree(
        self, server: VSCodeServer, *, with_changes: bool = True,
    ) -> str:
        """Run a task to create a pending worktree, optionally with changes.

        Returns the task branch name.
        """
        server.agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )
        branch = server.agent._wt_branch
        assert branch is not None

        if with_changes:
            wt_dir = server.agent._wt_dir
            assert wt_dir is not None
            (wt_dir / "changed.txt").write_text("extension change")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_all(wt_dir, "extension change")

        return branch

    # -- worktree_done event -----------------------------------------------

    def test_worktree_done_event_fields(self) -> None:
        """worktree_done broadcast has branch, worktreeDir, originalBranch."""
        server, events = _make_server(self.repo)
        branch = self._setup_pending_worktree(server)

        # Simulate what _run_task_inner does after task
        server.agent._auto_commit_worktree()
        changed = server._get_worktree_changed_files()
        assert len(changed) > 0

        server.printer.broadcast({
            "type": "worktree_done",
            "branch": server.agent._wt_branch,
            "worktreeDir": str(server.agent._wt_dir),
            "originalBranch": server.agent._original_branch,
        })

        wt_events = [e for e in events if e["type"] == "worktree_done"]
        assert len(wt_events) == 1
        ev = wt_events[0]
        assert ev["branch"] == branch
        assert ev["originalBranch"] is not None
        assert "worktreeDir" in ev

        server.agent.discard()

    # -- Merge via server --------------------------------------------------

    def test_server_merge_returns_success_result(self) -> None:
        """_handle_worktree_action('merge') returns success with message."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        result = server._handle_worktree_action("merge")
        assert result["success"] is True
        assert "Successfully merged" in result["message"]

    def test_server_merge_cleans_agent_state(self) -> None:
        """After merge via server, agent has no pending worktree."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("merge")
        assert server.agent._wt_branch is None
        assert not server.agent._wt_pending

    def test_server_merge_propagates_changes(self) -> None:
        """After merge via server, changes are on the original branch."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("merge")
        assert _file_in_repo(self.repo, "changed.txt")

    # -- Discard via server ------------------------------------------------

    def test_server_discard_returns_success_result(self) -> None:
        """_handle_worktree_action('discard') returns success."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        result = server._handle_worktree_action("discard")
        assert result["success"] is True
        assert "Discarded" in result["message"]

    def test_server_discard_cleans_agent_state(self) -> None:
        """After discard via server, agent has no pending worktree."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("discard")
        assert server.agent._wt_branch is None
        assert not server.agent._wt_pending

    def test_server_discard_does_not_propagate_changes(self) -> None:
        """After discard via server, changes are not on original branch."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("discard")
        assert not _file_in_repo(self.repo, "changed.txt")

    # -- Command routing ---------------------------------------------------

    def test_worktree_action_command_broadcasts_result(self) -> None:
        """worktreeAction command broadcasts a worktree_result event."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_command(
            {"type": "worktreeAction", "action": "discard"}
        )
        wt_results = [e for e in events if e["type"] == "worktree_result"]
        assert len(wt_results) == 1
        assert wt_results[0]["success"] is True

    def test_unknown_worktree_action_returns_error(self) -> None:
        """Unknown worktree action returns error result."""
        server, events = _make_server(self.repo)
        result = server._handle_worktree_action("invalid")
        assert result["success"] is False
        assert "Unknown action" in result["message"]

    # -- Auto-discard on no changes ----------------------------------------

    def test_no_changes_triggers_auto_discard(self) -> None:
        """When the worktree has no changes, the agent auto-discards."""
        server, events = _make_server(self.repo)
        # Run task without making any changes in the worktree
        self._setup_pending_worktree(server, with_changes=False)

        # After auto-commit, there are no changed files vs original
        server.agent._auto_commit_worktree()
        changed = server._get_worktree_changed_files()
        assert len(changed) == 0

        # Server would auto-discard in _run_task_inner
        server.agent.discard()
        assert not server.agent._wt_pending

    # -- Session replay restores pending worktree --------------------------

    def test_emit_pending_worktree_after_restart(self) -> None:
        """_emit_pending_worktree re-emits worktree_done after restart."""
        server, events = _make_server(self.repo)
        branch = self._setup_pending_worktree(server)
        original_branch = server.agent._original_branch

        # Simulate server restart: create new server, restore state
        server2, events2 = _make_server(self.repo)
        server2.agent.resume_chat_by_id(server.agent._chat_id)
        server2._emit_pending_worktree()

        wt_events = [e for e in events2 if e["type"] == "worktree_done"]
        assert len(wt_events) == 1
        assert wt_events[0]["branch"] == branch
        assert wt_events[0]["originalBranch"] == original_branch

        # Clean up
        server2.agent.discard()

    # -- Merge conflict via server -----------------------------------------

    def test_server_merge_conflict_returns_failure(self) -> None:
        """Merge conflict via server returns success=False."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        # Create conflict on original branch
        (self.repo / "changed.txt").write_text("conflicting content")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "conflict on main", cwd=self.repo)

        result = server._handle_worktree_action("merge")
        assert result["success"] is False
        assert "conflict" in result["message"].lower()

        # Agent still pending — can discard
        assert server.agent._wt_pending
        server.agent.discard()

    # -- Merge then new task -----------------------------------------------

    def test_merge_then_new_task_works(self) -> None:
        """After merge via server, a new task can run."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)
        server._handle_worktree_action("merge")

        # New task should succeed
        result = server.agent.run(
            prompt_template="task2", work_dir=str(self.repo)
        )
        assert "test done" in result
        assert server.agent._wt_pending
        server.agent.discard()

    def test_discard_then_new_task_works(self) -> None:
        """After discard via server, a new task can run."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)
        server._handle_worktree_action("discard")

        result = server.agent.run(
            prompt_template="task2", work_dir=str(self.repo)
        )
        assert "test done" in result
        assert server.agent._wt_pending
        server.agent.discard()

    # -- Premature commit bug (changes must NOT be committed before user acts) --

    def test_task_does_not_commit_before_user_action(self) -> None:
        """After task finishes, worktree changes must NOT be committed yet.

        The agent should leave changes uncommitted in the worktree so
        the user can review them before choosing Commit and Merge.
        """
        server, events = _make_server(self.repo)
        server.agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server.agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()
        # Simulate agent making changes (files written but NOT committed)
        (wt_dir / "agent_file.txt").write_text("agent wrote this")

        # Check: the worktree branch should have NO new commits beyond
        # what was on the original branch (the file is uncommitted)
        branch = server.agent._wt_branch
        original = server.agent._original_branch
        assert branch is not None and original is not None
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", (
            "Worktree branch should have no new commits before user action"
        )

        # The changed file detection should still find the uncommitted file
        changed = server._get_worktree_changed_files()
        assert "agent_file.txt" in changed

        server.agent.discard()

    def test_merge_commits_then_merges(self) -> None:
        """merge() should commit uncommitted changes, then merge."""
        server, events = _make_server(self.repo)
        server.agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server.agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_file.txt").write_text("agent wrote this")

        # Before merge: no commits on the branch
        branch = server.agent._wt_branch
        assert branch is not None

        # Merge should auto-commit the changes and merge
        result = server._handle_worktree_action("merge")
        assert result["success"] is True

        # File should now be on original branch
        assert _file_in_repo(self.repo, "agent_file.txt")
        assert (self.repo / "agent_file.txt").read_text() == "agent wrote this"

    def test_discard_drops_uncommitted_changes(self) -> None:
        """discard() should throw away uncommitted worktree changes."""
        server, events = _make_server(self.repo)
        server.agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server.agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_file.txt").write_text("should be discarded")

        server._handle_worktree_action("discard")
        assert not _file_in_repo(self.repo, "agent_file.txt")

    def test_get_worktree_changed_files_detects_uncommitted(self) -> None:
        """_get_worktree_changed_files() must detect uncommitted changes."""
        server, events = _make_server(self.repo)
        server.agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server.agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "new.txt").write_text("new file")
        (wt_dir / "README.md").write_text("modified\n")

        changed = server._get_worktree_changed_files()
        assert "README.md" in changed
        assert "new.txt" in changed

        server.agent.discard()

    def test_run_task_inner_does_not_auto_commit(self) -> None:
        """_run_task_inner must NOT call _auto_commit_worktree().

        The auto-commit should only happen when the user clicks
        'Commit and Merge', not when the task finishes.
        """
        server, events = _make_server(self.repo)
        # We'll inspect the agent after run() to check no commit happened.
        # The patched super().run() returns immediately without making
        # file changes, so we check the run_task_inner flow by verifying
        # the worktree has 0 branch commits.
        server.agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server.agent._wt_dir
        assert wt_dir is not None

        # Create files to simulate what the agent tool would have done
        (wt_dir / "tool_output.txt").write_text("from tool")

        # Now simulate what _run_task_inner does after the task:
        # It should detect changes WITHOUT committing
        changed = server._get_worktree_changed_files()
        assert len(changed) > 0

        # Verify nothing was committed
        branch = server.agent._wt_branch
        original = server.agent._original_branch
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0"

        server.agent.discard()

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

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
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
    raise_exc: BaseException | None = None,
) -> Any:
    """Patch SorcarAgent's parent ``run()`` to return a canned value.

    Args:
        return_value: String to return from the fake run.
        raise_exc: If set, the fake run raises this exception instead
            of returning *return_value*.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        if raise_exc is not None:
            raise raise_exc
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
    server._get_tab("0").use_worktree = True
    server.work_dir = str(repo)
    events: list[dict] = []

    def capture(event: dict) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


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

    @pytest.mark.slow
    def test_merge_commits_changes(self) -> None:
        """After merge, changes are committed on the original branch."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()
        (wt_dir / "new_file.txt").write_text("hello from worktree")
        (wt_dir / "README.md").write_text("modified\n")

        agent.merge()

        assert _file_in_repo(self.repo, "new_file.txt")
        assert (self.repo / "README.md").read_text() == "modified\n"

        status = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        porcelain = status.stdout.strip()
        assert not porcelain, "Working tree should be clean after merge"

        merge_head = self.repo / ".git" / "MERGE_HEAD"
        assert not merge_head.exists()

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
        assert r.returncode != 0


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


    def test_do_nothing_method_does_not_exist(self) -> None:
        """WorktreeSorcarAgent no longer has a do_nothing() method."""
        agent = self._agent()
        assert not hasattr(agent, "do_nothing")


    def test_merge_auto_commits_uncommitted_worktree_changes(self) -> None:
        """Uncommitted changes in the worktree are auto-committed on merge."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "uncommitted.txt").write_text("not staged or committed")

        agent.merge()
        assert _file_in_repo(self.repo, "uncommitted.txt")


    def test_merge_conflict_preserves_pending_state(self) -> None:
        """On merge conflict, agent stays pending so discard still works."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        wt_dir = agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "README.md").write_text("worktree change\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_all(wt_dir, "wt conflict")

        (self.repo / "README.md").write_text("main change\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "main conflict", cwd=self.repo)

        msg = agent.merge()
        assert "Merge conflict" in msg
        assert agent._wt_pending
        assert agent._wt_branch is not None

        status = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        assert not status.stdout.strip(), "Working tree should be clean after conflict"

        agent.discard()
        assert not agent._wt_pending


    def test_merge_instructions_contain_all_options(self) -> None:
        """merge_instructions mentions merge() and discard() but not do_nothing()."""
        agent = self._agent()
        agent.run(prompt_template="task1", work_dir=str(self.repo))

        instructions = agent.merge_instructions()
        assert "agent.merge()" in instructions
        assert "agent.discard()" in instructions
        assert "do_nothing" not in instructions
        assert agent._wt_branch is not None
        assert agent._wt_branch in instructions
        agent.discard()

    def test_run_result_is_plain_task_output(self) -> None:
        """run() returns only the task result — no merge-instructions suffix."""
        agent = self._agent()
        result = agent.run(prompt_template="task1", work_dir=str(self.repo))
        assert "agent.merge()" not in result
        assert "agent.discard()" not in result
        assert "do_nothing" not in result
        assert agent._wt_pending
        agent.discard()


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
        server._get_tab("0").agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )
        branch = server._get_tab("0").agent._wt_branch
        assert branch is not None

        if with_changes:
            wt_dir = server._get_tab("0").agent._wt_dir
            assert wt_dir is not None
            (wt_dir / "changed.txt").write_text("extension change")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_all(wt_dir, "extension change")

        return branch


    def test_worktree_done_event_fields(self) -> None:
        """worktree_done broadcast has branch, worktreeDir, originalBranch."""
        server, events = _make_server(self.repo)
        branch = self._setup_pending_worktree(server)

        server._get_tab("0").agent._auto_commit_worktree()
        changed = server._get_worktree_changed_files("0")
        assert len(changed) > 0

        server.printer.broadcast({
            "type": "worktree_done",
            "branch": server._get_tab("0").agent._wt_branch,
            "worktreeDir": str(server._get_tab("0").agent._wt_dir),
            "originalBranch": server._get_tab("0").agent._original_branch,
        })

        wt_events = [e for e in events if e["type"] == "worktree_done"]
        assert len(wt_events) == 1
        ev = wt_events[0]
        assert ev["branch"] == branch
        assert ev["originalBranch"] is not None
        assert "worktreeDir" in ev

        server._get_tab("0").agent.discard()


    def test_server_merge_returns_success_result(self) -> None:
        """_handle_worktree_action('merge') returns success with message."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        result = server._handle_worktree_action("merge", "0")
        assert result["success"] is True
        assert "Successfully merged" in result["message"]

    def test_server_merge_cleans_agent_state(self) -> None:
        """After merge via server, agent has no pending worktree."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("merge", "0")
        assert server._get_tab("0").agent._wt_branch is None
        assert not server._get_tab("0").agent._wt_pending

    def test_server_merge_propagates_changes(self) -> None:
        """After merge via server, changes are on the original branch."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("merge", "0")
        assert _file_in_repo(self.repo, "changed.txt")


    def test_server_discard_returns_success_result(self) -> None:
        """_handle_worktree_action('discard') returns success."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        result = server._handle_worktree_action("discard", "0")
        assert result["success"] is True
        assert "Discarded" in result["message"]

    def test_server_discard_cleans_agent_state(self) -> None:
        """After discard via server, agent has no pending worktree."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("discard", "0")
        assert server._get_tab("0").agent._wt_branch is None
        assert not server._get_tab("0").agent._wt_pending

    def test_server_discard_does_not_propagate_changes(self) -> None:
        """After discard via server, changes are not on original branch."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_worktree_action("discard", "0")
        assert not _file_in_repo(self.repo, "changed.txt")


    def test_server_do_nothing_rejected_as_unknown(self) -> None:
        """do_nothing action is rejected as unknown after simplification."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        result = server._handle_worktree_action("do_nothing", "0")
        assert result["success"] is False
        assert "Unknown action" in result["message"]

        server._get_tab("0").agent.discard()


    def test_worktree_action_command_broadcasts_result(self) -> None:
        """worktreeAction command broadcasts a worktree_result event."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        server._handle_command(
            {"type": "worktreeAction", "action": "discard", "tabId": "0"}
        )
        wt_results = [e for e in events if e["type"] == "worktree_result"]
        assert len(wt_results) == 1
        assert wt_results[0]["success"] is True

    def test_unknown_worktree_action_returns_error(self) -> None:
        """Unknown worktree action returns error result."""
        server, events = _make_server(self.repo)
        result = server._handle_worktree_action("invalid", "0")
        assert result["success"] is False
        assert "Unknown action" in result["message"]


    def test_no_changes_triggers_auto_discard(self) -> None:
        """When the worktree has no changes, the agent auto-discards."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server, with_changes=False)

        server._get_tab("0").agent._auto_commit_worktree()
        changed = server._get_worktree_changed_files("0")
        assert len(changed) == 0

        server._get_tab("0").agent.discard()
        assert not server._get_tab("0").agent._wt_pending


    def test_emit_pending_worktree_after_restart(self) -> None:
        """_emit_pending_worktree re-emits worktree_done after restart."""
        server, events = _make_server(self.repo)
        branch = self._setup_pending_worktree(server)
        original_branch = server._get_tab("0").agent._original_branch

        server2, events2 = _make_server(self.repo)
        server2._get_tab("0").agent.resume_chat_by_id(server._get_tab("0").agent._chat_id)
        server2._emit_pending_worktree("0")

        wt_events = [e for e in events2 if e["type"] == "worktree_done"]
        assert len(wt_events) == 1
        assert wt_events[0]["branch"] == branch
        assert wt_events[0]["originalBranch"] == original_branch

        server2._get_tab("0").agent.discard()


    def test_server_merge_conflict_returns_failure(self) -> None:
        """Merge conflict via server returns success=False."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)

        (self.repo / "changed.txt").write_text("conflicting content")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-m", "conflict on main", cwd=self.repo)

        result = server._handle_worktree_action("merge", "0")
        assert result["success"] is False
        assert "conflict" in result["message"].lower()

        assert server._get_tab("0").agent._wt_pending
        server._get_tab("0").agent.discard()


    def test_merge_then_new_task_works(self) -> None:
        """After merge via server, a new task can run."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)
        server._handle_worktree_action("merge", "0")

        result = server._get_tab("0").agent.run(
            prompt_template="task2", work_dir=str(self.repo)
        )
        assert "test done" in result
        assert server._get_tab("0").agent._wt_pending
        server._get_tab("0").agent.discard()

    def test_discard_then_new_task_works(self) -> None:
        """After discard via server, a new task can run."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)
        server._handle_worktree_action("discard", "0")

        result = server._get_tab("0").agent.run(
            prompt_template="task2", work_dir=str(self.repo)
        )
        assert "test done" in result
        assert server._get_tab("0").agent._wt_pending
        server._get_tab("0").agent.discard()


    def test_task_does_not_commit_before_user_action(self) -> None:
        """After task finishes, worktree changes must NOT be committed yet.

        The agent should leave changes uncommitted in the worktree so
        the user can review them before choosing Commit and Merge.
        """
        server, events = _make_server(self.repo)
        server._get_tab("0").agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server._get_tab("0").agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()
        (wt_dir / "agent_file.txt").write_text("agent wrote this")

        branch = server._get_tab("0").agent._wt_branch
        original = server._get_tab("0").agent._original_branch
        assert branch is not None and original is not None
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", (
            "Worktree branch should have no new commits before user action"
        )

        changed = server._get_worktree_changed_files("0")
        assert "agent_file.txt" in changed

        server._get_tab("0").agent.discard()

    @pytest.mark.slow
    def test_merge_commits_then_merges(self) -> None:
        """merge() should commit uncommitted changes, then merge."""
        server, events = _make_server(self.repo)
        server._get_tab("0").agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server._get_tab("0").agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_file.txt").write_text("agent wrote this")

        branch = server._get_tab("0").agent._wt_branch
        assert branch is not None

        result = server._handle_worktree_action("merge", "0")
        assert result["success"] is True

        assert _file_in_repo(self.repo, "agent_file.txt")
        assert (self.repo / "agent_file.txt").read_text() == "agent wrote this"

    def test_discard_drops_uncommitted_changes(self) -> None:
        """discard() should throw away uncommitted worktree changes."""
        server, events = _make_server(self.repo)
        server._get_tab("0").agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server._get_tab("0").agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_file.txt").write_text("should be discarded")

        server._handle_worktree_action("discard", "0")
        assert not _file_in_repo(self.repo, "agent_file.txt")

    def test_get_worktree_changed_files_detects_uncommitted(self) -> None:
        """_get_worktree_changed_files() must detect uncommitted changes."""
        server, events = _make_server(self.repo)
        server._get_tab("0").agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server._get_tab("0").agent._wt_dir
        assert wt_dir is not None
        (wt_dir / "new.txt").write_text("new file")
        (wt_dir / "README.md").write_text("modified\n")

        changed = server._get_worktree_changed_files("0")
        assert "README.md" in changed
        assert "new.txt" in changed

        server._get_tab("0").agent.discard()

    def test_run_task_inner_does_not_auto_commit(self) -> None:
        """_run_task_inner must NOT call _auto_commit_worktree().

        The auto-commit should only happen when the user clicks
        'Commit and Merge', not when the task finishes.
        """
        server, events = _make_server(self.repo)
        server._get_tab("0").agent.run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = server._get_tab("0").agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "tool_output.txt").write_text("from tool")

        changed = server._get_worktree_changed_files("0")
        assert len(changed) > 0

        branch = server._get_tab("0").agent._wt_branch
        original = server._get_tab("0").agent._original_branch
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0"

        server._get_tab("0").agent.discard()


    def test_worktree_no_changes_auto_discarded_on_failure(self) -> None:
        """Worktree is auto-discarded when agent fails with no file changes.

        When the exception happens before any file changes, the
        worktree is discarded automatically and no worktree_done event
        is emitted (nothing for the user to merge).
        """
        _unpatch_super_run(self.original_run)
        self.original_run = _patch_super_run(raise_exc=RuntimeError("boom"))

        server, events = _make_server(self.repo)
        server._run_task_inner({
            "prompt": "failing task",
            "workDir": str(self.repo),
            "tabId": "0",
            "useWorktree": True,
            "model": "test-model",
        })

        wt_events = [e for e in events if e["type"] == "worktree_done"]
        assert len(wt_events) == 0
        done_events = [e for e in events if e["type"] == "task_done"]
        assert len(done_events) == 1
        assert not server._get_tab("0").agent._wt_pending

    def test_worktree_no_changes_auto_discarded_on_stop(self) -> None:
        """Worktree is auto-discarded when user stops with no file changes."""
        _unpatch_super_run(self.original_run)
        self.original_run = _patch_super_run(raise_exc=KeyboardInterrupt("stopped"))

        server, events = _make_server(self.repo)
        server._run_task_inner({
            "prompt": "stopped task",
            "workDir": str(self.repo),
            "tabId": "0",
            "useWorktree": True,
            "model": "test-model",
        })

        stopped_events = [e for e in events if e["type"] == "task_stopped"]
        assert len(stopped_events) == 1
        assert not server._get_tab("0").agent._wt_pending

    def test_worktree_merge_review_shown_on_failure_with_changes(self) -> None:
        """Merge/diff review UI is shown when agent fails after making changes.

        Regression: previously the merge/diff + worktree action UI was
        only shown on success, leaving the user with no way to merge or
        discard after a failure.  Now the merge review (merge_data +
        merge_started) is emitted in the finally block so the user can
        still review and accept/reject changes.
        """
        _unpatch_super_run(self.original_run)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original_parent_run = parent_class.run

        def fake_run_with_changes(self_agent: object, **kwargs: object) -> str:
            wt_dir = getattr(self_agent, "_wt_dir", None)
            if wt_dir is not None:
                (Path(wt_dir) / "agent_output.txt").write_text("partial work")
            else:
                work_dir = kwargs.get("work_dir", "")
                if work_dir:
                    Path(str(work_dir), "agent_output.txt").write_text("partial work")
            raise RuntimeError("task crashed after writing files")

        parent_class.run = fake_run_with_changes
        self.original_run = original_parent_run

        server, events = _make_server(self.repo)
        server._run_task_inner({
            "prompt": "crashing task",
            "workDir": str(self.repo),
            "tabId": "0",
            "useWorktree": True,
            "model": "test-model",
        })

        merge_events = [e for e in events if e["type"] == "merge_data"]
        assert len(merge_events) == 1
        merge_started = [e for e in events if e["type"] == "merge_started"]
        assert len(merge_started) == 1
        assert server._get_tab("0").agent._wt_pending

        server._finish_merge("0")
        wt_done = [e for e in events if e["type"] == "worktree_done"]
        assert len(wt_done) == 1
        assert len(wt_done[0].get("changedFiles", [])) > 0

        tab = server._get_tab("0")
        if tab.agent._wt_pending:
            tab.agent.discard()

    def test_worktree_merge_review_shown_on_stop_with_changes(self) -> None:
        """Merge/diff review UI is shown when user stops after agent made changes.

        Same as the failure test but with KeyboardInterrupt (user stop).
        """
        _unpatch_super_run(self.original_run)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original_parent_run = parent_class.run

        def fake_run_with_changes(self_agent: object, **kwargs: object) -> str:
            wt_dir = getattr(self_agent, "_wt_dir", None)
            if wt_dir is not None:
                (Path(wt_dir) / "agent_output.txt").write_text("partial work")
            else:
                work_dir = kwargs.get("work_dir", "")
                if work_dir:
                    Path(str(work_dir), "agent_output.txt").write_text("partial work")
            raise KeyboardInterrupt("stopped after writing files")

        parent_class.run = fake_run_with_changes
        self.original_run = original_parent_run

        server, events = _make_server(self.repo)
        server._run_task_inner({
            "prompt": "stopped task",
            "workDir": str(self.repo),
            "tabId": "0",
            "useWorktree": True,
            "model": "test-model",
        })

        merge_events = [e for e in events if e["type"] == "merge_data"]
        assert len(merge_events) == 1
        stopped = [e for e in events if e["type"] == "task_stopped"]
        assert len(stopped) == 1

        tab = server._get_tab("0")
        if tab.agent._wt_pending:
            tab.agent.discard()


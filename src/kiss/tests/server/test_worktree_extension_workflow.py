# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_worktree_extension_workflow import (  # noqa: F401
    _file_in_repo,
    _make_repo,
    _patch_super_run,
    _redirect_db,
    _restore_db,
    _unpatch_super_run,
)


def _agent(server: VSCodeServer, tab_id: str = "0") -> WorktreeSorcarAgent:
    """Return the tab's registered agent, asserting it exists for mypy."""
    state = agent_state.find_by_tab(tab_id)
    assert state is not None and state.agent is not None
    return state.agent


def _make_server(repo: Path) -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    server.work_dir = str(repo)
    state = agent_state.AgentState(
        "task-0",
        agent=WorktreeSorcarAgent("Sorcar VS Code"),
        tab_id="0",
        server_owned=True,
    )
    state.use_worktree = True
    agent_state.register(state)
    events: list[dict] = []

    def capture(event: dict) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


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
        agent_state.agent_states.clear()

    def _setup_pending_worktree(
        self, server: VSCodeServer, *, with_changes: bool = True,
    ) -> str:
        """Run a task to create a pending worktree, optionally with changes.

        Returns the task branch name.
        """
        _agent(server).run(
            prompt_template="task1", work_dir=str(self.repo)
        )
        branch = _agent(server)._wt_branch
        assert branch is not None

        if with_changes:
            wt_dir = _agent(server)._wt_dir
            assert wt_dir is not None
            (wt_dir / "changed.txt").write_text("extension change")
            GitWorktreeOps.stage_all(wt_dir)
            GitWorktreeOps.commit_all(wt_dir, "extension change")

        return branch


    def test_worktree_done_event_fields(self) -> None:
        """worktree_done broadcast has branch, worktreeDir, originalBranch."""
        server, events = _make_server(self.repo)
        branch = self._setup_pending_worktree(server)

        _agent(server)._auto_commit_worktree()
        changed = server._get_worktree_changed_files("0")
        assert len(changed) > 0

        server.printer.broadcast({
            "type": "worktree_done",
            "branch": _agent(server)._wt_branch,
            "worktreeDir": str(_agent(server)._wt_dir),
            "originalBranch": _agent(server)._original_branch,
        })

        wt_events = [e for e in events if e["type"] == "worktree_done"]
        assert len(wt_events) == 1
        ev = wt_events[0]
        assert ev["branch"] == branch
        assert ev["originalBranch"] is not None
        assert "worktreeDir" in ev

        _agent(server).discard()


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
        assert _agent(server)._wt_branch is None
        assert not _agent(server)._wt_pending

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
        assert _agent(server)._wt_branch is None
        assert not _agent(server)._wt_pending

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

        _agent(server).discard()


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


    def test_no_changes_triggers_auto_discard(self) -> None:
        """When the worktree has no changes, the agent auto-discards."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server, with_changes=False)

        _agent(server)._auto_commit_worktree()
        changed = server._get_worktree_changed_files("0")
        assert len(changed) == 0

        _agent(server).discard()
        assert not _agent(server)._wt_pending


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

        assert _agent(server)._wt_pending
        _agent(server).discard()


    def test_merge_then_new_task_works(self) -> None:
        """After merge via server, a new task can run."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)
        server._handle_worktree_action("merge", "0")

        result = _agent(server).run(
            prompt_template="task2", work_dir=str(self.repo)
        )
        assert "test done" in result
        assert _agent(server)._wt_pending
        _agent(server).discard()

    def test_discard_then_new_task_works(self) -> None:
        """After discard via server, a new task can run."""
        server, events = _make_server(self.repo)
        self._setup_pending_worktree(server)
        server._handle_worktree_action("discard", "0")

        result = _agent(server).run(
            prompt_template="task2", work_dir=str(self.repo)
        )
        assert "test done" in result
        assert _agent(server)._wt_pending
        _agent(server).discard()


    def test_task_does_not_commit_before_user_action(self) -> None:
        """After task finishes, worktree changes must NOT be committed yet.

        The agent should leave changes uncommitted in the worktree so
        the user can review them before choosing Commit and Merge.
        """
        server, events = _make_server(self.repo)
        _agent(server).run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = _agent(server)._wt_dir
        assert wt_dir is not None and wt_dir.exists()
        (wt_dir / "agent_file.txt").write_text("agent wrote this")

        branch = _agent(server)._wt_branch
        original = _agent(server)._original_branch
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

        _agent(server).discard()

    @pytest.mark.slow
    def test_merge_commits_then_merges(self) -> None:
        """merge() should commit uncommitted changes, then merge."""
        server, events = _make_server(self.repo)
        _agent(server).run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = _agent(server)._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_file.txt").write_text("agent wrote this")

        branch = _agent(server)._wt_branch
        assert branch is not None

        result = server._handle_worktree_action("merge", "0")
        assert result["success"] is True

        assert _file_in_repo(self.repo, "agent_file.txt")
        assert (self.repo / "agent_file.txt").read_text() == "agent wrote this"

    def test_discard_drops_uncommitted_changes(self) -> None:
        """discard() should throw away uncommitted worktree changes."""
        server, events = _make_server(self.repo)
        _agent(server).run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = _agent(server)._wt_dir
        assert wt_dir is not None
        (wt_dir / "agent_file.txt").write_text("should be discarded")

        server._handle_worktree_action("discard", "0")
        assert not _file_in_repo(self.repo, "agent_file.txt")

    def test_get_worktree_changed_files_detects_uncommitted(self) -> None:
        """_get_worktree_changed_files() must detect uncommitted changes."""
        server, events = _make_server(self.repo)
        _agent(server).run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = _agent(server)._wt_dir
        assert wt_dir is not None
        (wt_dir / "new.txt").write_text("new file")
        (wt_dir / "README.md").write_text("modified\n")

        changed = server._get_worktree_changed_files("0")
        assert "README.md" in changed
        assert "new.txt" in changed

        _agent(server).discard()

    def test_run_task_inner_does_not_auto_commit(self) -> None:
        """_run_task_inner must NOT call _auto_commit_worktree().

        The auto-commit should only happen when the user clicks
        'Commit and Merge', not when the task finishes.
        """
        server, events = _make_server(self.repo)
        _agent(server).run(
            prompt_template="task1", work_dir=str(self.repo)
        )

        wt_dir = _agent(server)._wt_dir
        assert wt_dir is not None

        (wt_dir / "tool_output.txt").write_text("from tool")

        changed = server._get_worktree_changed_files("0")
        assert len(changed) > 0

        branch = _agent(server)._wt_branch
        original = _agent(server)._original_branch
        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0"

        _agent(server).discard()


    def test_worktree_no_changes_preserved_on_failure(self) -> None:
        """Worktree branch is preserved when agent fails with no file changes.

        The user explicitly enabled the worktree workflow (and left
        auto-commit off), so the branch must remain in git for manual
        inspection / merge / discard even when the task failed before
        making any file changes.  Previously this path auto-discarded
        the branch, which surfaced as the "worktree branch is not
        getting created" symptom in the
        ``use_worktree=True`` + ``autoCommit=False`` mode.

        The branch is preserved (``_wt_pending`` stays True) but the
        ``worktree_done`` event — which the frontend renders as the
        "Auto-commit and merge or Discard?" prompt — must NOT be
        broadcast when there are no changes to merge.  Showing that
        prompt for an empty worktree is meaningless and was the user-
        reported bug.
        """
        _unpatch_super_run(self.original_run)
        self.original_run = _patch_super_run(raise_exc=RuntimeError("boom"))

        server, events = _make_server(self.repo)
        server._run_task_inner({
            "prompt": "failing task",
            "workDir": str(self.repo),
            "tabId": "0",
            "useWorktree": True,
            "model": "",
        })

        wt_events = [e for e in events if e["type"] == "worktree_done"]
        assert wt_events == []
        done_events = [e for e in events if e["type"] == "task_done"]
        assert len(done_events) == 1
        assert _agent(server)._wt_pending

    def test_worktree_no_changes_preserved_on_stop(self) -> None:
        """Worktree branch is preserved when user stops with no file changes.

        ``_wt_pending`` stays True so the branch survives, but no
        ``worktree_done`` is broadcast — there is nothing to merge or
        discard, so the frontend prompt would be meaningless.
        """
        _unpatch_super_run(self.original_run)
        self.original_run = _patch_super_run(raise_exc=KeyboardInterrupt("stopped"))

        server, events = _make_server(self.repo)
        server._run_task_inner({
            "prompt": "stopped task",
            "workDir": str(self.repo),
            "tabId": "0",
            "useWorktree": True,
            "model": "",
        })

        stopped_events = [e for e in events if e["type"] == "task_stopped"]
        assert len(stopped_events) == 1
        assert _agent(server)._wt_pending
        wt_events = [e for e in events if e["type"] == "worktree_done"]
        assert wt_events == []

    def test_worktree_done_shown_on_failure_with_changes(self) -> None:
        """Merge/Discard buttons are shown when agent fails after making changes.

        Regression: previously the worktree action UI was only shown on
        success, leaving the user with no way to merge or discard after
        a failure.  Now ``worktree_done`` is broadcast in the finally
        block so the user can still merge or discard the changes.
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
            "model": "",
        })

        wt_done = [e for e in events if e["type"] == "worktree_done"]
        assert len(wt_done) == 1
        assert len(wt_done[0].get("changedFiles", [])) > 0
        assert _agent(server)._wt_pending

        agent = _agent(server)
        if agent._wt_pending:
            agent.discard()

    def test_worktree_done_shown_on_stop_with_changes(self) -> None:
        """Merge/Discard buttons are shown when user stops after agent made changes.

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
            "model": "",
        })

        wt_done = [e for e in events if e["type"] == "worktree_done"]
        assert len(wt_done) == 1
        assert len(wt_done[0].get("changedFiles", [])) > 0
        stopped = [e for e in events if e["type"] == "task_stopped"]
        assert len(stopped) == 1

        agent = _agent(server)
        if agent._wt_pending:
            agent.discard()

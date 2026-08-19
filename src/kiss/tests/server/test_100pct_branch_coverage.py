# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Server-only branch-coverage tests extracted from
``kiss.tests.agents.third_party_agents.test_100pct_branch_coverage``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server); the
``_channel_cli`` coverage stayed behind in
tests/agents/third_party_agents.

Targets remaining uncovered branches in:
  json_printer.py (_format_tool_call, peek_recording)
  server.py (merge-conflict/changed-files/result-summary paths)

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_100pct_branch_coverage import (
    _redirect_db,
    _restore_db,
)


def _register_worktree_state(tab_id: str) -> AgentState:
    """Register a server-owned worktree state for *tab_id* and return it."""
    state = AgentState(
        f"task-{tab_id}",
        agent=WorktreeSorcarAgent("Sorcar VS Code"),
        tab_id=tab_id,
        server_owned=True,
    )
    state.use_worktree = True
    agent_state.register(state)
    return state


class TestFormatToolCallBranches:
    """Cover _format_tool_call branches in json_printer.py."""

    def test_format_tool_call_with_all_fields(self) -> None:
        """All optional fields present in tool_input."""
        p = JsonPrinter()
        p._thread_local.task_id = "t1"
        p.start_recording()
        p._format_tool_call("Edit", {
            "file_path": "/path/to/file.py",
            "description": "edit desc",
            "command": "some cmd",
            "content": "file content",
            "old_string": "old",
            "new_string": "new",
            "extra_param": "extra_val",
        })
        events = p.stop_recording()
        ev = events[0]
        assert ev["type"] == "tool_call"
        assert ev["name"] == "Edit"
        assert ev["path"] == "/path/to/file.py"
        assert ev["description"] == "edit desc"
        assert ev["command"] == "some cmd"
        assert ev["content"] == "file content"
        assert ev["old_string"] == "old"
        assert ev["new_string"] == "new"
        assert "extras" in ev

class TestVSCodeServerUncoveredBranches:
    """Cover remaining uncovered branches in VSCodeServer."""

    def test_check_merge_conflict_no_branches(self) -> None:
        """_check_merge_conflict returns False when no wt_branch."""
        server = VSCodeServer()
        state = _register_worktree_state("0")
        try:
            assert state.agent is not None
            state.agent._wt = None
            assert server._check_merge_conflict("0") is False
        finally:
            agent_state.unregister(state.task_id, state)

    def test_get_worktree_changed_files_no_branches(self) -> None:
        """_get_worktree_changed_files returns [] when no branches."""
        server = VSCodeServer()
        state = _register_worktree_state("0")
        try:
            assert state.agent is not None
            state.agent._wt = None
            assert server._get_worktree_changed_files("0") == []
        finally:
            agent_state.unregister(state.task_id, state)

    def test_check_merge_conflict_dirty_worktree(self, tmp_path: Path) -> None:
        """_check_merge_conflict detects dirty files that overlap with merge."""
        saved = _redirect_db(str(tmp_path))
        try:
            repo = tmp_path / "repo"
            repo.mkdir()
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=repo, capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=repo, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("content")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)

            subprocess.run(["git", "checkout", "-b", "test-branch"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("branch content")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "mod"], cwd=repo, capture_output=True)
            subprocess.run(["git", "checkout", "main"], cwd=repo, capture_output=True)

            (repo / "f.txt").write_text("dirty local change")

            wt_dir = repo / ".kiss-worktrees" / "test-wt"
            subprocess.run(
                ["git", "worktree", "add", "-b", "test-wt", str(wt_dir)],
                cwd=repo, capture_output=True, check=True,
            )
            (wt_dir / "f.txt").write_text("worktree content")
            subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "wt mod"], cwd=wt_dir, capture_output=True,
            )

            server = VSCodeServer()
            state = _register_worktree_state("0")
            assert state.agent is not None
            state.agent._wt = GitWorktree(
                repo_root=repo, branch="test-wt",
                original_branch="main",
                wt_dir=wt_dir,
            )
            server.work_dir = str(repo)

            try:
                assert server._check_merge_conflict("0") is True
            finally:
                agent_state.unregister(state.task_id, state)
        finally:
            _restore_db(saved)

class TestVSCodeServerExtractResultSummary:
    """Cover _extract_result_summary."""

    def test_extract_result_summary_with_result_event(self) -> None:
        """_extract_result_summary finds the result event."""
        server = VSCodeServer()
        server.printer._thread_local.task_id = "t1"
        server.printer.start_recording()
        server.printer.broadcast({"type": "text_delta", "text": "hello"})
        import yaml
        text = yaml.dump({"success": True, "summary": "All done"})
        server.printer.broadcast({"type": "result", "text": text, "summary": "All done"})
        summary = server._extract_result_summary()
        assert summary == "All done"
        server.printer.stop_recording()

class TestBrowserPrinterPeekRecording:
    """Cover peek_recording for empty/non-existent recording."""

    def test_peek_active_recording(self) -> None:
        """peek_recording returns current events without stopping."""
        p = JsonPrinter()
        p._thread_local.task_id = "t1"
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "hello"})
        events = p.peek_recording()
        assert len(events) == 1
        p.broadcast({"type": "text_delta", "text": " world"})
        events2 = p.peek_recording()
        assert len(events2) == 1
        assert events2[0]["text"] == "hello world"
        p.stop_recording()

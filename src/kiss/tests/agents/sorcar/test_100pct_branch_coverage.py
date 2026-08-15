# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

Targets remaining uncovered branches in:
  _channel_cli.py (channel-agent CLI helpers)
  persistence.py: lines 263, 426
  sorcar_agent.py: lines 251-252
  chat_sorcar_agent.py: lines 130->134, 132-133
  useful_tools.py: lines 184, 204
  worktree_sorcar_agent.py: lines 187, 209-211, 313-314, 351
  json_printer.py: lines 205-215, 248, 254, 259-260, 281-285, 294, 302-310,
                 319-323, 329-330, 332, 333->335, 336, 340, 342, 344->346,
                 349, 352, 355, 358, 363-365, 367-368, 376
  server.py: lines 315->341, 319, 361->369, 416, 733-740

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent, _generate_commit_message
from kiss.agents.third_party_agents._channel_cli import (
    _build_arg_parser,
    _build_run_kwargs,
)
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer

_SavedState = tuple[Path, "sqlite3.Connection | None", Path]


def _redirect_db(tmpdir: str) -> _SavedState:
    old: _SavedState = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: _SavedState) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class TestCliHelpers:
    """Cover uncovered branches in _channel_cli.py."""

    def test_build_run_kwargs(self) -> None:
        """_build_run_kwargs builds kwargs from parsed args."""
        with tempfile.TemporaryDirectory() as d:
            parser = _build_arg_parser()
            args = parser.parse_args(["-t", "do something", "-w", d, "-e", "http://localhost:1234"])
            kwargs = _build_run_kwargs(args)
            assert kwargs["prompt_template"] == "do something"
            assert kwargs["work_dir"] == d
            assert kwargs["model_config"]["base_url"] == "http://localhost:1234"
            assert kwargs["web_tools"] is True


class TestPersistenceUncoveredBranches:
    """Cover remaining persistence.py branches."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestWorktreeCommitMessageBranches:
    """Cover commit message generation branches."""

    @pytest.mark.slow
    def test_generate_commit_message_with_staged_changes(self, tmp_path: Path) -> None:
        """Commit message generation with staged changes exercises the LLM path.

        Creates a real repo with staged changes; the method either succeeds
        (returning an LLM-generated message) or catches an exception and
        returns the fallback, covering one of the two code paths.
        """
        saved = _redirect_db(str(tmp_path))
        try:
            repo = tmp_path / "commitgen"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=repo, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("initial")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("modified content")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)

            msg = _generate_commit_message(repo)
            assert isinstance(msg, str) and len(msg) > 0
        finally:
            _restore_db(saved)


class TestBrowserPrinterPrintBranches:
    """Cover all print() type branches in json_printer.py."""

    def _make_printer(self) -> JsonPrinter:
        p = JsonPrinter()
        p.start_recording()
        return p


class TestFormatToolCallBranches:
    """Cover _format_tool_call branches (lines 336-358)."""

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


class TestVSCodeServerUncoveredBranches:
    """Cover remaining uncovered branches in VSCodeServer."""

    def test_check_merge_conflict_no_branches(self) -> None:
        """_check_merge_conflict returns False when no wt_branch (line 733)."""
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

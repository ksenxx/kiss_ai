"""Integration tests for 100% branch coverage of sorcar/ and vscode/ modules.

Targets remaining uncovered branches in:
  cli_helpers.py: lines 23, 53->39, 106-119, 137-142, 153-155, 172-180, 200-203
  persistence.py: lines 263, 426
  sorcar_agent.py: lines 251-252
  stateful_sorcar_agent.py: lines 130->134, 132-133
  useful_tools.py: lines 184, 204
  worktree_sorcar_agent.py: lines 187, 209-211, 313-314, 351
  browser_ui.py: lines 205-215, 248, 254, 259-260, 281-285, 294, 302-310,
                 319-323, 329-330, 332, 333->335, 336, 340, 342, 344->346,
                 349, 352, 355, 358, 363-365, 367-368, 376
  server.py: lines 315->341, 319, 361->369, 416, 733-740

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.cli_helpers import (
    _apply_chat_args,
    _build_arg_parser,
    _build_run_kwargs,
    _print_recent_chats,
)
from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import _generate_commit_message
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers for DB isolation
# ---------------------------------------------------------------------------

_SavedState = tuple[Path, "sqlite3.Connection | None", Path]


def _redirect_db(tmpdir: str) -> _SavedState:
    old: _SavedState = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_db(saved: _SavedState) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


# ---------------------------------------------------------------------------
# cli_helpers.py
# ---------------------------------------------------------------------------


class TestCliHelpers:
    """Cover uncovered branches in cli_helpers.py."""

    def test_print_recent_chats_with_data(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_print_recent_chats with populated chats prints session data."""
        saved = _redirect_db(str(tmp_path))
        try:
            _, chat_id = th._add_task("task one")
            th._save_task_result(result="result one", task="task one")
            # Add a task with long text to hit truncation branches
            long_text = "X" * 300
            th._add_task(long_text, chat_id=chat_id)
            th._save_task_result(result="R" * 300, task=long_text)
            # Add a task with empty result to cover 53->39 (if result_text: False)
            th._add_task("task no result", chat_id=chat_id)
            th._save_task_result(result="", task="task no result")
            _print_recent_chats()
            out = capsys.readouterr().out
            assert "Chat ID:" in out
        finally:
            _restore_db(saved)

    def test_apply_chat_args_chat_id(self, tmp_path: Path) -> None:
        """_apply_chat_args with --chat-id resumes that session."""
        saved = _redirect_db(str(tmp_path))
        try:
            agent = StatefulSorcarAgent("test")
            args = argparse.Namespace(new=False, chat_id=1000)
            _apply_chat_args(agent, args)
            assert agent.chat_id == 1000
        finally:
            _restore_db(saved)

    def test_apply_chat_args_no_options(self, tmp_path: Path) -> None:
        """_apply_chat_args with neither new nor chat_id and no task is a no-op."""
        saved = _redirect_db(str(tmp_path))
        try:
            agent = StatefulSorcarAgent("test")
            args = argparse.Namespace(new=False, chat_id=None)
            _apply_chat_args(agent, args, task="")
        finally:
            _restore_db(saved)

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


# ---------------------------------------------------------------------------
# persistence.py — uncovered branches
# ---------------------------------------------------------------------------


class TestPersistenceUncoveredBranches:
    """Cover remaining persistence.py branches."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._saved = _redirect_db(self._tmpdir)

    def teardown_method(self) -> None:
        _restore_db(self._saved)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# sorcar_agent.py — _resolve_task branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# stateful_sorcar_agent.py — run() exception branch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# useful_tools.py — Read/Write error paths
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# worktree_sorcar_agent.py — _generate_commit_message branches
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# browser_ui.py — comprehensive branch coverage
# ---------------------------------------------------------------------------


class TestBrowserPrinterPrintBranches:
    """Cover all print() type branches in browser_ui.py."""

    def _make_printer(self) -> tuple[BaseBrowserPrinter, queue.Queue[dict]]:
        p = BaseBrowserPrinter()
        cq: queue.Queue[dict] = queue.Queue()
        p._client_queue = cq
        return p, cq


class TestFormatToolCallBranches:
    """Cover _format_tool_call branches (lines 336-358)."""

    def test_format_tool_call_with_all_fields(self) -> None:
        """All optional fields present in tool_input."""
        p = BaseBrowserPrinter()
        cq: queue.Queue[dict] = queue.Queue()
        p._client_queue = cq
        p._format_tool_call("Edit", {
            "file_path": "/path/to/file.py",
            "description": "edit desc",
            "command": "some cmd",
            "content": "file content",
            "old_string": "old",
            "new_string": "new",
            "extra_param": "extra_val",
        })
        ev = cq.get_nowait()
        assert ev["type"] == "tool_call"
        assert ev["name"] == "Edit"
        assert ev["path"] == "/path/to/file.py"
        assert ev["description"] == "edit desc"
        assert ev["command"] == "some cmd"
        assert ev["content"] == "file content"
        assert ev["old_string"] == "old"
        assert ev["new_string"] == "new"
        assert "extras" in ev


# ---------------------------------------------------------------------------
# server.py — uncovered branches
# ---------------------------------------------------------------------------


class TestVSCodeServerUncoveredBranches:
    """Cover remaining uncovered branches in VSCodeServer."""

    def test_check_merge_conflict_no_branches(self) -> None:
        """_check_merge_conflict returns False when no wt_branch (line 733)."""
        server = VSCodeServer()
        tab = server._get_tab("0")
        tab.use_worktree = True
        tab.worktree_agent._wt = None  # type: ignore[attr-defined]
        assert server._check_merge_conflict() is False

    def test_get_worktree_changed_files_no_branches(self) -> None:
        """_get_worktree_changed_files returns [] when no branches."""
        server = VSCodeServer()
        tab = server._get_tab("0")
        tab.use_worktree = True
        tab.worktree_agent._wt = None  # type: ignore[attr-defined]
        assert server._get_worktree_changed_files() == []

    def test_check_merge_conflict_dirty_worktree(self, tmp_path: Path) -> None:
        """_check_merge_conflict detects dirty files that overlap with merge."""
        saved = _redirect_db(str(tmp_path))
        try:
            repo = tmp_path / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=repo, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("content")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)

            # Create a branch that modifies f.txt
            subprocess.run(["git", "checkout", "-b", "test-branch"], cwd=repo, capture_output=True)
            (repo / "f.txt").write_text("branch content")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "mod"], cwd=repo, capture_output=True)
            subprocess.run(["git", "checkout", "main"], cwd=repo, capture_output=True)

            # Dirty the same file in the working tree
            (repo / "f.txt").write_text("dirty local change")

            server = VSCodeServer()
            tab = server._get_tab("0")
            tab.use_worktree = True
            tab.worktree_agent._wt = GitWorktree(
                repo_root=repo, branch="test-branch",
                original_branch="main",
                wt_dir=repo / ".kiss-worktrees" / "test-branch",
            )
            server.work_dir = str(repo)

            # Should detect overlap between dirty file and merge changeset
            assert server._check_merge_conflict("0") is True
        finally:
            _restore_db(saved)

    def test_handle_worktree_action_unknown(self) -> None:
        """_handle_worktree_action with unknown action (server.py line end)."""
        server = VSCodeServer()
        server._get_tab("0").use_worktree = True
        result = server._handle_worktree_action("unknown_action", tab_id="0")
        assert result["success"] is False
        assert "Unknown action" in result["message"]


class TestVSCodeServerExtractResultSummary:
    """Cover _extract_result_summary."""

    def test_extract_result_summary_with_result_event(self) -> None:
        """_extract_result_summary finds the result event."""
        server = VSCodeServer()
        rec_id = 1
        server.printer.start_recording(rec_id)
        server.printer.broadcast({"type": "text_delta", "text": "hello"})
        import yaml
        text = yaml.dump({"success": True, "summary": "All done"})
        server.printer.broadcast({"type": "result", "text": text, "summary": "All done"})
        summary = server._extract_result_summary(rec_id)
        assert summary == "All done"
        server.printer.stop_recording(rec_id)


class TestBrowserPrinterPeekRecording:
    """Cover peek_recording for empty/non-existent recording."""

    def test_peek_active_recording(self) -> None:
        """peek_recording returns current events without stopping."""
        p = BaseBrowserPrinter()
        p.start_recording(42)
        p.broadcast({"type": "text_delta", "text": "hello"})
        events = p.peek_recording(42)
        assert len(events) == 1
        # Recording still active
        p.broadcast({"type": "text_delta", "text": " world"})
        events2 = p.peek_recording(42)
        assert len(events2) == 1  # coalesced
        assert events2[0]["text"] == "hello world"
        p.stop_recording(42)

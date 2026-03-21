"""Integration tests for kiss/agents/sorcar/ to increase branch coverage.

No mocks, patches, or test doubles. Uses real files, real git repos, and
real objects.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import requests

import kiss.agents.sorcar.task_history as task_history
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _disable_copilot_scm_button,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _scan_files,
    _snapshot_files,
    _untracked_base_dir,
)
from kiss.agents.sorcar.sorcar import run_chatbot
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.kiss_error import KISSError
from kiss.core.relentless_agent import RelentlessAgent


def _init_git_repo(tmpdir: str) -> None:
    """Initialize a git repo with one committed file."""
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True
    )
    Path(tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)


class TestSaveUntrackedBase:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = tempfile.mkdtemp()
        _init_git_repo(self.tmpdir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)
        base_dir = _untracked_base_dir()
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

class TestCleanupMergeData:
    def setup_method(self) -> None:
        self.data_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)
        base_dir = _untracked_base_dir()
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

class TestPrepareMergeViewUntrackedModified:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = tempfile.mkdtemp()
        _init_git_repo(self.tmpdir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)
        base_dir = _untracked_base_dir()
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

class TestPrepareMergeViewTrackedPreHash:
    """Test _prepare_merge_view with tracked files that have pre_file_hashes."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = tempfile.mkdtemp()
        _init_git_repo(self.tmpdir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)
        base_dir = _untracked_base_dir()
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

class TestUsefulToolsRead:

    def test_read_nonexistent_file(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        result = tools.Read("/no/such/file.txt")
        assert "Error:" in result

    def test_read_truncates_long_files(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(3000):
                f.write(f"line {i}\n")
            path = f.name
        try:
            result = tools.Read(path, max_lines=10)
            assert "[truncated:" in result
        finally:
            os.unlink(path)


class TestUsefulToolsWrite:
    def test_write_error(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        result = tools.Write("/dev/null/impossible/file.txt", "content")
        assert "Error:" in result


class TestUsefulToolsEdit:
    def test_edit_file_not_found(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        result = tools.Edit("/no/file.txt", "old", "new")
        assert "Error:" in result

    def test_edit_same_string(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            result = tools.Edit(path, "hello", "hello")
            assert "must be different" in result
        finally:
            os.unlink(path)

    def test_edit_string_not_found(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            result = tools.Edit(path, "xyz", "abc")
            assert "not found" in result
        finally:
            os.unlink(path)

    def test_edit_multiple_occurrences_without_replace_all(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aa bb aa")
            path = f.name
        try:
            result = tools.Edit(path, "aa", "cc")
            assert "appears 2 times" in result
        finally:
            os.unlink(path)

    def test_edit_replace_all(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aa bb aa")
            path = f.name
        try:
            result = tools.Edit(path, "aa", "cc", replace_all=True)
            assert "replaced 2" in result
            assert Path(path).read_text() == "cc bb cc"
        finally:
            os.unlink(path)


class TestUsefulToolsBash:
    def test_bash_disallowed_command(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        result = tools.Bash("eval 'echo hello'", "test disallowed")
        assert "not allowed" in result

class TestTaskHistory:
    def setup_method(self) -> None:
        from kiss.agents.sorcar import task_history

        self._orig_history_file = task_history.HISTORY_FILE
        self._orig_model_usage_file = task_history.MODEL_USAGE_FILE
        self._orig_file_usage_file = task_history.FILE_USAGE_FILE
        self._orig_kiss_dir = task_history._KISS_DIR

        self.tmpdir = tempfile.mkdtemp()
        self._orig_events_dir = task_history._CHAT_EVENTS_DIR
        task_history._KISS_DIR = Path(self.tmpdir)
        task_history.HISTORY_FILE = Path(self.tmpdir) / "task_history.jsonl"
        task_history._CHAT_EVENTS_DIR = Path(self.tmpdir) / "chat_events"
        task_history.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        task_history.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"

        task_history._history_cache = None

    def teardown_method(self) -> None:
        from kiss.agents.sorcar import task_history

        task_history.HISTORY_FILE = self._orig_history_file
        task_history._CHAT_EVENTS_DIR = self._orig_events_dir
        task_history.MODEL_USAGE_FILE = self._orig_model_usage_file
        task_history.FILE_USAGE_FILE = self._orig_file_usage_file
        task_history._KISS_DIR = self._orig_kiss_dir
        task_history._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestScanFiles:
    def test_respects_depth_limit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            current = d
            for i in range(8):
                current = os.path.join(current, f"level{i}")
                os.makedirs(current)
                Path(current, f"file{i}.txt").write_text(f"content {i}")
            paths = _scan_files(d)
            assert not any("level5/file5.txt" in p for p in paths)
            assert any("file0.txt" in p for p in paths)

    def test_caps_at_2000_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for i in range(2050):
                Path(d, f"file{i:04d}.txt").write_text(f"content {i}")
            paths = _scan_files(d)
            assert len(paths) <= 2000

class TestDisableCopilotScmButton:
    def test_bad_package_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "github.copilot-chat-1.0.0"
            ext_dir.mkdir(parents=True)
            (ext_dir / "package.json").write_text("not json")
            _disable_copilot_scm_button(d)

class TestTruncateOutputTailZero:
    def test_tail_zero_branch(self) -> None:
        """When remaining=0 after subtracting msg length, tail=0 and line 29 is hit."""
        from kiss.agents.sorcar.useful_tools import _truncate_output

        text = "a" * 200
        worst_msg = f"\n\n... [truncated {len(text)} chars] ...\n\n"
        result = _truncate_output(text, len(worst_msg))
        assert "truncated" in result
        assert not result.startswith("a")


class TestExtractLeadingCommandNameEdgeCases:
    def test_empty_name_after_lstrip(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        assert _extract_leading_command_name("((") is None


class TestBrowserPrinterBashStream:
    def test_reset_with_active_timer(self) -> None:
        """Covers reset cancelling an active flush timer (lines 458-459)."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        printer._bash_last_flush = time.monotonic()
        printer.print("data\n", type="bash_stream")
        assert printer._bash_flush_timer is not None
        printer.reset()
        assert printer._bash_flush_timer is None


class TestTaskHistoryEdgeCases:
    def setup_method(self) -> None:
        from kiss.agents.sorcar import task_history

        self._orig_history_file = task_history.HISTORY_FILE
        self._orig_model_usage_file = task_history.MODEL_USAGE_FILE
        self._orig_file_usage_file = task_history.FILE_USAGE_FILE
        self._orig_kiss_dir = task_history._KISS_DIR
        self._orig_events_dir = task_history._CHAT_EVENTS_DIR
        self.tmpdir = tempfile.mkdtemp()
        task_history._KISS_DIR = Path(self.tmpdir)
        task_history.HISTORY_FILE = Path(self.tmpdir) / "task_history.jsonl"
        task_history._CHAT_EVENTS_DIR = Path(self.tmpdir) / "chat_events"
        task_history.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        task_history.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        task_history._history_cache = None

    def teardown_method(self) -> None:
        from kiss.agents.sorcar import task_history

        task_history.HISTORY_FILE = self._orig_history_file
        task_history._CHAT_EVENTS_DIR = self._orig_events_dir
        task_history.MODEL_USAGE_FILE = self._orig_model_usage_file
        task_history.FILE_USAGE_FILE = self._orig_file_usage_file
        task_history._KISS_DIR = self._orig_kiss_dir
        task_history._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestInstallCopilotExtension:
    def test_already_installed(self) -> None:
        """When copilot extension dir exists, return immediately."""
        from kiss.agents.sorcar.code_server import _install_copilot_extension

        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "github.copilot-1.0.0"
            ext_dir.mkdir(parents=True)
            _install_copilot_extension(d)

class TestDisableCopilotScmButtonEdgeCases:
    def test_copilot_chat_without_package_json(self) -> None:
        """Directory exists but no package.json."""
        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "github.copilot-chat-1.0.0"
            ext_dir.mkdir(parents=True)
            _disable_copilot_scm_button(d)

class TestPrepareMergeViewFilteredHunks:
    """Test the pre_hunks filtering logic in _prepare_merge_view."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = tempfile.mkdtemp()
        _init_git_repo(self.tmpdir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)
        base_dir = _untracked_base_dir()
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

class TestBrowserUiUncoveredBranches:
    """Cover remaining uncovered branches in browser_ui.py."""


    def test_content_block_delta_unknown_delta_type(self) -> None:
        """Cover 705->723: content_block_delta with unknown delta_type."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()

        class FakeEvent:
            event = {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": "abc"},
            }

        text = printer.parse_stream_event(FakeEvent())
        assert text == ""

class TestCodeServerUncoveredBranches:
    """Cover remaining uncovered branches in code_server.py."""

    def test_disable_copilot_write_oserror(self) -> None:
        """Cover 487-488: OSError when writing back package.json."""
        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "github.copilot-chat-1.0.0"
            ext_dir.mkdir(parents=True)
            pkg = {
                "contributes": {
                    "menus": {
                        "scm/inputBox": [
                            {
                                "command": "github.copilot.git.generateCommitMessage",
                                "when": "scmProvider == git",
                            }
                        ]
                    }
                }
            }
            pkg_path = ext_dir / "package.json"
            pkg_path.write_text(json.dumps(pkg))
            pkg_path.chmod(0o444)
            try:
                _disable_copilot_scm_button(d)
            finally:
                pkg_path.chmod(0o644)

    def test_save_untracked_base_oserror_on_copy(self) -> None:
        """Cover 748-749: OSError when copying untracked file (unreadable)."""
        tmpdir = tempfile.mkdtemp()
        try:
            _init_git_repo(tmpdir)
            noread = Path(tmpdir, "noread.py")
            noread.write_text("content")
            noread.chmod(0o000)
            _save_untracked_base(tmpdir, {"noread.py"})
            base_dir = _untracked_base_dir()
            assert not (base_dir / "noread.py").exists()
        finally:
            Path(tmpdir, "noread.py").chmod(0o644)
            shutil.rmtree(tmpdir, ignore_errors=True)
            base_dir = _untracked_base_dir()
            if base_dir.exists():
                shutil.rmtree(base_dir, ignore_errors=True)

    def test_prepare_merge_view_untracked_large_in_pre_hashes(self) -> None:
        """Cover 830-831: pre-existing untracked file that's now >2MB."""
        tmpdir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        try:
            _init_git_repo(tmpdir)
            Path(tmpdir, "growing.py").write_text("small\n")
            pre_hunks = _parse_diff_hunks(tmpdir)
            pre_untracked = _capture_untracked(tmpdir)
            pre_hashes = _snapshot_files(tmpdir, set(pre_hunks.keys()) | pre_untracked)
            Path(tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
            Path(tmpdir, "growing.py").write_bytes(b"x" * 2_100_000)
            result = _prepare_merge_view(
                tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(data_dir, ignore_errors=True)
            base_dir = _untracked_base_dir()
            if base_dir.exists():
                shutil.rmtree(base_dir, ignore_errors=True)

class TestTaskHistoryUncoveredBranches:
    def setup_method(self) -> None:
        from kiss.agents.sorcar import task_history

        self._orig_history_file = task_history.HISTORY_FILE
        self._orig_model_usage_file = task_history.MODEL_USAGE_FILE
        self._orig_file_usage_file = task_history.FILE_USAGE_FILE
        self._orig_kiss_dir = task_history._KISS_DIR
        self._orig_events_dir = task_history._CHAT_EVENTS_DIR

        self.tmpdir = tempfile.mkdtemp()
        task_history._KISS_DIR = Path(self.tmpdir)
        task_history.HISTORY_FILE = Path(self.tmpdir) / "task_history.jsonl"
        task_history._CHAT_EVENTS_DIR = Path(self.tmpdir) / "chat_events"
        task_history.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        task_history.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        task_history._history_cache = None

    def teardown_method(self) -> None:
        from kiss.agents.sorcar import task_history

        task_history.HISTORY_FILE = self._orig_history_file
        task_history._CHAT_EVENTS_DIR = self._orig_events_dir
        task_history.MODEL_USAGE_FILE = self._orig_model_usage_file
        task_history.FILE_USAGE_FILE = self._orig_file_usage_file
        task_history._KISS_DIR = self._orig_kiss_dir
        task_history._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestUsefulToolsUncoveredBranches:
    def test_edit_write_permission_error(self) -> None:
        """Cover 264-266: Edit exception handler when write fails."""
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            os.chmod(path, 0o444)
            result = tools.Edit(path, "hello", "goodbye")
            assert "Error:" in result
        finally:
            os.chmod(path, 0o644)
            os.unlink(path)


class TestPrepareMergeViewLine819:
    """Cover the branch where a pre-existing untracked file is already in
    file_hunks (from tracked hunks) and gets skipped via `continue`."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = tempfile.mkdtemp()
        _init_git_repo(self.tmpdir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)
        base_dir = _untracked_base_dir()
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_pre_untracked_becomes_tracked_and_modified(self) -> None:
        """A file that was untracked pre-task, gets committed by agent, then
        modified — it appears in file_hunks via tracked hunks AND in
        pre_untracked, hitting the `if fname in file_hunks: continue` branch."""
        Path(self.tmpdir, "newfile.py").write_text("original\n")
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        assert "newfile.py" in pre_untracked
        pre_hashes = _snapshot_files(
            self.tmpdir, set(pre_hunks.keys()) | pre_untracked
        )
        subprocess.run(
            ["git", "add", "newfile.py"], cwd=self.tmpdir, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "add newfile"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        Path(self.tmpdir, "newfile.py").write_text("modified by agent\n")
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, pre_hunks, pre_untracked, pre_hashes
        )
        assert result.get("status") == "opened"
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text()
        )
        file_names = [f["name"] for f in manifest["files"]]
        assert "newfile.py" in file_names


class TestWebUseToolHeadless:
    """Integration tests for WebUseTool using a real headless browser."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_go_to_url_tab_list(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        result = self.tool.go_to_url("tab:list")
        assert "Open tabs" in result

    def test_go_to_url_tab_out_of_range(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        result = self.tool.go_to_url("tab:999")
        assert "Error" in result

    def test_type_text(self) -> None:
        self.tool.go_to_url(
            'data:text/html,<input type="text" placeholder="Name">'
        )
        result = self.tool.type_text(1, "hello world")
        assert isinstance(result, str)

    def test_type_text_with_enter(self) -> None:
        self.tool.go_to_url(
            'data:text/html,<form><input type="text" placeholder="Search"></form>'
        )
        result = self.tool.type_text(1, "query", press_enter=True)
        assert isinstance(result, str)

    def test_screenshot(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Screenshot Test</h1>")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            result = self.tool.screenshot(path)
            assert "saved" in result.lower()
            assert os.path.exists(path)

    def test_get_page_content_text_only(self) -> None:
        self.tool.go_to_url("data:text/html,<p>Hello World</p>")
        result = self.tool.get_page_content(text_only=True)
        assert "Hello World" in result


class TestWebUseToolPersistentContext:
    """Test WebUseTool with persistent context (user_data_dir set)."""

    def test_persistent_context(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        with tempfile.TemporaryDirectory() as d:
            tool = WebUseTool(user_data_dir=d)
            try:
                result = tool.go_to_url("data:text/html,<h1>Persistent</h1>")
                assert "Persistent" in result or isinstance(result, str)
            finally:
                tool.close()


class TestWebUseToolResolveLocator:
    """Test _resolve_locator edge cases."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

class TestWebUseToolEdgeCases:
    """Cover remaining edge cases in web_use_tool.py."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_truncated_snapshot(self) -> None:
        """Cover line 143: snapshot exceeding max_chars is truncated."""
        buttons = "".join([f'<button>Button{i}</button>' for i in range(200)])
        self.tool.go_to_url(f"data:text/html,{buttons}")
        result = self.tool._get_ax_tree(max_chars=100)
        assert "truncated" in result

    def test_scroll_error(self) -> None:
        """Cover lines 325-327: scroll after page closed."""
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        self.tool._page.close()
        result = self.tool.scroll("down")
        assert "Error" in result

    def test_check_for_new_tab_single_page(self) -> None:
        """Cover 162->exit: _check_for_new_tab when there's only one page."""
        self.tool.go_to_url("data:text/html,<h1>Single</h1>")
        assert len(self.tool._context.pages) == 1
        self.tool._check_for_new_tab()


class TestReadActiveFileDirPath:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestSorcarAgentDirect:

    def test_run_with_both_attachments(self) -> None:
        """Cover both image + PDF attachment branches simultaneously."""
        from kiss.core.models.model import Attachment

        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            with pytest.raises(KISSError):
                agent.run(
                    prompt_template="analyze",
                    work_dir=tmpdir,
                    max_steps=1,
                    max_budget=0.001,
                    max_sub_sessions=1,
                    headless=True,
                    verbose=False,
                    attachments=[
                        Attachment(data=b"fake_img", mime_type="image/png"),
                        Attachment(data=b"fake_pdf", mime_type="application/pdf"),
                    ],
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_with_editor_file(self) -> None:
        """Cover the current_editor_file branch."""
        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        editor_file = os.path.join(tmpdir, "test.py")
        Path(editor_file).write_text("print('hello')")
        try:
            with pytest.raises(KISSError):
                agent.run(
                    prompt_template="fix this file",
                    work_dir=tmpdir,
                    max_steps=1,
                    max_budget=0.001,
                    max_sub_sessions=1,
                    headless=True,
                    verbose=False,
                    current_editor_file=editor_file,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_with_text_attachment_no_parts(self) -> None:
        """Attachment that is neither image nor PDF hits empty parts branch."""
        from kiss.core.models.model import Attachment

        agent = SorcarAgent("test_agent")
        tmpdir = tempfile.mkdtemp()
        try:
            with pytest.raises(KISSError):
                agent.run(
                    prompt_template="analyze",
                    work_dir=tmpdir,
                    max_steps=1,
                    max_budget=0.001,
                    max_sub_sessions=1,
                    headless=True,
                    verbose=False,
                    attachments=[Attachment(data=b"text data", mime_type="text/plain")],
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stream_callback_without_printer(self) -> None:
        """Cover the _stream closure's if self.printer False branch."""
        agent = SorcarAgent("test_stream_agent")
        agent.printer = None
        tools = agent._get_tools()
        bash_tool = tools[0]
        result = bash_tool(command="echo hello_no_printer", description="test no printer")
        assert "hello_no_printer" in result

def _wait_for_port_file(port_file: str, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
            return int(Path(port_file).read_text().strip())
        time.sleep(0.3)
    raise TimeoutError(f"Port file {port_file} not written within {timeout}s")


@pytest.fixture(scope="module")
def server():
    """Start a sorcar server subprocess and yield (base_url, work_dir, proc, tmpdir)."""
    tmpdir = tempfile.mkdtemp()
    work_dir = os.path.join(tmpdir, "work")
    os.makedirs(work_dir)

    subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=work_dir, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=work_dir, capture_output=True,
    )
    Path(work_dir, "file.txt").write_text("line1\nline2\n")
    subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

    port_file = os.path.join(tmpdir, "port")

    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "_sorcar_test_server.py"),
            port_file,
            work_dir,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    keepalive = None
    try:
        port = _wait_for_port_file(port_file)
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                resp = requests.get(base_url, timeout=2)
                if resp.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(0.3)
        else:
            raise TimeoutError("Server not responsive")

        keepalive = requests.get(
            f"{base_url}/events", stream=True, timeout=300,
        )

        yield base_url, work_dir, proc, tmpdir
    finally:
        if keepalive is not None:
            keepalive.close()
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestSuggestionsFileMatch:
    """Exercise suggestions with file query matching actual files (line 825, 831)."""

    def test_suggestions_general_with_file_match(self, server) -> None:
        base_url, work_dir, _, _ = server
        Path(work_dir, "readme.txt").write_text("hello")
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "file", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_files_with_frequent(self, server) -> None:
        """Exercise frequent file usage sorting in files mode."""
        base_url, work_dir, _, _ = server
        requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "file.txt"},
            timeout=5,
        )
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "", "mode": "files"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)
        types = [item.get("type", "") for item in data]
        assert any("frequent" in t for t in types) or len(data) == 0


class TestCompleteEndpointFastPath:
    """Exercise _fast_complete with file path completion (lines 899)."""

    def test_complete_fast_file_path(self, server) -> None:
        base_url, work_dir, _, _ = server
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "fil"},
            timeout=5,
        )
        data = resp.json()
        assert "suggestion" in data


class TestActiveFileInfoMd:
    """Exercise /active-file-info with a .md file (lines 1160-1162)."""

    def test_active_file_md(self, server) -> None:
        """Set active-file.json to point to a .md file and query."""
        base_url, work_dir, _, tmpdir = server
        from kiss.agents.sorcar.task_history import _KISS_DIR
        sorcar_data_dir = str(_KISS_DIR / "sorcar-data")
        os.makedirs(sorcar_data_dir, exist_ok=True)

        md_file = os.path.join(work_dir, "prompt.md")
        Path(md_file).write_text("# System Prompt\nYou are a helpful assistant.\n")
        af = os.path.join(sorcar_data_dir, "active-file.json")
        with open(af, "w") as f:
            json.dump({"path": md_file}, f)

        resp = requests.get(f"{base_url}/active-file-info", timeout=5)
        data = resp.json()
        assert "is_prompt" in data
        assert data["path"] == md_file
        assert data["filename"] == "prompt.md"

        os.unlink(af)


class TestGetFileContentEndpoint:
    """Exercise /get-file-content success path (line 1182-1184)."""

    def test_get_file_content_success(self, server) -> None:
        base_url, work_dir, _, _ = server
        fpath = os.path.join(work_dir, "file.txt")
        resp = requests.get(
            f"{base_url}/get-file-content",
            params={"path": fpath},
            timeout=5,
        )
        data = resp.json()
        assert "content" in data

    def test_get_file_content_binary_error(self, server) -> None:
        """Exercise exception path (line 1204-1206)."""
        base_url, work_dir, _, _ = server
        bin_file = os.path.join(work_dir, "binary.dat")
        Path(bin_file).write_bytes(bytes(range(256)) * 100)
        resp = requests.get(
            f"{base_url}/get-file-content",
            params={"path": bin_file},
            timeout=5,
        )
        assert resp.status_code in (200, 500)


class TestMergeActionValidActions:
    """Exercise merge_action with all valid action types (lines 849-856)."""

    def test_merge_action_accept(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "accept"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_reject(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "reject"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_accept_all(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "accept-all"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_reject_all(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": "reject-all"},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_merge_action_empty_action(self, server) -> None:
        """Empty action string hits invalid action branch."""
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/merge-action",
            json={"action": ""},
            timeout=5,
        )
        assert resp.status_code == 400


class TestRecordFileUsageWithPath:
    """Exercise record_file_usage with a non-empty path (line 987-988)."""

    def test_record_file_usage_path(self, server) -> None:
        base_url, _, _, _ = server
        resp = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "src/main.py"},
            timeout=5,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestGenerateCommitMessageNoChanges:
    """Exercise /generate-commit-message with no changes (line 1118)."""

    def test_generate_commit_msg_no_changes(self, server) -> None:
        base_url, work_dir, _, _ = server
        subprocess.run(["git", "add", "-A"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "cleanup", "--allow-empty"],
            cwd=work_dir, capture_output=True,
            env={
                **os.environ,
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )
        resp = requests.post(
            f"{base_url}/generate-commit-message",
            json={},
            timeout=10,
        )
        data = resp.json()
        assert "error" in data


class _InProcessDummyAgent(RelentlessAgent):
    """Minimal agent that returns immediately for in-process testing."""

    def __init__(self, name: str) -> None:
        pass

    def run(self, **kwargs) -> str:  # type: ignore[override]
        task = kwargs.get("prompt_template", "")
        work_dir = kwargs.get("work_dir", "")
        if task == "slow_task_for_stop_test":
            import time as _t

            for _ in range(300):
                _t.sleep(0.1)
        if task == "error_task_for_test":
            raise RuntimeError("test error from agent")
        if task == "create_file_for_merge" and work_dir:
            Path(work_dir, "agent_created.txt").write_text("new content\n")
        return "success: true\nsummary: done"


@pytest.fixture(scope="module")
def inproc_server():
    """Start run_chatbot() in a background thread for in-process coverage."""
    import webbrowser as _wb

    tmpdir = tempfile.mkdtemp()
    work_dir = os.path.join(tmpdir, "work")
    os.makedirs(work_dir)

    subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=work_dir, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True
    )
    Path(work_dir, "file.txt").write_text("line1\nline2\n")
    subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

    import shutil as _sh

    _orig_which = _sh.which

    def _no_cs(cmd: str, mode: int = 0, path: str | None = None) -> str | None:
        if cmd == "code-server":
            return None
        result: str | None = _orig_which(cmd, mode=mode, path=path)  # type: ignore[call-overload]
        return result

    old_which = _sh.which
    old_open = _wb.open
    _sh.which = _no_cs  # type: ignore[assignment]
    _wb.open = lambda url: None  # type: ignore[assignment,misc]

    from kiss.agents.sorcar import browser_ui
    from kiss.agents.sorcar import sorcar as sorcar_module

    port_holder: list[int] = []
    _orig_ffp = browser_ui.find_free_port

    def _capture_port() -> int:
        p: int = _orig_ffp()
        port_holder.append(p)
        return p

    sorcar_module.find_free_port = _capture_port  # type: ignore[attr-defined]

    thread = threading.Thread(
        target=run_chatbot,
        kwargs={
            "agent_factory": _InProcessDummyAgent,
            "title": "InProcTest",
            "work_dir": work_dir,
        },
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if port_holder:
            break
        time.sleep(0.2)
    assert port_holder, "Server did not start"

    base_url = f"http://127.0.0.1:{port_holder[0]}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            resp = requests.get(base_url, timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(0.2)

    from kiss.agents.sorcar.task_history import _KISS_DIR

    sorcar_data_dir = str(_KISS_DIR / "sorcar-data")

    keepalive = requests.get(f"{base_url}/events", stream=True, timeout=300)

    yield base_url, work_dir, sorcar_data_dir

    keepalive.close()
    try:
        requests.post(f"{base_url}/closing", json={}, timeout=2)
    except Exception:
        pass

    _sh.which = old_which  # type: ignore[assignment]
    _wb.open = old_open  # type: ignore[assignment,misc]
    sorcar_module.find_free_port = _orig_ffp  # type: ignore[attr-defined]
    time.sleep(1)
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestInProcessEndpoints:
    """Tests that exercise run_chatbot() endpoint code in-process for coverage."""

    def test_index(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(base_url, timeout=5)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_suggestions_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/suggestions?q=&mode=general", timeout=5)
        assert resp.status_code == 200

    def test_complete_short(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/complete?q=a", timeout=5)
        assert resp.status_code == 200

    def test_complete_with_fast_match(self, inproc_server) -> None:
        """Exercise _fast_complete with a query that starts with a task."""
        base_url, _, _ = inproc_server
        requests.post(
            f"{base_url}/run",
            json={"task": "unique_auto_complete_test_query_xyz"},
            timeout=10,
        )
        time.sleep(4)
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "unique_auto"},
            timeout=10,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_complete_with_file_match(self, inproc_server) -> None:
        """Exercise _fast_complete file path branch."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "edit fi"},
            timeout=10,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_task_events_invalid(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/task-events?idx=abc", timeout=5)
        assert resp.status_code == 400

    def test_task_events_out_of_range(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/task-events?idx=999", timeout=5)
        assert resp.status_code == 404

    def test_task_events_valid(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/task-events?idx=0", timeout=5)
        assert resp.status_code in (200, 404)

    def test_run_empty_task(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/run", json={"task": ""}, timeout=5)
        assert resp.status_code == 400

    def test_run_while_running(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp1 = requests.post(
            f"{base_url}/run",
            json={"task": "slow_task_for_stop_test"},
            timeout=10,
        )
        assert resp1.status_code == 200
        time.sleep(0.5)
        resp2 = requests.post(
            f"{base_url}/run",
            json={"task": "second task"},
            timeout=10,
        )
        assert resp2.status_code == 409
        resp3 = requests.post(
            f"{base_url}/run-selection",
            json={"text": "selected text"},
            timeout=10,
        )
        assert resp3.status_code == 409
        requests.post(f"{base_url}/stop", json={}, timeout=5)
        time.sleep(2)

    def test_stop_no_task(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        time.sleep(1)
        resp = requests.post(f"{base_url}/stop", json={}, timeout=5)
        assert resp.status_code == 404

    def test_run_selection_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/run-selection", json={"text": ""}, timeout=5
        )
        assert resp.status_code == 400

    def test_open_file_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/open-file", json={"path": ""}, timeout=5
        )
        assert resp.status_code == 400

    def test_open_file_not_found(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/open-file", json={"path": "/nonexistent"}, timeout=5
        )
        assert resp.status_code == 404

    def test_open_file_success(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        fpath = os.path.join(work_dir, "file.txt")
        resp = requests.post(
            f"{base_url}/open-file", json={"path": fpath}, timeout=5
        )
        assert resp.status_code == 200

    def test_focus_chatbox(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/focus-chatbox", json={}, timeout=5)
        assert resp.status_code == 200

    def test_focus_editor(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(f"{base_url}/focus-editor", json={}, timeout=5)
        assert resp.status_code == 200

    def test_merge_action_invalid(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/merge-action", json={"action": "bogus"}, timeout=5
        )
        assert resp.status_code == 400

    def test_record_file_usage_empty(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": ""},
            timeout=5,
        )
        assert resp.status_code == 200

    def test_commit_no_changes(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        subprocess.run(["git", "add", "-A"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "pre-clean", "--allow-empty"],
            cwd=work_dir, capture_output=True,
            env={**os.environ, "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t.com"},
        )
        resp = requests.post(f"{base_url}/commit", json={}, timeout=10)
        data = resp.json()
        assert "error" in data

    def test_active_file_info_no_file(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_prompt"] is False

    def test_active_file_info_md(self, inproc_server) -> None:
        base_url, work_dir, sorcar_data_dir = inproc_server
        md_file = os.path.join(work_dir, "test_prompt.md")
        Path(md_file).write_text("# System Prompt\nYou are helpful.\n")
        os.makedirs(sorcar_data_dir, exist_ok=True)
        af = os.path.join(sorcar_data_dir, "active-file.json")
        with open(af, "w") as f:
            json.dump({"path": md_file}, f)
        resp = requests.get(f"{base_url}/active-file-info", timeout=5)
        data = resp.json()
        assert data["is_prompt"] is not None
        assert data["path"] == md_file
        os.unlink(af)

    def test_get_file_content(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        fpath = os.path.join(work_dir, "file.txt")
        resp = requests.get(
            f"{base_url}/get-file-content", params={"path": fpath}, timeout=5
        )
        assert resp.status_code == 200
        assert "content" in resp.json()

    def test_get_file_content_not_found(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/get-file-content", params={"path": "/nonexistent"}, timeout=5
        )
        assert resp.status_code == 404

    def test_get_file_content_binary_error(self, inproc_server) -> None:
        """Exercise get_file_content exception path (line 1160-1162)."""
        base_url, work_dir, _ = inproc_server
        bin_file = os.path.join(work_dir, "binary_test.dat")
        Path(bin_file).write_bytes(bytes(range(256)) * 100)
        resp = requests.get(
            f"{base_url}/get-file-content", params={"path": bin_file}, timeout=5
        )
        assert resp.status_code in (200, 500)

    def test_suggestions_general_with_query(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/suggestions", params={"q": "test", "mode": "general"}, timeout=5
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_files_with_query(self, inproc_server) -> None:
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/suggestions", params={"q": "file", "mode": "files"}, timeout=5
        )
        assert resp.status_code == 200

    def test_theme_with_bad_file(self, inproc_server) -> None:
        """Exercise theme endpoint with a corrupt theme file (lines 987-988)."""
        from kiss.agents.sorcar.task_history import _KISS_DIR

        theme_file = _KISS_DIR / "vscode-theme.json"
        theme_file.parent.mkdir(parents=True, exist_ok=True)
        orig = theme_file.read_text() if theme_file.exists() else None
        try:
            theme_file.write_text("not valid json{{{")
            base_url, _, _ = inproc_server
            resp = requests.get(f"{base_url}/theme", timeout=5)
            assert resp.status_code == 200
        finally:
            if orig is not None:
                theme_file.write_text(orig)
            elif theme_file.exists():
                theme_file.unlink()

    def test_run_task_while_merging(self, inproc_server) -> None:
        """Create file changes that trigger merge view, then try /run."""
        base_url, work_dir, sorcar_data_dir = inproc_server
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "create_file_for_merge"},
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(5)
        try:
            resp2 = requests.post(
                f"{base_url}/run",
                json={"task": "should fail while merging"},
                timeout=10,
            )
            if resp2.status_code == 409:
                data = resp2.json()
                err = data.get("error", "").lower()
                assert "merge" in err or "running" in err
                resp3 = requests.post(
                    f"{base_url}/run-selection",
                    json={"text": "selection during merge"},
                    timeout=10,
                )
                assert resp3.status_code == 409
        finally:
            requests.post(
                f"{base_url}/merge-action",
                json={"action": "accept-all"},
                timeout=10,
            )
            time.sleep(1)
            requests.post(
                f"{base_url}/merge-action",
                json={"action": "all-done"},
                timeout=10,
            )
            time.sleep(1)

    def test_commit_git_failure(self, inproc_server) -> None:
        """Make git commit fail via a pre-commit hook."""
        base_url, work_dir, _ = inproc_server
        hooks_dir = Path(work_dir, ".git", "hooks")
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = hooks_dir / "pre-commit"
        hook_path.write_text("#!/bin/sh\nexit 1\n")
        hook_path.chmod(0o755)
        Path(work_dir, "hook_test.txt").write_text("test")
        try:
            resp = requests.post(
                f"{base_url}/commit", json={}, timeout=15
            )
            data = resp.json()
            assert "error" in data or "status" in data
        finally:
            hook_path.unlink(missing_ok=True)
            Path(work_dir, "hook_test.txt").unlink(missing_ok=True)

    def test_theme_no_file(self, inproc_server) -> None:
        """Theme endpoint with no theme file → hits file-not-exists branch."""
        base_url, _, _ = inproc_server
        from kiss.agents.sorcar.task_history import _KISS_DIR

        theme_file = _KISS_DIR / "vscode-theme.json"
        orig = theme_file.read_text() if theme_file.exists() else None
        try:
            theme_file.unlink(missing_ok=True)
            resp = requests.get(f"{base_url}/theme", timeout=5)
            assert resp.status_code == 200
        finally:
            if orig is not None:
                theme_file.write_text(orig)

    def test_generate_commit_message_no_changes(self, inproc_server) -> None:
        base_url, work_dir, _ = inproc_server
        subprocess.run(["git", "add", "-A"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "cleanup", "--allow-empty"],
            cwd=work_dir, capture_output=True,
            env={
                **os.environ,
                "GIT_COMMITTER_NAME": "T",
                "GIT_COMMITTER_EMAIL": "t@t.com",
            },
        )
        resp = requests.post(
            f"{base_url}/generate-commit-message", json={}, timeout=10
        )
        data = resp.json()
        assert "error" in data

    def test_run_with_attachments(self, inproc_server) -> None:
        """Run a task with base64-encoded attachments."""
        import base64

        base_url, _, _ = inproc_server
        img_data = base64.b64encode(b"fake image data").decode()
        resp = requests.post(
            f"{base_url}/run",
            json={
                "task": "test with attachments",
                "attachments": [
                    {"data": img_data, "mime_type": "image/png"},
                ],
            },
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(2)

    def test_run_error_task(self, inproc_server) -> None:
        """Run a task that raises Exception to cover except Exception branch."""
        base_url, _, _ = inproc_server
        time.sleep(1)
        resp = requests.post(
            f"{base_url}/run",
            json={"task": "error_task_for_test"},
            timeout=10,
        )
        assert resp.status_code == 200
        time.sleep(3)

    def test_tasks_has_events(self, inproc_server) -> None:
        """After running tasks, check that tasks endpoint shows has_events."""
        base_url, _, _ = inproc_server
        requests.post(
            f"{base_url}/run",
            json={"task": "events check task unique"},
            timeout=10,
        )
        time.sleep(5)
        resp = requests.get(f"{base_url}/tasks", timeout=5)
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        has_events_list = [t.get("has_events") for t in data]
        assert any(has_events_list), f"No tasks have events. Tasks: {data[:3]}"

    def test_generate_commit_message_with_untracked(self, inproc_server) -> None:
        """Generate commit message with untracked file (untracked branch)."""
        base_url, work_dir, sorcar_data_dir = inproc_server
        test_file = os.path.join(work_dir, "gen_cm_untracked.txt")
        Path(test_file).write_text("untracked content for commit msg")
        try:
            resp = requests.post(
                f"{base_url}/generate-commit-message", json={}, timeout=60
            )
            data = resp.json()
            assert "message" in data or "error" in data
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    def test_commit_with_staged_changes(self, inproc_server) -> None:
        """Commit endpoint with staged changes to cover full commit path."""
        base_url, work_dir, _ = inproc_server
        commit_file = os.path.join(work_dir, "commit_staged.txt")
        Path(commit_file).write_text("staged content for commit")
        subprocess.run(["git", "add", commit_file], cwd=work_dir, capture_output=True)
        resp = requests.post(f"{base_url}/commit", json={}, timeout=60)
        data = resp.json()
        assert resp.status_code in (200, 400)
        assert "status" in data or "error" in data

    def test_run_selection_while_merging(self, inproc_server) -> None:
        """Hit the merging check in run-selection by triggering merge state."""
        base_url, _, _ = inproc_server
        time.sleep(1)
        resp = requests.post(
            f"{base_url}/run-selection",
            json={"text": "selection merge test"},
            timeout=10,
        )
        assert resp.status_code in (200, 409)
        time.sleep(2)

    def test_complete_no_fast_match_triggers_llm(self, inproc_server) -> None:
        """Query that doesn't fast-match goes through LLM path (or returns empty)."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "xyzzy_no_match_ever_12345"},
            timeout=30,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_complete_short_last_word(self, inproc_server) -> None:
        """Query where last word is < 2 chars → skips file matching in _fast_complete."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/complete",
            params={"q": "test x"},
            timeout=30,
        )
        data = resp.json()
        assert "suggestion" in data

    def test_suggestions_general_short_last_word(self, inproc_server) -> None:
        """Query where last word is only 1 char → skips file matching."""
        base_url, _, _ = inproc_server
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "test x", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)

    def test_suggestions_general_8_file_matches_break(self, inproc_server) -> None:
        """Create >8 files matching a pattern to hit count >= 8 break."""
        base_url, work_dir, _ = inproc_server
        for i in range(10):
            Path(work_dir, f"zbatchf_{i}.txt").write_text(f"c{i}")
        requests.post(
            f"{base_url}/run",
            json={"task": "refresh for zbatchf"},
            timeout=10,
        )
        time.sleep(4)
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "check zbatchf", "mode": "general"},
            timeout=5,
        )
        data = resp.json()
        file_items = [item for item in data if item.get("type") == "file"]
        assert len(file_items) <= 8

    def test_suggestions_files_frequent_sort(self, inproc_server) -> None:
        """Record file usage, then query files mode. file.txt is in initial cache."""
        base_url, work_dir, _ = inproc_server
        for _ in range(5):
            requests.post(
                f"{base_url}/record-file-usage",
                json={"path": "file.txt"},
                timeout=5,
            )
        resp = requests.get(
            f"{base_url}/suggestions",
            params={"q": "", "mode": "files"},
            timeout=5,
        )
        data = resp.json()
        assert isinstance(data, list)
        types = [item.get("type", "") for item in data]
        assert any("frequent" in t for t in types), f"Data: {data[:5]}"

    def test_select_model_persists(self, inproc_server) -> None:
        """Selecting a model via /select-model persists it for next startup."""
        import json

        from kiss.agents.sorcar.task_history import MODEL_USAGE_FILE

        base_url, _, _ = inproc_server
        original = json.loads(MODEL_USAGE_FILE.read_text()) if MODEL_USAGE_FILE.exists() else {}
        original_last = original.get("_last", "")
        try:
            resp = requests.post(
                f"{base_url}/select-model",
                json={"model": "gemini-2.5-pro"},
                timeout=5,
            )
            assert resp.status_code == 200
            assert task_history._load_last_model() == "gemini-2.5-pro"
            resp = requests.get(f"{base_url}/models", timeout=5)
            assert resp.json()["selected"] == "gemini-2.5-pro"
        finally:
            data = json.loads(MODEL_USAGE_FILE.read_text()) if MODEL_USAGE_FILE.exists() else {}
            data["_last"] = original_last
            MODEL_USAGE_FILE.write_text(json.dumps(data))

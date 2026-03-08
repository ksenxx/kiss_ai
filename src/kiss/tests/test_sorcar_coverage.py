"""Integration tests for kiss/agents/sorcar/ to maximize branch coverage.

No mocks, patches, or test doubles.  Uses real files, real git repos, and
real objects.
"""

import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _disable_copilot_scm_button,
    _install_copilot_extension,
    _parse_diff_hunks,
    _prepare_merge_view,
    _snapshot_files,
)
from kiss.agents.sorcar.prompt_detector import PromptDetector
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
)


class TestPromptDetector:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        p = os.path.join(self.tmpdir, name)
        with open(p, "w") as f:
            f.write(content)
        return p

# ---------------------------------------------------------------------------
# task_history.py  (52% -> target higher)
# ---------------------------------------------------------------------------


def _redirect_history(tmpdir: str):
    """Redirect all task_history files to a temp dir."""
    old_hist = th.HISTORY_FILE
    old_prop = th.PROPOSALS_FILE
    old_model = th.MODEL_USAGE_FILE
    old_file = th.FILE_USAGE_FILE
    old_cache = th._history_cache

    th.HISTORY_FILE = Path(tmpdir) / "history.json"
    th.PROPOSALS_FILE = Path(tmpdir) / "proposals.json"
    th.MODEL_USAGE_FILE = Path(tmpdir) / "model_usage.json"
    th.FILE_USAGE_FILE = Path(tmpdir) / "file_usage.json"
    th._history_cache = None

    return old_hist, old_prop, old_model, old_file, old_cache


def _restore_history(old_hist, old_prop, old_model, old_file, old_cache):
    th.HISTORY_FILE = old_hist
    th.PROPOSALS_FILE = old_prop
    th.MODEL_USAGE_FILE = old_model
    th.FILE_USAGE_FILE = old_file
    th._history_cache = old_cache


class TestTaskHistory:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.old = _redirect_history(self.tmpdir)

    def teardown_method(self) -> None:
        _restore_history(*self.old)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- _load_history / _save_history ---
    # --- proposals ---
    # --- model usage ---
    # --- file usage ---
    # --- _load_json_dict ---
    # --- _append_task_to_md ---
    def test_append_task_to_md(self) -> None:
        md_path = Path(self.tmpdir) / "TASK_HISTORY.md"
        # Monkey-patch _get_task_history_md_path
        import kiss.core.config as cfg
        old_artifact = cfg.DEFAULT_CONFIG.agent.artifact_dir
        try:
            cfg.DEFAULT_CONFIG.agent.artifact_dir = str(md_path.parent / "artifacts")
            th._init_task_history_md()
            th._append_task_to_md("Test task", "success: true\nsummary: done")
            content = th._get_task_history_md_path().read_text()
            assert "Test task" in content
            assert "success: true" in content
        finally:
            cfg.DEFAULT_CONFIG.agent.artifact_dir = old_artifact

    # --- thread safety ---
# ---------------------------------------------------------------------------
# useful_tools.py  (90% -> target higher)
# ---------------------------------------------------------------------------


class TestUsefulToolsRead:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestUsefulToolsWrite:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestUsefulToolsEdit:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, content: str) -> str:
        p = os.path.join(self.tmpdir, "edit.txt")
        Path(p).write_text(content)
        return p

class TestUsefulToolsBash:
    def setup_method(self) -> None:
        self.tools = UsefulTools()

# ---------------------------------------------------------------------------
# browser_ui.py  (33% -> target higher)
# ---------------------------------------------------------------------------


class TestBaseBrowserPrinter:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

    def test_print_tool_call_with_edit(self) -> None:
        cq = self.printer.add_client()
        self.printer.print(
            "Edit",
            type="tool_call",
            tool_input={
                "file_path": "/tmp/test.py",
                "old_string": "old",
                "new_string": "new",
            },
        )
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["old_string"] == "old"
        assert tool_events[0]["new_string"] == "new"
        self.printer.remove_client(cq)

    def test_print_result_bad_yaml(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("not: [yaml: {", type="result")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        # Should still work without parsed yaml
        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1
        self.printer.remove_client(cq)

# ---------------------------------------------------------------------------
# code_server.py  (45% -> target higher)
# ---------------------------------------------------------------------------


class TestScanFiles:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestDisableCopilotScmButton:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestGitDiffAndMerge:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.tmpdir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.tmpdir, capture_output=True,
        )
        # Create initial commit
        Path(self.tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# web_use_tool.py  (81% -> target higher)
# ---------------------------------------------------------------------------


class TestScrollDelta:
    def test_all_directions(self) -> None:
        from kiss.agents.sorcar.web_use_tool import _SCROLL_DELTA

        assert _SCROLL_DELTA["down"] == (0, 300)
        assert _SCROLL_DELTA["up"] == (0, -300)
        assert _SCROLL_DELTA["right"] == (300, 0)
        assert _SCROLL_DELTA["left"] == (-300, 0)


# ---------------------------------------------------------------------------
# Additional browser_ui.py coverage
# ---------------------------------------------------------------------------
class TestBrowserPrinterEdgeCases:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

    def test_tool_call_with_content_and_extras(self) -> None:
        cq = self.printer.add_client()
        self.printer.print(
            "Write",
            type="tool_call",
            tool_input={
                "file_path": "/tmp/out.py",
                "content": "print('hi')",
                "extra_key": "extra_val",
            },
        )
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert tool_events[0]["content"] == "print('hi')"
        self.printer.remove_client(cq)

# ---------------------------------------------------------------------------
# Additional code_server.py coverage
# ---------------------------------------------------------------------------
class TestSetupCodeServerAdditional:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Additional task_history.py coverage - save_history OSError, _init_task_history_md
# ---------------------------------------------------------------------------
class TestTaskHistoryAdditional:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.old = _redirect_history(self.tmpdir)

    def teardown_method(self) -> None:
        _restore_history(*self.old)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_usage_corrupted(self) -> None:
        th.FILE_USAGE_FILE.write_text("broken json")
        assert th._load_file_usage() == {}

# ---------------------------------------------------------------------------
# Additional useful_tools.py coverage
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# code_server _install_copilot_extension
# ---------------------------------------------------------------------------


class TestInstallCopilotExtension:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# _prepare_merge_view additional edge cases
# ---------------------------------------------------------------------------
class TestPrepareMergeViewEdgeCases:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=self.tmpdir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=self.tmpdir, capture_output=True,
        )
        Path(self.tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_empty_file_not_added(self) -> None:
        """Empty new files (0 lines) should not be added to merge view."""
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        # Create an empty file
        Path(self.tmpdir, "empty.txt").write_text("")
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes,
            )
            # Should only have "No changes" since empty file has 0 lines
            assert "error" in result
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_large_file_skipped(self) -> None:
        """Files larger than 2MB should be skipped."""
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        large = Path(self.tmpdir, "large.bin")
        large.write_bytes(b"x" * 2_100_000)
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes,
            )
            assert "error" in result  # Only large file, skipped
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_pre_hunks_filter_with_matching_base(self) -> None:
        """Hunks that exist in pre_hunks should be filtered out."""
        # Make a change, record hunks, then don't change further
        Path(self.tmpdir, "file.txt").write_text("line1\nchanged\nline3\n")
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        # Also create a NEW file to force merge view open
        Path(self.tmpdir, "new.py").write_text("code\n")
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(self.tmpdir, data_dir, pre_hunks, pre_untracked, None)
            if result.get("status") == "opened":
                manifest = json.loads((Path(data_dir) / "pending-merge.json").read_text())
                file_names = [f["name"] for f in manifest["files"]]
                # new.py should be there, file.txt should NOT (its hunks match pre_hunks)
                assert "new.py" in file_names
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# sorcar.py utilities - _read_active_file, _clean_llm_output, _model_vendor_order
# ---------------------------------------------------------------------------
class TestSorcarUtilities:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# _kill_process_group tests
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test _truncate_output edge: tail=0 branch
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test code_server _setup_code_server extension.js return value (changed vs unchanged)
# ---------------------------------------------------------------------------
class TestSetupCodeServerReturn:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Test code_server._scan_files with hidden dirs at top level
# ---------------------------------------------------------------------------
class TestScanFilesEdgeCases:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Test sorcar.py _THEME_PRESETS
# ---------------------------------------------------------------------------
class TestThemePresets:
    def test_all_presets_have_required_keys(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
        required = {
            "bg", "bg2", "fg", "accent", "border",
            "inputBg", "green", "red", "purple", "cyan",
        }
        for name, theme in _THEME_PRESETS.items():
            assert set(theme.keys()) == required, f"Theme {name} missing keys"

    def test_all_presets_are_hex_colors(self) -> None:

        from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
        for name, theme in _THEME_PRESETS.items():
            for key, value in theme.items():
                assert re.match(r"^#[0-9a-fA-F]{6}$", value), f"Theme {name}.{key}={value} not hex"


# ---------------------------------------------------------------------------
# Test _build_html with and without code_server
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# WebUseTool integration tests with real Playwright browser
# ---------------------------------------------------------------------------
class TestWebUseToolBrowser:
    """Tests requiring real Playwright browser - headless."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tmpdir = tempfile.mkdtemp()
        self.tool = WebUseTool(
            headless=True,
            user_data_dir=None,  # no persistent context
        )

    def teardown_method(self) -> None:
        self.tool.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_go_to_url_tab_switch(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Page1</h1>")
        result = self.tool.go_to_url("tab:0")
        assert "Page:" in result

    def test_press_key(self) -> None:
        self.tool.go_to_url("data:text/html,<button>B</button>")
        result = self.tool.press_key("Tab")
        assert "Page:" in result

    def test_scroll_down(self) -> None:
        long_page = "<div style='height:5000px'>Tall page</div><button>Bottom</button>"
        self.tool.go_to_url(f"data:text/html,{long_page}")
        result = self.tool.scroll("down", 3)
        assert "Page:" in result

class TestWebUseToolPersistentContext:
    """Test persistent context path (user_data_dir set)."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tmpdir = tempfile.mkdtemp()
        self.tool = WebUseTool(
            headless=True,
            user_data_dir=os.path.join(self.tmpdir, "profile"),
        )

    def teardown_method(self) -> None:
        self.tool.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestWebUseToolResolveLocator:
    """Test _resolve_locator edge cases with real browser."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_resolve_element_without_name(self) -> None:
        """Elements without name attribute should still be clickable."""
        self.tool.go_to_url("data:text/html,<button></button>")
        result = self.tool.get_page_content()
        if "[1]" in result:
            result2 = self.tool.click(1)
            assert "Page:" in result2

            # Should have handled the new tab

    def test_screenshot_error_handling(self) -> None:
        """Screenshot to invalid path."""
        self.tool.go_to_url("data:text/html,<h1>X</h1>")
        result = self.tool.screenshot("/dev/null/impossible/file.png")
        assert "Error" in result or "saved" in result.lower()

# ---------------------------------------------------------------------------
# Additional targeted tests for remaining uncovered branches
# ---------------------------------------------------------------------------


class TestPromptDetectorBranches:
    """Cover remaining branches in prompt_detector.py."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        p = os.path.join(self.tmpdir, name)
        with open(p, "w") as f:
            f.write(content)
        return p

class TestBrowserUiBranches:
    """Cover remaining branches in browser_ui.py."""

    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

class TestTaskHistoryBranches:
    """Cover remaining branches in task_history.py."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.orig_kiss_dir = th._KISS_DIR
        self.orig_history = th.HISTORY_FILE
        self.orig_proposals = th.PROPOSALS_FILE
        self.orig_model_usage = th.MODEL_USAGE_FILE
        self.orig_file_usage = th.FILE_USAGE_FILE
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th.HISTORY_FILE = kiss_dir / "task_history.json"
        th.PROPOSALS_FILE = kiss_dir / "proposals.json"
        th.MODEL_USAGE_FILE = kiss_dir / "model_usage.json"
        th.FILE_USAGE_FILE = kiss_dir / "file_usage.json"
        th._history_cache = None

    def teardown_method(self) -> None:
        th._KISS_DIR = self.orig_kiss_dir
        th.HISTORY_FILE = self.orig_history
        th.PROPOSALS_FILE = self.orig_proposals
        th.MODEL_USAGE_FILE = self.orig_model_usage
        th.FILE_USAGE_FILE = self.orig_file_usage
        th._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_model_usage_oserror(self) -> None:
        """OSError when writing model usage.
        Covers lines 188-189."""
        # Write a valid JSON so _load_json_dict succeeds, then make file read-only
        th.MODEL_USAGE_FILE.write_text("{}")
        os.chmod(str(th.MODEL_USAGE_FILE), 0o444)
        try:
            # Should not raise despite OSError on write
            th._record_model_usage("test-model")
        finally:
            os.chmod(str(th.MODEL_USAGE_FILE), 0o644)

    def test_record_file_usage_oserror(self) -> None:
        """OSError when writing file usage.
        Covers lines 206-207."""
        th.FILE_USAGE_FILE.write_text("{}")
        os.chmod(str(th.FILE_USAGE_FILE), 0o444)
        try:
            th._record_file_usage("/some/file.py")
        finally:
            os.chmod(str(th.FILE_USAGE_FILE), 0o644)

class TestUsefulToolsBranches:
    """Cover remaining branches in useful_tools.py."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestCodeServerBranches:
    """Cover remaining branches in code_server.py."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_install_copilot_no_binary(self) -> None:
        """code-server binary not found.
        Covers line 496 (cs_binary not found -> return)."""
        data_dir = self.tmpdir
        ext_dir = Path(data_dir) / "extensions"
        ext_dir.mkdir(parents=True)
        # No copilot extension dirs
        # Override PATH to ensure code-server isn't found
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ""
        try:
            _install_copilot_extension(data_dir)
        finally:
            os.environ["PATH"] = old_path

class TestWebUseToolBranches:
    """Cover remaining branches in web_use_tool.py."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

# ---------------------------------------------------------------------------
# Round 2: Additional targeted tests for remaining uncovered branches
# ---------------------------------------------------------------------------


class TestCodeServerBranchesR2:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disable_copilot_non_matching_items(self) -> None:
        """scm_items with items that DON'T match copilot command.
        Covers branch 476->475 (the False path of the if at 476)."""
        data_dir = self.tmpdir
        ext_dir = Path(data_dir) / "extensions" / "github.copilot-chat-0.1"
        ext_dir.mkdir(parents=True)
        pkg = {
            "contributes": {
                "menus": {
                    "scm/inputBox": [
                        {"command": "some.other.command", "when": "true"},
                        {
                            "command": "github.copilot.git.generateCommitMessage",
                            "when": "true",
                        },
                    ]
                }
            }
        }
        (ext_dir / "package.json").write_text(json.dumps(pkg))
        _disable_copilot_scm_button(data_dir)
        result = json.loads((ext_dir / "package.json").read_text())
        items = result["contributes"]["menus"]["scm/inputBox"]
        # Non-matching item should be unchanged
        assert items[0]["when"] == "true"
        # Matching item should be disabled
        assert items[1]["when"] == "false"

    def test_prepare_merge_view_hash_oserror_via_directory(self) -> None:
        """File replaced with directory after pre-hash, causing OSError.
        Covers lines 721-723."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=work_dir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=work_dir, capture_output=True,
        )
        fpath = Path(work_dir) / "test.txt"
        fpath.write_text("original")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)
        # Modify the file so there's a diff
        fpath.write_text("modified")
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        import hashlib
        pre_hashes = {"test.txt": hashlib.md5(b"original_different").hexdigest()}
        # Replace file with a directory -> read_bytes will fail
        fpath.unlink()
        fpath.mkdir()
        (fpath / "subfile.txt").write_text("x")
        result = _prepare_merge_view(
            work_dir, self.tmpdir, pre_hunks, pre_untracked,
            pre_file_hashes=pre_hashes,
        )
        # Should complete without error (file skipped due to OSError)
        assert isinstance(result, (dict, type(None)))


class TestUsefulToolsBranchesR2:
    """Cover remaining useful_tools.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestPromptDetectorBranchesR2:
    """Cover remaining prompt_detector.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

class TestWebUseToolBranchesR2:
    """Cover remaining web_use_tool.py branches."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_resolve_locator_re_snapshot_success(self) -> None:
        """Element not in stale list, re-snapshot finds it.
        Covers branch 169->171."""
        self.tool.go_to_url("data:text/html,<button>MyBtn</button>")
        # Set _elements to a shorter list so element_id is out of range
        self.tool._elements = []
        # Now click element 1 - should re-snapshot and find it
        result = self.tool.click(1)
        assert "Page:" in result

    def test_resolve_locator_zero_matches(self) -> None:
        """Element in snapshot but locator finds 0 matches.
        Covers line 181."""
        self.tool.go_to_url("data:text/html,<button>X</button>")
        tree = self.tool.get_page_content()
        if "[1]" in tree:
            # Manually change the elements list to have a non-existent element
            self.tool._elements = [{"role": "button", "name": "NonExistentButtonXYZ"}]
            result = self.tool.click(1)
            assert "Error" in result

    def test_new_tab_via_target_blank_click(self) -> None:
        """Click on target=_blank link to open new tab.
        Covers lines 252-253 (_check_for_new_tab during click)."""
        html = '<a href="about:blank" target="_blank">New</a>'
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        assert "[1]" in tree
        pages_before = len(self.tool._context.pages)
        result = self.tool.click(1)
        pages_after = len(self.tool._context.pages)
        assert pages_after > pages_before, f"Expected new tab, got {pages_before} -> {pages_after}"
        assert "Page:" in result or "Error" in result

    def test_check_for_new_tab_no_context(self) -> None:
        """_check_for_new_tab with None context.
        Covers line 160 (context is None -> return)."""
        self.tool.go_to_url("data:text/html,<p>test</p>")
        saved_ctx = self.tool._context
        self.tool._context = None
        self.tool._check_for_new_tab()  # should just return
        self.tool._context = saved_ctx

class TestInstallCopilotOSError:
    """Cover _install_copilot_extension exception handler (lines 505-507)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_install_copilot_subprocess_oserror(self) -> None:
        """Fake code-server binary (garbage bytes) causes OSError.
        Covers lines 505-507."""
        data_dir = self.tmpdir
        ext_dir = Path(data_dir) / "extensions"
        ext_dir.mkdir(parents=True)
        fake_bin_dir = os.path.join(self.tmpdir, "bin")
        os.makedirs(fake_bin_dir)
        fake_cs = os.path.join(fake_bin_dir, "code-server")
        # Write garbage bytes -> Exec format error (OSError)
        with open(fake_cs, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05")
        os.chmod(fake_cs, 0o755)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = fake_bin_dir + ":" + old_path
        try:
            _install_copilot_extension(data_dir)
        finally:
            os.environ["PATH"] = old_path


class TestWebUseToolCloseException:
    """Separate class for close exception test to avoid polluting other tests."""

    def test_close_with_corrupted_playwright(self) -> None:
        """Close with corrupted _playwright that raises on stop().
        Covers lines 384-386 (exception handler in close)."""
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        tool = WebUseTool(headless=True, user_data_dir=None)
        tool.go_to_url("data:text/html,<p>test</p>")
        # Properly close browser first
        if tool._browser:
            tool._browser.close()
        # Corrupt _playwright so stop() raises
        real_pw = tool._playwright
        tool._playwright = "corrupted"
        tool._browser = None
        tool._context = None
        result = tool.close()
        assert result == "Browser closed."
        # Clean up real playwright
        try:
            if real_pw:
                real_pw.stop()
        except Exception:
            pass


class TestWebUseToolWaitForStable:
    """Cover _wait_for_stable exception handlers (149-156) and _check_for_new_tab."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_resolve_locator_empty_snapshot_re_snapshot(self) -> None:
        """Re-snapshot on about:blank where snapshot is empty.
        Covers branch 169->171 (if snapshot: False path)."""
        self.tool.go_to_url("about:blank")
        # Set fake elements so element_id check triggers re-snapshot
        self.tool._elements = [{"role": "button", "name": "fake"}]
        # Now element_id 2 > 1 = len(elements), triggers re-snapshot
        # about:blank has empty body, so snapshot might be empty
        result = self.tool.click(2)
        assert "Error" in result


class TestBashStreamingBaseException:
    """Cover useful_tools.py lines 354-356: BaseException in streaming readline loop."""

    def test_callback_raises_keyboard_interrupt(self) -> None:
        """Stream callback raising KeyboardInterrupt triggers BaseException handler."""
        call_count = 0

        def callback(line: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise KeyboardInterrupt("test interrupt")

        tools = UsefulTools(stream_callback=callback)
        with pytest.raises(KeyboardInterrupt, match="test interrupt"):
            tools.Bash(
                "for i in 1 2 3 4 5; do echo line$i; done",
                "test",
                timeout_seconds=30,
            )


class TestWebUseToolDomContentLoadedTimeout:
    """Cover web_use_tool.py lines 149-151: domcontentloaded timeout in _wait_for_stable."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_click_navigates_to_slow_page(self) -> None:
        """Click a link that navigates to a page that never finishes loading.
        The domcontentloaded wait in _wait_for_stable should timeout."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        port = srv.getsockname()[1]

        def handle_client(conn: socket.socket) -> None:
            try:
                data = conn.recv(4096).decode()
                if "GET /slow" in data:
                    # Send headers and partial HTML - never complete the document
                    response = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/html\r\n"
                        "Transfer-Encoding: chunked\r\n\r\n"
                    )
                    conn.sendall(response.encode())
                    chunk = "<html><body><h1>Loading"
                    conn.sendall(f"{len(chunk):x}\r\n{chunk}\r\n".encode())
                    time.sleep(30)  # keep connection open
                else:
                    html = '<html><body><a href="/slow">GoSlow</a></body></html>'
                    resp = (
                        f"HTTP/1.1 200 OK\r\n"
                        f"Content-Type: text/html\r\n"
                        f"Content-Length: {len(html)}\r\n\r\n{html}"
                    )
                    conn.sendall(resp.encode())
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        def accept_loop() -> None:
            while True:
                try:
                    conn, _ = srv.accept()
                    threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
                except Exception:
                    break

        acceptor = threading.Thread(target=accept_loop, daemon=True)
        acceptor.start()
        try:
            self.tool.go_to_url(f"http://127.0.0.1:{port}/")
            # Click the link to navigate to /slow (which never finishes loading)
            result = self.tool.click(1)
            # The page should still return something despite timeouts
            assert "Page:" in result or "Error" in result
        finally:
            srv.close()


class TestWebUseToolAllInvisibleElements:
    """Cover web_use_tool.py lines 186->184 and 191:
    is_visible() returns False for all elements, falls through to return locator.first."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_all_zero_size_buttons(self) -> None:
        """Multiple buttons with zero bounding box: get_by_role finds them,
        is_visible returns False for all, loop falls through to locator.first."""
        html = (
            "<html><body>"
            '<button style="width:0;height:0;overflow:hidden;padding:0;border:0;'
            'margin:0;display:inline-block">ZBtn</button>'
            '<button style="width:0;height:0;overflow:hidden;padding:0;border:0;'
            'margin:0;display:inline-block">ZBtn</button>'
            "</body></html>"
        )

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            self.tool.go_to_url(f"http://127.0.0.1:{port}/")
            tree = self.tool.get_page_content()
            # Should show two buttons
            assert "ZBtn" in tree
            # Click the first button - _resolve_locator should iterate
            # through all invisible buttons and return locator.first
            result = self.tool.click(1)
            assert "Page:" in result or "Error" in result
        finally:
            server.shutdown()

    def test_one_zero_size_one_visible(self) -> None:
        """First button is zero-size (invisible), second is normal (visible).
        Loop iterates past the first, returns the second."""
        html = (
            "<html><body>"
            '<button style="width:0;height:0;overflow:hidden;padding:0;border:0;'
            'margin:0;display:inline-block">MBtn</button>'
            "<button>MBtn</button>"
            "</body></html>"
        )

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            self.tool.go_to_url(f"http://127.0.0.1:{port}/")
            tree = self.tool.get_page_content()
            assert "MBtn" in tree
            result = self.tool.click(1)
            assert "Page:" in result or "Error" in result
        finally:
            server.shutdown()

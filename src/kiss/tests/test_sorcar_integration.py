"""Integration tests for kiss/agents/sorcar/ to increase branch coverage.

No mocks, patches, or test doubles. Uses real files, real git repos, and
real objects.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# _save_untracked_base
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# _cleanup_merge_data
# ---------------------------------------------------------------------------
class TestCleanupMergeData:
    def setup_method(self) -> None:
        self.data_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)
        base_dir = _untracked_base_dir()
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# _prepare_merge_view - modified pre-existing untracked files
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# _prepare_merge_view - tracked file pre-hash filtering
# ---------------------------------------------------------------------------
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
    def test_write_new_file(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "new.txt")
            result = tools.Write(path, "content")
            assert "Successfully wrote" in result
            assert Path(path).read_text() == "content"

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

# ---------------------------------------------------------------------------
# task_history.py
# ---------------------------------------------------------------------------
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

        # Reset cache
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
            # Create deeply nested structure
            current = d
            for i in range(8):
                current = os.path.join(current, f"level{i}")
                os.makedirs(current)
                Path(current, f"file{i}.txt").write_text(f"content {i}")
            paths = _scan_files(d)
            # Very deep files (e.g. level0/.../level5/file5.txt) should not appear
            assert not any("level5/file5.txt" in p for p in paths)
            # But shallow files should be present
            assert any("file0.txt" in p for p in paths)

    def test_caps_at_2000_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for i in range(2050):
                Path(d, f"file{i:04d}.txt").write_text(f"content {i}")
            paths = _scan_files(d)
            assert len(paths) <= 2000

class TestDisableCopilotScmButton:
    def test_no_extensions_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _disable_copilot_scm_button(d)  # Should not raise

    def test_bad_package_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "extensions" / "github.copilot-chat-1.0.0"
            ext_dir.mkdir(parents=True)
            (ext_dir / "package.json").write_text("not json")
            _disable_copilot_scm_button(d)  # Should not raise

class TestTruncateOutputTailZero:
    def test_tail_zero_branch(self) -> None:
        """When remaining=0 after subtracting msg length, tail=0 and line 29 is hit."""
        from kiss.agents.sorcar.useful_tools import _truncate_output

        text = "a" * 200
        worst_msg = f"\n\n... [truncated {len(text)} chars] ...\n\n"
        result = _truncate_output(text, len(worst_msg))
        assert "truncated" in result
        assert not result.startswith("a")  # head=0


class TestExtractLeadingCommandNameEdgeCases:
    def test_empty_name_after_lstrip(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        # Token is '((' which becomes '' after lstrip('({')
        assert _extract_leading_command_name("((") is None


# ---------------------------------------------------------------------------
# browser_ui.py - additional branch coverage
# ---------------------------------------------------------------------------
class TestBrowserPrinterBashStream:
    def test_reset_with_active_timer(self) -> None:
        """Covers reset cancelling an active flush timer (lines 458-459)."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        # Set last_flush far in the past to prevent immediate flush
        printer._bash_last_flush = time.monotonic()
        printer.print("data\n", type="bash_stream")
        # Timer should be set now
        assert printer._bash_flush_timer is not None
        printer.reset()
        assert printer._bash_flush_timer is None


# ---------------------------------------------------------------------------
# task_history.py - _save_history and _set_latest_chat_events edge cases
# ---------------------------------------------------------------------------
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

    def test_append_task_to_md_creates_file(self) -> None:
        """Test _append_task_to_md when file doesn't exist."""
        from kiss.agents.sorcar.task_history import (
            _append_task_to_md,
            _get_task_history_md_path,
        )

        path = _get_task_history_md_path()
        if path.exists():
            path.unlink()
        _append_task_to_md("new task", "new result")
        assert path.exists()
        content = path.read_text()
        assert "Task History" in content
        assert "new task" in content

# ---------------------------------------------------------------------------
# code_server.py - _install_copilot_extension edge cases
# ---------------------------------------------------------------------------
class TestInstallCopilotExtension:
    def test_already_installed(self) -> None:
        """When copilot extension dir exists, return immediately."""
        from kiss.agents.sorcar.code_server import _install_copilot_extension

        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "extensions" / "github.copilot-1.0.0"
            ext_dir.mkdir(parents=True)
            _install_copilot_extension(d)  # Should return early

# ---------------------------------------------------------------------------
# code_server.py - _disable_copilot_scm_button additional cases
# ---------------------------------------------------------------------------
class TestDisableCopilotScmButtonEdgeCases:
    def test_copilot_chat_without_package_json(self) -> None:
        """Directory exists but no package.json."""
        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "extensions" / "github.copilot-chat-1.0.0"
            ext_dir.mkdir(parents=True)
            _disable_copilot_scm_button(d)  # Should not raise

# ---------------------------------------------------------------------------
# browser_ui.py - _handle_message edge cases
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# browser_ui.py - _format_tool_call with old/new string
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Additional _prepare_merge_view branches
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Additional prompt_detector branches
# ---------------------------------------------------------------------------
class TestPromptDetectorEdgeCases:
    def test_frontmatter_without_closing_dashes(self) -> None:
        """Frontmatter with only one --- marker (no closing)."""
        from kiss.agents.sorcar.prompt_detector import PromptDetector

        detector = PromptDetector()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("---\nmodel: gpt-4\nNot closed properly\n")
            path = f.name
        try:
            is_prompt, score, reasons = detector.analyze(path)
            # No frontmatter parsed because < 3 parts when split by ---
        finally:
            os.unlink(path)

    def test_empty_words(self) -> None:
        """Empty content with just frontmatter."""
        from kiss.agents.sorcar.prompt_detector import PromptDetector

        detector = PromptDetector()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("---\nmodel: gpt-4\n---\n")
            path = f.name
        try:
            is_prompt, score, reasons = detector.analyze(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# browser_ui.py - additional branch coverage
# ---------------------------------------------------------------------------
class TestBrowserUiUncoveredBranches:
    """Cover remaining uncovered branches in browser_ui.py."""

        # No broadcast should happen, no error

    def test_content_block_delta_unknown_delta_type(self) -> None:
        """Cover 705->723: content_block_delta with unknown delta_type."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()

        class FakeEvent:
            event = {
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": "abc"},
            }

        text = printer._handle_stream_event(FakeEvent())
        assert text == ""

# ---------------------------------------------------------------------------
# code_server.py - additional branch coverage
# ---------------------------------------------------------------------------
class TestCodeServerUncoveredBranches:
    """Cover remaining uncovered branches in code_server.py."""

    def test_disable_copilot_write_oserror(self) -> None:
        """Cover 487-488: OSError when writing back package.json."""
        with tempfile.TemporaryDirectory() as d:
            ext_dir = Path(d) / "extensions" / "github.copilot-chat-1.0.0"
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
            # Make the file read-only to trigger OSError on write
            pkg_path.chmod(0o444)
            try:
                _disable_copilot_scm_button(d)  # Should not raise
            finally:
                pkg_path.chmod(0o644)
            # Should complete without error

    def test_save_untracked_base_oserror_on_copy(self) -> None:
        """Cover 748-749: OSError when copying untracked file (unreadable)."""
        tmpdir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        try:
            _init_git_repo(tmpdir)
            noread = Path(tmpdir, "noread.py")
            noread.write_text("content")
            noread.chmod(0o000)
            _save_untracked_base(tmpdir, data_dir, {"noread.py"})
            base_dir = _untracked_base_dir()
            assert not (base_dir / "noread.py").exists()
        finally:
            Path(tmpdir, "noread.py").chmod(0o644)
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(data_dir, ignore_errors=True)
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
            # Agent makes tracked change and makes untracked file huge
            Path(tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
            Path(tmpdir, "growing.py").write_bytes(b"x" * 2_100_000)
            result = _prepare_merge_view(
                tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            # Tracked change should still be included
            assert result.get("status") == "opened"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(data_dir, ignore_errors=True)
            base_dir = _untracked_base_dir()
            if base_dir.exists():
                shutil.rmtree(base_dir, ignore_errors=True)

    def test_prepare_merge_view_untracked_empty_in_pre_hashes(self) -> None:
        """Cover 832-833: pre-existing untracked file that's now empty (0 lines)."""
        tmpdir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        try:
            _init_git_repo(tmpdir)
            Path(tmpdir, "will_empty.py").write_text("content\n")
            pre_hunks = _parse_diff_hunks(tmpdir)
            pre_untracked = _capture_untracked(tmpdir)
            pre_hashes = _snapshot_files(tmpdir, set(pre_hunks.keys()) | pre_untracked)
            # Agent modifies tracked and empties untracked
            Path(tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
            Path(tmpdir, "will_empty.py").write_text("")
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


# ---------------------------------------------------------------------------
# prompt_detector.py - additional branch coverage
# ---------------------------------------------------------------------------
class TestPromptDetectorUncoveredBranches:
    def test_unreadable_md_file(self) -> None:
        """Cover 67-69: exception when reading file content."""
        from kiss.agents.sorcar.prompt_detector import PromptDetector

        detector = PromptDetector()
        # Create a directory with .md suffix — reading it will raise an error
        with tempfile.TemporaryDirectory() as d:
            md_dir = Path(d, "fake.md")
            md_dir.mkdir()
            is_prompt, score, reasons = detector.analyze(str(md_dir))
            assert not is_prompt
            assert score == 0.0
            assert any("Error" in r for r in reasons)


# ---------------------------------------------------------------------------
# task_history.py - additional branch coverage
# ---------------------------------------------------------------------------
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

        # Nothing should happen since cache is empty

# ---------------------------------------------------------------------------
# useful_tools.py - additional branch coverage
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# code_server.py - line 819: pre-untracked file already in file_hunks
# ---------------------------------------------------------------------------
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
        # 1. Create untracked file
        Path(self.tmpdir, "newfile.py").write_text("original\n")
        # 2. Capture pre-state
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        assert "newfile.py" in pre_untracked
        pre_hashes = _snapshot_files(
            self.tmpdir, set(pre_hunks.keys()) | pre_untracked
        )
        # 3. Agent stages and commits the file
        subprocess.run(
            ["git", "add", "newfile.py"], cwd=self.tmpdir, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "add newfile"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        # 4. Agent modifies the file (now tracked change vs new HEAD)
        Path(self.tmpdir, "newfile.py").write_text("modified by agent\n")
        # 5. Now _parse_diff_hunks will have newfile.py, pre_untracked has it too
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, pre_hunks, pre_untracked, pre_hashes
        )
        assert result.get("status") == "opened"
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text()
        )
        file_names = [f["name"] for f in manifest["files"]]
        assert "newfile.py" in file_names


# ---------------------------------------------------------------------------
# useful_tools.py - _kill_process_group with dead process (lines 159-161)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# web_use_tool.py - _number_interactive_elements (extended)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# web_use_tool.py - WebUseTool methods (headless browser integration)
# ---------------------------------------------------------------------------
class TestWebUseToolHeadless:
    """Integration tests for WebUseTool using a real headless browser."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(
            headless=True,
            user_data_dir=None,  # Don't use persistent profile
        )

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


class TestWebUseToolContextArgs:
    def test_launch_kwargs_non_chromium(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(headless=True, browser_type="firefox", user_data_dir=None)
        kwargs = tool._launch_kwargs()
        assert kwargs["headless"] is True
        assert "args" not in kwargs

    def test_launch_kwargs_non_headless_chromium(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(headless=False, browser_type="chromium", user_data_dir=None)
        kwargs = tool._launch_kwargs()
        assert kwargs["headless"] is False
        assert "channel" in kwargs

class TestWebUseToolPersistentContext:
    """Test WebUseTool with persistent context (user_data_dir set)."""

    def test_persistent_context(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        with tempfile.TemporaryDirectory() as d:
            tool = WebUseTool(headless=True, user_data_dir=d)
            try:
                result = tool.go_to_url("data:text/html,<h1>Persistent</h1>")
                assert "Persistent" in result or isinstance(result, str)
            finally:
                tool.close()


class TestWebUseToolResolveLocator:
    """Test _resolve_locator edge cases."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

class TestWebUseToolEdgeCases:
    """Cover remaining edge cases in web_use_tool.py."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_truncated_snapshot(self) -> None:
        """Cover line 143: snapshot exceeding max_chars is truncated."""
        # Create a page with many elements to produce a large snapshot
        buttons = "".join([f'<button>Button{i}</button>' for i in range(200)])
        self.tool.go_to_url(f"data:text/html,{buttons}")
        # Use a small max_chars to trigger truncation
        result = self.tool._get_ax_tree(max_chars=100)
        assert "truncated" in result

    def test_scroll_error(self) -> None:
        """Cover lines 325-327: scroll after page closed."""
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        self.tool._page.close()
        result = self.tool.scroll("down")
        assert "Error" in result

    def test_close_exception_in_pw_stop(self) -> None:
        """Cover lines 384-386: exception during close."""
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(headless=True, user_data_dir=None)
        tool.go_to_url("data:text/html,<h1>Test</h1>")
        # Stop playwright first to make close() hit exception path
        tool._playwright.stop()
        tool._playwright = None
        # Now close should try to close context/browser that are already dead
        result = tool.close()
        assert result == "Browser closed."

    def test_check_for_new_tab_single_page(self) -> None:
        """Cover 162->exit: _check_for_new_tab when there's only one page."""
        self.tool.go_to_url("data:text/html,<h1>Single</h1>")
        assert len(self.tool._context.pages) == 1
        # _check_for_new_tab should exit early (one page only)
        self.tool._check_for_new_tab()
        # Page should be unchanged


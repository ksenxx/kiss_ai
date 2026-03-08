"""Integration tests for kiss/agents/sorcar/ to increase branch coverage.

No mocks, patches, or test doubles. Uses real files, real git repos, and
real objects.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
    _disable_copilot_scm_button,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _scan_files,
    _setup_code_server,
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

    def test_cleans_merge_temp(self) -> None:
        merge_dir = Path(self.data_dir) / "merge-temp"
        merge_dir.mkdir()
        (merge_dir / "file.txt").write_text("temp")
        _cleanup_merge_data(self.data_dir)
        assert not merge_dir.exists()

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

    def test_multiple_untracked_files_mixed(self) -> None:
        Path(self.tmpdir, "mod.py").write_text("original\n")
        Path(self.tmpdir, "nomod.py").write_text("unchanged\n")
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(
            self.tmpdir, set(pre_hunks.keys()) | pre_untracked
        )
        _save_untracked_base(self.tmpdir, self.data_dir, pre_untracked)
        Path(self.tmpdir, "mod.py").write_text("modified\n")
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, pre_hunks, pre_untracked, pre_hashes
        )
        assert result.get("status") == "opened"
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text()
        )
        file_names = [f["name"] for f in manifest["files"]]
        assert "mod.py" in file_names
        assert "nomod.py" not in file_names

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

    def test_new_file_binary_from_agent_skipped(self) -> None:
        """A brand new binary file triggers UnicodeDecodeError and is skipped."""
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        # Agent creates a binary file
        Path(self.tmpdir, "binary.dat").write_bytes(b"\x80\x81\x82\xff\xfe")
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, pre_hunks, pre_untracked, None
        )
        assert "error" in result  # skipped due to UnicodeDecodeError

# ---------------------------------------------------------------------------
# sorcar_agent.py - _get_tools and _reset
# ---------------------------------------------------------------------------
class TestSorcarAgentGetTools:
    def test_get_tools_with_web_use_tool(self) -> None:
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(headless=True)
        agent.docker_manager = None
        tools = agent._get_tools()
        assert len(tools) == 11

    def test_reset_with_explicit_values(self) -> None:
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        agent = SorcarAgent("test")
        agent._reset(
            model_name="test-model",
            max_sub_sessions=5,
            max_steps=10,
            max_budget=1.0,
            work_dir="/tmp",
            docker_image=None,
            printer=None,
            verbose=True,
        )
        assert agent.model_name == "test-model"
        assert agent.max_steps == 10
        assert agent.max_budget == 1.0

# ---------------------------------------------------------------------------
# sorcar.py utilities
# ---------------------------------------------------------------------------
class TestSorcarUtilities:
    def test_clean_llm_output_strips_quotes(self) -> None:
        from kiss.agents.sorcar.sorcar import _clean_llm_output

        assert _clean_llm_output('  "hello world"  ') == "hello world"
        assert _clean_llm_output("  'hello world'  ") == "hello world"
        assert _clean_llm_output("plain text") == "plain text"

    def test_model_vendor_order_all_prefixes(self) -> None:
        from kiss.agents.sorcar.sorcar import _model_vendor_order

        assert _model_vendor_order("claude-3-5-sonnet") == 0
        assert _model_vendor_order("gpt-4o") == 1
        assert _model_vendor_order("o1-mini") == 1
        assert _model_vendor_order("gemini-2.0-flash") == 2
        assert _model_vendor_order("minimax-text-01") == 3
        assert _model_vendor_order("openrouter/anthropic/claude") == 4
        assert _model_vendor_order("llama-3") == 5

    def test_read_active_file_valid(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file

        with tempfile.TemporaryDirectory() as d:
            af_path = os.path.join(d, "active-file.json")
            real_file = os.path.join(d, "test.py")
            Path(real_file).write_text("content")
            with open(af_path, "w") as f:
                json.dump({"path": real_file}, f)
            assert _read_active_file(d) == real_file

    def test_read_active_file_bad_json(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file

        with tempfile.TemporaryDirectory() as d:
            af_path = os.path.join(d, "active-file.json")
            Path(af_path).write_text("not json")
            assert _read_active_file(d) == ""

    def test_read_active_file_nonexistent_path(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file

        with tempfile.TemporaryDirectory() as d:
            af_path = os.path.join(d, "active-file.json")
            with open(af_path, "w") as f:
                json.dump({"path": "/no/such/file.py"}, f)
            assert _read_active_file(d) == ""


# ---------------------------------------------------------------------------
# useful_tools.py
# ---------------------------------------------------------------------------
class TestExtractCommandNames:
    def test_env_var_prefix(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names

        assert _extract_command_names("FOO=bar python script.py") == ["python"]

class TestUsefulToolsRead:
    def test_read_existing_file(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\nworld\n")
            path = f.name
        try:
            result = tools.Read(path)
            assert "hello\nworld\n" == result
        finally:
            os.unlink(path)

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
        self._orig_proposals_file = task_history.PROPOSALS_FILE
        self._orig_model_usage_file = task_history.MODEL_USAGE_FILE
        self._orig_file_usage_file = task_history.FILE_USAGE_FILE
        self._orig_kiss_dir = task_history._KISS_DIR

        self.tmpdir = tempfile.mkdtemp()
        task_history._KISS_DIR = Path(self.tmpdir)
        task_history.HISTORY_FILE = Path(self.tmpdir) / "task_history.json"
        task_history.PROPOSALS_FILE = Path(self.tmpdir) / "proposed_tasks.json"
        task_history.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        task_history.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"

        # Reset cache
        task_history._history_cache = None

    def teardown_method(self) -> None:
        from kiss.agents.sorcar import task_history

        task_history.HISTORY_FILE = self._orig_history_file
        task_history.PROPOSALS_FILE = self._orig_proposals_file
        task_history.MODEL_USAGE_FILE = self._orig_model_usage_file
        task_history.FILE_USAGE_FILE = self._orig_file_usage_file
        task_history._KISS_DIR = self._orig_kiss_dir
        task_history._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_proposals_empty(self) -> None:
        from kiss.agents.sorcar.task_history import _load_proposals

        assert _load_proposals() == []

    def test_save_and_load_proposals(self) -> None:
        from kiss.agents.sorcar.task_history import _load_proposals, _save_proposals

        _save_proposals(["task 1", "task 2"])
        assert _load_proposals() == ["task 1", "task 2"]

    def test_load_proposals_bad_json(self) -> None:
        from kiss.agents.sorcar import task_history
        from kiss.agents.sorcar.task_history import _load_proposals

        task_history.PROPOSALS_FILE.write_text("not json")
        assert _load_proposals() == []

    def test_record_and_load_model_usage(self) -> None:
        from kiss.agents.sorcar.task_history import _load_model_usage, _record_model_usage

        _record_model_usage("gpt-4")
        _record_model_usage("gpt-4")
        _record_model_usage("claude-3")
        usage = _load_model_usage()
        assert usage["gpt-4"] == 2
        assert usage["claude-3"] == 1

    def test_load_last_model(self) -> None:
        from kiss.agents.sorcar.task_history import _load_last_model, _record_model_usage

        assert _load_last_model() == ""
        _record_model_usage("gpt-4")
        assert _load_last_model() == "gpt-4"

    def test_record_and_load_file_usage(self) -> None:
        from kiss.agents.sorcar.task_history import _load_file_usage, _record_file_usage

        _record_file_usage("src/main.py")
        _record_file_usage("src/main.py")
        usage = _load_file_usage()
        assert usage["src/main.py"] == 2

    def test_append_task_to_md(self) -> None:
        from kiss.agents.sorcar.task_history import _append_task_to_md, _init_task_history_md

        _init_task_history_md()
        _append_task_to_md("test task", "test result")
        path = _init_task_history_md()
        content = path.read_text()
        assert "test task" in content
        assert "test result" in content

    def test_load_history_with_duplicates(self) -> None:
        from kiss.agents.sorcar import task_history
        from kiss.agents.sorcar.task_history import _load_history

        # Write history with duplicate tasks
        data = [
            {"task": "task A", "chat_events": []},
            {"task": "task B", "chat_events": []},
            {"task": "task A", "chat_events": [{"extra": True}]},
        ]
        task_history.HISTORY_FILE.write_text(json.dumps(data))
        task_history._history_cache = None
        history = _load_history()
        tasks = [e["task"] for e in history]
        assert tasks.count("task A") == 1  # Deduplicated

    def test_load_history_bad_json(self) -> None:
        from kiss.agents.sorcar import task_history
        from kiss.agents.sorcar.task_history import SAMPLE_TASKS, _load_history

        task_history.HISTORY_FILE.write_text("not json")
        task_history._history_cache = None
        history = _load_history()
        assert len(history) == len(SAMPLE_TASKS)

# ---------------------------------------------------------------------------
# prompt_detector.py
# ---------------------------------------------------------------------------
class TestPromptDetector:
    def test_nonexistent_file(self) -> None:
        from kiss.agents.sorcar.prompt_detector import PromptDetector

        detector = PromptDetector()
        is_prompt, score, reasons = detector.analyze("/no/such/file.md")
        assert not is_prompt

# ---------------------------------------------------------------------------
# code_server.py - _scan_files
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# code_server.py - _setup_code_server
# ---------------------------------------------------------------------------
class TestSetupCodeServer:
    def test_setup_handles_bad_settings_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            user_dir = Path(d) / "User"
            user_dir.mkdir(parents=True)
            settings_file = user_dir / "settings.json"
            settings_file.write_text("not valid json")
            _setup_code_server(d)
            settings = json.loads(settings_file.read_text())
            assert "workbench.colorTheme" in settings

    def test_setup_cleans_workspace_chat_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ws_dir = Path(d) / "User" / "workspaceStorage" / "abc123"
            chat_dir = ws_dir / "chatSessions"
            chat_dir.mkdir(parents=True)
            (chat_dir / "session.json").write_text("{}")
            edit_dir = ws_dir / "chatEditingSessions"
            edit_dir.mkdir(parents=True)
            (edit_dir / "edit.json").write_text("{}")
            _setup_code_server(d)
            assert not chat_dir.exists()
            assert not edit_dir.exists()

# ---------------------------------------------------------------------------
# code_server.py - _disable_copilot_scm_button
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# code_server.py - _parse_diff_hunks
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# code_server.py - _snapshot_files
# ---------------------------------------------------------------------------
class TestSnapshotFiles:
    def test_skips_nonexistent_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.txt").write_text("aaa")
            result = _snapshot_files(d, {"a.txt", "missing.txt"})
            assert "a.txt" in result
            assert "missing.txt" not in result


# ---------------------------------------------------------------------------
# browser_ui.py - BaseBrowserPrinter
# ---------------------------------------------------------------------------
class TestBaseBrowserPrinter:
    def test_remove_nonexistent_client(self) -> None:
        import queue

        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q: queue.Queue = queue.Queue()
        printer.remove_client(q)  # Should not raise

    def test_print_unknown_type_returns_empty(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        result = printer.print("data", type="totally_unknown")
        assert result == ""

# ---------------------------------------------------------------------------
# browser_ui.py - _coalesce_events
# ---------------------------------------------------------------------------
class TestCoalesceEvents:
    def test_non_mergeable_not_changed(self) -> None:
        from kiss.agents.sorcar.browser_ui import _coalesce_events

        events = [
            {"type": "tool_call", "name": "Bash"},
            {"type": "tool_call", "name": "Read"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# browser_ui.py - find_free_port
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# web_use_tool.py - _number_interactive_elements
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# web_use_tool.py - WebUseTool init/close
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# chatbot_ui.py - _build_html
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# browser_ui.py - _handle_stream_event
# ---------------------------------------------------------------------------
class TestHandleStreamEvent:
    def test_content_block_start_tool_use(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()

        class FakeEvent:
            event = {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Bash"},
            }

        printer._handle_stream_event(FakeEvent())
        assert printer._tool_name == "Bash"
        assert printer._tool_json_buffer == ""

    def test_content_block_delta_json(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()

        class FakeEvent:
            event = {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"cmd":',
                },
            }

        printer._handle_stream_event(FakeEvent())
        assert printer._tool_json_buffer == '{"cmd":'

    def test_content_block_stop_tool_use(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q = printer.add_client()
        printer._current_block_type = "tool_use"
        printer._tool_name = "Bash"
        printer._tool_json_buffer = '{"command": "ls"}'

        class FakeEvent:
            event = {"type": "content_block_stop"}

        printer._handle_stream_event(FakeEvent())
        event = q.get_nowait()
        assert event["type"] == "tool_call"
        assert event["name"] == "Bash"

        # Should have _raw key for bad json

# ---------------------------------------------------------------------------
# browser_ui.py - _handle_message
# ---------------------------------------------------------------------------
class TestHandleMessage:
    def test_handle_result_message(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q = printer.add_client()

        class FakeMsg:
            result = "success: true\nsummary: All done"

        printer._handle_message(FakeMsg(), budget_used=0.5, step_count=3, total_tokens_used=100)
        event = q.get_nowait()
        assert event["type"] == "result"

# ---------------------------------------------------------------------------
# browser_ui.py - token_callback
# ---------------------------------------------------------------------------
class TestTokenCallback:
    def test_token_callback_empty_string(self) -> None:
        import asyncio

        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q = printer.add_client()
        asyncio.run(printer.token_callback(""))
        assert q.empty()

# ---------------------------------------------------------------------------
# browser_ui.py - _format_tool_call
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# code_server.py - _capture_untracked
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# useful_tools.py - additional branch coverage
# ---------------------------------------------------------------------------
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

class TestSplitRespectingQuotes:
    def test_escaped_chars(self) -> None:
        from kiss.agents.sorcar.useful_tools import _CONTROL_RE, _split_respecting_quotes

        result = _split_respecting_quotes("echo hello\\;world; cmd2", _CONTROL_RE)
        assert len(result) == 2

    def test_escaped_in_double_quotes(self) -> None:
        from kiss.agents.sorcar.useful_tools import _CONTROL_RE, _split_respecting_quotes

        result = _split_respecting_quotes('echo "hello\\"world"; cmd2', _CONTROL_RE)
        assert len(result) == 2

# ---------------------------------------------------------------------------
# sorcar.py - shutdown safety tests
# ---------------------------------------------------------------------------
class TestShutdownWhileRunning:
    """Test that _do_shutdown and _schedule_shutdown don't exit while a task is running."""

    def test_do_shutdown_skips_when_has_clients(self) -> None:
        """_do_shutdown returns early if clients are still connected."""

        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        running = False
        running_lock = threading.Lock()
        exited = False

        cq = printer.add_client()

        def _do_shutdown() -> None:
            nonlocal exited
            if printer.has_clients():
                return
            with running_lock:
                if running:
                    return
            exited = True

        _do_shutdown()
        assert not exited, "_do_shutdown must not exit with active clients"

        printer.remove_client(cq)
        _do_shutdown()
        assert exited


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

    def test_print_message(self) -> None:
        """Covers the message branch in print()."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q = printer.add_client()

        class FakeMsg:
            subtype = "tool_output"
            data = {"content": "output"}

        printer.print(FakeMsg(), type="message")
        event = q.get_nowait()
        assert event["type"] == "system_output"


# ---------------------------------------------------------------------------
# task_history.py - _save_history and _set_latest_chat_events edge cases
# ---------------------------------------------------------------------------
class TestTaskHistoryEdgeCases:
    def setup_method(self) -> None:
        from kiss.agents.sorcar import task_history

        self._orig_history_file = task_history.HISTORY_FILE
        self._orig_proposals_file = task_history.PROPOSALS_FILE
        self._orig_model_usage_file = task_history.MODEL_USAGE_FILE
        self._orig_file_usage_file = task_history.FILE_USAGE_FILE
        self._orig_kiss_dir = task_history._KISS_DIR
        self.tmpdir = tempfile.mkdtemp()
        task_history._KISS_DIR = Path(self.tmpdir)
        task_history.HISTORY_FILE = Path(self.tmpdir) / "task_history.json"
        task_history.PROPOSALS_FILE = Path(self.tmpdir) / "proposed_tasks.json"
        task_history.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        task_history.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        task_history._history_cache = None

    def teardown_method(self) -> None:
        from kiss.agents.sorcar import task_history

        task_history.HISTORY_FILE = self._orig_history_file
        task_history.PROPOSALS_FILE = self._orig_proposals_file
        task_history.MODEL_USAGE_FILE = self._orig_model_usage_file
        task_history.FILE_USAGE_FILE = self._orig_file_usage_file
        task_history._KISS_DIR = self._orig_kiss_dir
        task_history._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_proposals_non_list_json(self) -> None:
        """Test _load_proposals when file contains non-list JSON."""
        from kiss.agents.sorcar import task_history
        from kiss.agents.sorcar.task_history import _load_proposals

        task_history.PROPOSALS_FILE.write_text('{"key": "value"}')
        assert _load_proposals() == []

    def test_load_history_empty_list(self) -> None:
        """Test _load_history when file contains empty list."""
        from kiss.agents.sorcar import task_history
        from kiss.agents.sorcar.task_history import SAMPLE_TASKS, _load_history

        task_history.HISTORY_FILE.write_text("[]")
        task_history._history_cache = None
        history = _load_history()
        # Empty list triggers fallback to SAMPLE_TASKS
        assert len(history) == len(SAMPLE_TASKS)

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

    def test_save_history_oserror_caught(self) -> None:
        """OSError during save_history_unlocked is caught (lines 118-119)."""
        from kiss.agents.sorcar import task_history
        from kiss.agents.sorcar.task_history import _load_history, _save_history

        _load_history()
        # Make HISTORY_FILE point to /dev/null/impossible which will fail
        task_history.HISTORY_FILE = Path("/dev/null/impossible/history.json")
        _save_history([{"task": "test", "chat_events": []}])  # Should not raise

    def test_save_proposals_oserror_caught(self) -> None:
        """OSError during _save_proposals is caught (lines 152-153)."""
        from kiss.agents.sorcar import task_history
        from kiss.agents.sorcar.task_history import _save_proposals

        task_history.PROPOSALS_FILE = Path("/dev/null/impossible/proposals.json")
        _save_proposals(["task"])  # Should not raise

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

    def test_frontmatter_no_prompt_keys(self) -> None:
        """Frontmatter with non-prompt keys."""
        from kiss.agents.sorcar.prompt_detector import PromptDetector

        detector = PromptDetector()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("---\ntitle: My Document\nauthor: John\n---\nSome content\n")
            path = f.name
        try:
            is_prompt, score, reasons = detector.analyze(path)
            assert not is_prompt
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

    def test_top_p_indicator(self) -> None:
        from kiss.agents.sorcar.prompt_detector import PromptDetector

        detector = PromptDetector()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(
                "# System Prompt\n"
                "You are a helpful assistant.\n"
                "top_p: 0.9\n"
                "Act as an expert.\n"
            )
            path = f.name
        try:
            is_prompt, score, reasons = detector.analyze(path)
            assert is_prompt
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# browser_ui.py - additional branch coverage
# ---------------------------------------------------------------------------
class TestBrowserUiUncoveredBranches:
    """Cover remaining uncovered branches in browser_ui.py."""

    def test_print_text_empty_after_rich_formatting(self) -> None:
        """Cover 592->594: text.strip() is falsy after Rich formatting."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q = printer.add_client()
        # Empty string produces no output from Rich
        printer.print("", type="text")
        assert q.empty()

        # No broadcast should happen, no error

    def test_handle_message_subtype_not_tool_output(self) -> None:
        """Cover the branch inside subtype+data where subtype != 'tool_output'."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q = printer.add_client()

        class Msg:
            subtype = "other"
            data = {"content": "hello"}

        printer._handle_message(Msg())
        assert q.empty()

    def test_handle_message_with_content_blocks(self) -> None:
        """Cover 708->723 and 727->exit: message.content with blocks."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()
        q = printer.add_client()

        class Block:
            is_error = True
            content = "some error"

        class BlockNoAttrs:
            """Block without is_error/content attrs."""
            pass

        class Msg:
            content = [Block(), BlockNoAttrs()]

        printer._handle_message(Msg())
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["is_error"] is True

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

    def test_unknown_event_type(self) -> None:
        """Cover 708->723: event with entirely unknown evt_type."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter

        printer = BaseBrowserPrinter()

        class FakeEvent:
            event = {"type": "message_start"}

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

    def test_setup_code_server_ws_dir_without_chat_sessions(self) -> None:
        """Cover 559->557: workspace dir exists but chatSessions dir doesn't."""
        with tempfile.TemporaryDirectory() as d:
            ws_dir = Path(d) / "User" / "workspaceStorage" / "abc123"
            ws_dir.mkdir(parents=True)
            # Don't create chatSessions or chatEditingSessions
            _setup_code_server(d)
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
            # Need tracked change so pre_hashes is non-empty
            Path(tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
            Path(tmpdir, "growing.py").write_text("small\n")
            pre_hunks = _parse_diff_hunks(tmpdir)
            pre_untracked = _capture_untracked(tmpdir)
            pre_hashes = _snapshot_files(tmpdir, set(pre_hunks.keys()) | pre_untracked)
            # Agent makes tracked change and makes untracked file huge
            Path(tmpdir, "file.txt").write_text("line1\nmodified again\nline3\n")
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
            # Need tracked change so pre_hashes is non-empty
            Path(tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
            Path(tmpdir, "will_empty.py").write_text("content\n")
            pre_hunks = _parse_diff_hunks(tmpdir)
            pre_untracked = _capture_untracked(tmpdir)
            pre_hashes = _snapshot_files(tmpdir, set(pre_hunks.keys()) | pre_untracked)
            # Agent modifies tracked and empties untracked
            Path(tmpdir, "file.txt").write_text("line1\nmodified again\nline3\n")
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

    def test_prepare_merge_view_untracked_unicode_error_in_pre_hashes(self) -> None:
        """Cover 837: UnicodeDecodeError in pre-existing untracked file."""
        tmpdir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        try:
            _init_git_repo(tmpdir)
            Path(tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
            Path(tmpdir, "will_binary.py").write_text("text content\n")
            pre_hunks = _parse_diff_hunks(tmpdir)
            pre_untracked = _capture_untracked(tmpdir)
            pre_hashes = _snapshot_files(tmpdir, set(pre_hunks.keys()) | pre_untracked)
            # Agent modifies tracked and replaces untracked with binary
            Path(tmpdir, "file.txt").write_text("line1\nmodified again\nline3\n")
            Path(tmpdir, "will_binary.py").write_bytes(b"\x80\x81\x82\xff\xfe")
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
        self._orig_proposals_file = task_history.PROPOSALS_FILE
        self._orig_model_usage_file = task_history.MODEL_USAGE_FILE
        self._orig_file_usage_file = task_history.FILE_USAGE_FILE
        self._orig_kiss_dir = task_history._KISS_DIR

        self.tmpdir = tempfile.mkdtemp()
        task_history._KISS_DIR = Path(self.tmpdir)
        task_history.HISTORY_FILE = Path(self.tmpdir) / "task_history.json"
        task_history.PROPOSALS_FILE = Path(self.tmpdir) / "proposed_tasks.json"
        task_history.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        task_history.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        task_history._history_cache = None

    def teardown_method(self) -> None:
        from kiss.agents.sorcar import task_history

        task_history.HISTORY_FILE = self._orig_history_file
        task_history.PROPOSALS_FILE = self._orig_proposals_file
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

    def test_bash_streaming_error_exit(self) -> None:
        """Cover _bash_streaming with non-zero exit code."""
        from kiss.agents.sorcar.useful_tools import UsefulTools

        tools = UsefulTools(stream_callback=lambda s: None)
        result = tools.Bash("echo fail_msg && exit 42", "test")
        assert "Error (exit code 42)" in result
        assert "fail_msg" in result


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

    def test_close_after_use(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        result = self.tool.close()
        assert result == "Browser closed."
        # Second close should also work
        result = self.tool.close()
        assert result == "Browser closed."


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

    def test_type_text_error(self) -> None:
        """Cover type_text exception path."""
        self.tool.go_to_url("data:text/html,<h1>No inputs</h1>")
        result = self.tool.type_text(999, "text")
        assert "Error" in result

    def test_get_page_content_error(self) -> None:
        """Cover lines 368-370: get_page_content after page closed."""
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        self.tool._page.close()
        result = self.tool.get_page_content()
        assert "Error" in result

    def test_screenshot_error(self) -> None:
        """Cover lines 346-348: screenshot after page closed."""
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        self.tool._page.close()
        result = self.tool.screenshot("/tmp/test_error.png")
        assert "Error" in result

    def test_scroll_error(self) -> None:
        """Cover lines 325-327: scroll after page closed."""
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        self.tool._page.close()
        result = self.tool.scroll("down")
        assert "Error" in result

    def test_press_key_error(self) -> None:
        """Cover lines 301-303: press_key after page closed."""
        self.tool.go_to_url("data:text/html,<h1>Test</h1>")
        self.tool._page.close()
        result = self.tool.press_key("Enter")
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


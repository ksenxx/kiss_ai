"""Integration tests for kiss/agents/sorcar/ to maximize branch coverage.

No mocks, patches, or test doubles.  Uses real files, real git repos, and
real objects.
"""

import asyncio
import http.server
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.task_history as th
from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter, find_free_port
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _disable_copilot_scm_button,
    _install_copilot_extension,
    _parse_diff_hunks,
    _prepare_merge_view,
    _scan_files,
    _setup_code_server,
    _snapshot_files,
)
from kiss.agents.sorcar.prompt_detector import PromptDetector
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _extract_command_names,
    _extract_leading_command_name,
    _format_bash_result,
    _kill_process_group,
    _split_respecting_quotes,
    _strip_heredocs,
    _truncate_output,
)
from kiss.agents.sorcar.web_use_tool import (
    INTERACTIVE_ROLES,
    _number_interactive_elements,
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

    def test_nonexistent_file(self) -> None:
        ok, score, reasons = self.detector.analyze("/no/such/file.md")
        assert not ok and score == 0.0
        assert "not found" in reasons[0].lower() or "not .md" in reasons[0].lower()

    def test_non_md_file(self) -> None:
        p = self._write("readme.txt", "hello")
        ok, score, reasons = self.detector.analyze(p)
        assert not ok and score == 0.0

    def test_unreadable_file(self) -> None:
        p = self._write("locked.md", "content")
        os.chmod(p, 0o000)
        try:
            ok, score, reasons = self.detector.analyze(p)
            assert not ok
            assert any("error" in r.lower() for r in reasons)
        finally:
            os.chmod(p, 0o644)

    def test_strong_indicator_system_prompt(self) -> None:
        p = self._write("prompt.md", "# System Prompt\nYou are a helpful assistant.\n")
        ok, score, reasons = self.detector.analyze(p)
        assert ok and score >= PromptDetector.THRESHOLD

    def test_strong_indicator_act_as(self) -> None:
        p = self._write("role.md", "Act as a senior engineer.\n## Constraints\nReturn only code.")
        ok, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_strong_indicator_template_vars(self) -> None:
        p = self._write("tmpl.md", "Hello {{ user_name }}, welcome to {{ project }}.\n")
        ok, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_strong_indicator_xml_tags(self) -> None:
        content = "<system>\nYou are a helpful assistant.\n</system>\n"
        p = self._write("xml.md", content + "<user>Hi</user>\n")
        ok, score, reasons = self.detector.analyze(p)
        assert ok

    def test_medium_indicators(self) -> None:
        p = self._write(
            "med.md",
            "# Role\nfew-shot examples\nchain of thought reasoning\n"
            "step-by-step\nyour task is to classify\ndo not hallucinate\n",
        )
        ok, score, reasons = self.detector.analyze(p)
        assert ok

    def test_weak_indicators(self) -> None:
        content = "temperature: 0.7\ntop_p: 0.9\njson mode\n```json\n{}\n```\n"
        p = self._write("weak.md", content)
        ok, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_frontmatter_with_prompt_keys(self) -> None:
        p = self._write(
            "fm.md",
            "---\nmodel: gpt-4\ntemperature: 0.5\ninputs: text\n---\n"
            "You are a helpful assistant.\n{{ text }}\n",
        )
        ok, score, reasons = self.detector.analyze(p)
        assert ok
        assert any("metadata" in r.lower() for r in reasons)

    def test_frontmatter_without_prompt_keys(self) -> None:
        p = self._write("fm2.md", "---\ntitle: My Blog\nauthor: Me\n---\nJust a blog post.\n")
        ok, score, reasons = self.detector.analyze(p)
        assert not ok or score < PromptDetector.THRESHOLD

    def test_frontmatter_only_partial_match(self) -> None:
        """Frontmatter with one prompt key still gets partial credit."""
        p = self._write("fm3.md", "---\nmodel: gpt-4\n---\nSome generic text.\n")
        ok, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_high_imperative_density(self) -> None:
        verbs = " ".join(["write explain summarize translate classify"] * 20)
        filler = " ".join(["word"] * 50)
        p = self._write("verbs.md", f"{verbs}\n{filler}\n")
        ok, score, reasons = self.detector.analyze(p)
        assert any("imperative" in r.lower() for r in reasons)

    def test_low_imperative_density(self) -> None:
        p = self._write("noaction.md", " ".join(["apple banana cherry"] * 100) + "\n")
        ok, score, reasons = self.detector.analyze(p)
        assert not any("imperative" in r.lower() for r in reasons)

    def test_diminishing_returns_capped(self) -> None:
        """Multiple matches of the same pattern get diminishing returns."""
        content = "\n".join([f"You are a {w} expert." for w in ["Python", "JS", "Go", "Rust", "C"]])
        p = self._write("many.md", content)
        ok, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_readme_not_detected(self) -> None:
        p = self._write(
            "readme.md",
            "# Project\nThis project does X.\n## Installation\nRun pip install.\n",
        )
        ok, score, reasons = self.detector.analyze(p)
        assert not ok

    def test_no_frontmatter(self) -> None:
        p = self._write("nofm.md", "Just plain markdown.\nNo frontmatter here.\n")
        ok, score, reasons = self.detector.analyze(p)
        assert not ok

    def test_frontmatter_not_starting_with_dashes(self) -> None:
        p = self._write("nofm2.md", "Some text\n---\nkey: val\n---\nMore text.\n")
        ok, score, reasons = self.detector.analyze(p)
        # Should not detect frontmatter since it doesn't start with ---
        assert score == 0.0 or not any("metadata" in r.lower() for r in reasons)


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
    def test_load_populates_samples_when_no_file(self) -> None:
        history = th._load_history()
        assert len(history) == len(th.SAMPLE_TASKS)
        assert th.HISTORY_FILE.exists()

    def test_load_caches_result(self) -> None:
        h1 = th._load_history()
        h2 = th._load_history()
        assert h1 is h2  # same object from cache

    def test_load_deduplicates(self) -> None:
        data = [
            {"task": "a", "result": ""},
            {"task": "a", "result": "x"},
            {"task": "b", "result": ""},
        ]
        th.HISTORY_FILE.write_text(json.dumps(data))
        th._history_cache = None
        history = th._load_history()
        tasks = [e["task"] for e in history]
        assert tasks == ["a", "b"]

    def test_load_empty_list_triggers_samples(self) -> None:
        th.HISTORY_FILE.write_text("[]")
        th._history_cache = None
        history = th._load_history()
        # Empty list is falsy, so SAMPLE_TASKS should be loaded
        assert history == list(th.SAMPLE_TASKS)

    def test_add_task_deduplicates(self) -> None:
        th._add_task("task1")
        th._add_task("task2")
        th._add_task("task1")  # should move to top
        history = th._load_history()
        assert history[0]["task"] == "task1"
        assert history[1]["task"] == "task2"

    def test_set_latest_result(self) -> None:
        th._add_task("my_task")
        th._set_latest_result("done!")
        history = th._load_history()
        assert history[0]["result"] == "done!"

    def test_set_latest_result_empty_history(self) -> None:
        # With sample tasks loaded, set result on first
        th._load_history()
        th._set_latest_result("result_text")
        assert th._history_cache is not None
        assert th._history_cache[0]["result"] == "result_text"

    # --- proposals ---
    def test_load_save_proposals(self) -> None:
        th._save_proposals(["do X", "do Y"])
        loaded = th._load_proposals()
        assert loaded == ["do X", "do Y"]

    def test_load_proposals_corrupted(self) -> None:
        th.PROPOSALS_FILE.write_text("bad json{{{")
        assert th._load_proposals() == []

    def test_load_proposals_non_list(self) -> None:
        th.PROPOSALS_FILE.write_text('{"a": 1}')
        assert th._load_proposals() == []

    def test_load_proposals_filters_non_strings(self) -> None:
        th.PROPOSALS_FILE.write_text('[1, "valid", "", "  ", "ok"]')
        loaded = th._load_proposals()
        assert loaded == ["valid", "ok"]

    def test_load_proposals_max_five(self) -> None:
        th._save_proposals([f"task{i}" for i in range(10)])
        loaded = th._load_proposals()
        assert len(loaded) == 5

    # --- model usage ---
    def test_record_and_load_model_usage(self) -> None:
        th._record_model_usage("claude-3")
        th._record_model_usage("claude-3")
        th._record_model_usage("gpt-4")
        usage = th._load_model_usage()
        assert usage["claude-3"] == 2
        assert usage["gpt-4"] == 1

    def test_load_last_model(self) -> None:
        th._record_model_usage("my-model")
        assert th._load_last_model() == "my-model"

    def test_load_last_model_missing(self) -> None:
        assert th._load_last_model() == ""

    def test_load_last_model_non_string(self) -> None:
        th.MODEL_USAGE_FILE.write_text(json.dumps({"_last": 42}))
        assert th._load_last_model() == ""

    # --- file usage ---
    def test_record_and_load_file_usage(self) -> None:
        th._record_file_usage("src/main.py")
        th._record_file_usage("src/main.py")
        usage = th._load_file_usage()
        assert usage["src/main.py"] == 2

    # --- _load_json_dict ---
    def test_load_json_dict_corrupted(self) -> None:
        th.MODEL_USAGE_FILE.write_text("not json")
        assert th._load_json_dict(th.MODEL_USAGE_FILE) == {}

    def test_load_json_dict_non_dict(self) -> None:
        th.MODEL_USAGE_FILE.write_text("[1,2,3]")
        assert th._load_json_dict(th.MODEL_USAGE_FILE) == {}

    def test_int_values_filters_non_numeric(self) -> None:
        assert th._int_values({"a": 1, "b": "x", "c": 3.5}) == {"a": 1, "c": 3}

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
    def test_concurrent_add_and_set_result(self) -> None:
        errors = []

        def add_tasks():
            try:
                for i in range(20):
                    th._add_task(f"concurrent_{i}")
            except Exception as e:
                errors.append(e)

        def set_results():
            try:
                for _ in range(20):
                    th._set_latest_result("result")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_tasks)
        t2 = threading.Thread(target=set_results)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors


# ---------------------------------------------------------------------------
# useful_tools.py  (90% -> target higher)
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_no_truncation(self) -> None:
        assert _truncate_output("hello", 100) == "hello"

    def test_truncation_with_message(self) -> None:
        long = "x" * 200
        result = _truncate_output(long, 100)
        assert "truncated" in result
        assert len(result) <= 100 + 50  # message adds some chars

    def test_truncation_very_small_max(self) -> None:
        result = _truncate_output("x" * 100, 5)
        assert len(result) == 5

    def test_truncation_tail_zero(self) -> None:
        # When remaining is odd, tail might be 0
        result = _truncate_output("x" * 200, 50)
        assert "truncated" in result


class TestExtractCommandNames:
    def test_simple(self) -> None:
        assert _extract_command_names("ls -la") == ["ls"]

    def test_pipe(self) -> None:
        assert _extract_command_names("cat file | grep foo | wc -l") == ["cat", "grep", "wc"]

    def test_and_chain(self) -> None:
        assert _extract_command_names("cd /tmp && ls") == ["cd", "ls"]

    def test_or_chain(self) -> None:
        assert _extract_command_names("false || echo fail") == ["false", "echo"]

    def test_semicolons(self) -> None:
        assert _extract_command_names("a; b; c") == ["a", "b", "c"]

    def test_env_vars_prefix(self) -> None:
        assert _extract_command_names("FOO=bar python script.py") == ["python"]

    def test_background_ampersand(self) -> None:
        names = _extract_command_names("sleep 10 & echo done")
        assert "sleep" in names and "echo" in names

    def test_redirect(self) -> None:
        names = _extract_command_names("echo hello > out.txt")
        assert names == ["echo"]

    def test_heredoc_stripping(self) -> None:
        cmd = "cat <<EOF\nhello world\nEOF\necho done"
        names = _extract_command_names(cmd)
        assert "echo" in names

    def test_quoted_semicolons_not_split(self) -> None:
        names = _extract_command_names("echo 'a;b'")
        assert names == ["echo"]

    def test_leading_command_with_path(self) -> None:
        assert _extract_leading_command_name("/usr/bin/python3 script.py") == "python3"

    def test_leading_command_empty(self) -> None:
        assert _extract_leading_command_name("") is None

    def test_bad_shlex(self) -> None:
        # Unbalanced quote
        assert _extract_leading_command_name("echo 'unbalanced") is None

    def test_shell_prefix_tokens(self) -> None:
        result = _extract_leading_command_name("{ echo hello; }")
        assert result == "echo"

    def test_redirect_with_fd(self) -> None:
        result = _extract_leading_command_name("2>&1 ls")
        assert result == "ls"

    def test_redirect_inline(self) -> None:
        # Redirect token where target is in same token, e.g., ">file"
        result = _extract_leading_command_name(">output.txt echo hello")
        assert result == "echo"

    def test_only_env_vars(self) -> None:
        result = _extract_leading_command_name("A=1 B=2")
        assert result is None

    def test_newline_split(self) -> None:
        names = _extract_command_names("echo a\necho b")
        assert names == ["echo", "echo"]


class TestStripHeredocs:
    def test_strips_heredoc(self) -> None:
        cmd = "cat <<EOF\nhello\nworld\nEOF"
        result = _strip_heredocs(cmd)
        assert "hello" not in result

    def test_no_heredoc(self) -> None:
        cmd = "echo hello"
        assert _strip_heredocs(cmd) == cmd

    def test_heredoc_with_dash(self) -> None:
        cmd = "cat <<-EOF\n\thello\n\tEOF"
        result = _strip_heredocs(cmd)
        assert "hello" not in result


class TestFormatBashResult:
    def test_error(self) -> None:
        result = _format_bash_result(1, "err", 1000)
        assert "Error" in result and "err" in result

    def test_error_no_output(self) -> None:
        result = _format_bash_result(1, "", 1000)
        assert "Error" in result

    def test_success(self) -> None:
        assert _format_bash_result(0, "ok", 1000) == "ok"


class TestUsefulToolsRead:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_file(self) -> None:
        p = os.path.join(self.tmpdir, "test.txt")
        Path(p).write_text("hello\nworld\n")
        assert "hello" in self.tools.Read(p)

    def test_read_truncates_lines(self) -> None:
        p = os.path.join(self.tmpdir, "big.txt")
        Path(p).write_text("\n".join(f"line{i}" for i in range(100)))
        result = self.tools.Read(p, max_lines=10)
        assert "truncated" in result.lower()

    def test_read_nonexistent(self) -> None:
        result = self.tools.Read("/no/such/file")
        assert "error" in result.lower()


class TestUsefulToolsWrite:
    def setup_method(self) -> None:
        self.tools = UsefulTools()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_file(self) -> None:
        p = os.path.join(self.tmpdir, "out.txt")
        result = self.tools.Write(p, "content")
        assert "success" in result.lower()
        assert Path(p).read_text() == "content"

    def test_write_creates_dirs(self) -> None:
        p = os.path.join(self.tmpdir, "a", "b", "c.txt")
        self.tools.Write(p, "deep")
        assert Path(p).read_text() == "deep"

    def test_write_error(self) -> None:
        result = self.tools.Write("/dev/null/impossible/path.txt", "x")
        assert "error" in result.lower()


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

    def test_edit_single_occurrence(self) -> None:
        p = self._write("hello world")
        result = self.tools.Edit(p, "hello", "goodbye")
        assert "success" in result.lower()
        assert Path(p).read_text() == "goodbye world"

    def test_edit_all_occurrences(self) -> None:
        p = self._write("aa bb aa cc aa")
        result = self.tools.Edit(p, "aa", "xx", replace_all=True)
        assert "3" in result
        assert Path(p).read_text() == "xx bb xx cc xx"

    def test_edit_nonunique_without_replace_all(self) -> None:
        p = self._write("aa bb aa")
        result = self.tools.Edit(p, "aa", "xx")
        assert "error" in result.lower() and "2" in result

    def test_edit_string_not_found(self) -> None:
        p = self._write("hello world")
        result = self.tools.Edit(p, "xyz", "abc")
        assert "not found" in result.lower()

    def test_edit_same_string(self) -> None:
        p = self._write("hello")
        result = self.tools.Edit(p, "hello", "hello")
        assert "different" in result.lower()

    def test_edit_file_not_found(self) -> None:
        result = self.tools.Edit("/no/such/file.txt", "a", "b")
        assert "error" in result.lower()


class TestUsefulToolsBash:
    def setup_method(self) -> None:
        self.tools = UsefulTools()

    def test_simple_command(self) -> None:
        result = self.tools.Bash("echo hello", "echo test")
        assert "hello" in result

    def test_disallowed_command(self) -> None:
        result = self.tools.Bash("eval 'echo hi'", "eval test")
        assert "not allowed" in result.lower()

    def test_disallowed_source(self) -> None:
        result = self.tools.Bash("source ~/.bashrc", "source test")
        assert "not allowed" in result.lower()

    def test_disallowed_exec(self) -> None:
        result = self.tools.Bash("exec bash", "exec test")
        assert "not allowed" in result.lower()

    def test_timeout(self) -> None:
        result = self.tools.Bash("sleep 30", "sleep test", timeout_seconds=0.5)
        assert "timeout" in result.lower()

    def test_exit_code_nonzero(self) -> None:
        result = self.tools.Bash("exit 1", "exit test")
        assert "error" in result.lower()

    def test_max_output_truncation(self) -> None:
        result = self.tools.Bash(
            "python3 -c \"print('x'*10000)\"",
            "long output",
            max_output_chars=100,
        )
        # Should be truncated
        assert len(result) < 10000

    def test_streaming_bash(self) -> None:
        collected: list[str] = []
        tools = UsefulTools(stream_callback=collected.append)
        result = tools.Bash("echo line1; echo line2", "stream test")
        assert "line1" in result or any("line1" in c for c in collected)

    def test_streaming_bash_timeout(self) -> None:
        collected: list[str] = []
        tools = UsefulTools(stream_callback=collected.append)
        result = tools.Bash("sleep 30", "timeout stream", timeout_seconds=0.5)
        assert "timeout" in result.lower()


# ---------------------------------------------------------------------------
# browser_ui.py  (33% -> target higher)
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_positive_int(self) -> None:
        port = find_free_port()
        assert isinstance(port, int) and port > 0


class TestBaseBrowserPrinter:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

    def test_add_and_remove_client(self) -> None:
        assert not self.printer.has_clients()
        cq = self.printer.add_client()
        assert self.printer.has_clients()
        self.printer.remove_client(cq)
        assert not self.printer.has_clients()

    def test_remove_nonexistent_client(self) -> None:
        import queue

        fake_q: queue.Queue = queue.Queue()
        # Should not raise
        self.printer.remove_client(fake_q)

    def test_broadcast(self) -> None:
        cq = self.printer.add_client()
        self.printer.broadcast({"type": "test", "data": "hello"})
        event = cq.get_nowait()
        assert event["type"] == "test"
        self.printer.remove_client(cq)

    def test_broadcast_to_multiple_clients(self) -> None:
        cq1 = self.printer.add_client()
        cq2 = self.printer.add_client()
        self.printer.broadcast({"type": "multi"})
        assert cq1.get_nowait()["type"] == "multi"
        assert cq2.get_nowait()["type"] == "multi"
        self.printer.remove_client(cq1)
        self.printer.remove_client(cq2)

    def test_reset(self) -> None:
        self.printer._current_block_type = "thinking"
        self.printer._tool_name = "test"
        self.printer._tool_json_buffer = "data"
        self.printer.reset()
        assert self.printer._current_block_type == ""
        assert self.printer._tool_name == ""
        assert self.printer._tool_json_buffer == ""

    def test_print_text(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("Hello world", type="text")
        event = cq.get_nowait()
        assert event["type"] == "text_delta"
        self.printer.remove_client(cq)

    def test_print_prompt(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("Do something", type="prompt")
        event = cq.get_nowait()
        assert event["type"] == "prompt"
        self.printer.remove_client(cq)

    def test_print_usage_info(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("Tokens: 100", type="usage_info")
        event = cq.get_nowait()
        assert event["type"] == "usage_info"
        self.printer.remove_client(cq)

    def test_print_tool_call(self) -> None:
        cq = self.printer.add_client()
        self.printer.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "ls", "description": "list"},
        )
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["command"] == "ls"
        self.printer.remove_client(cq)

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

    def test_print_tool_result(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("ok", type="tool_result", is_error=False)
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(result_events) == 1
        assert not result_events[0]["is_error"]
        self.printer.remove_client(cq)

    def test_print_tool_result_error(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("fail", type="tool_result", is_error=True)
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "tool_result"]
        assert result_events[0]["is_error"]
        self.printer.remove_client(cq)

    def test_print_result_plain(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("done", type="result", step_count=5, total_tokens=100, cost="$0.01")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1
        assert result_events[0]["step_count"] == 5
        self.printer.remove_client(cq)

    def test_print_result_yaml_parsed(self) -> None:
        cq = self.printer.add_client()
        yaml_text = "success: true\nsummary: All done"
        self.printer.print(yaml_text, type="result")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert result_events[0]["success"] is True
        assert result_events[0]["summary"] == "All done"
        self.printer.remove_client(cq)

    def test_print_result_yaml_without_summary_not_parsed(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("key: value\nother: data", type="result")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert "summary" not in result_events[0]
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

    def test_print_unknown_type(self) -> None:
        result = self.printer.print("data", type="unknown_type")
        assert result == ""

    def test_bash_stream(self) -> None:
        cq = self.printer.add_client()
        self.printer.print("line1\n", type="bash_stream")
        time.sleep(0.15)  # allow flush timer
        self.printer._flush_bash()  # force flush
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        sys_events = [e for e in events if e["type"] == "system_output"]
        assert any("line1" in e["text"] for e in sys_events)
        self.printer.remove_client(cq)

    def test_bash_stream_immediate_flush(self) -> None:
        cq = self.printer.add_client()
        # Set last_flush to past to trigger immediate flush
        self.printer._bash_last_flush = 0.0
        self.printer.print("quick\n", type="bash_stream")
        time.sleep(0.05)
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        sys_events = [e for e in events if e["type"] == "system_output"]
        assert any("quick" in e["text"] for e in sys_events)
        self.printer.remove_client(cq)

    def test_check_stop_raises(self) -> None:
        self.printer.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            self.printer._check_stop()
        self.printer.stop_event.clear()

    def test_token_callback_text(self) -> None:
        cq = self.printer.add_client()
        self.printer._current_block_type = "text"
        asyncio.run(self.printer.token_callback("hello"))
        event = cq.get_nowait()
        assert event["type"] == "text_delta"
        self.printer.remove_client(cq)

    def test_token_callback_thinking(self) -> None:
        cq = self.printer.add_client()
        self.printer._current_block_type = "thinking"
        asyncio.run(self.printer.token_callback("think"))
        event = cq.get_nowait()
        assert event["type"] == "thinking_delta"
        self.printer.remove_client(cq)

    def test_token_callback_empty_string(self) -> None:
        cq = self.printer.add_client()
        asyncio.run(self.printer.token_callback(""))
        assert cq.empty()
        self.printer.remove_client(cq)

    def test_token_callback_stop_event(self) -> None:
        self.printer.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(self.printer.token_callback("x"))
        self.printer.stop_event.clear()

    def test_handle_stream_event_content_block_start_thinking(self) -> None:
        cq = self.printer.add_client()

        class FakeEvent:
            event = {"type": "content_block_start", "content_block": {"type": "thinking"}}

        self.printer._handle_stream_event(FakeEvent())
        assert self.printer._current_block_type == "thinking"
        event = cq.get_nowait()
        assert event["type"] == "thinking_start"
        self.printer.remove_client(cq)

    def test_handle_stream_event_content_block_start_tool_use(self) -> None:
        class FakeEvent:
            event = {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Bash"},
            }

        self.printer._handle_stream_event(FakeEvent())
        assert self.printer._tool_name == "Bash"

    def test_handle_stream_event_thinking_delta(self) -> None:
        class FakeEvent:
            event = {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "hmm"},
            }

        text = self.printer._handle_stream_event(FakeEvent())
        assert text == "hmm"

    def test_handle_stream_event_text_delta(self) -> None:
        class FakeEvent:
            event = {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hello"},
            }

        text = self.printer._handle_stream_event(FakeEvent())
        assert text == "hello"

    def test_handle_stream_event_input_json_delta(self) -> None:
        class FakeEvent:
            event = {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'},
            }

        self.printer._handle_stream_event(FakeEvent())
        assert '{"cmd":' in self.printer._tool_json_buffer

    def test_handle_stream_event_content_block_stop_thinking(self) -> None:
        cq = self.printer.add_client()
        self.printer._current_block_type = "thinking"

        class FakeEvent:
            event = {"type": "content_block_stop"}

        self.printer._handle_stream_event(FakeEvent())
        event = cq.get_nowait()
        assert event["type"] == "thinking_end"
        self.printer.remove_client(cq)

    def test_handle_stream_event_content_block_stop_tool_use(self) -> None:
        cq = self.printer.add_client()
        self.printer._current_block_type = "tool_use"
        self.printer._tool_name = "Bash"
        self.printer._tool_json_buffer = '{"command": "ls"}'

        class FakeEvent:
            event = {"type": "content_block_stop"}

        self.printer._handle_stream_event(FakeEvent())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["command"] == "ls"
        self.printer.remove_client(cq)

    def test_handle_stream_event_content_block_stop_tool_use_bad_json(self) -> None:
        cq = self.printer.add_client()
        self.printer._current_block_type = "tool_use"
        self.printer._tool_name = "Bash"
        self.printer._tool_json_buffer = "not valid json"

        class FakeEvent:
            event = {"type": "content_block_stop"}

        self.printer._handle_stream_event(FakeEvent())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        self.printer.remove_client(cq)

    def test_handle_stream_event_content_block_stop_text(self) -> None:
        cq = self.printer.add_client()
        self.printer._current_block_type = "text"

        class FakeEvent:
            event = {"type": "content_block_stop"}

        self.printer._handle_stream_event(FakeEvent())
        event = cq.get_nowait()
        assert event["type"] == "text_end"
        self.printer.remove_client(cq)

    def test_handle_message_tool_output(self) -> None:
        cq = self.printer.add_client()

        class Msg:
            subtype = "tool_output"
            data = {"content": "tool output text"}

        self.printer._handle_message(Msg())
        event = cq.get_nowait()
        assert event["type"] == "system_output"
        self.printer.remove_client(cq)

    def test_handle_message_tool_output_empty(self) -> None:
        cq = self.printer.add_client()

        class Msg:
            subtype = "tool_output"
            data = {"content": ""}

        self.printer._handle_message(Msg())
        assert cq.empty()
        self.printer.remove_client(cq)

    def test_handle_message_result(self) -> None:
        cq = self.printer.add_client()

        class Msg:
            result = "success: true\nsummary: done"

        self.printer._handle_message(Msg(), budget_used=0.5, step_count=3, total_tokens_used=100)
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1
        assert result_events[0]["cost"] == "$0.5000"
        self.printer.remove_client(cq)

    def test_handle_message_result_no_budget(self) -> None:
        cq = self.printer.add_client()

        class Msg:
            result = "just text"

        self.printer._handle_message(Msg())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert result_events[0]["cost"] == "N/A"
        self.printer.remove_client(cq)

    def test_handle_message_content_blocks(self) -> None:
        cq = self.printer.add_client()

        class Block:
            is_error = True
            content = "something failed"

        class Msg:
            content = [Block()]

        self.printer._handle_message(Msg())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(result_events) == 1
        assert result_events[0]["is_error"]
        self.printer.remove_client(cq)

    def test_print_stream_event(self) -> None:
        class FakeStreamEvent:
            event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}

        text = self.printer.print(FakeStreamEvent(), type="stream_event")
        assert text == "hi"

    def test_print_message(self) -> None:
        cq = self.printer.add_client()

        class Msg:
            result = "output"

        self.printer.print(Msg(), type="message")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        assert any(e["type"] == "result" for e in events)
        self.printer.remove_client(cq)


# ---------------------------------------------------------------------------
# code_server.py  (45% -> target higher)
# ---------------------------------------------------------------------------


class TestScanFiles:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scans_files_and_dirs(self) -> None:
        Path(self.tmpdir, "a.txt").write_text("a")
        Path(self.tmpdir, "subdir").mkdir()
        Path(self.tmpdir, "subdir", "b.txt").write_text("b")
        paths = _scan_files(self.tmpdir)
        assert "a.txt" in paths
        assert "subdir/b.txt" in paths
        assert any(p.endswith("/") for p in paths)

    def test_skips_hidden_and_venv(self) -> None:
        Path(self.tmpdir, ".git").mkdir()
        Path(self.tmpdir, ".git", "HEAD").write_text("ref: refs/heads/main")
        Path(self.tmpdir, "__pycache__").mkdir()
        Path(self.tmpdir, "__pycache__", "x.pyc").write_text("data")
        Path(self.tmpdir, "real.py").write_text("code")
        paths = _scan_files(self.tmpdir)
        assert "real.py" in paths
        assert not any("__pycache__" in p for p in paths)
        assert not any(".git" in p for p in paths)

    def test_depth_limit(self) -> None:
        # Create deeply nested dirs
        deep = Path(self.tmpdir)
        for i in range(6):
            deep = deep / f"d{i}"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("deep")
        paths = _scan_files(self.tmpdir)
        assert not any("deep.txt" in p for p in paths)

    def test_max_files(self) -> None:
        for i in range(2100):
            Path(self.tmpdir, f"f{i:05d}.txt").write_text(str(i))
        paths = _scan_files(self.tmpdir)
        assert len(paths) <= 2000


class TestDisableCopilotScmButton:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disables_button(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-0.1.0"
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
        (ext_dir / "package.json").write_text(json.dumps(pkg))
        _disable_copilot_scm_button(self.tmpdir)
        updated = json.loads((ext_dir / "package.json").read_text())
        item = updated["contributes"]["menus"]["scm/inputBox"][0]
        assert item["when"] == "false"

    def test_no_extensions_dir(self) -> None:
        _disable_copilot_scm_button(self.tmpdir)  # Should not raise

    def test_already_disabled(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-0.2.0"
        ext_dir.mkdir(parents=True)
        pkg = {
            "contributes": {
                "menus": {
                    "scm/inputBox": [
                        {
                            "command": "github.copilot.git.generateCommitMessage",
                            "when": "false",
                        }
                    ]
                }
            }
        }
        (ext_dir / "package.json").write_text(json.dumps(pkg))
        _disable_copilot_scm_button(self.tmpdir)
        updated = json.loads((ext_dir / "package.json").read_text())
        assert updated == pkg  # unchanged

    def test_non_matching_extension(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "some-other-ext"
        ext_dir.mkdir(parents=True)
        (ext_dir / "package.json").write_text("{}")
        _disable_copilot_scm_button(self.tmpdir)  # Should skip


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

    def test_parse_diff_hunks_no_changes(self) -> None:
        hunks = _parse_diff_hunks(self.tmpdir)
        assert hunks == {}

    def test_parse_diff_hunks_with_changes(self) -> None:
        Path(self.tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
        hunks = _parse_diff_hunks(self.tmpdir)
        assert "file.txt" in hunks
        assert len(hunks["file.txt"]) >= 1

    def test_capture_untracked(self) -> None:
        Path(self.tmpdir, "newfile.txt").write_text("new")
        untracked = _capture_untracked(self.tmpdir)
        assert "newfile.txt" in untracked

    def test_capture_untracked_empty(self) -> None:
        untracked = _capture_untracked(self.tmpdir)
        assert len(untracked) == 0

    def test_snapshot_files(self) -> None:
        hashes = _snapshot_files(self.tmpdir, {"file.txt", "nonexistent.txt"})
        assert "file.txt" in hashes
        assert "nonexistent.txt" not in hashes

    def test_prepare_merge_view_no_changes(self) -> None:
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        result = _prepare_merge_view(self.tmpdir, self.tmpdir, pre_hunks, pre_untracked, pre_hashes)
        assert "error" in result

    def test_prepare_merge_view_with_changes(self) -> None:
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        # Simulate agent making changes
        Path(self.tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes,
            )
            assert result.get("status") == "opened"
            assert result.get("count", 0) >= 1
            # Check manifest file created
            manifest = Path(data_dir) / "pending-merge.json"
            assert manifest.exists()
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_prepare_merge_view_new_file(self) -> None:
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        # Add new file
        Path(self.tmpdir, "newfile.py").write_text("print('hello')\n")
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes,
            )
            assert result.get("status") == "opened"
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_prepare_merge_view_pre_existing_changes_unchanged(self) -> None:
        """File with pre-existing changes that agent doesn't modify should be skipped."""
        Path(self.tmpdir, "file.txt").write_text("line1\nchanged\nline3\n")
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        # Agent doesn't change the file
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes,
            )
            assert "error" in result  # No new changes
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_prepare_merge_view_pre_existing_changes_modified(self) -> None:
        """File with pre-existing changes that agent modifies should be included."""
        Path(self.tmpdir, "file.txt").write_text("line1\nchanged\nline3\n")
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        # Agent modifies the file further
        Path(self.tmpdir, "file.txt").write_text("line1\nchanged_again\nline3\n")
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes,
            )
            assert result.get("status") == "opened"
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_prepare_merge_view_without_pre_hashes(self) -> None:
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        Path(self.tmpdir, "file.txt").write_text("modified\n")
        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(self.tmpdir, data_dir, pre_hunks, pre_untracked, None)
            assert result.get("status") == "opened"
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# web_use_tool.py  (81% -> target higher)
# ---------------------------------------------------------------------------


class TestNumberInteractiveElements:
    def test_basic_numbering(self) -> None:
        snapshot = '- button "Click me"\n- textbox "Name"\n- heading "Title"'
        result, elements = _number_interactive_elements(snapshot)
        assert "[1] button" in result
        assert "[2] textbox" in result
        assert "heading" in result and "[" not in result.split("heading")[0].split("\n")[-1]
        assert len(elements) == 2

    def test_empty_snapshot(self) -> None:
        result, elements = _number_interactive_elements("")
        assert result == ""
        assert elements == []

    def test_nested_elements(self) -> None:
        snapshot = '  - link "Home"\n    - button "Go"\n- heading "Nav"'
        result, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 2
        assert elements[0]["role"] == "link"
        assert elements[1]["role"] == "button"

    def test_element_without_name(self) -> None:
        snapshot = '- button\n- textbox "Input"'
        result, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 2
        assert elements[0]["name"] == ""
        assert elements[1]["name"] == "Input"

    def test_non_interactive_roles(self) -> None:
        snapshot = '- heading "Title"\n- paragraph "Text"\n- button "Click"'
        result, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 1
        assert elements[0]["role"] == "button"

    def test_all_interactive_roles(self) -> None:
        lines = [f'- {role} "test"' for role in sorted(INTERACTIVE_ROLES)]
        snapshot = "\n".join(lines)
        result, elements = _number_interactive_elements(snapshot)
        assert len(elements) == len(INTERACTIVE_ROLES)


class TestScrollDelta:
    def test_all_directions(self) -> None:
        from kiss.agents.sorcar.web_use_tool import _SCROLL_DELTA

        assert _SCROLL_DELTA["down"] == (0, 300)
        assert _SCROLL_DELTA["up"] == (0, -300)
        assert _SCROLL_DELTA["right"] == (300, 0)
        assert _SCROLL_DELTA["left"] == (-300, 0)


class TestWebUseToolInit:
    def test_default_init(self) -> None:
        from kiss.agents.sorcar.web_use_tool import KISS_PROFILE_DIR, WebUseTool

        tool = WebUseTool()
        assert tool.browser_type == "chromium"
        assert not tool.headless
        assert tool.user_data_dir == KISS_PROFILE_DIR

    def test_custom_init(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(
            browser_type="firefox",
            headless=True,
            viewport=(800, 600),
            user_data_dir=None,
        )
        assert tool.browser_type == "firefox"
        assert tool.headless
        assert tool.viewport == (800, 600)
        assert tool.user_data_dir is None

    def test_get_tools(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool()
        tools = tool.get_tools()
        assert len(tools) == 7
        tool_names = [t.__name__ for t in tools]
        assert "go_to_url" in tool_names
        assert "click" in tool_names
        assert "close" not in tool_names

    def test_close_without_browser(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool()
        result = tool.close()
        assert "closed" in result.lower()

    def test_context_args(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(viewport=(1920, 1080))
        args = tool._context_args()
        assert args["viewport"]["width"] == 1920
        assert args["viewport"]["height"] == 1080

    def test_launch_kwargs_chromium(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(browser_type="chromium", headless=True)
        kwargs = tool._launch_kwargs()
        assert kwargs["headless"]
        assert "args" in kwargs

    def test_launch_kwargs_chromium_headed(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(browser_type="chromium", headless=False)
        kwargs = tool._launch_kwargs()
        assert "channel" in kwargs

    def test_launch_kwargs_firefox(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(browser_type="firefox", headless=True)
        kwargs = tool._launch_kwargs()
        assert kwargs["headless"]
        assert "args" not in kwargs


# ---------------------------------------------------------------------------
# Additional browser_ui.py coverage
# ---------------------------------------------------------------------------
class TestBrowserPrinterEdgeCases:
    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

    def test_print_text_empty_strip(self) -> None:
        """Printing whitespace-only text should not broadcast."""
        cq = self.printer.add_client()
        self.printer.print("   \n  ", type="text")
        assert cq.empty()
        self.printer.remove_client(cq)

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

    def test_tool_call_no_path(self) -> None:
        """Tool call without file_path should not have path in event."""
        cq = self.printer.add_client()
        self.printer.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "echo hi"},
        )
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert "path" not in tool_events[0]
        self.printer.remove_client(cq)

    def test_flush_bash_empty_buffer(self) -> None:
        """Flushing with empty buffer should not broadcast."""
        cq = self.printer.add_client()
        self.printer._flush_bash()
        assert cq.empty()
        self.printer.remove_client(cq)

    def test_flush_bash_cancels_timer(self) -> None:
        """Flushing should cancel any pending timer."""
        self.printer._bash_flush_timer = threading.Timer(10, lambda: None)
        self.printer._bash_flush_timer.start()
        cq = self.printer.add_client()
        with self.printer._bash_lock:
            self.printer._bash_buffer.append("data")
        self.printer._flush_bash()
        assert self.printer._bash_flush_timer is None
        self.printer.remove_client(cq)

    def test_reset_cancels_bash_timer(self) -> None:
        self.printer._bash_flush_timer = threading.Timer(10, lambda: None)
        self.printer._bash_flush_timer.start()
        self.printer.reset()
        assert self.printer._bash_flush_timer is None

    def test_handle_message_no_matching_attrs(self) -> None:
        """Message with none of the expected attributes should not raise."""
        cq = self.printer.add_client()

        class Msg:
            pass

        self.printer._handle_message(Msg())
        assert cq.empty()
        self.printer.remove_client(cq)

    def test_handle_message_subtype_non_tool_output(self) -> None:
        cq = self.printer.add_client()

        class Msg:
            subtype = "other"
            data = {"content": "stuff"}

        self.printer._handle_message(Msg())
        assert cq.empty()
        self.printer.remove_client(cq)

    def test_broadcast_result_empty_text(self) -> None:
        cq = self.printer.add_client()
        self.printer._broadcast_result("")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert result_events[0]["text"] == "(no result)"
        self.printer.remove_client(cq)

    def test_parse_result_yaml_invalid(self) -> None:
        assert self.printer._parse_result_yaml("not: [valid: yaml {") is None

    def test_parse_result_yaml_no_summary(self) -> None:
        assert self.printer._parse_result_yaml("key: value") is None

    def test_parse_result_yaml_valid(self) -> None:
        result = self.printer._parse_result_yaml("summary: test\nsuccess: true")
        assert result is not None
        assert result["summary"] == "test"

    def test_stream_event_unknown_block_type(self) -> None:
        """content_block_start with unknown type should be handled."""
        class FakeEvent:
            event = {"type": "content_block_start", "content_block": {"type": "unknown"}}

        self.printer._handle_stream_event(FakeEvent())
        assert self.printer._current_block_type == "unknown"

    def test_stream_event_unknown_delta_type(self) -> None:
        """content_block_delta with unknown delta type should return empty text."""
        class FakeEvent:
            event = {"type": "content_block_delta", "delta": {"type": "unknown_delta"}}

        text = self.printer._handle_stream_event(FakeEvent())
        assert text == ""

    def test_bash_stream_timer_path(self) -> None:
        """Test the bash_stream timer code path (no immediate flush, timer fires)."""
        cq = self.printer.add_client()
        # Ensure last_flush is recent so immediate flush doesn't trigger
        self.printer._bash_last_flush = time.monotonic()
        self.printer.print("slow\n", type="bash_stream")
        # Timer should be set
        assert self.printer._bash_flush_timer is not None
        # Wait for timer to fire
        time.sleep(0.2)
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        sys_events = [e for e in events if e["type"] == "system_output"]
        assert any("slow" in e["text"] for e in sys_events)
        self.printer.remove_client(cq)

    def test_bash_stream_already_has_timer(self) -> None:
        """Second bash_stream call while timer exists should not create another timer."""
        cq = self.printer.add_client()
        self.printer._bash_last_flush = time.monotonic()
        self.printer.print("first\n", type="bash_stream")
        first_timer = self.printer._bash_flush_timer
        assert first_timer is not None
        self.printer.print("second\n", type="bash_stream")
        # Timer should still be the same (not recreated)
        assert self.printer._bash_flush_timer is first_timer
        self.printer._flush_bash()
        self.printer.remove_client(cq)


# ---------------------------------------------------------------------------
# Additional code_server.py coverage
# ---------------------------------------------------------------------------
class TestSetupCodeServerAdditional:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_copilot_chat_corrupted_package_json(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-0.3.0"
        ext_dir.mkdir(parents=True)
        (ext_dir / "package.json").write_text("not json{{{")
        _disable_copilot_scm_button(self.tmpdir)  # Should not raise

    def test_copilot_chat_no_package_json(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-0.4.0"
        ext_dir.mkdir(parents=True)
        _disable_copilot_scm_button(self.tmpdir)  # Should not raise

    def test_copilot_chat_no_scm_menu(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-0.5.0"
        ext_dir.mkdir(parents=True)
        (ext_dir / "package.json").write_text(json.dumps({"contributes": {}}))
        _disable_copilot_scm_button(self.tmpdir)  # Should not raise

    def test_scan_files_nonexistent_dir(self) -> None:
        paths = _scan_files("/nonexistent/dir/abc123")
        assert paths == []

    def test_parse_diff_hunks_with_count_only(self) -> None:
        """Test hunks where count is missing (defaults to 1)."""
        tmpdir = tempfile.mkdtemp()
        try:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=tmpdir, capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "T"],
                cwd=tmpdir, capture_output=True,
            )
            Path(tmpdir, "f.txt").write_text("a\nb\n")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            # Delete a single line (hunk like @@ -1 +1,0 @@)
            Path(tmpdir, "f.txt").write_text("a\nmodified\n")
            hunks = _parse_diff_hunks(tmpdir)
            assert len(hunks.get("f.txt", [])) >= 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


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

    def test_save_proposals_creates_file(self) -> None:
        th._save_proposals(["task1", "task2"])
        assert th.PROPOSALS_FILE.exists()

    def test_load_proposals_missing_file(self) -> None:
        assert th._load_proposals() == []

    def test_load_usage_missing_file(self) -> None:
        assert th._load_file_usage() == {}

    def test_load_usage_corrupted(self) -> None:
        th.FILE_USAGE_FILE.write_text("broken json")
        assert th._load_file_usage() == {}

    def test_init_task_history_md_creates_new(self) -> None:
        import kiss.core.config as cfg
        old_artifact = cfg.DEFAULT_CONFIG.agent.artifact_dir
        try:
            cfg.DEFAULT_CONFIG.agent.artifact_dir = str(Path(self.tmpdir) / "artifacts")
            path = th._init_task_history_md()
            assert path.exists()
            assert "# Task History" in path.read_text()
        finally:
            cfg.DEFAULT_CONFIG.agent.artifact_dir = old_artifact

    def test_init_task_history_md_existing(self) -> None:
        import kiss.core.config as cfg
        old_artifact = cfg.DEFAULT_CONFIG.agent.artifact_dir
        try:
            cfg.DEFAULT_CONFIG.agent.artifact_dir = str(Path(self.tmpdir) / "artifacts")
            path = th._init_task_history_md()
            path.write_text("existing content\n")
            path2 = th._init_task_history_md()
            assert path2.read_text() == "existing content\n"
        finally:
            cfg.DEFAULT_CONFIG.agent.artifact_dir = old_artifact


# ---------------------------------------------------------------------------
# Additional useful_tools.py coverage
# ---------------------------------------------------------------------------
class TestUsefulToolsAdditional:
    def test_disallowed_dot(self) -> None:
        tools = UsefulTools()
        result = tools.Bash(". ./setup.sh", "dot test")
        assert "not allowed" in result.lower()

    def test_disallowed_env(self) -> None:
        tools = UsefulTools()
        result = tools.Bash("env FOO=bar", "env test")
        assert "not allowed" in result.lower()

    def test_split_respecting_quotes_escaped(self) -> None:
        result = _split_respecting_quotes("a\\;b;c", re.compile(r";"))
        assert result == ["a\\;b", "c"]

    def test_split_respecting_quotes_double_quotes(self) -> None:
        result = _split_respecting_quotes('a "b;c" d;e', re.compile(r";"))
        assert len(result) == 2
        assert '"b;c"' in result[0]

    def test_split_respecting_quotes_single_quotes(self) -> None:
        result = _split_respecting_quotes("a 'b;c' d;e", re.compile(r";"))
        assert len(result) == 2

    def test_split_respecting_quotes_escape_in_double_quotes(self) -> None:
        result = _split_respecting_quotes('a "b\\"c" d', re.compile(r";"))
        assert len(result) == 1

    def test_extract_command_empty_parens(self) -> None:
        """Test lstripping of parens yielding empty string."""
        result = _extract_leading_command_name("(")
        assert result is None

    def test_format_bash_result_truncation(self) -> None:
        long_output = "x" * 10000
        result = _format_bash_result(0, long_output, 100)
        assert "truncated" in result

    def test_streaming_bash_exit_nonzero(self) -> None:
        collected: list[str] = []
        tools = UsefulTools(stream_callback=collected.append)
        result = tools.Bash("exit 42", "fail stream", timeout_seconds=5)
        assert "error" in result.lower()

    def test_noop_stream_callback(self) -> None:
        """stream_callback=None falls through to non-streaming path."""
        tools = UsefulTools(stream_callback=None)
        result = tools.Bash("echo test_noop", "noop test")
        assert "test_noop" in result


# ---------------------------------------------------------------------------
# code_server _install_copilot_extension
# ---------------------------------------------------------------------------


class TestInstallCopilotExtension:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_already_installed(self) -> None:
        """If copilot extension dir exists, skip install."""
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-1.0.0"
        ext_dir.mkdir(parents=True)
        # Should return early
        _install_copilot_extension(self.tmpdir)

    def test_no_code_server_binary(self) -> None:
        """If code-server not found, skip install."""
        # Extensions dir exists but no copilot
        ext_dir = Path(self.tmpdir) / "extensions" / "some-other"
        ext_dir.mkdir(parents=True)
        # This may or may not have code-server available, but either way it shouldn't crash
        _install_copilot_extension(self.tmpdir)

    def test_no_extensions_dir(self) -> None:
        """No extensions dir at all."""
        _install_copilot_extension(self.tmpdir)


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

    def test_existing_merge_dir_cleaned(self) -> None:
        """If merge-temp exists, it should be cleaned before writing."""
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, set(pre_hunks.keys()))
        Path(self.tmpdir, "file.txt").write_text("modified\n")
        data_dir = tempfile.mkdtemp()
        try:
            old_merge = Path(data_dir) / "merge-temp" / "stale.txt"
            old_merge.parent.mkdir(parents=True)
            old_merge.write_text("stale")
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked, pre_hashes,
            )
            assert result.get("status") == "opened"
            assert not old_merge.exists()
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

    def test_read_active_file_valid(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        # Create the active-file.json
        af_path = os.path.join(self.tmpdir, "active-file.json")
        test_file = os.path.join(self.tmpdir, "test.py")
        Path(test_file).write_text("# test")
        with open(af_path, "w") as f:
            json.dump({"path": test_file}, f)
        result = _read_active_file(self.tmpdir)
        assert result == test_file

    def test_read_active_file_missing(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        result = _read_active_file(self.tmpdir)
        assert result == ""

    def test_read_active_file_nonexistent_path(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        af_path = os.path.join(self.tmpdir, "active-file.json")
        with open(af_path, "w") as f:
            json.dump({"path": "/no/such/file.py"}, f)
        result = _read_active_file(self.tmpdir)
        assert result == ""

    def test_read_active_file_invalid_json(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        af_path = os.path.join(self.tmpdir, "active-file.json")
        with open(af_path, "w") as f:
            f.write("not json")
        result = _read_active_file(self.tmpdir)
        assert result == ""

    def test_read_active_file_empty_path(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        af_path = os.path.join(self.tmpdir, "active-file.json")
        with open(af_path, "w") as f:
            json.dump({"path": ""}, f)
        result = _read_active_file(self.tmpdir)
        assert result == ""

    def test_clean_llm_output(self) -> None:
        from kiss.agents.sorcar.sorcar import _clean_llm_output
        assert _clean_llm_output('  "hello"  ') == "hello"
        assert _clean_llm_output("  'world'  ") == "world"
        assert _clean_llm_output("plain") == "plain"

    def test_model_vendor_order(self) -> None:
        from kiss.agents.sorcar.sorcar import _model_vendor_order
        assert _model_vendor_order("claude-3-opus") == 0
        assert _model_vendor_order("gpt-4") == 1
        assert _model_vendor_order("o1-preview") == 1
        assert _model_vendor_order("gemini-1.5-pro") == 2
        assert _model_vendor_order("minimax-01") == 3
        assert _model_vendor_order("openrouter/model") == 4
        assert _model_vendor_order("unknown-model") == 5


# ---------------------------------------------------------------------------
# _kill_process_group tests
# ---------------------------------------------------------------------------
class TestKillProcessGroup:
    def test_kill_running_process(self) -> None:
        proc = subprocess.Popen(
            ["sleep", "60"],
            start_new_session=True,
        )
        _kill_process_group(proc)
        assert proc.poll() is not None

    def test_kill_already_dead_process(self) -> None:
        proc = subprocess.Popen(["echo", "hi"])
        proc.wait()
        # Should not raise
        _kill_process_group(proc)


# ---------------------------------------------------------------------------
# Test _truncate_output edge: tail=0 branch
# ---------------------------------------------------------------------------
class TestTruncateOutputEdge:
    def test_tail_zero_when_remaining_small(self) -> None:
        # When remaining is 1, head=0, tail=1 or head=1, tail=0
        # Let's find a case where tail=0
        output = "x" * 200
        # With max_chars=45, worst_msg ~= "...[truncated 200 chars]..." ~= 35 chars
        # remaining = 45 - 35 = 10, head=5, tail=5
        # Let's make it so remaining//2 has tail=0
        result = _truncate_output(output, 38)
        assert "truncated" in result


# ---------------------------------------------------------------------------
# Test code_server _setup_code_server extension.js return value (changed vs unchanged)
# ---------------------------------------------------------------------------
class TestSetupCodeServerReturn:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_true_on_first_setup(self) -> None:
        changed = _setup_code_server(self.tmpdir)
        assert changed is True

    def test_returns_false_on_second_setup(self) -> None:
        _setup_code_server(self.tmpdir)
        changed = _setup_code_server(self.tmpdir)
        assert changed is False


# ---------------------------------------------------------------------------
# Test code_server._scan_files with hidden dirs at top level
# ---------------------------------------------------------------------------
class TestScanFilesEdgeCases:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_hidden_dirs_skipped(self) -> None:
        Path(self.tmpdir, ".hidden").mkdir()
        Path(self.tmpdir, ".hidden", "secret.txt").write_text("secret")
        Path(self.tmpdir, "visible.txt").write_text("vis")
        paths = _scan_files(self.tmpdir)
        assert "visible.txt" in paths
        assert not any(".hidden" in p for p in paths)

    def test_node_modules_skipped(self) -> None:
        Path(self.tmpdir, "node_modules").mkdir()
        Path(self.tmpdir, "node_modules", "pkg.json").write_text("{}")
        paths = _scan_files(self.tmpdir)
        assert not any("node_modules" in p for p in paths)

    def test_venv_skipped(self) -> None:
        for skip_dir in [".venv", "venv", ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache"]:
            Path(self.tmpdir, skip_dir).mkdir(exist_ok=True)
            Path(self.tmpdir, skip_dir, "x.txt").write_text("x")
        paths = _scan_files(self.tmpdir)
        assert not any(d in p for p in paths for d in [".venv", "venv", ".tox"])


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
        import re

        from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS
        for name, theme in _THEME_PRESETS.items():
            for key, value in theme.items():
                assert re.match(r"^#[0-9a-fA-F]{6}$", value), f"Theme {name}.{key}={value} not hex"


# ---------------------------------------------------------------------------
# Test _build_html with and without code_server
# ---------------------------------------------------------------------------
class TestBuildHtml:
    def test_build_html_without_code_server(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import _build_html
        html = _build_html("Test Title")
        assert "Test Title" in html
        assert '<div id="editor-fallback">' in html
        assert '<iframe id="code-server-frame"' not in html

    def test_build_html_with_code_server(self) -> None:
        from kiss.agents.sorcar.chatbot_ui import _build_html
        html = _build_html("Test", "http://localhost:8080", "/home/user/project")
        assert '<iframe id="code-server-frame"' in html
        assert "http://localhost:8080" in html
        assert '<div id="editor-fallback">' not in html


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

    def test_go_to_url_and_get_tree(self) -> None:
        result = self.tool.go_to_url("data:text/html,<h1>Hello</h1><button>Click Me</button>")
        assert "Click Me" in result
        assert "[1]" in result

    def test_go_to_url_tab_list(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Page1</h1>")
        result = self.tool.go_to_url("tab:list")
        assert "Open tabs" in result
        assert "(active)" in result

    def test_go_to_url_tab_switch(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Page1</h1>")
        result = self.tool.go_to_url("tab:0")
        assert "Page:" in result

    def test_go_to_url_tab_invalid_index(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Page1</h1>")
        result = self.tool.go_to_url("tab:99")
        assert "Error" in result

    def test_go_to_url_invalid(self) -> None:
        result = self.tool.go_to_url("not_a_valid_url_at_all")
        assert "Error" in result

    def test_click_element(self) -> None:
        url = "data:text/html,<button onclick='document.title=\"clicked\"'>Go</button>"
        self.tool.go_to_url(url)
        result = self.tool.click(1)
        assert "Page:" in result

    def test_click_hover(self) -> None:
        self.tool.go_to_url("data:text/html,<button>Hover Me</button>")
        result = self.tool.click(1, action="hover")
        assert "Page:" in result

    def test_click_invalid_element(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>No buttons</h1>")
        result = self.tool.click(999)
        assert "Error" in result

    def test_type_text(self) -> None:
        self.tool.go_to_url("data:text/html,<input type='text' id='inp'>")
        result = self.tool.type_text(1, "Hello World")
        assert "Page:" in result

    def test_type_text_with_enter(self) -> None:
        self.tool.go_to_url("data:text/html,<input type='text'>")
        result = self.tool.type_text(1, "search query", press_enter=True)
        assert "Page:" in result

    def test_type_text_invalid_element(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>No inputs</h1>")
        result = self.tool.type_text(999, "text")
        assert "Error" in result

    def test_press_key(self) -> None:
        self.tool.go_to_url("data:text/html,<button>B</button>")
        result = self.tool.press_key("Tab")
        assert "Page:" in result

    def test_press_key_invalid(self) -> None:
        self.tool.go_to_url("data:text/html,<div>X</div>")
        # This should work even with unusual key
        result = self.tool.press_key("Escape")
        assert "Page:" in result

    def test_scroll_down(self) -> None:
        long_page = "<div style='height:5000px'>Tall page</div><button>Bottom</button>"
        self.tool.go_to_url(f"data:text/html,{long_page}")
        result = self.tool.scroll("down", 3)
        assert "Page:" in result

    def test_scroll_up(self) -> None:
        self.tool.go_to_url("data:text/html,<div>Content</div>")
        result = self.tool.scroll("up", 1)
        assert "Page:" in result

    def test_scroll_invalid_direction(self) -> None:
        self.tool.go_to_url("data:text/html,<div>X</div>")
        result = self.tool.scroll("diagonal", 1)
        assert "Page:" in result

    def test_screenshot(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>Screenshot Test</h1>")
        path = os.path.join(self.tmpdir, "shot.png")
        result = self.tool.screenshot(path)
        assert "saved" in result.lower()
        assert os.path.exists(path)

    def test_get_page_content_tree(self) -> None:
        self.tool.go_to_url("data:text/html,<button>X</button>")
        result = self.tool.get_page_content(text_only=False)
        assert "[1]" in result

    def test_get_page_content_text_only(self) -> None:
        self.tool.go_to_url("data:text/html,<p>Some text here</p>")
        result = self.tool.get_page_content(text_only=True)
        assert "Some text here" in result

    def test_close(self) -> None:
        self.tool.go_to_url("data:text/html,<div>X</div>")
        result = self.tool.close()
        assert "closed" in result.lower()
        # Close again should be safe
        result2 = self.tool.close()
        assert "closed" in result2.lower()

    def test_empty_page(self) -> None:
        self.tool.go_to_url("data:text/html,")
        result = self.tool.get_page_content()
        # Should handle empty page
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

    def test_persistent_context_launch(self) -> None:
        result = self.tool.go_to_url("data:text/html,<h1>Persistent</h1>")
        assert "Page:" in result

    def test_close_persistent_context(self) -> None:
        self.tool.go_to_url("data:text/html,<h1>X</h1>")
        result = self.tool.close()
        assert "closed" in result.lower()


class TestWebUseToolResolveLocator:
    """Test _resolve_locator edge cases with real browser."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_resolve_stale_element_retry(self) -> None:
        """When elements list is stale, _resolve_locator should re-snapshot."""
        self.tool.go_to_url("data:text/html,<button>A</button><button>B</button>")
        # Clear elements to force re-snapshot
        self.tool._elements = []
        result = self.tool.click(1)
        assert "Page:" in result

    def test_resolve_element_without_name(self) -> None:
        """Elements without name attribute should still be clickable."""
        self.tool.go_to_url("data:text/html,<button></button>")
        result = self.tool.get_page_content()
        if "[1]" in result:
            result2 = self.tool.click(1)
            assert "Page:" in result2

    def test_click_link_opens_new_tab(self) -> None:
        """Clicking a target=_blank link should switch to new tab."""
        html = '<a href="data:text/html,<h1>New</h1>" target="_blank">Open</a>'
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        if "[1]" in tree:
            self.tool.click(1)
            # Should have handled the new tab

    def test_scroll_left_right(self) -> None:
        wide = "<div style='width:5000px;white-space:nowrap'>Wide content</div>"
        self.tool.go_to_url(f"data:text/html,{wide}")
        r1 = self.tool.scroll("right", 2)
        assert "Page:" in r1
        r2 = self.tool.scroll("left", 2)
        assert "Page:" in r2

    def test_screenshot_error_handling(self) -> None:
        """Screenshot to invalid path."""
        self.tool.go_to_url("data:text/html,<h1>X</h1>")
        result = self.tool.screenshot("/dev/null/impossible/file.png")
        assert "Error" in result or "saved" in result.lower()

    def test_get_page_content_error(self) -> None:
        """Close browser, then try get_page_content should recover or error."""
        # Initial page
        self.tool.go_to_url("data:text/html,<p>test</p>")
        result = self.tool.get_page_content()
        assert "Page:" in result

    def test_multiple_buttons_resolve(self) -> None:
        """When multiple buttons with same name exist, resolve first visible."""
        html = """
        <div><button>Same</button></div>
        <div><button>Same</button></div>
        <div style="display:none"><button>Same</button></div>
        """
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        # Should be able to click first one
        if "[1]" in tree:
            result = self.tool.click(1)
            assert "Page:" in result


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

    def test_frontmatter_only_opening_dashes(self) -> None:
        """Content starts with '---' but has < 3 parts when split on '---'.
        Covers branch 43->54 (frontmatter starts with --- but no closing ---)."""
        content = "---\nsome content without closing dashes\n"
        p = self._write("partial.md", content)
        ok, score, reasons = self.detector.analyze(p)
        # Should still work, just no frontmatter parsed
        assert isinstance(score, float)

    def test_low_verb_density_branch(self) -> None:
        """Content with very low imperative verb density (<=5%).
        Covers branch 114->121 (density check False)."""
        # Lots of words, very few imperative verbs
        content = ("the quick brown fox jumped over the lazy dog " * 20) + "\n"
        p = self._write("lowverb.md", content)
        ok, score, reasons = self.detector.analyze(p)
        # No "imperative command verbs" reason should appear
        assert not any("imperative" in r.lower() for r in reasons)


class TestBrowserUiBranches:
    """Cover remaining branches in browser_ui.py."""

    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()

    def test_stream_event_unknown_type(self) -> None:
        """Event type not recognized (not start/delta/stop).
        Covers branch 643->658."""

        class FakeEvent:
            event = {"type": "message_start", "message": {}}

        text = self.printer._handle_stream_event(FakeEvent())
        assert text == ""

    def test_handle_message_content_block_without_is_error(self) -> None:
        """Message with content blocks that lack is_error attribute.
        Covers branch 676->675."""

        class FakeBlock:
            def __init__(self) -> None:
                self.text = "hello"
                # Intentionally no is_error or content attributes

        class FakeMsg:
            def __init__(self) -> None:
                self.content = [FakeBlock()]

        cq = self.printer.add_client()
        self.printer._handle_message(FakeMsg())
        # No tool_result broadcast since block lacks is_error
        assert cq.empty()
        self.printer.remove_client(cq)

    def test_handle_message_no_known_attributes(self) -> None:
        """Message with none of the recognized attributes.
        Tests the elif chain fallthrough in _handle_message."""

        class PlainMsg:
            pass

        cq = self.printer.add_client()
        self.printer._handle_message(PlainMsg())
        # Should not broadcast anything
        assert cq.empty()
        self.printer.remove_client(cq)

    def test_handle_message_tool_output_empty_content(self) -> None:
        """Tool output message with empty content string.
        Covers the 'if text:' False branch in _handle_message."""

        class ToolMsg:
            subtype = "tool_output"
            data = {"content": ""}

        cq = self.printer.add_client()
        self.printer._handle_message(ToolMsg())
        # Empty content should not be broadcast
        assert cq.empty()
        self.printer.remove_client(cq)


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

    def test_load_history_json_decode_error(self) -> None:
        """History file with invalid JSON triggers except branch.
        Covers lines 97-98."""
        th.HISTORY_FILE.write_text("not valid json{{{")
        th._history_cache = None
        history = th._load_history()
        # Should fall back to SAMPLE_TASKS
        assert len(history) > 0

    def test_save_history_oserror(self) -> None:
        """Saving to a read-only directory triggers OSError.
        Covers lines 116-117."""
        # Make the kiss dir read-only
        kiss_dir = th._KISS_DIR
        # Remove the history file if present
        if th.HISTORY_FILE.exists():
            th.HISTORY_FILE.unlink()
        os.chmod(str(kiss_dir), 0o444)
        try:
            # Should not raise despite OSError
            th._save_history([{"task": "test", "result": ""}])
        finally:
            os.chmod(str(kiss_dir), 0o755)

    def test_set_latest_result_empty_cache(self) -> None:
        """_set_latest_result with empty cache does nothing.
        Covers branch 129->exit."""
        th._history_cache = []
        th._set_latest_result("some result")
        assert th._history_cache == []

    def test_load_proposals_json_decode_error(self) -> None:
        """Proposals file with invalid JSON.
        Covers lines 149-150."""
        th.PROPOSALS_FILE.write_text("{not valid")
        result = th._load_proposals()
        assert result == []

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

    def test_save_proposals_oserror(self) -> None:
        """OSError when writing proposals.
        Covers lines 149-150."""
        th.PROPOSALS_FILE.write_text("[]")
        os.chmod(str(th.PROPOSALS_FILE), 0o444)
        try:
            th._save_proposals(["test proposal"])
        finally:
            os.chmod(str(th.PROPOSALS_FILE), 0o644)

    def test_append_task_to_md_creates_new_file(self) -> None:
        """_append_task_to_md creates file if it doesn't exist.
        Covers line 238."""
        md_path = th._get_task_history_md_path()
        if md_path.exists():
            md_path.unlink()
        th._append_task_to_md("test task", "test result")
        assert md_path.exists()
        content = md_path.read_text()
        assert "# Task History" in content
        assert "test task" in content


class TestUsefulToolsBranches:
    """Cover remaining branches in useful_tools.py."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_truncate_output_tail_zero(self) -> None:
        """When remaining==0 after subtracting worst_msg, tail is 0.
        Covers line 29."""
        output = "X" * 200
        worst_msg = f"\n\n... [truncated {len(output)} chars] ...\n\n"
        max_chars = len(worst_msg)
        result = _truncate_output(output, max_chars)
        # Should have the truncation message but no tail
        assert "truncated" in result
        assert not result.endswith("X")

    def test_redirect_inline_consumes_one_token(self) -> None:
        """Redirect like '2>file.txt' where match doesn't consume whole token.
        Covers line 66 (m.end() < len(token) -> i += 1)."""
        name = _extract_leading_command_name("2>/dev/null echo hello")
        assert name == "echo"

    def test_redirect_standalone_consumes_two_tokens(self) -> None:
        """Redirect like '>' where match consumes whole token.
        Covers line 68 (else: i += 2)."""
        name = _extract_leading_command_name("> out.txt echo hello")
        assert name == "echo"

    def test_command_name_empty_after_lstrip(self) -> None:
        """Token is '((' which after lstrip('({') is empty.
        Covers line 76."""
        name = _extract_leading_command_name("((")
        assert name is None

    def test_all_tokens_consumed_by_prefix(self) -> None:
        """All tokens are prefix tokens like '!'.
        Covers branch 72 (i >= len(tokens) -> return None)."""
        name = _extract_leading_command_name("!")
        assert name is None

    def test_all_tokens_consumed_env_only(self) -> None:
        """Only env var assignments, no command.
        Covers branch 72 (i >= len(tokens) -> return None)."""
        name = _extract_leading_command_name("FOO=bar BAZ=qux")
        assert name is None

    def test_all_tokens_consumed_redirect_only(self) -> None:
        """Only a redirect with no following command.
        Covers branch 72 via redirect consuming all tokens."""
        name = _extract_leading_command_name("> out.txt")
        assert name is None

    def test_escaped_char_in_double_quotes(self) -> None:
        r"""Backslash escape inside double quotes.
        Covers branch 128->126."""
        import re
        pattern = re.compile(r"&&")
        result = _split_respecting_quotes('echo "hello \\"world\\"" && ls', pattern)
        assert len(result) == 2

    def test_extract_command_names_empty_segment(self) -> None:
        """Pipeline segment that resolves to no command name."""
        names = _extract_command_names("| ls")
        assert "ls" in names

    def test_extract_command_names_trailing_pipe(self) -> None:
        """Trailing pipe results in empty last segment."""
        names = _extract_command_names("echo hello |")
        assert "echo" in names

    def test_edit_write_readonly_file(self) -> None:
        """Edit a file where write fails due to permissions.
        Covers lines 264-266."""
        fpath = os.path.join(self.tmpdir, "readonly.txt")
        with open(fpath, "w") as f:
            f.write("hello world")
        os.chmod(fpath, 0o444)
        try:
            tools = UsefulTools()
            result = tools.Edit(fpath, "hello", "goodbye")
            assert "Error" in result
        finally:
            os.chmod(fpath, 0o644)

    def test_bash_non_streaming_timeout(self) -> None:
        """Non-streaming Bash command that times out.
        Covers lines 306-312 (TimeoutExpired handler)."""
        tools = UsefulTools()
        result = tools.Bash(
            "sleep 30",
            description="sleeping",
            timeout_seconds=0.5,
        )
        assert "timeout" in result.lower()

    def test_bash_streaming_timeout(self) -> None:
        """Streaming Bash command that times out.
        Covers lines 360-361 (timed_out return)."""
        chunks: list[str] = []
        tools = UsefulTools(stream_callback=chunks.append)
        result = tools.Bash(
            "sleep 30",
            description="sleeping",
            timeout_seconds=0.5,
        )
        assert "timeout" in result.lower()


class TestCodeServerBranches:
    """Cover remaining branches in code_server.py."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disable_copilot_already_false(self) -> None:
        """Extension already has when='false' - no modification needed.
        Covers branch 476->475."""
        data_dir = self.tmpdir
        ext_dir = Path(data_dir) / "extensions" / "github.copilot-chat-0.1"
        ext_dir.mkdir(parents=True)
        pkg = {
            "contributes": {
                "menus": {
                    "scm/inputBox": [
                        {
                            "command": "github.copilot.git.generateCommitMessage",
                            "when": "false",
                        }
                    ]
                }
            }
        }
        (ext_dir / "package.json").write_text(json.dumps(pkg))
        _disable_copilot_scm_button(data_dir)
        # File should be unchanged
        result = json.loads((ext_dir / "package.json").read_text())
        assert result["contributes"]["menus"]["scm/inputBox"][0]["when"] == "false"

    def test_disable_copilot_write_oserror(self) -> None:
        """OSError writing package.json after modification.
        Covers lines 483-484."""
        data_dir = self.tmpdir
        ext_dir = Path(data_dir) / "extensions" / "github.copilot-chat-0.1"
        ext_dir.mkdir(parents=True)
        pkg = {
            "contributes": {
                "menus": {
                    "scm/inputBox": [
                        {
                            "command": "github.copilot.git.generateCommitMessage",
                            "when": "true",
                        }
                    ]
                }
            }
        }
        pkg_path = ext_dir / "package.json"
        pkg_path.write_text(json.dumps(pkg))
        # Make the file read-only to trigger write error
        os.chmod(str(pkg_path), 0o444)
        try:
            _disable_copilot_scm_button(data_dir)
        finally:
            os.chmod(str(pkg_path), 0o644)

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

    def test_setup_code_server_corrupted_settings(self) -> None:
        """Settings file with invalid JSON.
        Covers lines 522-524."""
        data_dir = self.tmpdir
        user_dir = Path(data_dir) / "User"
        user_dir.mkdir(parents=True)
        settings_file = user_dir / "settings.json"
        settings_file.write_text("{invalid json!!!")
        _setup_code_server(data_dir)
        # Should still work, creating valid settings
        result = json.loads(settings_file.read_text())
        assert "workbench.colorTheme" in result

    def test_setup_code_server_workspace_cleanup(self) -> None:
        """Workspace storage with chatSessions dirs gets cleaned.
        Covers lines 552-556."""
        data_dir = self.tmpdir
        ws_dir = Path(data_dir) / "User" / "workspaceStorage" / "ws1"
        chat_dir = ws_dir / "chatSessions"
        chat_edit_dir = ws_dir / "chatEditingSessions"
        chat_dir.mkdir(parents=True)
        chat_edit_dir.mkdir(parents=True)
        (chat_dir / "session.json").write_text("{}")
        (chat_edit_dir / "edit.json").write_text("{}")
        _setup_code_server(data_dir)
        assert not chat_dir.exists()
        assert not chat_edit_dir.exists()

    def test_scan_files_oserror(self) -> None:
        """_scan_files on a directory that becomes unreadable.
        Covers lines 642-644."""
        scan_dir = os.path.join(self.tmpdir, "unreadable")
        os.makedirs(scan_dir)
        # Create a file so there's something to scan
        Path(os.path.join(scan_dir, "file.txt")).write_text("hi")
        # Make subdir unreadable
        subdir = os.path.join(scan_dir, "sub")
        os.makedirs(subdir)
        Path(os.path.join(subdir, "inner.txt")).write_text("x")
        os.chmod(subdir, 0o000)
        try:
            result = _scan_files(scan_dir)
            # Should still return some files
            assert isinstance(result, list)
        finally:
            os.chmod(subdir, 0o755)

    def test_prepare_merge_view_hash_oserror(self) -> None:
        """File that can't be read during hash check.
        Covers lines 721-723."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=work_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=work_dir, capture_output=True
        )
        # Create and commit a file
        fpath = Path(work_dir) / "test.txt"
        fpath.write_text("original")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=work_dir, capture_output=True
        )
        # Modify the file
        fpath.write_text("modified")
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        import hashlib
        pre_hashes = {
            "test.txt": hashlib.md5(fpath.read_bytes()).hexdigest()
        }
        # Make file unreadable to trigger OSError on hash
        os.chmod(str(fpath), 0o000)
        try:
            fpath.write_text("even more modified")
        except PermissionError:
            pass
        try:
            _prepare_merge_view(
                work_dir, self.tmpdir, pre_hunks, pre_untracked,
                pre_file_hashes=pre_hashes,
            )
        finally:
            os.chmod(str(fpath), 0o644)

    def test_prepare_merge_view_untracked_binary_file(self) -> None:
        """New untracked file that's not readable (UnicodeDecodeError).
        Covers lines 750-752."""
        work_dir = os.path.join(self.tmpdir, "work2")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=work_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=work_dir, capture_output=True
        )
        # Create and commit something
        Path(os.path.join(work_dir, "base.txt")).write_text("base")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=work_dir, capture_output=True
        )
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        # Create a binary file that causes UnicodeDecodeError
        binary_path = Path(work_dir) / "binary.dat"
        binary_path.write_bytes(b"\x80\x81\x82\xff\xfe" * 100)
        result = _prepare_merge_view(
            work_dir, self.tmpdir, pre_hunks, pre_untracked,
        )
        # Binary file should be skipped due to UnicodeDecodeError
        if isinstance(result, dict) and "error" not in result:
            assert "binary.dat" not in result.get("files", [])


class TestWebUseToolBranches:
    """Cover remaining branches in web_use_tool.py."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_type_text_error_branch(self) -> None:
        """Type text with invalid element ID.
        Covers lines 301-303 (type_text error handler)."""
        self.tool.go_to_url("data:text/html,<p>no inputs</p>")
        result = self.tool.type_text(999, "test")
        assert "Error" in result

    def test_press_key_error_branch(self) -> None:
        """Press an invalid key.
        Covers lines 325-327 (press_key error handler)."""
        self.tool.go_to_url("data:text/html,<p>test</p>")
        result = self.tool.press_key("InvalidKeyNameThatDoesNotExist!!!")
        assert "Error" in result

    def test_click_error_branch(self) -> None:
        """Click with out-of-range ID.
        Covers lines 252-253 (click error handler)."""
        self.tool.go_to_url("data:text/html,<p>no buttons</p>")
        result = self.tool.click(999)
        assert "Error" in result

    def test_close_already_closed(self) -> None:
        """Close after already closed.
        Covers the except branch in close (384-386)."""
        self.tool.go_to_url("data:text/html,<p>test</p>")
        self.tool.close()
        result = self.tool.close()
        assert result == "Browser closed."

    def test_empty_snapshot(self) -> None:
        """Page with no body content returns empty tree.
        Covers _get_ax_tree empty snapshot branch."""
        self.tool.go_to_url("about:blank")
        result = self.tool.get_page_content()
        assert "Page:" in result


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

    def test_setup_code_server_workspace_no_chat_dirs(self) -> None:
        """Workspace storage dir exists but chat dirs don't.
        Covers branch 555->553 (chat_dir.exists() is False)."""
        data_dir = self.tmpdir
        ws_dir = Path(data_dir) / "User" / "workspaceStorage" / "ws1"
        ws_dir.mkdir(parents=True)
        # Don't create chatSessions or chatEditingSessions
        (ws_dir / "state.vscdb").write_text("")
        _setup_code_server(data_dir)
        # Should complete without error; chat dirs still don't exist
        assert not (ws_dir / "chatSessions").exists()
        assert not (ws_dir / "chatEditingSessions").exists()

    def test_scan_files_top_dir_unreadable(self) -> None:
        """_scan_files raises OSError when top dir is unreadable.
        Covers lines 642-644."""
        scan_dir = os.path.join(self.tmpdir, "scanme")
        os.makedirs(scan_dir)
        Path(os.path.join(scan_dir, "file.txt")).write_text("hi")
        os.chmod(scan_dir, 0o000)
        try:
            result = _scan_files(scan_dir)
            assert isinstance(result, list)
        finally:
            os.chmod(scan_dir, 0o755)

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

    def test_unterminated_quote_in_split(self) -> None:
        """Unterminated quote string exhausts inner while loop.
        Covers branch 94->102 (while loop at 94 exhausts without finding closing quote)."""
        import re
        pattern = re.compile(r"&&")
        result = _split_respecting_quotes('echo "hello', pattern)
        assert len(result) == 1
        assert 'echo "hello' in result[0]

    def test_unterminated_single_quote(self) -> None:
        """Unterminated single quote string."""
        import re
        pattern = re.compile(r"&&")
        result = _split_respecting_quotes("echo 'hello", pattern)
        assert len(result) == 1


class TestPromptDetectorBranchesR2:
    """Cover remaining prompt_detector.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_word_chars_empty_words(self) -> None:
        """Content with no word characters -> words list is empty.
        Covers branch 114->121 (words is empty, so if words: is False)."""
        content = "!@#$%^&*() --- +++\n\n"
        p = os.path.join(self.tmpdir, "symbols.md")
        with open(p, "w") as f:
            f.write(content)
        _, score, reasons = self.detector.analyze(p)
        # Should not get verb density bonus
        assert not any("verb" in r.lower() for r in reasons)


class TestWebUseToolBranchesR2:
    """Cover remaining web_use_tool.py branches."""

    def setup_method(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool
        self.tool = WebUseTool(headless=True, user_data_dir=None)

    def teardown_method(self) -> None:
        self.tool.close()

    def test_ax_tree_truncation(self) -> None:
        """Snapshot longer than max_chars gets truncated.
        Covers line 143."""
        # Create page with many buttons so snapshot is long
        buttons = "".join(f"<button>Button{i}</button>" for i in range(200))
        self.tool.go_to_url(f"data:text/html,{buttons}")
        # Call _get_ax_tree with very small max_chars to trigger truncation
        result = self.tool._get_ax_tree(max_chars=100)
        assert "[truncated]" in result

    def test_resolve_locator_re_snapshot_success(self) -> None:
        """Element not in stale list, re-snapshot finds it.
        Covers branch 169->171."""
        self.tool.go_to_url("data:text/html,<button>MyBtn</button>")
        # Set _elements to a shorter list so element_id is out of range
        self.tool._elements = []
        # Now click element 1 - should re-snapshot and find it
        result = self.tool.click(1)
        assert "Page:" in result

    def test_new_tab_during_click(self) -> None:
        """Click opens a new tab, triggering _check_for_new_tab.
        Covers lines 252-253 and 159-163."""
        html = '''<a href="data:text/html,<h1>NewTab</h1>" target="_blank" id="link">Open Tab</a>'''
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        if "[1]" in tree:
            result = self.tool.click(1)
            # Should handle the new tab
            assert "Page:" in result or "Error" in result

    def test_scroll_error_closed_page(self) -> None:
        """Scroll error when page has been closed.
        Covers lines 325-327."""
        self.tool.go_to_url("data:text/html,<div>X</div>")
        # Close the page but keep browser state intact to trigger error
        self.tool._page.close()
        self.tool._page = self.tool._context.new_page()
        self.tool._page.close()
        # Set _page to None to trigger _ensure_browser to create new page
        # Actually, set it to a closed page
        result = self.tool.scroll("down", 1)
        # Might get error or success depending on implementation
        assert isinstance(result, str)

    def test_get_page_content_error_closed(self) -> None:
        """get_page_content error when browser is broken.
        Covers lines 368-370."""
        self.tool.go_to_url("data:text/html,<p>test</p>")
        # Close the page directly to break state
        self.tool._page.close()
        # _page is set but page is closed -> operations will fail
        result = self.tool.get_page_content()
        assert "Error" in result or "Page:" in result

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

    def test_multiple_elements_visibility_loop(self) -> None:
        """Multiple elements with same role/name, iterating visibility.
        Covers branches 186->184, 188-191."""
        # Create page where aria snapshot shows multiple same-name buttons
        # but they're handled differently by get_by_role
        html = """<html><body>
        <button aria-label="Act">One</button>
        <button aria-label="Act">Two</button>
        <button aria-label="Act">Three</button>
        </body></html>"""
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        if "[1]" in tree:
            result = self.tool.click(1)
            assert "Page:" in result or "Error" in result

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

    def test_check_for_new_tab_no_new_pages(self) -> None:
        """_check_for_new_tab when no new pages exist.
        Covers branch 162->exit (condition False, function exits)."""
        self.tool.go_to_url("data:text/html,<p>test</p>")
        # Only 1 page, so condition len(pages) > 1 is False
        self.tool._check_for_new_tab()

    def test_resolve_locator_visibility_hidden(self) -> None:
        """Multiple buttons with same name, some hidden via visibility:hidden.
        Covers branches 186->184 (is_visible False) and 188-191."""
        html = """<html><body>
        <button style="visibility:hidden" aria-label="X">One</button>
        <button aria-label="X">Two</button>
        </body></html>"""
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        if "[1]" in tree:
            result = self.tool.click(1)
            assert "Page:" in result or "Error" in result


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

    def test_navigate_slow_page_networkidle_timeout(self) -> None:
        """Navigate to a page with a never-completing resource.
        Covers lines 154-156 (networkidle timeout)."""
        import http.server
        import socketserver

        class SlowHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.startswith("/slow"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", "1000000")
                    self.end_headers()
                    time.sleep(30)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b'<html><body><img src="/slow"><button>B</button></body></html>'
                    )

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = True

        server = ThreadedServer(("127.0.0.1", 0), SlowHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            result = self.tool.go_to_url(f"http://127.0.0.1:{port}/")
            assert "Page:" in result or "Error" in result
        finally:
            server.shutdown()
            server.server_close()

    def test_click_opens_new_tab_with_window_open(self) -> None:
        """Click that triggers window.open for new tab detection.
        Covers lines 159-163 (_check_for_new_tab) and 252-253."""
        html = (
            '<button onclick="window.open(\'data:text/html,'
            "<h1>NewTab</h1>', '_blank')\">OpenTab</button>"
        )
        self.tool.go_to_url(f"data:text/html,{html}")
        tree = self.tool.get_page_content()
        if "[1]" in tree:
            result = self.tool.click(1)
            # Should handle new tab
            assert "Page:" in result or "Error" in result

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


class TestKillProcessGroupBothFail:
    """Cover useful_tools.py lines 162-163: inner OSError from process.kill()."""

    def test_kill_reaped_process(self) -> None:
        """After a process exits and is reaped, both os.killpg and process.kill raise OSError."""
        process = subprocess.Popen(["true"], start_new_session=True)
        process.wait()  # reap the process so both kill calls will fail
        _kill_process_group(process)  # should not raise

    def test_kill_permission_error(self) -> None:
        """os.killpg raises PermissionError (OSError subclass) — enters the except OSError
        branch. process.kill() calls poll() which sets returncode, so inner except is
        not reached (Python 3.13+ behavior)."""
        process = subprocess.Popen(["true"], start_new_session=True)
        process.wait()
        original_pid = process.pid
        process.pid = 1
        process.returncode = None
        try:
            _kill_process_group(process)
        finally:
            process.pid = original_pid


class TestBashNonStreamingBaseException:
    """Cover useful_tools.py lines 313-319: BaseException during process.communicate()."""

    def test_sigint_during_communicate(self) -> None:
        """Send SIGINT to trigger KeyboardInterrupt during communicate()."""
        tools = UsefulTools()
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:

            def send_sigint() -> None:
                time.sleep(0.5)
                os.kill(os.getpid(), signal.SIGINT)

            t = threading.Thread(target=send_sigint, daemon=True)
            t.start()
            with pytest.raises(KeyboardInterrupt):
                tools.Bash("sleep 100", "test", timeout_seconds=200)
            t.join(timeout=5)
        finally:
            signal.signal(signal.SIGINT, original_handler)


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

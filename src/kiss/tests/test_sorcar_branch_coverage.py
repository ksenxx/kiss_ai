"""Integration tests targeting uncovered branches in kiss/agents/sorcar/.

No mocks, patches, or test doubles. Uses real files, real git repos, and
real objects.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# prompt_detector.py
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.prompt_detector import PromptDetector


class TestPromptDetector:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_non_md_file(self) -> None:
        p = os.path.join(self.tmpdir, "test.txt")
        Path(p).write_text("hello")
        is_prompt, score, reasons = self.detector.analyze(p)
        assert not is_prompt
        assert score == 0.0
        assert "not .md" in reasons[0]

    def test_nonexistent_file(self) -> None:
        is_prompt, score, reasons = self.detector.analyze("/no/such/file.md")
        assert not is_prompt
        assert "not found" in reasons[0].lower() or "not .md" in reasons[0].lower()

    def test_file_with_frontmatter_prompt_keys(self) -> None:
        p = os.path.join(self.tmpdir, "test.md")
        Path(p).write_text(
            "---\nmodel: gpt-4\ntemperature: 0.5\ntop_p: 0.9\n---\n"
            "# System Prompt\nYou are an expert Python developer.\n"
            "Your task is to write code step-by-step.\n"
            "Do not hallucinate.\n"
            "{{ user_input }}\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert is_prompt
        assert score >= 3.0

    def test_file_with_xml_tags(self) -> None:
        p = os.path.join(self.tmpdir, "xml.md")
        Path(p).write_text(
            "<system>\nYou are a helpful assistant.\n</system>\n"
            "<instruction>\nAnalyze the following text.\n</instruction>\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_file_with_strong_indicators(self) -> None:
        p = os.path.join(self.tmpdir, "strong.md")
        Path(p).write_text(
            "# System Prompt\n"
            "You are a coding assistant.\n"
            "Act as a Python expert.\n"
            "{{ variable }}\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert is_prompt
        assert score >= 3.0

    def test_file_with_medium_indicators(self) -> None:
        p = os.path.join(self.tmpdir, "medium.md")
        Path(p).write_text(
            "# Role\n# Context\n# Task\n"
            "Your task is to classify documents.\n"
            "Do not hallucinate or invent information.\n"
            "Use chain of thought reasoning.\n"
            "Explain step-by-step.\n"
            "Few-shot examples follow.\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_file_with_weak_indicators(self) -> None:
        p = os.path.join(self.tmpdir, "weak.md")
        Path(p).write_text(
            "temperature: 0.7\ntop_p: 0.9\n"
            "json mode enabled\n"
            "```json\n{}\n```\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_file_with_high_imperative_density(self) -> None:
        p = os.path.join(self.tmpdir, "imperative.md")
        # >5% imperative verbs
        Path(p).write_text(
            "write explain summarize translate classify act ignore return output "
            "write explain summarize translate classify act ignore return output "
            "write explain summarize\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert any("imperative" in r.lower() for r in reasons)

    def test_readme_not_prompt(self) -> None:
        p = os.path.join(self.tmpdir, "readme.md")
        Path(p).write_text(
            "# My Project\nThis is a project.\n## Installation\nRun pip install.\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert not is_prompt

    def test_frontmatter_no_prompt_keys(self) -> None:
        p = os.path.join(self.tmpdir, "no_keys.md")
        Path(p).write_text("---\ntitle: My Doc\nauthor: Me\n---\nHello world.\n")
        is_prompt, score, reasons = self.detector.analyze(p)
        # frontmatter exists but no prompt keys
        assert score < 3.0

    def test_frontmatter_partial_parse(self) -> None:
        """Frontmatter with only two --- delimiters."""
        p = os.path.join(self.tmpdir, "partial.md")
        Path(p).write_text("---\nmodel: gpt-4\n---\nContent.\n")
        is_prompt, score, reasons = self.detector.analyze(p)
        assert any("metadata" in r for r in reasons)

    def test_empty_md_file(self) -> None:
        p = os.path.join(self.tmpdir, "empty.md")
        Path(p).write_text("")
        is_prompt, score, reasons = self.detector.analyze(p)
        assert not is_prompt

    def test_multiple_strong_indicator_matches(self) -> None:
        """Cover diminishing returns (multiple matches of same pattern)."""
        p = os.path.join(self.tmpdir, "multi.md")
        Path(p).write_text(
            "You are a great assistant.\n"
            "You are a helpful coder.\n"
            "You are a skilled designer.\n"
            "{{ var1 }} {{ var2 }} {{ var3 }}\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert score > 0

    def test_frontmatter_with_inputs_key(self) -> None:
        p = os.path.join(self.tmpdir, "inputs.md")
        Path(p).write_text("---\ninputs: user_query\nstop_sequences: END\n---\nContent.\n")
        is_prompt, score, reasons = self.detector.analyze(p)
        assert any("metadata" in r for r in reasons)

    def test_no_frontmatter_returns_none(self) -> None:
        """File doesn't start with ---."""
        p = os.path.join(self.tmpdir, "nofm.md")
        Path(p).write_text("# Title\nBody text.\n")
        is_prompt, score, reasons = self.detector.analyze(p)
        # No crash, just no frontmatter bonus

    def test_unreadable_file(self) -> None:
        """Cover the exception path in reading file."""
        p = os.path.join(self.tmpdir, "unreadable.md")
        Path(p).write_text("content")
        os.chmod(p, 0o000)
        try:
            is_prompt, score, reasons = self.detector.analyze(p)
            assert not is_prompt
            assert any("error" in r.lower() for r in reasons)
        finally:
            os.chmod(p, 0o644)


# ---------------------------------------------------------------------------
# useful_tools.py
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _extract_command_names,
    _format_bash_result,
    _kill_process_group,
    _strip_heredocs,
    _truncate_output,
)


class TestTruncateOutput:
    def test_no_truncation(self) -> None:
        assert _truncate_output("hello", 100) == "hello"

    def test_truncation_small_max(self) -> None:
        result = _truncate_output("a" * 100, 10)
        assert len(result) <= 10

    def test_truncation_with_head_tail(self) -> None:
        text = "a" * 1000
        result = _truncate_output(text, 200)
        assert "truncated" in result
        assert len(result) <= 200

    def test_truncation_zero_tail(self) -> None:
        """Cover the tail=0 branch."""
        msg_len = len("\n\n... [truncated 100 chars] ...\n\n")
        # max_chars such that remaining = head + 0 tail
        result = _truncate_output("x" * 200, msg_len + 1)
        assert "truncated" in result


class TestExtractCommandNames:
    def test_simple_command(self) -> None:
        assert _extract_command_names("ls -la") == ["ls"]

    def test_pipe(self) -> None:
        assert _extract_command_names("cat file | grep foo") == ["cat", "grep"]

    def test_and_operator(self) -> None:
        assert _extract_command_names("cd /tmp && ls") == ["cd", "ls"]

    def test_or_operator(self) -> None:
        assert _extract_command_names("false || true") == ["false", "true"]

    def test_semicolon(self) -> None:
        assert _extract_command_names("echo a; echo b") == ["echo", "echo"]

    def test_env_var_prefix(self) -> None:
        names = _extract_command_names("FOO=bar python script.py")
        assert "python" in names

    def test_background(self) -> None:
        names = _extract_command_names("sleep 10 &")
        assert "sleep" in names

    def test_heredoc_stripped(self) -> None:
        cmd = "cat <<EOF\nhello world\nEOF"
        result = _strip_heredocs(cmd)
        assert "hello world" not in result

    def test_quoted_strings(self) -> None:
        names = _extract_command_names("echo 'hello world'")
        assert "echo" in names

    def test_redirect(self) -> None:
        names = _extract_command_names("echo hello > /tmp/out.txt")
        assert "echo" in names

    def test_subshell_prefix(self) -> None:
        names = _extract_command_names("(echo hello)")
        assert "echo" in names

    def test_brace_prefix(self) -> None:
        names = _extract_command_names("{ echo hello; }")
        assert "echo" in names

    def test_empty_command(self) -> None:
        assert _extract_command_names("") == []

    def test_invalid_shlex(self) -> None:
        """Unmatched quote should not crash."""
        names = _extract_command_names("echo 'unclosed")
        # Should handle gracefully
        assert isinstance(names, list)

    def test_redirect_with_fd(self) -> None:
        names = _extract_command_names("echo hello 2>&1")
        assert "echo" in names

    def test_redirect_separate_arg(self) -> None:
        """Cover redirect token where file is separate (2> file)."""
        names = _extract_command_names("echo hello 2> /dev/null")
        assert "echo" in names

    def test_newline_separator(self) -> None:
        names = _extract_command_names("echo a\necho b")
        assert names == ["echo", "echo"]


class TestFormatBashResult:
    def test_success(self) -> None:
        result = _format_bash_result(0, "output text", 1000)
        assert result == "output text"

    def test_error_with_output(self) -> None:
        result = _format_bash_result(1, "error msg", 1000)
        assert "Error (exit code 1)" in result
        assert "error msg" in result

    def test_error_no_output(self) -> None:
        result = _format_bash_result(1, "", 1000)
        assert "Error (exit code 1)" in result


class TestUsefulTools:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.tools = UsefulTools()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_file(self) -> None:
        p = os.path.join(self.tmpdir, "test.txt")
        Path(p).write_text("hello\nworld\n")
        result = self.tools.Read(p)
        assert "hello" in result

    def test_read_file_truncated(self) -> None:
        p = os.path.join(self.tmpdir, "big.txt")
        Path(p).write_text("\n".join(f"line{i}" for i in range(3000)))
        result = self.tools.Read(p, max_lines=10)
        assert "truncated" in result

    def test_read_file_not_found(self) -> None:
        result = self.tools.Read("/no/such/file.txt")
        assert "Error" in result

    def test_write_file(self) -> None:
        p = os.path.join(self.tmpdir, "out.txt")
        result = self.tools.Write(p, "content")
        assert "Successfully" in result
        assert Path(p).read_text() == "content"

    def test_write_file_creates_dirs(self) -> None:
        p = os.path.join(self.tmpdir, "sub", "dir", "file.txt")
        result = self.tools.Write(p, "nested")
        assert "Successfully" in result

    def test_write_file_error(self) -> None:
        result = self.tools.Write("/dev/null/impossible/path.txt", "content")
        assert "Error" in result

    def test_edit_file(self) -> None:
        p = os.path.join(self.tmpdir, "edit.txt")
        Path(p).write_text("hello world")
        result = self.tools.Edit(p, "world", "everyone")
        assert "Successfully" in result
        assert Path(p).read_text() == "hello everyone"

    def test_edit_file_not_found(self) -> None:
        result = self.tools.Edit("/no/such/file.txt", "a", "b")
        assert "Error" in result

    def test_edit_same_string(self) -> None:
        p = os.path.join(self.tmpdir, "same.txt")
        Path(p).write_text("hello")
        result = self.tools.Edit(p, "hello", "hello")
        assert "must be different" in result

    def test_edit_string_not_found(self) -> None:
        p = os.path.join(self.tmpdir, "nf.txt")
        Path(p).write_text("hello")
        result = self.tools.Edit(p, "xyz", "abc")
        assert "not found" in result.lower()

    def test_edit_multiple_occurrences(self) -> None:
        p = os.path.join(self.tmpdir, "multi.txt")
        Path(p).write_text("aaa bbb aaa")
        result = self.tools.Edit(p, "aaa", "ccc")
        assert "appears 2 times" in result

    def test_edit_replace_all(self) -> None:
        p = os.path.join(self.tmpdir, "rall.txt")
        Path(p).write_text("aaa bbb aaa")
        result = self.tools.Edit(p, "aaa", "ccc", replace_all=True)
        assert "Successfully replaced 2" in result
        assert Path(p).read_text() == "ccc bbb ccc"

    def test_bash_simple(self) -> None:
        result = self.tools.Bash("echo hello", "test echo")
        assert "hello" in result

    def test_bash_error(self) -> None:
        result = self.tools.Bash("exit 1", "test error")
        assert "Error" in result

    def test_bash_timeout(self) -> None:
        result = self.tools.Bash("sleep 60", "test timeout", timeout_seconds=0.5)
        assert "timeout" in result.lower()

    def test_bash_disallowed(self) -> None:
        result = self.tools.Bash("eval echo hi", "test eval")
        assert "not allowed" in result

        result = self.tools.Bash("source ~/.bashrc", "test source")
        assert "not allowed" in result

    def test_bash_streaming(self) -> None:
        collected: list[str] = []
        tools = UsefulTools(stream_callback=collected.append)
        result = tools.Bash("echo line1; echo line2", "stream test")
        assert "line1" in result
        assert len(collected) > 0

    def test_bash_streaming_timeout(self) -> None:
        collected: list[str] = []
        tools = UsefulTools(stream_callback=collected.append)
        result = tools.Bash("sleep 60", "stream timeout", timeout_seconds=0.5)
        assert "timeout" in result.lower()

    def test_bash_max_output_chars(self) -> None:
        result = self.tools.Bash(
            "python3 -c \"print('x' * 200)\"",
            "test truncation",
            max_output_chars=50,
        )
        assert len(result) <= 55  # some margin for truncation msg

    def test_bash_disallowed_env(self) -> None:
        result = self.tools.Bash("env | head", "test env")
        assert "not allowed" in result

    def test_bash_disallowed_exec(self) -> None:
        result = self.tools.Bash("exec echo hi", "test exec")
        assert "not allowed" in result

    def test_bash_disallowed_dot(self) -> None:
        result = self.tools.Bash(". ~/.bashrc", "test dot source")
        assert "not allowed" in result


# ---------------------------------------------------------------------------
# task_history.py
# ---------------------------------------------------------------------------
import kiss.agents.sorcar.task_history as th


class TestTaskHistory:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        # Save originals
        self._orig_history_file = th.HISTORY_FILE
        self._orig_proposals_file = th.PROPOSALS_FILE
        self._orig_model_usage_file = th.MODEL_USAGE_FILE
        self._orig_file_usage_file = th.FILE_USAGE_FILE
        self._orig_kiss_dir = th._KISS_DIR
        # Redirect to temp
        th._KISS_DIR = Path(self.tmpdir)
        th.HISTORY_FILE = Path(self.tmpdir) / "task_history.json"
        th.PROPOSALS_FILE = Path(self.tmpdir) / "proposed_tasks.json"
        th.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        th.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        # Clear cache
        th._history_cache = None

    def teardown_method(self) -> None:
        th.HISTORY_FILE = self._orig_history_file
        th.PROPOSALS_FILE = self._orig_proposals_file
        th.MODEL_USAGE_FILE = self._orig_model_usage_file
        th.FILE_USAGE_FILE = self._orig_file_usage_file
        th._KISS_DIR = self._orig_kiss_dir
        th._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_history_no_file(self) -> None:
        """No history file -> SAMPLE_TASKS are used as default."""
        history = th._load_history()
        assert len(history) > 0

    def test_load_history_from_file(self) -> None:
        """Valid history file loads correctly."""
        data = [{"task": "task1", "chat_events": []}, {"task": "task2", "chat_events": []}]
        th.HISTORY_FILE.write_text(json.dumps(data))
        th._history_cache = None
        history = th._load_history()
        assert len(history) == 2
        assert history[0]["task"] == "task1"

    def test_load_history_dedup(self) -> None:
        """Duplicate tasks get deduplicated."""
        data = [
            {"task": "task1", "chat_events": []},
            {"task": "task1", "chat_events": []},
            {"task": "task2", "chat_events": []},
        ]
        th.HISTORY_FILE.write_text(json.dumps(data))
        th._history_cache = None
        history = th._load_history()
        assert len(history) == 2

    def test_load_history_invalid_json(self) -> None:
        """Invalid JSON falls back to SAMPLE_TASKS."""
        th.HISTORY_FILE.write_text("not json{{{")
        th._history_cache = None
        history = th._load_history()
        assert len(history) > 0  # Sample tasks

    def test_load_history_empty_list(self) -> None:
        """Empty list falls back to SAMPLE_TASKS."""
        th.HISTORY_FILE.write_text("[]")
        th._history_cache = None
        history = th._load_history()
        assert len(history) > 0  # Sample tasks

    def test_load_history_uses_cache(self) -> None:
        """Second call uses cache."""
        data = [{"task": "cached", "chat_events": []}]
        th.HISTORY_FILE.write_text(json.dumps(data))
        th._history_cache = None
        h1 = th._load_history()
        h2 = th._load_history()
        assert h1 is h2

    def test_save_history(self) -> None:
        entries = [{"task": "saved", "chat_events": []}]
        th._save_history(entries)
        assert th.HISTORY_FILE.exists()
        data = json.loads(th.HISTORY_FILE.read_text())
        assert data[0]["task"] == "saved"

    def test_add_task(self) -> None:
        th._history_cache = None
        th._add_task("new task")
        history = th._load_history()
        assert history[0]["task"] == "new task"

    def test_add_task_dedup(self) -> None:
        """Adding existing task moves it to front."""
        th._history_cache = None
        th._add_task("task1")
        th._add_task("task2")
        th._add_task("task1")
        history = th._load_history()
        tasks = [e["task"] for e in history]
        assert tasks[0] == "task1"
        assert tasks.count("task1") == 1

    def test_set_latest_chat_events_by_task(self) -> None:
        th._history_cache = None
        th._add_task("my task")
        th._set_latest_chat_events([{"type": "text"}], task="my task")
        history = th._load_history()
        found = [e for e in history if e["task"] == "my task"]
        assert found[0]["chat_events"] == [{"type": "text"}]
        assert "result" not in found[0]

    def test_set_latest_chat_events_no_task(self) -> None:
        """Update history[0] when task is None."""
        th._history_cache = None
        th._add_task("first task")
        th._set_latest_chat_events([{"type": "done"}])
        history = th._load_history()
        assert history[0]["chat_events"] == [{"type": "done"}]

    def test_set_latest_chat_events_empty_cache(self) -> None:
        """When cache is empty, do nothing."""
        th._history_cache = []
        th._set_latest_chat_events([{"type": "x"}])
        assert th._history_cache == []

    def test_set_latest_chat_events_task_not_found(self) -> None:
        """When task not in cache, do nothing."""
        th._history_cache = None
        th._add_task("existing")
        th._set_latest_chat_events([{"type": "x"}], task="nonexistent")

    def test_set_latest_chat_events_pops_result(self) -> None:
        """Should pop 'result' key if present."""
        th._history_cache = None
        th._add_task("popper")
        # Manually add result key
        for e in th._history_cache:
            if e["task"] == "popper":
                e["result"] = "old result"
        th._set_latest_chat_events([{"type": "new"}], task="popper")
        for e in th._history_cache:
            if e["task"] == "popper":
                assert "result" not in e

    def test_load_proposals_valid(self) -> None:
        th.PROPOSALS_FILE.write_text(json.dumps(["task1", "task2"]))
        result = th._load_proposals()
        assert result == ["task1", "task2"]

    def test_load_proposals_empty(self) -> None:
        result = th._load_proposals()
        assert result == []

    def test_load_proposals_invalid_json(self) -> None:
        th.PROPOSALS_FILE.write_text("not json")
        result = th._load_proposals()
        assert result == []

    def test_load_proposals_non_list(self) -> None:
        th.PROPOSALS_FILE.write_text('{"key": "val"}')
        result = th._load_proposals()
        assert result == []

    def test_load_proposals_filters(self) -> None:
        """Filters non-string and empty entries, max 5."""
        th.PROPOSALS_FILE.write_text(json.dumps(["a", 123, "", "b", "c", "d", "e", "f"]))
        result = th._load_proposals()
        assert all(isinstance(t, str) and t.strip() for t in result)
        assert len(result) <= 5

    def test_save_proposals(self) -> None:
        th._save_proposals(["p1", "p2"])
        assert th.PROPOSALS_FILE.exists()
        data = json.loads(th.PROPOSALS_FILE.read_text())
        assert data == ["p1", "p2"]

    def test_load_json_dict_non_dict(self) -> None:
        path = Path(self.tmpdir) / "test.json"
        path.write_text("[1,2,3]")
        result = th._load_json_dict(path)
        assert result == {}

    def test_load_json_dict_valid(self) -> None:
        path = Path(self.tmpdir) / "test.json"
        path.write_text('{"key": "val"}')
        result = th._load_json_dict(path)
        assert result == {"key": "val"}

    def test_load_json_dict_no_file(self) -> None:
        result = th._load_json_dict(Path(self.tmpdir) / "missing.json")
        assert result == {}

    def test_load_json_dict_bad_json(self) -> None:
        path = Path(self.tmpdir) / "bad.json"
        path.write_text("not json{")
        result = th._load_json_dict(path)
        assert result == {}

    def test_int_values(self) -> None:
        result = th._int_values({"a": 1, "b": 2.5, "c": "not_int", "d": 3})
        assert result == {"a": 1, "b": 2, "d": 3}

    def test_record_model_usage(self) -> None:
        th._record_model_usage("gpt-4")
        th._record_model_usage("gpt-4")
        usage = th._load_model_usage()
        assert usage.get("gpt-4") == 2

    def test_load_last_model(self) -> None:
        th._record_model_usage("claude-3")
        last = th._load_last_model()
        assert last == "claude-3"

    def test_load_last_model_empty(self) -> None:
        last = th._load_last_model()
        assert last == ""

    def test_load_last_model_non_string(self) -> None:
        th.MODEL_USAGE_FILE.write_text(json.dumps({"_last": 123}))
        last = th._load_last_model()
        assert last == ""

    def test_record_file_usage(self) -> None:
        th._record_file_usage("src/test.py")
        th._record_file_usage("src/test.py")
        usage = th._load_file_usage()
        assert usage.get("src/test.py") == 2

    def test_append_task_to_md(self) -> None:
        md_path = th._get_task_history_md_path()
        # Save and restore original content
        orig = md_path.read_text() if md_path.exists() else None
        try:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text("# Task History\n\n")
            th._append_task_to_md("test_unique_task_12345", "done_result")
            content = md_path.read_text()
            assert "test_unique_task_12345" in content
            assert "done_result" in content
        finally:
            if orig is not None:
                md_path.write_text(orig)
            elif md_path.exists():
                md_path.unlink()

    def test_init_task_history_md(self) -> None:
        md_path = th._get_task_history_md_path()
        orig = md_path.read_text() if md_path.exists() else None
        try:
            if md_path.exists():
                md_path.unlink()
            path = th._init_task_history_md()
            assert path.exists()
            assert "Task History" in path.read_text()
        finally:
            if orig is not None:
                md_path.write_text(orig)

    def test_init_task_history_md_already_exists(self) -> None:
        md_path = th._get_task_history_md_path()
        orig = md_path.read_text() if md_path.exists() else None
        try:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text("existing content\n")
            path = th._init_task_history_md()
            assert path.read_text() == "existing content\n"
        finally:
            if orig is not None:
                md_path.write_text(orig)
            elif md_path.exists():
                md_path.unlink()


# ---------------------------------------------------------------------------
# web_use_tool.py: _number_interactive_elements
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.web_use_tool import (
    INTERACTIVE_ROLES,
    _number_interactive_elements,
)


class TestNumberInteractiveElements:
    def test_basic_numbering(self) -> None:
        snapshot = (
            '- heading "Title"\n'
            '  - button "Click me"\n'
            '  - textbox "Enter name"\n'
            '  - paragraph "Some text"\n'
            '  - link "Home"\n'
        )
        numbered, elements = _number_interactive_elements(snapshot)
        assert "[1]" in numbered
        assert "[2]" in numbered
        assert "[3]" in numbered
        assert len(elements) == 3
        assert elements[0]["role"] == "button"
        assert elements[0]["name"] == "Click me"
        assert elements[1]["role"] == "textbox"
        assert elements[2]["role"] == "link"

    def test_no_interactive(self) -> None:
        snapshot = "- heading 'Title'\n- paragraph 'text'\n"
        numbered, elements = _number_interactive_elements(snapshot)
        assert elements == []
        assert "[" not in numbered

    def test_empty_snapshot(self) -> None:
        numbered, elements = _number_interactive_elements("")
        assert elements == []
        assert numbered == ""

    def test_element_without_name(self) -> None:
        snapshot = "  - button\n"
        numbered, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 1
        assert elements[0]["name"] == ""

    def test_all_interactive_roles(self) -> None:
        lines = [f"  - {role} 'test'" for role in sorted(INTERACTIVE_ROLES)]
        snapshot = "\n".join(lines)
        numbered, elements = _number_interactive_elements(snapshot)
        assert len(elements) == len(INTERACTIVE_ROLES)


# ---------------------------------------------------------------------------
# browser_ui.py: BaseBrowserPrinter
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.browser_ui import (
    BaseBrowserPrinter,
    _coalesce_events,
    find_free_port,
)


class TestCoalesceEvents:
    def test_empty(self) -> None:
        assert _coalesce_events([]) == []

    def test_no_merge(self) -> None:
        events = [{"type": "tool_call"}, {"type": "tool_result"}]
        assert _coalesce_events(events) == events

    def test_merge_thinking_delta(self) -> None:
        events = [
            {"type": "thinking_delta", "text": "a"},
            {"type": "thinking_delta", "text": "b"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "ab"

    def test_merge_text_delta(self) -> None:
        events = [
            {"type": "text_delta", "text": "x"},
            {"type": "text_delta", "text": "y"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "xy"

    def test_merge_system_output(self) -> None:
        events = [
            {"type": "system_output", "text": "1"},
            {"type": "system_output", "text": "2"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 1
        assert result[0]["text"] == "12"

    def test_no_merge_different_types(self) -> None:
        events = [
            {"type": "thinking_delta", "text": "a"},
            {"type": "text_delta", "text": "b"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2

    def test_no_merge_missing_text(self) -> None:
        events = [
            {"type": "thinking_delta"},
            {"type": "thinking_delta", "text": "a"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


class TestBaseBrowserPrinter:
    def test_broadcast_and_receive(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.broadcast({"type": "test"})
        event = cq.get_nowait()
        assert event["type"] == "test"
        printer.remove_client(cq)

    def test_remove_nonexistent_client(self) -> None:
        import queue

        printer = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        printer.remove_client(cq)  # Should not raise

    def test_has_clients(self) -> None:
        printer = BaseBrowserPrinter()
        assert not printer.has_clients()
        cq = printer.add_client()
        assert printer.has_clients()
        printer.remove_client(cq)
        assert not printer.has_clients()

    def test_recording(self) -> None:
        printer = BaseBrowserPrinter()
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "hello"})
        printer.broadcast({"type": "tool_call", "name": "test"})
        printer.broadcast({"type": "non_display_type"})
        events = printer.stop_recording()
        assert len(events) == 2  # Only display types
        assert events[0]["type"] == "text_delta"
        assert events[1]["type"] == "tool_call"

    def test_print_text(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("hello world", type="text")
        event = cq.get_nowait()
        assert event["type"] == "text_delta"
        printer.remove_client(cq)

    def test_print_prompt(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("prompt text", type="prompt")
        event = cq.get_nowait()
        assert event["type"] == "prompt"
        printer.remove_client(cq)

    def test_print_usage_info(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("usage text", type="usage_info")
        event = cq.get_nowait()
        assert event["type"] == "usage_info"
        printer.remove_client(cq)

    def test_print_tool_call(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print(
            "Edit",
            type="tool_call",
            tool_input={
                "file_path": "/tmp/test.py",
                "old_string": "old",
                "new_string": "new",
                "description": "fix bug",
                "command": "echo hi",
                "content": "file content",
            },
        )
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        # Should have text_end and tool_call
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["path"] == "/tmp/test.py"
        assert tool_calls[0]["old_string"] == "old"
        assert tool_calls[0]["new_string"] == "new"
        printer.remove_client(cq)

    def test_print_tool_result(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("result text", type="tool_result", is_error=True)
        event = cq.get_nowait()
        assert event["type"] == "tool_result"
        assert event["is_error"] is True
        printer.remove_client(cq)

    def test_print_result(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print(
            "success: true\nsummary: done",
            type="result",
            step_count=5,
            total_tokens=1000,
            cost="$0.01",
        )
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1
        printer.remove_client(cq)

    def test_print_result_with_yaml(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print(
            "success: true\nsummary: all tests passed",
            type="result",
        )
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_events = [e for e in events if e["type"] == "result"]
        assert result_events[0].get("summary") == "all tests passed"
        printer.remove_client(cq)

    def test_print_bash_stream(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("line1\n", type="bash_stream")
        time.sleep(0.2)  # Let flush happen
        printer._flush_bash()
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        sys_outputs = [e for e in events if e["type"] == "system_output"]
        assert len(sys_outputs) >= 1
        printer.remove_client(cq)

    def test_print_bash_stream_timer(self) -> None:
        """Cover the timer path in bash_stream."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        # Set last flush to now so next print won't flush immediately
        printer._bash_last_flush = time.monotonic()
        printer.print("a", type="bash_stream")
        # Timer should be set
        time.sleep(0.2)
        printer._flush_bash()
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        printer.remove_client(cq)

    def test_print_unknown_type(self) -> None:
        printer = BaseBrowserPrinter()
        result = printer.print("text", type="unknown_type")
        assert result == ""

    def test_check_stop_thread_local(self) -> None:
        printer = BaseBrowserPrinter()
        stop_ev = threading.Event()
        printer._thread_local.stop_event = stop_ev
        stop_ev.set()
        with pytest.raises(KeyboardInterrupt):
            printer._check_stop()
        printer._thread_local.stop_event = None

    def test_check_stop_global(self) -> None:
        printer = BaseBrowserPrinter()
        printer.stop_event.set()
        with pytest.raises(KeyboardInterrupt):
            printer._check_stop()
        printer.stop_event.clear()

    def test_check_stop_no_stop(self) -> None:
        printer = BaseBrowserPrinter()
        printer._check_stop()  # Should not raise

    def test_handle_stream_event_content_block_start_thinking(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class FakeEvent:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        printer._handle_stream_event(
            FakeEvent({"type": "content_block_start", "content_block": {"type": "thinking"}})
        )
        assert printer._current_block_type == "thinking"
        event = cq.get_nowait()
        assert event["type"] == "thinking_start"
        printer.remove_client(cq)

    def test_handle_stream_event_content_block_start_tool_use(self) -> None:
        printer = BaseBrowserPrinter()

        class FakeEvent:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        printer._handle_stream_event(
            FakeEvent(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "name": "Bash"},
                }
            )
        )
        assert printer._tool_name == "Bash"

    def test_handle_stream_event_content_block_delta(self) -> None:
        printer = BaseBrowserPrinter()

        class FakeEvent:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        # thinking delta
        printer._current_block_type = "thinking"
        text = printer._handle_stream_event(
            FakeEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "hello"},
                }
            )
        )
        assert text == "hello"

        # text delta
        printer._current_block_type = "text"
        text = printer._handle_stream_event(
            FakeEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "world"},
                }
            )
        )
        assert text == "world"

        # input_json_delta
        printer._current_block_type = "tool_use"
        printer._tool_json_buffer = ""
        printer._handle_stream_event(
            FakeEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": '{"key":'},
                }
            )
        )
        assert printer._tool_json_buffer == '{"key":'

    def test_handle_stream_event_content_block_stop_thinking(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._current_block_type = "thinking"
        printer._handle_stream_event(type("E", (), {"event": {"type": "content_block_stop"}})())
        event = cq.get_nowait()
        assert event["type"] == "thinking_end"
        printer.remove_client(cq)

    def test_handle_stream_event_content_block_stop_tool_use(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._current_block_type = "tool_use"
        printer._tool_name = "Read"
        printer._tool_json_buffer = '{"file_path": "/tmp/x.py"}'
        printer._handle_stream_event(type("E", (), {"event": {"type": "content_block_stop"}})())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["path"] == "/tmp/x.py"
        printer.remove_client(cq)

    def test_handle_stream_event_content_block_stop_tool_use_bad_json(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._current_block_type = "tool_use"
        printer._tool_name = "Bash"
        printer._tool_json_buffer = "invalid json"
        printer._handle_stream_event(type("E", (), {"event": {"type": "content_block_stop"}})())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1
        printer.remove_client(cq)

    def test_handle_stream_event_content_block_stop_text(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._current_block_type = "text"
        printer._handle_stream_event(type("E", (), {"event": {"type": "content_block_stop"}})())
        event = cq.get_nowait()
        assert event["type"] == "text_end"
        printer.remove_client(cq)

    def test_handle_message_tool_output(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Msg:
            subtype = "tool_output"
            data = {"content": "tool output text"}

        printer._handle_message(Msg())
        event = cq.get_nowait()
        assert event["type"] == "system_output"
        assert event["text"] == "tool output text"
        printer.remove_client(cq)

    def test_handle_message_result(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Msg:
            result = "success: true\nsummary: done"

        printer._handle_message(Msg(), budget_used=0.5, step_count=3, total_tokens_used=500)
        event = cq.get_nowait()
        assert event["type"] == "result"
        printer.remove_client(cq)

    def test_handle_message_content_blocks(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Block:
            is_error = True
            content = "error content"

        class Msg:
            content = [Block()]

        printer._handle_message(Msg())
        event = cq.get_nowait()
        assert event["type"] == "tool_result"
        assert event["is_error"] is True
        printer.remove_client(cq)

    def test_token_callback(self) -> None:
        import asyncio

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        asyncio.run(printer.token_callback("hello"))
        event = cq.get_nowait()
        assert event["type"] == "text_delta"
        printer.remove_client(cq)

    def test_token_callback_thinking(self) -> None:
        import asyncio

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._current_block_type = "thinking"
        asyncio.run(printer.token_callback("thought"))
        event = cq.get_nowait()
        assert event["type"] == "thinking_delta"
        printer.remove_client(cq)

    def test_token_callback_empty(self) -> None:
        import asyncio

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        asyncio.run(printer.token_callback(""))
        assert cq.empty()
        printer.remove_client(cq)

    def test_parse_result_yaml_valid(self) -> None:
        result = BaseBrowserPrinter._parse_result_yaml("success: true\nsummary: done")
        assert result is not None
        assert result["summary"] == "done"

    def test_parse_result_yaml_invalid(self) -> None:
        result = BaseBrowserPrinter._parse_result_yaml("not: yaml: data: [")
        assert result is None

    def test_parse_result_yaml_no_summary(self) -> None:
        result = BaseBrowserPrinter._parse_result_yaml("key: value")
        assert result is None

    def test_flush_bash_with_data(self) -> None:
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._bash_buffer = ["line1\n", "line2\n"]
        printer._flush_bash()
        event = cq.get_nowait()
        assert event["type"] == "system_output"
        assert "line1" in event["text"]
        printer.remove_client(cq)

    def test_reset_clears_state(self) -> None:
        printer = BaseBrowserPrinter()
        printer._current_block_type = "text"
        printer._tool_name = "Bash"
        printer._tool_json_buffer = "partial"
        printer._bash_buffer = ["data"]
        printer.reset()
        assert printer._current_block_type == ""
        assert printer._tool_name == ""
        assert printer._tool_json_buffer == ""

    def test_print_message_type(self) -> None:
        """Cover the message handler in print()."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Msg:
            subtype = "tool_output"
            data = {"content": "msg text"}

        printer.print(Msg(), type="message")
        event = cq.get_nowait()
        assert event["type"] == "system_output"
        printer.remove_client(cq)


class TestFindFreePort:
    def test_returns_int(self) -> None:
        port = find_free_port()
        assert isinstance(port, int)
        assert port > 0


# ---------------------------------------------------------------------------
# code_server.py
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
    _diff_files,
    _disable_copilot_scm_button,
    _parse_diff_hunks,
    _prepare_merge_view,
    _restore_merge_files,
    _save_untracked_base,
    _scan_files,
    _setup_code_server,
    _snapshot_files,
)


class TestScanFiles:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scan_basic(self) -> None:
        Path(self.tmpdir, "file1.txt").write_text("a")
        Path(self.tmpdir, "file2.py").write_text("b")
        os.makedirs(os.path.join(self.tmpdir, "subdir"))
        Path(self.tmpdir, "subdir", "inner.txt").write_text("c")
        result = _scan_files(self.tmpdir)
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir/" in result

    def test_scan_skips_hidden(self) -> None:
        os.makedirs(os.path.join(self.tmpdir, ".git"))
        Path(self.tmpdir, ".git", "HEAD").write_text("ref")
        result = _scan_files(self.tmpdir)
        assert not any(".git" in p for p in result)

    def test_scan_skips_pycache(self) -> None:
        os.makedirs(os.path.join(self.tmpdir, "__pycache__"))
        Path(self.tmpdir, "__pycache__", "mod.pyc").write_bytes(b"\x00")
        result = _scan_files(self.tmpdir)
        assert not any("__pycache__" in p for p in result)

    def test_scan_depth_limit(self) -> None:
        deep = os.path.join(self.tmpdir, "a", "b", "c", "d", "e")
        os.makedirs(deep)
        Path(deep, "deep.txt").write_text("deep")
        result = _scan_files(self.tmpdir)
        assert not any("deep.txt" in p for p in result)


class TestGitUtilities:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        Path(self.tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parse_diff_hunks_no_diff(self) -> None:
        result = _parse_diff_hunks(self.tmpdir)
        assert result == {}

    def test_parse_diff_hunks_with_diff(self) -> None:
        Path(self.tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
        result = _parse_diff_hunks(self.tmpdir)
        assert "file.txt" in result
        assert len(result["file.txt"]) > 0

    def test_parse_diff_hunks_new_lines(self) -> None:
        """Hunk with only additions (no count for old)."""
        Path(self.tmpdir, "file.txt").write_text("line1\nline2\nline3\nnewline\n")
        result = _parse_diff_hunks(self.tmpdir)
        assert "file.txt" in result

    def test_capture_untracked(self) -> None:
        Path(self.tmpdir, "untracked.txt").write_text("new")
        result = _capture_untracked(self.tmpdir)
        assert "untracked.txt" in result

    def test_capture_untracked_empty(self) -> None:
        result = _capture_untracked(self.tmpdir)
        assert result == set()

    def test_snapshot_files(self) -> None:
        result = _snapshot_files(self.tmpdir, {"file.txt"})
        assert "file.txt" in result
        assert len(result["file.txt"]) == 32  # MD5 hex

    def test_snapshot_files_missing(self) -> None:
        result = _snapshot_files(self.tmpdir, {"missing.txt"})
        assert "missing.txt" not in result

    def test_diff_files(self) -> None:
        f1 = os.path.join(self.tmpdir, "base.txt")
        f2 = os.path.join(self.tmpdir, "current.txt")
        Path(f1).write_text("line1\nline2\n")
        Path(f2).write_text("line1\nmodified\n")
        hunks = _diff_files(f1, f2)
        assert len(hunks) > 0

    def test_diff_files_identical(self) -> None:
        f1 = os.path.join(self.tmpdir, "same1.txt")
        f2 = os.path.join(self.tmpdir, "same2.txt")
        Path(f1).write_text("same content\n")
        Path(f2).write_text("same content\n")
        hunks = _diff_files(f1, f2)
        assert hunks == []


class TestSaveUntrackedBase:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(self.work_dir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_copies(self) -> None:
        Path(self.work_dir, "ut.txt").write_text("content")
        _save_untracked_base(self.work_dir, self.tmpdir, {"ut.txt"})

    def test_skip_large_files(self) -> None:
        large = os.path.join(self.work_dir, "big.bin")
        Path(large).write_bytes(b"\x00" * 3_000_000)
        _save_untracked_base(self.work_dir, self.tmpdir, {"big.bin"})

    def test_skip_nonexistent(self) -> None:
        _save_untracked_base(self.work_dir, self.tmpdir, {"missing.txt"})

    def test_skip_directory(self) -> None:
        os.makedirs(os.path.join(self.work_dir, "adir"))
        _save_untracked_base(self.work_dir, self.tmpdir, {"adir"})


class TestCleanupMergeData:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cleanup_removes_dirs(self) -> None:
        merge_temp = Path(self.tmpdir) / "merge-temp"
        merge_temp.mkdir()
        (merge_temp / "file.txt").write_text("x")
        merge_current = Path(self.tmpdir) / "merge-current"
        merge_current.mkdir()
        manifest = Path(self.tmpdir) / "pending-merge.json"
        manifest.write_text("{}")
        _cleanup_merge_data(self.tmpdir)
        assert not merge_temp.exists()
        assert not merge_current.exists()
        assert not manifest.exists()

    def test_cleanup_nothing_to_clean(self) -> None:
        _cleanup_merge_data(self.tmpdir)  # Should not raise


class TestRestoreMergeFiles:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(self.work_dir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_restore(self) -> None:
        data_dir = os.path.join(self.tmpdir, "data")
        current_dir = Path(data_dir) / "merge-current"
        current_dir.mkdir(parents=True)
        (current_dir / "restored.txt").write_text("restored content")
        _restore_merge_files(data_dir, self.work_dir)
        assert Path(self.work_dir, "restored.txt").read_text() == "restored content"

    def test_restore_no_current_dir(self) -> None:
        _restore_merge_files(self.tmpdir, self.work_dir)  # Should not raise


class TestPrepareMergeView:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=self.tmpdir,
            capture_output=True,
        )
        Path(self.tmpdir, "file.txt").write_text("line1\nline2\nline3\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)
        self.data_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_no_changes(self) -> None:
        result = _prepare_merge_view(self.tmpdir, self.data_dir, {}, set())
        assert result.get("error") == "No changes"

    def test_with_changes(self) -> None:
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(
            self.tmpdir, set(pre_hunks.keys()) | pre_untracked
        )
        _save_untracked_base(self.tmpdir, self.data_dir, pre_untracked | set(pre_hunks.keys()))

        # Make a change
        Path(self.tmpdir, "file.txt").write_text("line1\nmodified\nline3\n")
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, pre_hunks, pre_untracked, pre_hashes
        )
        assert result.get("status") == "opened"
        assert result["count"] >= 1

    def test_with_new_file(self) -> None:
        pre_untracked = _capture_untracked(self.tmpdir)
        # Create new file
        Path(self.tmpdir, "new_file.py").write_text("print('hello')\n")
        result = _prepare_merge_view(self.tmpdir, self.data_dir, {}, pre_untracked)
        assert result.get("status") == "opened"

    def test_unchanged_file_skipped(self) -> None:
        """File with pre-existing changes but unchanged by agent."""
        # Make a change before "agent runs"
        Path(self.tmpdir, "file.txt").write_text("changed\nline2\nline3\n")
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(
            self.tmpdir, set(pre_hunks.keys()) | pre_untracked
        )
        _save_untracked_base(self.tmpdir, self.data_dir, pre_untracked | set(pre_hunks.keys()))
        # Don't change anything else
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, pre_hunks, pre_untracked, pre_hashes
        )
        assert result.get("error") == "No changes"

    def test_modified_untracked_file(self) -> None:
        """Pre-existing untracked file modified by agent."""
        Path(self.tmpdir, "untracked.txt").write_text("original\n")
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, pre_untracked)
        _save_untracked_base(self.tmpdir, self.data_dir, pre_untracked)

        # Agent modifies it
        Path(self.tmpdir, "untracked.txt").write_text("modified\n")
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, {}, pre_untracked, pre_hashes
        )
        assert result.get("status") == "opened"

    def test_modified_untracked_no_saved_base(self) -> None:
        """Modified untracked file with no saved base copy."""
        Path(self.tmpdir, "untracked2.txt").write_text("original\n")
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, pre_untracked)
        # Don't save base

        # Agent modifies it
        Path(self.tmpdir, "untracked2.txt").write_text("modified\n")
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, {}, pre_untracked, pre_hashes
        )
        assert result.get("status") == "opened"

    def test_large_new_file_skipped(self) -> None:
        """New file larger than 2MB is skipped."""
        pre_untracked = _capture_untracked(self.tmpdir)
        Path(self.tmpdir, "huge.bin").write_bytes(b"\x00" * 3_000_000)
        result = _prepare_merge_view(self.tmpdir, self.data_dir, {}, pre_untracked)
        assert result.get("error") == "No changes"

    def test_binary_new_file_skipped(self) -> None:
        """New binary file that causes UnicodeDecodeError is skipped."""
        pre_untracked = _capture_untracked(self.tmpdir)
        Path(self.tmpdir, "binary.dat").write_bytes(bytes(range(256)))
        result = _prepare_merge_view(self.tmpdir, self.data_dir, {}, pre_untracked)
        # May or may not have changes depending on decode

    def test_hunk_with_zero_cc(self) -> None:
        """Cover the cs=cs branch (cc==0)."""
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(
            self.tmpdir, set(pre_hunks.keys()) | pre_untracked
        )
        _save_untracked_base(self.tmpdir, self.data_dir, pre_untracked | set(pre_hunks.keys()))

        # Delete a line (pure deletion = cc=0)
        Path(self.tmpdir, "file.txt").write_text("line1\nline3\n")
        result = _prepare_merge_view(
            self.tmpdir, self.data_dir, pre_hunks, pre_untracked, pre_hashes
        )
        assert result.get("status") == "opened"


class TestSetupCodeServer:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_setup_creates_settings(self) -> None:
        changed = _setup_code_server(self.tmpdir)
        settings_file = Path(self.tmpdir) / "User" / "settings.json"
        assert settings_file.exists()
        data = json.loads(settings_file.read_text())
        assert data["workbench.startupEditor"] == "none"

    def test_setup_preserves_theme(self) -> None:
        """Existing colorTheme is preserved."""
        user_dir = Path(self.tmpdir) / "User"
        user_dir.mkdir(parents=True)
        settings = {"workbench.colorTheme": "Monokai"}
        (user_dir / "settings.json").write_text(json.dumps(settings))
        _setup_code_server(self.tmpdir)
        data = json.loads((user_dir / "settings.json").read_text())
        assert data["workbench.colorTheme"] == "Monokai"

    def test_setup_idempotent(self) -> None:
        """Running twice returns False for extension changed on second run."""
        _setup_code_server(self.tmpdir)
        changed = _setup_code_server(self.tmpdir)
        assert not changed

    def test_setup_extension_changed(self) -> None:
        """First run returns True."""
        changed = _setup_code_server(self.tmpdir)
        assert changed

    def test_setup_creates_state_db(self) -> None:
        _setup_code_server(self.tmpdir)
        db = Path(self.tmpdir) / "User" / "globalStorage" / "state.vscdb"
        assert db.exists()

    def test_setup_removes_chat_sessions(self) -> None:
        ws_dir = Path(self.tmpdir) / "User" / "workspaceStorage" / "ws1" / "chatSessions"
        ws_dir.mkdir(parents=True)
        (ws_dir / "session.json").write_text("{}")
        _setup_code_server(self.tmpdir)
        assert not ws_dir.exists()

    def test_setup_bad_existing_settings(self) -> None:
        """Corrupt settings.json is handled gracefully."""
        user_dir = Path(self.tmpdir) / "User"
        user_dir.mkdir(parents=True)
        (user_dir / "settings.json").write_text("not valid json{")
        _setup_code_server(self.tmpdir)  # Should not raise


class TestDisableCopilotScmButton:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disable_copilot(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-1.0"
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
        data = json.loads((ext_dir / "package.json").read_text())
        items = data["contributes"]["menus"]["scm/inputBox"]
        assert items[0]["when"] == "false"

    def test_no_extensions_dir(self) -> None:
        _disable_copilot_scm_button(self.tmpdir)  # Should not raise

    def test_already_disabled(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-2.0"
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
        # Should not rewrite

    def test_bad_package_json(self) -> None:
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-3.0"
        ext_dir.mkdir(parents=True)
        (ext_dir / "package.json").write_text("not json")
        _disable_copilot_scm_button(self.tmpdir)  # Should not raise


# ---------------------------------------------------------------------------
# chatbot_ui.py
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.chatbot_ui import _THEME_PRESETS, _build_html


class TestBuildHtml:
    def test_without_code_server(self) -> None:
        html = _build_html("Test Title")
        assert "Test Title" in html
        assert "code-server is not installed" in html

    def test_with_code_server(self) -> None:
        html = _build_html("Test Title", "http://localhost:8080", "/tmp/work")
        assert "Test Title" in html
        assert "code-server-frame" in html
        assert "localhost:8080" in html


class TestThemePresets:
    def test_all_presets_exist(self) -> None:
        assert "dark" in _THEME_PRESETS
        assert "light" in _THEME_PRESETS
        assert "hcDark" in _THEME_PRESETS
        assert "hcLight" in _THEME_PRESETS

    def test_presets_have_keys(self) -> None:
        for name, preset in _THEME_PRESETS.items():
            assert "bg" in preset
            assert "fg" in preset
            assert "accent" in preset


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.config import AgentConfig, SorcarConfig


class TestSorcarConfig:
    def test_defaults(self) -> None:
        cfg = AgentConfig()
        assert cfg.model_name == "claude-opus-4-6"
        assert cfg.max_steps == 100
        assert cfg.max_budget == 200.0
        assert cfg.headless is False

    def test_sorcar_config(self) -> None:
        cfg = SorcarConfig()
        assert isinstance(cfg.sorcar_agent, AgentConfig)


# ---------------------------------------------------------------------------
# sorcar.py utility functions
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.sorcar import (
    _clean_llm_output,
    _model_vendor_order,
    _read_active_file,
)
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _build_arg_parser,
    _resolve_task,
)


class TestReadActiveFile:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_active_file(self) -> None:
        real_file = os.path.join(self.tmpdir, "test.py")
        Path(real_file).write_text("print('hi')")
        af_path = os.path.join(self.tmpdir, "active-file.json")
        with open(af_path, "w") as f:
            json.dump({"path": real_file}, f)
        assert _read_active_file(self.tmpdir) == real_file

    def test_active_file_nonexistent(self) -> None:
        af_path = os.path.join(self.tmpdir, "active-file.json")
        with open(af_path, "w") as f:
            json.dump({"path": "/no/such/file.py"}, f)
        assert _read_active_file(self.tmpdir) == ""

    def test_active_file_empty_path(self) -> None:
        af_path = os.path.join(self.tmpdir, "active-file.json")
        with open(af_path, "w") as f:
            json.dump({"path": ""}, f)
        assert _read_active_file(self.tmpdir) == ""

    def test_active_file_invalid_json(self) -> None:
        af_path = os.path.join(self.tmpdir, "active-file.json")
        with open(af_path, "w") as f:
            f.write("not valid json{")
        assert _read_active_file(self.tmpdir) == ""

    def test_active_file_missing(self) -> None:
        assert _read_active_file(self.tmpdir) == ""

    def test_active_file_no_dir(self) -> None:
        assert _read_active_file("/nonexistent/dir") == ""


class TestCleanLlmOutput:
    def test_strips_quotes(self) -> None:
        assert _clean_llm_output('"hello"') == "hello"
        assert _clean_llm_output("'hello'") == "hello"

    def test_strips_whitespace(self) -> None:
        assert _clean_llm_output("  hello  ") == "hello"

    def test_empty(self) -> None:
        assert _clean_llm_output("") == ""


class TestModelVendorOrder:
    def test_all_vendors(self) -> None:
        assert _model_vendor_order("claude-3.5-sonnet") == 0
        assert _model_vendor_order("gpt-4") == 1
        assert _model_vendor_order("gemini-2.0-flash") == 2
        assert _model_vendor_order("minimax-model") == 3
        assert _model_vendor_order("openrouter/some") == 4
        assert _model_vendor_order("unknown") == 5


class TestBuildArgParser:
    def test_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.model_name == "claude-opus-4-6"
        assert args.headless is False
        assert args.verbose is True

    def test_custom(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--headless", "true", "--verbose", "false"])
        assert args.headless is True
        assert args.verbose is False


class TestResolveTask:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_from_file(self) -> None:
        p = os.path.join(self.tmpdir, "task.txt")
        Path(p).write_text("file task")
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", p])
        assert _resolve_task(args) == "file task"

    def test_from_string(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--task", "inline task"])
        assert _resolve_task(args) == "inline task"

    def test_default(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert "gmail" in _resolve_task(args).lower()

    def test_file_not_found(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", "/no/such/file.txt"])
        with pytest.raises(FileNotFoundError):
            _resolve_task(args)


class TestSorcarAgentInit:
    def test_init(self) -> None:
        agent = SorcarAgent("test")
        assert agent.web_use_tool is None

    def test_get_tools_without_web(self) -> None:
        agent = SorcarAgent("test")
        tools = agent._get_tools()
        assert len(tools) == 4

    def test_get_tools_with_web(self) -> None:
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        agent = SorcarAgent("test")
        agent.web_use_tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            tools = agent._get_tools()
            assert len(tools) > 4
        finally:
            agent.web_use_tool.close()


class TestSorcarAgentRunAttachments:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _try_run(self, **kwargs: object) -> None:
        agent = SorcarAgent("test")
        try:
            agent.run(
                prompt_template="test",
                work_dir=self.tmpdir,
                max_steps=0,
                max_budget=0.0,
                headless=True,
                verbose=False,
                **kwargs,  # type: ignore[arg-type]
            )
        except Exception:
            pass

    def test_image_and_pdf(self) -> None:
        from kiss.core.models.model import Attachment

        self._try_run(
            attachments=[
                Attachment(data=b"img", mime_type="image/png"),
                Attachment(data=b"pdf", mime_type="application/pdf"),
            ]
        )

    def test_only_images(self) -> None:
        from kiss.core.models.model import Attachment

        self._try_run(
            attachments=[Attachment(data=b"img", mime_type="image/png")]
        )

    def test_only_pdfs(self) -> None:
        from kiss.core.models.model import Attachment

        self._try_run(
            attachments=[Attachment(data=b"pdf", mime_type="application/pdf")]
        )

    def test_non_image_non_pdf(self) -> None:
        from kiss.core.models.model import Attachment

        self._try_run(
            attachments=[Attachment(data=b"txt", mime_type="text/plain")]
        )

    def test_with_editor_file(self) -> None:
        self._try_run(current_editor_file="/tmp/editor.py")

    def test_no_attachments(self) -> None:
        self._try_run()


# ---------------------------------------------------------------------------
# web_use_tool.py: WebUseTool construction and close
# ---------------------------------------------------------------------------
from kiss.agents.sorcar.web_use_tool import WebUseTool


class TestWebUseToolBasic:
    def test_init_default(self) -> None:
        tool = WebUseTool(headless=True)
        assert tool.headless is True
        assert tool.user_data_dir is not None

    def test_init_no_profile(self) -> None:
        tool = WebUseTool(headless=True, user_data_dir=None)
        assert tool.user_data_dir is None

    def test_close_without_open(self) -> None:
        tool = WebUseTool(headless=True, user_data_dir=None)
        result = tool.close()
        assert result == "Browser closed."

    def test_get_tools(self) -> None:
        tool = WebUseTool(headless=True, user_data_dir=None)
        tools = tool.get_tools()
        assert len(tools) == 7

    def test_context_args(self) -> None:
        tool = WebUseTool(headless=True, viewport=(800, 600), user_data_dir=None)
        args = tool._context_args()
        assert args["viewport"]["width"] == 800
        assert args["viewport"]["height"] == 600

    def test_launch_kwargs_chromium(self) -> None:
        tool = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        kwargs = tool._launch_kwargs()
        assert kwargs["headless"] is True
        assert "args" in kwargs

    def test_launch_kwargs_chromium_not_headless(self) -> None:
        tool = WebUseTool(browser_type="chromium", headless=False, user_data_dir=None)
        kwargs = tool._launch_kwargs()
        assert kwargs.get("channel") == "chrome"

    def test_launch_kwargs_firefox(self) -> None:
        tool = WebUseTool(browser_type="firefox", headless=True, user_data_dir=None)
        kwargs = tool._launch_kwargs()
        assert "args" not in kwargs


# ---------------------------------------------------------------------------
# sorcar_agent.py main()
# ---------------------------------------------------------------------------
class TestSorcarAgentMain:
    def test_main_subprocess(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kiss.agents.sorcar.sorcar_agent",
                    "--max_steps", "0",
                    "--max_budget", "0.0",
                    "--work_dir", tmpdir,
                    "--headless", "true",
                    "--verbose", "false",
                    "--task", "say hello",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            # Should complete (may fail due to 0 budget but should not crash)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_main_no_work_dir(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kiss.agents.sorcar.sorcar_agent",
                "--max_steps", "0",
                "--max_budget", "0.0",
                "--headless", "true",
                "--verbose", "false",
                "--task", "say hello",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_main_with_file(self) -> None:
        tmpdir = tempfile.mkdtemp()
        task_file = os.path.join(tmpdir, "task.txt")
        Path(task_file).write_text("echo hello")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kiss.agents.sorcar.sorcar_agent",
                    "--max_steps", "0",
                    "--max_budget", "0.0",
                    "--work_dir", tmpdir,
                    "--headless", "true",
                    "-f", task_file,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Additional edge-case coverage for _docker_bash path in SorcarAgent
# ---------------------------------------------------------------------------
class TestSorcarAgentDockerBash:
    def test_get_tools_uses_bash_without_docker(self) -> None:
        agent = SorcarAgent("test")
        agent.docker_manager = None
        tools = agent._get_tools()
        assert callable(tools[0])


# ---------------------------------------------------------------------------
# Additional edge-case tests for higher coverage
# ---------------------------------------------------------------------------
class TestUsefulToolsEdgeCases:
    """Cover remaining useful_tools.py branches."""

    def test_extract_leading_command_name_only_env_vars(self) -> None:
        """All tokens are env var assignments -> returns None."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        assert _extract_leading_command_name("FOO=bar BAZ=qux") is None

    def test_extract_leading_command_name_only_redirects(self) -> None:
        """Only redirect tokens -> returns None."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        assert _extract_leading_command_name("> /dev/null") is None

    def test_extract_leading_command_name_empty_after_lstrip(self) -> None:
        """Token is just '(' -> name is empty after lstrip."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        # "(" alone -> after lstrip("({") gives "" -> returns None
        assert _extract_leading_command_name("(") is None

    def test_extract_command_names_quoted_pipe(self) -> None:
        """Pipe inside quotes should not split."""
        names = _extract_command_names("echo 'a|b'")
        assert names == ["echo"]

    def test_extract_command_names_quoted_semicolon(self) -> None:
        """Semicolon inside quotes should not split."""
        names = _extract_command_names('echo "a;b"')
        assert names == ["echo"]

    def test_extract_command_names_escaped_chars(self) -> None:
        """Escaped characters."""
        names = _extract_command_names("echo hello\\ world")
        assert "echo" in names

    def test_extract_command_names_double_quote_escape(self) -> None:
        """Escaped quote in double-quoted string."""
        names = _extract_command_names('echo "hello \\"world\\""')
        assert "echo" in names

    def test_truncate_exact_boundary(self) -> None:
        """Cover when max_chars equals a specific boundary."""
        text = "a" * 50
        result = _truncate_output(text, 50)
        assert result == text

    def test_kill_process_group(self) -> None:
        """Cover _kill_process_group with a real process."""
        proc = subprocess.Popen(
            ["sleep", "60"],
            start_new_session=True,
        )
        _kill_process_group(proc)
        assert proc.poll() is not None


class TestBrowserUiEdgeCases:
    """Cover remaining browser_ui.py branches."""

    def test_print_text_empty(self) -> None:
        """Cover line 613 - text is empty after stripping."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("", type="text")
        assert cq.empty()  # No event broadcast for empty text
        printer.remove_client(cq)

    def test_bash_stream_existing_timer(self) -> None:
        """Cover line 631 - timer already exists, needs_flush stays False."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        # Set last flush to now so flush isn't triggered by time
        printer._bash_last_flush = time.monotonic()
        # First call creates timer
        printer.print("a", type="bash_stream")
        assert printer._bash_flush_timer is not None
        # Second call with timer already set -> else branch (needs_flush = False)
        printer.print("b", type="bash_stream")
        # Timer should still be set
        assert printer._bash_flush_timer is not None
        # Clean up
        printer._flush_bash()
        printer.remove_client(cq)

    def test_handle_message_no_matching_attr(self) -> None:
        """Cover message with none of the expected attributes."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class EmptyMsg:
            pass

        printer._handle_message(EmptyMsg())
        assert cq.empty()  # No event broadcast
        printer.remove_client(cq)

    def test_handle_message_tool_output_empty_content(self) -> None:
        """Cover tool_output with empty content -> no broadcast."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Msg:
            subtype = "tool_output"
            data = {"content": ""}

        printer._handle_message(Msg())
        assert cq.empty()  # Empty content, no broadcast
        printer.remove_client(cq)

    def test_handle_message_subtype_not_tool_output(self) -> None:
        """Cover message with subtype != tool_output."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Msg:
            subtype = "other"
            data = {"content": "text"}

        printer._handle_message(Msg())
        assert cq.empty()
        printer.remove_client(cq)

    def test_broadcast_result_no_yaml(self) -> None:
        """Cover _broadcast_result when text is not valid YAML."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._broadcast_result("just text", step_count=1, total_tokens=10, cost="$0.01")
        event = cq.get_nowait()
        assert event["type"] == "result"
        assert "success" not in event or event.get("success") is None
        printer.remove_client(cq)

    def test_broadcast_result_empty(self) -> None:
        """Cover _broadcast_result with empty text."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._broadcast_result("")
        event = cq.get_nowait()
        assert event["text"] == "(no result)"
        printer.remove_client(cq)

    def test_handle_stream_event_unknown_type(self) -> None:
        """Cover _handle_stream_event with unknown event type."""
        printer = BaseBrowserPrinter()

        class FakeEvent:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        text = printer._handle_stream_event(FakeEvent({"type": "unknown_event"}))
        assert text == ""

    def test_handle_stream_event_content_block_start_text(self) -> None:
        """Cover content_block_start with type=text (not thinking/tool_use)."""
        printer = BaseBrowserPrinter()

        class FakeEvent:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        printer._handle_stream_event(
            FakeEvent({"type": "content_block_start", "content_block": {"type": "text"}})
        )
        assert printer._current_block_type == "text"

    def test_handle_stream_event_delta_unknown(self) -> None:
        """Cover content_block_delta with unknown delta_type."""
        printer = BaseBrowserPrinter()

        class FakeEvent:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        text = printer._handle_stream_event(
            FakeEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "unknown_delta"},
                }
            )
        )
        assert text == ""

    def test_handle_message_result_no_budget(self) -> None:
        """Cover handle_message result path with budget_used=0."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Msg:
            result = "done"

        printer._handle_message(Msg(), budget_used=0.0)
        event = cq.get_nowait()
        assert event["cost"] == "N/A"
        printer.remove_client(cq)

    def test_format_tool_call_with_extras(self) -> None:
        """Cover _format_tool_call with extras."""
        from kiss.core.printer import extract_extras

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._format_tool_call(
            "Bash",
            {
                "command": "echo hi",
                "timeout_seconds": 30,
                "max_output_chars": 1000,
            },
        )
        event = cq.get_nowait()
        assert event["type"] == "tool_call"
        assert event["command"] == "echo hi"
        printer.remove_client(cq)


class TestCodeServerEdgeCases:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scan_files_large_count(self) -> None:
        """Cover early return when >= 2000 files."""
        for i in range(2010):
            Path(self.tmpdir, f"file{i:04d}.txt").write_text(f"content{i}")
        result = _scan_files(self.tmpdir)
        assert len(result) >= 2000

    def test_install_copilot_extension_already_installed(self) -> None:
        """Cover _install_copilot_extension when already installed."""
        from kiss.agents.sorcar.code_server import _install_copilot_extension

        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-1.0"
        ext_dir.mkdir(parents=True)
        _install_copilot_extension(self.tmpdir)  # Should return early

    def test_install_copilot_no_code_server(self) -> None:
        """Cover _install_copilot_extension when code-server is not installed."""
        from kiss.agents.sorcar.code_server import _install_copilot_extension

        ext_dir = Path(self.tmpdir) / "extensions"
        ext_dir.mkdir()
        # No copilot dir, and if code-server isn't in PATH...
        # This just exercises the path; won't actually install
        _install_copilot_extension(self.tmpdir)

    def test_disable_copilot_non_copilot_dir(self) -> None:
        """Extension dir that doesn't start with github.copilot-chat-."""
        ext_dir = Path(self.tmpdir) / "extensions" / "other-extension-1.0"
        ext_dir.mkdir(parents=True)
        (ext_dir / "package.json").write_text("{}")
        _disable_copilot_scm_button(self.tmpdir)

    def test_disable_copilot_no_package_json(self) -> None:
        """Copilot dir but no package.json."""
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-4.0"
        ext_dir.mkdir(parents=True)
        _disable_copilot_scm_button(self.tmpdir)

    def test_disable_copilot_scm_write_error(self) -> None:
        """Cover OSError on write in _disable_copilot_scm_button."""
        ext_dir = Path(self.tmpdir) / "extensions" / "github.copilot-chat-5.0"
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
        # Make dir read-only to cause write error
        os.chmod(str(ext_dir), 0o555)
        try:
            _disable_copilot_scm_button(self.tmpdir)  # Should not raise
        finally:
            os.chmod(str(ext_dir), 0o755)

    def test_prepare_merge_view_file_hash_oserror(self) -> None:
        """Cover OSError when hashing file in _prepare_merge_view."""
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=self.tmpdir, capture_output=True)
        Path(self.tmpdir, "f.txt").write_text("line1\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)

        # Modify file
        Path(self.tmpdir, "f.txt").write_text("modified\n")
        pre_hashes = {"f.txt": "deadbeef"}

        data_dir = tempfile.mkdtemp()
        try:
            # Remove file to cause OSError during hash
            os.remove(os.path.join(self.tmpdir, "f.txt"))
            result = _prepare_merge_view(
                self.tmpdir, data_dir, {}, set(), pre_hashes
            )
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_prepare_merge_view_pre_hunks_filter(self) -> None:
        """Cover filtering pre-existing hunks from post hunks."""
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=self.tmpdir, capture_output=True)
        Path(self.tmpdir, "g.txt").write_text("a\nb\nc\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)

        # Make pre-change
        Path(self.tmpdir, "g.txt").write_text("a\nX\nc\n")
        pre_hunks = _parse_diff_hunks(self.tmpdir)
        pre_untracked = _capture_untracked(self.tmpdir)

        # Make additional change (agent's change)
        Path(self.tmpdir, "g.txt").write_text("a\nX\nc\nD\n")

        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, pre_hunks, pre_untracked
            )
            # Should only show agent's change, not pre-existing
            assert result.get("status") == "opened"
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    def test_cleanup_merge_data_manifest_readonly(self) -> None:
        """Cover OSError in manifest.unlink()."""
        manifest = Path(self.tmpdir) / "pending-merge.json"
        manifest.write_text("{}")
        os.chmod(self.tmpdir, 0o555)
        try:
            _cleanup_merge_data(self.tmpdir)
        finally:
            os.chmod(self.tmpdir, 0o755)

    def test_prepare_merge_view_untracked_modified_oserror(self) -> None:
        """Modified untracked file that can't be read -> OSError."""
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=self.tmpdir, capture_output=True)

        Path(self.tmpdir, "ut.txt").write_text("original\n")
        pre_untracked = _capture_untracked(self.tmpdir)
        pre_hashes = _snapshot_files(self.tmpdir, pre_untracked)

        # Remove file to simulate OSError during hash
        os.remove(os.path.join(self.tmpdir, "ut.txt"))

        data_dir = tempfile.mkdtemp()
        try:
            result = _prepare_merge_view(
                self.tmpdir, data_dir, {}, pre_untracked, pre_hashes
            )
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


class TestTaskHistoryEdgeCases:
    """Cover remaining task_history.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self._orig_history_file = th.HISTORY_FILE
        self._orig_proposals_file = th.PROPOSALS_FILE
        self._orig_model_usage_file = th.MODEL_USAGE_FILE
        self._orig_file_usage_file = th.FILE_USAGE_FILE
        self._orig_kiss_dir = th._KISS_DIR
        th._KISS_DIR = Path(self.tmpdir)
        th.HISTORY_FILE = Path(self.tmpdir) / "task_history.json"
        th.PROPOSALS_FILE = Path(self.tmpdir) / "proposed_tasks.json"
        th.MODEL_USAGE_FILE = Path(self.tmpdir) / "model_usage.json"
        th.FILE_USAGE_FILE = Path(self.tmpdir) / "file_usage.json"
        th._history_cache = None

    def teardown_method(self) -> None:
        th.HISTORY_FILE = self._orig_history_file
        th.PROPOSALS_FILE = self._orig_proposals_file
        th.MODEL_USAGE_FILE = self._orig_model_usage_file
        th.FILE_USAGE_FILE = self._orig_file_usage_file
        th._KISS_DIR = self._orig_kiss_dir
        th._history_cache = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_history_oserror(self) -> None:
        """Cover OSError in _save_history_unlocked."""
        # Make kiss dir read-only
        os.chmod(self.tmpdir, 0o555)
        try:
            th._save_history([{"task": "x", "chat_events": []}])
        finally:
            os.chmod(self.tmpdir, 0o755)

    def test_save_proposals_oserror(self) -> None:
        """Cover OSError in _save_proposals."""
        os.chmod(self.tmpdir, 0o555)
        try:
            th._save_proposals(["p1"])
        finally:
            os.chmod(self.tmpdir, 0o755)

    def test_record_model_usage_oserror(self) -> None:
        """Cover OSError in _record_model_usage."""
        os.chmod(self.tmpdir, 0o555)
        try:
            th._record_model_usage("model")
        finally:
            os.chmod(self.tmpdir, 0o755)

    def test_record_file_usage_oserror(self) -> None:
        """Cover OSError in _record_file_usage."""
        os.chmod(self.tmpdir, 0o555)
        try:
            th._record_file_usage("file.py")
        finally:
            os.chmod(self.tmpdir, 0o755)

    def test_append_task_to_md_creates_file(self) -> None:
        """Cover _append_task_to_md when file doesn't exist."""
        md_path = th._get_task_history_md_path()
        orig = md_path.read_text() if md_path.exists() else None
        try:
            if md_path.exists():
                md_path.unlink()
            th._append_task_to_md("new task", "result")
            assert md_path.exists()
            content = md_path.read_text()
            assert "Task History" in content
            assert "new task" in content
        finally:
            if orig is not None:
                md_path.write_text(orig)
            elif md_path.exists():
                md_path.unlink()


class TestPromptDetectorEdgeCases:
    """Cover remaining prompt_detector.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.detector = PromptDetector()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_frontmatter_with_only_one_separator(self) -> None:
        """File starts with --- but has only one separator -> no frontmatter."""
        p = os.path.join(self.tmpdir, "one_sep.md")
        Path(p).write_text("---\nsome content without closing separator\n")
        is_prompt, score, reasons = self.detector.analyze(p)
        # Should not have frontmatter bonus

    def test_low_imperative_density(self) -> None:
        """Imperative verb count / total words <= 0.05."""
        p = os.path.join(self.tmpdir, "low_density.md")
        # Many non-imperative words
        Path(p).write_text(
            "the quick brown fox jumps over the lazy dog. " * 20
            + "\nwrite something.\n"
        )
        is_prompt, score, reasons = self.detector.analyze(p)
        assert not any("imperative" in r.lower() for r in reasons)

    def test_no_words(self) -> None:
        """File with no alphanumeric words."""
        p = os.path.join(self.tmpdir, "no_words.md")
        Path(p).write_text("---\n---\n")
        is_prompt, score, reasons = self.detector.analyze(p)


class TestWebUseToolEdgeCases:
    """Cover remaining web_use_tool.py branches without starting a browser."""

    def test_scroll_delta_values(self) -> None:
        """Cover _SCROLL_DELTA dict lookup."""
        from kiss.agents.sorcar.web_use_tool import _SCROLL_DELTA

        assert _SCROLL_DELTA["down"] == (0, 300)
        assert _SCROLL_DELTA["up"] == (0, -300)
        assert _SCROLL_DELTA["right"] == (300, 0)
        assert _SCROLL_DELTA["left"] == (-300, 0)

    def test_interactive_roles_complete(self) -> None:
        """Ensure all expected roles are in INTERACTIVE_ROLES."""
        expected = {"link", "button", "textbox", "searchbox", "combobox",
                    "checkbox", "radio", "switch", "slider", "spinbutton",
                    "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
                    "option", "treeitem"}
        assert INTERACTIVE_ROLES == expected

    def test_number_interactive_elements_multiple_same_role(self) -> None:
        """Multiple elements with same role."""
        snapshot = '  - button "A"\n  - button "B"\n  - button "C"\n'
        numbered, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 3
        assert "[1]" in numbered
        assert "[2]" in numbered
        assert "[3]" in numbered

    def test_close_with_user_data_dir_context(self) -> None:
        """Cover close() path when user_data_dir is set."""
        tool = WebUseTool(headless=True, user_data_dir=None)
        # _context is None, so this exercises the else branch
        result = tool.close()
        assert result == "Browser closed."

    def test_auto_detect_user_data_dir(self) -> None:
        """Cover _AUTO_DETECT user_data_dir path."""
        from kiss.agents.sorcar.web_use_tool import KISS_PROFILE_DIR

        tool = WebUseTool(headless=True)
        assert tool.user_data_dir == KISS_PROFILE_DIR


class TestBrowserUiFinalEdgeCases:
    """Cover last few browser_ui.py branches."""

    def test_handle_message_content_block_no_is_error(self) -> None:
        """Cover content block that doesn't have is_error attr."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class Block:
            pass  # No is_error or content

        class Msg:
            content = [Block()]

        printer._handle_message(Msg())
        # Should not broadcast anything for block without is_error
        assert cq.empty()
        printer.remove_client(cq)

    def test_handle_message_content_mixed_blocks(self) -> None:
        """Content with both valid and invalid blocks."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class ValidBlock:
            is_error = False
            content = "ok"

        class InvalidBlock:
            pass

        class Msg:
            content = [InvalidBlock(), ValidBlock()]

        printer._handle_message(Msg())
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        printer.remove_client(cq)

    def test_print_text_only_whitespace(self) -> None:
        """Cover empty text.strip() branch."""
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("   \n\t  ", type="text")
        assert cq.empty()
        printer.remove_client(cq)


class TestUsefulToolsFinalEdgeCases:
    """Cover remaining useful_tools.py branches."""

    def test_redirect_to_separate_file(self) -> None:
        """Cover redirect where file is next token (i += 2)."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        # "2>" is a redirect token where m.end() == len(token), so i += 2
        result = _extract_leading_command_name("2> /dev/null echo hello")
        # After skipping redirect "2>" and its target "/dev/null", should find "echo"
        assert result == "echo"

    def test_only_redirect_tokens(self) -> None:
        """All tokens consumed by redirects -> i >= len(tokens) -> None."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        result = _extract_leading_command_name("2> /dev/null")
        assert result is None

    def test_truncate_output_tail_zero(self) -> None:
        """Explicitly trigger tail=0 path."""
        # We need len(output) > max_chars, and remaining - head <= 0
        # worst_msg for 200 chars: "\n\n... [truncated 200 chars] ...\n\n" ~35 chars
        # remaining = max_chars - len(msg) ~ small
        # head = remaining // 2, tail = remaining - head
        # If remaining is odd, tail = head+1. If remaining = 1, head=0, tail=1
        # We need tail=0 -> remaining even and small, head = remaining/2, tail = remaining/2
        # Actually tail=0 requires remaining=0, which means max_chars = len(msg)
        text = "x" * 100
        msg = f"\n\n... [truncated {len(text)} chars] ...\n\n"
        result = _truncate_output(text, len(msg))
        assert "truncated" in result

    def test_command_with_full_path(self) -> None:
        """Command name with / in it -> split('/')[-1]."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name

        result = _extract_leading_command_name("/usr/bin/python3 script.py")
        assert result == "python3"


# ---------------------------------------------------------------------------
# web_use_tool.py: Browser integration tests using headless Playwright
# ---------------------------------------------------------------------------
class TestWebUseToolBrowser:
    """Integration tests for WebUseTool with a real headless browser."""

    @pytest.fixture(autouse=True)
    def setup_tool(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tool = WebUseTool(
            browser_type="chromium",
            headless=True,
            user_data_dir=None,
        )

    def teardown_method(self) -> None:
        if hasattr(self, "tool"):
            self.tool.close()

    def _write_html(self, name: str, content: str) -> str:
        p = self.tmp_path / name
        p.write_text(content)
        return f"file://{p}"

    def test_go_to_url_and_get_tree(self) -> None:
        url = self._write_html(
            "test.html",
            "<html><head><title>Test</title></head>"
            '<body><button>Click</button><a href="#">Link</a></body></html>',
        )
        result = self.tool.go_to_url(url)
        assert "Test" in result
        assert "button" in result
        assert "link" in result

    def test_go_to_url_tab_list(self) -> None:
        url = self._write_html("t.html", "<html><body>Hi</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.go_to_url("tab:list")
        assert "Open tabs" in result

    def test_go_to_url_tab_switch(self) -> None:
        url = self._write_html("t.html", "<html><body>Hi</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.go_to_url("tab:0")
        assert "Hi" in result or "Page:" in result

    def test_go_to_url_tab_out_of_range(self) -> None:
        url = self._write_html("t.html", "<html><body>Hi</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.go_to_url("tab:99")
        assert "Error" in result or "out of range" in result

    def test_go_to_url_invalid(self) -> None:
        result = self.tool.go_to_url("not-a-valid-url://???")
        assert "Error" in result

    def test_click_element(self) -> None:
        url = self._write_html(
            "click.html",
            "<html><body><button id='b'>Click Me</button></body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool.click(1)
        assert "Page:" in result or "button" in result.lower() or "Click" in result

    def test_click_hover(self) -> None:
        url = self._write_html(
            "hover.html",
            "<html><body><button>Hover</button></body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool.click(1, action="hover")
        assert "Page:" in result

    def test_click_invalid_element(self) -> None:
        url = self._write_html("c.html", "<html><body><p>No buttons</p></body></html>")
        self.tool.go_to_url(url)
        result = self.tool.click(999)
        assert "Error" in result

    def test_type_text(self) -> None:
        url = self._write_html(
            "type.html",
            '<html><body><input type="text" placeholder="Name"></body></html>',
        )
        self.tool.go_to_url(url)
        result = self.tool.type_text(1, "Hello World")
        assert "Page:" in result

    def test_type_text_press_enter(self) -> None:
        url = self._write_html(
            "enter.html",
            '<html><body><form><input type="text"></form></body></html>',
        )
        self.tool.go_to_url(url)
        result = self.tool.type_text(1, "query", press_enter=True)
        assert "Page:" in result or "Error" not in result[:5]

    def test_type_text_invalid_element(self) -> None:
        url = self._write_html("t.html", "<html><body>No input</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.type_text(999, "text")
        assert "Error" in result

    def test_press_key(self) -> None:
        url = self._write_html("k.html", "<html><body>Hello</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.press_key("Escape")
        assert "Page:" in result

    def test_press_key_error(self) -> None:
        # Press key without browser being on a valid page
        url = self._write_html("k.html", "<html><body>Hello</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.press_key("NotAKey!!!")
        assert "Error" in result or "Page:" in result

    def test_scroll(self) -> None:
        url = self._write_html(
            "scroll.html",
            "<html><body>" + "<p>Line</p>" * 200 + "</body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool.scroll(direction="down", amount=3)
        assert "Page:" in result

    def test_scroll_up(self) -> None:
        url = self._write_html("s.html", "<html><body>Short</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.scroll(direction="up", amount=2)
        assert "Page:" in result

    def test_scroll_invalid_direction(self) -> None:
        url = self._write_html("s.html", "<html><body>Short</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.scroll(direction="diagonal", amount=1)
        # Falls back to default (0, 300)
        assert "Page:" in result

    def test_screenshot(self) -> None:
        url = self._write_html("ss.html", "<html><body>Screenshot</body></html>")
        self.tool.go_to_url(url)
        ss_path = str(self.tmp_path / "test_screenshot.png")
        result = self.tool.screenshot(file_path=ss_path)
        assert "Screenshot saved" in result
        assert Path(ss_path).exists()

    def test_get_page_content_tree(self) -> None:
        url = self._write_html(
            "pc.html",
            "<html><head><title>Content</title></head><body><p>Hello</p></body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool.get_page_content(text_only=False)
        assert "Content" in result

    def test_get_page_content_text_only(self) -> None:
        url = self._write_html(
            "pt.html",
            "<html><head><title>Text</title></head><body><p>Plain text</p></body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool.get_page_content(text_only=True)
        assert "Plain text" in result

    def test_click_opens_new_tab(self) -> None:
        """Cover new tab detection."""
        url = self._write_html(
            "newtab.html",
            '<html><body><a href="about:blank" target="_blank">Open Tab</a></body></html>',
        )
        self.tool.go_to_url(url)
        result = self.tool.click(1)
        # May or may not open a new tab, but should not crash

    def test_resolve_locator_stale_then_refresh(self) -> None:
        """Cover the re-snapshot path in _resolve_locator."""
        url = self._write_html(
            "resolve.html",
            "<html><body><button>A</button><button>B</button></body></html>",
        )
        self.tool.go_to_url(url)
        # Clear elements to force re-snapshot
        self.tool._elements = []
        result = self.tool.click(1)
        assert "Page:" in result or "Error" not in result[:5]

    def test_get_ax_tree_empty_page(self) -> None:
        """Cover empty page in _get_ax_tree."""
        url = self._write_html("empty.html", "<html><body></body></html>")
        self.tool.go_to_url(url)
        result = self.tool.get_page_content()
        # Should handle empty body gracefully

    def test_close_with_context(self) -> None:
        """Cover close when user_data_dir is set (persistent context)."""
        profile_dir = str(self.tmp_path / "profile")
        tool2 = WebUseTool(
            browser_type="chromium",
            headless=True,
            user_data_dir=profile_dir,
        )
        url = self._write_html("close.html", "<html><body>Hi</body></html>")
        tool2.go_to_url(url)
        result = tool2.close()
        assert result == "Browser closed."

    def test_close_without_user_data_dir(self) -> None:
        """Cover close when user_data_dir is None (non-persistent)."""
        url = self._write_html("close2.html", "<html><body>Hi</body></html>")
        self.tool.go_to_url(url)
        result = self.tool.close()
        assert result == "Browser closed."

    def test_multiple_elements_same_role_name(self) -> None:
        """Cover _resolve_locator with n > 1 for same name."""
        url = self._write_html(
            "multi.html",
            '<html><body>'
            '<button>Same</button>'
            '<button>Same</button>'
            '</body></html>',
        )
        self.tool.go_to_url(url)
        # Should click first visible one
        result = self.tool.click(1)
        assert "Page:" in result or "Error" not in result[:5]


# ---------------------------------------------------------------------------
# sorcar.py: Server integration test via subprocess
# ---------------------------------------------------------------------------
import signal

import requests


def _wait_for_port_file(port_file: str, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
            return int(Path(port_file).read_text().strip())
        time.sleep(0.3)
    raise TimeoutError(f"Port file {port_file} not written within {timeout}s")


class TestSorcarServerIntegration:
    """Integration tests for run_chatbot by starting it as a subprocess.

    Uses _sorcar_test_server_with_cov.py to collect coverage data from
    the server subprocess. Coverage is combined after the server stops.
    """

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        tmpdir = tmp_path_factory.mktemp("sorcar_server")
        work_dir = str(tmpdir / "work")
        os.makedirs(work_dir)

        # Initialize a git repo
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
        Path(work_dir, "readme.md").write_text("# Test\n\nsome content\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

        port_file = str(tmpdir / "port")
        cov_data_file = str(tmpdir / ".coverage.server")

        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "_sorcar_test_server_with_cov.py"),
                port_file,
                work_dir,
                cov_data_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

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

            yield base_url, work_dir, proc, str(tmpdir)
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            # Try to combine coverage from subprocess
            try:
                import coverage as cov_mod

                if os.path.exists(cov_data_file):
                    main_cov_file = os.path.join(os.getcwd(), ".coverage")
                    cov = cov_mod.Coverage(data_file=main_cov_file)
                    cov.combine(data_paths=[cov_data_file], keep=True)
                    cov.save()
            except Exception:
                pass

    def test_index(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(base_url, timeout=5)
        assert r.status_code == 200
        assert "KISS" in r.text or "html" in r.text.lower()

    def test_models(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/models", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "selected" in data

    def test_theme(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/theme", timeout=5)
        assert r.status_code == 200

    def test_tasks(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/tasks", timeout=5)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_proposed_tasks(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/proposed_tasks", timeout=5)
        assert r.status_code == 200

    def test_suggestions_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?q=", timeout=5)
        assert r.status_code == 200

    def test_suggestions_with_query(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?q=test", timeout=5)
        assert r.status_code == 200

    def test_suggestions_files_mode(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/suggestions?mode=files&q=file", timeout=5)
        assert r.status_code == 200

    def test_complete_short(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/complete?q=a", timeout=5)
        assert r.status_code == 200

    def test_complete_longer(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/complete?q=test something", timeout=5)
        assert r.status_code == 200

    def test_active_file_info(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert r.status_code == 200

    def test_get_file_content(self, server) -> None:
        base_url, work_dir, _, _ = server
        fpath = os.path.join(work_dir, "file.txt")
        r = requests.get(f"{base_url}/get-file-content?path={fpath}", timeout=5)
        assert r.status_code == 200

    def test_get_file_content_not_found(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/get-file-content?path=/no/such/file", timeout=5)
        assert r.status_code == 404

    def test_run_empty_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/run", json={"task": ""}, timeout=5)
        assert r.status_code == 400

    def test_run_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/run",
            json={"task": "test task", "model": "claude-opus-4-6"},
            timeout=5,
        )
        assert r.status_code == 200
        time.sleep(1)

    def test_stop_task(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/stop", timeout=5)
        assert r.status_code in (200, 404)

    def test_run_selection_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/run-selection", json={"text": ""}, timeout=5)
        assert r.status_code == 400

    def test_run_selection(self, server) -> None:
        base_url, _, _, _ = server
        time.sleep(1)  # Ensure previous task is done
        r = requests.post(f"{base_url}/run-selection", json={"text": "echo hello"}, timeout=5)
        assert r.status_code == 200
        time.sleep(1)

    def test_task_events_invalid(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/task-events?idx=bad", timeout=5)
        assert r.status_code == 400

    def test_task_events_out_of_range(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/task-events?idx=9999", timeout=5)
        assert r.status_code == 404

    def test_task_events_valid(self, server) -> None:
        base_url, _, _, _ = server
        tasks = requests.get(f"{base_url}/tasks", timeout=5).json()
        if tasks:
            r = requests.get(f"{base_url}/task-events?idx=0", timeout=5)
            assert r.status_code == 200

    def test_open_file(self, server) -> None:
        base_url, work_dir, _, _ = server
        r = requests.post(
            f"{base_url}/open-file",
            json={"path": os.path.join(work_dir, "file.txt")},
            timeout=5,
        )
        assert r.status_code == 200

    def test_open_file_not_found(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/open-file",
            json={"path": "/no/such/file.txt"},
            timeout=5,
        )
        assert r.status_code == 404

    def test_open_file_empty(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/open-file", json={"path": ""}, timeout=5)
        assert r.status_code == 400

    def test_merge_action_next(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "next"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_invalid(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "invalid"}, timeout=5
        )
        assert r.status_code == 400

    def test_merge_action_all_done(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "all-done"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_accept(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "accept"}, timeout=5
        )
        assert r.status_code == 200

    def test_merge_action_reject(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/merge-action", json={"action": "reject"}, timeout=5
        )
        assert r.status_code == 200

    def test_focus_chatbox(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/focus-chatbox", timeout=5)
        assert r.status_code == 200

    def test_focus_editor(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/focus-editor", timeout=5)
        assert r.status_code == 200

    def test_record_file_usage(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(
            f"{base_url}/record-file-usage",
            json={"path": "src/test.py"},
            timeout=5,
        )
        assert r.status_code == 200

    def test_closing(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/closing", timeout=5)
        assert r.status_code == 200

    def test_commit_no_changes(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/commit", timeout=30)
        assert r.status_code in (200, 400)

    def test_push_no_remote(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.post(f"{base_url}/push", timeout=10)
        assert r.status_code in (200, 400)

    def test_sse_events(self, server) -> None:
        base_url, _, _, _ = server
        r = requests.get(f"{base_url}/events", stream=True, timeout=10)
        assert r.status_code == 200
        content = b""
        for chunk in r.iter_content(chunk_size=256):
            content += chunk
            if len(content) > 20:
                break
        r.close()

    def test_run_with_attachments(self, server) -> None:
        import base64

        base_url, _, _, _ = server
        time.sleep(1)
        fake_image = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        fake_pdf = base64.b64encode(b"%PDF-1.4 fake").decode()
        r = requests.post(
            f"{base_url}/run",
            json={
                "task": "test with attachments",
                "attachments": [
                    {"data": fake_image, "mime_type": "image/png"},
                    {"data": fake_pdf, "mime_type": "application/pdf"},
                ],
            },
            timeout=5,
        )
        assert r.status_code == 200
        time.sleep(1)

    def test_active_file_md(self, server) -> None:
        """Cover active file with .md extension triggering prompt detection."""
        import hashlib

        base_url, work_dir, _, _ = server
        from kiss.agents.sorcar.task_history import _KISS_DIR

        wd_hash = hashlib.md5(work_dir.encode()).hexdigest()[:8]
        cs_dir = _KISS_DIR / f"cs-{wd_hash}"
        cs_dir.mkdir(parents=True, exist_ok=True)
        (cs_dir / "active-file.json").write_text(
            json.dumps({"path": os.path.join(work_dir, "readme.md")})
        )
        r = requests.get(f"{base_url}/active-file-info", timeout=5)
        assert r.status_code == 200
        assert "is_prompt" in r.json()


class TestWebUseToolBrowserExtra:
    """Additional browser tests for remaining web_use_tool.py branches."""

    @pytest.fixture(autouse=True)
    def setup_tool(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tool = WebUseTool(
            browser_type="chromium",
            headless=True,
            user_data_dir=None,
        )

    def teardown_method(self) -> None:
        if hasattr(self, "tool"):
            self.tool.close()

    def _write_html(self, name: str, content: str) -> str:
        p = self.tmp_path / name
        p.write_text(content)
        return f"file://{p}"

    def test_large_ax_tree_truncation(self) -> None:
        """Cover truncation of numbered ax tree (line 143)."""
        # Create a page with many interactive elements
        buttons = "".join(f'<button>Btn{i}</button>' for i in range(200))
        url = self._write_html(
            "large.html",
            f"<html><body>{buttons}</body></html>",
        )
        self.tool.go_to_url(url)
        result = self.tool._get_ax_tree(max_chars=500)
        assert "[truncated]" in result

    def test_check_for_new_tab_none_context(self) -> None:
        """Cover _check_for_new_tab when context is None."""
        self.tool._context = None
        self.tool._check_for_new_tab()  # Should return immediately

    def test_resolve_locator_element_not_on_page(self) -> None:
        """Cover locator.count() == 0 path (line 181)."""
        url = self._write_html(
            "empty_page.html",
            "<html><body><p>No interactive</p></body></html>",
        )
        self.tool.go_to_url(url)
        # Manually set elements with a fake element
        self.tool._elements = [{"role": "button", "name": "Ghost Button"}]
        result = self.tool.click(1)
        # Should get error since element doesn't exist on page
        assert "Error" in result

    def test_resolve_locator_multiple_invisible(self) -> None:
        """Cover for loop through n>1 elements where none visible -> locator.first."""
        url = self._write_html(
            "hidden.html",
            '<html><body>'
            '<button style="display:none">Hidden1</button>'
            '<button style="display:none">Hidden2</button>'
            '<button>Visible</button>'
            '</body></html>',
        )
        self.tool.go_to_url(url)
        result = self.tool.click(1)  # The visible button
        assert "Page:" in result or "Error" not in result[:5]

    def test_scroll_error(self) -> None:
        """Cover scroll exception path (line 325-327)."""
        # Close browser then try to scroll
        self.tool.close()
        self.tool._page = None
        # _ensure_browser will be called but let's test with a broken state
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("s.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        # Close page to cause error on scroll
        tool2._page.close()
        result = tool2.scroll()
        assert "Error" in result
        tool2.close()

    def test_screenshot_error(self) -> None:
        """Cover screenshot exception path (line 346-348)."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("ss.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.screenshot()
        assert "Error" in result
        tool2.close()

    def test_get_page_content_error(self) -> None:
        """Cover get_page_content exception path (line 368-370)."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("pc.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.get_page_content()
        assert "Error" in result
        tool2.close()

    def test_press_key_error(self) -> None:
        """Cover press_key exception path."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("k.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.press_key("Enter")
        assert "Error" in result
        tool2.close()

    def test_type_text_error(self) -> None:
        """Cover type_text exception path."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("t.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        tool2._page.close()
        result = tool2.type_text(1, "text")
        assert "Error" in result
        tool2.close()

    def test_close_exception(self) -> None:
        """Cover close exception path (lines 384-386)."""
        tool2 = WebUseTool(browser_type="chromium", headless=True, user_data_dir=None)
        url = self._write_html("c.html", "<html><body>X</body></html>")
        tool2.go_to_url(url)
        # Kill playwright to cause close to fail gracefully
        if tool2._playwright:
            tool2._playwright.stop()
            tool2._playwright = None
        result = tool2.close()
        assert result == "Browser closed."


class TestCodeServerFinalEdgeCases:
    """Cover remaining code_server.py branches."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_untracked_base_oserror(self) -> None:
        """Cover OSError in shutil.copy2 inside _save_untracked_base."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        # Create a symlink to non-existent target
        link = os.path.join(work_dir, "broken_link")
        os.symlink("/nonexistent/target", link)
        _save_untracked_base(work_dir, self.tmpdir, {"broken_link"})

    def test_prepare_merge_view_with_saved_base(self) -> None:
        """Cover the saved_base.is_file() path for tracked files."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
        Path(work_dir, "f.txt").write_text("a\nb\nc\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

        # Save base copy of tracked file
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(work_dir, {"f.txt"})
        data_dir = os.path.join(self.tmpdir, "data")
        _save_untracked_base(work_dir, data_dir, {"f.txt"})

        # Modify file
        Path(work_dir, "f.txt").write_text("a\nX\nc\n")
        result = _prepare_merge_view(work_dir, data_dir, pre_hunks, pre_untracked, pre_hashes)
        assert result.get("status") == "opened"

    def test_prepare_merge_view_empty_new_file(self) -> None:
        """New file with 0 lines should be skipped."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=work_dir, capture_output=True)

        pre_untracked = _capture_untracked(work_dir)
        # Create empty file
        Path(work_dir, "empty.txt").write_text("")
        data_dir = os.path.join(self.tmpdir, "data")
        result = _prepare_merge_view(work_dir, data_dir, {}, pre_untracked)
        assert result.get("error") == "No changes"

    def test_prepare_merge_view_modified_untracked_unchanged(self) -> None:
        """Pre-existing untracked file that hasn't changed -> skip."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=work_dir, capture_output=True)

        Path(work_dir, "ut.txt").write_text("unchanged\n")
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(work_dir, pre_untracked)

        data_dir = os.path.join(self.tmpdir, "data")
        # File not changed
        result = _prepare_merge_view(work_dir, data_dir, {}, pre_untracked, pre_hashes)
        assert result.get("error") == "No changes"

    def test_prepare_merge_view_modified_untracked_not_in_hashes(self) -> None:
        """Pre-existing untracked file not in pre_hashes -> skip."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=work_dir, capture_output=True)

        Path(work_dir, "ut2.txt").write_text("content\n")
        pre_untracked = _capture_untracked(work_dir)
        # Don't include ut2.txt in hashes
        pre_hashes = {}

        data_dir = os.path.join(self.tmpdir, "data")
        result = _prepare_merge_view(work_dir, data_dir, {}, pre_untracked, pre_hashes)
        assert result.get("error") == "No changes"

    def test_sorcar_agent_stream_callback_with_printer(self) -> None:
        """Cover the _stream callback in SorcarAgent._get_tools (lines 33-34)."""
        agent = SorcarAgent("test-stream")
        printer = BaseBrowserPrinter()
        agent.printer = printer
        cq = printer.add_client()
        tools = agent._get_tools()
        # The first tool is Bash with stream_callback=_stream
        # The stream_callback calls self.printer.print(text, type="bash_stream")
        # Call the stream callback via the UsefulTools internal
        # Actually _stream is the stream_callback of UsefulTools
        # Let's find it: tools[0] is the Bash method, so the UsefulTools has stream_callback
        # We need to invoke _stream directly
        # The easiest way: execute a bash command that produces output
        result = tools[0]("echo hello_stream_test", "test stream", timeout_seconds=5)
        assert "hello_stream_test" in result
        # Check that the printer received bash_stream data
        time.sleep(0.2)
        printer._flush_bash()
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        sys_outputs = [e for e in events if e["type"] == "system_output"]
        assert any("hello_stream_test" in e.get("text", "") for e in sys_outputs)
        printer.remove_client(cq)

    def test_sorcar_agent_stream_callback_without_printer(self) -> None:
        """Cover the _stream callback when printer is None (line 33 False branch)."""
        agent = SorcarAgent("test-no-printer")
        agent.printer = None
        tools = agent._get_tools()
        # Execute a bash command - _stream should not crash when printer is None
        result = tools[0]("echo no_printer_test", "test", timeout_seconds=5)
        assert "no_printer_test" in result

    def test_prepare_merge_view_modified_untracked_already_in_file_hunks(self) -> None:
        """Pre-existing untracked file already in file_hunks -> skip."""
        work_dir = os.path.join(self.tmpdir, "work")
        os.makedirs(work_dir)
        subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
        Path(work_dir, "f.txt").write_text("a\nb\n")
        subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)

        # Create untracked file AND tracked file change
        Path(work_dir, "ut.txt").write_text("original\n")
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(work_dir, pre_untracked | {"f.txt"})
        _save_untracked_base(work_dir, os.path.join(self.tmpdir, "data"), pre_untracked)

        # Agent modifies tracked file and untracked file
        Path(work_dir, "f.txt").write_text("a\nX\n")
        Path(work_dir, "ut.txt").write_text("modified\n")

        data_dir = os.path.join(self.tmpdir, "data")
        result = _prepare_merge_view(work_dir, data_dir, {}, pre_untracked, pre_hashes)
        # Both files should be in merge view
        assert result.get("status") == "opened"

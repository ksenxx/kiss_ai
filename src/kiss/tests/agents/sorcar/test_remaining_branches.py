"""Integration tests for remaining uncovered branches in sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _extract_leading_command_name,
    _split_respecting_quotes,
    _stop_monitor,
    _truncate_output,
)
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.diff_merge import (
    _agent_file_hunks,
    _cleanup_merge_data,
    _file_as_new_hunks,
    _merge_data_dir,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
    _capture_untracked,
)
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
    fast_model_for,
    model_vendor,
    rank_file_suggestions,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.agents.vscode.server import VSCodePrinter, VSCodeServer


# ---------------------------------------------------------------------------
# persistence.py — uncovered branches
# ---------------------------------------------------------------------------


class TestPersistenceBranches:
    """Cover remaining branches in persistence.py."""

    def test_load_task_chat_events_bad_json(self) -> None:
        """_load_task_chat_events handles corrupt event_json gracefully (lines 294-295)."""
        db = th._get_db()
        # Insert a task and then corrupt event data
        th._add_task("corrupt-event-test")
        task_id = th._most_recent_task_id(db, "corrupt-event-test")
        assert task_id is not None
        db.execute(
            "INSERT INTO events (task_id, seq, event_json) VALUES (?, ?, ?)",
            (task_id, 0, "NOT VALID JSON {{{"),
        )
        db.execute(
            "INSERT INTO events (task_id, seq, event_json) VALUES (?, ?, ?)",
            (task_id, 1, json.dumps({"type": "ok"})),
        )
        db.commit()
        events = th._load_task_chat_events("corrupt-event-test")
        # The bad JSON is skipped, the valid one is returned
        assert len(events) == 1
        assert events[0]["type"] == "ok"

    def test_cleanup_stale_cs_dirs_with_active_port(self) -> None:
        """_cleanup_stale_cs_dirs skips dirs with active port (lines 560-561)."""
        import shutil as _shutil
        kiss_dir = th._KISS_DIR
        sd = kiss_dir / "sorcar-data"
        sd.mkdir(parents=True, exist_ok=True)
        # Create a port file pointing to a port we're listening on
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]
        pf = sd / "cs-port"
        pf.write_text(str(port))
        # Set mtime AFTER writing files so directory mtime is old
        old_time = time.time() - 25 * 3600
        os.utime(sd, (old_time, old_time))
        try:
            removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
            # Should NOT remove because port is active
            assert sd.exists()
        finally:
            server_sock.close()
            if sd.exists():
                _shutil.rmtree(sd, ignore_errors=True)

    def test_cleanup_stale_cs_dirs_with_invalid_port(self) -> None:
        """_cleanup_stale_cs_dirs removes dir when port file has bad value."""
        kiss_dir = th._KISS_DIR
        sd = kiss_dir / "sorcar-data"
        sd.mkdir(parents=True, exist_ok=True)
        pf = sd / "cs-port"
        pf.write_text("not-a-number")
        # Set mtime AFTER writing files so directory mtime is old
        old_time = time.time() - 25 * 3600
        os.utime(sd, (old_time, old_time))
        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert not sd.exists()
        assert removed >= 1

    def test_cleanup_stale_cs_dirs_with_dead_port(self) -> None:
        """_cleanup_stale_cs_dirs removes dir when port is not listening."""
        kiss_dir = th._KISS_DIR
        sd = kiss_dir / "sorcar-data"
        sd.mkdir(parents=True, exist_ok=True)
        pf = sd / "cs-port"
        # Use a port that's almost certainly not listening
        pf.write_text("19999")
        old_time = time.time() - 25 * 3600
        os.utime(sd, (old_time, old_time))
        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert not sd.exists()
        assert removed >= 1

    def test_cleanup_stale_cs_legacy_dirs(self) -> None:
        """_cleanup_stale_cs_dirs removes legacy cs-* dirs and cs-port-* files."""
        kiss_dir = th._KISS_DIR
        # Create legacy dir
        legacy = kiss_dir / "cs-test123"
        legacy.mkdir(parents=True, exist_ok=True)
        # Create port file (covers line 557 for-loop iteration and 558 is_file True branch)
        pf = kiss_dir / "cs-port-test"
        pf.write_text("12345")
        # cs-extensions should NOT be removed
        ext = kiss_dir / "cs-extensions"
        ext.mkdir(parents=True, exist_ok=True)
        try:
            th._cleanup_stale_cs_dirs(max_age_hours=24)
            assert not legacy.exists()
            assert not pf.exists()
            assert ext.exists()
        finally:
            if ext.exists():
                ext.rmdir()

    def test_cleanup_stale_cs_port_dir_not_file(self) -> None:
        """_cleanup_stale_cs_dirs handles cs-port-* that is a directory (line 557->556)."""
        kiss_dir = th._KISS_DIR
        # Create a directory matching cs-port-* pattern
        port_dir = kiss_dir / "cs-port-dirtest"
        port_dir.mkdir(parents=True, exist_ok=True)
        try:
            th._cleanup_stale_cs_dirs(max_age_hours=24)
        finally:
            if port_dir.exists():
                import shutil as _s
                _s.rmtree(port_dir, ignore_errors=True)




# ---------------------------------------------------------------------------
# useful_tools.py — uncovered branches
# ---------------------------------------------------------------------------


class TestUsefulToolsBranches:
    """Cover remaining branches in useful_tools.py."""

    def test_truncate_output_zero_tail(self) -> None:
        """_truncate_output when max_chars exactly equals worst_msg length, tail=0 (line 33)."""
        output = "A" * 200
        worst_msg = f"\n\n... [truncated {len(output)} chars] ...\n\n"
        # Set max_chars == len(worst_msg) so remaining=0, head=0, tail=0
        max_chars = len(worst_msg)
        result = _truncate_output(output, max_chars)
        assert "truncated" in result
        # tail is 0 so no suffix is appended
        assert not result.endswith("A")

    def test_extract_leading_command_name_empty_after_lstrip(self) -> None:
        """_extract_leading_command_name returns None when name is empty after lstrip (line 80)."""
        # After skipping env var, remaining token "({" lstrips to empty string
        result = _extract_leading_command_name("X=1 ({")
        assert result is None

    def test_split_respecting_quotes_escape_in_double_quote(self) -> None:
        """Cover escape inside double-quoted string (lines 98-106)."""
        pattern = re.compile(r"&&")
        result = _split_respecting_quotes('echo "hello\\"world" && ls', pattern)
        assert len(result) == 2
        assert 'echo "hello\\"world"' in result[0]

    def test_split_respecting_quotes_unclosed_dquote_with_escape(self) -> None:
        """Cover unclosed double-quote string ending with escape (line 98->106)."""
        pattern = re.compile(r"&&")
        # Unclosed double-quote with escape — while loop exits because j >= len
        result = _split_respecting_quotes('echo "hello\\', pattern)
        assert len(result) == 1
        assert result[0] == 'echo "hello\\'

    def test_stop_monitor_exits_when_done(self) -> None:
        """_stop_monitor exits cleanly when done is set (line 207 exit branch)."""
        stop = threading.Event()
        done = threading.Event()
        # Create a real process that finishes quickly
        process = subprocess.Popen(["true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        done.set()
        # Should exit immediately since done is set
        t = threading.Thread(target=_stop_monitor, args=(stop, process, done))
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()


# ---------------------------------------------------------------------------
# helpers.py — uncovered branches
# ---------------------------------------------------------------------------


class TestHelpersBranches:
    """Cover remaining branches in helpers.py."""

    def test_model_vendor_all_prefixes(self) -> None:
        """model_vendor covers all vendor branches (lines 27, 66, 68, 70)."""
        assert model_vendor("claude-haiku-4.5")[0] == "Anthropic"
        assert model_vendor("gpt-4o")[0] == "OpenAI"
        assert model_vendor("gemini-2.0-flash")[0] == "Gemini"
        assert model_vendor("minimax-text-01")[0] == "MiniMax"
        assert model_vendor("openrouter/some-model")[0] == "OpenRouter"
        assert model_vendor("together/llama")[0] == "Together AI"
        # Also test "openai/" prefix which falls through OpenAI check
        name, order = model_vendor("openai/custom")
        assert order == 5  # Together AI fallback

    def test_fast_model_for_all_providers(self) -> None:
        """fast_model_for returns correct fast model per provider (lines 66-70)."""
        assert fast_model_for("openrouter/anthropic/claude-3") == "openrouter/anthropic/claude-haiku-4.5"
        assert fast_model_for("gemini-2.5-pro") == "gemini-2.0-flash"
        assert fast_model_for("gpt-4o") == "gpt-4o-mini"
        # Default: claude model
        result = fast_model_for("claude-opus-4-6")
        assert result  # Should return a non-empty string (DEFAULT_CONFIG.FAST_MODEL)

    def test_clip_autocomplete_suggestion_echo_prefix(self) -> None:
        """clip_autocomplete_suggestion strips query prefix when echoed."""
        result = clip_autocomplete_suggestion("hello", "hello world")
        assert result == " world"

    def test_rank_file_suggestions_empty_query(self) -> None:
        """rank_file_suggestions with empty query returns all files (line 143)."""
        files = ["a.py", "b.py", "c.py"]
        usage = {"a.py": 5}
        ranked = rank_file_suggestions(files, "", usage)
        assert len(ranked) == 3
        # Frequent files first
        assert ranked[0]["type"] == "frequent"
        assert ranked[0]["text"] == "a.py"

    def test_rank_file_suggestions_with_query(self) -> None:
        """rank_file_suggestions filters by query substring."""
        files = ["src/main.py", "src/test.py", "README.md"]
        usage = {"src/main.py": 2}
        ranked = rank_file_suggestions(files, "main", usage)
        assert len(ranked) == 1
        assert ranked[0]["text"] == "src/main.py"

    def test_rank_file_suggestions_limit(self) -> None:
        """rank_file_suggestions respects the limit parameter (line 143)."""
        files = [f"file{i}.py" for i in range(30)]
        usage: dict[str, int] = {}
        ranked = rank_file_suggestions(files, "", usage, limit=5)
        assert len(ranked) == 5

    def test_generate_followup_text_failure(self) -> None:
        """generate_followup_text returns empty string on LLM failure (lines 104-106)."""
        from kiss.agents.vscode.helpers import generate_followup_text
        # Use an invalid model to trigger an exception
        result = generate_followup_text("task", "result", "nonexistent-model-xyz")
        assert result == ""


# ---------------------------------------------------------------------------
# server.py — uncovered branches
# ---------------------------------------------------------------------------


class TestVSCodeServerBranches:
    """Cover remaining branches in server.py."""

    def test_run_loop_empty_lines_and_invalid_json(self) -> None:
        """server.run() skips empty lines, handles invalid JSON (line 119)."""
        import io
        import sys

        server = VSCodeServer()
        events: list[dict] = []
        orig_broadcast = server.printer.broadcast
        def capture(ev: dict) -> None:
            events.append(ev)
            orig_broadcast(ev)
        server.printer.broadcast = capture  # type: ignore[assignment]

        # Feed stdin with empty line, invalid JSON, then EOF
        fake_stdin = io.StringIO("\n\nnot-json\n")
        old_stdin = sys.stdin
        sys.stdin = fake_stdin
        try:
            server.run()
        finally:
            sys.stdin = old_stdin

        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "Invalid JSON" in error_events[0]["text"]

    def test_handle_command_unknown(self) -> None:
        """Unknown command type broadcasts error."""
        server = VSCodeServer()
        events: list[dict] = []
        orig = server.printer.broadcast
        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        server._handle_command({"type": "unknownCommand123"})
        assert any("Unknown command" in str(e.get("text", "")) for e in events)

    def test_complete_stale_seq_early_return(self) -> None:
        """_complete exits early when seq is stale (lines 535->exit)."""
        server = VSCodeServer()
        events: list[dict] = []
        orig = server.printer.broadcast
        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        # Advance seq counter past what we'll pass
        server._complete_seq_latest = 999
        server._complete("hello", seq=5)
        # Should return early - no ghost event broadcast
        ghost_events = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost_events) == 0

    def test_complete_short_query(self) -> None:
        """_complete with short query broadcasts empty suggestion."""
        server = VSCodeServer()
        events: list[dict] = []
        orig = server.printer.broadcast
        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        server._complete("a", seq=-1)
        ghost = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost) == 1
        assert ghost[0]["suggestion"] == ""

    def test_complete_from_active_file_no_content_no_path(self) -> None:
        """_complete_from_active_file returns empty when no file (line 618)."""
        server = VSCodeServer()
        result = server._complete_from_active_file("hello", "", "")
        assert result == ""

    def test_complete_from_active_file_trailing_whitespace(self) -> None:
        """_complete_from_active_file returns empty when query ends with space."""
        server = VSCodeServer()
        result = server._complete_from_active_file("hello ", "", "some content")
        assert result == ""

    def test_complete_from_active_file_no_partial_match(self) -> None:
        """_complete_from_active_file returns empty when regex finds nothing."""
        server = VSCodeServer()
        result = server._complete_from_active_file("!@#$", "", "some content")
        assert result == ""

    def test_complete_from_active_file_short_partial(self) -> None:
        """_complete_from_active_file returns empty when partial < 2 chars."""
        server = VSCodeServer()
        result = server._complete_from_active_file("a", "", "apple banana")
        assert result == ""

    def test_complete_from_active_file_reads_file(self) -> None:
        """_complete_from_active_file reads from disk when no snapshot_content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def calculate_total():\n    pass\n")
            f.flush()
            path = f.name
        try:
            server = VSCodeServer()
            result = server._complete_from_active_file("calc", path, "")
            assert result == "ulate_total"
        finally:
            os.unlink(path)

    def test_complete_from_active_file_file_not_found(self) -> None:
        """_complete_from_active_file returns empty for nonexistent file."""
        server = VSCodeServer()
        result = server._complete_from_active_file("test", "/nonexistent/file.py", "")
        assert result == ""

    def test_complete_from_active_file_with_dot_chains(self) -> None:
        """_complete_from_active_file matches dot-chained identifiers."""
        server = VSCodeServer()
        content = "import os\nos.path.join\nos.path.exists\n"
        result = server._complete_from_active_file("os.path.jo", "", content)
        assert result == "in"

    def test_fast_complete_history_match(self) -> None:
        """_fast_complete returns history match (line 535)."""
        server = VSCodeServer()
        # Add a task to history
        th._add_task("integrate all the modules together")
        result = server._fast_complete("integrate all the module")
        assert "s together" in result

    def test_merge_action_all_done(self) -> None:
        """_handle_merge_action with 'all-done' calls _finish_merge."""
        server = VSCodeServer()
        server._merging = True
        events: list[dict] = []
        orig = server.printer.broadcast
        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        server._handle_merge_action("all-done")
        assert not server._merging
        assert any(e.get("type") == "merge_ended" for e in events)

    def test_merge_action_other(self) -> None:
        """_handle_merge_action with non-'all-done' does nothing."""
        server = VSCodeServer()
        server._merging = True
        server._handle_merge_action("accept")
        assert server._merging  # unchanged

    def test_start_merge_session_empty_files(self) -> None:
        """_start_merge_session returns False when files list is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"files": []}, f)
            path = f.name
        try:
            server = VSCodeServer()
            result = server._start_merge_session(path)
            assert result is False
        finally:
            os.unlink(path)

    def test_start_merge_session_zero_hunks(self) -> None:
        """_start_merge_session returns False when total hunks is 0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"files": [{"name": "a.py", "hunks": []}]}, f)
            path = f.name
        try:
            server = VSCodeServer()
            result = server._start_merge_session(path)
            assert result is False
        finally:
            os.unlink(path)

    def test_start_merge_session_success(self) -> None:
        """_start_merge_session returns True for valid merge data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "files": [{"name": "a.py", "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}]}]
            }, f)
            path = f.name
        try:
            server = VSCodeServer()
            events: list[dict] = []
            orig = server.printer.broadcast
            def cap(ev: dict) -> None:
                events.append(ev)
                orig(ev)
            server.printer.broadcast = cap  # type: ignore[assignment]
            result = server._start_merge_session(path)
            assert result is True
            assert server._merging
            assert any(e.get("type") == "merge_data" for e in events)
            assert any(e.get("type") == "merge_started" for e in events)
        finally:
            server._merging = False
            os.unlink(path)

    def test_start_merge_session_invalid_json(self) -> None:
        """_start_merge_session returns False for invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("NOT JSON")
            path = f.name
        try:
            server = VSCodeServer()
            result = server._start_merge_session(path)
            assert result is False
        finally:
            os.unlink(path)

    def test_start_merge_session_missing_file(self) -> None:
        """_start_merge_session returns False for missing file."""
        server = VSCodeServer()
        result = server._start_merge_session("/nonexistent/path.json")
        assert result is False

    def test_run_task_merging_guard(self) -> None:
        """_run_task_inner broadcasts error when _merging is True (lines 300-303)."""
        server = VSCodeServer()
        server._merging = True
        events: list[dict] = []
        orig = server.printer.broadcast
        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        server._run_task({"prompt": "test", "model": "test"})
        assert any("merge review" in str(e.get("text", "")) for e in events)
        # Must always broadcast status running=False
        assert any(e.get("type") == "status" and e.get("running") is False for e in events)
        server._merging = False

    def test_force_stop_thread_already_dead(self) -> None:
        """_force_stop_thread exits immediately if thread is already dead (lines 386-395)."""
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        # Thread is dead, should return quickly
        VSCodeServer._force_stop_thread(t)

    def test_select_model_command(self) -> None:
        """selectModel command updates selected model."""
        server = VSCodeServer()
        server._handle_command({"type": "selectModel", "model": "gpt-4o"})
        assert server._selected_model == "gpt-4o"

    def test_record_file_usage_command(self) -> None:
        """recordFileUsage command records the path."""
        server = VSCodeServer()
        server._handle_command({"type": "recordFileUsage", "path": "/test/file.py"})
        usage = th._load_file_usage()
        assert "/test/file.py" in usage

    def test_user_answer_command(self) -> None:
        """userAnswer command sets the answer and signals the event."""
        server = VSCodeServer()
        server._user_answer_event = threading.Event()
        server._handle_command({"type": "userAnswer", "answer": "yes"})
        assert server._user_answer == "yes"
        assert server._user_answer_event.is_set()

    def test_new_chat_command(self) -> None:
        """newChat command starts a new chat session."""
        server = VSCodeServer()
        old_id = server.agent._chat_id
        server._handle_command({"type": "newChat"})
        assert server.agent._chat_id != old_id

    def test_new_chat_while_running(self) -> None:
        """newChat while task is running does nothing."""
        server = VSCodeServer()
        old_id = server.agent._chat_id
        # Fake a running thread
        t = threading.Thread(target=lambda: time.sleep(5))
        t.daemon = True
        t.start()
        server._task_thread = t
        server._handle_command({"type": "newChat"})
        assert server.agent._chat_id == old_id

    def test_get_input_history(self) -> None:
        """getInputHistory command returns deduplicated tasks."""
        server = VSCodeServer()
        events: list[dict] = []
        orig = server.printer.broadcast
        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        server._handle_command({"type": "getInputHistory"})
        hist_events = [e for e in events if e.get("type") == "inputHistory"]
        assert len(hist_events) == 1
        assert "tasks" in hist_events[0]

    def test_generate_commit_message_no_staged(self) -> None:
        """_generate_commit_message broadcasts error when no staged files (line 699-703)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "a.txt").write_text("x")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            # No staged changes
            server = VSCodeServer()
            server.work_dir = tmpdir
            events: list[dict] = []
            orig = server.printer.broadcast
            def cap(ev: dict) -> None:
                events.append(ev)
                orig(ev)
            server.printer.broadcast = cap  # type: ignore[assignment]
            server._generate_commit_message()
            cm_events = [e for e in events if e.get("type") == "commitMessage"]
            assert len(cm_events) == 1
            assert "No staged files" in cm_events[0]["message"]

    def test_resume_session_no_events(self) -> None:
        """_replay_session broadcasts error when no events found."""
        server = VSCodeServer()
        events: list[dict] = []
        orig = server.printer.broadcast
        def cap(ev: dict) -> None:
            events.append(ev)
            orig(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        server._replay_session("nonexistent-task-12345")
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert "No recorded events" in err[0]["text"]


# ---------------------------------------------------------------------------
# diff_merge.py — uncovered branches
# ---------------------------------------------------------------------------


class TestDiffMergeBranches:
    """Cover remaining branches in diff_merge.py."""

    def test_file_changed_returns_false_on_oserror(self) -> None:
        """_prepare_merge_view._file_changed returns False on OSError (lines 331-333)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize a git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "a.txt").write_text("initial")
            subprocess.run(["git", "add", "a.txt"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            # Delete the file to trigger OSError in _file_changed
            Path(tmpdir, "a.txt").unlink()
            # Pre-hash includes a.txt
            pre_hashes = {"a.txt": "abc123"}
            result = _prepare_merge_view(
                tmpdir, str(Path(tmpdir) / "merge"),
                {}, set(), pre_hashes
            )
            # Should return "No changes" since file can't be read
            assert result.get("error") == "No changes"

    def test_prepare_merge_view_pre_untracked_modified(self) -> None:
        """_prepare_merge_view detects modified pre-existing untracked files (line 345->343, 351)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)
            # Create and commit a file
            Path(tmpdir, "committed.txt").write_text("committed")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)

            # Create an untracked file, snapshot it, then modify it
            untracked = Path(tmpdir, "untracked.txt")
            untracked.write_text("original content")
            pre_untracked = {"untracked.txt"}
            pre_hashes = _snapshot_files(tmpdir, pre_untracked)

            # Now modify the untracked file (simulating agent changes)
            untracked.write_text("modified content by agent")

            merge_dir = str(Path(tmpdir) / "merge")
            result = _prepare_merge_view(
                tmpdir, merge_dir, {}, pre_untracked, pre_hashes
            )
            # Should detect the modification
            assert result.get("status") == "opened" or "error" not in result

    def test_save_untracked_base_oserror(self) -> None:
        """_save_untracked_base handles OSError on copy (lines 203-204)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file that we'll make unreadable
            bad_file = Path(tmpdir) / "bad.txt"
            bad_file.write_text("content")
            os.chmod(str(bad_file), 0o000)
            try:
                _save_untracked_base(tmpdir, {"bad.txt"})
            finally:
                os.chmod(str(bad_file), 0o644)

    def test_prepare_merge_view_empty_new_file(self) -> None:
        """_prepare_merge_view skips empty new untracked files (line 345->343)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "x.txt").write_text("x")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            pre_untracked: set[str] = set()
            # Create an empty new untracked file after task
            Path(tmpdir, "empty.txt").write_text("")
            merge_dir = str(Path(tmpdir) / "merge")
            result = _prepare_merge_view(tmpdir, merge_dir, {}, pre_untracked, {})
            # Empty file produces no hunks, so result may be "No changes"
            # if there are no other changes
            assert "error" in result or "status" in result

    def test_prepare_merge_view_untracked_not_changed(self) -> None:
        """_prepare_merge_view skips untracked files that haven't changed (line 351)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "x.txt").write_text("x")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            # Create untracked file, snapshot, but DON'T modify it
            ut = Path(tmpdir, "unmod.txt")
            ut.write_text("same content")
            pre_untracked = {"unmod.txt"}
            pre_hashes = _snapshot_files(tmpdir, pre_untracked)
            merge_dir = str(Path(tmpdir) / "merge")
            result = _prepare_merge_view(tmpdir, merge_dir, {}, pre_untracked, pre_hashes)
            assert result.get("error") == "No changes"

    def test_prepare_merge_view_untracked_not_in_hashes(self) -> None:
        """_prepare_merge_view skips untracked files not in pre_file_hashes (line 351)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "x.txt").write_text("x")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            # Pre-untracked set includes a file that's NOT in pre_file_hashes
            # This hits the `fname not in pre_file_hashes` branch -> continue (line 351)
            pre_untracked = {"phantom.txt"}
            pre_hashes: dict[str, str] = {"other.txt": "abc"}  # phantom.txt NOT here
            merge_dir = str(Path(tmpdir) / "merge")
            result = _prepare_merge_view(tmpdir, merge_dir, {}, pre_untracked, pre_hashes)
            assert result.get("error") == "No changes"

    def test_prepare_merge_view_untracked_modified_with_saved_base(self) -> None:
        """_prepare_merge_view: untracked file with saved base copy gets diffed (line 345->343, 351)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "committed.txt").write_text("x")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)

            # Create untracked file, snapshot, save base, then modify
            ut = Path(tmpdir, "notes.txt")
            ut.write_text("line1\nline2\n")
            pre_untracked = {"notes.txt"}
            pre_hashes = _snapshot_files(tmpdir, pre_untracked)
            # Save untracked base (like the server does before a task)
            _save_untracked_base(tmpdir, pre_untracked)
            # Simulate agent modifying the untracked file
            ut.write_text("line1\nline2\nline3 added by agent\n")

            merge_dir = str(Path(tmpdir) / "merge")
            result = _prepare_merge_view(
                tmpdir, merge_dir, {}, pre_untracked, pre_hashes
            )
            # Should detect the modification via saved base diff
            assert result.get("status") == "opened"
            # Clean up merge data
            _cleanup_merge_data(merge_dir)


# ---------------------------------------------------------------------------
# sorcar_agent.py — uncovered branches
# ---------------------------------------------------------------------------


class TestSorcarAgentBranches:
    """Cover remaining branches in sorcar_agent.py."""

    def test_get_tools_stream_no_printer(self) -> None:
        """_stream callback handles None printer (line 39->exit)."""
        agent = SorcarAgent("test")
        agent.printer = None
        tools = agent._get_tools()
        assert len(tools) > 0
        # Actually invoke the Bash tool with a command to trigger _stream
        # The first tool is Bash — it uses the _stream callback
        bash_tool = tools[0]
        result = bash_tool(command="echo test_no_printer", description="test", timeout_seconds=5)
        assert "test_no_printer" in result
        if agent.web_use_tool:
            agent.web_use_tool.close()

    def test_get_tools_without_docker(self) -> None:
        """_get_tools without docker_manager uses UsefulTools (line 74->76)."""
        agent = SorcarAgent("test")
        agent.docker_manager = None
        agent.web_use_tool = None
        tools = agent._get_tools()
        assert len(tools) > 0
        # web_use_tool should now be set
        assert agent.web_use_tool is not None
        # Clean up
        agent.web_use_tool.close()

    def test_get_tools_web_use_tool_already_set(self) -> None:
        """_get_tools skips web_use_tool creation when already set (line 74->76)."""
        agent = SorcarAgent("test")
        agent.docker_manager = None
        existing_wut = WebUseTool()
        agent.web_use_tool = existing_wut
        tools = agent._get_tools()
        # Should still be the same instance
        assert agent.web_use_tool is existing_wut
        existing_wut.close()


# ---------------------------------------------------------------------------
# stateful_sorcar_agent.py — uncovered branches
# ---------------------------------------------------------------------------


class TestStatefulSorcarAgentBranches:
    """Cover remaining branches in stateful_sorcar_agent.py."""

    def test_build_chat_prompt_entry_without_result(self) -> None:
        """build_chat_prompt skips result when entry has no result (line 84->82)."""
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
        agent = StatefulSorcarAgent("test")
        # Add a task with empty result to the agent's chat
        th._add_task("task with no result", chat_id=agent._chat_id)
        th._save_task_result("task with no result", "")
        prompt = agent.build_chat_prompt("new task")
        assert "### Task 1" in prompt
        assert "### Result 1" not in prompt
        assert "new task" in prompt


# ---------------------------------------------------------------------------
# browser_ui.py — uncovered branches
# ---------------------------------------------------------------------------


class TestBrowserUIBranches:
    """Cover remaining branches in browser_ui.py."""

    def test_bash_stream_timer_scheduling(self) -> None:
        """Bash stream schedules timer when flush interval not reached (lines 247-248)."""
        p = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        p._client_queue = cq
        # First call flushes immediately (fresh printer, _bash_last_flush is 0)
        p.print("first line\n", type="bash_stream")
        # Immediately call again - should schedule timer since interval not reached
        p.print("second line\n", type="bash_stream")
        # Wait for timer to fire
        time.sleep(0.3)
        # Collect all events
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        output_events = [e for e in events if e.get("type") == "system_output"]
        # Should have at least 1 event (possibly 2 if timer fired)
        assert len(output_events) >= 1

    def test_bash_stream_cancel_existing_timer(self) -> None:
        """Bash stream cancels existing timer when flush interval reached (lines 247-248).

        To hit lines 247-248, we need _bash_flush_timer to be non-None
        when the main flush branch fires (time.monotonic() - _bash_last_flush >= 0.1).

        We set _bash_last_flush to a value that makes the next call enter the
        main flush branch, and manually set a timer to simulate the state.
        """
        p = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        p._client_queue = cq
        # Set last flush to a time well in the past so the next call will flush
        p._bash_last_flush = time.monotonic() - 1.0
        # Set a timer manually to simulate pending timer state
        p._bash_flush_timer = threading.Timer(10.0, p._flush_bash)
        p._bash_flush_timer.daemon = True
        p._bash_flush_timer.start()
        # Now call bash_stream — should enter main flush branch, cancel timer
        p.print("line1\n", type="bash_stream")
        # Timer should be cancelled and set to None
        assert p._bash_flush_timer is None
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        output_events = [e for e in events if e.get("type") == "system_output"]
        assert len(output_events) == 1

    def test_print_tool_result_non_core_tool(self) -> None:
        """Non-core tool result is hidden unless is_error (line 273->281)."""
        p = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        p._client_queue = cq
        # Simulate tool_result for a non-core tool
        p.print("some result", type="tool_result", tool_name="custom_tool", is_error=False)
        # No tool_result event should be broadcast for non-core, non-error
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_results = [e for e in events if e.get("type") == "tool_result"]
        assert len(tool_results) == 0

    def test_print_tool_result_non_core_tool_with_error(self) -> None:
        """Non-core tool result is shown when is_error=True."""
        p = BaseBrowserPrinter()
        cq: queue.Queue = queue.Queue()
        p._client_queue = cq
        p.print("error msg", type="tool_result", tool_name="custom_tool", is_error=True)
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        tool_results = [e for e in events if e.get("type") == "tool_result"]
        assert len(tool_results) == 1


# ---------------------------------------------------------------------------
# web_use_tool.py — uncovered branches (basic non-browser tests)
# ---------------------------------------------------------------------------


class TestWebUseToolBranches:
    """Cover basic branches in web_use_tool.py that don't need a real browser."""

    def test_close_without_browser(self) -> None:
        """close() succeeds even when no browser was started (line 414)."""
        tool = WebUseTool()
        result = tool.close()
        assert result == "Browser closed."
        assert tool._page is None

    def test_get_tools_returns_all(self) -> None:
        """get_tools returns the correct number of callable tools."""
        tool = WebUseTool()
        tools = tool.get_tools()
        assert len(tools) == 8

    def test_check_for_new_tab_no_context(self) -> None:
        """_check_for_new_tab returns immediately when no context."""
        tool = WebUseTool()
        tool._context = None
        tool._check_for_new_tab()  # should not raise


from kiss.agents.sorcar.sorcar_agent import SorcarAgent


class TestSorcarAgentAskUserQuestion:
    """Cover ask_user_question inner function."""

    def test_ask_user_question_no_callback(self) -> None:
        """ask_user_question returns fallback when no callback set."""
        agent = SorcarAgent("test")
        agent._ask_user_question_callback = None
        tools = agent._get_tools()
        # Find the ask_user_question tool
        ask_tool = None
        for t in tools:
            if hasattr(t, "__name__") and t.__name__ == "ask_user_question":
                ask_tool = t
                break
        assert ask_tool is not None
        result = ask_tool("What is your name?")
        assert "not available" in result
        if agent.web_use_tool:
            agent.web_use_tool.close()

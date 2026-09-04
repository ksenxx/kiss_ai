# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for remaining uncovered branches in sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.server.helpers import (
    clip_autocomplete_suggestion,
)
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _git(tmpdir: str, *args: str) -> None:
    """Run a git command in tmpdir, suppressing output."""
    subprocess.run(["git", *args], cwd=tmpdir, capture_output=True, check=True)


def _file_suffix(
    server: VSCodeServer,
    query: str,
    snapshot_file: str = "",
    snapshot_content: str | None = None,
    chat_id: str = "",
) -> str:
    """Longest identifier-completion suffix for *query*.

    Exercises ``_active_file_identifier_matches`` (the production
    identifier harvester behind the fast-complete dropdown) the way
    the ghost-text pipeline consumes it: the longest-first match list
    is reduced to the top match's remaining suffix.
    """
    matches = server._active_file_identifier_matches(
        query, snapshot_file, snapshot_content, chat_id,
    )
    if not matches:
        return ""
    m = re.search(r"([\w][\w.]*)$", query)
    assert m is not None
    return matches[0][len(m.group(1)):]


class TestHelpersBranches:
    """Cover remaining branches in helpers.py."""

    def test_clip_autocomplete_suggestion_keeps_suffix_prefix(self) -> None:
        """A suffix that itself begins with the query is NOT re-stripped.

        Suggestions are always continuation suffixes (the call sites
        strip the query before calling), so a suffix starting with the
        query text — e.g. completing ``hellohello world`` after typing
        ``hello`` — must survive intact.
        """
        result = clip_autocomplete_suggestion("hello", "hello world")
        assert result == "hello world"


class TestVSCodeServerBranches:
    """Cover remaining branches in server.py."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache()

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        th._invalidate_chat_context_cache()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

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

    def test_file_suffix_trailing_whitespace(self) -> None:
        """_file_suffix returns empty when query ends with space."""
        server = VSCodeServer()
        result = _file_suffix(server, "hello ", "", "some content")
        assert result == ""

    def test_file_suffix_no_partial_match(self) -> None:
        """_file_suffix returns empty when regex finds nothing."""
        server = VSCodeServer()
        result = _file_suffix(server, "!@#$", "", "some content")
        assert result == ""

    def test_file_suffix_short_partial(self) -> None:
        """_file_suffix returns empty when partial < 2 chars."""
        server = VSCodeServer()
        result = _file_suffix(server, "a", "", "apple banana")
        assert result == ""

    def test_file_suffix_reads_file(self) -> None:
        """_file_suffix reads from disk when no snapshot_content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def calculate_total():\n    pass\n")
            f.flush()
            path = f.name
        try:
            server = VSCodeServer()
            result = _file_suffix(server, "calc", path, None)
            assert result == "ulate_total"
        finally:
            os.unlink(path)

    def test_file_suffix_file_not_found(self) -> None:
        """_file_suffix returns empty for nonexistent file."""
        server = VSCodeServer()
        result = _file_suffix(server, "test", "/nonexistent/file.py", "")
        assert result == ""

    def test_file_suffix_uses_chat_history(self) -> None:
        """_file_suffix harvests identifiers from prior chat tasks."""
        task_id, chat_id = th._add_task("first task with calculate_total_amount usage")
        th._save_task_result(
            "the result mentions parse_xml_payload again", task_id=task_id,
        )
        server = VSCodeServer()
        assert _file_suffix(server,
            "calc", "", "", chat_id,
        ) == "ulate_total_amount"
        assert _file_suffix(server,
            "parse_xml", "", "", chat_id,
        ) == "_payload"
        assert _file_suffix(server, "calc", "", "") == ""

    def test_file_suffix_combines_file_and_chat(self) -> None:
        """File content and chat history both contribute candidates."""
        _task_id, chat_id = th._add_task("chat had wonderful_widget_factory in it")
        server = VSCodeServer()
        file_content = "class HelperUtil:\n    pass\n"
        assert _file_suffix(server,
            "Help", "", file_content, chat_id,
        ) == "erUtil"
        assert _file_suffix(server,
            "wonderful", "", file_content, chat_id,
        ) == "_widget_factory"

    def test_file_suffix_caches_chat_context(self) -> None:
        """Chat-context text is cached between keystrokes in the same chat."""
        task_id, chat_id = th._add_task("first wonderful_alpha_token here")
        th._save_task_result("nothing useful", task_id=task_id)
        server = VSCodeServer()

        assert _file_suffix(server,
            "wonderful_a", "", "", chat_id,
        ) == "lpha_token"

        db = th._get_db()
        db.execute(
            "UPDATE task_history SET task = ? WHERE id = ?",
            ("first beta_zero_marker different", task_id),
        )
        db.commit()
        assert _file_suffix(server,
            "wonderful_a", "", "", chat_id,
        ) == "lpha_token"

        th._invalidate_chat_context_cache(chat_id)
        assert _file_suffix(server,
            "beta_zero", "", "", chat_id,
        ) == "_marker"
        assert _file_suffix(server,
            "wonderful_a", "", "", chat_id,
        ) == ""

        th._save_task_result(
            "gamma_three_signal appears now", task_id=task_id,
        )
        assert _file_suffix(server,
            "gamma_three", "", "", chat_id,
        ) == "_signal"

        th._add_task("delta_four_indicator was added", chat_id=chat_id)
        assert _file_suffix(server,
            "delta_four", "", "", chat_id,
        ) == "_indicator"

    def test_fast_complete_history_match(self) -> None:
        """_complete returns history match via broadcast."""
        server = VSCodeServer()
        events: list[dict] = []  # type: ignore[type-arg]
        def cap(ev: dict) -> None:  # type: ignore[type-arg]
            events.append(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        th._add_task("integrate all the modules together")
        server._complete("integrate all the module")
        ghost = [e for e in events if e.get("type") == "ghost"]
        assert len(ghost) == 1
        assert "s together" in ghost[0]["suggestion"]

    def test_record_file_usage_command(self) -> None:
        """recordFileUsage command records the path."""
        server = VSCodeServer()
        server._handle_command({"type": "recordFileUsage", "path": "/test/file.py"})
        usage = th._load_file_usage()
        assert "/test/file.py" in usage

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

    def test_get_input_history_deduplicates_across_full_history(self) -> None:
        """Deduplication should keep the newest copy even when duplicates span >100 rows."""
        server = VSCodeServer()
        events: list[dict] = []

        def cap(ev: dict) -> None:
            events.append(ev)

        server.printer.broadcast = cap  # type: ignore[assignment]
        th._add_task("repeated-task")
        for i in range(100):
            th._add_task(f"middle-task-{i:03d}")
        th._add_task("repeated-task")

        server._get_input_history()

        hist_event = next(e for e in events if e.get("type") == "inputHistory")
        tasks = hist_event["tasks"]
        assert tasks.count("repeated-task") == 1
        assert tasks[0] == "repeated-task"
        assert "middle-task-000" in tasks


class TestBrowserUIBranches:
    """Cover remaining branches in json_printer.py."""

    def test_bash_stream_cancel_existing_timer(self) -> None:
        """Bash stream cancels existing timer when flush interval reached.

        Sets up per-tab bash state with a pending timer and old last_flush,
        then verifies the timer is cancelled and buffer is flushed.
        """
        p = JsonPrinter()
        p._thread_local.task_id = "0"
        with p._bash_lock:
            bs = p._bash_state
            bs.last_flush = time.monotonic() - 1.0
            bs.timer = threading.Timer(10.0, p._flush_bash)
            bs.timer.daemon = True
            bs.timer.start()
        p.start_recording()
        p.print("line1\n", type="bash_stream")
        with p._bash_lock:
            assert p._bash_state.timer is None
        events = p.stop_recording()
        output_events = [e for e in events if e.get("type") == "system_output"]
        assert len(output_events) == 1

    def test_print_tool_result_non_core_tool(self) -> None:
        """Non-core tool result is now rendered (policy: render every tool's
        return value EXCEPT ``finish``). Verify the event reaches the
        recording for a custom (non-core) tool.
        """
        p = JsonPrinter()
        p._thread_local.task_id = "0"
        p.start_recording()
        p.print("some result", type="tool_result", tool_name="custom_tool", is_error=False)
        events = p.stop_recording()
        tool_results = [e for e in events if e.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["content"] == "some result"


class TestServerCompleteEmptyQuery:
    """Cover the empty-query branch of the complete command (line 188->exit)."""

    def test_complete_command_empty_query(self) -> None:
        """Sending complete command with empty query doesn't start thread."""
        server = VSCodeServer()
        server._handle_command({"type": "complete", "query": ""})
        assert server._complete_seq_latest.get("", -1) >= 0

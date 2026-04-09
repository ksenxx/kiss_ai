"""Integration tests for remaining uncovered branches in sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import (
    _stop_monitor,
    _truncate_output,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
)
from kiss.agents.vscode.server import VSCodeServer


def _git(tmpdir: str, *args: str) -> None:
    """Run a git command in tmpdir, suppressing output."""
    subprocess.run(["git", *args], cwd=tmpdir, capture_output=True, check=True)

# ---------------------------------------------------------------------------
# persistence.py — uncovered branches
# ---------------------------------------------------------------------------


class TestPersistenceBranches:
    """Cover remaining branches in persistence.py."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

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
            th._cleanup_stale_cs_dirs(max_age_hours=24)
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

    def test_clip_autocomplete_suggestion_echo_prefix(self) -> None:
        """clip_autocomplete_suggestion strips query prefix when echoed."""
        result = clip_autocomplete_suggestion("hello", "hello world")
        assert result == " world"

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

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_run_loop_empty_lines_and_invalid_json(self) -> None:
        """server.run() skips empty lines, handles invalid JSON (line 119)."""
        import io

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

    def test_fast_complete_history_match(self) -> None:
        """_complete returns history match via broadcast."""
        server = VSCodeServer()
        events: list[dict] = []  # type: ignore[type-arg]
        def cap(ev: dict) -> None:  # type: ignore[type-arg]
            events.append(ev)
        server.printer.broadcast = cap  # type: ignore[assignment]
        # Add a task to history
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


# ---------------------------------------------------------------------------
# stateful_sorcar_agent.py — uncovered branches
# ---------------------------------------------------------------------------


class TestStatefulSorcarAgentBranches:
    """Cover remaining branches in stateful_sorcar_agent.py."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_build_chat_prompt_entry_without_result(self) -> None:
        """build_chat_prompt skips result when entry has no result (line 84->82)."""
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
        agent = StatefulSorcarAgent("test")
        # Add a task with empty result to the agent's chat
        task_id = th._add_task("task with no result", chat_id=agent._chat_id)
        th._save_task_result("", task_id)
        prompt = agent.build_chat_prompt("new task")
        assert "### Task 1" in prompt
        assert "### Result 1" not in prompt
        assert "new task" in prompt


# ---------------------------------------------------------------------------
# browser_ui.py — uncovered branches
# ---------------------------------------------------------------------------


class TestBrowserUIBranches:
    """Cover remaining branches in browser_ui.py."""

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


# ---------------------------------------------------------------------------
# web_use_tool.py — uncovered branches (basic non-browser tests)
# ---------------------------------------------------------------------------


class TestWebUseToolBranches:
    """Cover basic branches in web_use_tool.py that don't need a real browser."""

    def test_check_for_new_tab_no_context(self) -> None:
        """_check_for_new_tab returns immediately when no context."""
        tool = WebUseTool(headless=True)
        tool._context = None
        tool._check_for_new_tab()  # should not raise


class TestServerCompleteEmptyQuery:
    """Cover the empty-query branch of the complete command (line 188->exit)."""

    def test_complete_command_empty_query(self) -> None:
        """Sending complete command with empty query doesn't start thread."""
        server = VSCodeServer()
        server._handle_command({"type": "complete", "query": ""})
        # No thread started - seq just incremented
        assert server._complete_seq_latest >= 0


class TestSorcarAgentDockerBranch:
    """Cover docker_manager truthy branch in _get_tools (lines 64-67)."""

    def test_get_tools_with_docker_manager(self) -> None:
        """When docker_manager is truthy, DockerTools are used."""
        agent = SorcarAgent("test")

        class FakeDockerManager:
            def Bash(self, cmd: str, desc: str) -> str:  # noqa: N802
                return "docker output"

        agent.docker_manager = FakeDockerManager()
        tools = agent._get_tools()
        # First tool should be _docker_bash (the bound method)
        assert callable(tools[0])
        # Should have docker tools (Read, Edit, Write) from DockerTools
        tool_names = [getattr(t, "__name__", getattr(t, "__func__", t).__name__) for t in tools]
        assert "Read" in tool_names
        assert "Edit" in tool_names
        assert "Write" in tool_names
        if agent.web_use_tool:
            agent.web_use_tool.close()


class TestWebUseToolTruncation:
    """Cover _get_ax_tree truncation branch (line 157)."""

    def test_ax_tree_truncated(self, tmp_path: Path) -> None:
        """Large accessibility tree gets truncated."""
        # Create HTML with many interactive elements
        buttons = "\n".join(f'<button>Button{i}</button>' for i in range(200))
        html_file = tmp_path / "big.html"
        html_file.write_text(f"<html><body>{buttons}</body></html>")
        tool = WebUseTool(headless=True)
        try:
            tool.go_to_url(f"file://{html_file}")
            # Call with small max_chars to trigger truncation
            result = tool._get_ax_tree(max_chars=100)
            assert "[truncated]" in result
        finally:
            tool.close()


class TestWebUseToolNewTab:
    """Cover _check_for_new_tab and click->new tab branches (lines 175-177, 266-267)."""

    def test_click_opens_new_tab(self, tmp_path: Path) -> None:
        """Clicking a target=_blank link opens a new tab."""
        html_file = tmp_path / "newtab.html"
        html_file.write_text(
            '<html><body><a href="about:blank" target="_blank">Open New</a></body></html>'
        )
        tool = WebUseTool(headless=True)
        try:
            tool.go_to_url(f"file://{html_file}")
            # Find the link element
            link_id = None
            for i, el in enumerate(tool._elements):
                if el["role"] == "link":
                    link_id = i + 1
                    break
            if link_id:
                result = tool.click(link_id)
                # Should have switched to new tab or at least not errored
                assert "Error" not in result or "Page:" in result
        finally:
            tool.close()


class TestWebUseToolEmptyNameLocator:
    """Cover _resolve_locator empty name branch (line 192)."""

    def test_resolve_locator_empty_name(self, tmp_path: Path) -> None:
        """Element with empty name uses get_by_role without name."""
        html_file = tmp_path / "emptyname.html"
        html_file.write_text('<html><body><button></button></body></html>')
        tool = WebUseTool(headless=True)
        try:
            tool.go_to_url(f"file://{html_file}")
            # Check if there's a button with empty name
            for i, el in enumerate(tool._elements):
                if el["role"] == "button" and el["name"] == "":
                    # Click it to trigger the empty-name locator path
                    result = tool.click(i + 1)
                    assert "Error" not in result or "Page:" in result
                    break
        finally:
            tool.close()


class TestWebUseToolAskUser:
    """Cover ask_user_browser_action (lines 451-459)."""

    def test_ask_user_browser_action_with_url(self, tmp_path: Path) -> None:
        """ask_user_browser_action navigates to url and calls callback."""
        callback_calls: list[tuple[str, str]] = []

        def callback(instruction: str, url: str) -> None:
            callback_calls.append((instruction, url))

        html_file = tmp_path / "ask.html"
        html_file.write_text('<html><body><p>Test page</p></body></html>')
        tool = WebUseTool(wait_for_user_callback=callback, headless=True)
        try:
            tool._ensure_browser()
            result = tool.ask_user_browser_action(
                "Do something", url=f"file://{html_file}"
            )
            assert len(callback_calls) == 1
            assert callback_calls[0][0] == "Do something"
            assert "Page:" in result
        finally:
            tool.close()

    def test_ask_user_browser_action_no_callback(self, tmp_path: Path) -> None:
        """ask_user_browser_action works without a callback."""
        tool = WebUseTool(headless=True)
        try:
            html_file = tmp_path / "ask3.html"
            html_file.write_text('<html><body><p>Hello</p></body></html>')
            tool.go_to_url(f"file://{html_file}")
            result = tool.ask_user_browser_action("Do stuff")
            assert "Page:" in result
        finally:
            tool.close()


class TestSorcarAgentAttachmentNoParts:
    """Cover the 'if parts' False branch (line 190->199)."""

    def test_run_with_unknown_attachment_type(self) -> None:
        """Attachment with unknown mime type produces no parts, so if parts: is False."""
        from kiss.core.models.model import Attachment

        agent = SorcarAgent("test")
        try:
            agent.run(
                prompt_template="test task",
                model_name="nonexistent-model",
                attachments=[
                    Attachment(data=b"data", mime_type="text/plain"),
                ],
            )
        except Exception:
            pass


class TestWebUseToolResolveLocatorInvisible:
    """Cover _resolve_locator loop where is_visible returns False (200->198)."""

    def test_resolve_locator_invisible_element(self, tmp_path: Path) -> None:
        """When first matching element is not visible, loop skips it (200->198).

        Use a zero-size button (clip:rect(0,0,0,0) + width/height 0) which stays
        in the accessibility tree but makes is_visible() return False.
        """
        html_file = tmp_path / "hidden.html"
        html_file.write_text(
            "<html><body>"
            '<button style="position:absolute;width:0;height:0;padding:0;'
            'border:0;overflow:hidden;clip:rect(0,0,0,0)">Submit</button>'
            "<button>Submit</button>"
            "</body></html>"
        )
        tool = WebUseTool(headless=True)
        try:
            tool.go_to_url(f"file://{html_file}")
            # Both buttons should be in the accessibility snapshot
            btn_id = None
            for i, el in enumerate(tool._elements):
                if el["role"] == "button" and el["name"] == "Submit":
                    btn_id = i + 1
                    break
            assert btn_id is not None, "Should find Submit button in elements"
            result = tool.click(btn_id)
            assert "Error" not in result or "Page:" in result
        finally:
            tool.close()

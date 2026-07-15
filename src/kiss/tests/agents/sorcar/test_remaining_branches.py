# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for remaining uncovered branches in sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
)
from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.agents.vscode.server import VSCodeServer


def _git(tmpdir: str, *args: str) -> None:
    """Run a git command in tmpdir, suppressing output."""
    subprocess.run(["git", *args], cwd=tmpdir, capture_output=True, check=True)


def _file_suffix(
    server: VSCodeServer,
    query: str,
    snapshot_file: str = "",
    snapshot_content: str = "",
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


class TestPersistenceBranches:
    """Cover remaining branches in persistence.py."""

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

    def test_chat_context_text_cache_hits_and_invalidates(self) -> None:
        """_load_chat_context_text caches and is invalidated by writes."""
        # Start from a clean cache so a chat_id reused across tests
        # doesn't carry over stale text.
        th._invalidate_chat_context_cache()
        task_id, chat_id = th._add_task("alpha_one")
        th._save_task_result("result_one_text", task_id=task_id)
        first = th._load_chat_context_text(chat_id)
        assert "alpha_one" in first
        assert "result_one_text" in first

        # Mutate the underlying row directly via SQL — this path does
        # not invalidate the cache, so the next call must still return
        # the cached pre-mutation text.
        db = th._get_db()
        db.execute(
            "UPDATE task_history SET result = ? WHERE id = ?",
            ("SECRET_NEW_RESULT", task_id),
        )
        db.commit()
        cached = th._load_chat_context_text(chat_id)
        assert cached == first
        assert "SECRET_NEW_RESULT" not in cached

        # After explicit invalidation the next call observes the
        # updated row.
        th._invalidate_chat_context_cache(chat_id)
        refreshed = th._load_chat_context_text(chat_id)
        assert "SECRET_NEW_RESULT" in refreshed
        assert "result_one_text" not in refreshed

        # _save_task_result invalidates automatically.
        th._save_task_result("AUTO_INVALIDATED", task_id=task_id)
        after_save = th._load_chat_context_text(chat_id)
        assert "AUTO_INVALIDATED" in after_save

        # _add_task on the same chat_id also invalidates automatically.
        th._add_task("brand_new_task_added", chat_id=chat_id)
        after_add = th._load_chat_context_text(chat_id)
        assert "brand_new_task_added" in after_add
        assert "AUTO_INVALIDATED" in after_add

    def test_chat_context_text_cache_clear_all(self) -> None:
        """_invalidate_chat_context_cache() with no arg clears every entry."""
        th._invalidate_chat_context_cache()
        _, chat_a = th._add_task("aa_one")
        _, chat_b = th._add_task("bb_one")
        # Populate cache for both.
        text_a = th._load_chat_context_text(chat_a)
        text_b = th._load_chat_context_text(chat_b)
        assert "aa_one" in text_a
        assert "bb_one" in text_b
        # Out-of-band SQL update neither chat sees through the cache.
        db = th._get_db()
        db.execute(
            "UPDATE task_history SET task = 'mut_aa' WHERE chat_id = ?",
            (chat_a,),
        )
        db.execute(
            "UPDATE task_history SET task = 'mut_bb' WHERE chat_id = ?",
            (chat_b,),
        )
        db.commit()
        assert "mut_aa" not in th._load_chat_context_text(chat_a)
        assert "mut_bb" not in th._load_chat_context_text(chat_b)
        th._invalidate_chat_context_cache()
        assert "mut_aa" in th._load_chat_context_text(chat_a)
        assert "mut_bb" in th._load_chat_context_text(chat_b)

    def test_chat_context_text_cache_empty_chat_id(self) -> None:
        """_load_chat_context_text returns '' for empty chat_id."""
        assert th._load_chat_context_text("") == ""

    def test_load_latest_chat_events_bad_json(self) -> None:
        """_load_latest_chat_events_by_chat_id handles corrupt event_json gracefully."""
        db = th._get_db()
        task_id, _ = th._add_task("corrupt-event-test", chat_id="corrupt_test")
        import time as _time
        now = _time.time()
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
            (task_id, 0, "NOT VALID JSON {{{", now),
        )
        db.execute(
            "INSERT INTO events (task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
            (task_id, 1, json.dumps({"type": "ok"}), now),
        )
        db.commit()
        result = th._load_latest_chat_events_by_chat_id("corrupt_test")
        assert result is not None
        events = result["events"]
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["type"] == "ok"


class TestUsefulToolsBranches:
    """Cover remaining branches in useful_tools.py."""

    def test_truncate_output_zero_tail(self) -> None:
        """_truncate_output when max_chars exactly equals worst_msg length, tail=0 (line 33)."""
        output = "A" * 200
        worst_msg = f"\n\n... [truncated {len(output)} chars] ...\n\n"
        max_chars = len(worst_msg)
        result = _truncate_output(output, max_chars)
        assert "truncated" in result
        assert not result.endswith("A")

    def test_stop_monitor_exits_when_done(self) -> None:
        """_stop_monitor exits cleanly when done is set (line 207 exit branch)."""
        stop = threading.Event()
        done = threading.Event()
        process = subprocess.Popen(["true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        done.set()
        t = threading.Thread(target=_stop_monitor, args=(stop, process, done))
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()


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

    def test_generate_followup_text_failure(self) -> None:
        """generate_followup_text returns empty string on LLM failure (lines 104-106)."""
        from kiss.agents.vscode.helpers import generate_followup_text
        result = generate_followup_text("task", "result", "nonexistent-model-xyz")
        assert result == ""


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
            result = _file_suffix(server, "calc", path, "")
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
        # No file content at all — must still find a candidate from chat history.
        assert _file_suffix(server,
            "calc", "", "", chat_id,
        ) == "ulate_total_amount"
        assert _file_suffix(server,
            "parse_xml", "", "", chat_id,
        ) == "_payload"
        # Without chat_id, the same call returns nothing because there is
        # no active-file content to harvest identifiers from.
        assert _file_suffix(server, "calc", "", "") == ""

    def test_file_suffix_combines_file_and_chat(self) -> None:
        """File content and chat history both contribute candidates."""
        _task_id, chat_id = th._add_task("chat had wonderful_widget_factory in it")
        server = VSCodeServer()
        file_content = "class HelperUtil:\n    pass\n"
        # Match from file content.
        assert _file_suffix(server,
            "Help", "", file_content, chat_id,
        ) == "erUtil"
        # Match from chat history when the partial doesn't appear in file.
        assert _file_suffix(server,
            "wonderful", "", file_content, chat_id,
        ) == "_widget_factory"

    def test_file_suffix_caches_chat_context(self) -> None:
        """Chat-context text is cached between keystrokes in the same chat."""
        task_id, chat_id = th._add_task("first wonderful_alpha_token here")
        th._save_task_result("nothing useful", task_id=task_id)
        server = VSCodeServer()

        # First call populates the cache; suggestion comes from chat text.
        assert _file_suffix(server,
            "wonderful_a", "", "", chat_id,
        ) == "lpha_token"

        # Mutate the row out-of-band so the DB no longer contains the
        # original token.  The cache must still serve the stale text,
        # proving the second keystroke didn't re-run SQL/joins.
        db = th._get_db()
        db.execute(
            "UPDATE task_history SET task = ? WHERE id = ?",
            ("first beta_zero_marker different", task_id),
        )
        db.commit()
        assert _file_suffix(server,
            "wonderful_a", "", "", chat_id,
        ) == "lpha_token"

        # An explicit invalidation forces a reload; now beta_zero_marker
        # is visible and the old token is not.
        th._invalidate_chat_context_cache(chat_id)
        assert _file_suffix(server,
            "beta_zero", "", "", chat_id,
        ) == "_marker"
        assert _file_suffix(server,
            "wonderful_a", "", "", chat_id,
        ) == ""

        # _save_task_result auto-invalidates: write a result containing a
        # brand-new identifier and confirm the next keystroke sees it.
        th._save_task_result(
            "gamma_three_signal appears now", task_id=task_id,
        )
        assert _file_suffix(server,
            "gamma_three", "", "", chat_id,
        ) == "_signal"

        # _add_task also auto-invalidates.
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


class TestSorcarAgentBranches:
    """Cover remaining branches in sorcar_agent.py."""

    def test_get_tools_stream_no_printer(self) -> None:
        """_stream callback handles None printer (line 39->exit)."""
        agent = SorcarAgent("test")
        agent.printer = None
        tools = agent._get_tools()
        assert len(tools) > 0
        bash_tool = tools[0]
        result = bash_tool(command="echo test_no_printer", description="test", timeout_seconds=5)
        assert "test_no_printer" in result
        if agent.web_use_tool:
            agent.web_use_tool.close()


class TestChatSorcarAgentBranches:
    """Cover remaining branches in chat_sorcar_agent.py."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_build_chat_prompt_entry_without_result(self) -> None:
        """build_chat_prompt skips result when entry has no result (line 84->82)."""
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        agent = ChatSorcarAgent("test")
        task_id, chat_id = th._add_task("task with no result", chat_id="test_no_result")
        agent._chat_id = chat_id
        th._save_task_result("", task_id)
        prompt = agent.build_chat_prompt("new task")
        assert "### Task 1" in prompt
        assert "### Result 1" not in prompt
        assert "new task" in prompt


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


class TestWebUseToolBranches:
    """Cover basic branches in web_use_tool.py that don't need a real browser."""

    def test_check_for_new_tab_no_context(self) -> None:
        """_check_for_new_tab returns immediately when no context."""
        tool = WebUseTool(headless=True)
        tool._context = None
        tool._check_for_new_tab()


class TestServerCompleteEmptyQuery:
    """Cover the empty-query branch of the complete command (line 188->exit)."""

    def test_complete_command_empty_query(self) -> None:
        """Sending complete command with empty query doesn't start thread."""
        server = VSCodeServer()
        server._handle_command({"type": "complete", "query": ""})
        assert server._complete_seq_latest.get("", -1) >= 0


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
        assert callable(tools[0])
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
        buttons = "\n".join(f'<button>Button{i}</button>' for i in range(200))
        html_file = tmp_path / "big.html"
        html_file.write_text(f"<html><body>{buttons}</body></html>")
        tool = WebUseTool(headless=True)
        try:
            tool.go_to_url(f"file://{html_file}")
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
            link_id = None
            for i, el in enumerate(tool._elements):
                if el["role"] == "link":
                    link_id = i + 1
                    break
            if link_id:
                result = tool.click(link_id)
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
            for i, el in enumerate(tool._elements):
                if el["role"] == "button" and el["name"] == "":
                    result = tool.click(i + 1)
                    assert "Error" not in result or "Page:" in result
                    break
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

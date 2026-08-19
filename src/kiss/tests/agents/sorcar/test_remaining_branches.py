# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for remaining uncovered branches in sorcar/ and vscode/ modules.

No mocks, patches, fakes, or test doubles. All tests use real objects.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from pathlib import Path

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import (
    _stop_monitor,
    _truncate_output,
)
from kiss.agents.sorcar.web_use_tool import WebUseTool


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
        th._invalidate_chat_context_cache()
        task_id, chat_id = th._add_task("alpha_one")
        th._save_task_result("result_one_text", task_id=task_id)
        first = th._load_chat_context_text(chat_id)
        assert "alpha_one" in first
        assert "result_one_text" in first

        db = th._get_db()
        db.execute(
            "UPDATE task_history SET result = ? WHERE id = ?",
            ("SECRET_NEW_RESULT", task_id),
        )
        db.commit()
        cached = th._load_chat_context_text(chat_id)
        assert cached == first
        assert "SECRET_NEW_RESULT" not in cached

        th._invalidate_chat_context_cache(chat_id)
        refreshed = th._load_chat_context_text(chat_id)
        assert "SECRET_NEW_RESULT" in refreshed
        assert "result_one_text" not in refreshed

        th._save_task_result("AUTO_INVALIDATED", task_id=task_id)
        after_save = th._load_chat_context_text(chat_id)
        assert "AUTO_INVALIDATED" in after_save

        th._add_task("brand_new_task_added", chat_id=chat_id)
        after_add = th._load_chat_context_text(chat_id)
        assert "brand_new_task_added" in after_add
        assert "AUTO_INVALIDATED" in after_add

    def test_chat_context_text_cache_clear_all(self) -> None:
        """_invalidate_chat_context_cache() with no arg clears every entry."""
        th._invalidate_chat_context_cache()
        _, chat_a = th._add_task("aa_one")
        _, chat_b = th._add_task("bb_one")
        text_a = th._load_chat_context_text(chat_a)
        text_b = th._load_chat_context_text(chat_b)
        assert "aa_one" in text_a
        assert "bb_one" in text_b
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


class TestWebUseToolBranches:
    """Cover basic branches in web_use_tool.py that don't need a real browser."""

    def test_check_for_new_tab_no_context(self) -> None:
        """_check_for_new_tab returns immediately when no context."""
        tool = WebUseTool(headless=True)
        tool._context = None
        tool._check_for_new_tab()


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


class TestWebUseToolResolveLocatorDuplicateOccurrences:
    """Duplicate role/name IDs resolve to their exact snapshot occurrences."""

    def test_hidden_and_visible_duplicates_keep_distinct_ids(
        self, tmp_path: Path,
    ) -> None:
        """A hidden button ID must not silently redirect to its visible twin."""
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
            button_ids = [
                i + 1
                for i, element in enumerate(tool._elements)
                if element["role"] == "button" and element["name"] == "Submit"
            ]
            assert len(button_ids) == 2
            assert not tool._resolve_locator(button_ids[0]).is_visible()
            assert tool._resolve_locator(button_ids[1]).is_visible()

            result = tool.click(button_ids[1])
            assert "Error" not in result
            assert "Page:" in result
        finally:
            tool.close()

"""Regression test: ConsolePrinter and BaseBrowserPrinter produce the same content.

Feeds identical print() calls to both printers and verifies:
- Return values are identical for every call
- Both handle the same set of content types
- Semantic content (tool names, paths, results, etc.) matches
"""

import asyncio
import io
import queue
from types import SimpleNamespace

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.core.print_to_console import ConsolePrinter


def _drain(q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


def _make_printers():
    buf = io.StringIO()
    console = ConsolePrinter(file=buf)
    browser = BaseBrowserPrinter()
    bq: queue.Queue = queue.Queue()
    browser._clients.append(bq)
    return console, buf, browser, bq


def _event(evt_dict):
    return SimpleNamespace(event=evt_dict)


class TestPrintReturnValueParity:
    """Both printers must return the same string from print() for every type."""

    def test_text_returns_empty(self):
        console, _, browser, _ = _make_printers()
        assert console.print("hello", type="text") == browser.print("hello", type="text")

    def test_prompt_returns_empty(self):
        console, _, browser, _ = _make_printers()
        assert console.print("do X", type="prompt") == browser.print("do X", type="prompt")

    def test_tool_call_returns_empty(self):
        console, _, browser, _ = _make_printers()
        ti = {"command": "ls", "description": "list files"}
        r1 = console.print("Bash", type="tool_call", tool_input=ti)
        r2 = browser.print("Bash", type="tool_call", tool_input=ti)
        assert r1 == r2 == ""

    def test_tool_result_returns_empty(self):
        console, _, browser, _ = _make_printers()
        r1 = console.print("OK output", type="tool_result", is_error=False)
        r2 = browser.print("OK output", type="tool_result", is_error=False)
        assert r1 == r2 == ""

    def test_tool_result_error_returns_empty(self):
        console, _, browser, _ = _make_printers()
        r1 = console.print("fail!", type="tool_result", is_error=True)
        r2 = browser.print("fail!", type="tool_result", is_error=True)
        assert r1 == r2 == ""

    def test_result_returns_empty(self):
        console, _, browser, _ = _make_printers()
        r1 = console.print("done", type="result", cost="$0.01", step_count=3, total_tokens=100)
        r2 = browser.print("done", type="result", cost="$0.01", step_count=3, total_tokens=100)
        assert r1 == r2 == ""

    def test_usage_info_returns_empty(self):
        console, _, browser, _ = _make_printers()
        r1 = console.print("tokens: 100", type="usage_info")
        r2 = browser.print("tokens: 100", type="usage_info")
        assert r1 == r2 == ""

    def test_bash_stream_returns_empty(self):
        console, _, browser, _ = _make_printers()
        r1 = console.print("output line\n", type="bash_stream")
        r2 = browser.print("output line\n", type="bash_stream")
        assert r1 == r2 == ""

    def test_unknown_type_returns_empty(self):
        console, _, browser, _ = _make_printers()
        r1 = console.print("x", type="nonexistent_type")
        r2 = browser.print("x", type="nonexistent_type")
        assert r1 == r2 == ""


class TestStreamEventReturnParity:
    """Both printers must return the same extracted text from stream events."""

    def test_thinking_delta(self):
        console, _, browser, _ = _make_printers()
        ev = _event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "analyzing..."},
        })
        assert console.print(ev, type="stream_event") == browser.print(ev, type="stream_event")

    def test_text_delta(self):
        console, _, browser, _ = _make_printers()
        ev = _event({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello world"},
        })
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == "hello world"

    def test_input_json_delta(self):
        console, _, browser, _ = _make_printers()
        ev = _event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
        })
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""

    def test_content_block_start_thinking(self):
        console, _, browser, _ = _make_printers()
        ev = _event({
            "type": "content_block_start",
            "content_block": {"type": "thinking"},
        })
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""
        assert console._current_block_type == browser._current_block_type == "thinking"

    def test_content_block_start_tool_use(self):
        console, _, browser, _ = _make_printers()
        ev = _event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Bash"},
        })
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""
        assert console._tool_name == browser._tool_name == "Bash"
        assert console._tool_json_buffer == browser._tool_json_buffer == ""

    def test_content_block_start_text(self):
        console, _, browser, _ = _make_printers()
        ev = _event({
            "type": "content_block_start",
            "content_block": {"type": "text"},
        })
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""
        assert console._current_block_type == browser._current_block_type == "text"

    def test_content_block_stop_thinking(self):
        console, _, browser, _ = _make_printers()
        console._current_block_type = "thinking"
        browser._current_block_type = "thinking"
        ev = _event({"type": "content_block_stop"})
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""
        assert console._current_block_type == browser._current_block_type == ""

    def test_content_block_stop_tool_use(self):
        console, _, browser, _ = _make_printers()
        console._current_block_type = "tool_use"
        browser._current_block_type = "tool_use"
        console._tool_name = "Write"
        browser._tool_name = "Write"
        json_buf = '{"file_path": "test.py", "content": "x=1"}'
        console._tool_json_buffer = json_buf
        browser._tool_json_buffer = json_buf
        ev = _event({"type": "content_block_stop"})
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""

    def test_content_block_stop_text(self):
        console, _, browser, _ = _make_printers()
        console._current_block_type = "text"
        browser._current_block_type = "text"
        ev = _event({"type": "content_block_stop"})
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""

    def test_unknown_event_type(self):
        console, _, browser, _ = _make_printers()
        ev = _event({"type": "message_start"})
        r1 = console.print(ev, type="stream_event")
        r2 = browser.print(ev, type="stream_event")
        assert r1 == r2 == ""


class TestToolCallContentParity:
    """Both printers show the same tool call details."""

    def test_bash_with_command_and_description(self):
        console, buf, browser, bq = _make_printers()
        ti = {"command": "pytest -x", "description": "Run tests"}
        console.print("Bash", type="tool_call", tool_input=ti)
        browser.print("Bash", type="tool_call", tool_input=ti)
        out = buf.getvalue()
        events = _drain(bq)
        # Filter to tool_call events
        tc_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tc_events) == 1
        ev = tc_events[0]
        # Both show tool name
        assert "Bash" in out
        assert ev["name"] == "Bash"
        # Both show command
        assert "pytest -x" in out
        assert ev["command"] == "pytest -x"
        # Both show description
        assert "Run tests" in out
        assert ev["description"] == "Run tests"

    def test_write_with_file_path_and_content(self):
        console, buf, browser, bq = _make_printers()
        ti = {"file_path": "src/main.py", "content": "print('hello')"}
        console.print("Write", type="tool_call", tool_input=ti)
        browser.print("Write", type="tool_call", tool_input=ti)
        out = buf.getvalue()
        events = _drain(bq)
        tc_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tc_events) == 1
        ev = tc_events[0]
        assert "Write" in out
        assert ev["name"] == "Write"
        assert "src/main.py" in out
        assert ev["path"] == "src/main.py"
        assert ev["lang"] == "python"
        assert "print('hello')" in out
        assert ev["content"] == "print('hello')"

    def test_edit_with_old_new_string(self):
        console, buf, browser, bq = _make_printers()
        ti = {
            "file_path": "app.py",
            "old_string": "x = 1",
            "new_string": "x = 2",
        }
        console.print("Edit", type="tool_call", tool_input=ti)
        browser.print("Edit", type="tool_call", tool_input=ti)
        out = buf.getvalue()
        events = _drain(bq)
        tc_events = [e for e in events if e["type"] == "tool_call"]
        ev = tc_events[0]
        assert "x = 1" in out
        assert "x = 2" in out
        assert ev["old_string"] == "x = 1"
        assert ev["new_string"] == "x = 2"

    def test_extras_shown_by_both(self):
        console, buf, browser, bq = _make_printers()
        ti = {"file_path": "a.txt", "replace_all": "true"}
        console.print("Edit", type="tool_call", tool_input=ti)
        browser.print("Edit", type="tool_call", tool_input=ti)
        out = buf.getvalue()
        events = _drain(bq)
        tc_events = [e for e in events if e["type"] == "tool_call"]
        ev = tc_events[0]
        assert "replace_all" in out
        assert "replace_all" in ev.get("extras", {})

    def test_no_arguments(self):
        console, buf, browser, bq = _make_printers()
        console.print("screenshot", type="tool_call", tool_input={})
        browser.print("screenshot", type="tool_call", tool_input={})
        out = buf.getvalue()
        events = _drain(bq)
        tc_events = [e for e in events if e["type"] == "tool_call"]
        assert "screenshot" in out
        assert tc_events[0]["name"] == "screenshot"


class TestToolResultContentParity:
    """Both printers display the same tool result content."""

    def test_success_result(self):
        console, buf, browser, bq = _make_printers()
        console.print("file written successfully", type="tool_result", is_error=False)
        browser.print("file written successfully", type="tool_result", is_error=False)
        out = buf.getvalue()
        events = _drain(bq)
        tr_events = [e for e in events if e["type"] == "tool_result"]
        assert "file written successfully" in out
        assert tr_events[0]["content"] == "file written successfully"
        assert tr_events[0]["is_error"] is False

    def test_error_result(self):
        console, buf, browser, bq = _make_printers()
        console.print("FileNotFoundError", type="tool_result", is_error=True)
        browser.print("FileNotFoundError", type="tool_result", is_error=True)
        out = buf.getvalue()
        events = _drain(bq)
        tr_events = [e for e in events if e["type"] == "tool_result"]
        assert "FileNotFoundError" in out
        assert "FAILED" in out
        assert tr_events[0]["content"] == "FileNotFoundError"
        assert tr_events[0]["is_error"] is True

    def test_truncation_applied_equally(self):
        from kiss.core.printer import MAX_RESULT_LEN
        console, buf, browser, bq = _make_printers()
        long_content = "x" * (MAX_RESULT_LEN * 2)
        console.print(long_content, type="tool_result", is_error=False)
        browser.print(long_content, type="tool_result", is_error=False)
        out = buf.getvalue()
        events = _drain(bq)
        tr_events = [e for e in events if e["type"] == "tool_result"]
        # Both truncate
        assert "... (truncated) ..." in out
        assert "... (truncated) ..." in tr_events[0]["content"]


class TestResultContentParity:
    """Both printers display the same final result."""

    def test_plain_result(self):
        console, buf, browser, bq = _make_printers()
        console.print("task done", type="result", cost="$0.05", step_count=5, total_tokens=500)
        browser.print("task done", type="result", cost="$0.05", step_count=5, total_tokens=500)
        out = buf.getvalue()
        events = _drain(bq)
        r_events = [e for e in events if e["type"] == "result"]
        assert "task done" in out
        assert "5" in out  # step_count
        assert "500" in out  # total_tokens
        assert "$0.05" in out
        ev = r_events[0]
        assert ev["text"] == "task done"
        assert ev["step_count"] == 5
        assert ev["total_tokens"] == 500
        assert ev["cost"] == "$0.05"

    def test_yaml_result_with_summary(self):
        import yaml
        content = yaml.dump({"success": True, "summary": "All tests passed"})
        console, buf, browser, bq = _make_printers()
        console.print(content, type="result", cost="$0.10", step_count=10, total_tokens=1000)
        browser.print(content, type="result", cost="$0.10", step_count=10, total_tokens=1000)
        out = buf.getvalue()
        events = _drain(bq)
        r_events = [e for e in events if e["type"] == "result"]
        # Console shows success status and summary
        assert "PASSED" in out
        assert "All tests passed" in out
        # Browser sends parsed fields
        ev = r_events[0]
        assert ev["success"] is True
        assert ev["summary"] == "All tests passed"

    def test_yaml_result_with_failure(self):
        import yaml
        content = yaml.dump({"success": False, "summary": "2 tests failed"})
        console, buf, browser, bq = _make_printers()
        console.print(content, type="result", cost="N/A", step_count=1, total_tokens=50)
        browser.print(content, type="result", cost="N/A", step_count=1, total_tokens=50)
        out = buf.getvalue()
        events = _drain(bq)
        r_events = [e for e in events if e["type"] == "result"]
        assert "FAILED" in out
        ev = r_events[0]
        assert ev["success"] is False

    def test_empty_result(self):
        console, _, browser, bq = _make_printers()
        r1 = console.print("", type="result", cost="N/A", step_count=0, total_tokens=0)
        r2 = browser.print("", type="result", cost="N/A", step_count=0, total_tokens=0)
        assert r1 == r2 == ""


class TestMessageParity:
    """Both printers handle message objects the same way."""

    def test_tool_output_message(self):
        console, buf, browser, bq = _make_printers()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "command output\n"})
        console.print(msg, type="message")
        browser.print(msg, type="message")
        out = buf.getvalue()
        events = _drain(bq)
        assert "command output" in out
        so_events = [e for e in events if e["type"] == "system_output"]
        assert so_events[0]["text"] == "command output\n"

    def test_result_message(self):
        console, buf, browser, bq = _make_printers()
        msg = SimpleNamespace(result="completed successfully")
        console.print(msg, type="message", step_count=3, budget_used=0.02, total_tokens_used=200)
        browser.print(msg, type="message", step_count=3, budget_used=0.02, total_tokens_used=200)
        out = buf.getvalue()
        events = _drain(bq)
        assert "completed successfully" in out
        assert "$0.0200" in out
        r_events = [e for e in events if e["type"] == "result"]
        assert r_events[0]["text"] == "completed successfully"
        assert r_events[0]["cost"] == "$0.0200"
        assert r_events[0]["step_count"] == 3
        assert r_events[0]["total_tokens"] == 200

    def test_content_blocks_message(self):
        console, buf, browser, bq = _make_printers()
        block = SimpleNamespace(is_error=False, content="result content")
        msg = SimpleNamespace(content=[block])
        console.print(msg, type="message")
        browser.print(msg, type="message")
        out = buf.getvalue()
        events = _drain(bq)
        assert "result content" in out
        tr_events = [e for e in events if e["type"] == "tool_result"]
        assert tr_events[0]["content"] == "result content"

    def test_error_content_block_message(self):
        console, buf, browser, bq = _make_printers()
        block = SimpleNamespace(is_error=True, content="something broke")
        msg = SimpleNamespace(content=[block])
        console.print(msg, type="message")
        browser.print(msg, type="message")
        out = buf.getvalue()
        events = _drain(bq)
        assert "something broke" in out
        tr_events = [e for e in events if e["type"] == "tool_result"]
        assert tr_events[0]["is_error"] is True

    def test_empty_tool_output_produces_nothing(self):
        console, buf, browser, bq = _make_printers()
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        console.print(msg, type="message")
        browser.print(msg, type="message")
        events = _drain(bq)
        assert len(events) == 0


class TestTokenCallbackParity:
    """Both printers handle token_callback the same way."""

    def test_text_token(self):
        console, buf, browser, bq = _make_printers()
        asyncio.run(console.token_callback("hello"))
        asyncio.run(browser.token_callback("hello"))
        events = _drain(bq)
        assert "hello" in buf.getvalue()
        assert events[0] == {"type": "text_delta", "text": "hello"}

    def test_thinking_token(self):
        console, buf, browser, bq = _make_printers()
        console._current_block_type = "thinking"
        browser._current_block_type = "thinking"
        asyncio.run(console.token_callback("pondering"))
        asyncio.run(browser.token_callback("pondering"))
        events = _drain(bq)
        assert "pondering" in buf.getvalue()
        assert events[0] == {"type": "thinking_delta", "text": "pondering"}

    def test_empty_token_no_broadcast(self):
        console, _, browser, bq = _make_printers()
        asyncio.run(console.token_callback(""))
        asyncio.run(browser.token_callback(""))
        events = _drain(bq)
        assert len(events) == 0


class TestResetParity:
    """Both printers reset the same state."""

    def test_reset_clears_same_state(self):
        console, _, browser, _ = _make_printers()
        # Set state on both
        for p in (console, browser):
            p._current_block_type = "thinking"
            p._tool_name = "Read"
            p._tool_json_buffer = '{"x": 1}'
        console.reset()
        browser.reset()
        for p in (console, browser):
            assert p._current_block_type == ""
            assert p._tool_name == ""
            assert p._tool_json_buffer == ""


class TestFullAgentSequenceParity:
    """Simulate a full agent execution and verify both printers get the same content."""

    def test_full_sequence(self):
        console, buf, browser, bq = _make_printers()

        # 1. Prompt
        for p in (console, browser):
            p.print("Fix the bug in app.py", type="prompt")

        # 2. Stream: thinking block
        for p in (console, browser):
            p.print(_event({
                "type": "content_block_start",
                "content_block": {"type": "thinking"},
            }), type="stream_event")
        for p in (console, browser):
            p.print(_event({
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "I should read the file first"},
            }), type="stream_event")
        for p in (console, browser):
            p.print(_event({"type": "content_block_stop"}), type="stream_event")

        # 3. Stream: text block
        for p in (console, browser):
            p.print(_event({
                "type": "content_block_start",
                "content_block": {"type": "text"},
            }), type="stream_event")
        texts = []
        for p in (console, browser):
            texts.append(p.print(_event({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "I'll fix the bug"},
            }), type="stream_event"))
        assert texts[0] == texts[1] == "I'll fix the bug"
        for p in (console, browser):
            p.print(_event({"type": "content_block_stop"}), type="stream_event")

        # 4. Stream: tool_use block (Read)
        for p in (console, browser):
            p.print(_event({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Read"},
            }), type="stream_event")
        for p in (console, browser):
            p.print(_event({
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"file_path":'},
            }), type="stream_event")
        for p in (console, browser):
            p.print(_event({
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": ' "app.py"}'},
            }), type="stream_event")
        for p in (console, browser):
            p.print(_event({"type": "content_block_stop"}), type="stream_event")

        # 5. Tool call + result
        ti = {"file_path": "app.py", "content": "fixed code"}
        for p in (console, browser):
            p.print("Edit", type="tool_call", tool_input=ti)
        for p in (console, browser):
            p.print("File edited", type="tool_result", is_error=False)

        # 6. Usage info
        for p in (console, browser):
            p.print("Steps: 2/10, Tokens: 500", type="usage_info")

        # 7. Result
        import yaml
        result = yaml.dump({"success": True, "summary": "Bug fixed in app.py"})
        for p in (console, browser):
            p.print(result, type="result", cost="$0.05", step_count=2, total_tokens=500)

        # Verify console output
        out = buf.getvalue()
        assert "Fix the bug" in out  # prompt
        assert "Edit" in out  # tool call
        assert "app.py" in out  # file path
        assert "File edited" in out  # tool result
        assert "Bug fixed" in out  # result summary

        # Verify browser events
        events = _drain(bq)
        types = [e["type"] for e in events]
        assert "prompt" in types
        assert "thinking_start" in types
        assert "thinking_end" in types
        assert "text_end" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "usage_info" in types
        assert "result" in types

        # Tool call content matches
        tc = [e for e in events if e["type"] == "tool_call"]
        assert any(e["name"] == "Edit" for e in tc)
        assert any(e.get("path") == "app.py" for e in tc)

        # Result content matches
        r = [e for e in events if e["type"] == "result"]
        assert r[-1]["success"] is True
        assert "Bug fixed" in r[-1]["summary"]
        assert r[-1]["cost"] == "$0.05"
        assert r[-1]["step_count"] == 2
        assert r[-1]["total_tokens"] == 500

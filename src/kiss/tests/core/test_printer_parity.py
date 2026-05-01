"""Regression test: ConsolePrinter and BaseBrowserPrinter produce the same content.

Feeds identical print() calls to both printers and verifies:
- Return values are identical for every call
- Both handle the same set of content types
- Semantic content (tool names, paths, results, etc.) matches
"""

import io
from types import SimpleNamespace

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.print_to_console import ConsolePrinter


def _make_printers():
    buf = io.StringIO()
    console = ConsolePrinter(file=buf)
    browser = BaseBrowserPrinter()
    browser.start_recording()
    return console, buf, browser


def _drain(browser: BaseBrowserPrinter) -> list[dict]:
    """Stop recording and return all recorded events."""
    return browser.stop_recording()


class TestPrintReturnValueParity:
    """Both printers must return the same string from print() for every type."""

    def test_text_returns_empty(self):
        console, _, browser = _make_printers()
        assert console.print("hello", type="text") == browser.print("hello", type="text")

    def test_unknown_type_returns_empty(self):
        console, _, browser = _make_printers()
        r1 = console.print("x", type="nonexistent_type")
        r2 = browser.print("x", type="nonexistent_type")
        assert r1 == r2 == ""


class TestToolCallContentParity:
    """Both printers show the same tool call details."""

    def test_bash_with_command_and_description(self):
        console, buf, browser = _make_printers()
        ti = {"command": "pytest -x", "description": "Run tests"}
        console.print("Bash", type="tool_call", tool_input=ti)
        browser.print("Bash", type="tool_call", tool_input=ti)
        out = buf.getvalue()
        events = _drain(browser)
        tc_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tc_events) == 1
        ev = tc_events[0]
        assert "Bash" in out
        assert ev["name"] == "Bash"
        assert "pytest -x" in out
        assert ev["command"] == "pytest -x"
        assert "Run tests" in out
        assert ev["description"] == "Run tests"

    def test_edit_with_old_new_string(self):
        console, buf, browser = _make_printers()
        ti = {
            "file_path": "app.py",
            "old_string": "x = 1",
            "new_string": "x = 2",
        }
        console.print("Edit", type="tool_call", tool_input=ti)
        browser.print("Edit", type="tool_call", tool_input=ti)
        out = buf.getvalue()
        events = _drain(browser)
        tc_events = [e for e in events if e["type"] == "tool_call"]
        ev = tc_events[0]
        assert "x = 1" in out
        assert "x = 2" in out
        assert ev["old_string"] == "x = 1"
        assert ev["new_string"] == "x = 2"


class TestToolResultContentParity:
    """Both printers display the same tool result content."""

    def test_truncation_applied_on_error(self):
        from kiss.core.printer import MAX_RESULT_LEN
        console, buf, browser = _make_printers()
        long_content = "x" * (MAX_RESULT_LEN * 2)
        console.print(long_content, type="tool_result", is_error=True, tool_name="Read")
        browser.print(long_content, type="tool_result", is_error=True, tool_name="Read")
        out = buf.getvalue()
        events = _drain(browser)
        tr_events = [e for e in events if e["type"] == "tool_result"]
        assert "... (truncated) ..." in out
        assert "... (truncated) ..." in tr_events[0]["content"]


class TestMessageParity:
    """Both printers handle message objects the same way."""

    def test_tool_output_message(self):
        console, buf, browser = _make_printers()
        msg = SimpleNamespace(subtype="tool_output", data={"content": "command output\n"})
        console.print(msg, type="message")
        browser.print(msg, type="message")
        out = buf.getvalue()
        events = _drain(browser)
        assert "command output" in out
        so_events = [e for e in events if e["type"] == "system_output"]
        assert so_events[0]["text"] == "command output\n"

    def test_result_message(self):
        console, buf, browser = _make_printers()
        msg = SimpleNamespace(result="completed successfully")
        console.print(msg, type="message", budget_used=0.02, total_tokens_used=200)
        browser.print(msg, type="message", budget_used=0.02, total_tokens_used=200)
        out = buf.getvalue()
        events = _drain(browser)
        assert "completed successfully" in out
        assert "$0.0200" in out
        r_events = [e for e in events if e["type"] == "result"]
        assert r_events[0]["text"] == "completed successfully"
        assert r_events[0]["cost"] == "$0.0200"
        assert r_events[0]["total_tokens"] == 200

    def test_error_content_block_message(self):
        console, buf, browser = _make_printers()
        block = SimpleNamespace(is_error=True, content="something broke")
        msg = SimpleNamespace(content=[block])
        console.print(msg, type="message")
        browser.print(msg, type="message")
        out = buf.getvalue()
        events = _drain(browser)
        assert "something broke" in out
        tr_events = [e for e in events if e["type"] == "tool_result"]
        assert tr_events[0]["is_error"] is True

    def test_empty_tool_output_produces_nothing(self):
        console, buf, browser = _make_printers()
        msg = SimpleNamespace(subtype="tool_output", data={"content": ""})
        console.print(msg, type="message")
        browser.print(msg, type="message")
        events = _drain(browser)
        assert len(events) == 0


class TestTokenCallbackParity:
    """Both printers handle token_callback the same way."""

    def test_empty_token_no_broadcast(self):
        console, _, browser = _make_printers()
        console.token_callback("")
        browser.token_callback("")
        events = _drain(browser)
        assert len(events) == 0


class TestFullAgentSequenceParity:
    """Simulate a full agent execution via callbacks/print() and verify
    both printers produce the same content."""

    def test_full_sequence(self):
        console, buf, browser = _make_printers()

        for p in (console, browser):
            p.print("Fix the bug in app.py", type="prompt")

        for p in (console, browser):
            p.thinking_callback(True)
            p.token_callback("I should read the file first")
            p.thinking_callback(False)

        for p in (console, browser):
            p.token_callback("I'll fix the bug")

        ti = {"file_path": "app.py", "content": "fixed code"}
        for p in (console, browser):
            p.print("Edit", type="tool_call", tool_input=ti)
        for p in (console, browser):
            p.print("File edited", type="tool_result", is_error=False, tool_name="Edit")

        import yaml
        result = yaml.dump({"success": True, "summary": "Bug fixed in app.py"})
        for p in (console, browser):
            p.print(result, type="result", cost="$0.05", total_tokens=500)

        out = buf.getvalue()
        assert "Fix the bug" in out
        assert "Edit" in out
        assert "app.py" in out
        assert "Bug fixed" in out
        assert "Thinking" in out
        assert "I should read the file first" in out
        assert "I'll fix the bug" in out

        events = _drain(browser)
        types = [e["type"] for e in events]
        assert "prompt" in types
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types
        assert "text_delta" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "result" in types

        td = [e for e in events if e["type"] == "thinking_delta"]
        assert any("I should read the file first" in e["text"] for e in td)
        td2 = [e for e in events if e["type"] == "text_delta"]
        assert any("I'll fix the bug" in e["text"] for e in td2)

        tc = [e for e in events if e["type"] == "tool_call"]
        assert any(e["name"] == "Edit" for e in tc)
        assert any(e.get("path") == "app.py" for e in tc)

        r = [e for e in events if e["type"] == "result"]
        assert r[-1]["success"] is True
        assert "Bug fixed" in r[-1]["summary"]
        assert r[-1]["cost"] == "$0.05"
        assert r[-1]["total_tokens"] == 500

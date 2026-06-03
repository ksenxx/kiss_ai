# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: every tool's return value reaches the console & webview.

Before this fix, only ``Bash``/``Read``/``Edit``/``Write`` (and errors) were
shown. The return value of ``run_parallel``, ``ask_user_question``,
``update_settings``, and the ``WebUseTool`` methods was silently dropped both
in the terminal and in the chat sidebar. The fix flips the whitelist into a
single-entry blacklist (``finish``) so every other tool result is rendered.

Only ``finish`` should still be suppressed at the ``tool_result`` layer
because the agentic loop renders it again as a dedicated ``result`` panel.
"""

import io

from kiss.agents.vscode.json_printer import JsonPrinter
from kiss.core.print_to_console import ConsolePrinter

_TASK_COUNTER = 0


def _new_printers():
    global _TASK_COUNTER
    _TASK_COUNTER += 1
    buf = io.StringIO()
    console = ConsolePrinter(file=buf)
    browser = JsonPrinter()
    browser._thread_local.task_id = f"test-tool-result-{_TASK_COUNTER}"
    browser.start_recording()
    return console, buf, browser


def _tool_results(browser: JsonPrinter) -> list[dict]:
    return [e for e in browser.stop_recording() if e["type"] == "tool_result"]


class TestNonCoreToolResultsAreShown:
    """Non-core tools must now appear in both the console and the webview."""

    def test_run_parallel_result_broadcast_and_printed(self):
        console, buf, browser = _new_printers()
        payload = "- success: true\n  summary: did the thing\n"
        console.print(payload, type="tool_result", tool_name="run_parallel")
        browser.print(payload, type="tool_result", tool_name="run_parallel")
        evs = _tool_results(browser)
        assert len(evs) == 1
        assert evs[0]["content"] == payload
        assert evs[0]["is_error"] is False
        assert "did the thing" in buf.getvalue()
        assert "RESULT" in buf.getvalue()

    def test_ask_user_question_result_visible(self):
        console, buf, browser = _new_printers()
        console.print("blue", type="tool_result", tool_name="ask_user_question")
        browser.print("blue", type="tool_result", tool_name="ask_user_question")
        evs = _tool_results(browser)
        assert len(evs) == 1
        assert evs[0]["content"] == "blue"
        assert "blue" in buf.getvalue()

    def test_update_settings_result_visible(self):
        console, buf, browser = _new_printers()
        msg = "Updated: model=claude-sonnet-4"
        console.print(msg, type="tool_result", tool_name="update_settings")
        browser.print(msg, type="tool_result", tool_name="update_settings")
        evs = _tool_results(browser)
        assert len(evs) == 1
        assert msg in evs[0]["content"]
        assert msg in buf.getvalue()

    def test_web_use_tool_result_visible(self):
        console, buf, browser = _new_printers()
        payload = "<html>some page</html>"
        console.print(payload, type="tool_result", tool_name="go_to_url")
        browser.print(payload, type="tool_result", tool_name="go_to_url")
        evs = _tool_results(browser)
        assert len(evs) == 1
        assert payload in evs[0]["content"]
        assert payload in buf.getvalue()


class TestFinishStillSuppressed:
    """``finish`` is rendered as a dedicated result panel; suppress the dup."""

    def test_finish_tool_result_not_broadcast(self):
        console, buf, browser = _new_printers()
        console.print("done", type="tool_result", tool_name="finish")
        browser.print("done", type="tool_result", tool_name="finish")
        assert _tool_results(browser) == []
        # ConsolePrinter must not emit a RESULT rule for finish either.
        assert "RESULT" not in buf.getvalue()


class TestCoreToolsStillVisible:
    """Sanity check: the previously-working core-tool case still works."""

    def test_edit_result_visible(self):
        console, buf, browser = _new_printers()
        console.print("ok", type="tool_result", tool_name="Edit")
        browser.print("ok", type="tool_result", tool_name="Edit")
        evs = _tool_results(browser)
        assert len(evs) == 1
        assert evs[0]["content"] == "ok"
        assert "RESULT" in buf.getvalue()

    def test_error_still_visible(self):
        console, buf, browser = _new_printers()
        console.print(
            "boom", type="tool_result", tool_name="run_parallel", is_error=True,
        )
        browser.print(
            "boom", type="tool_result", tool_name="run_parallel", is_error=True,
        )
        evs = _tool_results(browser)
        assert len(evs) == 1
        assert evs[0]["is_error"] is True
        assert "FAILED" in buf.getvalue()

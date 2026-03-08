"""Integration tests for text wrapping in the sorcar chat window.

Verifies that tool call headers (Edit, Read, Write), usage info, and
other chat elements properly wrap long text instead of overflowing.

No mocks, patches, or test doubles.
"""

from __future__ import annotations

import queue
import re

from kiss.agents.sorcar.browser_ui import OUTPUT_CSS, BaseBrowserPrinter
from kiss.agents.sorcar.chatbot_ui import CHATBOT_CSS, _build_html


def _css_block(css: str, selector: str) -> str:
    """Extract the CSS block for a given selector from a CSS string."""
    # Find selector followed by {, then capture up to matching }
    pattern = re.escape(selector) + r"\s*\{([^}]*)\}"
    match = re.search(pattern, css)
    return match.group(1) if match else ""


class TestToolCallHeaderWrapping:
    """Verify .tc-h (tool call header) wraps long content."""

    def test_tc_h_has_flex_wrap(self) -> None:
        block = _css_block(OUTPUT_CSS, ".tc-h")
        assert "flex-wrap:wrap" in block

    def test_tc_h_is_flex_container(self) -> None:
        block = _css_block(OUTPUT_CSS, ".tc-h")
        assert "display:flex" in block

    def test_tp_has_word_break(self) -> None:
        """File path element (.tp) should break long paths."""
        block = _css_block(OUTPUT_CSS, ".tp")
        assert "word-break:break-all" in block

    def test_tp_has_min_width_zero(self) -> None:
        """File path element needs min-width:0 to shrink in flex."""
        block = _css_block(OUTPUT_CSS, ".tp")
        assert "min-width:0" in block

    def test_td_has_word_break(self) -> None:
        """Description element (.td) should break long text."""
        block = _css_block(OUTPUT_CSS, ".td")
        assert "word-break:break-word" in block

    def test_td_has_min_width_zero(self) -> None:
        """Description element needs min-width:0 to shrink in flex."""
        block = _css_block(OUTPUT_CSS, ".td")
        assert "min-width:0" in block


class TestUsageInfoWrapping:
    """Verify .usage element wraps long text."""

    def test_usage_has_pre_wrap(self) -> None:
        block = _css_block(OUTPUT_CSS, ".usage")
        assert "white-space:pre-wrap" in block

    def test_usage_has_word_break(self) -> None:
        block = _css_block(OUTPUT_CSS, ".usage")
        assert "word-break:break-word" in block

    def test_usage_no_nowrap(self) -> None:
        block = _css_block(OUTPUT_CSS, ".usage")
        assert "nowrap" not in block

    def test_usage_no_overflow_x(self) -> None:
        block = _css_block(OUTPUT_CSS, ".usage")
        assert "overflow-x" not in block

    def test_usage_has_overflow_wrap(self) -> None:
        block = _css_block(OUTPUT_CSS, ".usage")
        assert "overflow-wrap:break-word" in block


class TestBuildHtmlContainsWrappingCSS:
    """Verify _build_html output includes the wrapping CSS."""

    def setup_method(self) -> None:
        self.html = _build_html("Test", "", "/tmp")

    def test_html_contains_flex_wrap_for_tc_h(self) -> None:
        assert "flex-wrap:wrap" in self.html

    def test_html_contains_word_break_for_tp(self) -> None:
        assert "word-break:break-all" in self.html

    def test_html_contains_word_break_for_td(self) -> None:
        assert "word-break:break-word" in self.html

    def test_html_contains_pre_wrap_for_usage(self) -> None:
        assert "white-space:pre-wrap" in self.html

    def test_html_usage_block_no_nowrap(self) -> None:
        """The .usage CSS block in the HTML should not have nowrap."""
        block = _css_block(self.html, ".usage")
        assert "nowrap" not in block


class TestToolCallBroadcastLongContent:
    """Verify tool call events carry full long paths and descriptions."""

    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()
        self.cq = self.printer.add_client()

    def teardown_method(self) -> None:
        self.printer.remove_client(self.cq)

    def _drain(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self.cq.get_nowait())
            except queue.Empty:
                break
        return events

    def test_edit_long_path(self) -> None:
        long_path = "/very/deep/nested/" + "subdir/" * 20 + "file.py"
        self.printer.print(
            "Edit",
            type="tool_call",
            tool_input={
                "file_path": long_path,
                "old_string": "a",
                "new_string": "b",
            },
        )
        events = self._drain()
        tc = [e for e in events if e["type"] == "tool_call"]
        assert len(tc) == 1
        assert tc[0]["path"] == long_path
        assert tc[0]["name"] == "Edit"

    def test_read_long_path(self) -> None:
        long_path = "/workspace/" + "a" * 200 + "/config.yaml"
        self.printer.print(
            "Read",
            type="tool_call",
            tool_input={"file_path": long_path},
        )
        events = self._drain()
        tc = [e for e in events if e["type"] == "tool_call"]
        assert len(tc) == 1
        assert tc[0]["path"] == long_path

    def test_write_long_path_and_description(self) -> None:
        long_path = "/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u.txt"
        long_desc = "Writing a very long description " * 10
        self.printer.print(
            "Write",
            type="tool_call",
            tool_input={
                "file_path": long_path,
                "content": "hello",
                "description": long_desc,
            },
        )
        events = self._drain()
        tc = [e for e in events if e["type"] == "tool_call"]
        assert len(tc) == 1
        assert tc[0]["path"] == long_path
        assert tc[0]["description"] == long_desc

    def test_tool_call_no_path(self) -> None:
        self.printer.print(
            "Bash",
            type="tool_call",
            tool_input={"command": "echo hi"},
        )
        events = self._drain()
        tc = [e for e in events if e["type"] == "tool_call"]
        assert len(tc) == 1
        assert "path" not in tc[0]
        assert tc[0]["command"] == "echo hi"


class TestUsageInfoBroadcastLongContent:
    """Verify usage_info events carry full long text."""

    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()
        self.cq = self.printer.add_client()

    def teardown_method(self) -> None:
        self.printer.remove_client(self.cq)

    def test_long_usage_text(self) -> None:
        long_text = (
            "Steps: 42/100, Tokens: 123456/200000, "
            "Budget: $1.2345/$200.00, "
            "Global Budget: $5.6789/$200.00"
        )
        self.printer.print(long_text, type="usage_info")
        event = self.cq.get_nowait()
        assert event["type"] == "usage_info"
        assert event["text"] == long_text

    def test_very_long_usage_text(self) -> None:
        long_text = "x" * 500
        self.printer.print(long_text, type="usage_info")
        event = self.cq.get_nowait()
        assert event["type"] == "usage_info"
        assert event["text"] == long_text

    def test_multiline_usage_text(self) -> None:
        text = "Line1: value\nLine2: value\nLine3: value"
        self.printer.print(text, type="usage_info")
        event = self.cq.get_nowait()
        assert event["type"] == "usage_info"
        assert event["text"] == text


class TestChatbotCSSWrapping:
    """Verify CHATBOT_CSS overrides don't break wrapping."""

    def test_chatbot_usage_no_nowrap(self) -> None:
        """CHATBOT_CSS .usage override should not reintroduce nowrap."""
        block = _css_block(CHATBOT_CSS, ".usage")
        assert "nowrap" not in block

    def test_assistant_panel_usage_no_nowrap(self) -> None:
        """#assistant-panel .usage should not have nowrap."""
        block = _css_block(CHATBOT_CSS, "#assistant-panel .usage")
        assert "nowrap" not in block

    def test_assistant_panel_tc_h_no_nowrap(self) -> None:
        block = _css_block(CHATBOT_CSS, "#assistant-panel .tc-h")
        assert "nowrap" not in block


class TestEventHandlerJSToolCallRendering:
    """Verify the JS event handler for tool_call creates proper HTML structure."""

    def test_js_contains_tp_class(self) -> None:
        from kiss.agents.sorcar.browser_ui import EVENT_HANDLER_JS

        assert 'class="tp"' in EVENT_HANDLER_JS or "class=\"tp\"" in EVENT_HANDLER_JS

    def test_js_contains_td_class(self) -> None:
        from kiss.agents.sorcar.browser_ui import EVENT_HANDLER_JS

        assert 'class="td"' in EVENT_HANDLER_JS or "class=\"td\"" in EVENT_HANDLER_JS

    def test_js_contains_tn_class(self) -> None:
        from kiss.agents.sorcar.browser_ui import EVENT_HANDLER_JS

        assert 'class="tn"' in EVENT_HANDLER_JS or "class=\"tn\"" in EVENT_HANDLER_JS

    def test_js_usage_handler_creates_usage_div(self) -> None:
        from kiss.agents.sorcar.browser_ui import EVENT_HANDLER_JS

        assert "usage_info" in EVENT_HANDLER_JS
        assert "usage" in EVENT_HANDLER_JS

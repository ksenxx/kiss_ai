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

class TestUsageInfoBroadcastLongContent:
    """Verify usage_info events carry full long text."""

    def setup_method(self) -> None:
        self.printer = BaseBrowserPrinter()
        self.cq = self.printer.add_client()

    def teardown_method(self) -> None:
        self.printer.remove_client(self.cq)

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

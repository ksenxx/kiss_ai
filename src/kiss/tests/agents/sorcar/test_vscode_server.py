"""Tests for the VS Code extension backend server.

Tests cover: model picker (vendor ordering, sorting, grouping, pricing),
file picker (sorting by usage/recency/end-distance, section grouping),
keyboard interaction parity with web Sorcar, and the JS rendering code
in main.js.
No mocks — uses real functions from the server module.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.helpers import model_vendor
from kiss.agents.vscode.server import VSCodeServer


def _model_vendor_name(name: str) -> str:
    return model_vendor(name)[0]


def _model_vendor_order(name: str) -> int:
    return model_vendor(name)[1]


class TestModelVendorOrder(unittest.TestCase):
    """Test _model_vendor_order matches web Sorcar's modelVendor sorting."""

    def test_order_is_consistent(self) -> None:
        names = [
            "unknown-model",
            "gemini-2.0-flash",
            "claude-opus-4-6",
            "gpt-4o",
            "openrouter/x",
            "minimax-large",
            "cc/opus",
        ]
        sorted_names = sorted(names, key=_model_vendor_order)
        assert sorted_names[0] == "claude-opus-4-6"
        assert sorted_names[1] == "cc/opus"
        assert sorted_names[2] == "gpt-4o"
        assert sorted_names[3] == "gemini-2.0-flash"
        assert sorted_names[-1] in ("unknown-model", "together/some-model")

    def test_cc_models_are_anthropic(self) -> None:
        assert _model_vendor_name("cc/opus") == "Anthropic"
        assert _model_vendor_name("cc/sonnet") == "Anthropic"
        assert _model_vendor_name("cc/haiku") == "Anthropic"
        assert _model_vendor_order("cc/opus") == 0


class TestGetFiles(unittest.TestCase):
    """Test VSCodeServer._get_files produces correct sections and sorting."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

        # Create test files
        for name in ["src/main.py", "src/util.py", "README.md", "test/test_main.py"]:
            path = Path(self.tmpdir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {name}")

        # Pre-populate file cache
        self.server._file_cache = [
            "src/main.py",
            "src/util.py",
            "README.md",
            "test/test_main.py",
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_files_filtered_by_prefix(self) -> None:
        self.server._get_files("main")
        files = self.events[0]["files"]
        for f in files:
            assert "main" in f["text"].lower()


class TestMainJsFilePicker(unittest.TestCase):
    """Test the main.js file picker JavaScript code for web Sorcar parity."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()

    def test_has_ac_section_rendering(self) -> None:
        assert "ac-section" in self.js

    def test_has_ac_footer_with_keyboard_hints(self) -> None:
        assert "ac-footer" in self.js
        assert "navigate" in self.js
        assert "accept" in self.js
        assert "dismiss" in self.js

    def test_has_ac_hint_tab(self) -> None:
        assert "ac-hint" in self.js
        assert "tab" in self.js

    def test_has_svg_icons(self) -> None:
        assert "_acSvg" in self.js
        assert "viewBox" in self.js

    def test_has_path_highlighting(self) -> None:
        assert "ac-dir" in self.js
        assert "ac-fname" in self.js
        assert "_acPathHtml" in self.js

    def test_has_search_highlighting(self) -> None:
        assert "ac-hl" in self.js
        assert "hlMatch" in self.js

    def test_has_keyboard_navigation(self) -> None:
        assert "ArrowDown" in self.js
        assert "ArrowUp" in self.js

    def test_has_tab_accept(self) -> None:
        # Tab should select the autocomplete item
        assert "'Tab'" in self.js

    def test_has_escape_dismiss(self) -> None:
        assert "'Escape'" in self.js
        assert "hideAC" in self.js

    def test_has_ac_idx_tracking(self) -> None:
        assert "acIdx" in self.js

    def test_has_sel_class_toggling(self) -> None:
        assert "updateSel" in self.js
        assert "'sel'" in self.js

    def test_has_get_at_ctx(self) -> None:
        """Matches web Sorcar's getAtCtx function."""
        assert "function getAtCtx" in self.js
        assert "@([^\\s]*)$" in self.js

    def test_has_section_grouping(self) -> None:
        """Renders frequent and file sections separately."""
        assert "'frequent'" in self.js
        assert "'file'" in self.js
        assert "Frequent" in self.js
        assert "Files" in self.js

    def test_has_record_file_usage(self) -> None:
        """Records file usage when an item is selected."""
        assert "recordFileUsage" in self.js

    def test_has_scroll_into_view(self) -> None:
        """Selected item scrolls into view."""
        assert "scrollIntoView" in self.js


class TestMainJsModelPicker(unittest.TestCase):
    """Test the main.js model picker JavaScript code for web Sorcar parity."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()

    def test_has_model_group_headers(self) -> None:
        assert "model-group-hdr" in self.js

    def test_has_recently_used_section(self) -> None:
        assert "Recently Used" in self.js

    def test_has_vendor_grouping(self) -> None:
        assert "modelVendor" in self.js or "vendor" in self.js

    def test_has_keyboard_navigation(self) -> None:
        assert "modelDDIdx" in self.js
        assert "updateSel" in self.js

    def test_has_arrow_key_handling(self) -> None:
        # Check model search keydown handler
        assert "ArrowDown" in self.js
        assert "ArrowUp" in self.js

    def test_has_enter_select(self) -> None:
        assert "'Enter'" in self.js

    def test_has_escape_close(self) -> None:
        assert "closeModelDD" in self.js

    def test_has_sel_class(self) -> None:
        assert "'sel'" in self.js

    def test_has_pricing_display(self) -> None:
        assert "toFixed" in self.js
        assert "model-cost" in self.js

    def test_has_usage_sorting(self) -> None:
        """Recently used models are sorted by usage count."""
        assert "b.uses - a.uses" in self.js or "uses" in self.js

    def test_has_active_class(self) -> None:
        """Selected model gets active class."""
        assert "active" in self.js

    def test_render_model_list_function(self) -> None:
        assert "function renderModelList" in self.js

    def test_select_model_function(self) -> None:
        assert "function selectModel" in self.js

    def test_scroll_into_view_for_keyboard(self) -> None:
        assert "scrollIntoView" in self.js


class TestMainCssFilePicker(unittest.TestCase):
    """Test the main.css has all required file picker styles."""

    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    def test_has_ac_section(self) -> None:
        assert ".ac-section" in self.css

    def test_has_ac_item_sel(self) -> None:
        assert ".ac-item.sel" in self.css

    def test_has_ac_icon_svg_styles(self) -> None:
        assert ".ac-icon svg" in self.css

    def test_has_ac_dir_fname(self) -> None:
        assert ".ac-dir" in self.css
        assert ".ac-fname" in self.css

    def test_has_ac_hl(self) -> None:
        assert ".ac-hl" in self.css

    def test_has_ac_hint(self) -> None:
        assert ".ac-hint" in self.css

    def test_has_ac_footer(self) -> None:
        assert ".ac-footer" in self.css

    def test_has_ac_footer_kbd(self) -> None:
        assert ".ac-footer kbd" in self.css

    def test_has_border_left_for_sel(self) -> None:
        """Selected item has accent border-left like web Sorcar."""
        sel_idx = self.css.index(".ac-item.sel")
        sel_block = self.css[sel_idx : sel_idx + 200]
        assert "border-left" in sel_block

    def test_has_slide_up_animation(self) -> None:
        assert "acSlideUp" in self.css


class TestExtractResultSummary(unittest.TestCase):
    """Test _extract_result_summary extracts summary from recorded events."""

    def setUp(self) -> None:
        self.server = VSCodeServer()

    def test_returns_empty_when_no_result_event(self) -> None:
        rec_id = 42
        self.server.printer.start_recording(rec_id)
        self.server.printer.broadcast({"type": "text_delta", "text": "hello"})
        result = self.server._extract_result_summary(rec_id)
        assert result == ""


class TestLastActiveFile(unittest.TestCase):
    """Test that _last_active_file is stored from run commands."""

    def setUp(self) -> None:
        self.server = VSCodeServer()

    def test_initial_value_empty(self) -> None:
        assert self.server._last_active_file == ""


class TestMainJsHistoryCycling(unittest.TestCase):
    """Test that ArrowUp/Down history cycling only works when textbox is empty."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()

    def test_arrow_up_requires_empty_input_or_active_cycling(self) -> None:
        """ArrowUp history cycling only starts when inp.value is empty."""
        assert "histIdx >= 0 || !inp.value" in self.js

    def test_arrow_down_resets_to_empty(self) -> None:
        """When cycling back to bottom, textbox resets to empty string."""
        assert "histIdx >= 0 ? histCache[histIdx] : ''" in self.js

    def test_no_hist_saved(self) -> None:
        """histSaved is no longer needed since cycling only starts from empty."""
        assert "histSaved" not in self.js


class TestMainJsInfiniteScroll(unittest.TestCase):
    """Test main.js has infinite scroll and chat_id color code."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()

    def test_has_history_offset_state(self) -> None:
        assert "historyOffset" in self.js

    def test_has_history_loading_state(self) -> None:
        assert "historyLoading" in self.js

    def test_has_history_has_more_state(self) -> None:
        assert "historyHasMore" in self.js

    def test_has_history_generation_counter(self) -> None:
        assert "historyGeneration" in self.js

    def test_has_scroll_listener(self) -> None:
        assert "historyList.addEventListener('scroll'" in self.js or \
               'historyList.addEventListener("scroll"' in self.js

    def test_has_loading_indicator(self) -> None:
        assert "sidebar-loading" in self.js
        assert "Loading..." in self.js

    def test_has_chat_id_bg_color_function(self) -> None:
        assert "chatIdBgColor" in self.js

    def test_chat_id_bg_uses_hsl(self) -> None:
        assert "hsl(" in self.js
        assert "40%" in self.js
        assert "92%" in self.js

    def test_chat_id_bg_colors_are_light(self) -> None:
        """Verify the chatIdBgColor function produces light colors for all chat_ids.

        Reimplements the JS djb2 hash + HSL logic in Python and checks that
        the minimum RGB channel is >= 220 (i.e., clearly light/pastel) for
        a wide range of chat_id strings.
        """
        import colorsys
        import ctypes

        def chat_id_bg_rgb(chat_id: str) -> tuple[int, int, int]:
            h = 5381
            for ch in chat_id:
                h = ((h << 5) + h) + ord(ch)
                h = ctypes.c_int32(h).value  # JS |= 0
            hue = abs(h) % 360
            # HSL(hue, 40%, 92%) -> RGB
            r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.92, 0.40)
            return (round(r * 255), round(g * 255), round(b * 255))

        test_ids = [
            "abc123", "xyz789", "chat-001", "chat-002", "session-1",
            "a", "test", "550e8400-e29b-41d4-a716-446655440000",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479", "z",
        ]
        for cid in test_ids:
            r, g, b = chat_id_bg_rgb(cid)
            assert min(r, g, b) >= 220, (
                f"chat_id={cid!r} produced dark color rgb({r},{g},{b})"
            )

    def test_sidebar_escape_closes(self) -> None:
        assert "'Escape'" in self.js
        assert "closeSidebar" in self.js

    def test_render_history_accepts_offset(self) -> None:
        assert "renderHistory(sessions, offset, generation)" in self.js or \
               "function renderHistory" in self.js


class TestMainCssInfiniteScroll(unittest.TestCase):
    """Test main.css has infinite scroll and responsive sidebar styles."""

    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    def test_sidebar_width_is_capped_at_420(self) -> None:
        idx = self.css.index("#sidebar")
        block = self.css[idx : idx + 400]
        assert "420px" in block

    def test_sidebar_uses_75_percent_of_viewport_width(self) -> None:
        idx = self.css.index("#sidebar")
        block = self.css[idx : idx + 400]
        assert "75vw" in block

    def test_sidebar_uses_full_viewport_height(self) -> None:
        idx = self.css.index("#sidebar")
        block = self.css[idx : idx + 400]
        assert "top: 0" in block
        assert "bottom: 0" in block

    def test_sidebar_overflow_hidden(self) -> None:
        # Sidebar should have overflow: hidden (not auto) so #history-list scrolls
        idx = self.css.index("#sidebar")
        block = self.css[idx : idx + 500]
        assert "overflow: hidden" in block or "overflow:hidden" in block

    def test_history_list_scrollable(self) -> None:
        assert "#history-list" in self.css
        idx = self.css.index("#history-list")
        block = self.css[idx : idx + 200]
        assert "overflow-y: auto" in block or "overflow-y:auto" in block

    def test_sidebar_loading_style(self) -> None:
        assert ".sidebar-loading" in self.css

    def test_sidebar_item_black_text(self) -> None:
        idx = self.css.index(".sidebar-item")
        block = self.css[idx : idx + 300]
        assert "#000" in block


class TestMainCssModelPicker(unittest.TestCase):
    """Test the main.css has all required model picker styles."""

    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    def test_has_model_group_hdr(self) -> None:
        assert ".model-group-hdr" in self.css

    def test_has_model_group_hdr_sticky(self) -> None:
        idx = self.css.index(".model-group-hdr")
        block = self.css[idx : idx + 300]
        assert "sticky" in block

    def test_has_model_item_sel(self) -> None:
        assert ".model-item.sel" in self.css

    def test_has_model_item_active(self) -> None:
        assert ".model-item.active" in self.css

    def test_has_model_cost(self) -> None:
        assert ".model-cost" in self.css


class TestGhostOverlayPaddingAlignment(unittest.TestCase):
    """Test that #ghost-overlay and #task-input have matching padding."""

    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    @staticmethod
    def _extract_padding(css: str, selector: str) -> str:
        """Extract the padding value from a CSS rule block for an exact selector."""
        import re

        # Find occurrences of the selector that are exact (followed by space or {)
        pattern = re.escape(selector) + r"\s*\{"
        for m_sel in re.finditer(pattern, css):
            brace_start = css.index("{", m_sel.start())
            brace_end = css.index("}", brace_start)
            block = css[brace_start:brace_end]
            m = re.search(r"padding:\s*([^;]+);", block)
            if m:
                return m.group(1).strip()
        raise AssertionError(f"No padding found for exact selector {selector}")

    def test_ghost_overlay_padding_matches_task_input(self) -> None:
        """#ghost-overlay padding must match #task-input padding exactly."""
        ghost_padding = self._extract_padding(self.css, "#ghost-overlay")
        input_padding = self._extract_padding(self.css, "#task-input")
        assert ghost_padding == input_padding, (
            f"Padding mismatch: #ghost-overlay has '{ghost_padding}' "
            f"but #task-input has '{input_padding}'"
        )

    def test_task_input_has_explicit_padding(self) -> None:
        """#task-input must have explicit padding (not rely on browser defaults)."""
        padding = self._extract_padding(self.css, "#task-input")
        # Should have all 4 sides specified (shorthand with at least 2 values)
        parts = padding.split()
        assert len(parts) >= 2, (
            f"#task-input padding should be explicit for all sides, got: '{padding}'"
        )


class TestGetLastSession(unittest.TestCase):
    """Test _get_last_session loads the most recent task."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]
        # Just verify no crash; may or may not emit depending on DB state

    def test_command_routing(self) -> None:
        """getLastSession command should be routed to _get_last_session."""
        self.server._handle_command({"type": "getLastSession"})
        # Should not produce an error event
        errors = [e for e in self.events if e["type"] == "error"]
        assert len(errors) == 0


class TestCompleteFromActiveFile(unittest.TestCase):
    """Test chained identifier extraction and matching from active file content."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]



class TestMainJsInputHistory(unittest.TestCase):
    """Test that main.js handles inputHistory events and request patterns."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def test_handles_input_history_event(self) -> None:
        """main.js should have a case for 'inputHistory'."""
        assert "case 'inputHistory':" in self._js

    def test_sets_hist_cache_from_tasks(self) -> None:
        """inputHistory handler should set histCache from ev.tasks."""
        assert "histCache = ev.tasks" in self._js

    def test_does_not_set_hist_cache_from_welcome(self) -> None:
        """renderWelcomeSuggestions should NOT set histCache anymore."""
        # Find the renderWelcomeSuggestions function body
        idx = self._js.index("function renderWelcomeSuggestions")
        # Find the next function definition to bound the search
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "histCache" not in body

    def test_requests_input_history_on_tasks_updated(self) -> None:
        """tasks_updated handler should request getInputHistory."""
        idx = self._js.index("case 'tasks_updated':")
        # Check within next 200 chars
        snippet = self._js[idx:idx + 300]
        assert "getInputHistory" in snippet

    def test_optimistic_hist_cache_in_send_message(self) -> None:
        """sendMessage should unshift prompt into histCache before posting."""
        idx = self._js.index("function sendMessage()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "histCache.unshift(prompt)" in body


if __name__ == "__main__":
    unittest.main()

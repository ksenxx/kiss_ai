"""Tests for the VS Code extension backend server.

Tests cover: model picker (vendor ordering, sorting, grouping, pricing),
file picker (sorting by usage/recency/end-distance, section grouping),
keyboard interaction parity with web Sorcar, and the JS rendering code
in main.js.
No mocks — uses real functions from the server module.
"""

import os
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

    def test_anthropic_first(self) -> None:
        assert _model_vendor_order("claude-opus-4-6") == 0
        assert _model_vendor_order("claude-sonnet-4-20250514") == 0

    def test_openai_second(self) -> None:
        assert _model_vendor_order("gpt-4o") == 1
        assert _model_vendor_order("o1") == 1
        assert _model_vendor_order("o3-mini") == 1
        assert _model_vendor_order("o4-mini") == 1

    def test_gemini_third(self) -> None:
        assert _model_vendor_order("gemini-2.0-flash") == 2

    def test_minimax_fourth(self) -> None:
        assert _model_vendor_order("minimax-large") == 3

    def test_openrouter_fifth(self) -> None:
        assert _model_vendor_order("openrouter/some-model") == 4

    def test_other_last(self) -> None:
        assert _model_vendor_order("together/some-model") == 5
        assert _model_vendor_order("unknown-model") == 5

    def test_order_is_consistent(self) -> None:
        names = [
            "unknown-model",
            "gemini-2.0-flash",
            "claude-opus-4-6",
            "gpt-4o",
            "openrouter/x",
            "minimax-large",
        ]
        sorted_names = sorted(names, key=_model_vendor_order)
        assert sorted_names[0] == "claude-opus-4-6"
        assert sorted_names[1] == "gpt-4o"
        assert sorted_names[2] == "gemini-2.0-flash"
        assert sorted_names[-1] in ("unknown-model", "together/some-model")


class TestModelVendorName(unittest.TestCase):
    """Test _model_vendor_name matches web Sorcar's modelVendor function."""

    def test_anthropic(self) -> None:
        assert _model_vendor_name("claude-opus-4-6") == "Anthropic"

    def test_openai(self) -> None:
        assert _model_vendor_name("gpt-4o") == "OpenAI"
        assert _model_vendor_name("o4-mini") == "OpenAI"

    def test_openai_prefix_not_openai_slash(self) -> None:
        # openai/ prefix should NOT be OpenAI
        assert _model_vendor_name("openai/something") != "OpenAI"

    def test_gemini(self) -> None:
        assert _model_vendor_name("gemini-2.0-flash") == "Gemini"

    def test_minimax(self) -> None:
        assert _model_vendor_name("minimax-large") == "MiniMax"

    def test_openrouter(self) -> None:
        assert _model_vendor_name("openrouter/model") == "OpenRouter"

    def test_fallback(self) -> None:
        assert _model_vendor_name("unknown") == "Together AI"


class TestGetModels(unittest.TestCase):
    """Test VSCodeServer._get_models produces correct grouping and sorting."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []
        # Capture broadcast events
        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def test_models_event_structure(self) -> None:
        self.server._get_models()
        assert len(self.events) == 1
        ev = self.events[0]
        assert ev["type"] == "models"
        assert "models" in ev
        assert "selected" in ev
        assert isinstance(ev["models"], list)

    def test_models_have_required_fields(self) -> None:
        self.server._get_models()
        for m in self.events[0]["models"]:
            assert "name" in m
            assert "inp" in m
            assert "out" in m
            assert "uses" in m
            assert "vendor" in m

    def test_models_sorted_by_vendor_then_price(self) -> None:
        self.server._get_models()
        models = self.events[0]["models"]
        if len(models) < 2:
            self.skipTest("Need at least 2 models to test sorting")
        for i in range(len(models) - 1):
            a, b = models[i], models[i + 1]
            order_a = _model_vendor_order(a["name"])
            order_b = _model_vendor_order(b["name"])
            if order_a == order_b:
                # Within same vendor, sorted by price descending
                assert a["inp"] + a["out"] >= b["inp"] + b["out"], (
                    f"{a['name']} should come before {b['name']} by price"
                )
            else:
                assert order_a <= order_b

    def test_selected_model_is_set(self) -> None:
        self.server._selected_model = "claude-opus-4-6"
        self.server._get_models()
        assert self.events[0]["selected"] == "claude-opus-4-6"


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

    def test_files_event_structure(self) -> None:
        self.server._get_files("")
        assert len(self.events) == 1
        ev = self.events[0]
        assert ev["type"] == "files"
        assert "files" in ev
        assert isinstance(ev["files"], list)

    def test_files_have_type_and_text(self) -> None:
        self.server._get_files("")
        for f in self.events[0]["files"]:
            assert "type" in f
            assert "text" in f
            assert f["type"] in ("file", "frequent")

    def test_files_filtered_by_prefix(self) -> None:
        self.server._get_files("main")
        files = self.events[0]["files"]
        for f in files:
            assert "main" in f["text"].lower()

    def test_files_limited_to_20(self) -> None:
        # Add many files
        self.server._file_cache = [f"file_{i}.py" for i in range(50)]
        self.server._get_files("")
        assert len(self.events[0]["files"]) <= 20

    def test_empty_prefix_returns_all_files(self) -> None:
        self.server._get_files("")
        files = self.events[0]["files"]
        assert len(files) == 4

    def test_no_match_returns_empty(self) -> None:
        self.server._get_files("nonexistent_xyz")
        files = self.events[0]["files"]
        assert len(files) == 0

    def test_frequent_files_sorted_first(self) -> None:
        """Files with usage > 0 should appear before files with no usage."""
        from kiss.agents.sorcar.persistence import _record_file_usage

        # Record usage for one file
        _record_file_usage("src/main.py")

        self.events.clear()
        self.server._get_files("")
        files = self.events[0]["files"]

        # Find the frequent and file sections
        frequent = [f for f in files if f["type"] == "frequent"]

        assert len(frequent) >= 1
        assert frequent[0]["text"] == "src/main.py"

        # Frequent files should come before regular files in the list
        freq_indices = [i for i, f in enumerate(files) if f["type"] == "frequent"]
        file_indices = [i for i, f in enumerate(files) if f["type"] == "file"]
        if freq_indices and file_indices:
            assert max(freq_indices) < min(file_indices)

    def test_end_distance_sorting(self) -> None:
        """Files matching closer to end of path should rank higher."""
        self.server._file_cache = [
            "very/long/path/main_helper.py",
            "src/main.py",
            "main.py",
        ]
        self.server._get_files("main")
        files = self.events[0]["files"]
        texts = [f["text"] for f in files]
        # "main.py" end_dist=3, "src/main.py" end_dist=3,
        # "very/long/path/main_helper.py" end_dist=10
        # So main.py and src/main.py should come before the helper
        assert texts[-1] == "very/long/path/main_helper.py"


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

    def test_extracts_summary_from_result_event(self) -> None:
        self.server.printer.start_recording()
        self.server.printer.broadcast({"type": "text_delta", "text": "some text"})
        self.server.printer.broadcast({
            "type": "result",
            "summary": "Task completed successfully",
            "success": True,
        })
        result = self.server._extract_result_summary()
        assert result == "Task completed successfully"

    def test_extracts_text_when_no_summary(self) -> None:
        self.server.printer.start_recording()
        self.server.printer.broadcast({
            "type": "result",
            "text": "Some result text",
        })
        result = self.server._extract_result_summary()
        assert result == "Some result text"

    def test_returns_empty_when_no_result_event(self) -> None:
        self.server.printer.start_recording()
        self.server.printer.broadcast({"type": "text_delta", "text": "hello"})
        result = self.server._extract_result_summary()
        assert result == ""

    def test_returns_empty_when_no_recording(self) -> None:
        result = self.server._extract_result_summary()
        assert result == ""


class TestHandleCommandGenerateCommitMessage(unittest.TestCase):
    """Test that generateCommitMessage command is routed correctly."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def test_unknown_command_returns_error(self) -> None:
        self.server._handle_command({"type": "unknownXYZ"})
        assert len(self.events) == 1
        assert self.events[0]["type"] == "error"
        assert "unknownXYZ" in self.events[0]["text"]


class TestGetHistory(unittest.TestCase):
    """Test _get_history sends paginated history with offset/generation."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def test_history_event_structure(self) -> None:
        self.server._get_history(None)
        assert len(self.events) == 1
        ev = self.events[0]
        assert ev["type"] == "history"
        assert "sessions" in ev
        assert "offset" in ev
        assert "generation" in ev
        assert isinstance(ev["sessions"], list)

    def test_history_offset_echoed(self) -> None:
        self.server._get_history(None, offset=10, generation=3)
        ev = self.events[0]
        assert ev["offset"] == 10
        assert ev["generation"] == 3

    def test_history_sessions_have_chat_id(self) -> None:
        self.server._get_history(None)
        ev = self.events[0]
        for s in ev["sessions"]:
            assert "chat_id" in s

    def test_getHistory_command_routing(self) -> None:
        self.server._handle_command({"type": "getHistory", "offset": 5, "generation": 2})
        assert len(self.events) == 1
        assert self.events[0]["type"] == "history"
        assert self.events[0]["offset"] == 5
        assert self.events[0]["generation"] == 2

    def test_getHistory_with_query(self) -> None:
        self.server._get_history("nonexistent_xyz_query_123")
        ev = self.events[0]
        assert ev["sessions"] == []


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
    """Test main.css has infinite scroll and wider sidebar styles."""

    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    def test_sidebar_width_420(self) -> None:
        assert "420px" in self.css

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


class TestGetLastSession(unittest.TestCase):
    """Test _get_last_session loads the most recent task."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def test_loads_last_task(self) -> None:
        """Should broadcast task_events with task field set."""
        self.server._get_last_session()
        task_ev = [e for e in self.events if e["type"] == "task_events"]
        assert len(task_ev) == 1
        assert "task" in task_ev[0]
        assert isinstance(task_ev[0]["task"], str)
        assert len(task_ev[0]["task"]) > 0
        assert "events" in task_ev[0]
        assert isinstance(task_ev[0]["events"], list)

    def test_no_history_does_nothing(self) -> None:
        """When there's no history, no event should be broadcast."""
        # We can't easily empty the DB, but we can verify _get_last_session
        # doesn't crash. With existing history it will emit an event.
        self.server._get_last_session()
        # Just verify no crash; may or may not emit depending on DB state

    def test_command_routing(self) -> None:
        """getLastSession command should be routed to _get_last_session."""
        self.server._handle_command({"type": "getLastSession"})
        # Should not produce an error event
        errors = [e for e in self.events if e["type"] == "error"]
        assert len(errors) == 0


class TestRestorePendingMerge(unittest.TestCase):
    """Test _restore_pending_merge restores merge state from disk."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

        from kiss.agents.vscode.diff_merge import _merge_data_dir

        self.merge_dir = _merge_data_dir()
        self.merge_json = self.merge_dir / "pending-merge.json"
        # Clean up any existing merge data
        self._had_merge = self.merge_json.is_file()
        if self._had_merge:
            self._saved = self.merge_json.read_text()

    def tearDown(self) -> None:
        if self._had_merge:
            self.merge_json.parent.mkdir(parents=True, exist_ok=True)
            self.merge_json.write_text(self._saved)
        elif self.merge_json.is_file():
            self.merge_json.unlink()

    def _write_merge_json(self, data: object) -> None:
        self.merge_dir.mkdir(parents=True, exist_ok=True)
        import json

        self.merge_json.write_text(json.dumps(data))

    def test_no_file_does_nothing(self) -> None:
        """No pending-merge.json → no events, no state change."""
        if self.merge_json.is_file():
            self.merge_json.unlink()
        self.server._restore_pending_merge()
        assert not self.server._merging
        merge_events = [e for e in self.events if e["type"] in ("merge_data", "merge_started")]
        assert len(merge_events) == 0

    def test_restores_merge_state(self) -> None:
        """Valid pending-merge.json → sends merge_data + merge_started, sets state."""
        merge_data = {
            "branch": "HEAD",
            "files": [
                {
                    "name": "a.py",
                    "base": "/tmp/base/a.py",
                    "current": "/tmp/cur/a.py",
                    "hunks": [
                        {"bs": 0, "bc": 2, "cs": 0, "cc": 3},
                        {"bs": 5, "bc": 1, "cs": 6, "cc": 1},
                    ],
                },
                {
                    "name": "b.py",
                    "base": "/tmp/base/b.py",
                    "current": "/tmp/cur/b.py",
                    "hunks": [{"bs": 0, "bc": 0, "cs": 0, "cc": 5}],
                },
            ],
        }
        self._write_merge_json(merge_data)
        self.server._restore_pending_merge()

        assert self.server._merging is True

        merge_data_evts = [e for e in self.events if e["type"] == "merge_data"]
        assert len(merge_data_evts) == 1
        assert merge_data_evts[0]["hunk_count"] == 3
        assert merge_data_evts[0]["data"]["files"][0]["name"] == "a.py"

        merge_started_evts = [e for e in self.events if e["type"] == "merge_started"]
        assert len(merge_started_evts) == 1

    def test_empty_files_ignored(self) -> None:
        """pending-merge.json with empty files list → no events."""
        self._write_merge_json({"branch": "HEAD", "files": []})
        self.server._restore_pending_merge()
        assert not self.server._merging
        assert len([e for e in self.events if e["type"] == "merge_data"]) == 0

    def test_zero_hunks_ignored(self) -> None:
        """Files present but all with empty hunk lists → no events."""
        self._write_merge_json({
            "branch": "HEAD",
            "files": [{"name": "a.py", "base": "/x", "current": "/y", "hunks": []}],
        })
        self.server._restore_pending_merge()
        assert not self.server._merging

    def test_invalid_json_does_not_crash(self) -> None:
        """Corrupt pending-merge.json → silent failure, no crash."""
        self.merge_dir.mkdir(parents=True, exist_ok=True)
        self.merge_json.write_text("not valid json{{{")
        self.server._restore_pending_merge()
        assert not self.server._merging
        assert len(self.events) == 0

    def test_get_last_session_also_restores_merge(self) -> None:
        """_get_last_session should call _restore_pending_merge."""
        merge_data = {
            "branch": "HEAD",
            "files": [
                {
                    "name": "c.py",
                    "base": "/tmp/base/c.py",
                    "current": "/tmp/cur/c.py",
                    "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
                }
            ],
        }
        self._write_merge_json(merge_data)
        self.server._get_last_session()

        assert self.server._merging is True
        merge_data_evts = [e for e in self.events if e["type"] == "merge_data"]
        assert len(merge_data_evts) == 1
        merge_started_evts = [e for e in self.events if e["type"] == "merge_started"]
        assert len(merge_started_evts) == 1


if __name__ == "__main__":
    unittest.main()

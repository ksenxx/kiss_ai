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

from kiss.agents.vscode.server import (
    VSCodeServer,
    _model_vendor,
)


def _model_vendor_name(name: str) -> str:
    return _model_vendor(name)[0]


def _model_vendor_order(name: str) -> int:
    return _model_vendor(name)[1]


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
        from kiss.agents.sorcar.task_history import _record_file_usage

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
        assert "updateACSel" in self.js
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
        assert "updateModelSel" in self.js

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


if __name__ == "__main__":
    unittest.main()

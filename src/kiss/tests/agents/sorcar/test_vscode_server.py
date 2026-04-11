"""Tests for the VS Code extension backend server.

Tests cover: model picker (vendor ordering, sorting, grouping, pricing),
file picker (sorting by usage/recency/end-distance, section grouping),
keyboard interaction parity with web Sorcar, and the JS rendering code
in main.js.
No mocks — uses real functions from the server module.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.vscode.helpers import model_vendor
from kiss.agents.vscode.server import VSCodeServer


def _set_agent_wt(agent: object, repo: Path, branch: str, original: str) -> None:
    """Helper to set agent._wt with a GitWorktree for testing."""
    slug = branch.replace("/", "_")
    agent._wt = GitWorktree(  # type: ignore[attr-defined]
        repo_root=repo,
        branch=branch,
        original_branch=original,
        wt_dir=repo / ".kiss-worktrees" / slug,
    )


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


class TestLastActiveFile(unittest.TestCase):
    """Test that _last_active_file is stored from run commands."""

    def setUp(self) -> None:
        self.server = VSCodeServer()


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


class TestMainJsWorktreeToggle(unittest.TestCase):
    """Test that main.js has the worktree toggle button wiring."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def test_has_worktree_toggle_btn_element(self) -> None:
        assert "worktree-toggle-btn" in self._js

    def test_toggle_adds_active_class(self) -> None:
        assert "worktreeToggleBtn.classList.toggle('active')" in self._js

    def test_worktree_toggle_btn_variable(self) -> None:
        assert "worktreeToggleBtn" in self._js


class TestMainCssWorktreeToggle(unittest.TestCase):
    """Test that main.css has styles for the worktree toggle button."""

    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    def test_has_worktree_toggle_btn_base_style(self) -> None:
        assert "#worktree-toggle-btn" in self.css

    def test_has_worktree_toggle_btn_active_style(self) -> None:
        assert "#worktree-toggle-btn.active" in self.css

    def test_active_uses_accent_color(self) -> None:
        idx = self.css.index("#worktree-toggle-btn.active")
        block = self.css[idx : idx + 200]
        assert "accent" in block


class TestSorcarTabWorktreeToggle(unittest.TestCase):
    """Test that SorcarTab.ts HTML includes the worktree toggle button."""

    html: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.html = (base / "vscode" / "src" / "SorcarTab.ts").read_text()

    def test_has_worktree_toggle_btn(self) -> None:
        assert 'id="worktree-toggle-btn"' in self.html

    def test_has_use_worktree_tooltip(self) -> None:
        assert 'data-tooltip="Use worktree"' in self.html

    def test_button_is_between_upload_and_history(self) -> None:
        upload_idx = self.html.index('id="upload-btn"')
        worktree_idx = self.html.index('id="worktree-toggle-btn"')
        history_idx = self.html.index('id="history-btn"')
        assert upload_idx < worktree_idx < history_idx

    def test_has_tree_svg_icon(self) -> None:
        """Button should have a git-branch-like tree SVG icon."""
        idx = self.html.index('id="worktree-toggle-btn"')
        # Look at the next 500 chars for the SVG
        block = self.html[idx : idx + 500]
        assert "<svg" in block
        assert "viewBox" in block


class TestWorktreeServerIntegration(unittest.TestCase):
    """Integration tests for worktree support in VSCodeServer."""

    def _git(self, *args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=self.repo, capture_output=True,
        )

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        self._git("init")
        self._git("config", "user.email", "test@test.com")
        self._git("config", "user.name", "Test")
        (self.repo / "file.txt").write_text("hello")
        self._git("add", ".")
        self._git("commit", "-m", "init")

        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_handle_worktree_action_merge(self) -> None:
        """Merge action calls agent.merge() and returns result."""
        self._git("checkout", "-b", "kiss/merge-test")
        (self.repo / "merged.txt").write_text("merged content")
        self._git("add", ".")
        self._git("commit", "-m", "add merged")
        self._git("checkout", "main")

        self.server._use_worktree = True
        _set_agent_wt(self.server._worktree_agent, self.repo, "kiss/merge-test", "main")

        result = self.server._handle_worktree_action("merge")
        assert result["success"] is True
        assert "Successfully merged" in result["message"]
        # Branch should be cleaned up
        assert self.server._worktree_agent._wt_branch is None

    def test_handle_worktree_action_discard(self) -> None:
        """Discard action removes worktree branch."""
        self._git("checkout", "-b", "kiss/discard-test")
        self._git("checkout", "main")

        self.server._use_worktree = True
        _set_agent_wt(self.server._worktree_agent, self.repo, "kiss/discard-test", "main")

        result = self.server._handle_worktree_action("discard")
        assert result["success"] is True
        assert "Discarded" in result["message"]
        assert self.server._worktree_agent._wt_branch is None

    def test_worktree_action_command_routing(self) -> None:
        """worktreeAction command is routed to _handle_worktree_action."""
        self._git("checkout", "-b", "kiss/route-test")
        (self.repo / "route.txt").write_text("route content")
        self._git("add", ".")
        self._git("commit", "-m", "add route")
        self._git("checkout", "main")

        self.server._use_worktree = True
        _set_agent_wt(self.server._worktree_agent, self.repo, "kiss/route-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "merge"})
        wt_events = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(wt_events) == 1
        assert wt_events[0]["success"] is True

    def test_merge_broadcasts_progress_before_result(self) -> None:
        """Merge action broadcasts worktree_progress before worktree_result."""
        self._git("checkout", "-b", "kiss/progress-test")
        (self.repo / "progress.txt").write_text("progress content")
        self._git("add", ".")
        self._git("commit", "-m", "add progress")
        self._git("checkout", "main")

        self.server._use_worktree = True
        _set_agent_wt(self.server._worktree_agent, self.repo, "kiss/progress-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "merge"})
        progress_events = [e for e in self.events if e["type"] == "worktree_progress"]
        assert len(progress_events) == 1
        assert "Generating commit message" in progress_events[0]["message"]
        # Progress must come before result
        relevant = ("worktree_progress", "worktree_result")
        types = [e["type"] for e in self.events if e["type"] in relevant]
        assert types == ["worktree_progress", "worktree_result"]

    def test_discard_does_not_broadcast_progress(self) -> None:
        """Discard action does not broadcast worktree_progress."""
        self._git("checkout", "-b", "kiss/no-progress-test")
        self._git("checkout", "main")

        self.server._use_worktree = True
        _set_agent_wt(self.server._worktree_agent, self.repo, "kiss/no-progress-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "discard"})
        progress_events = [e for e in self.events if e["type"] == "worktree_progress"]
        assert len(progress_events) == 0


class TestAgentToggle(unittest.TestCase):
    """Tests for worktree toggle switching between agents."""

    _JS_PATH = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media" / "main.js"
    _TS_PATH = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src" / "SorcarTab.ts"
    _js: str
    _ts: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = cls._JS_PATH.read_text()
        cls._ts = cls._TS_PATH.read_text()

    def test_server_defaults_to_stateful_agent(self) -> None:
        """Server agent is StatefulSorcarAgent by default (worktree off)."""
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        server = VSCodeServer()
        assert server._use_worktree is False
        assert isinstance(server.agent, StatefulSorcarAgent)
        assert not isinstance(server.agent, WorktreeSorcarAgent)

    def test_server_uses_worktree_agent_when_enabled(self) -> None:
        """Server agent is WorktreeSorcarAgent when worktree mode is on."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        server = VSCodeServer()
        server._use_worktree = True
        assert isinstance(server.agent, WorktreeSorcarAgent)

    def test_server_agent_switches_dynamically(self) -> None:
        """Agent switches when _use_worktree is toggled."""
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        server = VSCodeServer()
        assert isinstance(server.agent, StatefulSorcarAgent)
        assert not isinstance(server.agent, WorktreeSorcarAgent)
        server._use_worktree = True
        assert isinstance(server.agent, WorktreeSorcarAgent)
        server._use_worktree = False
        assert isinstance(server.agent, StatefulSorcarAgent)
        assert not isinstance(server.agent, WorktreeSorcarAgent)

    def test_js_sends_use_worktree_in_submit(self) -> None:
        """main.js includes useWorktree in submit message."""
        assert "useWorktree" in self._js
        assert "worktreeToggleBtn.classList.contains('active')" in self._js

    def test_ts_passes_use_worktree_to_start_task(self) -> None:
        """SorcarTab.ts passes useWorktree to _startTask."""
        assert "message.useWorktree" in self._ts
        assert "useWorktree" in self._ts

    def test_ts_start_task_includes_use_worktree(self) -> None:
        """_startTask sends useWorktree in agent command."""
        # Check that _startTask accepts useWorktree and passes it through
        assert "useWorktree?: boolean" in self._ts
        # Check it's included in the command object
        lines = self._ts.split("\n")
        in_start_task = False
        found_use_worktree_in_command = False
        for line in lines:
            if "_startTask" in line and "useWorktree" in line:
                in_start_task = True
            if in_start_task and "useWorktree" in line and "sendCommand" not in line:
                found_use_worktree_in_command = True
        assert found_use_worktree_in_command

    def test_worktree_action_rejected_when_not_enabled(self) -> None:
        """Worktree action fails gracefully when worktree mode is off."""
        server = VSCodeServer()
        result = server._handle_worktree_action("merge")
        assert result["success"] is False
        assert "not enabled" in result["message"]


class TestWorktreeActionNotifications(unittest.TestCase):
    """Tests for VS Code notification behavior on worktree actions."""

    _TS_PATH = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src" / "SorcarTab.ts"
    _ts: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._ts = cls._TS_PATH.read_text()

    def test_worktree_action_resolve_field_exists(self) -> None:
        """SorcarTab has a _worktreeActionResolve field for progress tracking."""
        assert "_worktreeActionResolve" in self._ts

    def test_merge_shows_progress_notification(self) -> None:
        """Clicking merge shows a progress notification with merge message."""
        assert "Committing and merging worktree" in self._ts
        assert "withProgress" in self._ts
        assert "ProgressLocation.Notification" in self._ts

    def test_discard_shows_progress_notification(self) -> None:
        """Clicking discard shows a progress notification with discard message."""
        assert "Discarding worktree" in self._ts

    def test_success_result_shows_info_message(self) -> None:
        """Successful worktree_result shows an information message."""
        assert "showInformationMessage" in self._ts

    def test_failure_result_shows_error_message(self) -> None:
        """Failed worktree_result shows an error message."""
        assert "showErrorMessage" in self._ts

    def test_progress_resolved_on_result(self) -> None:
        """The progress notification is resolved when worktree_result arrives."""
        # Check that _worktreeActionResolve is called and nulled
        assert "this._worktreeActionResolve();" in self._ts
        assert "this._worktreeActionResolve = null;" in self._ts

    def test_progress_title_varies_by_action(self) -> None:
        """Progress title differs for merge vs discard actions."""
        # The code should branch on wtAction to pick the right message
        assert "wtAction === 'merge'" in self._ts
        assert "wtAction === 'discard'" in self._ts

    def test_worktree_progress_field_exists(self) -> None:
        """SorcarTab has a _worktreeProgress field for progress reporting."""
        assert "_worktreeProgress" in self._ts

    def test_worktree_progress_captured_in_with_progress(self) -> None:
        """The progress reporter is captured in the withProgress callback."""
        assert "this._worktreeProgress = progress" in self._ts

    def test_worktree_progress_handler_updates_notification(self) -> None:
        """worktree_progress messages update the progress notification."""
        assert "worktree_progress" in self._ts
        assert ".report(" in self._ts

    def test_worktree_progress_cleared_on_result(self) -> None:
        """_worktreeProgress is cleared when worktree_result arrives."""
        assert "this._worktreeProgress = null" in self._ts


class TestMainJsParallelToggle(unittest.TestCase):
    """Test that main.js has the parallel toggle button wiring."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def test_has_parallel_toggle_btn_element(self) -> None:
        assert "parallel-toggle-btn" in self._js

    def test_toggle_adds_active_class(self) -> None:
        assert "parallelToggleBtn.classList.toggle('active')" in self._js

    def test_parallel_toggle_btn_variable(self) -> None:
        assert "parallelToggleBtn" in self._js

    def test_js_sends_use_parallel_in_submit(self) -> None:
        """main.js includes useParallel in submit message."""
        assert "useParallel" in self._js
        assert "parallelToggleBtn.classList.contains('active')" in self._js


class TestMainCssParallelToggle(unittest.TestCase):
    """Test that main.css has styles for the parallel toggle button."""

    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    def test_has_parallel_toggle_btn_base_style(self) -> None:
        assert "#parallel-toggle-btn" in self.css

    def test_has_parallel_toggle_btn_active_style(self) -> None:
        assert "#parallel-toggle-btn.active" in self.css

    def test_active_uses_accent_color(self) -> None:
        idx = self.css.index("#parallel-toggle-btn.active")
        block = self.css[idx : idx + 200]
        assert "accent" in block


class TestSorcarTabParallelToggle(unittest.TestCase):
    """Test that SorcarTab.ts HTML includes the parallel toggle button."""

    html: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.html = (base / "vscode" / "src" / "SorcarTab.ts").read_text()

    def test_has_parallel_toggle_btn(self) -> None:
        assert 'id="parallel-toggle-btn"' in self.html

    def test_has_use_parallelism_tooltip(self) -> None:
        assert 'data-tooltip="Use parallelism"' in self.html

    def test_button_is_between_worktree_and_history(self) -> None:
        worktree_idx = self.html.index('id="worktree-toggle-btn"')
        parallel_idx = self.html.index('id="parallel-toggle-btn"')
        history_idx = self.html.index('id="history-btn"')
        assert worktree_idx < parallel_idx < history_idx

    def test_has_svg_icon(self) -> None:
        """Button should have a parallel-lines SVG icon."""
        idx = self.html.index('id="parallel-toggle-btn"')
        block = self.html[idx : idx + 500]
        assert "<svg" in block
        assert "viewBox" in block

    def test_ts_passes_use_parallel_to_start_task(self) -> None:
        """SorcarTab.ts passes useParallel to _startTask."""
        assert "message.useParallel" in self.html
        assert "useParallel" in self.html

    def test_ts_start_task_includes_use_parallel(self) -> None:
        """_startTask sends useParallel in agent command."""
        assert "useParallel?: boolean" in self.html
        lines = self.html.split("\n")
        in_start_task = False
        found = False
        for line in lines:
            if "_startTask" in line and "useParallel" in line:
                in_start_task = True
            if in_start_task and "useParallel" in line and "sendCommand" not in line:
                found = True
        assert found


class TestServerParallelToggle(unittest.TestCase):
    """Tests for parallel toggle in VSCodeServer."""

    def test_server_defaults_parallel_off(self) -> None:
        """_use_parallel is False by default."""
        server = VSCodeServer()
        assert server._use_parallel is False

    def test_server_parses_use_parallel_from_command(self) -> None:
        """_run_task_inner sets _use_parallel from cmd dict."""
        import inspect

        src = inspect.getsource(VSCodeServer._run_task_inner)
        assert 'useParallel' in src

    def test_agent_run_receives_is_parallel(self) -> None:
        """agent.run() call includes is_parallel=self._use_parallel."""
        import inspect

        src = inspect.getsource(VSCodeServer._run_task_inner)
        assert "is_parallel" in src


class TestMergeSession(unittest.TestCase):
    """Tests for _start_merge_session, _handle_merge_action, _finish_merge,
    and _restore_pending_merge."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.merge_dir = Path(self.tmpdir) / "merge_dir"
        self.merge_dir.mkdir()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_merge_json(self, files: list[dict] | None = None) -> str:
        """Write a pending-merge.json and return its path."""
        import json as _json

        if files is None:
            # Create a dummy base file and current file for a real merge
            base = self.merge_dir / "merge-temp" / "a.txt"
            base.parent.mkdir(parents=True, exist_ok=True)
            base.write_text("old line\n")
            current = Path(self.tmpdir) / "a.txt"
            current.write_text("new line\n")
            files = [{
                "name": "a.txt",
                "base": str(base),
                "current": str(current),
                "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
            }]
        merge_json = self.merge_dir / "pending-merge.json"
        merge_json.write_text(_json.dumps({"branch": "HEAD", "files": files}))
        return str(merge_json)

    def test_start_merge_session_broadcasts_merge_data_and_started(self) -> None:
        """_start_merge_session broadcasts merge_data and merge_started events."""
        path = self._write_merge_json()
        result = self.server._start_merge_session(path)
        assert result is True
        types = [e["type"] for e in self.events]
        assert "merge_data" in types
        assert "merge_started" in types
        # merge_data must come before merge_started
        assert types.index("merge_data") < types.index("merge_started")
        # Server is now in merging state
        assert self.server._merging is True

    def test_start_merge_session_includes_hunk_count(self) -> None:
        """merge_data event includes correct hunk_count."""
        path = self._write_merge_json()
        self.server._start_merge_session(path)
        md = [e for e in self.events if e["type"] == "merge_data"][0]
        assert md["hunk_count"] == 1

    def test_start_merge_session_returns_false_for_empty_files(self) -> None:
        """Returns False when merge JSON has no files."""
        path = self._write_merge_json(files=[])
        result = self.server._start_merge_session(path)
        assert result is False
        assert self.server._merging is False

    def test_start_merge_session_returns_false_for_zero_hunks(self) -> None:
        """Returns False when all files have zero hunks."""
        current = Path(self.tmpdir) / "b.txt"
        current.write_text("content\n")
        path = self._write_merge_json(files=[{
            "name": "b.txt",
            "base": str(current),
            "current": str(current),
            "hunks": [],
        }])
        result = self.server._start_merge_session(path)
        assert result is False

    def test_start_merge_session_returns_false_for_missing_file(self) -> None:
        """Returns False when merge JSON file doesn't exist."""
        result = self.server._start_merge_session("/nonexistent/merge.json")
        assert result is False
        assert self.server._merging is False

    def test_start_merge_session_returns_false_for_invalid_json(self) -> None:
        """Returns False when merge JSON is malformed."""
        bad = self.merge_dir / "bad.json"
        bad.write_text("not json")
        result = self.server._start_merge_session(str(bad))
        assert result is False

    def test_handle_merge_action_all_done_finishes_merge(self) -> None:
        """mergeAction all-done calls _finish_merge and resets state."""
        path = self._write_merge_json()
        self.server._start_merge_session(path)
        self.events.clear()

        self.server._handle_merge_action("all-done")
        assert self.server._merging is False
        types = [e["type"] for e in self.events]
        assert "merge_ended" in types

    def test_handle_merge_action_unknown_is_noop(self) -> None:
        """Non-'all-done' actions are no-ops on the Python side."""
        self.server._merging = True
        self.server._handle_merge_action("accept")
        # Still merging — only all-done finishes
        assert self.server._merging is True

    def test_finish_merge_cleans_up_data_dir(self) -> None:
        """_finish_merge removes the merge data directory."""
        import kiss.agents.vscode.diff_merge as dm
        import kiss.agents.vscode.server as srv_mod

        orig_dm = dm._merge_data_dir
        orig_srv = srv_mod._merge_data_dir
        dm._merge_data_dir = lambda: self.merge_dir  # type: ignore[assignment]
        srv_mod._merge_data_dir = lambda: self.merge_dir  # type: ignore[assignment]
        try:
            path = self._write_merge_json()
            self.server._start_merge_session(path)
            assert self.merge_dir.exists()
            self.server._finish_merge()
            assert not self.merge_dir.exists()
        finally:
            dm._merge_data_dir = orig_dm  # type: ignore[assignment]
            srv_mod._merge_data_dir = orig_srv  # type: ignore[assignment]

    def test_merging_blocks_new_tasks(self) -> None:
        """Cannot start a task while merge review is in progress."""
        self.server._merging = True
        # Simulate _run_task_inner rejecting a task
        self.server._run_task_inner({"prompt": "test", "model": "m"})
        errors = [e for e in self.events if e["type"] == "error"]
        assert any("merge review" in e["text"] for e in errors)

    def test_restore_pending_merge_from_disk(self) -> None:
        """_restore_pending_merge re-opens a merge session from disk."""
        import kiss.agents.vscode.diff_merge as dm
        import kiss.agents.vscode.server as srv_mod

        orig = dm._merge_data_dir
        dm._merge_data_dir = lambda: self.merge_dir  # type: ignore[assignment]
        # Also patch the imported reference in server module
        orig_srv = srv_mod._merge_data_dir
        srv_mod._merge_data_dir = lambda: self.merge_dir  # type: ignore[assignment]
        try:
            self._write_merge_json()
            self.server._restore_pending_merge()
            assert self.server._merging is True
            types = [e["type"] for e in self.events]
            assert "merge_data" in types
        finally:
            dm._merge_data_dir = orig  # type: ignore[assignment]
            srv_mod._merge_data_dir = orig_srv  # type: ignore[assignment]

    def test_merge_command_routing(self) -> None:
        """mergeAction command is routed through _handle_command."""
        path = self._write_merge_json()
        self.server._start_merge_session(path)
        self.events.clear()

        import kiss.agents.vscode.diff_merge as dm
        import kiss.agents.vscode.server as srv_mod

        orig = dm._merge_data_dir
        dm._merge_data_dir = lambda: self.merge_dir  # type: ignore[assignment]
        orig_srv = srv_mod._merge_data_dir
        srv_mod._merge_data_dir = lambda: self.merge_dir  # type: ignore[assignment]
        try:
            self.server._handle_command({"type": "mergeAction", "action": "all-done"})
        finally:
            dm._merge_data_dir = orig  # type: ignore[assignment]
            srv_mod._merge_data_dir = orig_srv  # type: ignore[assignment]
        assert self.server._merging is False
        types = [e["type"] for e in self.events]
        assert "merge_ended" in types


class TestMergeDiffViewColumn(unittest.TestCase):
    """Verify MergeManager opens files in ViewColumn.One to preserve the
    chat webview, and only opens one file (not all changed files)."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "MergeManager.ts").read_text()

    def _get_method_body(self, method: str) -> str:
        """Extract from method definition to the next top-level member."""
        # Find method definition line (e.g. "  private async _doOpenMerge(")
        import re as _re

        escaped = _re.escape(method)
        pat = _re.compile(
            rf"^\s+(?:private\s+|public\s+)?(?:async\s+)?{escaped}\(",
            _re.MULTILINE,
        )
        m = pat.search(self._ts)
        assert m, f"Method {method} not found"
        start = m.start()
        # Find the next top-level member declaration
        for marker in ("\n  private ", "\n  public ", "\n  async ", "\n  dispose"):
            try:
                end = self._ts.index(marker, m.end())
                return self._ts[start:end]
            except ValueError:
                continue
        return self._ts[start:]

    def test_do_open_merge_uses_view_column_one(self) -> None:
        """_doOpenMerge passes viewColumn: vscode.ViewColumn.One to showTextDocument."""
        body = self._get_method_body("_doOpenMerge")
        assert "viewColumn: vscode.ViewColumn.One" in body

    def test_do_open_merge_has_single_show_text_document(self) -> None:
        """_doOpenMerge calls showTextDocument only once (for the first file),
        not inside the for loop for every file."""
        body = self._get_method_body("_doOpenMerge")
        count = body.count("showTextDocument")
        assert count == 1, f"Expected 1 showTextDocument call, got {count}"

    def test_do_open_merge_uses_workspace_apply_edit(self) -> None:
        """_doOpenMerge uses WorkspaceEdit for base-line insertions instead
        of ed.edit() which requires a visible editor."""
        body = self._get_method_body("_doOpenMerge")
        assert "WorkspaceEdit" in body
        assert "applyEdit" in body

    def test_navigate_hunk_uses_view_column_one(self) -> None:
        """_navigateHunk opens files in ViewColumn.One."""
        body = self._get_method_body("_navigateHunk")
        assert "viewColumn: vscode.ViewColumn.One" in body

    def test_get_or_open_editor_uses_view_column_one(self) -> None:
        """_getOrOpenEditor opens files in ViewColumn.One."""
        body = self._get_method_body("_getOrOpenEditor")
        assert "viewColumn: vscode.ViewColumn.One" in body

    def test_do_open_merge_does_not_execute_revert_command(self) -> None:
        """_doOpenMerge no longer calls executeCommand to revert
        (which requires the document to be the active editor)."""
        body = self._get_method_body("_doOpenMerge")
        # The old code used executeCommand('workbench.action.files.revert').
        # The new code uses WorkspaceEdit to revert dirty documents.
        assert "executeCommand" not in body

    def test_do_open_merge_tracks_first_file_fp(self) -> None:
        """_doOpenMerge tracks firstFileFp to show only one file."""
        body = self._get_method_body("_doOpenMerge")
        assert "firstFileFp" in body


class TestSorcarTabOpensFilesInLeftSplit(unittest.TestCase):
    """Verify SorcarTab opens files in ViewColumn.One (the left split)
    so they never replace the chat webview panel which lives in a later column."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "SorcarTab.ts").read_text()

    def _extract_case_block(self, case_label: str) -> str:
        """Extract a switch-case block from _handleMessage."""
        import re as _re

        pat = _re.compile(
            rf"case\s+'{_re.escape(case_label)}'",
            _re.MULTILINE,
        )
        m = pat.search(self._ts)
        assert m, f"Case '{case_label}' not found"
        start = m.start()
        # Find the next case or closing brace
        next_case = _re.search(r"\n\s+case\s+'", self._ts[m.end():])
        if next_case:
            return self._ts[start : m.end() + next_case.start()]
        return self._ts[start:]

    def test_open_file_uses_view_column_one(self) -> None:
        """openFile handler opens files in ViewColumn.One."""
        block = self._extract_case_block("openFile")
        assert "viewColumn: vscode.ViewColumn.One" in block

    def test_submit_file_path_uses_view_column_one(self) -> None:
        """submit handler (file-path shortcut) opens files in ViewColumn.One."""
        block = self._extract_case_block("submit")
        assert "viewColumn: vscode.ViewColumn.One" in block


class TestFilePathDoesNotPopulateTaskPanel(unittest.TestCase):
    """Regression: typing a file path in the textbox and opening it must NOT
    populate the fixed task panel.

    Root cause was that sendMessage() in main.js used to set the task panel
    text (setTaskText, currentTaskName, resetAdjacentState, vscode.setState)
    *before* the extension determined whether to run a task or open a file.
    The fix moved all task-panel state management into the 'setTaskText' event
    handler, which is only sent by _startTask() — never by the file-open path.
    """

    _js: str = ""
    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()
        cls._ts = (base / "vscode" / "src" / "SorcarTab.ts").read_text()

    # -- main.js: sendMessage must NOT touch task panel state ----------------

    def _get_send_message_body(self) -> str:
        start = self._js.index("function sendMessage()")
        end = self._js.index("\n  function ", start + 1)
        return self._js[start:end]

    def test_send_message_does_not_call_set_task_text(self) -> None:
        body = self._get_send_message_body()
        assert "setTaskText" not in body

    def test_send_message_does_not_set_current_task_name(self) -> None:
        body = self._get_send_message_body()
        assert "currentTaskName" not in body

    def test_send_message_does_not_call_reset_adjacent_state(self) -> None:
        body = self._get_send_message_body()
        assert "resetAdjacentState" not in body

    def test_send_message_does_not_call_set_state(self) -> None:
        body = self._get_send_message_body()
        assert "vscode.setState" not in body

    def test_send_message_does_not_hide_welcome(self) -> None:
        body = self._get_send_message_body()
        assert "welcome.style.display" not in body

    # -- main.js: setTaskText event handler DOES manage task panel state -----

    def _get_set_task_text_handler(self) -> str:
        start = self._js.index("case 'setTaskText':")
        end = self._js.index("break;", start) + len("break;")
        return self._js[start:end]

    def test_set_task_text_handler_sets_current_task_name(self) -> None:
        body = self._get_set_task_text_handler()
        assert "currentTaskName = stt" in body

    def test_set_task_text_handler_calls_reset_adjacent_state(self) -> None:
        body = self._get_set_task_text_handler()
        assert "resetAdjacentState()" in body

    def test_set_task_text_handler_calls_set_state(self) -> None:
        body = self._get_set_task_text_handler()
        assert "vscode.setState({ task: stt })" in body

    def test_set_task_text_handler_hides_welcome(self) -> None:
        body = self._get_set_task_text_handler()
        assert "welcome.style.display = 'none'" in body

    def test_set_task_text_handler_calls_set_task_text(self) -> None:
        body = self._get_set_task_text_handler()
        assert "setTaskText(ev.text" in body

    # -- SorcarTab.ts: file-open path returns before _startTask --------------

    def test_ts_file_open_returns_before_start_task(self) -> None:
        """The submit handler opens the file and returns *before* _startTask."""
        submit_idx = self._ts.index("case 'submit':")
        submit_end = self._ts.index("break;", submit_idx)
        submit_body = self._ts[submit_idx:submit_end]
        # The file-open block must contain 'return' before _startTask
        file_check_idx = submit_body.index("isFile()")
        return_idx = submit_body.index("return;", file_check_idx)
        start_task_idx = submit_body.index("this._startTask(")
        assert return_idx < start_task_idx

    def test_ts_start_task_sends_set_task_text(self) -> None:
        """_startTask sends setTaskText to the webview (the only source)."""
        start = self._ts.index("private _startTask(")
        end = self._ts.index("\n  private ", start + 1)
        body = self._ts[start:end]
        assert "setTaskText" in body


class TestCollapsiblePanelsJS(unittest.TestCase):
    """Test that main.js makes panels collapsible via addCollapse."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    # -- addCollapse helper function --

    def test_has_add_collapse_function(self) -> None:
        assert "function addCollapse(panelEl, headerEl)" in self._js

    def test_add_collapse_inserts_chevron(self) -> None:
        """addCollapse inserts a .collapse-chv span into the header."""
        idx = self._js.index("function addCollapse(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "collapse-chv" in body
        assert "insertBefore" in body

    def test_add_collapse_toggles_collapsed_class(self) -> None:
        """Clicking the header toggles 'collapsed' class on the panel."""
        idx = self._js.index("function addCollapse(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "classList.toggle('collapsed')" in body

    def test_add_collapse_stops_propagation(self) -> None:
        """Click handler calls stopPropagation to avoid side effects."""
        idx = self._js.index("function addCollapse(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "stopPropagation" in body

    # -- LLM panel is collapsible --

    def test_llm_panel_has_header(self) -> None:
        """LLM panel gets a .llm-panel-hdr div with 'Response' label."""
        assert "llm-panel-hdr" in self._js
        # Verify it's created with 'Response' text
        idx = self._js.index("llm-panel-hdr")
        block = self._js[idx : idx + 200]
        assert "Response" in block

    def test_llm_panel_calls_add_collapse(self) -> None:
        """processOutputEvent calls addCollapse on the LLM panel."""
        idx = self._js.index("var lHdr = mkEl('div', 'llm-panel-hdr')")
        block = self._js[idx : idx + 200]
        assert "addCollapse(llmPanel, lHdr)" in block

    def test_adjacent_llm_panel_calls_add_collapse(self) -> None:
        """renderAdjacentTask also calls addCollapse on LLM panels."""
        idx = self._js.index("var aLHdr = mkEl('div', 'llm-panel-hdr')")
        block = self._js[idx : idx + 200]
        assert "addCollapse(adjLlmPanel, aLHdr)" in block

    # -- Result card is collapsible --

    def test_result_card_never_collapsible(self) -> None:
        """Result card does NOT call addCollapse — it is always visible."""
        assert "addCollapse(rc, rc.querySelector('.rc-h'))" not in self._js

    # -- System prompt / Prompt are collapsible --

    def test_prompt_calls_add_collapse(self) -> None:
        """System prompt and prompt panels call addCollapse."""
        assert "addCollapse(el, el.querySelector('.' + cls + '-h'))" in self._js

    # -- Tool result is collapsible --

    def test_tool_result_has_tr_content_wrapper(self) -> None:
        """Tool result content is wrapped in .tr-content div."""
        assert "tr-content" in self._js

    def test_tool_result_calls_add_collapse(self) -> None:
        """Tool result calls addCollapse with .rl as the header."""
        assert "addCollapse(r, r.querySelector('.rl'))" in self._js

    # -- Usage info is collapsible --

    def test_usage_has_header_and_content(self) -> None:
        """Usage info has .usage-hdr and .usage-content elements."""
        assert "usage-hdr" in self._js
        assert "usage-content" in self._js

    def test_usage_calls_add_collapse(self) -> None:
        """Usage info calls addCollapse with the header."""
        assert "addCollapse(u, uHdr)" in self._js

    # -- Error/stopped banners are collapsible --

    def test_error_banner_has_tr_content(self) -> None:
        """Error/stopped banners wrap content in .tr-content."""
        # Find the task_error/task_stopped handler in handleEvent
        idx = self._js.index("case 'task_error':")
        block = self._js[idx : idx + 500]
        assert "tr-content" in block

    def test_error_banner_calls_add_collapse(self) -> None:
        """Error/stopped banners call addCollapse."""
        idx = self._js.index("case 'task_error':")
        block = self._js[idx : idx + 500]
        assert "addCollapse(banner, banner.querySelector('.rl'))" in block

    def test_adjacent_error_banners_collapsible(self) -> None:
        """Error banners in adjacent task replay are also collapsible."""
        idx = self._js.index("'task_error') {")
        block = self._js[idx : idx + 600]
        assert "tr-content" in block
        assert "addCollapse(banner" in block

    # -- Merge info is collapsible --

    def test_merge_info_has_header_and_body(self) -> None:
        """Merge info uses .merge-info-hdr and .merge-info-body."""
        assert "merge-info-hdr" in self._js
        assert "merge-info-body" in self._js

    def test_merge_info_calls_add_collapse(self) -> None:
        """Merge info calls addCollapse with the header."""
        assert "addCollapse(mc, mc.querySelector('.merge-info-hdr'))" in self._js

    # -- Excluded panels are NOT collapsible --

    def test_followup_bar_not_collapsible(self) -> None:
        """Followup suggestion ('Suggested next') should NOT be collapsible."""
        # Find the followup_suggestion handler
        idx = self._js.index("case 'followup_suggestion':")
        end = self._js.index("break;", idx) + len("break;")
        body = self._js[idx:end]
        assert "addCollapse" not in body
        assert "collapsed" not in body

    def test_task_panel_not_collapsible(self) -> None:
        """The fixed task panel (setTaskText function) should NOT be collapsible."""
        idx = self._js.index("function setTaskText(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "addCollapse" not in body
        assert "collapsed" not in body


class TestAutoCollapseOlderPanelsJS(unittest.TestCase):
    """Test that older collapsible panels are auto-collapsed during running."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    # -- collapseOlderPanels function --

    def test_has_collapse_older_panels_function(self) -> None:
        assert "function collapseOlderPanels()" in self._js

    def test_collapse_older_panels_checks_is_running(self) -> None:
        """Only auto-collapses when the agent is running."""
        idx = self._js.index("function collapseOlderPanels()")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "if (!isRunning) return;" in body

    def test_collapse_older_panels_queries_scope_collapsible(self) -> None:
        """Uses :scope > .collapsible to find direct children only."""
        idx = self._js.index("function collapseOlderPanels()")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "querySelectorAll(':scope > .collapsible')" in body

    def test_collapse_older_panels_keeps_last_expanded(self) -> None:
        """Loop iterates to length-1, leaving the last panel expanded."""
        idx = self._js.index("function collapseOlderPanels()")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "panels.length - 1" in body
        assert "classList.add('collapsed')" in body

    # -- addCollapse marks panels as collapsible --

    def test_add_collapse_adds_collapsible_class(self) -> None:
        """addCollapse adds 'collapsible' class to the panel element."""
        idx = self._js.index("function addCollapse(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "classList.add('collapsible')" in body

    # -- collapseOlderPanels called in processOutputEvent --

    def test_called_after_llm_panel_appended(self) -> None:
        """collapseOlderPanels is called after O.appendChild(llmPanel)."""
        idx = self._js.index("O.appendChild(llmPanel);")
        block = self._js[idx : idx + 100]
        assert "collapseOlderPanels()" in block

    def test_called_after_handle_output_event_to_output(self) -> None:
        """collapseOlderPanels is called when target === O after handleOutputEvent."""
        idx = self._js.index("function processOutputEvent(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "if (target === O) collapseOlderPanels();" in body

    # -- collapseOlderPanels called in handleEvent --

    def test_called_after_merge_info_appended(self) -> None:
        """collapseOlderPanels is called after merge-info panel appended."""
        idx = self._js.index("case 'merge_data':")
        end = self._js.index("break;", idx) + len("break;")
        body = self._js[idx:end]
        assert "collapseOlderPanels()" in body

    def test_called_after_error_banner_appended(self) -> None:
        """collapseOlderPanels is called after error/stopped banner appended."""
        idx = self._js.index("case 'task_error':\n")
        end = self._js.index("break;", idx) + len("break;")
        body = self._js[idx:end]
        assert "collapseOlderPanels()" in body

    # -- Not called during history replay --

    def test_not_called_in_render_adjacent_task(self) -> None:
        """collapseOlderPanels is NOT called in renderAdjacentTask."""
        idx = self._js.index("function renderAdjacentTask(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "collapseOlderPanels" not in body

    def test_not_called_in_replay_task_events(self) -> None:
        """collapseOlderPanels is NOT called in replayTaskEvents."""
        idx = self._js.index("function replayTaskEvents(")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "collapseOlderPanels" not in body


class TestCollapsiblePanelsCSS(unittest.TestCase):
    """Test that main.css has all required collapsible panel styles."""

    _css: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._css = (base / "vscode" / "media" / "main.css").read_text()

    def test_has_collapse_chv_style(self) -> None:
        assert ".collapse-chv" in self._css

    def test_collapse_chv_rotates_when_collapsed(self) -> None:
        assert ".collapsed .collapse-chv" in self._css
        idx = self._css.index(".collapsed .collapse-chv")
        block = self._css[idx : idx + 100]
        assert "rotate(-90deg)" in block

    def test_llm_panel_hdr_style(self) -> None:
        assert ".llm-panel-hdr" in self._css

    def test_llm_panel_collapsed_hides_children(self) -> None:
        assert ".llm-panel.collapsed > :not(.llm-panel-hdr)" in self._css

    def test_rc_never_collapsed(self) -> None:
        """Result card has no CSS collapse rule — it is always visible."""
        assert ".rc.collapsed .rc-body" not in self._css

    def test_prompt_collapsed_hides_body(self) -> None:
        assert ".system-prompt.collapsed .system-prompt-body" in self._css
        assert ".prompt.collapsed .prompt-body" in self._css

    def test_tr_collapsed_hides_content(self) -> None:
        assert ".tr.collapsed .tr-content" in self._css

    def test_usage_hdr_style(self) -> None:
        assert ".usage-hdr" in self._css

    def test_usage_collapsed_hides_content(self) -> None:
        assert ".usage.collapsed .usage-content" in self._css

    def test_merge_info_collapsed_hides_body(self) -> None:
        assert ".merge-info.collapsed .merge-info-body" in self._css

    def test_has_collapsible_section_comment(self) -> None:
        assert "Collapsible panels" in self._css

    def test_bash_panel_hdr_style(self) -> None:
        assert ".bash-panel-hdr" in self._css

    def test_bash_panel_hdr_hover(self) -> None:
        assert ".bash-panel-hdr:hover" in self._css

    def test_bash_panel_collapsed_hides_content(self) -> None:
        assert ".bash-panel.collapsed .bash-panel-content" in self._css

    def test_bash_panel_content_has_max_height(self) -> None:
        """max-height and overflow-y moved from .bash-panel to .bash-panel-content."""
        idx = self._css.index(".bash-panel-content")
        block = self._css[idx : idx + 300]
        assert "max-height" in block
        assert "overflow-y" in block

    def test_bash_panel_no_max_height(self) -> None:
        """The .bash-panel wrapper itself should not have max-height."""
        # Find the .bash-panel { ... } block (not .bash-panel-content or -hdr)
        import re

        m = re.search(r"\.bash-panel\s*\{([^}]+)\}", self._css)
        assert m is not None
        block = m.group(1)
        assert "max-height" not in block


class TestBashPanelCollapsibleJS(unittest.TestCase):
    """Test that bash panels are made collapsible in main.js."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def _tool_call_block(self) -> str:
        """Extract the tool_call case block from handleOutputEvent."""
        idx = self._js.index("case 'tool_call':")
        end = self._js.index("case 'tool_result':", idx)
        return self._js[idx:end]

    def test_bash_panel_has_header(self) -> None:
        """A bash-panel-hdr div is created inside bash-panel."""
        block = self._tool_call_block()
        assert "bash-panel-hdr" in block

    def test_bash_panel_header_text_is_output(self) -> None:
        """The bash-panel header displays 'Output'."""
        block = self._tool_call_block()
        assert "'Output'" in block

    def test_bash_panel_has_content_div(self) -> None:
        """A bash-panel-content div is created for streaming output."""
        block = self._tool_call_block()
        assert "bash-panel-content" in block

    def test_bash_panel_calls_add_collapse(self) -> None:
        """addCollapse is called on the bash-panel with the header."""
        block = self._tool_call_block()
        assert "addCollapse(bp, bpHdr)" in block

    def test_bash_panel_state_points_to_content(self) -> None:
        """tState.bashPanel is set to the content div, not the wrapper."""
        block = self._tool_call_block()
        assert "tState.bashPanel = bpContent" in block

    def test_bash_panel_content_appended_to_wrapper(self) -> None:
        """The content div is appended inside the bash-panel wrapper."""
        block = self._tool_call_block()
        assert "bp.appendChild(bpContent)" in block

    def test_bash_panel_header_appended_before_content(self) -> None:
        """Header is appended before content within the bash-panel."""
        block = self._tool_call_block()
        hdr_pos = block.index("bp.appendChild(bpHdr)")
        content_pos = block.index("bp.appendChild(bpContent)")
        assert hdr_pos < content_pos


class TestCollapseAllExceptResultJS(unittest.TestCase):
    """Test that loading a task collapses all panels except the Result panel."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def test_collapse_all_except_result_function_exists(self) -> None:
        """collapseAllExceptResult function is defined."""
        assert "function collapseAllExceptResult(container)" in self._js

    def test_queries_collapsible_panels(self) -> None:
        """Function queries .collapsible elements in the container."""
        idx = self._js.index("function collapseAllExceptResult(container)")
        block = self._js[idx : idx + 300]
        assert "container.querySelectorAll('.collapsible')" in block

    def test_skips_result_panels(self) -> None:
        """Function skips panels that have the .rc class (Result)."""
        idx = self._js.index("function collapseAllExceptResult(container)")
        block = self._js[idx : idx + 300]
        assert "classList.contains('rc')" in block

    def test_collapses_non_result_panels(self) -> None:
        """Function adds 'collapsed' class to non-Result panels."""
        idx = self._js.index("function collapseAllExceptResult(container)")
        block = self._js[idx : idx + 300]
        assert "classList.add('collapsed')" in block

    def test_called_in_replay_task_events(self) -> None:
        """collapseAllExceptResult(O) is called in replayTaskEvents."""
        idx = self._js.index("function replayTaskEvents(events)")
        end = self._js.index("function ", idx + 10)
        block = self._js[idx:end]
        assert "collapseAllExceptResult(O)" in block

    def test_called_in_render_adjacent_task(self) -> None:
        """collapseAllExceptResult(container) is called in renderAdjacentTask."""
        idx = self._js.index("function renderAdjacentTask(direction, task, events)")
        end = self._js.index("\n  function ", idx + 10)
        block = self._js[idx:end]
        assert "collapseAllExceptResult(container)" in block

    def test_not_called_during_live_streaming(self) -> None:
        """collapseAllExceptResult is NOT called in processOutputEvent."""
        idx = self._js.index("function processOutputEvent(ev)")
        end = self._js.index("\n  function ", idx + 10)
        block = self._js[idx:end]
        assert "collapseAllExceptResult" not in block


class TestDiffFilesDeletionAtStart(unittest.TestCase):
    """Regression: _diff_files must produce correct hunk positions for
    pure deletions at the beginning of a file.

    Root cause: _diff_files returned new_start=1 (instead of 0) when
    lines were deleted at the very start and the current file was non-empty.
    This caused _hunk_to_dict to produce cs=1, so the MergeManager
    inserted old (red) lines at position 1 instead of 0 — the deleted
    lines appeared AFTER the first surviving line instead of BEFORE it.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name: str, text: str) -> str:
        p = Path(self.tmpdir) / name
        p.write_text(text)
        return str(p)

    def test_start_deletion_cs_is_zero(self) -> None:
        """Deleting lines at the start must produce cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\nD\n")
        current = self._write("current.txt", "C\nD\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0, f"Expected cs=0, got cs={dicts[0]['cs']}"
        assert dicts[0]["bs"] == 0
        assert dicts[0]["bc"] == 2
        assert dicts[0]["cc"] == 0

    def test_middle_deletion_cs_correct(self) -> None:
        """Deleting lines in the middle must produce correct cs."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\nD\n")
        current = self._write("current.txt", "A\nD\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 1

    def test_delete_all_cs_is_zero(self) -> None:
        """Deleting all lines must produce cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\n")
        current = self._write("current.txt", "")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0

    def test_start_deletion_single_line(self) -> None:
        """Deleting a single line at the start produces cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\n")
        current = self._write("current.txt", "B\nC\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0
        assert dicts[0]["bc"] == 1

    def test_end_deletion_cs_correct(self) -> None:
        """Deleting lines at the end produces correct cs."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\n")
        current = self._write("current.txt", "A\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 1

    def test_multiple_hunks_including_start(self) -> None:
        """Multiple deletions including at the start all have correct cs."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\nD\nE\n")
        current = self._write("current.txt", "C\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        # First hunk: deletion of A,B at start → cs=0
        assert dicts[0]["cs"] == 0
        assert dicts[0]["bc"] == 2

    def test_start_insertion_cs_correct(self) -> None:
        """Inserting lines at the start produces cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "B\nC\n")
        current = self._write("current.txt", "A\nB\nC\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0
        assert dicts[0]["cc"] == 1
        assert dicts[0]["bc"] == 0

    def test_replacement_at_start(self) -> None:
        """Replacing lines at the start produces cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\n")
        current = self._write("current.txt", "X\nY\nC\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0
        assert dicts[0]["cc"] == 2
        assert dicts[0]["bc"] == 2


class TestWorktreeActionExceptionHandling(unittest.TestCase):
    """Regression: worktree actions must always broadcast worktree_result,
    even when the action raises an exception.

    Root cause: _handle_worktree_action was called without try/except in
    _handle_command, so a RuntimeError from wt.merge() (e.g. when _wt is
    None) would prevent the worktree_result broadcast, causing the VS Code
    UI to hang for the 120s timeout.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        subprocess.run(
            ["git", "init"], cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.repo, capture_output=True,
        )
        (self.repo / "file.txt").write_text("hello")
        subprocess.run(
            ["git", "add", "."], cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=self.repo, capture_output=True,
        )

        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_exception_still_broadcasts_result(self) -> None:
        """worktree_result is broadcast even when merge raises RuntimeError."""
        self.server._use_worktree = True
        # Don't set up _wt — so wt.merge() raises RuntimeError
        self.server._handle_command({"type": "worktreeAction", "action": "merge"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["message"]  # non-empty error message

    def test_discard_exception_still_broadcasts_result(self) -> None:
        """worktree_result is broadcast even when discard raises RuntimeError."""
        self.server._use_worktree = True
        self.server._handle_command({"type": "worktreeAction", "action": "discard"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False

    def test_do_nothing_exception_still_broadcasts_result(self) -> None:
        """worktree_result is broadcast even when do_nothing raises RuntimeError."""
        self.server._use_worktree = True
        self.server._handle_command({"type": "worktreeAction", "action": "do_nothing"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False

    def test_successful_merge_still_works(self) -> None:
        """Normal merge flow still works after the try/except addition."""
        subprocess.run(
            ["git", "checkout", "-b", "kiss/exc-test"],
            cwd=self.repo, capture_output=True,
        )
        (self.repo / "new.txt").write_text("new content")
        subprocess.run(
            ["git", "add", "."], cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add new"],
            cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=self.repo, capture_output=True,
        )

        self.server._use_worktree = True
        _set_agent_wt(
            self.server._worktree_agent,
            self.repo, "kiss/exc-test", "main",
        )

        self.server._handle_command({"type": "worktreeAction", "action": "merge"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is True


class TestExtractExtrasNoTruncation(unittest.TestCase):
    """Verify extract_extras does not truncate long argument values."""

    def test_long_value_not_truncated(self):
        from kiss.core.printer import extract_extras
        long_val = "x" * 500
        result = extract_extras({"custom_arg": long_val})
        assert result == {"custom_arg": long_val}
        assert "..." not in result["custom_arg"]

    def test_known_keys_excluded(self):
        from kiss.core.printer import extract_extras
        result = extract_extras({
            "file_path": "/a/b.py", "command": "ls", "extra": "val",
        })
        assert result == {"extra": "val"}


if __name__ == "__main__":
    unittest.main()

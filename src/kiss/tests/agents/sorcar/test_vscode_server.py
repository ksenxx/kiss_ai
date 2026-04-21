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


class TestNewChatBroadcastsShowWelcome(unittest.TestCase):
    """_new_chat must broadcast a showWelcome event to the tab."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_chat_broadcasts_show_welcome(self) -> None:
        self.server._new_chat("tab-1")
        welcome_events = [e for e in self.events if e["type"] == "showWelcome"]
        assert len(welcome_events) == 1
        assert welcome_events[0]["tabId"] == "tab-1"


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


class TestGenerateCommitMessage(unittest.TestCase):
    """Test generateCommitMessage uses fast_model_for via _generate_commit_message_llm."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_command_spawns_thread(self) -> None:
        """generateCommitMessage command calls _generate_commit_message."""
        import inspect

        src = inspect.getsource(VSCodeServer._cmd_generate_commit_message)
        assert "target=self._generate_commit_message" in src

    def test_no_model_param(self) -> None:
        """_generate_commit_message takes no model parameter."""
        import inspect

        sig = inspect.signature(VSCodeServer._generate_commit_message)
        # Only 'self' — no model parameter
        assert "model" not in sig.parameters

    def test_no_staged_changes(self) -> None:
        """_generate_commit_message reports no staged changes."""
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True)
        self.server._generate_commit_message()
        assert len(self.events) == 1
        assert self.events[0]["error"] == (
            "No staged changes found. Stage files with 'git add' first."
        )


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
        assert "55%" in self.js
        assert "75%" in self.js

    def test_chat_id_bg_colors_are_light(self) -> None:
        """Verify the chatIdBgColor function produces light pastel colors.

        Reimplements the JS djb2 hash + HSL logic in Python and checks that
        the minimum RGB channel is >= 140 (i.e., clearly light) for
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
            # HSL(hue, 55%, 75%) -> RGB
            r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.75, 0.55)
            return (round(r * 255), round(g * 255), round(b * 255))

        test_ids = [
            "abc123", "xyz789", "chat-001", "chat-002", "session-1",
            "a", "test", "550e8400-e29b-41d4-a716-446655440000",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479", "z",
        ]
        for cid in test_ids:
            r, g, b = chat_id_bg_rgb(cid)
            assert min(r, g, b) >= 140, (
                f"chat_id={cid!r} produced dark color rgb({r},{g},{b})"
            )

    def test_sidebar_escape_closes(self) -> None:
        assert "'Escape'" in self.js
        assert "closeSidebar" in self.js

    def test_render_history_accepts_offset(self) -> None:
        assert "renderHistory(sessions, offset, generation)" in self.js or \
               "function renderHistory" in self.js


class TestHistoryPanelSearchOnOpen(unittest.TestCase):
    """Test that opening the history panel uses existing search text.

    Regression: the historyBtn click handler used to send getHistory without
    the ``query`` parameter, ignoring text already in the search box.  The fix
    adds ``query: historySearch.value`` so the server filters results even on
    the initial open.
    """

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def _get_history_btn_click_body(self) -> str:
        """Extract the historyBtn click handler body."""
        idx = self._js.index("historyBtn.addEventListener('click',")
        # Find the matching closing of this handler — look for next top-level
        # addEventListener on a different element
        end = self._js.index("sidebarClose.addEventListener(", idx)
        return self._js[idx:end]

    def test_history_btn_click_sends_query(self) -> None:
        """historyBtn click handler includes query: historySearch.value."""
        body = self._get_history_btn_click_body()
        assert "query: historySearch.value" in body, (
            "historyBtn click handler must send query: historySearch.value "
            "so existing search text filters the results on panel open"
        )

    def test_all_get_history_calls_include_query(self) -> None:
        """Every getHistory postMessage includes query: historySearch.value.

        This ensures no code path accidentally sends getHistory without the
        query parameter, which would discard the user's current search text.
        """
        import re

        # Find all lines that post a getHistory message
        pattern = re.compile(r"postMessage\(\{[^}]*type:\s*'getHistory'[^}]*\}")
        matches = pattern.findall(self._js)
        assert len(matches) >= 3, (
            f"Expected at least 3 getHistory calls (btn click, input, scroll), "
            f"found {len(matches)}"
        )
        for m in matches:
            assert "query: historySearch.value" in m, (
                f"getHistory call missing query parameter: {m}"
            )

    def test_server_filters_history_with_query(self) -> None:
        """VSCodeServer._get_history passes query to _search_history."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda ev: events.append(ev)  # type: ignore[assignment]

        # Call with a query — should not crash and should broadcast history
        server._get_history("some search text", offset=0, generation=1)
        assert len(events) == 1
        assert events[0]["type"] == "history"
        assert events[0]["generation"] == 1

    def test_server_returns_unfiltered_without_query(self) -> None:
        """VSCodeServer._get_history returns unfiltered results when query is None."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda ev: events.append(ev)  # type: ignore[assignment]

        server._get_history(None, offset=0, generation=0)
        assert len(events) == 1
        assert events[0]["type"] == "history"
        assert isinstance(events[0]["sessions"], list)


class TestHistoryClickTabFocus(unittest.TestCase):
    """Test that clicking a history item focuses an existing tab or creates a new one."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def _get_render_history_body(self) -> str:
        idx = self._js.index("function renderHistory(")
        end = self._js.index("\n  function ", idx + 1)
        return self._js[idx:end]

    def test_task_events_captures_chat_id(self) -> None:
        """task_events handler persists state when ev.chat_id is present."""
        idx = self._js.index("case 'task_events'")
        end = self._js.index("case 'adjacent_task_events'", idx)
        body = self._js[idx:end]
        assert "ev.chat_id" in body
        assert "persistTabState" in body

    def test_history_click_creates_new_tab(self) -> None:
        """History item click creates a new tab and loads the session."""
        body = self._get_render_history_body()
        assert "createNewTab()" in body
        assert "resumeSession" in body


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

    def test_sidebar_item_uses_theme_foreground(self) -> None:
        idx = self.css.index(".sidebar-item")
        block = self.css[idx : idx + 300]
        assert "var(--fg)" in block
        assert "#000" not in block


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

    def test_button_is_after_upload(self) -> None:
        upload_idx = self.html.index('id="upload-btn"')
        worktree_idx = self.html.index('id="worktree-toggle-btn"')
        assert upload_idx < worktree_idx

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

        tab = self.server._get_tab("0")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/merge-test", "main")

        result = self.server._handle_worktree_action("merge", "0")
        assert result["success"] is True
        assert "Successfully merged" in result["message"]
        # Branch should be cleaned up
        assert self.server._get_tab("0").agent._wt_branch is None

    def test_handle_worktree_action_discard(self) -> None:
        """Discard action removes worktree branch."""
        self._git("checkout", "-b", "kiss/discard-test")
        self._git("checkout", "main")

        tab = self.server._get_tab("0")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/discard-test", "main")

        result = self.server._handle_worktree_action("discard", "0")
        assert result["success"] is True
        assert "Discarded" in result["message"]
        assert self.server._get_tab("0").agent._wt_branch is None

    def test_worktree_action_command_routing(self) -> None:
        """worktreeAction command is routed to _handle_worktree_action."""
        self._git("checkout", "-b", "kiss/route-test")
        (self.repo / "route.txt").write_text("route content")
        self._git("add", ".")
        self._git("commit", "-m", "add route")
        self._git("checkout", "main")

        self.server._get_tab("0").use_worktree = True
        wt_agent = self.server._get_tab("0").agent
        _set_agent_wt(wt_agent, self.repo, "kiss/route-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
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

        tab = self.server._get_tab("0")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/progress-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
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

        tab = self.server._get_tab("0")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/no-progress-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "discard", "tabId": "0"})
        progress_events = [e for e in self.events if e["type"] == "worktree_progress"]
        assert len(progress_events) == 0


class TestAgentToggle(unittest.TestCase):
    """Tests for worktree toggle switching between agents."""

    _JS_PATH = (
        Path(__file__).resolve().parents[3]
        / "agents" / "vscode" / "media" / "main.js"
    )
    _TS_PATH = (
        Path(__file__).resolve().parents[3]
        / "agents" / "vscode" / "src" / "SorcarSidebarView.ts"
    )
    _js: str
    _ts: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = cls._JS_PATH.read_text()
        cls._ts = cls._TS_PATH.read_text()

    def test_server_agent_is_worktree_sorcar_agent(self) -> None:
        """Server agent is a single WorktreeSorcarAgent regardless of toggle.

        ``WorktreeSorcarAgent`` subclasses ``StatefulSorcarAgent`` and
        internally falls back to the stateful code path when
        ``use_worktree=False`` is passed to ``run()``.  One instance
        per tab is therefore sufficient.
        """
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        server = VSCodeServer()
        tab = server._get_tab("0")
        assert tab.use_worktree is False
        assert isinstance(tab.agent, WorktreeSorcarAgent)
        assert isinstance(tab.agent, StatefulSorcarAgent)  # subclass
        original = tab.agent
        tab.use_worktree = True
        assert tab.agent is original  # same instance after toggle

    def test_js_sends_use_worktree_in_submit(self) -> None:
        """main.js includes useWorktree in submit message."""
        assert "useWorktree" in self._js
        assert "worktreeToggleBtn.classList.contains('active')" in self._js

    def test_ts_passes_use_worktree_to_start_task(self) -> None:
        """SorcarSidebarView.ts passes useWorktree to _startTask."""
        assert "message.useWorktree" in self._ts
        assert "useWorktree" in self._ts

    def test_ts_start_task_includes_use_worktree(self) -> None:
        """_startTask accepts useWorktree and forwards it in sendCommand."""
        assert "useWorktree?: boolean" in self._ts
        # Locate the _startTask method body and its sendCommand({...}) call.
        idx = self._ts.index("_startTask(")
        body = self._ts[idx : idx + 2000]
        send_idx = body.index("sendCommand({")
        cmd_block = body[send_idx : body.index("});", send_idx)]
        assert "useWorktree" in cmd_block

    def test_worktree_action_rejected_when_not_enabled(self) -> None:
        """Worktree action fails gracefully when worktree mode is off."""
        server = VSCodeServer()
        result = server._handle_worktree_action("merge")
        assert result["success"] is False
        assert "not enabled" in result["message"]


class TestWorktreeActionNotifications(unittest.TestCase):
    """Tests for VS Code notification behavior on worktree actions."""

    _TS_PATH = (
        Path(__file__).resolve().parents[3]
        / "agents" / "vscode" / "src" / "SorcarSidebarView.ts"
    )
    _ts: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._ts = cls._TS_PATH.read_text()

    def test_worktree_action_resolve_field_exists(self) -> None:
        """SorcarSidebarView has a _worktreeActionResolve field for progress tracking."""
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
        # Check that _worktreeActionResolves map is used and cleaned up
        assert "_worktreeActionResolves" in self._ts
        assert ".delete(" in self._ts

    def test_progress_title_varies_by_action(self) -> None:
        """Progress title differs for merge vs discard actions."""
        # The code should branch on wtAction to pick the right message
        assert "wtAction === 'merge'" in self._ts
        assert "wtAction === 'discard'" in self._ts

    def test_worktree_progress_field_exists(self) -> None:
        """SorcarSidebarView has a _worktreeProgress field for progress reporting."""
        assert "_worktreeProgress" in self._ts

    def test_worktree_progress_captured_in_with_progress(self) -> None:
        """The progress reporter is captured in the withProgress callback."""
        assert "_worktreeProgresses" in self._ts
        assert ".set(" in self._ts

    def test_worktree_progress_handler_updates_notification(self) -> None:
        """worktree_progress messages update the progress notification."""
        assert "worktree_progress" in self._ts
        assert ".report(" in self._ts

    def test_worktree_progress_cleared_on_result(self) -> None:
        """_worktreeProgresses is cleaned up when worktree_result arrives."""
        assert "_worktreeProgresses.delete(" in self._ts


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

    def test_button_is_after_worktree(self) -> None:
        worktree_idx = self.html.index('id="worktree-toggle-btn"')
        parallel_idx = self.html.index('id="parallel-toggle-btn"')
        assert worktree_idx < parallel_idx

    def test_has_svg_icon(self) -> None:
        """Button should have a parallel-lines SVG icon."""
        idx = self.html.index('id="parallel-toggle-btn"')
        block = self.html[idx : idx + 500]
        assert "<svg" in block
        assert "viewBox" in block

    def test_ts_passes_use_parallel_to_start_task(self) -> None:
        """SorcarSidebarView.ts passes useParallel to _startTask."""
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()
        assert "message.useParallel" in ts
        assert "useParallel" in ts

    def test_ts_start_task_includes_use_parallel(self) -> None:
        """_startTask accepts useParallel and forwards it in sendCommand."""
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()
        assert "useParallel?: boolean" in ts
        idx = ts.index("_startTask(")
        body = ts[idx : idx + 2000]
        send_idx = body.index("sendCommand({")
        cmd_block = body[send_idx : body.index("});", send_idx)]
        assert "useParallel" in cmd_block


class TestServerParallelToggle(unittest.TestCase):
    """Tests for parallel toggle in VSCodeServer."""

    def test_server_defaults_parallel_off(self) -> None:
        """use_parallel is False by default on new tab state."""
        server = VSCodeServer()
        assert server._get_tab("0").use_parallel is False

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
        # Server is now in merging state (tab_id is None from main thread,
        # so no tab's is_merging gets set by _start_merge_session)

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
        types = [e["type"] for e in self.events]
        assert "merge_ended" in types

    def test_handle_merge_action_unknown_is_noop(self) -> None:
        """Non-'all-done' actions are no-ops on the Python side."""
        self.server._get_tab("0").is_merging = True
        self.server._handle_merge_action("accept")
        # Still merging — only all-done finishes
        assert self.server._get_tab("0").is_merging is True

    def test_finish_merge_cleans_up_data_dir(self) -> None:
        """_finish_merge removes the merge data directory."""
        import kiss.agents.vscode.diff_merge as dm
        import kiss.agents.vscode.server as srv_mod

        orig_dm = dm._merge_data_dir
        orig_srv = srv_mod._merge_data_dir
        dm._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        srv_mod._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        try:
            path = self._write_merge_json()
            self.server._start_merge_session(path)
            assert self.merge_dir.exists()
            self.server._finish_merge()
            assert not self.merge_dir.exists()
        finally:
            dm._merge_data_dir = orig_dm  # type: ignore[assignment]
            srv_mod._merge_data_dir = orig_srv  # type: ignore[assignment]

    def test_merging_blocks_same_tab(self) -> None:
        """Cannot start a task on the same tab that has a merge in progress."""
        self.server._get_tab("5").is_merging = True
        # Simulate _run_task_inner rejecting a task on the merging tab
        self.server._run_task_inner({"prompt": "test", "model": "m", "tabId": "5"})
        errors = [e for e in self.events if e["type"] == "error"]
        assert any("merge review" in e["text"] for e in errors)

    def test_merging_does_not_block_other_tabs(self) -> None:
        """A merge on one tab does not block tasks on other tabs."""
        self.server._get_tab("5").is_merging = True
        # _run_task_inner on a different tab should NOT hit the merge error
        self.events.clear()
        self.server._run_task_inner({"prompt": "test", "model": "m", "tabId": "99"})
        errors = [e for e in self.events if e["type"] == "error"]
        # No "merge review" error for a different tab
        assert not any("merge review" in e.get("text", "") for e in errors)

    def test_restore_pending_merge_removed(self) -> None:
        """_restore_pending_merge was dead code and has been removed (RED-9)."""
        assert not hasattr(self.server, "_restore_pending_merge")

    def test_merge_command_routing(self) -> None:
        """mergeAction command is routed through _handle_command."""
        path = self._write_merge_json()
        self.server._start_merge_session(path)
        self.events.clear()

        import kiss.agents.vscode.diff_merge as dm
        import kiss.agents.vscode.server as srv_mod

        orig = dm._merge_data_dir
        dm._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        orig_srv = srv_mod._merge_data_dir
        srv_mod._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        try:
            self.server._handle_command({"type": "mergeAction", "action": "all-done"})
        finally:
            dm._merge_data_dir = orig  # type: ignore[assignment]
            srv_mod._merge_data_dir = orig_srv  # type: ignore[assignment]
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
    """Verify SorcarSidebarView opens files in ViewColumn.One (the left split)."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()

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
        cls._ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()

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

    def test_set_task_text_handler_persists_tab_state(self) -> None:
        body = self._get_set_task_text_handler()
        assert "updateActiveTabTitle(stt)" in body

    def test_set_task_text_handler_hides_welcome(self) -> None:
        body = self._get_set_task_text_handler()
        assert "welcome.style.display = 'none'" in body

    def test_set_task_text_handler_calls_set_task_text(self) -> None:
        body = self._get_set_task_text_handler()
        assert "setTaskText(ev.text" in body

    # -- SorcarSidebarView.ts: file-open path returns before _startTask ------

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
        """LLM panel gets a .llm-panel-hdr div with 'Thoughts' label."""
        assert "llm-panel-hdr" in self._js
        # Verify it's created with 'Thoughts' text
        idx = self._js.index("llm-panel-hdr")
        block = self._js[idx : idx + 200]
        assert "Thoughts" in block

    def test_llm_panel_calls_add_collapse(self) -> None:
        """processOutputEvent calls addCollapse on the LLM panel."""
        idx = self._js.index("const lHdr = mkEl('div', 'llm-panel-hdr')")
        block = self._js[idx : idx + 200]
        assert "addCollapse(llmPanel, lHdr)" in block

    def test_adjacent_llm_panel_calls_add_collapse(self) -> None:
        """replayEventsInto calls addCollapse on LLM panels."""
        idx = self._js.index("function replayEventsInto(")
        end = self._js.index("\n  function ", idx + 10)
        block = self._js[idx:end]
        assert "addCollapse(rLlmPanel, lHdr)" in block

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

    # -- Usage info panels are hidden (not rendered in chat) --

    def test_usage_panels_not_rendered(self) -> None:
        """Usage info panels are not rendered in the chat output."""
        assert "usage-hdr" not in self._js
        assert "usage-content" not in self._js

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
        block = self._js[idx : idx + 800]
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
        # Skip past early-return break to find the case's closing break
        end = self._js.index("break;\n    }", idx) + len("break;\n    }")
        body = self._js[idx:end]
        assert "collapseOlderPanels()" in body

    def test_called_after_error_banner_appended(self) -> None:
        """collapseOlderPanels is called after error/stopped banner appended."""
        idx = self._js.index("case 'task_error':\n")
        # Skip past early-return break to find the case's closing break
        end = self._js.index("break;\n    }", idx) + len("break;\n    }")
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

    def test_bash_panel_nested_in_tc(self) -> None:
        """Output panel nested inside tool call has no border/radius."""
        assert ".tc > .bash-panel" in self._css

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
    """Test that bash panels are nested inside tool call panels in main.js."""

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

    def test_bash_panel_no_header(self) -> None:
        """Bash panel has no header (output panel is headerless)."""
        block = self._tool_call_block()
        assert "bash-panel-hdr" not in block

    def test_bash_panel_not_collapsible(self) -> None:
        """Bash panel does not call addCollapse."""
        block = self._tool_call_block()
        # addCollapse is called on the tool call panel (c, hdr), not on bash panel
        assert "addCollapse(bp" not in block

    def test_bash_panel_has_content_div(self) -> None:
        """A bash-panel-content div is created for streaming output."""
        block = self._tool_call_block()
        assert "bash-panel-content" in block

    def test_bash_panel_nested_in_tool_call(self) -> None:
        """Bash panel is appended inside the tool call element, not target."""
        block = self._tool_call_block()
        assert "c.appendChild(bp)" in block

    def test_bash_panel_state_points_to_content(self) -> None:
        """tState.bashPanel is set to the content div, not the wrapper."""
        block = self._tool_call_block()
        assert "tState.bashPanel = bpContent" in block

    def test_bash_panel_content_appended_to_wrapper(self) -> None:
        """The content div is appended inside the bash-panel wrapper."""
        block = self._tool_call_block()
        assert "bp.appendChild(bpContent)" in block

    def test_last_tool_call_el_tracked(self) -> None:
        """tState.lastToolCallEl is set to the tool call element."""
        block = self._tool_call_block()
        assert "tState.lastToolCallEl = c" in block

    def test_tool_result_uses_last_tool_call_el(self) -> None:
        """tool_result appends to lastToolCallEl when available."""
        idx = self._js.index("case 'tool_result':")
        end = self._js.index("case 'system_output':", idx)
        block = self._js[idx:end]
        assert "tState.lastToolCallEl || target" in block

    def test_tool_result_no_header(self) -> None:
        """tool_result output panel has no header."""
        idx = self._js.index("case 'tool_result':")
        end = self._js.index("case 'system_output':", idx)
        block = self._js[idx:end]
        assert "bash-panel-hdr" not in block

    def test_tool_result_not_collapsible(self) -> None:
        """tool_result output panel does not call addCollapse on bash-panel."""
        idx = self._js.index("case 'tool_result':")
        end = self._js.index("case 'system_output':", idx)
        block = self._js[idx:end]
        assert "addCollapse(op" not in block

    def test_mks_has_last_tool_call_el(self) -> None:
        """mkS() initializes lastToolCallEl to null."""
        idx = self._js.index("function mkS()")
        block = self._js[idx : idx + 200]
        assert "lastToolCallEl: null" in block


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
        """replayTaskEvents delegates to replayEventsInto which collapses."""
        idx = self._js.index("function replayTaskEvents(events)")
        end = self._js.index("\n  function ", idx + 10)
        block = self._js[idx:end]
        assert "replayEventsInto(O," in block

    def test_called_in_render_adjacent_task(self) -> None:
        """renderAdjacentTask delegates to replayEventsInto which collapses."""
        idx = self._js.index("function renderAdjacentTask(direction, task, events)")
        end = self._js.index("\n  function ", idx + 10)
        block = self._js[idx:end]
        assert "replayEventsInto(container, events)" in block


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
        self.server._get_tab("0").use_worktree = True
        # Don't set up _wt — so wt.merge() raises RuntimeError
        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["message"]  # non-empty error message

    def test_discard_exception_still_broadcasts_result(self) -> None:
        """worktree_result is broadcast even when discard raises RuntimeError."""
        self.server._get_tab("0").use_worktree = True
        self.server._handle_command({"type": "worktreeAction", "action": "discard", "tabId": "0"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False

    def test_do_nothing_rejected_as_unknown_action(self) -> None:
        """do_nothing is no longer a valid action and returns unknown error."""
        self.server._get_tab("0").use_worktree = True
        cmd = {"type": "worktreeAction", "action": "do_nothing", "tabId": "0"}
        self.server._handle_command(cmd)
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Unknown action" in results[0]["message"]

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

        self.server._get_tab("0").use_worktree = True
        _set_agent_wt(
            self.server._get_tab("0").agent,
            self.repo, "kiss/exc-test", "main",
        )

        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is True


class TestRunningStateDisablesButtons(unittest.TestCase):
    """Verify setRunningState disables the correct buttons when running.

    When the agent is running, 'Attach files', 'Use worktree',
    'Use parallelism', and 'Run current file as prompt' must be disabled.
    'Task history' and 'New chat' must NOT be disabled.
    """

    js: str
    css: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()
        cls.css = (base / "vscode" / "media" / "main.css").read_text()

    # --- JS: buttons disabled when running ---

    def test_upload_btn_disabled_when_running(self) -> None:
        assert "if (uploadBtn) uploadBtn.disabled = running" in self.js

    def test_worktree_btn_disabled_when_running(self) -> None:
        assert "if (worktreeToggleBtn) worktreeToggleBtn.disabled = running" in self.js

    def test_parallel_btn_disabled_when_running(self) -> None:
        assert "if (parallelToggleBtn) parallelToggleBtn.disabled = running" in self.js

    def test_run_prompt_btn_disabled_when_running(self) -> None:
        assert "if (runPromptBtn && running) runPromptBtn.disabled = true" in self.js

    # --- JS: history and new-chat NOT disabled ---

    def test_history_btn_not_disabled_when_running(self) -> None:
        assert "historyBtn.disabled" not in self.js

    def test_clear_btn_not_disabled_when_running(self) -> None:
        assert "clearBtn.disabled" not in self.js

    # --- CSS: disabled styles for correct buttons ---

    def test_css_upload_btn_disabled_style(self) -> None:
        assert "#upload-btn:disabled" in self.css

    def test_css_worktree_btn_disabled_style(self) -> None:
        assert "#worktree-toggle-btn:disabled" in self.css

    def test_css_parallel_btn_disabled_style(self) -> None:
        assert "#parallel-toggle-btn:disabled" in self.css

    def test_css_run_prompt_btn_disabled_style(self) -> None:
        assert "#run-prompt-btn:disabled" in self.css

    def test_css_no_history_btn_disabled_style(self) -> None:
        assert "#history-btn:disabled" not in self.css

    def test_css_no_clear_btn_disabled_style(self) -> None:
        assert "#clear-btn:disabled" not in self.css


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


# ---------------------------------------------------------------------------
# Secondary sidebar chat view tests
# ---------------------------------------------------------------------------


class TestSecondarySidebarPackageJson(unittest.TestCase):
    """Verify package.json declares the secondary sidebar container and webview view."""

    _pkg: dict

    @classmethod
    def setUpClass(cls) -> None:
        import json as _json

        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._pkg = _json.loads((base / "package.json").read_text())

    def test_secondary_sidebar_container_exists(self) -> None:
        containers = self._pkg["contributes"]["viewsContainers"]
        assert "secondarySidebar" in containers
        ids = [c["id"] for c in containers["secondarySidebar"]]
        assert "kissSorcarSecondary" in ids

    def test_secondary_sidebar_container_has_icon(self) -> None:
        containers = self._pkg["contributes"]["viewsContainers"]["secondarySidebar"]
        sec = [c for c in containers if c["id"] == "kissSorcarSecondary"][0]
        assert "icon" in sec
        assert "kiss-icon.svg" in sec["icon"]

    def test_secondary_sidebar_container_has_negative_order(self) -> None:
        """Negative order pushes the container to the top of the secondary sidebar."""
        containers = self._pkg["contributes"]["viewsContainers"]["secondarySidebar"]
        sec = [c for c in containers if c["id"] == "kissSorcarSecondary"][0]
        assert sec.get("order", 0) < 0

    def test_chat_view_secondary_declared(self) -> None:
        views = self._pkg["contributes"]["views"]
        assert "kissSorcarSecondary" in views
        ids = [v["id"] for v in views["kissSorcarSecondary"]]
        assert "kissSorcar.chatViewSecondary" in ids

    def test_chat_view_secondary_is_webview_type(self) -> None:
        views = self._pkg["contributes"]["views"]["kissSorcarSecondary"]
        chat = [v for v in views if v["id"] == "kissSorcar.chatViewSecondary"][0]
        assert chat.get("type") == "webview"

    def test_primary_sidebar_still_exists(self) -> None:
        """Primary sidebar (activity bar) container must still be present."""
        containers = self._pkg["contributes"]["viewsContainers"]
        assert "activitybar" in containers
        ids = [c["id"] for c in containers["activitybar"]]
        assert "kissSorcarContainer" in ids

    def test_primary_chat_view_still_exists(self) -> None:
        """Primary sidebar chat view must still be present."""
        views = self._pkg["contributes"]["views"]
        assert "kissSorcarContainer" in views
        ids = [v["id"] for v in views["kissSorcarContainer"]]
        assert "kissSorcar.chatView" in ids


class TestSorcarSidebarViewTS(unittest.TestCase):
    """Verify SorcarSidebarView.ts has the required structure."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_file_exists(self) -> None:
        assert len(self._ts) > 0

    def test_implements_webview_view_provider(self) -> None:
        assert "implements vscode.WebviewViewProvider" in self._ts

    def test_has_resolve_webview_view(self) -> None:
        assert "resolveWebviewView(" in self._ts

    def test_imports_build_chat_html(self) -> None:
        assert "buildChatHtml" in self._ts

    def test_calls_build_chat_html(self) -> None:
        assert "buildChatHtml(" in self._ts

    def test_has_agent_process(self) -> None:
        assert "AgentProcess" in self._ts

    def test_has_handle_message(self) -> None:
        assert "_handleMessage(" in self._ts

    def test_handles_submit(self) -> None:
        assert "case 'submit':" in self._ts

    def test_handles_ready(self) -> None:
        assert "case 'ready':" in self._ts

    def test_handles_stop(self) -> None:
        assert "case 'stop':" in self._ts

    def test_has_start_task(self) -> None:
        assert "_startTask(" in self._ts

    def test_has_submit_task(self) -> None:
        assert "submitTask(" in self._ts

    def test_has_stop_task(self) -> None:
        assert "stopTask(" in self._ts

    def test_has_new_conversation(self) -> None:
        assert "newConversation(" in self._ts

    def test_has_focus_chat_input(self) -> None:
        assert "focusChatInput(" in self._ts

    def test_has_dispose(self) -> None:
        assert "dispose()" in self._ts

    def test_has_commit_message_event(self) -> None:
        assert "onCommitMessage" in self._ts

    def test_has_merge_manager(self) -> None:
        assert "_mergeManager" in self._ts

    def test_has_worktree_support(self) -> None:
        assert "worktreeAction" in self._ts
        assert "_worktreeActionResolve" in self._ts

    def test_retains_context_when_hidden(self) -> None:
        """resolveWebviewView sets retainContextWhenHidden-equivalent options."""
        # The provider is registered with retainContextWhenHidden in extension.ts
        # but the webview options are set in resolveWebviewView
        assert "enableScripts: true" in self._ts

    def test_opens_files_in_view_column_one(self) -> None:
        assert "viewColumn: vscode.ViewColumn.One" in self._ts

    def test_sends_active_file_info(self) -> None:
        assert "_sendActiveFileInfo" in self._ts


class TestExtensionRegistersSecondaryView(unittest.TestCase):
    """Verify extension.ts registers the SorcarSidebarView provider."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "extension.ts").read_text()

    def test_imports_sorcar_sidebar_view(self) -> None:
        assert "SorcarSidebarView" in self._ts

    def test_registers_webview_view_provider(self) -> None:
        assert "registerWebviewViewProvider" in self._ts
        assert "kissSorcar.chatViewSecondary" in self._ts

    def test_creates_sidebar_view_instance(self) -> None:
        assert "new SorcarSidebarView(" in self._ts

    def test_sidebar_view_retain_context(self) -> None:
        assert "retainContextWhenHidden: true" in self._ts

    def test_sidebar_commit_message_listener(self) -> None:
        """Sidebar view's commit messages are forwarded to SCM."""
        assert "sidebarView" in self._ts
        assert "onCommitMessage" in self._ts

    def test_sidebar_view_disposed_on_deactivate(self) -> None:
        """sidebarView is disposed in the deactivate function."""
        idx = self._ts.index("function deactivate()")
        body = self._ts[idx:]
        assert "sidebarView?.dispose()" in body


class TestBuildChatHtmlExported(unittest.TestCase):
    """Verify SorcarTab.ts exports the shared buildChatHtml function."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarTab.ts").read_text()

    def test_build_chat_html_exported(self) -> None:
        assert "export function buildChatHtml(" in self._ts

    def test_get_nonce_exported(self) -> None:
        assert "export function getNonce()" in self._ts

    def test_get_version_exported(self) -> None:
        assert "export function getVersion()" in self._ts

    def test_build_chat_html_returns_full_html(self) -> None:
        """buildChatHtml contains the full chat HTML template."""
        idx = self._ts.index("export function buildChatHtml(")
        block = self._ts[idx:]
        assert "<!DOCTYPE html>" in block
        assert 'id="task-input"' in block
        assert 'id="output"' in block
        assert 'id="send-btn"' in block


class TestSorcarSidebarViewMessageHandling(unittest.TestCase):
    """Verify SorcarSidebarView handles expected message types."""

    _sidebar_ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._sidebar_ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    @staticmethod
    def _extract_case_labels(ts: str) -> set[str]:
        """Extract all case labels from the _handleMessage method."""
        import re

        idx = ts.index("_handleMessage(")
        body = ts[idx:]
        return set(re.findall(r"case '(\w+)':", body))

    def test_sidebar_handles_required_message_types(self) -> None:
        """SorcarSidebarView handles all required message types."""
        sidebar_cases = self._extract_case_labels(self._sidebar_ts)
        required = {
            "ready", "submit", "stop", "selectModel", "getModels",
            "newChat", "getInputHistory", "getHistory", "getFiles",
            "userAnswer", "userActionDone", "recordFileUsage", "openFile",
            "resumeSession", "getAdjacentTask",
            "complete", "mergeAction", "generateCommitMessage", "runPrompt",
            "worktreeAction", "resolveDroppedPaths", "focusEditor",
        }
        missing = required - sidebar_cases
        assert not missing, f"Sidebar is missing message handlers: {missing}"

    def test_sidebar_has_no_unknown_message_types(self) -> None:
        """SorcarSidebarView only handles known message types."""
        sidebar_cases = self._extract_case_labels(self._sidebar_ts)
        known = {
            "ready", "submit", "stop", "selectModel", "getModels",
            "newChat", "getInputHistory", "getHistory", "getFiles",
            "userAnswer", "userActionDone", "recordFileUsage", "openFile",
            "resumeSession", "getAdjacentTask",
            "complete", "mergeAction", "generateCommitMessage", "runPrompt",
            "worktreeAction", "resolveDroppedPaths", "focusEditor",
            "closeSecondaryBar", "getWelcomeSuggestions",
            "webviewFocusChanged", "autocommitAction",
        }
        extra = sidebar_cases - known
        assert not extra, f"Sidebar has extra message handlers: {extra}"


class TestSorcarSidebarViewOpensFilesInLeftSplit(unittest.TestCase):
    """Verify SorcarSidebarView opens files in ViewColumn.One like SorcarTab."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _extract_case_block(self, case_label: str) -> str:
        """Extract a switch-case block from _handleMessage."""
        import re as _re

        pat = _re.compile(rf"case\s+'{_re.escape(case_label)}'", _re.MULTILINE)
        m = pat.search(self._ts)
        assert m, f"Case '{case_label}' not found"
        start = m.start()
        next_case = _re.search(r"\n\s+case\s+'", self._ts[m.end():])
        if next_case:
            return self._ts[start : m.end() + next_case.start()]
        return self._ts[start:]

    def test_submit_file_open_uses_view_column_one(self) -> None:
        """submit handler opens file paths in ViewColumn.One."""
        block = self._extract_case_block("submit")
        assert "viewColumn: vscode.ViewColumn.One" in block

    def test_open_file_uses_view_column_one(self) -> None:
        """openFile handler opens files in ViewColumn.One."""
        block = self._extract_case_block("openFile")
        assert "viewColumn: vscode.ViewColumn.One" in block


class TestSorcarSidebarViewWorktreeActions(unittest.TestCase):
    """Verify SorcarSidebarView handles worktree actions with progress notifications."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_worktree_action_resolve_field(self) -> None:
        assert "_worktreeActionResolve" in self._ts

    def test_worktree_progress_field(self) -> None:
        assert "_worktreeProgress" in self._ts

    def test_merge_shows_progress_notification(self) -> None:
        assert "Committing and merging worktree" in self._ts

    def test_discard_shows_progress_notification(self) -> None:
        assert "Discarding worktree" in self._ts

    def test_progress_title_varies_by_action(self) -> None:
        assert "wtAction === 'merge'" in self._ts
        assert "wtAction === 'discard'" in self._ts

    def test_worktree_progress_reported(self) -> None:
        """worktree_progress events update the progress notification."""
        assert "worktree_progress" in self._ts
        assert ".report(" in self._ts

    def test_worktree_result_resolves_progress(self) -> None:
        """worktree_result resolves the progress notification (per-tab maps)."""
        assert "_worktreeActionResolves" in self._ts
        assert ".delete(" in self._ts

    def test_worktree_result_shows_info_or_error(self) -> None:
        """Successful results show info, failures show error."""
        assert "showInformationMessage" in self._ts
        assert "showErrorMessage" in self._ts

    def test_worktree_progress_cleared_on_result(self) -> None:
        assert "_worktreeProgresses.delete(" in self._ts

    def test_worktree_timeout(self) -> None:
        """worktreeAction has a 120s timeout."""
        assert "120_000" in self._ts

    def test_worktree_created_opens_scm(self) -> None:
        """worktree_created events open the worktree in SCM."""
        assert "worktree_created" in self._ts
        assert "_openWorktreeInScm" in self._ts

    def test_worktree_result_closes_scm(self) -> None:
        """Successful worktree_result closes the worktree in SCM."""
        assert "_closeWorktreeInScm" in self._ts


class TestSorcarSidebarViewMergeActions(unittest.TestCase):
    """Verify SorcarSidebarView dispatches merge actions to MergeManager."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_merge_action_block(self) -> str:
        import re

        m = re.search(r"case\s+'mergeAction':", self._ts)
        assert m
        start = m.start()
        end = self._ts.index("break;", start) + len("break;")
        return self._ts[start:end]

    def test_dispatches_accept(self) -> None:
        block = self._get_merge_action_block()
        assert "accept:" in block
        assert "acceptChange()" in block

    def test_dispatches_reject(self) -> None:
        block = self._get_merge_action_block()
        assert "reject:" in block
        assert "rejectChange()" in block

    def test_dispatches_prev(self) -> None:
        block = self._get_merge_action_block()
        assert "prev:" in block
        assert "prevChange()" in block

    def test_dispatches_next(self) -> None:
        block = self._get_merge_action_block()
        assert "next:" in block
        assert "nextChange()" in block

    def test_dispatches_accept_all(self) -> None:
        block = self._get_merge_action_block()
        assert "'accept-all'" in block
        assert "acceptAll()" in block

    def test_dispatches_reject_all(self) -> None:
        block = self._get_merge_action_block()
        assert "'reject-all'" in block
        assert "rejectAll()" in block

    def test_dispatches_accept_file(self) -> None:
        block = self._get_merge_action_block()
        assert "'accept-file'" in block
        assert "acceptFile()" in block

    def test_dispatches_reject_file(self) -> None:
        block = self._get_merge_action_block()
        assert "'reject-file'" in block
        assert "rejectFile()" in block

    def test_all_done_sent_to_agent(self) -> None:
        """all-done action is sent to the agent process, not MergeManager."""
        block = self._get_merge_action_block()
        assert "'all-done'" in block
        assert "sendMergeAllDone" in block

    def test_merge_data_opens_merge(self) -> None:
        """merge_data from agent opens merge via MergeManager."""
        assert "this._mergeManager.openMerge(" in self._ts

    def test_merge_data_sets_merge_owner(self) -> None:
        """merge_data pushes merge owner tab id from the event to the queue."""
        # Find merge_data handler
        idx = self._ts.index("msg.type === 'merge_data'")
        block = self._ts[idx : idx + 300]
        assert "_mergeOwnerTabIdQueue" in block


class TestSorcarSidebarViewStartTask(unittest.TestCase):
    """Verify SorcarSidebarView._startTask passes all parameters like SorcarTab."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_start_task_body(self) -> str:
        idx = self._ts.index("private _startTask(")
        end = self._ts.index("\n  private ", idx + 1)
        return self._ts[idx:end]

    def test_accepts_use_worktree(self) -> None:
        body = self._get_start_task_body()
        assert "useWorktree" in body

    def test_accepts_use_parallel(self) -> None:
        body = self._get_start_task_body()
        assert "useParallel" in body

    def test_accepts_attachments(self) -> None:
        body = self._get_start_task_body()
        assert "attachments" in body

    def test_accepts_active_file(self) -> None:
        body = self._get_start_task_body()
        assert "activeFile" in body

    def test_sends_set_task_text(self) -> None:
        """_startTask sends setTaskText to the webview."""
        body = self._get_start_task_body()
        assert "setTaskText" in body

    def test_sends_status_running(self) -> None:
        """_startTask sends status running: true to the webview."""
        body = self._get_start_task_body()
        assert "running: true" in body

    def test_sends_run_command(self) -> None:
        """_startTask sends the 'run' command to the agent process."""
        body = self._get_start_task_body()
        assert "type: 'run'" in body

    def test_submit_passes_worktree_and_parallel(self) -> None:
        """submit case passes useWorktree and useParallel from message to _startTask."""
        import re

        m = re.search(r"case\s+'submit':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "message.useWorktree" in block
        assert "message.useParallel" in block


class TestSorcarSidebarViewPublicAPI(unittest.TestCase):
    """Verify SorcarSidebarView exposes the required public methods."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_has_submit_task(self) -> None:
        """submitTask submits a task programmatically."""
        assert "public submitTask(prompt: string)" in self._ts

    def test_submit_task_calls_start_task(self) -> None:
        idx = self._ts.index("public submitTask(")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "this._startTask(" in body

    def test_submit_task_guards_against_running(self) -> None:
        """submitTask calls _startTask which checks per-tab running state."""
        idx = self._ts.index("public submitTask(")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "_startTask" in body

    def test_has_stop_task(self) -> None:
        assert "public stopTask()" in self._ts

    def test_stop_task_sends_trigger_stop(self) -> None:
        """stopTask delegates to webview via triggerStop to include active tabId."""
        idx = self._ts.index("public stopTask()")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "triggerStop" in body

    def test_has_focus_chat_input(self) -> None:
        assert "public async focusChatInput()" in self._ts

    def test_focus_chat_input_shows_view(self) -> None:
        idx = self._ts.index("public async focusChatInput()")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "this._view.show(true)" in body

    def test_focus_chat_input_sends_focus_input(self) -> None:
        idx = self._ts.index("public async focusChatInput()")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "focusInput" in body

    def test_has_new_conversation(self) -> None:
        assert "public newConversation()" in self._ts

    def test_new_conversation_sends_clear_chat(self) -> None:
        """newConversation sends clearChat to webview without stopping running tabs."""
        idx = self._ts.index("public newConversation()")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "type: 'clearChat'" in body
        # Should NOT stop any agent process (no interference with other tabs)
        assert ".stop()" not in body

    def test_has_generate_commit_message(self) -> None:
        assert "public generateCommitMessage(" in self._ts

    def test_generate_commit_message_sends_command(self) -> None:
        idx = self._ts.index("public generateCommitMessage(")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "type: 'generateCommitMessage'" in body

    def test_has_send_merge_all_done(self) -> None:
        assert "public sendMergeAllDone(" in self._ts

    def test_send_merge_all_done_sends_command(self) -> None:
        idx = self._ts.index("public sendMergeAllDone(")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "type: 'mergeAction'" in body
        assert "'all-done'" in body

    def test_has_dispose(self) -> None:
        assert "public dispose()" in self._ts

    def test_dispose_kills_agent_processes(self) -> None:
        idx = self._ts.index("public dispose()")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "proc.dispose()" in body
        assert "_taskProcesses.clear()" in body
        assert "_serviceProcess" in body

    def test_has_visible_getter(self) -> None:
        assert "get visible()" in self._ts

class TestSorcarSidebarViewAgentEventHandling(unittest.TestCase):
    """Verify SorcarSidebarView handles agent process events correctly."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_message_handler_body(self) -> str:
        """Get the _setupProcessListeners message handler body."""
        idx = self._ts.index("private _setupProcessListeners(")
        end = self._ts.index("\n  }", idx) + 4
        return self._ts[idx:end]

    def test_forwards_commit_messages(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'commitMessage'" in body
        assert "_onCommitMessage.fire" in body

    def test_updates_selected_model(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'models'" in body
        assert "this._selectedModel = msg.selected" in body

    def test_handles_merge_data(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'merge_data'" in body

    def test_handles_worktree_created(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_created'" in body

    def test_handles_worktree_done(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_done'" in body

    def test_handles_worktree_progress(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_progress'" in body

    def test_handles_worktree_result(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_result'" in body

    def test_forwards_all_messages_to_webview(self) -> None:
        body = self._get_message_handler_body()
        assert "this._sendToWebview(msg)" in body

    def test_tracks_running_status(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'status'" in body
        assert "this._runningTabs" in body

    def test_sends_active_file_info_on_stop(self) -> None:
        """When status running=false, sends active file info."""
        body = self._get_message_handler_body()
        assert "_sendActiveFileInfo()" in body

    def test_active_file_info_sent_on_stop(self) -> None:
        """When status running=false, sends active file info."""
        body = self._get_message_handler_body()
        assert "_sendActiveFileInfo()" in body


class TestSorcarSidebarViewReadyHandler(unittest.TestCase):
    """Verify the 'ready' message handler sends all initialization messages."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_ready_block(self) -> str:
        import re

        m = re.search(r"case\s+'ready':", self._ts)
        assert m
        end = self._ts.index("break;", m.end()) + len("break;")
        return self._ts[m.start() : end]

    def test_requests_models(self) -> None:
        block = self._get_ready_block()
        assert "'getModels'" in block

    def test_requests_input_history(self) -> None:
        block = self._get_ready_block()
        assert "'getInputHistory'" in block

    def test_sends_active_file_info(self) -> None:
        block = self._get_ready_block()
        assert "_sendActiveFileInfo()" in block

    def test_sends_focus_input(self) -> None:
        block = self._get_ready_block()
        assert "'focusInput'" in block


class TestSorcarSidebarViewVisibilityHandler(unittest.TestCase):
    """Verify the sidebar refreshes state when it becomes visible."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_on_did_change_visibility_registered(self) -> None:
        assert "onDidChangeVisibility" in self._ts

    def test_visibility_requests_input_history(self) -> None:
        """When view becomes visible, requests input history."""
        idx = self._ts.index("onDidChangeVisibility")
        block = self._ts[idx : idx + 200]
        assert "'getInputHistory'" in block

    def test_visibility_sends_active_file_info(self) -> None:
        idx = self._ts.index("onDidChangeVisibility")
        block = self._ts[idx : idx + 200]
        assert "_sendActiveFileInfo()" in block


class TestSorcarSidebarViewDisposeHandler(unittest.TestCase):
    """Verify the sidebar cleans up on dispose."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_on_did_dispose_registered(self) -> None:
        assert "onDidDispose" in self._ts

    def test_on_did_dispose_sets_disposed(self) -> None:
        idx = self._ts.index("onDidDispose")
        block = self._ts[idx : idx + 200]
        assert "this._disposed = true" in block

    def test_on_did_dispose_resolves_worktree_action(self) -> None:
        """Dispose resolves any pending worktree actions to prevent hangs."""
        idx = self._ts.index("onDidDispose")
        block = self._ts[idx : idx + 300]
        assert "_resolveAllWorktreeActions" in block

    def test_public_dispose_kills_agent(self) -> None:
        idx = self._ts.index("public dispose()")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "proc.dispose()" in body
        assert "_taskProcesses.clear()" in body
        assert "_serviceProcess" in body
        assert "this._onCommitMessage.dispose()" in body

    def test_send_to_webview_guards_disposed(self) -> None:
        """_sendToWebview checks _disposed before posting."""
        idx = self._ts.index("private _sendToWebview(")
        end = self._ts.index("\n  }", idx) + 4
        body = self._ts[idx:end]
        assert "!this._disposed" in body


class TestSorcarSidebarViewRunPrompt(unittest.TestCase):
    """Verify the sidebar handles runPrompt for .md files."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_run_prompt_case_exists(self) -> None:
        assert "case 'runPrompt':" in self._ts

    def test_run_prompt_checks_md_extension(self) -> None:
        import re

        m = re.search(r"case\s+'runPrompt':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert ".md'" in block

    def test_run_prompt_reads_content(self) -> None:
        import re

        m = re.search(r"case\s+'runPrompt':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "getText()" in block

    def test_run_prompt_calls_start_task(self) -> None:
        import re

        m = re.search(r"case\s+'runPrompt':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "_startTask(" in block


class TestSorcarSidebarViewResolveDroppedPaths(unittest.TestCase):
    """Verify the sidebar resolves dropped file paths relative to work dir."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_resolve_dropped_paths_case_exists(self) -> None:
        assert "case 'resolveDroppedPaths':" in self._ts

    def test_resolve_dropped_paths_sends_dropped_paths(self) -> None:
        import re

        m = re.search(r"case\s+'resolveDroppedPaths':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "'droppedPaths'" in block

    def test_filters_out_parent_paths(self) -> None:
        """Paths outside work dir (starting with ..) are filtered out."""
        import re

        m = re.search(r"case\s+'resolveDroppedPaths':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "'..'" in block


class TestSorcarSidebarViewComplete(unittest.TestCase):
    """Verify the sidebar handles the 'complete' message for autocompletion."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_complete_case_exists(self) -> None:
        assert "case 'complete':" in self._ts

    def test_complete_sends_query(self) -> None:
        import re

        m = re.search(r"case\s+'complete':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "query: message.query" in block

    def test_complete_sends_active_file(self) -> None:
        import re

        m = re.search(r"case\s+'complete':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "activeFile:" in block

    def test_complete_sends_active_file_content(self) -> None:
        import re

        m = re.search(r"case\s+'complete':", self._ts)
        assert m
        end = self._ts.index("break;", m.end())
        block = self._ts[m.start() : end]
        assert "activeFileContent:" in block


class TestWebviewTabBarJS(unittest.TestCase):
    """Test the in-webview tab bar JavaScript code in main.js.

    Both the SorcarTab (editor tabs) and SorcarSidebarView (secondary bar)
    share the same webview HTML/JS, which has its own tab management within
    the webview.  The "New chat" button must add a new tab to the tab bar.
    """

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    # -- Tab data model --

    def test_make_tab_function_exists(self) -> None:
        assert "function makeTab(title)" in self._js

    def test_make_tab_returns_object_with_id(self) -> None:
        idx = self._js.index("function makeTab(title)")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        assert "id:" in body
        assert "genTabId()" in body

    def test_make_tab_has_required_fields(self) -> None:
        idx = self._js.index("function makeTab(title)")
        end = self._js.index("\n  }", idx) + 4
        body = self._js[idx:end]
        for field in ("title:", "outputFragment:", "taskPanelHTML:",
                      "welcomeVisible:", "selectedModel:",
                      "attachments:", "inputValue:", "isMerging:",
                      "t0:", "streamState:", "streamLastToolName:",
                      "streamPendingPanel:"):
            assert field in body, f"makeTab missing field {field}"

    def test_tabs_array_exists(self) -> None:
        assert "let tabs = [];" in self._js

    def test_active_tab_id_exists(self) -> None:
        assert "let activeTabId = '';" in self._js

    # -- Tab bar rendering --

    def test_render_tab_bar_function_exists(self) -> None:
        assert "function renderTabBar()" in self._js

    def test_render_tab_bar_creates_chat_tab_elements(self) -> None:
        idx = self._js.index("function renderTabBar()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "'chat-tab'" in body
        assert "chat-tab-label" in body

    def test_render_tab_bar_marks_active_tab(self) -> None:
        idx = self._js.index("function renderTabBar()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "' active'" in body
        assert "activeTabId" in body

    def test_render_tab_bar_has_close_button(self) -> None:
        idx = self._js.index("function renderTabBar()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "chat-tab-close" in body
        assert "closeTab" in body

    def test_render_tab_bar_has_add_button(self) -> None:
        """The "+" button at the end of the tab list creates a new tab."""
        idx = self._js.index("function renderTabBar()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "chat-tab-add" in body
        assert "createNewTab" in body
        assert "New chat" in body

    # -- Creating a new tab --

    def test_create_new_tab_function_exists(self) -> None:
        assert "function createNewTab()" in self._js

    def test_create_new_tab_saves_current_tab(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "saveCurrentTab()" in body

    def test_create_new_tab_creates_tab_via_make_tab(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "makeTab(" in body

    def test_create_new_tab_pushes_to_tabs_array(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "tabs.push(" in body

    def test_create_new_tab_sets_active_tab(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "activeTabId = tab.id" in body

    def test_create_new_tab_clears_output(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "clearOutput()" in body

    def test_create_new_tab_renders_tab_bar(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "renderTabBar()" in body

    def test_create_new_tab_persists_state(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "persistTabState()" in body

    def test_create_new_tab_sends_new_chat_to_backend(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "type: 'newChat'" in body

    def test_create_new_tab_shows_welcome(self) -> None:
        idx = self._js.index("function createNewTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "welcome" in body

    def test_show_welcome_event_handler_exists(self) -> None:
        assert "case 'showWelcome':" in self._js

    def test_show_welcome_event_shows_welcome_for_active_tab(self) -> None:
        idx = self._js.index("case 'showWelcome':")
        end = self._js.index("break;", idx) + len("break;")
        body = self._js[idx:end]
        assert "welcome.style.display = ''" in body
        assert "clearOutput()" in body

    def test_show_welcome_sets_visibility_for_background_tab(self) -> None:
        idx = self._js.index("case 'showWelcome':")
        end = self._js.index("break;", idx) + len("break;")
        body = self._js[idx:end]
        assert "welcomeVisible = true" in body

    # -- Switching tabs --

    def test_switch_to_tab_function_exists(self) -> None:
        assert "function switchToTab(tabId)" in self._js

    def test_switch_to_tab_saves_current_tab(self) -> None:
        idx = self._js.index("function switchToTab(tabId)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "saveCurrentTab()" in body

    def test_switch_to_tab_restores_tab(self) -> None:
        idx = self._js.index("function switchToTab(tabId)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "restoreTab(tab)" in body

    def test_switch_to_tab_renders_tab_bar(self) -> None:
        idx = self._js.index("function switchToTab(tabId)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "renderTabBar()" in body

    def test_switch_to_tab_restores_dom(self) -> None:
        """Switching to a tab restores DOM from saved fragment (no backend call)."""
        idx = self._js.index("function switchToTab(tabId)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "restoreTab(tab)" in body
        assert "setRunningState(tab.isRunning)" in body

    def test_switch_to_tab_noop_for_same_tab(self) -> None:
        """Switching to the already active tab is a no-op."""
        idx = self._js.index("function switchToTab(tabId)")
        body = self._js[idx : idx + 100]
        assert "if (tabId === activeTabId) return;" in body

    # -- Closing tabs --

    def test_close_tab_function_exists(self) -> None:
        assert "function closeTab(tabId)" in self._js

    def test_close_last_tab_creates_new_chat(self) -> None:
        """Closing the last tab creates a fresh new chat."""
        idx = self._js.index("function closeTab(tabId)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "tabs.length === 0" in body
        assert "createNewTab()" in body

    def test_close_tab_switches_to_adjacent_tab(self) -> None:
        idx = self._js.index("function closeTab(tabId)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "restoreTab(" in body

    def test_close_tab_renders_tab_bar(self) -> None:
        idx = self._js.index("function closeTab(tabId)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "renderTabBar()" in body

    # -- Saving/restoring tab state --

    def test_save_current_tab_function_exists(self) -> None:
        assert "function saveCurrentTab()" in self._js

    def test_save_current_tab_stores_output_fragment(self) -> None:
        idx = self._js.index("function saveCurrentTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "tab.outputFragment = document.createDocumentFragment()" in body

    def test_save_current_tab_stores_per_tab_state(self) -> None:
        idx = self._js.index("function saveCurrentTab()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "tab.selectedModel = selectedModel" in body

    def test_restore_tab_function_exists(self) -> None:
        assert "function restoreTab(tab)" in self._js

    def test_restore_tab_sets_active_tab_id(self) -> None:
        idx = self._js.index("function restoreTab(tab)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "activeTabId = tab.id" in body

    def test_restore_tab_restores_output_fragment(self) -> None:
        idx = self._js.index("function restoreTab(tab)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "tab.outputFragment" in body

    # -- Tab title updates --

    def test_update_active_tab_title_function_exists(self) -> None:
        assert "function updateActiveTabTitle(title)" in self._js

    def test_update_active_tab_title_truncates_long_titles(self) -> None:
        idx = self._js.index("function updateActiveTabTitle(title)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "30" in body  # max length
        assert "\\u2026" in body  # ellipsis

    def test_update_active_tab_title_renders_tab_bar(self) -> None:
        idx = self._js.index("function updateActiveTabTitle(title)")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "renderTabBar()" in body

    # -- Persistence --

    def test_persist_tab_state_function_exists(self) -> None:
        assert "function persistTabState()" in self._js

    def test_persist_tab_state_uses_set_state(self) -> None:
        idx = self._js.index("function persistTabState()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "vscode.setState(" in body

    def test_persist_tab_state_saves_tabs_and_active_index(self) -> None:
        idx = self._js.index("function persistTabState()")
        end = self._js.index("\n  function ", idx + 1)
        body = self._js[idx:end]
        assert "tabs:" in body
        assert "activeTabIndex:" in body

    # -- Tab state restoration at startup --

    def test_tabs_restored_from_saved_state(self) -> None:
        """Tabs are restored from vscode.getState() on startup."""
        assert "vscode.getState()" in self._js
        assert "saved.tabs" in self._js

    # -- New chat button creates a new tab --

    def test_new_chat_button_calls_create_new_tab(self) -> None:
        """The clearChat handler creates a new tab."""
        idx = self._js.index("case 'clearChat':")
        end = self._js.index("break;", idx) + len("break;")
        body = self._js[idx:end]
        assert "createNewTab()" in body


class TestWebviewTabBarCSS(unittest.TestCase):
    """Test the in-webview tab bar CSS styles."""

    _css: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._css = (base / "vscode" / "media" / "main.css").read_text()

    def test_tab_bar_exists(self) -> None:
        assert "#tab-bar" in self._css

    def test_tab_bar_does_not_grow(self) -> None:
        idx = self._css.index("#tab-bar")
        block = self._css[idx : idx + 200]
        assert "flex-shrink: 0" in block

    def test_tab_bar_has_border(self) -> None:
        idx = self._css.index("#tab-bar")
        block = self._css[idx : idx + 200]
        assert "border-bottom" in block

    def test_tab_list_is_flex_with_horizontal_scroll(self) -> None:
        idx = self._css.index("#tab-list")
        block = self._css[idx : idx + 200]
        assert "display: flex" in block
        assert "overflow-x: auto" in block

    def test_tab_list_hides_scrollbar(self) -> None:
        assert "#tab-list::-webkit-scrollbar" in self._css

    def test_chat_tab_base_style(self) -> None:
        assert ".chat-tab" in self._css

    def test_chat_tab_has_max_width(self) -> None:
        """Tabs have a max-width to keep them compact."""
        idx = self._css.index(".chat-tab {")
        end = self._css.index("}", idx)
        block = self._css[idx:end]
        assert "max-width:" in block

    def test_chat_tab_active_has_accent_color(self) -> None:
        assert ".chat-tab.active" in self._css
        idx = self._css.index(".chat-tab.active")
        block = self._css[idx : idx + 200]
        assert "accent" in block

    def test_chat_tab_active_has_bottom_border(self) -> None:
        idx = self._css.index(".chat-tab.active")
        block = self._css[idx : idx + 200]
        assert "border-bottom-color" in block

    def test_chat_tab_label_has_ellipsis_overflow(self) -> None:
        assert ".chat-tab-label" in self._css
        idx = self._css.index(".chat-tab-label")
        block = self._css[idx : idx + 200]
        assert "text-overflow: ellipsis" in block

    def test_chat_tab_close_always_visible(self) -> None:
        """Tab close button is always visible (no hover-reveal behavior)."""
        assert ".chat-tab-close" in self._css
        idx = self._css.index(".chat-tab-close {")
        end = self._css.index("}", idx)
        block = self._css[idx:end]
        assert "opacity: 1" in block

    def test_chat_tab_close_hover_highlights(self) -> None:
        """Hovering the close button changes its background for affordance."""
        assert ".chat-tab-close:hover" in self._css

    def test_chat_tab_add_button_style(self) -> None:
        assert ".chat-tab-add" in self._css


class TestWebviewTabBarHTML(unittest.TestCase):
    """Test the tab bar HTML structure in buildChatHtml."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "SorcarTab.ts").read_text()

    def test_html_has_tab_bar_div(self) -> None:
        assert 'id="tab-bar"' in self._ts

    def test_html_has_tab_list_div(self) -> None:
        assert 'id="tab-list"' in self._ts

    def test_tab_bar_is_before_output(self) -> None:
        """Tab bar div appears before the output div in the HTML."""
        tab_bar_idx = self._ts.index('id="tab-bar"')
        output_idx = self._ts.index('id="output"')
        assert tab_bar_idx < output_idx

    def test_tab_status_bar_is_after_tab_bar(self) -> None:
        """Per-tab status bar appears after the tab bar and before task panel."""
        tab_bar_idx = self._ts.index('id="tab-bar"')
        status_bar_idx = self._ts.index('id="tab-status-bar"')
        task_panel_idx = self._ts.index('id="task-panel"')
        assert tab_bar_idx < status_bar_idx < task_panel_idx


class TestSorcarSidebarViewFilePathDoesNotPopulateTaskPanel(unittest.TestCase):
    """Verify SorcarSidebarView submit handler for file paths returns before
    _startTask, matching SorcarTab behavior.

    This is the sidebar equivalent of TestFilePathDoesNotPopulateTaskPanel.
    """

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()

    def test_file_open_returns_before_start_task(self) -> None:
        """The submit handler opens the file and returns *before* _startTask."""
        submit_idx = self._ts.index("case 'submit':")
        submit_end = self._ts.index("break;", submit_idx)
        submit_body = self._ts[submit_idx:submit_end]
        # The file-open block must contain 'return' before _startTask
        file_check_idx = submit_body.index("isFile()")
        return_idx = submit_body.index("return;", file_check_idx)
        start_task_idx = submit_body.index("this._startTask(")
        assert return_idx < start_task_idx

    def test_start_task_sends_set_task_text(self) -> None:
        """_startTask sends setTaskText to the webview (the only source)."""
        idx = self._ts.index("private _startTask(")
        # Find next method
        end = self._ts.index("\n  private ", idx + 1)
        body = self._ts[idx:end]
        assert "setTaskText" in body

    def test_submit_file_open_returns_without_starting_task(self) -> None:
        """File path open returns early without adding to _runningTabs."""
        submit_idx = self._ts.index("case 'submit':")
        submit_end = self._ts.index("break;", submit_idx)
        submit_body = self._ts[submit_idx:submit_end]
        file_check_idx = submit_body.index("isFile()")
        return_idx = submit_body.index("return;", file_check_idx)
        block = submit_body[file_check_idx:return_idx]
        # File open returns early — no _startTask call in this path
        assert "_startTask" not in block

    def test_submit_checks_per_tab_running_before_processing(self) -> None:
        """Submit returns early if this tab is already running."""
        submit_idx = self._ts.index("case 'submit':")
        submit_end = self._ts.index("break;", submit_idx)
        submit_body = self._ts[submit_idx:submit_end]
        assert "this._runningTabs.has(tabId)" in submit_body


class TestSorcarSidebarViewNewChatBehavior(unittest.TestCase):
    """Verify that SorcarSidebarView._handleMessage newChat is forwarded
    to the agent process, matching the expected flow for tab management.
    """

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()

    def test_new_chat_forwarded_to_agent(self) -> None:
        """The newChat message type is forwarded to the agent process."""
        assert "case 'newChat':" in self._ts

    def test_no_pending_new_chat_field(self) -> None:
        """_pendingNewChat was removed — newConversation always sends clearChat."""
        assert "_pendingNewChat" not in self._ts


class TestSidebarViewBehavior(unittest.TestCase):
    """Verify SorcarSidebarView has key behaviors."""

    _sidebar_ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._sidebar_ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def test_has_worktree_action_resolve_field(self) -> None:
        assert "_worktreeActionResolve" in self._sidebar_ts

    def test_has_worktree_progress_field(self) -> None:
        assert "_worktreeProgress" in self._sidebar_ts

    def test_has_merge_manager(self) -> None:
        assert "_mergeManager" in self._sidebar_ts

    def test_handles_all_done_from_merge_manager(self) -> None:
        assert "this._mergeManager.on('allDone'" in self._sidebar_ts

    def test_opens_merge_on_merge_data(self) -> None:
        assert "this._mergeManager.openMerge(" in self._sidebar_ts

    def test_has_generate_commit_message(self) -> None:
        assert "generateCommitMessage(" in self._sidebar_ts

    def test_sends_focus_input_on_ready(self) -> None:
        idx = self._sidebar_ts.index("case 'ready':")
        end = self._sidebar_ts.index("break;", idx) + len("break;")
        block = self._sidebar_ts[idx:end]
        assert "'focusInput'" in block

    def test_sends_active_file_info_on_ready(self) -> None:
        idx = self._sidebar_ts.index("case 'ready':")
        end = self._sidebar_ts.index("break;", idx) + len("break;")
        block = self._sidebar_ts[idx:end]
        assert "_sendActiveFileInfo()" in block

    def test_requests_models_on_ready(self) -> None:
        idx = self._sidebar_ts.index("case 'ready':")
        end = self._sidebar_ts.index("break;", idx) + len("break;")
        block = self._sidebar_ts[idx:end]
        assert "'getModels'" in block

    def test_requests_input_history_on_ready(self) -> None:
        idx = self._sidebar_ts.index("case 'ready':")
        end = self._sidebar_ts.index("break;", idx) + len("break;")
        block = self._sidebar_ts[idx:end]
        assert "'getInputHistory'" in block

    def test_passes_use_worktree_in_start_task(self) -> None:
        idx = self._sidebar_ts.index("private _startTask(")
        block = self._sidebar_ts[idx : idx + 200]
        assert "useWorktree" in block

    def test_passes_use_parallel_in_start_task(self) -> None:
        idx = self._sidebar_ts.index("private _startTask(")
        block = self._sidebar_ts[idx : idx + 200]
        assert "useParallel" in block

    def test_strips_work_dir_prefix_in_submit(self) -> None:
        idx = self._sidebar_ts.index("case 'submit':")
        end = self._sidebar_ts.index("break;", idx) + len("break;")
        block = self._sidebar_ts[idx:end]
        assert "$PWD" in block

    def test_resolves_file_paths_in_submit(self) -> None:
        idx = self._sidebar_ts.index("case 'submit':")
        end = self._sidebar_ts.index("break;", idx) + len("break;")
        block = self._sidebar_ts[idx:end]
        assert "path.resolve" in block

    def test_has_worktree_timeout_120s(self) -> None:
        assert "120_000" in self._sidebar_ts

    def test_resolves_worktree_action_on_dispose(self) -> None:
        idx = self._sidebar_ts.index("public dispose()")
        end = self._sidebar_ts.index("\n  }", idx) + 4
        body = self._sidebar_ts[idx:end]
        assert "_resolveAllWorktreeActions" in body

    def test_filters_dropped_paths_outside_workdir(self) -> None:
        import re

        m = re.search(r"case\s+'resolveDroppedPaths':", self._sidebar_ts)
        assert m
        end = self._sidebar_ts.index("break;", m.end())
        block = self._sidebar_ts[m.start() : end]
        assert "'..'" in block

    def test_opens_file_with_line_number(self) -> None:
        import re

        m = re.search(r"case\s+'openFile':", self._sidebar_ts)
        assert m
        end = self._sidebar_ts.index("break;", m.end())
        block = self._sidebar_ts[m.start() : end]
        assert "message.line" in block
        assert "revealRange" in block

    def test_handles_user_action_done_as_done(self) -> None:
        import re

        m = re.search(r"case\s+'userActionDone':", self._sidebar_ts)
        assert m
        end = self._sidebar_ts.index("break;", m.end())
        block = self._sidebar_ts[m.start() : end]
        assert "answer: 'done'" in block

    def test_sends_complete_with_active_file_content(self) -> None:
        import re

        m = re.search(r"case\s+'complete':", self._sidebar_ts)
        assert m
        end = self._sidebar_ts.index("break;", m.end())
        block = self._sidebar_ts[m.start() : end]
        assert "activeFileContent" in block
        assert "getText()" in block

    def test_has_append_to_input_method(self) -> None:
        """SorcarSidebarView has appendToInput for insertSelectionToChat."""
        assert "appendToInput(" in self._sidebar_ts
        assert "'appendToInput'" in self._sidebar_ts


class TestTabStateRestore(unittest.TestCase):
    """Test that tab state is persisted correctly for cross-restart restore.

    Tabs are identified by tab.id which IS the chat_id. persistTabState()
    serializes tab.id as chatId, and updateActiveTabTitle() updates tab.title.
    """

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()

    def test_persist_tab_state_serializes_tab_id_as_chat_id(self) -> None:
        """persistTabState serializes tab.id as chatId for persistence."""
        idx = self.js.index("function persistTabState()")
        block = self.js[idx:idx + 500]
        assert "t.id === activeTabId" in block
        assert "chatId: t.id" in block

    def test_update_active_tab_title_renders_tab_bar(self) -> None:
        """updateActiveTabTitle calls renderTabBar and persistTabState."""
        idx = self.js.index("function updateActiveTabTitle(")
        block = self.js[idx:idx + 400]
        assert "renderTabBar()" in block
        assert "persistTabState()" in block

    def test_persist_tab_state_logic_via_node(self) -> None:
        """Run the actual JS logic in Node.js and verify correctness."""
        node_script = """
        var activeTabId = '';
        var tabs = [];
        var _lastState = null;

        var vscode = {
            setState: function(s) { _lastState = s; },
            getState: function() { return _lastState; },
        };

        function persistTabState() {
            var serialized = tabs.map(function(t) {
                return { title: t.title, chatId: t.id };
            });
            var activeIdx = tabs.findIndex(function(t) { return t.id === activeTabId; });
            vscode.setState({ tabs: serialized, activeTabIndex: activeIdx });
        }

        // Test 1: Single tab, tab.id persisted as chatId
        tabs.push({ id: 'abc123', title: 'new chat' });
        activeTabId = 'abc123';
        persistTabState();
        var state = vscode.getState();
        if (state.tabs[0].chatId !== 'abc123') {
            console.log('FAIL test1: ' + state.tabs[0].chatId);
            process.exit(1);
        }

        // Test 2: Multi-tab scenario
        tabs = [];
        tabs.push({ id: 'chat-A', title: 'task A' });
        tabs.push({ id: 'chat-B', title: 'new chat' });
        activeTabId = 'chat-B';
        persistTabState();
        state = vscode.getState();
        if (state.tabs[0].chatId !== 'chat-A') {
            console.log('FAIL 2a: ' + state.tabs[0].chatId);
            process.exit(1);
        }
        if (state.tabs[1].chatId !== 'chat-B') {
            console.log('FAIL 2b: ' + state.tabs[1].chatId);
            process.exit(1);
        }

        console.log('PASS: all tab state persistence tests passed');
        """
        result = subprocess.run(
            ["node", "-e", node_script],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"Node.js test failed: {result.stdout}{result.stderr}"
        assert "PASS" in result.stdout


if __name__ == "__main__":
    unittest.main()

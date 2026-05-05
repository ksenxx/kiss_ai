"""Tests that the settings panel saves configuration on close instead of via a button.

The "Save Configuration" button was removed. Now, closing the config sidebar
automatically collects and saves the form data.
"""

import re
import unittest
from pathlib import Path

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


class TestSaveButtonRemovedFromHTML(unittest.TestCase):
    """The cfg-save-btn element must not appear in any HTML template."""

    def test_sorcar_tab_has_no_save_button(self) -> None:
        ts = (_VSCODE_DIR / "src" / "SorcarTab.ts").read_text()
        assert "cfg-save-btn" not in ts

    def test_web_server_has_no_save_button(self) -> None:
        py = (_VSCODE_DIR / "web_server.py").read_text()
        assert "cfg-save-btn" not in py

    def test_main_js_has_no_save_button_reference(self) -> None:
        js = (_VSCODE_DIR / "media" / "main.js").read_text()
        assert "cfg-save-btn" not in js

    def test_css_has_no_save_button_style(self) -> None:
        css = (_VSCODE_DIR / "media" / "main.css").read_text()
        assert "config-save-btn" not in css


class TestCloseConfigSidebarSavesConfig(unittest.TestCase):
    """closeConfigSidebar() must save config when the panel is open."""

    _js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = (_VSCODE_DIR / "media" / "main.js").read_text()

    def _extract_close_fn(self) -> str:
        """Extract the closeConfigSidebar function body."""
        m = re.search(
            r"function closeConfigSidebar\(\)\s*\{",
            self._js,
        )
        assert m, "closeConfigSidebar function not found"
        start = m.start()
        brace = 0
        for i in range(m.end() - 1, len(self._js)):
            if self._js[i] == "{":
                brace += 1
            elif self._js[i] == "}":
                brace -= 1
                if brace == 0:
                    return self._js[start : i + 1]
        raise AssertionError("Could not extract closeConfigSidebar body")  # pragma: no cover

    def test_saves_config_when_open(self) -> None:
        """closeConfigSidebar checks if the sidebar is open before saving."""
        body = self._extract_close_fn()
        assert "configSidebar.classList.contains('open')" in body
        assert "collectConfigForm()" in body
        assert "saveConfig" in body

    def test_guarded_by_open_check(self) -> None:
        """The saveConfig post is inside the 'open' check, not unconditional."""
        body = self._extract_close_fn()
        # The if-check must come before the postMessage
        open_check_pos = body.index("classList.contains('open')")
        save_pos = body.index("saveConfig")
        assert open_check_pos < save_pos

    def test_no_standalone_save_button_listener(self) -> None:
        """There must be no event listener on cfg-save-btn."""
        assert "cfgSaveBtn.addEventListener" not in self._js
        assert "cfg-save-btn" not in self._js


class TestAllClosePaths(unittest.TestCase):
    """Every path that closes the config sidebar must go through closeConfigSidebar."""

    _js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = (_VSCODE_DIR / "media" / "main.js").read_text()

    def test_close_button_calls_close_fn(self) -> None:
        """configSidebarClose click handler calls closeConfigSidebar."""
        assert "configSidebarClose.addEventListener('click', closeConfigSidebar)" in self._js

    def test_overlay_calls_close_fn(self) -> None:
        """configSidebarOverlay click handler calls closeConfigSidebar."""
        assert "configSidebarOverlay.addEventListener('click', closeConfigSidebar)" in self._js

    def test_config_btn_toggle_calls_close_fn(self) -> None:
        """Clicking config button when open calls closeConfigSidebar."""
        # The config button handler toggles: if open -> closeConfigSidebar()
        pattern = re.compile(
            r"configBtn\.addEventListener\('click'.*?closeConfigSidebar\(\)",
            re.DOTALL,
        )
        assert pattern.search(self._js)

    def test_open_config_sidebar_calls_close_first(self) -> None:
        """openConfigSidebar calls closeConfigSidebar at the start for clean state."""
        m = re.search(
            r"function openConfigSidebar\(\)\s*\{(.*?)\n  \}",
            self._js,
            re.DOTALL,
        )
        assert m
        body = m.group(1)
        lines = [ln.strip() for ln in body.strip().splitlines()]
        assert lines[0] == "closeConfigSidebar();", (
            "openConfigSidebar must call closeConfigSidebar() first"
        )

    def test_open_history_sidebar_calls_close_config(self) -> None:
        """Opening the history sidebar closes the config sidebar."""
        assert "closeConfigSidebar();" in self._js

    def test_open_frequent_sidebar_calls_close_config(self) -> None:
        """openFrequentSidebar calls closeConfigSidebar."""
        m = re.search(
            r"function openFrequentSidebar\(\)\s*\{(.*?)\n  \}",
            self._js,
            re.DOTALL,
        )
        assert m
        assert "closeConfigSidebar();" in m.group(1)

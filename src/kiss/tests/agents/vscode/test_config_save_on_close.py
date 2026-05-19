"""Tests that the settings sub-tab saves configuration on close/switch.

The "Save Configuration" button was removed long ago, and the entire
``#config-sidebar`` panel has now been merged into ``#sidebar`` as a
``Settings`` sub-tab.  Configuration must be saved automatically when

* the unified sidebar is closed while the Settings tab is active, or
* the user switches away from the Settings tab to ``History`` /
  ``Frequent``.

These tests pin both behaviours by reading the real ``main.js``.
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


class TestConfigSidebarRemoved(unittest.TestCase):
    """The standalone ``#config-sidebar`` element is gone."""

    def test_no_config_sidebar_in_extension_html(self) -> None:
        ts = (_VSCODE_DIR / "src" / "SorcarTab.ts").read_text()
        assert 'id="config-sidebar"' not in ts
        assert 'id="config-sidebar-overlay"' not in ts
        assert 'id="config-sidebar-close"' not in ts

    def test_no_config_sidebar_in_webapp_html(self) -> None:
        py = (_VSCODE_DIR / "web_server.py").read_text()
        assert 'id="config-sidebar"' not in py
        assert 'id="config-sidebar-overlay"' not in py
        assert 'id="config-sidebar-close"' not in py

    def test_no_config_sidebar_references_in_js(self) -> None:
        js = (_VSCODE_DIR / "media" / "main.js").read_text()
        for sym in (
            "openConfigSidebar",
            "closeConfigSidebar",
            "configSidebar",
            "configSidebarOverlay",
            "configSidebarClose",
            "configBtn",
        ):
            assert sym not in js, f"{sym} should be gone from main.js"


class TestSettingsSavesOnCloseOrSwitch(unittest.TestCase):
    """``switchSidebarTab`` and ``closeSidebar`` flush the settings form."""

    _js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = (_VSCODE_DIR / "media" / "main.js").read_text()

    def _extract_fn(self, name: str) -> str:
        """Return the source of ``function <name>() { … }`` (balanced)."""
        m = re.search(r"function\s+" + re.escape(name) + r"\(", self._js)
        assert m, f"{name} not found"
        # Find the opening brace.
        i = self._js.index("{", m.end())
        depth = 1
        j = i + 1
        while j < len(self._js) and depth:
            c = self._js[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return self._js[m.start() : j + 1]
            j += 1
        raise AssertionError(f"unterminated {name}")

    def test_close_sidebar_saves_settings(self) -> None:
        body = self._extract_fn("closeSidebar")
        assert "currentSidebarTab" in body, (
            "closeSidebar must inspect currentSidebarTab"
        )
        assert "'settings'" in body, "closeSidebar must check the settings tab"
        assert "saveSettingsIfPopulated" in body or "saveConfig" in body, (
            "closeSidebar must save the settings form before closing"
        )

    def test_switch_sidebar_tab_saves_on_leaving_settings(self) -> None:
        body = self._extract_fn("switchSidebarTab")
        assert "currentSidebarTab" in body
        assert "'settings'" in body
        assert "saveSettingsIfPopulated" in body or "saveConfig" in body

    def test_save_settings_helper_is_guarded_by_form_populated(self) -> None:
        helper = self._extract_fn("saveSettingsIfPopulated")
        assert "configFormPopulated" in helper
        assert "collectConfigForm()" in helper
        assert "saveConfig" in helper

    def test_settings_tab_button_switches_to_settings(self) -> None:
        """Clicking the in-panel Settings button calls switchSidebarTab('settings')."""
        m = re.search(
            r"sidebarTabSettingsBtn\.addEventListener\('click'.*?\}\);",
            self._js,
            re.DOTALL,
        )
        assert m, "sidebarTabSettingsBtn click handler not found"
        assert "switchSidebarTab('settings')" in m.group(0)

    def test_switch_to_settings_posts_get_config(self) -> None:
        body = self._extract_fn("switchSidebarTab")
        # Settings branch must request the latest config from the backend.
        assert "getConfig" in body
        # And reset the populated flag so a subsequent close won't re-save.
        assert "configFormPopulated = false" in body

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that the standalone Settings panel saves configuration on close.

The "Save Configuration" button was removed long ago, and the
``#config-sidebar`` panel was first merged into the unified ``#sidebar``
and later split out into a dedicated ``#settings-panel`` that slides in
from the right.  Configuration is now saved automatically when the
standalone Settings panel is closed while its form is populated; the
helper ``saveSettingsIfPopulated`` is invoked by ``closeSettingsPanel``.

These tests pin the surviving auto-save behaviour by reading the real
``main.js``.
"""

import re
import unittest
from pathlib import Path

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"









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

    def test_close_settings_panel_saves_settings(self) -> None:
        body = self._extract_fn("closeSettingsPanel")
        assert "saveSettingsIfPopulated" in body, (
            "closeSettingsPanel must save the settings form before closing"
        )

    def test_open_settings_panel_posts_get_config(self) -> None:
        body = self._extract_fn("openSettingsPanel")
        # Opening the panel must request the latest config from the backend.
        assert "getConfig" in body
        # And reset the populated flag so a subsequent close won't re-save
        # stale values that were never populated by a fresh fetch.
        assert "configFormPopulated = false" in body

    def test_save_settings_helper_is_guarded_by_form_populated(self) -> None:
        helper = self._extract_fn("saveSettingsIfPopulated")
        assert "configFormPopulated" in helper
        assert "collectConfigForm()" in helper
        assert "saveConfig" in helper

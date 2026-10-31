# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that the standalone Settings panel saves configuration on close.

The "Save Configuration" button was removed long ago, and the
``#config-sidebar`` panel was first merged into the unified ``#sidebar``
and later split out into a dedicated ``#settings-panel`` that slides in
from the right.  The fields the user edited are saved automatically when the
standalone Settings panel is closed; the helper
``saveSettingsIfPopulated`` is invoked by ``closeSettingsPanel``.

These tests pin the surviving auto-save behaviour by reading the real
``main.js``.
"""

import re
import unittest
from pathlib import Path

_WEBAPP_DIR = Path(__file__).resolve().parents[2] / "webapp"









class TestSettingsSavesOnCloseOrSwitch(unittest.TestCase):
    """``switchSidebarTab`` and ``closeSidebar`` flush the settings form."""

    _js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = (_WEBAPP_DIR / "media" / "main.js").read_text()

    def _extract_fn(self, name: str) -> str:
        """Return the source of ``function <name>() { … }`` (balanced)."""
        m = re.search(r"function\s+" + re.escape(name) + r"\(", self._js)
        assert m, f"{name} not found"
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
        # The open sheet loads the settings of the machine the chat
        # header shows (loadSettingsSheet, run again when it changes).
        assert "loadSettingsSheet()" in self._extract_fn("openSettingsPanel")
        body = self._extract_fn("loadSettingsSheet")
        assert "getConfig" in body
        assert "settingsEditedFields.clear()" in body

    def test_save_settings_helper_sends_only_edited_fields(self) -> None:
        # The daemon merges the payload, so the close flushes the edited
        # fields alone and stays silent when nothing was edited.
        helper = self._extract_fn("saveSettingsIfPopulated")
        assert "settingsEditedFields.size === 0" in helper
        assert "collectConfigForm()" in helper
        assert "settingsEditedFields.has(id)" in self._extract_fn("collectConfigForm")
        assert "saveConfig" in helper

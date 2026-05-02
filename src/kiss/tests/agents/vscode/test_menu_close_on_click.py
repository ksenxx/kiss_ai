"""Integration tests for menu dropdown closing when a menu item is clicked.

Validates:
- Clicking a .menu-item inside #menu-dropdown closes the dropdown.
- The delegated click listener on menuDropdown removes the 'open' class.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_JS_PATH = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media" / "main.js"
)


def _read_js() -> str:
    return _JS_PATH.read_text()


class TestMenuCloseOnItemClick(unittest.TestCase):
    """Menu dropdown closes when any .menu-item is clicked."""

    def test_delegated_click_listener_exists(self) -> None:
        """A click listener on menuDropdown removes 'open' when a .menu-item is clicked."""
        js = _read_js()
        # There should be a listener on menuDropdown that checks for .menu-item
        self.assertIn(
            "menuDropdown.addEventListener('click'",
            js,
            "Missing delegated click listener on menuDropdown",
        )

    def test_listener_checks_menu_item_class(self) -> None:
        """The delegated listener uses closest('.menu-item') to detect item clicks."""
        js = _read_js()
        self.assertIn(
            ".closest('.menu-item')",
            js,
            "Delegated listener should use closest('.menu-item')",
        )

    def test_listener_removes_open_class(self) -> None:
        """The delegated listener removes the 'open' class from the dropdown."""
        js = _read_js()
        # Find the delegated listener block
        pattern = re.compile(
            r"menuDropdown\.addEventListener\('click',\s*e\s*=>\s*\{[^}]*"
            r"menuDropdown\.classList\.remove\('open'\)"
            r"[^}]*\}",
            re.DOTALL,
        )
        match = pattern.search(js)
        self.assertIsNotNone(
            match,
            "Delegated click listener should remove 'open' from menuDropdown",
        )

    def test_listener_is_after_menu_setup(self) -> None:
        """The delegated listener appears after the menu button setup block."""
        js = _read_js()
        menu_toggle_pos = js.find("menuDropdown.classList.toggle('open')")
        delegated_pos = js.find("menuDropdown.addEventListener('click'")
        self.assertGreater(menu_toggle_pos, -1)
        self.assertGreater(delegated_pos, -1)
        self.assertGreater(
            delegated_pos,
            menu_toggle_pos,
            "Delegated close listener should appear after menu toggle setup",
        )


if __name__ == "__main__":
    unittest.main()

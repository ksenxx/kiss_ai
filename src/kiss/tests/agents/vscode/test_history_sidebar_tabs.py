# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests pinning the sidebar layout after the tabbed sidebar
was replaced with separate panels.

The chat webview previously had two separate tab-bar buttons —
``history-btn`` and ``frequent-btn`` — each of which opened a distinct
sliding sidebar (``#sidebar`` and ``#frequent-sidebar``).  An
intermediate refactor merged them into a single ``#sidebar`` with
in-panel tabs (Running/History/Frequent/Settings); that has since been
replaced again with the current layout where:

* ``#sidebar`` is dedicated to History (running items render inline via
  the ``running-item`` class on history rows).
* Frequent tasks live in a standalone ``#frequent-panel``.
* Settings live in a standalone ``#settings-panel``.

These tests pin the elements that survived the refactor — primarily
that the legacy frequent-only tab-bar button and standalone
``#frequent-sidebar`` are gone in both surfaces and that
``#history-list`` remains the history root.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import _build_html

_VSCODE = Path(__file__).resolve().parents[3] / "agents" / "vscode"
_SORCAR_TAB_TS = _VSCODE / "src" / "SorcarTab.ts"


def _ext_html() -> str:
    return _SORCAR_TAB_TS.read_text(encoding="utf-8")


class TestSidebarTabsMarkup(unittest.TestCase):
    """HTML in both surfaces wires the current sidebar layout."""

    def test_frequent_btn_removed_from_extension_tab_bar(self) -> None:
        self.assertNotIn('id="frequent-btn"', _ext_html())

    def test_frequent_btn_removed_from_webapp_tab_bar(self) -> None:
        self.assertNotIn('id="frequent-btn"', _build_html())

    def test_separate_frequent_sidebar_removed_from_extension(self) -> None:
        self.assertNotIn('id="frequent-sidebar"', _ext_html())
        self.assertNotIn('id="frequent-sidebar-overlay"', _ext_html())
        self.assertNotIn('id="frequent-sidebar-close"', _ext_html())

    def test_separate_frequent_sidebar_removed_from_webapp(self) -> None:
        web = _build_html()
        self.assertNotIn('id="frequent-sidebar"', web)
        self.assertNotIn('id="frequent-sidebar-overlay"', web)
        self.assertNotIn('id="frequent-sidebar-close"', web)

    def test_extension_sidebar_has_history_list(self) -> None:
        """The extension sidebar still surfaces the history list."""
        html = _ext_html()
        self.assertIn('id="history-list"', html)

    def test_webapp_sidebar_has_history_list(self) -> None:
        """The standalone webapp sidebar still surfaces the history list."""
        html = _build_html()
        self.assertIn('id="history-list"', html)


if __name__ == "__main__":
    unittest.main()

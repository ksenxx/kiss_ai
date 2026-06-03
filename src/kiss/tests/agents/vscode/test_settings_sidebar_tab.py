# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests pinning the Settings UI after the unified-sidebar
refactor.

The chat webview previously had a separate ``config-btn`` in the tab-bar
that opened a dedicated ``#config-sidebar``.  An intermediate refactor
merged the History/Frequent/Settings panels into a single tabbed
``#sidebar``.  Both the VS Code extension (``SorcarTab.ts``) and the
standalone web server (``web_server.py::_build_html``) have since
split the Settings and Frequent UIs out into standalone
``#settings-panel`` / ``#frequent-panel`` siblings of ``#sidebar`` —
which now hosts only the History panel.

These tests pin the surviving invariants on both surfaces:

* The tab-bar no longer ships ``config-btn``.
* The legacy ``#config-sidebar`` markup is gone.
* The old tabbed-sidebar buttons (``sidebar-tab-history``,
  ``sidebar-tab-frequent``, ``sidebar-tab-settings``) and their
  ``-panel`` siblings for Frequent / Settings are gone.
* The Settings form (``cfg-*`` controls) lives inside
  ``#settings-panel`` on both surfaces.
* ``#frequent-panel`` exists with a ``#frequent-list`` on both
  surfaces.
* A ``#frequent-tasks-btn`` toggle lives in the model-picker footer on
  both surfaces.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import _build_html

_VSCODE = Path(__file__).resolve().parents[3] / "agents" / "vscode"
_SORCAR_TAB_TS = _VSCODE / "src" / "SorcarTab.ts"


def _ext_html() -> str:
    return _SORCAR_TAB_TS.read_text(encoding="utf-8")


def _section(html: str, container_id: str) -> str:
    """Return the markup of the ``<div>`` with the given id (balanced)."""
    pat = re.compile(r'<div\s+id="' + re.escape(container_id) + r'"')
    m = pat.search(html)
    if not m:
        return ""
    depth = 1
    i = m.end()
    while i < len(html) and depth:
        nxt = html.find("<", i)
        if nxt < 0:
            break
        if html.startswith("<div", nxt) and html[nxt + 4] in " \t\n>":
            depth += 1
            i = nxt + 4
        elif html.startswith("</div>", nxt):
            depth -= 1
            if depth == 0:
                return html[m.start() : nxt + len("</div>")]
            i = nxt + len("</div>")
        else:
            i = nxt + 1
    return html[m.start() :]


class TestSettingsTabMarkup(unittest.TestCase):
    """HTML in both surfaces wires the new Settings sub-tab correctly."""

    def test_config_btn_removed_from_extension_tab_bar(self) -> None:
        bar = _section(_ext_html(), "tab-bar")
        self.assertTrue(bar, "could not locate #tab-bar in SorcarTab.ts")
        self.assertNotIn('id="config-btn"', bar)

    def test_config_btn_removed_from_webapp_tab_bar(self) -> None:
        bar = _section(_build_html(), "tab-bar")
        self.assertTrue(bar, "could not locate #tab-bar in webapp HTML")
        self.assertNotIn('id="config-btn"', bar)

    def test_separate_config_sidebar_removed_from_extension(self) -> None:
        html = _ext_html()
        self.assertNotIn('id="config-sidebar"', html)
        self.assertNotIn('id="config-sidebar-overlay"', html)
        self.assertNotIn('id="config-sidebar-close"', html)

    def test_separate_config_sidebar_removed_from_webapp(self) -> None:
        html = _build_html()
        self.assertNotIn('id="config-sidebar"', html)
        self.assertNotIn('id="config-sidebar-overlay"', html)
        self.assertNotIn('id="config-sidebar-close"', html)

    def test_old_tabbed_sidebar_buttons_removed_from_extension(self) -> None:
        html = _ext_html()
        for el in (
            'id="sidebar-tab-history"',
            'id="sidebar-tab-frequent"',
            'id="sidebar-tab-settings"',
            'id="sidebar-tab-frequent-panel"',
            'id="sidebar-tab-settings-panel"',
        ):
            self.assertNotIn(
                el,
                html,
                f"{el} should be gone from SorcarTab.ts (tabbed-sidebar layout)",
            )

    def test_old_tabbed_sidebar_buttons_removed_from_webapp(self) -> None:
        html = _build_html()
        for el in (
            'id="sidebar-tab-history"',
            'id="sidebar-tab-frequent"',
            'id="sidebar-tab-settings"',
            'id="sidebar-tab-frequent-panel"',
            'id="sidebar-tab-settings-panel"',
        ):
            self.assertNotIn(
                el,
                html,
                f"{el} should be gone from webapp HTML (tabbed-sidebar layout)",
            )

    def test_webapp_sidebar_contains_only_history(self) -> None:
        sidebar = _section(_build_html(), "sidebar")
        self.assertTrue(sidebar, "could not locate #sidebar in webapp HTML")
        self.assertIn('id="sidebar-tab-history-panel"', sidebar)
        self.assertIn('id="history-list"', sidebar)
        # The settings / frequent forms must NOT be nested inside #sidebar.
        self.assertNotIn('id="frequent-list"', sidebar)
        self.assertNotIn('id="cfg-max-budget"', sidebar)
        self.assertNotIn('id="config-form"', sidebar)

    def test_extension_settings_panel_contains_config_form(self) -> None:
        # In the extension the settings UI lives in the standalone
        # ``#settings-panel`` (sibling of ``#sidebar``), not as a
        # sub-tab inside ``#sidebar``.
        panel = _section(_ext_html(), "settings-panel")
        self.assertTrue(panel, "#settings-panel missing in SorcarTab.ts")
        for inp in (
            'id="cfg-max-budget"',
            'id="cfg-custom-endpoint"',
            'id="cfg-custom-api-key"',
            'id="cfg-custom-headers"',
            'id="cfg-use-web-browser"',
            'id="cfg-remote-password"',
            'id="cfg-key-GEMINI_API_KEY"',
            'id="cfg-key-OPENAI_API_KEY"',
            'id="cfg-key-ANTHROPIC_API_KEY"',
        ):
            self.assertIn(
                inp,
                panel,
                f"{inp} must live inside #settings-panel in SorcarTab.ts",
            )

    def test_webapp_settings_panel_contains_config_form(self) -> None:
        panel = _section(_build_html(), "settings-panel")
        self.assertTrue(panel, "#settings-panel missing in webapp HTML")
        for inp in (
            'id="cfg-max-budget"',
            'id="cfg-custom-endpoint"',
            'id="cfg-custom-api-key"',
            'id="cfg-custom-headers"',
            'id="cfg-use-web-browser"',
            'id="cfg-remote-password"',
            'id="cfg-key-GEMINI_API_KEY"',
            'id="cfg-key-OPENAI_API_KEY"',
            'id="cfg-key-ANTHROPIC_API_KEY"',
        ):
            self.assertIn(
                inp,
                panel,
                f"{inp} must live inside #settings-panel in webapp HTML",
            )

    def test_extension_frequent_panel_present(self) -> None:
        panel = _section(_ext_html(), "frequent-panel")
        self.assertTrue(panel, "#frequent-panel missing in SorcarTab.ts")
        self.assertIn('id="frequent-list"', panel)

    def test_webapp_frequent_panel_present(self) -> None:
        panel = _section(_build_html(), "frequent-panel")
        self.assertTrue(panel, "#frequent-panel missing in webapp HTML")
        self.assertIn('id="frequent-list"', panel)

    def test_extension_has_frequent_tasks_btn(self) -> None:
        self.assertIn('id="frequent-tasks-btn"', _ext_html())

    def test_webapp_has_frequent_tasks_btn(self) -> None:
        self.assertIn('id="frequent-tasks-btn"', _build_html())

    def test_webapp_has_overlays(self) -> None:
        html = _build_html()
        self.assertIn('id="sidebar-overlay"', html)
        self.assertIn('id="frequent-overlay"', html)
        self.assertIn('id="settings-overlay"', html)

    def test_extension_has_overlays(self) -> None:
        html = _ext_html()
        self.assertIn('id="sidebar-overlay"', html)
        self.assertIn('id="frequent-overlay"', html)
        self.assertIn('id="settings-overlay"', html)


if __name__ == "__main__":
    unittest.main()

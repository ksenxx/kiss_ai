# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test for the History panel filter bar.

The History sidebar panel renders a single-line filter bar between
the search box and the history list with:

* four category checkboxes — ``Running`` / ``Errored`` / ``Succeeded``
  / ``Favorites``
* two date inputs — ``From`` / ``To``

This test asserts the HTML, CSS, and JS surfaces all line up: the
served page contains the form controls with the expected IDs in
the expected position, the stylesheet defines the filter-bar styles,
and the client-side filter helper plus listener wiring is present in
``main.js``.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from kiss.agents.vscode import web_server


class TestHistoryFilterPanel(unittest.TestCase):
    """Filter bar shows up under the search box and wires up filters."""

    def test_html_contains_filter_controls_below_search(self) -> None:
        """The history panel HTML embeds the filter controls between
        ``<div class="search-wrap">`` and ``<div id="history-list">``."""
        html = web_server._build_html()  # type: ignore[attr-defined]
        # Slice from the search wrap closing tag to the history list
        # opening tag — everything in between is the filter bar.
        m = re.search(
            r'id="history-search".*?</div>'
            r'(?P<bar>.*?)'
            r'<div id="history-list"',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(
            m, "filter bar must sit between search and history-list"
        )
        assert m is not None
        bar = m.group("bar")
        self.assertIn('class="history-filter-bar"', bar)
        for cid in ("hf-running", "hf-errors", "hf-completed"):
            self.assertIn(f'id="{cid}"', bar)
            self.assertIn("checked", bar.split(f'id="{cid}"', 1)[1][:40])
        self.assertIn('id="hf-favorite"', bar)
        for did in ("hf-from", "hf-to"):
            self.assertIn(f'id="{did}"', bar)
            self.assertIn('type="date"', bar)
        for label in ("Running", "Errored", "Succeeded", "Favorites"):
            self.assertIn(label, bar)

    def test_css_styles_filter_bar(self) -> None:
        """Stylesheet has rules for the new filter bar selectors."""
        css = (
            Path(web_server.__file__).parent / "media" / "main.css"
        ).read_text(encoding="utf-8")
        for selector in (
            ".history-filter-bar",
            ".history-filter-chk",
            ".history-filter-date",
        ):
            self.assertIn(selector, css)

    def test_js_wires_filter_visibility(self) -> None:
        """``main.js`` defines the visibility helper and binds change
        listeners on every filter control."""
        js = (
            Path(web_server.__file__).parent / "media" / "main.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function applyHistoryFilterVisibility", js)
        # The renderHistory pass stamps each row with its category
        # and timestamp so the helper can filter without re-running
        # the network round-trip.
        self.assertIn("dataset.category", js)
        self.assertIn("dataset.timestamp", js)
        # Listener wiring covers all filter controls.
        for cid in (
            "hf-running",
            "hf-errors",
            "hf-completed",
            "hf-favorite",
            "hf-from",
            "hf-to",
        ):
            self.assertIn(cid, js)


if __name__ == "__main__":
    unittest.main()

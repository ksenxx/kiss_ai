# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the settings-panel Tips button (remote webapp).

The chat webview template ``media/chat.html`` is shared between the
VS Code extension (``SorcarTab.buildChatHtml``) and the remote webapp
(``web_server._build_html``), so the Tips button must appear in the
page served by both hosts.

Contract locked in here:

* The HTML served by ``web_server._build_html()`` contains a single
  ``#tips-btn`` button inside the settings panel's
  ``.config-update-row``, placed immediately to the LEFT of the
  "Git Commit" button (``#autocommit-btn``).
* The button carries a lightbulb SVG icon, a visible "Tips" label, a
  tooltip and accessible label mentioning tips, and it precedes the
  ``tips.js`` script that wires its click handler, so the handler
  always finds it in the DOM.
"""

from __future__ import annotations

import re
import unittest

from kiss.agents.vscode import web_server


class TestTipsSettingsButtonRemoteHtml(unittest.TestCase):
    """``web_server._build_html`` serves the Tips button in order."""

    def setUp(self) -> None:
        self.html: str = web_server._build_html()  # type: ignore[attr-defined]

    def test_tips_button_directly_left_of_git_commit(self) -> None:
        """``#tips-btn`` sits immediately before ``#autocommit-btn``
        inside the settings ``.config-update-row``."""
        i_row = self.html.find('class="config-update-row"')
        i_tips = self.html.find('id="tips-btn"')
        i_commit = self.html.find('id="autocommit-btn"')
        self.assertGreater(i_row, -1, "settings button row must exist")
        self.assertGreater(i_tips, -1, "Tips button must exist")
        self.assertGreater(i_commit, -1, "Git Commit button must exist")
        self.assertLess(i_row, i_tips, "Tips button must be in the row")
        self.assertLess(i_tips, i_commit, "Tips must come before Git Commit")
        i_commit_open = self.html.rfind("<button", 0, i_commit)
        between = self.html[i_tips:i_commit_open]
        self.assertEqual(
            between.count("<button"),
            0,
            "no other button may sit between Tips and Git Commit",
        )

    def test_tips_button_is_unique(self) -> None:
        """The Tips button id appears exactly once."""
        self.assertEqual(self.html.count('id="tips-btn"'), 1)

    def test_tips_button_markup(self) -> None:
        """The Tips button is a labelled lightbulb SVG button with a
        tips tooltip and accessible label."""
        m = re.search(
            r'<button[^>]*id="tips-btn"[^>]*>(.*?)</button>',
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(m, "Tips button markup must be present")
        assert m is not None
        tag = self.html[m.start() : self.html.index(">", m.start()) + 1]
        self.assertRegex(tag, r'data-tooltip="[^"]*[Tt]ip')
        self.assertRegex(tag, r'aria-label="[^"]*[Tt]ip')
        self.assertRegex(tag, r'class="[^"]*config-update-btn')
        self.assertIn("<svg", m.group(1), "Tips button must carry an SVG")
        self.assertIn(
            "A6 6 0 0 0 6 8",
            m.group(1),
            "Tips button SVG must draw a lightbulb",
        )
        self.assertIn(
            "<span>Tips</span>",
            m.group(1),
            'Tips button must carry the visible label "Tips"',
        )

    def test_tips_button_precedes_tips_script(self) -> None:
        """The button markup appears before the ``tips.js`` script tag
        that wires its click handler."""
        i_btn = self.html.find('id="tips-btn"')
        m = re.search(r'<script[^>]*src="[^"]*tips\.js[^"]*"', self.html)
        self.assertIsNotNone(m, "tips.js must be loaded by the page")
        assert m is not None
        self.assertLess(i_btn, m.start())


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Tips bulb button (Python / remote webapp side).

The chat webview template ``media/chat.html`` is shared between the
VS Code extension (``SorcarTab.buildChatHtml``) and the remote webapp
(``web_server._build_html``), so the bulb button must appear in the
page served by both hosts.

Contract locked in here:

* The HTML served by ``web_server._build_html()`` contains a single
  ``#tips-btn`` button placed strictly between the "Inject promptlet"
  button (``#tricks-btn``) and the mic button (``#voice-btn``).
* The button is an SVG icon button with a tooltip and accessible
  label mentioning tips, and it precedes the ``tips.js`` script that
  wires its click handler, so the handler always finds it in the DOM.
"""

from __future__ import annotations

import re
import unittest

from kiss.agents.vscode import web_server


class TestTipsBulbButtonRemoteHtml(unittest.TestCase):
    """``web_server._build_html`` serves the bulb button in order."""

    def setUp(self) -> None:
        self.html: str = web_server._build_html()  # type: ignore[attr-defined]

    def test_bulb_button_between_promptlet_and_mic(self) -> None:
        """``#tips-btn`` sits strictly between ``#tricks-btn`` and
        ``#voice-btn`` in the served page."""
        i_tricks = self.html.find('id="tricks-btn"')
        i_tips = self.html.find('id="tips-btn"')
        i_voice = self.html.find('id="voice-btn"')
        self.assertGreater(i_tricks, -1, "promptlet button must exist")
        self.assertGreater(i_tips, -1, "bulb button must exist")
        self.assertGreater(i_voice, -1, "mic button must exist")
        self.assertLess(i_tricks, i_tips, "bulb must come after promptlet")
        self.assertLess(i_tips, i_voice, "bulb must come before mic")
        i_voice_open = self.html.rfind("<button", 0, i_voice)
        between = self.html[i_tricks:i_voice_open]
        self.assertEqual(
            between.count("<button"),
            1,
            "the bulb must be the only button between promptlet and mic",
        )

    def test_bulb_button_is_unique(self) -> None:
        """The bulb button id appears exactly once."""
        self.assertEqual(self.html.count('id="tips-btn"'), 1)

    def test_bulb_button_markup(self) -> None:
        """The bulb button is an SVG icon button with a tips tooltip
        and accessible label."""
        m = re.search(
            r'<button id="tips-btn"[^>]*>(.*?)</button>',
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(m, "bulb button markup must be present")
        assert m is not None
        tag = self.html[m.start() : self.html.index(">", m.start()) + 1]
        self.assertRegex(tag, r'data-tooltip="[^"]*[Tt]ip')
        self.assertRegex(tag, r'aria-label="[^"]*[Tt]ip')
        self.assertIn("<svg", m.group(1), "bulb button must carry an SVG")

    def test_bulb_button_precedes_tips_script(self) -> None:
        """The button markup appears before the ``tips.js`` script tag
        that wires its click handler."""
        i_btn = self.html.find('id="tips-btn"')
        m = re.search(r'<script[^>]*src="[^"]*tips\.js[^"]*"', self.html)
        self.assertIsNotNone(m, "tips.js must be loaded by the page")
        assert m is not None
        self.assertLess(i_btn, m.start())


if __name__ == "__main__":
    unittest.main()

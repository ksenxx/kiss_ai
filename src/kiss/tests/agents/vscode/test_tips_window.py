# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Tests that stay behind from ``kiss.tests.agents.vscode.test_tips_window``
(now ``kiss.tests.server.test_tips_window``): they assert on assets outside
kiss.core/kiss.agents.sorcar/kiss.server (e.g. the real
chat.html/main.js media content), so they keep their original
location while the server-only majority of the file moved to
tests/server.
"""


from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from kiss.server import web_server
from kiss.server.tips import read_tips


class TestTipsInRemoteHtml(unittest.TestCase):
    """``web_server._build_html`` wires the tips surface into the page."""

    def test_html_injects_tips_config_and_script(self) -> None:
        """The served page defines ``window.__TIPS__`` and loads
        ``tips.js`` with a content-hash cache-buster; no ``{{TIPS...}}``
        placeholder survives substitution."""
        html = web_server._build_html()  # type: ignore[attr-defined]
        m = re.search(r"window\.__TIPS__\s*=\s*(\{.*?\});</script>", html)
        self.assertIsNotNone(m, "window.__TIPS__ must be defined")
        assert m is not None
        cfg = json.loads(m.group(1).replace("<\\/", "</"))
        self.assertEqual(sorted(cfg), ["show", "tips"])
        self.assertFalse(cfg["show"], "remote webapp never auto-shows tips")
        self.assertEqual(cfg["tips"], read_tips())
        self.assertRegex(html, r'src="/media/tips\.js\?v=[0-9a-f]{16}"')
        self.assertNotIn("{{TIPS", html)

    def test_media_ships_tips_component(self) -> None:
        """``media/tips.js`` defines the ``<kiss-tips-panel>`` web
        component and ``chat.html`` loads it via the placeholders."""
        media = Path(web_server.MEDIA_DIR)
        tips_js = (media / "tips.js").read_text()
        self.assertIn("kiss-tips-panel", tips_js)
        self.assertIn("customElements.define", tips_js)
        chat_html = (media / "chat.html").read_text()
        self.assertIn("{{TIPS_JSON}}", chat_html)
        self.assertIn("{{TIPS_SRC}}", chat_html)


class TestReadTips(unittest.TestCase):
    """``read_tips`` against the real bundled ``src/kiss/TIPS.md``.

    Reads the actual package data file (outside the server layer), so it
    lives here rather than in ``kiss.tests.server.test_tips_window``.
    """

    def test_bundled_tips_md_yields_tips(self) -> None:
        """The bundled ``src/kiss/TIPS.md`` produces non-empty tips."""
        tips = read_tips()
        self.assertGreater(len(tips), 0)
        for tip in tips:
            self.assertTrue(tip.strip())

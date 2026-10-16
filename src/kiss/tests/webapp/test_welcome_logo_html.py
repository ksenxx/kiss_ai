# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The served chat page carries the welcome logos and theme sheets.

``web_server._build_html`` fills ``media/chat.html``: both welcome logos
(light and dark colourings) point at content-versioned ``/media/`` URLs
of transparent RGBA PNGs, the greeting stays under the logo, no
suggested-prompt chips and no remote URL / password block are rendered,
every template placeholder is filled, and both highlight sheets are
handed to ``main.js`` through ``window.__HLJS_THEME_CSS__`` so the code
blocks follow the page theme.  The live page half (logo survives a new
chat, the toggle swaps the sheet) is ``webapp/test/welcomeLogo.test.js``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

from kiss.server import web_server


def _asset_hash(name: str) -> str:
    data = (web_server.MEDIA_DIR / name).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


class _Tags(HTMLParser):
    """Collect ``(tag, attrs)`` for every start tag of the page."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.tags.append((tag, dict(attrs)))


def _by_id(tags: list[tuple[str, dict[str, str | None]]], element_id: str):
    return next((t for t in tags if t[1].get("id") == element_id), None)


class TestWelcomeLogoHtml(unittest.TestCase):
    """The welcome page markup the daemon serves."""

    def setUp(self) -> None:
        self.html = web_server._build_html()
        parser = _Tags()
        parser.feed(self.html)
        self.tags = parser.tags

    def test_both_logos_are_versioned_transparent_pngs(self) -> None:
        for element_id, name in (
            ("welcome-logo", "welcome-logo.png"),
            ("welcome-logo-dark", "welcome-logo-dark.png"),
        ):
            tag = _by_id(self.tags, element_id)
            self.assertIsNotNone(tag, f"the page must show #{element_id}")
            assert tag is not None
            self.assertEqual(tag[0], "img")
            classes = (tag[1].get("class") or "").split()
            self.assertIn("welcome-logo", classes)
            self.assertEqual(
                tag[1].get("alt"), "",
                "the logo is decorative: the heading names the product",
            )
            url = urlparse(tag[1].get("src") or "")
            self.assertEqual(url.path, f"/media/{name}")
            self.assertEqual(
                parse_qs(url.query).get("v"), [_asset_hash(name)],
                "the logo URL carries a content-hash cache-buster",
            )
            data = (web_server.MEDIA_DIR / name).read_bytes()
            self.assertEqual(data[1:4], b"PNG", f"media/{name} is a PNG")
            # Colour type 6 in the IHDR chunk: RGBA, so the art is
            # transparent and sits directly on the theme background.
            self.assertEqual(data[25], 6, f"media/{name} is an RGBA PNG")

    def test_greeting_without_chips_or_config_block(self) -> None:
        heading = re.search(
            r'<div id="welcome".*?<h2[^>]*>(.*?)</h2>', self.html, re.S
        )
        self.assertIsNotNone(heading, "the greeting heading is under the logo")
        assert heading is not None
        self.assertIn("Welcome to", heading.group(1))
        self.assertIsNone(_by_id(self.tags, "suggestions"))
        self.assertIsNone(
            _by_id(self.tags, "welcome-config"),
            "the remote URL / password block is not on the welcome page",
        )
        self.assertNotIn("{{", self.html, "every template placeholder is filled")

    def test_highlight_sheets_follow_the_theme(self) -> None:
        # Light Modern is the page's default theme: the linked sheet is
        # the light one, and both sheets are handed to main.js.
        link = _by_id(self.tags, "hljs-theme")
        self.assertIsNotNone(link)
        assert link is not None
        href = urlparse(link[1].get("href") or "")
        self.assertEqual(href.path, "/media/highlight-light.css")
        shim = re.search(
            r"window\.__HLJS_THEME_CSS__ = (\{.*?\});", self.html
        )
        self.assertIsNotNone(shim, "the page provides window.__HLJS_THEME_CSS__")
        assert shim is not None
        urls = json.loads(shim.group(1))
        for theme in ("dark", "light"):
            parsed = urlparse(urls[theme])
            self.assertEqual(parsed.path, f"/media/highlight-{theme}.css")
            self.assertEqual(
                parse_qs(parsed.query).get("v"),
                [_asset_hash(f"highlight-{theme}.css")],
            )
        self.assertNotIn("highlight-github", self.html, "the github sheets are gone")


if __name__ == "__main__":
    unittest.main()

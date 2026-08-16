# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Tests that stay behind from ``kiss.tests.agents.vscode.test_simplification_lockdown_web_server``
(now ``kiss.tests.server.test_simplification_lockdown_web_server``): they assert on assets outside
kiss.core/kiss.agents.sorcar/kiss.server (e.g. the real
chat.html/main.js media content), so they keep their original
location while the server-only majority of the file moved to
tests/server.
"""


from __future__ import annotations

import re

from kiss.server.web_server import MEDIA_DIR
from kiss.tests.server.test_simplification_lockdown_web_server import (
    _ServerTestBase,
)

_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\}\}")


class TestHttpEndpointMatrix(_ServerTestBase):
    """Lock down the HTTP responses produced by ``_process_request``."""

    async def test_chat_page_html_fully_substituted(self) -> None:
        """GET / serves chat.html with every {{...}} placeholder substituted."""
        status, headers, body = await self._http_get("/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/html; charset=utf-8")
        html = body.decode("utf-8")
        self.assertIsNone(
            _PLACEHOLDER_RE.search(html),
            "served chat page contains unsubstituted template placeholders",
        )
        self.assertIn('id="auth-modal"', html)
        self.assertRegex(
            html,
            r'"/media/main\.js\?v=[0-9a-f]+"',
            "served chat page is missing a cache-busted main.js script tag",
        )
        self.assertIn('class="remote-chat"', html)

    async def test_media_file_served_with_mime_type(self) -> None:
        """GET /media/main.css returns the exact file bytes with a CSS MIME."""
        expected = (MEDIA_DIR / "main.css").read_bytes()
        status, headers, body = await self._http_get("/media/main.css")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/css")
        self.assertEqual(headers["content-length"], str(len(expected)))
        self.assertEqual(body, expected)

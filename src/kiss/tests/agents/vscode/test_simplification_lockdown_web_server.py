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

from kiss.server.web_server import MEDIA_DIR
from kiss.tests.server.test_simplification_lockdown_web_server import (
    _ServerTestBase,
)


class TestHttpEndpointMatrix(_ServerTestBase):
    """Lock down the HTTP responses produced by ``_process_request``."""

    async def test_media_file_served_with_mime_type(self) -> None:
        """GET /media/main.css returns the exact file bytes with a CSS MIME."""
        expected = (MEDIA_DIR / "main.css").read_bytes()
        status, headers, body = await self._http_get("/media/main.css")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/css")
        self.assertEqual(headers["content-length"], str(len(expected)))
        self.assertEqual(body, expected)

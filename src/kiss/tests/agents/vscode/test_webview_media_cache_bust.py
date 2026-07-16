# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for cache-busted remote webview media URLs.

The user-visible regression can persist after source fixes when a
browser or VS Code webview reuses stale ``main.js`` / ``main.css`` from
cache.  These tests pin the remote web server half: generated HTML must
reference content-versioned media URLs, and the HTTP request handler
must serve those URLs even with query strings present.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unittest
from typing import cast

from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request

from kiss.server import web_server
from kiss.server.web_server import RemoteAccessServer


def _asset_hash(name: str) -> str:
    data = (web_server.MEDIA_DIR / name).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _asset_urls(html: str, name: str) -> list[str]:
    urls = re.findall(r'(?:href|src)="([^"]+)"', html)
    return [u for u in urls if f"/media/{name}" in u]


class TestWebviewMediaCacheBust(unittest.TestCase):
    """Generated webview HTML must not depend on stale cached assets."""

    def test_remote_html_uses_content_versioned_media_urls(self) -> None:
        html = web_server._build_html()
        for name in (
            "main.css",
            "highlight-github-dark.min.css",
            "highlight.min.js",
            "marked.min.js",
            "panelCopy.js",
            "main.js",
            "demo.js",
        ):
            urls = _asset_urls(html, name)
            self.assertEqual(urls, [f"/media/{name}?v={_asset_hash(name)}"])

    def test_remote_server_serves_cache_busted_media_urls(self) -> None:
        server = RemoteAccessServer(host="127.0.0.1", port=0)
        path = f"/media/main.js?v={_asset_hash('main.js')}"
        request = Request(path=path, headers=Headers())
        response = asyncio.run(
            server._process_request(cast(ServerConnection, object()), request),
        )
        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.body,
            (web_server.MEDIA_DIR / "main.js").read_bytes(),
        )
        self.assertEqual(
            response.headers["Cache-Control"],
            "no-cache, no-store, must-revalidate",
        )


if __name__ == "__main__":
    unittest.main()

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
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request

from kiss.server import web_server
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import posix_only


def _asset_hash(name: str) -> str:
    data = (web_server.MEDIA_DIR / name).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _asset_urls(html: str, name: str) -> list[str]:
    urls = re.findall(r'(?:href|src)="([^"]+)"', html)
    return [u for u in urls if f"/media/{name}" in u]


def _sw_version(sw_script: str) -> str:
    match = re.search(r'"version": "([0-9a-f]{16})"', sw_script)
    assert match is not None, "service worker script has no manifest version"
    return match.group(1)


class TestWebviewMediaCacheBust(unittest.TestCase):
    """Generated webview HTML must not depend on stale cached assets."""

    def test_remote_html_uses_content_versioned_media_urls(self) -> None:
        html = web_server._build_html()
        for name in (
            "main.css",
            "highlight-vscode-dark.css",
            "highlight.min.js",
            "marked.min.js",
            "panelCopy.js",
            "main.js",
        ):
            urls = _asset_urls(html, name)
            self.assertEqual(urls, [f"/media/{name}?v={_asset_hash(name)}"])

    def test_remote_server_serves_cache_busted_media_urls(self) -> None:
        server = RemoteAccessServer(host="127.0.0.1", port=0)
        path = f"/media/main.js?v={_asset_hash('main.js')}"
        request = Request(path=path, headers=Headers())
        # _process_request fails closed on a missing peer address while
        # the remote password is empty, so present a loopback peer.
        conn = SimpleNamespace(remote_address=("127.0.0.1", 0))
        response = asyncio.run(
            server._process_request(cast(ServerConnection, conn), request),
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


class TestMediaVersionFollowsFileChanges(unittest.TestCase):
    """A running daemon must hand out new ``?v=`` URLs after the media
    files change on disk.

    Regression: ``_media_url`` hashed each asset once per process, so a
    daemon that outlived an in-place upgrade served the fresh
    ``chat.html`` (re-read per request) with the OLD ``main.css`` URL.
    The remote webapp's service worker is cache first for ``/media``,
    so phones rendered the new markup — the "Working directory" bottom
    sheet — with a stylesheet that had no rules for it: the sheet sat
    unstyled below the chat.
    """

    def setUp(self) -> None:
        # A stand-in media directory: every packaged asset symlinked,
        # main.css a real copy this test can edit.
        self.media_dir = Path(tempfile.mkdtemp(prefix="kiss-media-"))
        for entry in web_server.MEDIA_DIR.iterdir():
            if entry.name == "main.css":
                shutil.copyfile(entry, self.media_dir / entry.name)
            else:
                (self.media_dir / entry.name).symlink_to(entry)
        self.real_media_dir = web_server.MEDIA_DIR
        web_server.MEDIA_DIR = self.media_dir
        web_server._MEDIA_VERSION_CACHE.clear()

    def tearDown(self) -> None:
        web_server.MEDIA_DIR = self.real_media_dir
        web_server._MEDIA_VERSION_CACHE.clear()
        shutil.rmtree(self.media_dir, ignore_errors=True)

    def _bump_mtime(self, path: Path) -> None:
        # Coarse-mtime file systems could stamp a same-second rewrite
        # with the old mtime; force a visibly later one.
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    def _wait_for_ctime_tick(self, after_ns: int) -> None:
        # The kernel stamps ctime from a coarse clock (one jiffy, a few
        # ms on Linux): a rewrite in the same tick as the stat that
        # produced *after_ns* would keep the ctime.  Create a scratch
        # file (creation always stamps ctime from the clock) until the
        # clock has moved past it.
        probe = self.media_dir / "ctime-probe"
        while True:
            probe.unlink(missing_ok=True)
            probe.write_bytes(b"")
            if probe.stat().st_ctime_ns > after_ns:
                return
            time.sleep(0.001)

    def _serve(self, url: str) -> bytes:
        server = RemoteAccessServer(host="127.0.0.1", port=0)
        conn = SimpleNamespace(remote_address=("127.0.0.1", 0))
        response = asyncio.run(
            server._process_request(
                cast(ServerConnection, conn), Request(path=url, headers=Headers()),
            ),
        )
        assert response is not None
        self.assertEqual(response.status_code, 200)
        return response.body or b""

    def test_new_css_on_disk_yields_new_url_sw_manifest_and_body(self) -> None:
        css = self.media_dir / "main.css"
        html_before = web_server._build_html()
        sw_before = web_server._build_service_worker()
        (url_before,) = _asset_urls(html_before, "main.css")
        self.assertIn(url_before, sw_before)
        # A repeat call with the file untouched is served from the cache.
        self.assertEqual(web_server._media_url("main.css"), url_before)

        rule = "\n#workdir-panel { outline: 1px solid red; }\n"
        # newline="\n": a text-mode write would turn the rule's LF into
        # CRLF on Windows and the served bytes would not contain it.
        css.write_text(css.read_text(encoding="utf-8") + rule, encoding="utf-8", newline="\n")
        self._bump_mtime(css)

        html_after = web_server._build_html()
        sw_after = web_server._build_service_worker()
        (url_after,) = _asset_urls(html_after, "main.css")
        self.assertNotEqual(url_after, url_before)
        self.assertEqual(url_after, f"/media/main.css?v={_asset_hash('main.css')}")
        self.assertIn(url_after, sw_after)
        self.assertNotIn(url_before, sw_after)
        # The manifest hash — and so the worker's cache name — moved too.
        self.assertNotEqual(_sw_version(sw_before), _sw_version(sw_after))
        self.assertIn(rule.encode("utf-8"), self._serve(url_after))

    @posix_only(
        "st_ctime is the file's creation time on Windows (CPython 3.13), which an "
        "in-place rewrite keeps, so the stat fingerprint cannot see this change there"
    )
    def test_same_size_replacement_with_preserved_mtime_yields_new_url(self) -> None:
        # `cp -p` / archive extraction keep the source's mtime and a
        # same-size edit keeps st_size: the fingerprint must still move
        # (ctime cannot be preserved from userspace).
        css = self.media_dir / "main.css"
        url_before = web_server._media_url("main.css")
        st = css.stat()
        self._wait_for_ctime_tick(st.st_ctime_ns)
        data = bytearray(css.read_bytes())
        data[-1:] = b"X" if data[-1:] != b"X" else b"Y"
        css.write_bytes(bytes(data))
        os.utime(css, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(css.stat().st_mtime_ns, st.st_mtime_ns)
        self.assertEqual(css.stat().st_size, st.st_size)
        url_after = web_server._media_url("main.css")
        self.assertNotEqual(url_after, url_before)
        self.assertEqual(url_after, f"/media/main.css?v={_asset_hash('main.css')}")

    def test_rewrite_with_same_bytes_keeps_the_url(self) -> None:
        css = self.media_dir / "main.css"
        url_before = web_server._media_url("main.css")
        css.write_bytes(css.read_bytes())
        self._bump_mtime(css)
        # The stat changed, so the bytes are re-hashed — to the same value.
        self.assertEqual(web_server._media_url("main.css"), url_before)


if __name__ == "__main__":
    unittest.main()

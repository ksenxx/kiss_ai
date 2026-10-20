# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the webapp identifies itself as the brand's standalone app.

The page links the brand icons (favicon, Apple touch icon) and a web app
manifest; ``/manifest.webmanifest`` names the app after the brand and
asks for ``display: standalone``, so a Chromium app-mode window
(``kiss-app``) and a browser's "install as app" show the product's name
and icon.  The branded case runs the same server code from a Seamless
Loop copy of the package (:mod:`kiss.tests.branded_package`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request

from kiss.core.brand import BRAND, PRODUCT_NAME
from kiss.server import web_server
from kiss.tests.branded_package import BRAND_ICON_PNG, SEAMLESS_LOOP, make_branded_package


def _get(path: str) -> tuple[int, Headers, bytes]:
    """GET *path* from a fresh server as a loopback client; return status, headers, body."""
    server = web_server.RemoteAccessServer(host="127.0.0.1", port=0)
    conn = SimpleNamespace(remote_address=("127.0.0.1", 0))
    response = asyncio.run(
        server._process_request(
            cast(ServerConnection, conn), Request(path=path, headers=Headers())
        ),
    )
    assert response is not None
    return response.status_code, response.headers, bytes(response.body or b"")


def _media_url(name: str) -> str:
    digest = hashlib.sha256((web_server.MEDIA_DIR / name).read_bytes()).hexdigest()[:16]
    return f"/media/{name}?v={digest}"


# The same requests, run from a branded copy of the package in a subprocess.
_BRANDED_PROBE = """
import asyncio, json, sys
from types import SimpleNamespace
from websockets.datastructures import Headers
from websockets.http11 import Request
from kiss.server import web_server

def get(path):
    server = web_server.RemoteAccessServer(host="127.0.0.1", port=0)
    conn = SimpleNamespace(remote_address=("127.0.0.1", 0))
    r = asyncio.run(server._process_request(conn, Request(path=path, headers=Headers())))
    return r.status_code, bytes(r.body or b"")

status, manifest = get("/manifest.webmanifest")
icon_src = json.loads(manifest)["icons"][0]["src"]
icon_status, icon = get(icon_src)
_, html = get("/")
json.dump({
    "status": status, "manifest": json.loads(manifest), "icon_status": icon_status,
    "icon_hex": icon.hex(), "html": html.decode(),
}, sys.stdout)
"""


def test_page_links_brand_icons_and_the_manifest() -> None:
    """The page head carries the cache-busted brand icons and the manifest link."""
    html = web_server._build_html()
    head = html[: html.index("</head>")]
    assert f'<link rel="icon" type="image/svg+xml" href="{_media_url("kiss-icon.svg")}">' in head
    assert (
        f'<link rel="icon" type="image/png" sizes="512x512" href="{_media_url("kiss-icon.png")}">'
        in head
    )
    assert f'<link rel="apple-touch-icon" href="{_media_url("kiss-icon.png")}">' in head
    assert '<link rel="manifest" href="/manifest.webmanifest">' in head
    # The icons are part of the offline app shell the service worker precaches.
    shell = web_server._app_shell_urls(web_server._brand_css_asset_names())
    assert _media_url("kiss-icon.svg") in shell and _media_url("kiss-icon.png") in shell


def test_manifest_names_the_brand_and_asks_for_a_standalone_window() -> None:
    """``/manifest.webmanifest`` is served with the brand's name, tagline and icons."""
    status, headers, body = _get("/manifest.webmanifest")
    assert status == 200
    assert headers["Content-Type"] == "application/manifest+json; charset=utf-8"
    assert headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
    manifest = json.loads(body)
    assert manifest["name"] == PRODUCT_NAME == "KISS Sorcar"
    assert manifest["short_name"] == BRAND["short_name"] == "KISS"
    assert manifest["description"] == BRAND["tagline"]
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/" and manifest["scope"] == "/" and manifest["id"] == "/"
    assert manifest["icons"] == [
        {"src": _media_url("kiss-icon.png"), "sizes": "512x512", "type": "image/png"},
        {"src": _media_url("kiss-icon.svg"), "sizes": "any", "type": "image/svg+xml"},
    ]
    for icon in manifest["icons"]:
        icon_status, icon_headers, icon_body = _get(icon["src"])
        assert icon_status == 200
        assert icon_headers["Content-Type"] == icon["type"]
        assert (
            icon_body == (web_server.MEDIA_DIR / Path(icon["src"].split("?")[0]).name).read_bytes()
        )


def test_branded_install_serves_its_own_name_and_icon(tmp_path: Path) -> None:
    """From a Seamless Loop copy the manifest, page title and icon are the brand's."""
    src = make_branded_package(tmp_path)
    home = str(tmp_path / "home")
    env = {
        "PYTHONPATH": str(src),
        "HOME": home,
        "PATH": "/usr/bin:/bin",
        # Path.home() reads USERPROFILE on Windows, and the interpreter
        # itself needs SYSTEMROOT there (Winsock fails with WinError 10106
        # without it).
        "USERPROFILE": home,
        **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ},
    }
    proc = subprocess.run(
        [sys.executable, "-c", _BRANDED_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    probe = json.loads(proc.stdout)
    assert probe["status"] == 200
    manifest = probe["manifest"]
    assert manifest["name"] == "Seamless Loop"
    assert manifest["short_name"] == "s10s"
    assert manifest["description"] == SEAMLESS_LOOP["tagline"]
    assert manifest["display"] == "standalone"
    icon_src = manifest["icons"][0]["src"]
    assert re.fullmatch(r"/media/kiss-icon\.png\?v=[0-9a-f]{16}", icon_src)
    assert icon_src != _media_url("kiss-icon.png"), (
        "the brand's icon must get its own cache-busted URL"
    )
    assert probe["icon_status"] == 200
    assert bytes.fromhex(probe["icon_hex"]) == BRAND_ICON_PNG
    assert f'href="{icon_src}"' in probe["html"]
    assert "<title>Seamless Loop" in probe["html"]
    assert "KISS Sorcar" not in probe["html"][: probe["html"].index("</head>")]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))

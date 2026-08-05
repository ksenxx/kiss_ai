# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: attaching an iPhone camera photo to the remote webapp.

An iPhone whose *Settings > Camera > Formats* is "High Efficiency" (the
factory default since the iPhone 7) stores photos as HEIC.  The webapp's file
picker uses ``accept="image/*,..."``, and because that wildcard matches
``image/heic``, WebKit deliberately skips its own JPEG transcoding step and
hands the page the raw HEIC bytes (WebKit bug 237219).  The OpenAI and
Anthropic vision APIs reject ``image/heic``, so before the fix the photo was
dropped further down the pipeline and the user saw nothing at all.

``media/main.js`` now re-encodes such a photo to a downscaled JPEG in the
browser.  These tests drive the production page served by the real
``RemoteAccessServer`` in a mobile-sized viewport:

* **WebKit** -- the engine an iPhone actually runs, and the only one that
  decodes HEIC -- must turn the picked HEIC into an ``image/jpeg``
  attachment no wider than 1568px, with the send button left usable.
* **Chromium** cannot decode HEIC at all, and must therefore surface a
  visible error chip instead of silently attaching bytes no model accepts.

The HEIC fixture is produced by macOS's ``sips``; the tests skip where no
HEIF encoder exists.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import subprocess
import threading
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

# main.js downscales to this many pixels on the long edge.
MAX_EDGE = 1568
PHOTO_WIDTH = 2000
PHOTO_HEIGHT = 1200


def _write_gradient_png(path: Path, width: int, height: int) -> None:
    """Write a smooth RGB gradient PNG, encoder-free.

    Args:
        path: Destination file.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            rows += bytes((x * 255 // width, y * 255 // height, 128))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


def _make_heic(tmp_path: Path) -> Path:
    """Return the path of a real HEIC photo, skipping if none can be made."""
    sips = shutil.which("sips")
    if sips is None:
        pytest.skip("no HEIF encoder available to build the fixture")
    png = tmp_path / "IMG_0001.png"
    heic = tmp_path / "IMG_0001.HEIC"
    _write_gradient_png(png, PHOTO_WIDTH, PHOTO_HEIGHT)
    subprocess.run(
        [sips, "-s", "format", "heic", str(png), "--out", str(heic)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    data = heic.read_bytes()
    assert data[4:8] == b"ftyp", "sips did not produce a HEIF container"
    return heic


def _serve(
    tmp_path: Path,
    ready: threading.Event,
    done: threading.Event,
    state: dict[str, object],
) -> None:
    """Run the production RemoteAccessServer until *done* is set."""
    from kiss.server.web_server import (
        RemoteAccessServer,
        _generate_self_signed_cert,
    )

    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)

    async def scenario() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=str(tmp_path),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp_path / "remote-url.json",
            uds_path=tmp_path / "sorcar.sock",
        )
        started = False
        try:
            await server.start_async()
            started = True
            assert server._ws_server is not None
            state["port"] = next(iter(server._ws_server.sockets)).getsockname()[1]
            ready.set()
            while not done.is_set():
                await asyncio.sleep(0.02)
        except BaseException as exc:  # pragma: no cover - defensive
            state["error"] = exc
            ready.set()
        finally:
            if started:
                await server.stop_async()

    asyncio.run(scenario())


@pytest.fixture
def remote_port(tmp_path: Path) -> Iterator[int]:
    """Boot the real remote webapp server and yield its port."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_serve, args=(tmp_path, ready, done, state), daemon=True
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        error = state.get("error")
        if isinstance(error, BaseException):
            raise AssertionError("RemoteAccessServer startup failed") from error
        port = state["port"]
        assert isinstance(port, int)
        yield port
    finally:
        done.set()
        thread.join(30)


def _attach(page: Page, path: Path) -> None:
    """Pick *path* through the webapp's own file picker."""
    with page.expect_file_chooser() as chooser:
        page.click("#upload-btn")
    chooser.value.set_files(str(path))


_READ_CHIP_JS = """
async () => {
  const chip = document.querySelector('#file-chips .file-chip');
  if (!chip) return null;
  const img = chip.querySelector('img');
  const out = {
    cls: chip.className,
    text: chip.textContent,
    src: img ? img.src : '',
    width: 0,
    height: 0,
  };
  if (out.src) {
    const probe = new Image();
    probe.src = out.src;
    await probe.decode();
    out.width = probe.naturalWidth;
    out.height = probe.naturalHeight;
  }
  return out;
}
"""


def _open_mobile_page(browser: Browser, port: int) -> Page:
    """Open the remote chat in an iPhone-sized viewport."""
    page = browser.new_page(
        ignore_https_errors=True,
        viewport={"width": 390, "height": 844},
    )
    page.goto(f"https://127.0.0.1:{port}/", wait_until="domcontentloaded")
    page.wait_for_selector("#upload-btn", state="visible")
    assert page.evaluate("document.body.classList.contains('remote-chat')")
    return page


@pytest.mark.timeout(300)
def test_webkit_attaches_iphone_heic_as_downscaled_jpeg(
    tmp_path: Path, remote_port: int
) -> None:
    """Safari's engine converts a picked HEIC into a JPEG attachment."""
    heic = _make_heic(tmp_path)
    with sync_playwright() as p:
        browser = p.webkit.launch()
        try:
            page = _open_mobile_page(browser, remote_port)
            _attach(page, heic)
            page.wait_for_selector(
                "#file-chips .file-chip:not(.pending)", state="visible"
            )
            chip = page.evaluate(_READ_CHIP_JS)
            send_disabled = page.evaluate(
                "document.getElementById('send-btn').disabled"
            )
            chip_count = page.evaluate(
                "document.querySelectorAll('#file-chips .file-chip').length"
            )
            error_chips = page.evaluate(
                "document.querySelectorAll('#file-chips .file-chip.error').length"
            )
        finally:
            browser.close()

    assert chip is not None, "the photo produced no attachment chip"
    assert "error" not in chip["cls"], f"conversion failed: {chip['text']}"
    assert chip["src"].startswith("data:image/jpeg;base64,"), (
        "the attachment kept a MIME type the vision APIs reject: "
        + chip["src"][:40]
    )
    assert chip["text"].startswith("IMG_0001.jpg"), chip["text"]
    assert max(chip["width"], chip["height"]) == MAX_EDGE
    assert chip["width"] == MAX_EDGE
    assert chip["height"] == round(PHOTO_HEIGHT * MAX_EDGE / PHOTO_WIDTH)
    assert chip_count == 1
    assert error_chips == 0
    assert not send_disabled, "the composer stayed blocked after the conversion"


@pytest.mark.timeout(300)
def test_chromium_reports_a_heic_it_cannot_decode(
    tmp_path: Path, remote_port: int
) -> None:
    """An engine without HEIC support says so instead of failing silently."""
    heic = _make_heic(tmp_path)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-certificate-errors"])
        try:
            page = _open_mobile_page(browser, remote_port)
            _attach(page, heic)
            page.wait_for_selector("#file-chips .file-chip.error", state="visible")
            chip = page.evaluate(_READ_CHIP_JS)
            send_disabled = page.evaluate(
                "document.getElementById('send-btn').disabled"
            )
            chip_count = page.evaluate(
                "document.querySelectorAll('#file-chips .file-chip').length"
            )
        finally:
            browser.close()

    assert chip is not None
    assert "error" in chip["cls"]
    assert "IMG_0001.HEIC" in chip["text"]
    assert chip_count == 1, "the undecodable photo was also attached"
    assert not send_disabled, "the failed conversion left the composer blocked"

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
from playwright.sync_api import Browser, Page, ViewportSize, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from kiss.tests.conftest import goto_retrying_network_change

# main.js downscales to this many pixels on the long edge.
MAX_EDGE = 1568
PHOTO_WIDTH = 2000
PHOTO_HEIGHT = 1200
# main.js re-encodes a supported image above this size (ATTACH_MAX_BYTES).
REENCODE_ABOVE_BYTES = 1500 * 1024
# 4 * 720 * 720 bytes stored: over that size.
DRAWING_EDGE = 720

_MOBILE_VIEWPORT = ViewportSize(width=390, height=844)

WEBKIT_START_SECS = 60
"""How long WebKit gets to start with its first page before the test is skipped by name.

The page's deadline is the launch's (``launch_persistent_context``
opens the two together); a page asked for after ``launch`` has none,
and a WebKit whose web content process never comes up (Playwright's
WebKit 2359 on macOS 26.6.2 starts but that process never gets its
network connection, with or without this product) held the test for
as long as the run allowed.  Like a Docker daemon that is not running,
a browser that opens no page is a capability the machine lacks: the
test skips and names it.
"""


def _write_png(
    path: Path, width: int, height: int, color_type: int, rows: bytes, level: int
) -> None:
    """Write an 8-bit PNG from its unfiltered scanlines, encoder-free.

    Args:
        path: Destination file.
        width: Image width in pixels.
        height: Image height in pixels.
        color_type: The PNG colour type (2: RGB, 6: RGBA).
        rows: The scanlines, each led by its filter byte (0: none).
        level: The zlib level of the image data (0 stores it as is).
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, level))
        + chunk(b"IEND", b"")
    )


def _write_gradient_png(path: Path, width: int, height: int) -> None:
    """Write a smooth RGB gradient PNG.

    Args:
        path: Destination file.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            rows += bytes((x * 255 // width, y * 255 // height, 128))
    _write_png(path, width, height, 2, bytes(rows), 9)


def _write_transparent_drawing_png(path: Path, edge: int) -> None:
    """Write a square RGBA PNG: an opaque black square on full transparency.

    The square fills the middle third.  The image data is stored
    uncompressed, so the file weighs about ``4 * edge * edge`` bytes.

    Args:
        path: Destination file.
        edge: Width and height in pixels.
    """
    clear = bytes(4)
    black = bytes((0, 0, 0, 255))
    third = edge // 3
    blank_row = b"\x00" + clear * edge
    drawn_row = b"\x00" + clear * third + black * third + clear * (edge - 2 * third)
    rows = blank_row * third + drawn_row * third + blank_row * (edge - 2 * third)
    _write_png(path, edge, edge, 6, rows, 0)


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
    from kiss.server.web_server import RemoteAccessServer
    from kiss.tests.local_ws import _generate_self_signed_cert

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
            local_endpoint_file=tmp_path / "sorcar-local.json",
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
    """Pick *path* through the webapp's own file picker.

    The attach button lives inside the composer's "..." overflow menu,
    so the menu is opened first.
    """
    page.click("#more-btn")
    page.wait_for_selector("#more-menu.open", state="visible")
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


_READ_PIXELS_JS = """
async (src) => {
  const probe = new Image();
  probe.src = src;
  await probe.decode();
  const canvas = document.createElement('canvas');
  canvas.width = probe.naturalWidth;
  canvas.height = probe.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(probe, 0, 0);
  const rgb = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data).slice(0, 3);
  return {corner: rgb(8, 8), middle: rgb(canvas.width >> 1, canvas.height >> 1)};
}
"""


def _load_mobile_page(page: Page, port: int) -> Page:
    """Load the remote chat in *page*, an iPhone-sized viewport, and wait for its composer."""
    goto_retrying_network_change(page, f"https://127.0.0.1:{port}/", wait_until="domcontentloaded")
    page.wait_for_selector("#more-btn", state="visible")
    assert page.evaluate("document.body.classList.contains('remote-chat')")
    return page


def _open_mobile_page(browser: Browser, port: int) -> Page:
    """Open the remote chat in a new iPhone-sized page of *browser*."""
    page = browser.new_page(ignore_https_errors=True, viewport=_MOBILE_VIEWPORT)
    return _load_mobile_page(page, port)


@pytest.mark.timeout(300)
def test_webkit_attaches_iphone_heic_as_downscaled_jpeg(
    tmp_path: Path, remote_port: int
) -> None:
    """Safari's engine converts a picked HEIC into a JPEG attachment."""
    heic = _make_heic(tmp_path)
    with sync_playwright() as p:
        if not Path(p.webkit.executable_path).exists():
            pytest.skip("Playwright WebKit is not installed (uv run playwright install webkit)")
        try:
            context = p.webkit.launch_persistent_context(
                tmp_path / "webkit-profile",
                timeout=WEBKIT_START_SECS * 1000,
                ignore_https_errors=True,
                viewport=_MOBILE_VIEWPORT,
            )
        except PlaywrightTimeoutError as exc:
            pytest.skip(
                f"Playwright's WebKit opens no page on this machine within {WEBKIT_START_SECS}s: "
                f"{exc}"
            )
        try:
            page = _load_mobile_page(context.pages[0], remote_port)
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
            context.close()

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


@pytest.mark.timeout(300)
def test_chromium_converts_a_transparent_drawing_onto_white(
    tmp_path: Path, remote_port: int
) -> None:
    """A see-through PNG too big to send as is becomes a JPEG on white.

    JPEG has no transparency.  Laid over the black of an opaque canvas,
    a black drawing on a transparent background came out as one black
    rectangle (audit 2026-11-12, W3).
    """
    drawing = tmp_path / "drawing.png"
    _write_transparent_drawing_png(drawing, DRAWING_EDGE)
    assert drawing.stat().st_size > REENCODE_ABOVE_BYTES
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-certificate-errors"])
        try:
            page = _open_mobile_page(browser, remote_port)
            _attach(page, drawing)
            page.wait_for_selector(
                "#file-chips .file-chip:not(.pending)", state="visible"
            )
            chip = page.evaluate(_READ_CHIP_JS)
            pixels = page.evaluate(_READ_PIXELS_JS, chip["src"])
        finally:
            browser.close()

    assert chip["src"].startswith("data:image/jpeg;base64,"), chip["text"]
    assert min(pixels["corner"]) >= 250, (
        f"the transparent background was encoded as {pixels['corner']}"
    )
    assert max(pixels["middle"]) <= 5, f"the black drawing was encoded as {pixels['middle']}"

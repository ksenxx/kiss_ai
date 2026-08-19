# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: composer buttons are spread out and never overlap.

The composer row below the input textbox (``#input-footer`` in
``media/chat.html``) holds the left button group (``#model-picker``:
burger menu ``#menu-btn``, model pill ``#model-btn``, attach files
``#upload-btn``, inject promptlet ``#tricks-btn``, mic ``#voice-btn``
and share ``#share-btn``) and the right group (``#input-actions``:
send/stop).

Historically ``remote-codex.css`` pulled the remote webapp's 36px
circular controls together with ``margin-left/right: -8px``, so the
adjacent touch targets physically overlapped each other (every inner
edge overlapped by 8-16px) and the burger/attach circles overlapped
the model pill.  In the VS Code webview ``#model-picker`` packed the
buttons with a 1px gap, and ``#input-footer`` drew a ``border-top``
separator line across the composer.

These tests assert the new layout on BOTH surfaces:

* every adjacent pair of composer controls keeps a positive gap
  between their bounding boxes (no overlapping touch targets),
* the remote controls keep their full 36px touch targets and the
  idle remote footer stays on a single row at typical phone widths
  (~390px and up; the model pill is capped and truncates instead of
  pushing buttons around, and narrower screens wrap without overlap),
* in cramped production states — a 320px phone or a 300px sidebar
  while a task runs, when the footer also shows the wait spinner AND
  the stop button next to send — the row may wrap, but no two
  controls ever overlap and the model pill never collapses below its
  72px floor,
* ``#input-footer`` no longer draws a separator line (zero border-top)
  on either surface.

The remote test drives the production ``RemoteAccessServer`` + a real
headless Chromium against the page it actually serves; the VS Code
webview test renders the real ``chat.html`` markup with the real
``main.css`` cascade (no ``remote-codex.css``, exactly like the
extension).  With the old CSS the overlap assertions fail with
negative gaps.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_remote_composer_full_width import (
    _build_test_page,
)

_LEFT_GROUP_IDS = [
    "menu-btn",
    "model-btn",
    "upload-btn",
    "tricks-btn",
    "voice-btn",
    "share-btn",
]
_ALL_IDS = _LEFT_GROUP_IDS + ["send-btn"]
# While a task runs, main.js shows send AND stop plus the wait spinner
# (setRunningState: sendBtn stays flex, stopBtn becomes flex, spinner
# turns active), so the running footer holds two more controls.
_RUNNING_IDS = _ALL_IDS + ["wait-spinner", "stop-btn"]

_ENTER_RUNNING_STATE_JS = """
() => {
  document.getElementById('stop-btn').style.display = 'flex';
  document.getElementById('wait-spinner').classList.add('active');
}
"""

# The served chat.html boots with #app hidden (display:none) and only
# main.js's setServerLoading(false) reveals it — an event driven by
# the daemon connection, which this pure-layout test has no stake in
# (and which occasionally never fires here because no daemon listens
# on the test UDS socket). Reveal the UI exactly the way
# setServerLoading(false) does, so the geometry is deterministic.
_REVEAL_APP_JS = """
() => {
  const overlay = document.getElementById('kiss-server-loading');
  if (overlay) overlay.style.display = 'none';
  const app = document.getElementById('app');
  if (app) app.style.display = '';
}
"""

_FOOTER_LAID_OUT_JS = """
() => {
  const footer = document.getElementById('input-footer');
  return !!footer && footer.getBoundingClientRect().width > 0;
}
"""

_GEOMETRY_JS = """
(ids) => {
  const out = {boxes: {}};
  for (const id of ids) {
    const r = document.getElementById(id).getBoundingClientRect();
    out.boxes[id] = {
      left: r.left, right: r.right, top: r.top, bottom: r.bottom,
      width: r.width, height: r.height,
    };
  }
  const footer = document.getElementById('input-footer');
  out.footerBorderTop =
    getComputedStyle(footer).borderTopWidth;
  return out;
}
"""


def _assert_buttons_spread_out(geometry: dict, min_gap: float) -> None:
    """Assert every adjacent pair of visible composer controls keeps
    at least *min_gap* px between their bounding boxes and that all
    controls sit on one row (equal vertical centers)."""
    boxes = geometry["boxes"]
    for a, b in zip(_LEFT_GROUP_IDS, _LEFT_GROUP_IDS[1:]):
        gap = boxes[b]["left"] - boxes[a]["right"]
        assert gap >= min_gap, (
            f"gap {a} -> {b} is {gap:.2f}px; the buttons below the "
            f"input textbox must be spread out by at least "
            f"{min_gap:.0f}px and must not overlap"
        )
    send_gap = boxes["send-btn"]["left"] - boxes["share-btn"]["right"]
    assert send_gap >= min_gap, (
        f"gap share-btn -> send-btn is {send_gap:.2f}px; the left and "
        "right button groups must not overlap"
    )
    centers = [
        (boxes[i]["top"] + boxes[i]["bottom"]) / 2 for i in _ALL_IDS
    ]
    assert max(centers) - min(centers) <= 1.0, (
        f"composer controls must share one row, got centers {centers}"
    )


def _assert_no_overlap(boxes: dict) -> None:
    """Assert no two composer controls' bounding boxes intersect.

    Used for the cramped layouts (narrow viewport, running state)
    where the row is allowed to wrap: whatever row each control lands
    on, controls must never sit on top of each other.
    """
    ids = list(boxes)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ra, rb = boxes[a], boxes[b]
            separated = (
                ra["right"] <= rb["left"] + 0.01
                or rb["right"] <= ra["left"] + 0.01
                or ra["bottom"] <= rb["top"] + 0.01
                or rb["bottom"] <= ra["top"] + 0.01
            )
            assert separated, (
                f"controls {a} ({ra}) and {b} ({rb}) overlap; composer "
                "buttons must never sit on top of each other"
            )


def _assert_no_separator(geometry: dict) -> None:
    """Assert #input-footer draws no separator line above the buttons."""
    assert geometry["footerBorderTop"] == "0px", (
        "#input-footer must not draw a separator line, got border-top "
        f"width {geometry['footerBorderTop']}"
    )


def _start_live_server(
    tmp_path: Path,
    ready: threading.Event,
    done: threading.Event,
    state: dict[str, object],
) -> None:
    """Run the production RemoteAccessServer until *done* is set.

    Stores the bound ephemeral port in ``state['port']`` (or the
    startup exception in ``state['error']``) and sets *ready*.
    """
    from kiss.server.web_server import (
        RemoteAccessServer,
        _generate_self_signed_cert,
    )

    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)
    # macOS caps AF_UNIX paths at 104 bytes; pytest's tmp_path can
    # exceed that, so the socket gets its own short temp name.
    uds_path = Path(tempfile.gettempdir()) / f"kgap-{uuid.uuid4().hex[:8]}.sock"

    async def scenario() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=str(tmp_path),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp_path / "remote-url.json",
            uds_path=uds_path,
        )
        started = False
        try:
            await server.start_async()
            started = True
            assert server._ws_server is not None
            state["port"] = next(
                iter(server._ws_server.sockets)
            ).getsockname()[1]
            ready.set()
            while not done.is_set():
                await asyncio.sleep(0.02)
        except BaseException as exc:  # pragma: no cover - defensive
            state["error"] = exc
            ready.set()
        finally:
            if started:
                await server.stop_async()
            uds_path.unlink(missing_ok=True)

    asyncio.run(scenario())


@pytest.mark.timeout(180)
def test_remote_composer_buttons_spread_out(tmp_path: Path) -> None:
    """Served page + real Chromium at phone width: the composer
    buttons are spread out with positive gaps (no overlapping touch
    targets), keep their 36px touch targets, stay on a single row, and
    the footer draws no separator line."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server,
        args=(tmp_path, ready, done, state),
        daemon=True,
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError(
                "RemoteAccessServer startup failed"
            ) from startup_error
        port = state["port"]

        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--ignore-certificate-errors"]
            )
            try:
                page = browser.new_page(
                    ignore_https_errors=True,
                    viewport={"width": 420, "height": 900},
                )
                page.goto(
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#input-footer", state="attached")
                page.evaluate(_REVEAL_APP_JS)
                page.wait_for_function(_FOOTER_LAID_OUT_JS, timeout=30000)
                page.wait_for_timeout(300)
                geometry = page.evaluate(_GEOMETRY_JS, _ALL_IDS)
                # Cramped production state: 320px phone while a task
                # runs (spinner active, send AND stop visible).
                page.set_viewport_size({"width": 320, "height": 800})
                page.evaluate(_ENTER_RUNNING_STATE_JS)
                page.wait_for_function(_FOOTER_LAID_OUT_JS, timeout=30000)
                page.wait_for_timeout(200)
                running = page.evaluate(_GEOMETRY_JS, _RUNNING_IDS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"
    thread_error = state.get("error")
    if isinstance(thread_error, BaseException):
        raise AssertionError(
            "RemoteAccessServer thread failed"
        ) from thread_error

    _assert_buttons_spread_out(geometry, min_gap=1.5)
    _assert_no_separator(geometry)
    boxes = geometry["boxes"]
    for bid in (
        "menu-btn",
        "upload-btn",
        "tricks-btn",
        "voice-btn",
        "share-btn",
    ):
        assert boxes[bid]["width"] == 36, boxes
        assert boxes[bid]["height"] == 36, boxes

    _assert_no_overlap(running["boxes"])
    # The pill may truncate its label but must never collapse below
    # its 72px floor (which would clip the 16px model icon); allow
    # only sub-pixel rounding slack.
    assert running["boxes"]["model-btn"]["width"] >= 71.5, running["boxes"]


@pytest.mark.timeout(120)
def test_extension_composer_buttons_spread_out() -> None:
    """VS Code webview surface (real chat.html markup + main.css only):
    the composer buttons are spread out with at least a 4px gap between
    bounding boxes and the footer draws no separator line."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                viewport={"width": 480, "height": 800}
            )
            page = context.new_page()
            html = _build_test_page("")
            # The extension page never loads remote-codex.css; strip it
            # from the synthesised cascade so only main.css applies,
            # exactly like the VS Code webview.
            codex_css = (
                Path(__file__).resolve().parents[4]
                / "kiss"
                / "agents"
                / "vscode"
                / "media"
                / "remote-codex.css"
            ).read_text(encoding="utf-8")
            html = html.replace(f"<style>{codex_css}</style>", "")
            assert codex_css not in html
            page.set_content(html, wait_until="load")
            page.wait_for_selector("#input-footer", state="attached")
            geometry = page.evaluate(_GEOMETRY_JS, _ALL_IDS)

            # Cramped production state: narrow VS Code sidebar while a
            # task runs (spinner active, send AND stop visible). The
            # footer may wrap, but nothing may overlap.
            narrow = browser.new_context(
                viewport={"width": 300, "height": 800}
            )
            narrow_page = narrow.new_page()
            narrow_page.set_content(html, wait_until="load")
            narrow_page.wait_for_selector("#input-footer", state="attached")
            narrow_page.evaluate(_ENTER_RUNNING_STATE_JS)
            narrow_page.wait_for_timeout(100)
            running = narrow_page.evaluate(_GEOMETRY_JS, _RUNNING_IDS)
            narrow.close()
        finally:
            browser.close()

    _assert_buttons_spread_out(geometry, min_gap=4.0)
    _assert_no_separator(geometry)
    _assert_no_overlap(running["boxes"])

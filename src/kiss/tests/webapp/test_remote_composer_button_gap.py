# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: composer buttons are spread out and never overlap.

The composer row below the input textbox (``#input-footer`` in
``media/chat.html``) holds the left button group (``#footer-tools``:
burger menu ``#menu-btn``, new chat ``#new-chat-btn``, inject promptlet
``#tricks-btn`` and the "..." overflow menu ``#more-btn``) and the
right group (``#input-actions``: model button ``#model-btn``, then
send/stop).  Mic, share and attach live inside the "..." menu and are
not part of the idle row.

Historically ``remote-codex.css`` pulled the remote webapp's 36px
circular controls together with ``margin-left/right: -8px``, so the
adjacent touch targets physically overlapped each other (every inner
edge overlapped by 8-16px) and the burger/attach circles overlapped
the model button, and ``#input-footer`` drew a ``border-top`` separator
line across the composer.

This test asserts the new layout:

* every adjacent pair of composer controls keeps a positive gap
  between their bounding boxes (no overlapping touch targets),
* the remote controls keep their full 36px touch targets and the
  idle remote footer stays on a single row at typical phone widths
  (~390px and up; the model button is capped and truncates instead of
  pushing buttons around, and narrower screens wrap without overlap),
* in the cramped production state — a 320px phone while a task runs,
  when the footer also shows the wait spinner AND the stop button
  next to send — the row may wrap, but no two controls ever overlap
  and the model button never collapses below its 72px floor,
* ``#input-footer`` no longer draws a separator line (zero border-top),
* the new-chat button draws a chat bubble with a "+" inside it, at
  the bubble's top right.

The test drives the production ``RemoteAccessServer`` + a real
headless Chromium against the page it actually serves.  With the old
CSS the overlap assertions fail with negative gaps.
"""

from __future__ import annotations

import asyncio
import threading
from operator import itemgetter
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.conftest import goto_retrying_network_change

_LEFT_GROUP_IDS = [
    "menu-btn",
    "new-chat-btn",
    "tricks-btn",
    "more-btn",
]
_ALL_IDS = _LEFT_GROUP_IDS + ["model-btn", "send-btn"]
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
# (and which occasionally never fires here because the page is loaded
# without authenticating). Reveal the UI exactly the way
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


# The new-chat button's icon, in the units of its 24x24 drawing: the
# box of the bubble's outline (its first path), the boxes of the other
# strokes, and whether each stroke's two ends lie inside the outline.
_NEW_CHAT_ICON_JS = """
() => {
  const svg = document.querySelector('#new-chat-btn svg');
  const [bubble, ...strokes] = svg.querySelectorAll('path');
  const box = (el) => {
    const b = el.getBBox();
    return {x: b.x, y: b.y, width: b.width, height: b.height};
  };
  const ends = (el) => [el.getPointAtLength(0), el.getPointAtLength(el.getTotalLength())];
  const drawn = svg.getBoundingClientRect();
  return {
    drawn: [drawn.width, drawn.height],
    bubble: box(bubble),
    strokes: strokes.map(box),
    endsInBubble: strokes.flatMap(ends).map((p) => bubble.isPointInFill(p)),
    outsideBubble: bubble.isPointInFill(new DOMPoint(23, 23)),
  };
}
"""


def _assert_new_chat_icon(icon: dict) -> None:
    """Assert the icon is a chat bubble with a "+" inside it, at the
    bubble's top right.

    Args:
        icon: The ``_NEW_CHAT_ICON_JS`` probe result.
    """
    assert icon["drawn"] == [16, 16], icon
    bubble = icon["bubble"]
    assert bubble["width"] >= 18 and bubble["height"] >= 18, (
        f"the bubble fills the 24x24 drawing: {icon}"
    )
    # The "+": an upright and a level stroke of one length that cross
    # at their middles.
    assert len(icon["strokes"]) == 2, icon
    upright, level = sorted(icon["strokes"], key=itemgetter("width"))
    assert upright["width"] == 0 and level["height"] == 0, icon
    assert upright["height"] == level["width"] > 0, icon
    centre_x = upright["x"]
    centre_y = level["y"]
    assert centre_x == level["x"] + level["width"] / 2, icon
    assert centre_y == upright["y"] + upright["height"] / 2, icon
    # Inside the bubble, in its top right quarter.
    assert icon["endsInBubble"] == [True] * 4, icon
    assert icon["outsideBubble"] is False, icon
    assert centre_x > bubble["x"] + bubble["width"] / 2, icon
    assert centre_y < bubble["y"] + bubble["height"] / 2, icon


def _assert_buttons_spread_out(geometry: dict, min_gap: float) -> None:
    """Assert every adjacent pair of visible composer controls keeps
    at least *min_gap* px between their bounding boxes and that all
    controls sit on one row (equal vertical centers).

    Args:
        geometry: The ``_GEOMETRY_JS`` probe result.
        min_gap: Minimum horizontal gap between adjacent boxes.
    """
    boxes = geometry["boxes"]
    for a, b in zip(_LEFT_GROUP_IDS, _LEFT_GROUP_IDS[1:]):
        gap = boxes[b]["left"] - boxes[a]["right"]
        assert gap >= min_gap, (
            f"gap {a} -> {b} is {gap:.2f}px; the buttons below the "
            f"input textbox must be spread out by at least "
            f"{min_gap:.0f}px and must not overlap"
        )
    model_gap = boxes["model-btn"]["left"] - boxes["more-btn"]["right"]
    assert model_gap >= min_gap, (
        f"gap more-btn -> model-btn is {model_gap:.2f}px; the left and "
        "right button groups must not overlap"
    )
    send_gap = boxes["send-btn"]["left"] - boxes["model-btn"]["right"]
    assert send_gap >= min_gap, (
        f"gap model-btn -> send-btn is {send_gap:.2f}px; the model button "
        "and the send button must not overlap"
    )
    centers = [
        (boxes[i]["top"] + boxes[i]["bottom"]) / 2
        for i in _ALL_IDS
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

    asyncio.run(scenario())


@pytest.mark.timeout(180)
def test_remote_composer_buttons_spread_out(tmp_path: Path) -> None:
    """Served page + real Chromium at phone width: the composer
    buttons are spread out with positive gaps (no overlapping touch
    targets), keep their 36px touch targets, stay on a single row, the
    footer draws no separator line, and the new-chat button shows a
    chat bubble with a "+" inside it."""
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
                goto_retrying_network_change(
                    page,
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#input-footer", state="attached")
                page.evaluate(_REVEAL_APP_JS)
                page.wait_for_function(_FOOTER_LAID_OUT_JS, timeout=30000)
                page.wait_for_timeout(300)
                geometry = page.evaluate(_GEOMETRY_JS, _ALL_IDS)
                new_chat_icon = page.evaluate(_NEW_CHAT_ICON_JS)
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
    _assert_new_chat_icon(new_chat_icon)
    boxes = geometry["boxes"]
    for bid in (
        "menu-btn",
        "new-chat-btn",
        "tricks-btn",
        "more-btn",
    ):
        assert boxes[bid]["width"] == 36, boxes
        assert boxes[bid]["height"] == 36, boxes

    _assert_no_overlap(running["boxes"])
    # The model button may truncate its label but must never collapse
    # below its 72px floor (which would clip the 16px model icon);
    # allow only sub-pixel rounding slack.
    assert running["boxes"]["model-btn"]["width"] >= 71.5, running["boxes"]

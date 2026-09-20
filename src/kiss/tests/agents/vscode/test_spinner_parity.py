# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Every running-task spinner is the composer's wait spinner.

Three surfaces show a "task is running" ring:

* the composer's ``#wait-spinner`` under the chat textbox;
* the ``.status-spinner`` at the left of every running row in the
  history panel and in the chat tab strip (sidebar webview, editor-tab
  webview and the kiss-web page all load the same ``main.css``);
* the editor-tab ICON of a running chat in the extension's editor-tabs
  mode (``media/spinner-running.svg``, set as ``WebviewPanel.iconPath``
  by ``SorcarPanelManager``).

The first two are one CSS rule, so the tests compare their *computed*
styles in a real Chromium: size, ring thickness, one bright leading arc
over a faint track, and the very same keyframe animation.  The editor
tab paints its icon as a CSS background image, where only an animation
inside the SVG itself can run, so the last test renders the SVG that
way and checks that it really turns and really is green.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_history_failed_red_cross import (
    _open_history_page,
    _post_history,
    _sample_sessions,
)

_MEDIA = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
_SPINNER_SVG = _MEDIA / "spinner-running.svg"


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


# The ring properties that must agree between the surfaces.  Colours are
# deliberately left out: each surface picks its own via `color`.
_RING_PROBE = """
(sel) => {
  const el = document.querySelector(sel);
  if (!el) return {error: 'no element ' + sel};
  const cs = getComputedStyle(el);
  const track = cs.borderBottomColor;
  return {
    width: cs.width,
    height: cs.height,
    borderRadius: cs.borderRadius,
    borderTopWidth: cs.borderTopWidth,
    borderRightWidth: cs.borderRightWidth,
    borderBottomWidth: cs.borderBottomWidth,
    borderLeftWidth: cs.borderLeftWidth,
    // A single leading arc: the top edge is bright, the other three
    // edges are the faint track.
    singleArc: cs.borderTopColor !== track &&
      cs.borderRightColor === track && cs.borderLeftColor === track,
    animationName: cs.animationName,
    animationDuration: cs.animationDuration,
    animationTimingFunction: cs.animationTimingFunction,
    animationIterationCount: cs.animationIterationCount,
    animationPlayState: cs.animationPlayState,
    opacity: cs.opacity,
    visible: el.offsetWidth > 0 && el.offsetHeight > 0,
  };
}
"""


def _ring(page, selector: str) -> dict:
    info = dict(page.evaluate(_RING_PROBE, selector))
    assert "error" not in info, info
    return info


def _wait_for_composer_spinner(page) -> None:
    """Wait until ``#wait-spinner`` is active (main.js shows it 250ms
    into a run) and has faded in (a 0.2s opacity transition)."""
    page.wait_for_function(
        "(el => el && getComputedStyle(el).opacity === '1')"
        "(document.querySelector('#wait-spinner.active'))",
        timeout=5000,
    )


def _assert_same_ring(a: dict, b: dict) -> None:
    for key in (
        "width", "height", "borderRadius",
        "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
        "animationName", "animationDuration", "animationTimingFunction",
        "animationIterationCount",
    ):
        assert a[key] == b[key], f"{key}: {a[key]!r} != {b[key]!r}\n{a}\n{b}"
    assert a["singleArc"] and b["singleArc"], (a, b)
    assert a["animationName"] not in ("none", ""), a
    assert a["animationIterationCount"] == "infinite", a


def test_history_row_spinner_is_the_composer_wait_spinner(_browser) -> None:
    """The running row's ring and the active ``#wait-spinner`` share
    every geometric and animation property and both spin."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        # The composer's spinner turns while the chat's task runs.
        page.evaluate(
            "() => window.__post({type: 'status', running: true,"
            " tabId: window._testApi.getActiveTabId(), startTs: 1700000000000,"
            " taskId: 'task-x'})"
        )
        _wait_for_composer_spinner(page)
        wait = _ring(page, "#wait-spinner.active")
        row = _ring(page, ".sidebar-item.running-item > .sidebar-item-running.status-spinner")
        _assert_same_ring(wait, row)
        assert wait["width"] == "12px" and wait["height"] == "12px", wait
        assert wait["borderTopWidth"] == "2px", wait
        assert wait["animationDuration"] == "0.8s", wait
        assert wait["animationTimingFunction"] == "linear", wait
        assert wait["animationPlayState"] == "running", wait
        assert row["animationPlayState"] == "running", row
        assert wait["opacity"] == "1" and wait["visible"], wait
        assert row["visible"], row
    finally:
        context.close()


def test_tab_strip_spinner_is_the_composer_wait_spinner(_browser) -> None:
    """The chat tab strip's running icon is the same ring too."""
    context, page = _open_history_page(_browser)
    try:
        page.evaluate(
            "() => window.__post({type: 'status', running: true,"
            " tabId: window._testApi.getActiveTabId(), startTs: 1700000000000,"
            " taskId: 'task-y'})"
        )
        _wait_for_composer_spinner(page)
        _assert_same_ring(
            _ring(page, "#wait-spinner.active"),
            _ring(page, ".chat-tab-spinner.status-spinner"),
        )
    finally:
        context.close()


def _screenshot_pixels(page, selector: str) -> Image.Image:
    png = page.locator(selector).screenshot()
    return Image.open(io.BytesIO(png)).convert("RGB")


def _green_pixel_coords(img: Image.Image) -> list[tuple[int, int]]:
    """(x, y) of every pixel that reads as green on the dark background."""
    data = img.tobytes()
    w = img.size[0]
    return [
        (i % w, i // w)
        for i in range(len(data) // 3)
        if data[3 * i + 1] > data[3 * i] + 40 and data[3 * i + 1] > data[3 * i + 2] + 40
    ]


def test_editor_tab_icon_is_a_green_ring_that_turns_as_a_background_image(
    _browser,
) -> None:
    """Painted the way the workbench paints a tab icon (a 16x16 CSS
    ``background-image``), the SVG must be visibly green and must keep
    rotating: two frames a fraction of a period apart differ, and the
    green pixels sit on the same ring the composer's spinner draws."""
    assert _SPINNER_SVG.is_file(), _SPINNER_SVG
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "tab.html"
        # The icon box the workbench gives a tab icon, scaled 8x so a
        # 2px stroke is measurable; `background-size: contain` as in
        # vscode's IconLabel.
        html.write_text(
            "<!DOCTYPE html><html><body style='margin:0;background:#1e1e1e'>"
            "<div id='icon' style='width:128px;height:128px;"
            f"background:url({_SPINNER_SVG.as_uri()}) center / contain no-repeat'>"
            "</div></body></html>",
            encoding="utf-8",
        )
        context = _browser.new_context(viewport={"width": 200, "height": 200})
        try:
            page = context.new_page()
            page.goto(html.as_uri(), wait_until="load")
            page.wait_for_timeout(100)
            first = _screenshot_pixels(page, "#icon")
            page.wait_for_timeout(300)  # 3/8 of a turn later
            second = _screenshot_pixels(page, "#icon")
        finally:
            context.close()

    green = _green_pixel_coords(first)
    assert len(green) > 200, f"the tab spinner is not green: {len(green)} green pixels"
    assert first.tobytes() != second.tobytes(), (
        "the tab spinner did not move between two frames 300ms apart: "
        "the SVG must rotate with SMIL, which runs inside a CSS background image"
    )
    # The bright arc lies on a ring of 12/16 of the box: no green pixel
    # may fall in the centre (inside r=4/16) or outside the box's ring.
    w, h = first.size
    cx, cy = w / 2, h / 2
    inner = (w / 16) * 3.5
    outer = (w / 16) * 6.5
    for img in (first, second):
        for x, y in _green_pixel_coords(img):
            d = ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) ** 0.5
            assert inner <= d <= outer, (
                f"green pixel at ({x},{y}) is off the 2px ring: d={d:.1f}"
            )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real Chromium: scrolling a rendered preview's whole page sideways is not a swipe.

On the phone layout a rendered .md / .html file is a sandboxed iframe
that carries the swipe detector of ``media/mobileSwipe.js``
(``MobileSwipe.bootstrapHtml()``, appended by main.js's
``withMobileSwipe``); the detector posts every swipe to the parent as
``{kissMobileSwipe: 'left' | 'right'}`` and the parent changes screen.
A preview wider than the phone (a wide table, a fixed-width page)
scrolls sideways through its viewport, the document's scrolling
element, whose ``overflow-x`` is ``visible``.  The detector used to
count only ``overflow-x: auto | scroll`` boxes as sideways scrollers,
so dragging such a page posted a swipe and the screen changed under
the reader's finger.

The parent page holds just mobileSwipe.js and the preview iframe
(``sandbox="allow-scripts"``, ``srcdoc`` = the preview + the
bootstrap, as main.js builds it).  Fingers are CDP touch events, so
Chromium scrolls the preview natively, as under a real finger.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Frame, Page, sync_playwright

_SWIPE_JS = (
    Path(__file__).resolve().parents[2] / "webapp" / "media" / "mobileSwipe.js"
).read_text(encoding="utf-8")

# mobileSwipe.js ignores a finger that was down longer than this.
_SWIPE_MAX_MS = 1500

# The parent: nothing in it can scroll, so a drag that reaches it
# (scroll chaining at the preview's edge) changes no scroll offset.
_PARENT_HTML = (
    "<!DOCTYPE html><html><head><style>"
    "html, body { margin: 0; overflow: hidden; }"
    "iframe { display: block; border: 0; width: 100vw; height: 100vh; }"
    "</style></head><body></body></html>"
)

_WIDE_BODY = (
    "<html><body style='margin: 0'>"
    "<table style='width: 1600px; height: 900px'><tr><td>wide table</td></tr></table>"
    "</body></html>"
)
# A standards-mode page scrolls through <html>, a quirks-mode one (no
# doctype, as many hand-written .html files) through <body>.
_WIDE_PREVIEWS = pytest.mark.parametrize(
    "wide", ["<!DOCTYPE html>" + _WIDE_BODY, _WIDE_BODY], ids=["standards", "quirks"],
)

_FITTING_PREVIEW = (
    "<!DOCTYPE html><html><body style='margin: 0'><p>fits the phone</p></body></html>"
)

# Mounts the preview as main.js does and records what it posts.
_MOUNT_JS = """html => new Promise(resolve => {
  window.__swipes = [];
  window.addEventListener('message', e => {
    if (!e.data) return;
    if (e.data.kissMobileSwipe) window.__swipes.push(e.data.kissMobileSwipe);
    if (e.data.marker) window.__marked = true;
  });
  const frame = document.createElement('iframe');
  frame.setAttribute('sandbox', 'allow-scripts');
  frame.onload = () => resolve();
  frame.srcdoc = html + MobileSwipe.bootstrapHtml();
  document.body.appendChild(frame);
})"""

# How long the preview saw the last finger down (its own clock).
_FINGER_CLOCK_JS = """() => {
  window.__fingerDownMs = null;
  let at = 0;
  const opts = {capture: true, passive: true};
  window.addEventListener('touchstart', () => { at = Date.now(); }, opts);
  document.addEventListener('touchend', () => {
    window.__fingerDownMs = Date.now() - at;
  }, opts);
}"""


@pytest.fixture(scope="module")
def phone() -> Iterator[Page]:
    """A phone-sized page in one headless Chromium for the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 700})
        yield page
        browser.close()


def _preview(page: Page, html: str) -> Frame:
    """Show *html* as a rendered preview and return its frame."""
    page.set_content(_PARENT_HTML)
    page.add_script_tag(content=_SWIPE_JS)
    page.evaluate(_MOUNT_JS, html)
    frame = page.frames[1]
    frame.evaluate(_FINGER_CLOCK_JS)
    return frame


def _drag_left(page: Page, frame: Frame) -> int:
    """One finger from x=330 to x=60 across the preview; return how
    long the preview saw it down (ms)."""
    cdp = page.context.new_cdp_session(page)
    cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": True})
    frame.evaluate("window.__fingerDownMs = null")
    cdp.send("Input.dispatchTouchEvent",
             {"type": "touchStart", "touchPoints": [{"x": 330, "y": 350}]})
    for x in (285, 240, 195, 150, 105, 60):
        cdp.send("Input.dispatchTouchEvent",
                 {"type": "touchMove", "touchPoints": [{"x": x, "y": 350}]})
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    cdp.detach()
    frame.wait_for_function("window.__fingerDownMs !== null")
    return int(frame.evaluate("window.__fingerDownMs"))


def _posted(page: Page, frame: Frame) -> list[str]:
    """The swipes the preview posted.  The preview then posts a marker:
    one window's messages to another arrive in order, so once the
    marker is in, so is every swipe posted before it."""
    page.evaluate("window.__marked = false")
    frame.evaluate("parent.postMessage({marker: true}, '*')")
    page.wait_for_function("window.__marked")
    return list(page.evaluate("window.__swipes"))


def _swift_drag_left(page: Page, html: str, scroll_to: str = "0") -> Frame:
    """Show *html* scrolled to x=*scroll_to* and drag it left with a
    finger the detector times as a swipe's; every touch event is a CDP
    round trip, so on a loaded machine a finger can take longer than a
    swipe may, and such a drag is made again on a fresh preview."""
    down = 0
    for _ in range(5):
        frame = _preview(page, html)
        frame.evaluate(f"scrollTo({scroll_to}, 0)")
        down = _drag_left(page, frame)
        if down < _SWIPE_MAX_MS:
            return frame
    raise AssertionError(f"the finger took {down} ms, never a swipe's time")


@_WIDE_PREVIEWS
def test_dragging_a_wide_preview_scrolls_it_and_posts_no_swipe(phone: Page, wide: str) -> None:
    frame = _swift_drag_left(phone, wide)
    assert frame.evaluate("scrollX") > 0, "Chromium scrolled the preview sideways"
    assert _posted(phone, frame) == []


def test_dragging_a_preview_that_fits_posts_the_swipe(phone: Page) -> None:
    frame = _swift_drag_left(phone, _FITTING_PREVIEW)
    assert _posted(phone, frame) == ["left"]


@pytest.mark.parametrize("overflow", ["hidden", "clip"])
def test_a_wide_preview_that_hides_its_overflow_still_swipes(phone: Page, overflow: str) -> None:
    """A page whose root clips its sideways overflow cannot be dragged
    sideways (the phone layout's own page is such a page), so the drag
    is a swipe."""
    wide = f"<!DOCTYPE html>{_WIDE_BODY}".replace(
        "<html>", f"<html style='overflow-x: {overflow}'>",
    )
    frame = _swift_drag_left(phone, wide)
    assert frame.evaluate("document.scrollingElement.scrollWidth") > 390
    assert frame.evaluate("scrollX") == 0
    assert _posted(phone, frame) == ["left"]


@_WIDE_PREVIEWS
def test_a_wide_preview_at_its_right_edge_still_swipes_left(phone: Page, wide: str) -> None:
    """An unscrolled scroller still swipes: the page cannot scroll
    further right, so the drag leaves its offset alone."""
    frame = _swift_drag_left(phone, wide, "document.scrollingElement.scrollWidth")
    right_edge = frame.evaluate(
        "document.scrollingElement.scrollWidth - document.scrollingElement.clientWidth",
    )
    assert right_edge > 0
    assert frame.evaluate("scrollX") == right_edge
    assert _posted(phone, frame) == ["left"]

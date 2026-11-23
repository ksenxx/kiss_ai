# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and reused as parameters)
"""E2E tests: the top of a phone's screen, around the camera.

An iPhone (iOS 26 and later) colours the band of its screen around the
camera itself, and blurs what lies under it, unless the element on top
at ONE point of the page (the middle of the top edge, 4px down) is
pinned to the window (``position: fixed``), at least 90% of the window
wide and more than 10px tall: the band is then that element's
``background-color``, solid.  A drawer or the scrim behind it used to
be that element while one was open, so the band took their colour, kept
it after they closed, and was a blur until the first one opened.

``#statusbar-tint`` (``media/chat.html``, ``media/remote-codex.css``) is
that element in every state of the page, in the chat's background
colour.  Chromium draws no such band, so these tests check what the
phone reads: the element on top at the point it looks at, that
element's size and its colour in every palette, with a toast, a menu, a
drawer and a sheet on screen and before any script has run; and that the
strip paints nothing and takes no tap.  Android colours its status bar
from ``<meta name="theme-color">``, which ``applyRemoteTheme``
(``media/main.js``) keeps on the chat's colour.

A REAL browser (Playwright Chromium) drives a REAL
:class:`RemoteAccessServer` as a phone.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_editing import _dismiss_toasts
from kiss.tests.webapp.test_mobile_swipe_hint import PHONE_UA, _open_phone_page
from kiss.tests.webapp.test_mobile_swipe_screens import (
    _swipe_left,
    _swipe_right,
    _touch_session,
    _wait_screen,
)
from kiss.tests.webapp.test_workspace_sections import (
    browser,  # noqa: F401  (module fixture used by param name)
)

# What the phone reads at the top of the page.  It finds the element on
# top at the point whether or not that element takes taps, so the probe
# makes every element take them while it looks (`onTop`), then looks
# again as a finger would (`tapped`).  `plain` is what must not be set
# on the strip or around it: a transform, filter, perspective or
# containment of an ancestor would pin the strip to that ancestor, not
# to the window, and the phone skips an element that is see-through.
_SAMPLE_JS = """() => {
  const strip = document.getElementById('statusbar-tint');
  const style = getComputedStyle(strip);
  const box = strip.getBoundingClientRect();
  const taps = document.createElement('style');
  taps.textContent = '* { pointer-events: auto !important; }';
  document.head.appendChild(taps);
  const onTop = document.elementFromPoint(innerWidth / 2, 4);
  taps.remove();
  const tapped = document.elementFromPoint(innerWidth / 2, 4);
  const plain = el => {
    const cs = getComputedStyle(el);
    return [cs.transform, cs.filter, cs.perspective, cs.contain, cs.opacity, cs.visibility];
  };
  return {
    onTop: onTop.id,
    tapped: tapped.id,
    parent: strip.parentElement.tagName,
    box: [box.left, box.top, box.width, box.height],
    window: innerWidth,
    position: style.position,
    display: style.display,
    colour: style.backgroundColor,
    image: style.backgroundImage,
    backdrop: style.backdropFilter,
    clip: style.backgroundClip,
    taps: style.pointerEvents,
    plain: [strip, document.body, document.documentElement].map(plain),
    chat: getComputedStyle(document.body).backgroundColor,
    themeColor: document.querySelector('meta[name="theme-color"]').content,
  };
}"""
_PLAIN = ["none", "none", "none", "none", "1", "visible"]
_LIGHT = "rgb(250, 250, 250)"
_SET_STRIP_VISIBILITY_JS = (
    "value => { document.getElementById('statusbar-tint').style.visibility = value; }"
)


def _sampled(page) -> dict:
    """What the phone reads at the top of the page, having checked that
    it is the strip: on top at the point, pinned to the window along
    its whole top edge, 12px tall, in the chat's background colour,
    opaque, and neither painting over the page nor taking a tap."""
    got: dict = page.evaluate(_SAMPLE_JS)
    assert got["onTop"] == "statusbar-tint", got
    assert got["parent"] == "BODY" and got["position"] == "fixed", got
    assert got["display"] == "block" and got["plain"] == [_PLAIN] * 3, got
    assert got["box"] == [0, 0, got["window"], 12], got
    assert got["colour"] == got["chat"], "the chat's background colour"
    assert got["colour"].startswith("rgb("), "an opaque colour"
    assert got["image"] == "none" and got["backdrop"] == "none", got
    assert got["clip"] == "text", "the colour is clipped to a text the strip does not have"
    assert got["taps"] == "none" and got["tapped"] != "statusbar-tint", got
    return got


def _top_band(page: Page) -> bytes:
    """The pixels of the 12px along the top of the page."""
    return page.screenshot(clip={"x": 0, "y": 0, "width": 390, "height": 12})


def test_strip_is_on_top_at_the_sampled_point_whatever_is_on_screen(browser, harness):
    context, page, _frames = _open_phone_page(browser, harness, PHONE_UA)
    try:
        # The chat, under the notice about swiping (a toast at the top).
        expect(page.locator(".kiss-notification")).to_be_visible(timeout=30000)
        assert _sampled(page)["colour"] == _LIGHT
        _dismiss_toasts(page)
        _sampled(page)
        page.tap("#more-btn")
        expect(page.locator("#more-menu")).to_be_visible()
        _sampled(page)
        page.keyboard.press("Escape")

        # The Chats drawer is another colour than the chat, and it lies
        # under the point: the strip still is what the phone reads, and
        # the drawer looks the same with the strip as without it.
        cdp = _touch_session(page)
        _swipe_right(cdp)
        _wait_screen(page, "chats")
        assert _sampled(page)["colour"] == _LIGHT
        drawer = page.evaluate(
            "getComputedStyle(document.getElementById('sidebar')).backgroundColor",
        )
        assert drawer != _LIGHT
        with_strip = _top_band(page)
        page.evaluate(_SET_STRIP_VISIBILITY_JS, "hidden")
        assert _top_band(page) == with_strip, "the strip paints over the page"
        page.evaluate(_SET_STRIP_VISIBILITY_JS, "")

        # The task-info drawer, then the sheet its Settings button opens,
        # over its scrim.  The last swipe rests before it lifts: a tap
        # follows it.
        _swipe_left(cdp)
        _wait_screen(page, "chat")
        _swipe_left(cdp, rest=True)
        _wait_screen(page, "info")
        _sampled(page)
        page.tap("#settings-btn")
        _wait_screen(page, "chat")
        expect(page.locator("#settings-panel")).to_be_visible()
        _sampled(page)
        page.tap("#settings-panel-close")
        expect(page.locator("#settings-panel")).to_be_hidden()
    finally:
        context.close()


def test_strip_and_theme_color_follow_the_palette(browser, harness):
    context, page, _frames = _open_phone_page(browser, harness, PHONE_UA)
    try:
        _dismiss_toasts(page)
        # The swipe rests before it lifts: taps follow it.
        _swipe_right(_touch_session(page), rest=True)
        _wait_screen(page, "chats")
        seen = [_sampled(page)]
        for button in ("#theme-btn", "#contrast-btn", "#theme-btn", "#contrast-btn"):
            page.tap(button)
            seen.append(_sampled(page))
        assert [palette["colour"] for palette in seen] == [
            _LIGHT,
            "rgb(11, 11, 11)",  # dark
            "rgb(0, 0, 0)",  # dark, high contrast
            "rgb(255, 255, 255)",  # light, high contrast
            _LIGHT,
        ]
        assert [palette["themeColor"] for palette in seen] == [
            palette["colour"] for palette in seen
        ]
    finally:
        context.close()


def test_strip_is_there_before_any_script_runs(browser, harness):
    """The first frame a phone draws has the strip on top already, over
    the "server is starting" screen."""
    context = browser.new_context(
        ignore_https_errors=True, java_script_enabled=False,
        viewport={"width": 390, "height": 844},
        user_agent=PHONE_UA, has_touch=True, is_mobile=True,
    )
    try:
        page = context.new_page()
        goto_retrying_network_change(page, harness.base_url + "/")
        expect(page.locator("#kiss-server-loading")).to_be_visible()
        got = _sampled(page)
        assert got["colour"] == _LIGHT
        assert got["themeColor"] == "#fafafa"
    finally:
        context.close()

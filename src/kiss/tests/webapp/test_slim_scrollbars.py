# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Every scrolling element wears the same slim scrollbar, out of sight until used.

In a real Chromium with native (non-overlay) scrollbars the bar is a
6px gutter holding a 4px pill thumb, on the history list as well as on
the list panes of the bottom sheets and the task text of a task panel.
Chromium drops the ``::-webkit-scrollbar`` styling on any element that
also sets the standard ``scrollbar-width`` / ``scrollbar-color``, which
used to leave those elements with its 10px or 15px native bar; the
gutter widths here catch that.

The gutter is bare until the user scrolls: the thumb shows for a scroll
the user makes (the wheel, a scroll key, a press on the bar) and under
the pointer, where it is stronger, and goes again shortly after the
last scroll.  A scroll the page makes itself shows none.  The sidebar's
list and the chat's transcript both behave so.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from playwright.sync_api import Page, sync_playwright

from kiss.server.web_server import _SHARE_PAGE_LIGHT_VARS_CSS
from kiss.tests.webapp.test_history_panel_tints import (
    _open_remote_desktop_page,
)

_GUTTER = 6
_THUMB = 4

# Elements that scroll their own content.  The bottom-sheet list is
# hidden until its sheet opens, so each is shown and overfilled.
_SCROLLERS = ["#history-list", "#tricks-list"]

# The scrollers whose bars the user meets most: the sidebar's list and
# the chat's transcript.
_LIST = "#history-list"
_CHAT = "#output"

Pixel = tuple[int, int, int]


@pytest.fixture(scope="module")
def _page():
    """The remote desktop page in a Chromium drawing real scrollbars."""
    with sync_playwright() as p:
        browser = p.chromium.launch(ignore_default_args=["--hide-scrollbars"])
        try:
            context, page = _open_remote_desktop_page(browser)
            page.evaluate(
                """([sels, chat]) => {
                  const tall = () => {
                    const d = document.createElement('div');
                    d.style.height = '3000px';
                    return d;
                  };
                  for (const s of sels.slice(1)) {
                    const el = document.querySelector(s);
                    el.parentElement.style.display = 'block';
                    el.style.cssText = 'display:block;height:100px;width:200px';
                    el.appendChild(tall());
                  }
                  const t = document.createElement('div');
                  t.className = 'task-panel-text';
                  t.id = 'probe-task-text';
                  t.style.cssText = 'width:200px;max-height:100px';
                  t.appendChild(tall());
                  document.body.appendChild(t);
                  document.querySelector(chat).appendChild(tall());
                }""",
                [_SCROLLERS, _CHAT],
            )
            yield page
            context.close()
        finally:
            browser.close()


def _gutter(page: Page, selector: str) -> int:
    """Width the vertical scrollbar takes from ``selector``."""
    return int(
        page.evaluate(
            "(s) => { const el = document.querySelector(s);"
            " return el.offsetWidth - el.clientWidth; }",
            selector,
        )
    )


def _gutter_rows(page: Page, selector: str) -> list[list[Pixel]]:
    """The gutter of ``selector`` in one picture: a row of pixels per
    pixel of its height, each from the content edge outwards."""
    box = page.evaluate(
        "(s) => { const el = document.querySelector(s);"
        " const r = el.getBoundingClientRect();"
        " return {right: r.right, top: r.top, height: Math.floor(r.height),"
        " bw: parseFloat(getComputedStyle(el).borderRightWidth) || 0}; }",
        selector,
    )
    png = page.screenshot(
        clip={
            "x": box["right"] - box["bw"] - _GUTTER,
            "y": box["top"],
            "width": _GUTTER,
            "height": box["height"],
        }
    )
    image = Image.open(io.BytesIO(png)).convert("RGB")
    assert image.size == (_GUTTER, box["height"]), image.size
    # Three bytes a pixel, row after row.
    rgb = image.tobytes()
    row_bytes = 3 * _GUTTER
    return [
        [(rgb[at], rgb[at + 1], rgb[at + 2]) for at in range(start, start + row_bytes, 3)]
        for start in range(0, len(rgb), row_bytes)
    ]


def _shows_bar(page: Page, selector: str) -> bool:
    """Whether ``selector`` is marked as being scrolled by the user."""
    return bool(
        page.evaluate("(s) => document.querySelector(s).classList.contains('scrolling')", selector)
    )


# Scroll an element from the page's own script and resolve once its
# scroll event has been dispatched (at once when it is there already).
_SCRIPTED_SCROLL_JS = """([s, top]) => new Promise(done => {
  const el = document.querySelector(s);
  if (el.scrollTop === top) { done(); return; }
  el.addEventListener('scroll', () => done(), {once: true});
  el.scrollTop = top;
})"""


def _at_rest(page: Page, selector: str) -> list[list[Pixel]]:
    """The gutter of ``selector`` with the pointer away and no scroll
    of the user's pending, after the page itself has scrolled it away
    from its top and back."""
    page.mouse.move(600, 400)
    page.wait_for_function(
        "(s) => !document.querySelector(s).classList.contains('scrolling')", arg=selector
    )
    for top in (40, 0):
        page.evaluate(_SCRIPTED_SCROLL_JS, [selector, top])
    return _gutter_rows(page, selector)


def _scrolled_by_wheel(page: Page, selector: str) -> list[list[Pixel]]:
    """The gutter of ``selector`` right after the user wheeled it down
    and back to its top.  The bar stays for 1.2 s after a scroll: a
    picture counts only when the bar was still up once it was taken."""
    box = page.locator(selector).bounding_box()
    assert box is not None
    for _attempt in range(3):
        page.mouse.move(box["x"] + 40, box["y"] + 40)
        page.mouse.wheel(0, 120)
        page.wait_for_function("(s) => document.querySelector(s).scrollTop > 0", arg=selector)
        page.mouse.wheel(0, -1000)
        page.wait_for_function("(s) => document.querySelector(s).scrollTop === 0", arg=selector)
        rows = _gutter_rows(page, selector)
        if _shows_bar(page, selector):
            return rows
    raise AssertionError(f"{selector}: the bar went before its picture was taken, three times")


def _luma(px: Pixel) -> float:
    return 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]


def _colours(rows: list[list[Pixel]]) -> set[Pixel]:
    return {px for row in rows for px in row}


@pytest.mark.parametrize("selector", [*_SCROLLERS, "#probe-task-text", _CHAT])
def test_every_scroller_has_the_slim_gutter(_page, selector: str) -> None:
    """Each scrolling element gives up exactly the 6px gutter."""
    assert _page.evaluate(
        "(s) => { const el = document.querySelector(s);"
        " return el.scrollHeight > el.clientHeight; }",
        selector,
    ), selector
    assert _gutter(_page, selector) == _GUTTER, selector


@pytest.mark.parametrize("selector", [_LIST, _CHAT])
def test_bar_is_out_of_sight_unless_the_user_scrolls(_page, selector: str) -> None:
    """The gutter of the sidebar's list and of the chat is bare at
    rest, a scroll of the page's own leaving it so; the user's wheel
    brings the thumb up and it goes again shortly after."""
    page = _page
    idle = _at_rest(page, selector)
    assert len(_colours(idle)) == 1, sorted(_colours(idle))
    assert not _shows_bar(page, selector)

    scrolled = _scrolled_by_wheel(page, selector)
    assert len(_colours(scrolled)) > 1, "the user's scroll shows the thumb"

    page.wait_for_function(
        "(s) => !document.querySelector(s).classList.contains('scrolling')", arg=selector
    )
    assert _gutter_rows(page, selector) == idle, "the thumb goes after the scroll"


def test_thumb_is_a_slim_pill_that_strengthens_under_the_pointer(_page) -> None:
    """While it shows, the history list's thumb is 4px wide with a 1px
    gap on either side and has rounded ends, the track below it bare;
    under the pointer it is stronger, scrolling or not."""
    page = _page
    idle = _at_rest(page, _LIST)
    bg = idle[0][0]
    rows = _scrolled_by_wheel(page, _LIST)
    rest, track = rows[12], rows[-12]
    # Track: every gutter pixel is the list background; nothing drawn.
    assert set(track) == {bg}, track
    # Thumb: 1px gap, 4 thumb pixels, 1px gap.
    assert rest[0] == bg and rest[-1] == bg, rest
    thumb = rest[1 : 1 + _THUMB]
    assert len(set(thumb)) == 1 and thumb[0] != bg, rest
    # Rounded end: on the thumb's first row (the row after its 1px gap)
    # the outer columns are only partly covered by the arc, so they
    # sit between the bare track and the thumb's interior.
    end = rows[1]
    assert _luma(bg) < _luma(end[1]) < _luma(end[2]), (bg, end)
    assert _luma(end[1]) < _luma(rest[1]), (end, rest)

    # The pointer on the thumb, the scroll long over.
    box = page.locator(_LIST).bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] - 3, box["y"] + 12)
    page.wait_for_function(
        "(s) => !document.querySelector(s).classList.contains('scrolling')", arg=_LIST
    )
    on_thumb = _gutter_rows(page, _LIST)[12]
    # The dark page: a stronger thumb is a lighter one.
    assert _luma(bg) < _luma(rest[2]) < _luma(on_thumb[2]), (bg, rest, on_thumb)
    assert on_thumb[0] == bg and on_thumb[-1] == bg, on_thumb


def test_press_on_the_bar_scrolls_and_shows_it(_page) -> None:
    """A press on the bar's track scrolls the list a page: the user's
    own scroll, so the list is marked as scrolled (which is what
    paints its thumb).  A press inside the list is no scroll."""
    page = _page
    _at_rest(page, _LIST)
    box = page.locator(_LIST).bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] - 3, box["y"] + box["height"] - 12)
    page.wait_for_function(
        "(s) => { const el = document.querySelector(s);"
        " return el.scrollTop > 0 && el.classList.contains('scrolling'); }",
        arg=_LIST,
        timeout=5000,
    )
    # The bar gone again, the user presses a day label of the list and
    # the page scrolls the list right after: no bar.
    page.wait_for_function(
        "(s) => !document.querySelector(s).classList.contains('scrolling')", arg=_LIST
    )
    page.locator(f"{_LIST} .history-day-sep").last.click()
    page.evaluate(_SCRIPTED_SCROLL_JS, [_LIST, 0])
    assert not _shows_bar(page, _LIST)


# A scroller for one test, over the page: a 100px box whose content a
# press held on its bar's track takes about ten seconds to run through.
_HELD = "#probe-held"
_ADD_HELD_SCROLLER_JS = """() => {
  const el = document.createElement('div');
  el.id = 'probe-held';
  el.style.cssText = 'position:fixed;left:300px;top:100px;width:200px;height:100px;'
    + 'overflow-y:auto;z-index:2147483647;background:var(--bg)';
  const tall = document.createElement('div');
  tall.style.height = '20000px';
  el.appendChild(tall);
  document.body.appendChild(el);
}"""


def test_bar_stays_while_a_press_held_on_it_scrolls(_page) -> None:
    """A press held on the bar's track scrolls its element page after
    page: the bar shows for as long as that scroll goes on, however
    long ago the press began, and goes once the user has let go."""
    page = _page
    page.evaluate(_ADD_HELD_SCROLLER_JS)
    try:
        # A user presses a bar that is on the screen.  Chromium drops a
        # press on the bar of a scroller it has not drawn yet, so the
        # press waits for a picture of the scroller, its bar still out
        # of sight.
        page.mouse.move(600, 400)
        assert len(_colours(_gutter_rows(page, _HELD))) == 1
        box = page.locator(_HELD).bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] - 3, box["y"] + box["height"] - 6)
        page.mouse.down()
        # Longer than the second an input counts for plus the 1.2 s a
        # bar stays after a scroll.
        page.wait_for_timeout(2600)
        top = page.evaluate("(s) => document.querySelector(s).scrollTop", _HELD)
        held_rows = _gutter_rows(page, _HELD)
        marked = _shows_bar(page, _HELD)
        # The press was still scrolling when the picture was taken.
        page.wait_for_function(
            "([s, top]) => document.querySelector(s).scrollTop > top",
            arg=[_HELD, top],
            timeout=5000,
        )
        page.mouse.up()
        assert marked, "the bar went while the held press was scrolling"
        assert len(_colours(held_rows)) > 1, "no thumb is painted during the held press"
        page.wait_for_function(
            "(s) => !document.querySelector(s).classList.contains('scrolling')",
            arg=_HELD,
            timeout=5000,
        )
        assert len(_colours(_gutter_rows(page, _HELD))) == 1
        # The press is over: a scroll the page makes now shows no bar.
        page.evaluate(_SCRIPTED_SCROLL_JS, [_HELD, 0])
        assert not _shows_bar(page, _HELD)
    finally:
        page.mouse.up()
        page.evaluate("document.getElementById('probe-held').remove()")


# Scroll an element from the page's own script until its last child
# covers its whole box; resolves to the new scrollTop once the scroll
# event has been dispatched.
_LAST_CHILD_OVER_BOX_JS = """(s) => new Promise(done => {
  const el = document.querySelector(s);
  const top = r => r.getBoundingClientRect().top;
  el.addEventListener('scroll', () => done(el.scrollTop), {once: true});
  el.scrollTop += top(el.lastElementChild) - top(el) + 500;
})"""


def test_scroll_key_with_nothing_focused_shows_the_bar(_page) -> None:
    """A click on the chat's transcript focuses nothing, and a scroll
    key pressed after it scrolls the transcript: the user's own scroll,
    so the transcript's bar shows."""
    page = _page
    _at_rest(page, _CHAT)
    top = page.evaluate(_LAST_CHILD_OVER_BOX_JS, _CHAT)
    box = page.locator(_CHAT).bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    assert page.evaluate("document.activeElement === document.body")
    assert not _shows_bar(page, _CHAT)
    page.keyboard.press("PageDown")
    # The transcript moved, and its scroll event marked it.
    page.wait_for_function(
        "([s, top]) => { const el = document.querySelector(s);"
        " return el.scrollTop > top && el.classList.contains('scrolling'); }",
        arg=[_CHAT, top],
        timeout=5000,
    )


def test_remote_light_theme_thumb_follows_the_light_foreground(_page) -> None:
    """On the remote page the light theme is a ``light-theme`` class on
    ``<body>`` (not ``<html>``), so the thumb colour has to be derived
    there too: with the production light-theme variables the thumb of
    a list being scrolled is clearly darker than the light background."""
    page = _page
    page.add_style_tag(content=_SHARE_PAGE_LIGHT_VARS_CSS)
    page.evaluate("() => document.body.classList.add('light-theme')")
    try:
        rest = _scrolled_by_wheel(page, _LIST)[12]
    finally:
        page.evaluate("() => document.body.classList.remove('light-theme')")
    bg, thumb = rest[0], rest[2]
    assert _luma(bg) > 200, rest
    # A 20% mix of the light theme's #0b0b0b foreground into the sidebar
    # background (#f5f5f3) sits 47 luma below it; a thumb still mixed
    # from the dark foreground (#f0efec) would come within 2.
    assert _luma(bg) - _luma(thumb) > 25, rest

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Every scrolling element wears the same slim scrollbar.

In a real Chromium with native (non-overlay) scrollbars the bar is a
6px gutter holding a 4px pill thumb, on the history list as well as on
the list panes of the bottom sheets and the task text of a task panel.
Chromium drops the ``::-webkit-scrollbar`` styling on any element that
also sets the standard ``scrollbar-width`` / ``scrollbar-color``, which
used to leave those elements with its 10px or 15px native bar; the
gutter widths here catch that.  The thumb is faint at rest and stronger
under the pointer.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from playwright.sync_api import sync_playwright

from kiss.server.web_server import _SHARE_PAGE_LIGHT_VARS_CSS
from kiss.tests.agents.vscode.test_history_panel_tints import (
    _open_remote_desktop_page,
)

_GUTTER = 6
_THUMB = 4

# Elements that scroll their own content.  The bottom-sheet lists are
# hidden until their sheet opens, so each is shown and overfilled.
_SCROLLERS = ["#history-list", "#frequent-list", "#tricks-list", "#workdir-list"]


@pytest.fixture(scope="module")
def _page():
    """The remote desktop page in a Chromium drawing real scrollbars."""
    with sync_playwright() as p:
        browser = p.chromium.launch(ignore_default_args=["--hide-scrollbars"])
        try:
            context, page = _open_remote_desktop_page(browser)
            page.evaluate(
                """(sels) => {
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
                }""",
                _SCROLLERS,
            )
            yield page
            context.close()
        finally:
            browser.close()


def _gutter(page, selector: str) -> int:
    """Width the vertical scrollbar takes from ``selector``."""
    return int(
        page.evaluate(
            "(s) => { const el = document.querySelector(s);"
            " return el.offsetWidth - el.clientWidth; }",
            selector,
        )
    )


def _gutter_row(page, selector: str, y_offset: int) -> list[tuple[int, int, int]]:
    """The gutter's pixels on one row, from the content edge outwards."""
    box = page.evaluate(
        "(s) => { const r = document.querySelector(s).getBoundingClientRect();"
        " return {right: r.right, top: r.top, bw: parseFloat(getComputedStyle("
        "document.querySelector(s)).borderRightWidth) || 0}; }",
        selector,
    )
    left = box["right"] - box["bw"] - _GUTTER
    png = page.screenshot(
        clip={"x": left, "y": box["top"] + y_offset, "width": _GUTTER, "height": 1}
    )
    raw = Image.open(io.BytesIO(png)).convert("RGB").tobytes()
    return [(raw[3 * x], raw[3 * x + 1], raw[3 * x + 2]) for x in range(_GUTTER)]


def _luma(px: tuple[int, int, int]) -> float:
    return 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]


@pytest.mark.parametrize("selector", [*_SCROLLERS, "#probe-task-text"])
def test_every_scroller_has_the_slim_gutter(_page, selector: str) -> None:
    """Each scrolling element gives up exactly the 6px gutter."""
    assert _page.evaluate(
        "(s) => { const el = document.querySelector(s);"
        " return el.scrollHeight > el.clientHeight; }",
        selector,
    ), selector
    assert _gutter(_page, selector) == _GUTTER, selector


def test_thumb_is_a_slim_pill_that_strengthens_under_the_pointer(_page) -> None:
    """The history list's thumb is 4px wide with a 1px gap on either
    side, has rounded ends and gets stronger under the pointer; the
    track below the thumb stays bare."""
    page = _page
    box = page.evaluate(
        "() => { const l = document.getElementById('history-list');"
        " l.scrollTop = 0; const r = l.getBoundingClientRect();"
        " return {right: r.right, top: r.top, bottom: r.bottom}; }"
    )
    page.mouse.move(600, 400)
    page.wait_for_timeout(50)
    rest = _gutter_row(page, "#history-list", 12)
    track = _gutter_row(page, "#history-list", int(box["bottom"] - box["top"]) - 12)
    # Track: every gutter pixel is the list background; nothing drawn.
    assert len(set(track)) == 1, track
    bg = track[0]
    # Thumb: 1px gap, 4 thumb pixels, 1px gap.
    assert rest[0] == bg and rest[-1] == bg, rest
    thumb = rest[1 : 1 + _THUMB]
    assert len(set(thumb)) == 1 and thumb[0] != bg, rest
    # Rounded end: on the thumb's first row (the row after its 1px gap)
    # the outer columns are only partly covered by the arc, so they
    # sit between the bare track and the thumb's interior.
    end = _gutter_row(page, "#history-list", 1)
    assert _luma(bg) < _luma(end[1]) < _luma(end[2]), (bg, end)
    assert _luma(end[1]) < _luma(rest[1]), (end, rest)

    page.mouse.move(box["right"] - 3, box["top"] + 12)
    page.wait_for_timeout(50)
    on_thumb = _gutter_row(page, "#history-list", 12)
    # The dark page: a stronger thumb is a lighter one.
    assert _luma(bg) < _luma(rest[2]) < _luma(on_thumb[2]), (bg, rest, on_thumb)
    assert on_thumb[0] == bg and on_thumb[-1] == bg, on_thumb


def test_remote_light_theme_thumb_follows_the_light_foreground(_page) -> None:
    """On the remote page the light theme is a ``light-theme`` class on
    ``<body>`` (not ``<html>``), so the thumb colour has to be derived
    there too: with the production Light Modern variables the resting
    thumb is clearly darker than the light background."""
    page = _page
    page.add_style_tag(content=_SHARE_PAGE_LIGHT_VARS_CSS)
    page.evaluate("() => document.body.classList.add('light-theme')")
    page.mouse.move(600, 400)
    page.wait_for_timeout(50)
    try:
        rest = _gutter_row(page, "#history-list", 12)
    finally:
        page.evaluate("() => document.body.classList.remove('light-theme')")
    bg, thumb = rest[0], rest[2]
    assert _luma(bg) > 200, rest
    # A 20% mix of Light Modern's #3b3b3b foreground into the sidebar
    # background sits about 35 luma below it; a thumb still mixed from
    # the dark foreground (#ccc) would come within 10.
    assert _luma(bg) - _luma(thumb) > 25, rest

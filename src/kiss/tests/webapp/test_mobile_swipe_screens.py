# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from
#   kiss.tests.server.test_explorer_scm_commands /
#   kiss.tests.webapp.test_workspace_sections are re-exported
#   for pytest by name)
"""The phone layout's four screens, stepped through by swiping.

Below the desktop breakpoint the remote page's four side-by-side panes
are four screens in the same order: the chats (history drawer), the
chat, the open file (a content tab in place of the chat) and the task
info (right drawer).  A horizontal touch swipe moves one screen in the
finger's direction and skips the file screen while no file is open.
A drag that scrolled something sideways, a two-finger drag, a vertical
drag, a short, a slow, or a selecting drag, one in the composer, or one
under an open sheet, is not a swipe.

The file screen's tab bar lists the open files, terminals and browser
pages and nothing else, also when there is only one.  A swipe right
from it goes back to the chat that was on screen before it (its machine
in the header, its task in the transcript), whichever chat the tab was
opened from.  Stop and "Save chat", which stay on the file screen, act
on that same chat.

The swipes are real touch sequences dispatched through the Chrome
DevTools protocol (``Input.dispatchTouchEvent``), so Chromium produces
the same touch events and native scrolling a finger would.
"""

from __future__ import annotations

import http.server
import os
import threading
from pathlib import Path

import pytest

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import persistence, web_stealth
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_editing import _dismiss_toasts
from kiss.tests.webapp.test_workspace_sections import (
    _explorer_row,
    _open_page,
    _open_page_mobile,
    _show_section,
    browser,  # noqa: F401  (module fixture used by param name)
)

# The screen on view, read the way a user sees it: an open drawer, else
# the active tab of the tab strip.
_SCREEN_JS = """() => {
  if (document.getElementById('sidebar').classList.contains('open')) return 'chats';
  if (document.getElementById('meta-panel').classList.contains('open')) return 'info';
  return document.querySelector('#tab-list .chat-tab.active.content-tab')
    ? 'file' : 'chat';
}"""

# A finger on the chat transcript / content view: mid-screen, clear of
# the tab strip above and the composer below.
_Y = 400
_LEFT_X = 60
_RIGHT_X = 330


def _screen(page) -> str:
    return str(page.evaluate(_SCREEN_JS))


def _touch_session(page):
    """A CDP session with touch emulation on, for dispatching fingers."""
    cdp = page.context.new_cdp_session(page)
    cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": True})
    return cdp


def _points(points: list[tuple[float, float]]) -> list[dict]:
    return [{"x": x, "y": y} for x, y in points]


def _drag(cdp, start: tuple[float, float], end: tuple[float, float],
          steps: int = 6, between=None) -> None:
    """One finger pressed at *start*, moved in *steps* to *end*, lifted.
    *between* runs after the moves and before the lift."""
    cdp.send("Input.dispatchTouchEvent",
             {"type": "touchStart", "touchPoints": _points([start])})
    for i in range(1, steps + 1):
        x = start[0] + (end[0] - start[0]) * i / steps
        y = start[1] + (end[1] - start[1]) * i / steps
        cdp.send("Input.dispatchTouchEvent",
                 {"type": "touchMove", "touchPoints": _points([(x, y)])})
    if between is not None:
        between()
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


# mobileSwipe.js ignores a finger that was down longer than this (a
# press-and-drag).
_SWIPE_MAX_MS = 1500

# Records, next to the detector's own listeners, how long the page saw
# the last finger down: the detector's clock, not the test's.  It starts
# on the window, before the detector's document listener, and stops on
# the document after it, so it never reads less than the detector.
_FINGER_CLOCK_JS = """() => {
  if (window.__fingerDownMs !== undefined) return;
  window.__fingerDownMs = 0;
  let at = 0;
  const opts = {capture: true, passive: true};
  window.addEventListener('touchstart', () => { at = Date.now(); }, opts);
  document.addEventListener('touchend', () => {
    window.__fingerDownMs = Date.now() - at;
  }, opts);
}"""

# Resolves, once the clock above has the lifted finger (it is reset to
# null before each drag), to that finger's time down and the screen
# then on view (``%s``: the screen function).  A finger on a rendered
# preview never reaches this document (the iframe's own detector posts
# the swipe), so after a second the time down stays null.
_LIFTED_JS = """new Promise(resolve => {
  const until = Date.now() + 1000;
  const read = () => (window.__fingerDownMs === null && Date.now() < until)
    ? setTimeout(read, 10) : resolve([window.__fingerDownMs, (%s)()]);
  read();
})"""


def _quick_drag(cdp, start: tuple[float, float], end: tuple[float, float],
                steps: int = 6) -> None:
    """:func:`_drag`, again when the detector may have ignored the
    finger as too slow: every touch event is a CDP round trip, and on a
    loaded machine the whole finger can take longer than a swipe may,
    which the detector rightly ignores.

    The page clock never reads less than the detector's, so a time
    under the limit is one the detector accepted.  At the limit or over
    it may still have taken the swipe: the clock also counts what the
    swipe does before the clock stops (turning the screen, a few ms).
    So the screen decides: the finger is dragged again only when the
    screen did not change, never after a drag the detector took (that
    would swipe twice).  The detector turns the screen in the same
    touchend dispatch, before the clock stops, so the two are read
    together once the clock has the finger.  A finger this document
    never saw (on a preview) is not judged."""
    before = cdp.send("Runtime.evaluate", {
        "expression": f"({_FINGER_CLOCK_JS})(), ({_SCREEN_JS})()",
        "returnByValue": True,
    })["result"]["value"]
    down = 0
    for _ in range(5):
        cdp.send("Runtime.evaluate", {"expression": "window.__fingerDownMs = null"})
        _drag(cdp, start, end, steps)
        down, screen = cdp.send("Runtime.evaluate", {
            "expression": _LIFTED_JS % _SCREEN_JS,
            "awaitPromise": True, "returnByValue": True,
        })["result"]["value"]
        if down is None or down < _SWIPE_MAX_MS or screen != before:
            return
    raise AssertionError(f"the finger took {down} ms, never a swipe's time")


def _swipe_left(cdp, y: float = _Y) -> None:
    """The finger moves right to left: the next screen."""
    _quick_drag(cdp, (_RIGHT_X, y), (_LEFT_X, y))


def _swipe_right(cdp, y: float = _Y) -> None:
    """The finger moves left to right: the previous screen."""
    _quick_drag(cdp, (_LEFT_X, y), (_RIGHT_X, y))


def _wait_screen(page, screen: str) -> None:
    page.wait_for_function(
        "s => (" + _SCREEN_JS + ")() === s", arg=screen, timeout=10000,
    )
    # Let the drawer transition settle before the next finger.
    page.wait_for_timeout(400)


def _settled_screen(page) -> str:
    page.wait_for_timeout(400)
    return _screen(page)


_VIEW = "#content-tab-area .content-tab-view:not([style*='display: none']) "


def _open_file_from_drawer(page, name: str) -> None:
    """Open *name* from the Explorer inside the task-info drawer; the
    drawer closes and the file is the active tab."""
    if _screen(page) != "info":
        page.click("#meta-drawer-btn")
        _wait_screen(page, "info")
    _show_section(page, "meta-explorer")
    page.wait_for_selector(".explorer-row.is-file", timeout=15000)
    _explorer_row(page, name).click()
    _wait_screen(page, "file")
    page.wait_for_selector(
        _VIEW + ".monaco-editor, " + _VIEW + ".content-code-fallback, "
        + _VIEW + ".content-html-frame",
        timeout=30000,
    )


def _open_mobile(browser, harness):
    context, page, _ = _open_page_mobile(browser, harness)
    _dismiss_toasts(page)
    return context, page, _touch_session(page)


def test_without_a_file_the_swipes_step_chats_chat_info(browser, harness):
    context, page, cdp = _open_mobile(browser, harness)
    try:
        assert _screen(page) == "chat"
        _swipe_right(cdp)
        _wait_screen(page, "chats")
        # The first screen: nothing further to the left.
        _swipe_right(cdp)
        assert _settled_screen(page) == "chats"
        _swipe_left(cdp)
        _wait_screen(page, "chat")
        # No file is open: the file screen is skipped on the way out...
        _swipe_left(cdp)
        _wait_screen(page, "info")
        # ...the last screen stops the finger...
        _swipe_left(cdp)
        assert _settled_screen(page) == "info"
        # ...and is skipped on the way back.
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        # The drawer buttons and the swipes agree on the screen: a drawer
        # opened by its button is left by the same swipe.
        page.click("#meta-drawer-btn")
        _wait_screen(page, "info")
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        page.click("#menu-btn")
        _wait_screen(page, "chats")
        _swipe_left(cdp)
        _wait_screen(page, "chat")
    finally:
        context.close()


def test_an_open_file_is_the_third_screen(browser, harness):
    context, page, cdp = _open_mobile(browser, harness)
    try:
        _open_file_from_drawer(page, "feature.txt")
        _swipe_left(cdp)
        _wait_screen(page, "info")
        _swipe_right(cdp)
        _wait_screen(page, "file")
        assert page.locator("#content-tab-area").is_visible()
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        assert page.locator("#output").is_visible()
        # The file is still open, so it is the next screen again.
        _swipe_left(cdp)
        _wait_screen(page, "file")
        # Chats, chat, file, info: the whole way across and back.
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        _swipe_right(cdp)
        _wait_screen(page, "chats")
        _swipe_left(cdp)
        _wait_screen(page, "chat")
        _swipe_left(cdp)
        _wait_screen(page, "file")
        _swipe_left(cdp)
        _wait_screen(page, "info")
        _swipe_right(cdp)
        _wait_screen(page, "file")
        # The chats drawer opened over the file screen leaves to the chat
        # screen (the screens keep their order).
        page.click("#menu-btn")
        _wait_screen(page, "chats")
        _swipe_left(cdp)
        _wait_screen(page, "chat")
        # A second file: the file screen shows the one viewed last, and
        # once that one is closed, the file that is still open.
        _open_file_from_drawer(page, "README.md")
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        _swipe_left(cdp)
        _wait_screen(page, "file")
        assert "README.md" in page.locator(
            "#tab-list .chat-tab.active.content-tab"
        ).inner_text()
        page.click("#tab-list .chat-tab.active.content-tab .chat-tab-close")
        page.wait_for_function(
            "document.querySelectorAll('#tab-list .chat-tab.content-tab').length === 1",
            timeout=10000,
        )
        if _screen(page) != "chat":
            _swipe_right(cdp)
            _wait_screen(page, "chat")
        _swipe_left(cdp)
        _wait_screen(page, "file")
        assert "feature.txt" in page.locator(
            "#tab-list .chat-tab.active.content-tab"
        ).inner_text()
    finally:
        context.close()


def test_mobile_chat_hides_editor_tabs_without_closing_them(browser, harness):
    """Editor tabs belong to the file screen, not the mobile chat header."""
    context, page, cdp = _open_mobile(browser, harness)
    try:
        _open_file_from_drawer(page, "feature.txt")
        _open_file_from_drawer(page, "README.md")
        files = page.locator("#tab-list .content-tab")
        file_ids = files.evaluate_all("els => els.map(el => el.dataset.tabId)")
        assert len(file_ids) == 2

        _swipe_right(cdp)
        _wait_screen(page, "chat")
        assert page.locator("#output").is_visible()
        assert page.locator("#task-input").is_visible()
        assert files.count() == 0
        # A lone chat needs no strip, including no empty editor-tab row.
        assert not page.locator("#tab-bar").is_visible()
        assert not page.locator("#content-tab-bar").is_visible()

        # Desktop still shows the retained files on its separate editor row.
        page.set_viewport_size({"width": 900, "height": 844})
        page.wait_for_selector("body.remote-desktop", state="attached")
        desktop_files = page.locator("#content-tab-list .content-tab")
        assert desktop_files.evaluate_all(
            "els => els.map(el => el.dataset.tabId)"
        ) == file_ids
        assert page.locator("#content-tab-bar").is_visible()
        assert files.count() == 0

        # Crossing back to mobile must not leak that row into the chat.
        page.set_viewport_size({"width": 899, "height": 844})
        page.wait_for_selector("body.remote-desktop", state="detached")
        assert files.count() == 0
        assert not page.locator("#tab-bar").is_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        _swipe_left(cdp)
        _wait_screen(page, "file")
        assert files.evaluate_all(
            "els => els.map(el => el.dataset.tabId)"
        ) == file_ids
        assert page.locator("#tab-bar").is_visible()
        assert page.locator("#content-tab-area").is_visible()
        assert "README.md" in page.locator(
            "#tab-list .content-tab.active"
        ).inner_text()

        # Both files remain selectable and closable on the editor screen.
        feature = files.filter(has_text="feature.txt")
        feature.click()
        assert "feature.txt" in page.locator(
            "#tab-list .content-tab.active"
        ).inner_text()
        feature.locator(".chat-tab-close").click()
        _wait_screen(page, "file")
        assert files.count() == 1
        assert "README.md" in files.inner_text()
        files.locator(".chat-tab-close").click()
        _wait_screen(page, "chat")
        assert not page.locator("#tab-bar").is_visible()
        # No file remains: the next swipe skips the editor screen.
        _swipe_left(cdp)
        _wait_screen(page, "info")
    finally:
        context.close()


def test_drags_that_are_not_swipes_keep_the_screen(browser, harness):
    context, page, cdp = _open_mobile(browser, harness)
    try:
        assert _screen(page) == "chat"
        # Mostly vertical: scrolling the transcript.
        _drag(cdp, (_LEFT_X, 300), (_RIGHT_X, 520))
        assert _settled_screen(page) == "chat"
        # Too short to be a swipe.
        _drag(cdp, (_LEFT_X, _Y), (_LEFT_X + 40, _Y))
        assert _settled_screen(page) == "chat"
        # Too slow: a press-and-drag, not a swipe.
        _drag(cdp, (_LEFT_X, _Y), (_RIGHT_X, _Y),
              between=lambda: page.wait_for_timeout(1700))
        assert _settled_screen(page) == "chat"
        # Two fingers from the start (a pinch).
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchStart",
            "touchPoints": _points([(_LEFT_X, _Y), (_LEFT_X, _Y + 80)]),
        })
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchMove",
            "touchPoints": _points([(_RIGHT_X, _Y), (_RIGHT_X, _Y + 80)]),
        })
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        assert _settled_screen(page) == "chat"
        # A second finger joining a one-finger drag, then both lifting.
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchStart", "touchPoints": _points([(_LEFT_X, _Y)]),
        })
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchStart",
            "touchPoints": _points([(_LEFT_X, _Y), (_LEFT_X, _Y + 80)]),
        })
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchMove",
            "touchPoints": _points([(_RIGHT_X, _Y), (_RIGHT_X, _Y + 80)]),
        })
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchEnd", "touchPoints": _points([(_RIGHT_X, _Y + 80)]),
        })
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        assert _settled_screen(page) == "chat"
        # A cancelled touch (the system took the finger).
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchStart", "touchPoints": _points([(_LEFT_X, _Y)]),
        })
        cdp.send("Input.dispatchTouchEvent", {
            "type": "touchMove", "touchPoints": _points([(_RIGHT_X, _Y)]),
        })
        cdp.send("Input.dispatchTouchEvent", {"type": "touchCancel", "touchPoints": []})
        assert _settled_screen(page) == "chat"
        # Dragging across the composer: its own swipes (accepting a
        # suggestion, cycling the prompt history).
        box = page.locator("#task-input").bounding_box()
        assert box is not None
        y = box["y"] + box["height"] / 2
        _drag(cdp, (box["x"] + 10, y), (box["x"] + box["width"] - 10, y))
        assert _settled_screen(page) == "chat"
        # A drag that made a text selection (long-press, then drag the
        # handle) selects; it does not swipe.
        page.evaluate(
            """() => {
              const p = document.createElement('p');
              p.id = 'swipe-probe-text';
              p.textContent = 'some words to select';
              document.getElementById('output').appendChild(p);
            }"""
        )
        _drag(
            cdp, (_RIGHT_X, _Y), (_LEFT_X, _Y),
            between=lambda: page.evaluate(
                "() => document.getSelection().selectAllChildren("
                "document.getElementById('swipe-probe-text'))"
            ),
        )
        assert _settled_screen(page) == "chat"
        page.evaluate("() => document.getSelection().removeAllRanges()")
        # The same drag with no selection made is a swipe.
        _swipe_left(cdp)
        _wait_screen(page, "info")
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        # A settings sheet over the page keeps the gesture.
        page.click("#more-btn")
        page.click("#settings-btn")
        page.wait_for_selector("#settings-panel.open", timeout=10000)
        page.wait_for_timeout(400)
        _swipe_right(cdp, y=300)
        assert _settled_screen(page) == "chat"
        assert page.locator("#settings-panel.open").count() == 1
    finally:
        context.close()


def test_a_sideways_scroll_is_not_a_swipe(browser, harness):
    """A wide transcript block (a long code line) scrolls under the
    finger; the swipe detector leaves the screen alone, and a drag on a
    Monaco editor that scrolled its long line does the same."""
    # One line somewhat wider than a 390px editor: a single drag scrolls
    # it to its end.
    (harness.work_dir / "wide.txt").write_text("x" * 60 + "\n" + "short\n")
    context, page, cdp = _open_mobile(browser, harness)
    try:
        page.evaluate(
            """() => {
              const box = document.createElement('div');
              box.id = 'swipe-probe-scroller';
              box.style.cssText =
                'overflow-x: auto; width: 300px; height: 120px;'
                + ' touch-action: pan-x;';
              const wide = document.createElement('div');
              wide.style.cssText = 'width: 1500px; height: 100px';
              box.appendChild(wide);
              // At the top of the transcript, clear of the composer.
              const out = document.getElementById('output');
              out.insertBefore(box, out.firstChild);
            }"""
        )
        box = page.locator("#swipe-probe-scroller").bounding_box()
        assert box is not None
        y = box["y"] + box["height"] / 2
        _drag(cdp, (box["x"] + 250, y), (box["x"] + 20, y), steps=12)
        assert page.evaluate(
            "document.getElementById('swipe-probe-scroller').scrollLeft"
        ) > 0
        assert _settled_screen(page) == "chat"
        # The block scrolled to its end no longer moves: the next drag on
        # it IS a swipe.
        page.evaluate(
            "() => { const b = document.getElementById('swipe-probe-scroller');"
            " b.scrollLeft = b.scrollWidth; }"
        )
        _quick_drag(cdp, (box["x"] + 250, y), (box["x"] + 20, y), steps=12)
        _wait_screen(page, "info")
        _swipe_right(cdp)
        _wait_screen(page, "chat")

        _open_file_from_drawer(page, "wide.txt")
        editor = page.locator(_VIEW + ".monaco-editor")
        if editor.count() == 0:
            pytest.skip("Monaco did not load (no CDN); the plain fallback "
                        "is a scroller the first half already covered")
        page.wait_for_function(
            "() => { const s = document.querySelector("
            "'#content-tab-area .monaco-editor .monaco-scrollable-element');"
            " return s && s.scrollWidth > 0; }",
            timeout=15000,
        )
        line = page.locator(_VIEW + ".monaco-editor .view-line").first
        line.wait_for(timeout=15000)
        lbox = line.bounding_box()
        assert lbox is not None
        y = lbox["y"] + lbox["height"] / 2
        before = page.evaluate(
            "document.querySelector('#content-tab-area .monaco-editor')"
            ".getBoundingClientRect().left"
        )
        page.wait_for_timeout(600)
        # A right-to-left drag scrolls the long line under the finger
        # (to its end: the line is a little wider than the editor)...
        _drag(cdp, (before + 300, y), (before + 60, y), steps=12)
        page.wait_for_function(
            "() => { const lines = document.querySelector("
            "'#content-tab-area .monaco-editor .lines-content');"
            " return lines && lines.getBoundingClientRect().left < "
            + str(before) + " - 50; }",
            timeout=5000,
        )
        # ...so the screen stays...
        assert _settled_screen(page) == "file"
        # ...and with the editor at its end the same drag moves nothing:
        # it is a swipe.  Few steps: Monaco spends ~100ms on every
        # synthetic touchmove, and a swipe must finish within 1.5s.
        _quick_drag(cdp, (before + 300, y), (before + 60, y), steps=4)
        _wait_screen(page, "info")
    finally:
        context.close()


def test_a_diff_tab_is_a_file_screen_and_swipes_like_one(browser, harness):
    """A file of the commit graph opens as a Monaco diff editor (two
    code editors in one tab): the tab is the file screen, swiping works
    with it open, and keeps working from the other screens while it
    stays open in the background."""
    context, page, cdp = _open_mobile(browser, harness)
    try:
        page.click("#meta-drawer-btn")
        _wait_screen(page, "info")
        _show_section(page, "meta-scm")
        page.wait_for_selector("#scm-graph .scm-commit.is-worktree", timeout=15000)
        page.locator("#scm-graph .scm-commit.is-worktree").click()
        # A modified file: the working tree against HEAD.
        modified = page.locator(
            "#scm-graph .scm-commit.is-worktree"
        ).locator("xpath=..").locator(
            ".scm-commit-files .scm-row[data-scm-status='M']"
        )
        modified.first.wait_for(timeout=15000)
        modified.first.click()
        _wait_screen(page, "file")
        page.wait_for_selector(
            _VIEW + ".monaco-diff-editor, " + _VIEW + ".content-code-fallback",
            timeout=30000,
        )
        page.wait_for_timeout(600)
        _swipe_left(cdp)
        _wait_screen(page, "info")
        _swipe_right(cdp)
        _wait_screen(page, "file")
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        # The diff tab stays open behind the chat and still lets the
        # chats drawer open.
        _swipe_right(cdp)
        _wait_screen(page, "chats")
        _swipe_left(cdp)
        _wait_screen(page, "chat")
        _swipe_left(cdp)
        _wait_screen(page, "file")
    finally:
        context.close()


def test_desktop_layout_ignores_swipes(browser, harness):
    """The desktop remote page shows all four panes side by side: there
    are no screens to swipe between."""
    context, page, _ = _open_page(browser, harness)
    try:
        cdp = _touch_session(page)
        assert page.evaluate("document.body.classList.contains('remote-desktop')")
        # The history pane is docked open; the task-info panel is docked
        # and never carries the drawer's `open`.
        page.wait_for_selector("#sidebar.open", state="attached", timeout=10000)
        _swipe_left(cdp)
        _swipe_left(cdp)
        page.wait_for_timeout(400)
        assert page.locator("#sidebar.open").count() == 1
        assert not page.evaluate(
            "document.getElementById('meta-panel').classList.contains('open')"
        )
        assert not page.evaluate(
            "document.getElementById('sidebar-overlay').classList.contains('open')"
        )
    finally:
        context.close()


# ---- The file screen's own tab bar, and the way back to the chat ------
#
# These tests leave chats open in the daemon's tab registry (a later
# page adopts them), so they come last in the module.


@pytest.fixture(scope="module")
def box(harness):
    """A second daemon, registered as the machine ``box``, with one chat."""
    daemon = start_remote_daemon(
        Path(harness.tmpdir) / "box-home", password="pw-box",
        seed=[{"chat_id": "box-chat", "task": "Box task", "result": "done"}],
    )
    mach.save_machines([])
    mach.add_machine("box", daemon.url, "pw-box")
    yield daemon
    daemon.stop()
    mach.save_machines([])


# A page 3000px wide: red up to x = 100, green beyond.  Its top third is
# a strip that scrolls sideways on its own, over the same two colours.
_BAND = "width:3000px;background:linear-gradient(90deg,#f00 100px,#0c0 100px)"
_BAND_PAGE = (
    "<!doctype html><html><head><title>Bands</title></head><body style='margin:0'>"
    "<div style='position:absolute;left:0;top:0;width:100vw;height:33vh;overflow-x:auto'>"
    f"<div style='height:100%;{_BAND}'></div></div>"
    f"<div style='height:100vh;{_BAND}'></div></body></html>"
).encode()
# How far down the page the strip and the document under it are touched
# and looked at, as fractions of the page's height.
_STRIP_DOWN = 0.15
_DOCUMENT_DOWN = 0.75

# The colour the streamed picture shows near its left edge, the fraction
# *down* of the way down: red while what is drawn there stands at its
# left edge, green once it is scrolled 100px or more sideways ('' before
# the first frame).
_BAND_JS = """(down) => {
  const img = document.querySelector('.browser-screen img.browser-frame');
  if (!img || !img.naturalWidth) return '';
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const g = canvas.getContext('2d');
  g.drawImage(img, 0, 0);
  const y = Math.round(img.naturalHeight * down);
  const [red, green] = g.getImageData(20, y, 1, 1).data;
  return red > green ? 'red' : 'green';
}"""


def _wait_band(page, down: float, colour: str) -> None:
    """Wait until the streamed picture shows *colour* at its left edge, *down* of the way down."""
    page.wait_for_function(
        f"([down, colour]) => ({_BAND_JS})(down) === colour", arg=[down, colour], timeout=60000,
    )


def _stays_on_the_page(page) -> bool:
    """Whether the browser tab is still on screen after a drag, once
    the swipe can no longer be waiting for the daemon's word on it (the
    view waits TOUCH_ANSWER_MS, 600 ms, for a late answer)."""
    page.wait_for_timeout(800)
    return _settled_screen(page) == "file"


class _BandPage(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(_BAND_PAGE)))
        self.end_headers()
        self.wfile.write(_BAND_PAGE)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def band_page(harness):
    """The daemon's own browser, headless, opening on the band page."""
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _BandPage)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    saved = {k: os.environ.get(k) for k in ("KISS_HEADLESS", "KISS_BROWSER_HOME")}
    os.environ["KISS_HEADLESS"] = "1"
    os.environ["KISS_BROWSER_HOME"] = f"http://127.0.0.1:{httpd.server_address[1]}/"
    yield
    # The browser goes before the display it draws on.
    harness.server._agent_server.browser_tabs.shutdown()
    web_stealth.stop_virtual_display()
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    httpd.shutdown()
    httpd.server_close()


def _bar(page) -> list[str] | None:
    """The labels on the tab bar, or None while the bar is hidden."""
    if not page.locator("#tab-bar").is_visible():
        return None
    labels: list[str] = page.locator("#tab-list .chat-tab .chat-tab-label").all_inner_texts()
    return labels


def _seed_chat(harness, chat_id: str, task: str) -> None:
    """A finished one-task chat in the harness daemon's history."""
    task_id, _ = persistence._add_task(
        task, chat_id, {"work_dir": str(harness.work_dir)},
    )
    persistence._save_task_result("done", task_id=task_id)


def _pick_chat(page, cdp, chat_id: str, task: str, machine: str = "") -> None:
    """Swipe to the Chats screen and pick *chat_id*: its task is on the
    chat screen, under a header that names *machine* ('' = this one)."""
    _swipe_right(cdp)
    _wait_screen(page, "chats")
    page.locator(
        f"#history-list .history-chat-group[data-chat-id='{chat_id}']"
        " .history-chat-header"
    ).click(timeout=30000)
    _wait_screen(page, "chat")
    page.wait_for_function(
        "m => document.getElementById('chat-machine').value === m",
        arg=machine, timeout=30000,
    )
    page.wait_for_selector("#remote-chat-loader", state="detached", timeout=30000)
    assert _chat_on_screen(page)[1] == task


def _chat_on_screen(page) -> tuple[str, str]:
    """The chat screen as the user reads it: the machine the header
    names and the task the transcript opens with."""
    assert page.locator("#chat-machine").is_visible()
    machine = page.locator("#chat-machine option:checked").inner_text()
    task = page.locator("#output .task-panel .task-panel-text").first.inner_text()
    return machine, task


def test_a_lone_file_or_terminal_has_its_bar_and_swipes_back_to_the_chat_left(
    browser, harness, box,
):
    """One file, then one terminal: the file screen's bar holds that one
    tab.  A swipe right, or closing the tab, goes back to the chat the
    file screen was swiped to from, on its machine, not to the chat the
    tab was opened from."""
    _seed_chat(harness, "local-chat", "Local task")
    context, page, cdp = _open_mobile(browser, harness)
    try:
        _pick_chat(page, cdp, "local-chat", "Local task")
        local = _chat_on_screen(page)[0]
        _open_file_from_drawer(page, "feature.txt")
        assert _bar(page) == ["feature.txt"]
        page.click("#more-btn")
        page.click("#terminal-btn")
        page.wait_for_selector(_VIEW + ".terminal-view", timeout=30000)
        assert _bar(page) == ["feature.txt", "Terminal"]
        page.click("#tab-list .chat-tab:not(.active) .chat-tab-close")
        assert _bar(page) == ["Terminal"]
        # Back on the chat the terminal was opened from: no bar for a
        # lone chat.  A draft keeps that chat open behind the next one.
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        assert _bar(page) is None
        assert _chat_on_screen(page) == (local, "Local task")
        page.fill("#task-input", "a draft keeps this chat open")
        _pick_chat(page, cdp, "box-chat", "Box task", machine="box")
        # The terminal of the local chat, looked at from the chat on box...
        _swipe_left(cdp)
        _wait_screen(page, "file")
        assert _bar(page) == ["Terminal"]
        # ...and the swipe back returns to that chat, on box.
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        assert _chat_on_screen(page) == ("box", "Box task")
        # So does closing the terminal from its bar.
        _swipe_left(cdp)
        _wait_screen(page, "file")
        page.click("#tab-list .chat-tab.active .chat-tab-close")
        _wait_screen(page, "chat")
        assert _bar(page) is None
        assert _chat_on_screen(page) == ("box", "Box task")
    finally:
        context.close()


def test_a_lone_browser_tab_has_its_bar_and_swipes_back_from_its_page(
    browser, harness, band_page,
):
    """The daemon's browser streamed into a tab is alone on the file
    screen and still has its bar.  A swipe right on the streamed page
    goes back to the chat swiped from while what is under the finger
    (the document, or a strip of it that scrolls on its own) stands at
    its left edge; scrolled sideways, the finger scrolls it back."""
    _seed_chat(harness, "alpha-chat", "Alpha task")
    _seed_chat(harness, "beta-chat", "Beta task")
    context, page, cdp = _open_mobile(browser, harness)
    try:
        _pick_chat(page, cdp, "alpha-chat", "Alpha task")
        page.fill("#task-input", "a draft keeps this chat open")
        _pick_chat(page, cdp, "beta-chat", "Beta task")
        page.click("#more-btn")
        page.click("#browser-btn")
        _wait_screen(page, "file")
        _wait_band(page, _DOCUMENT_DOWN, "red")
        bar = _bar(page)
        assert bar is not None and len(bar) == 1
        box = page.locator(".browser-screen").bounding_box()
        assert box is not None
        for down in (_DOCUMENT_DOWN, _STRIP_DOWN):
            y = box["y"] + box["height"] * down
            # From its left edge a swipe right leaves for the chat.
            _swipe_right(cdp, y=y)
            _wait_screen(page, "chat")
            assert _chat_on_screen(page)[1] == "Beta task"
            _swipe_left(cdp)
            _wait_screen(page, "file")
            # Leftwards the finger scrolls what is under it and stays...
            _drag(cdp, (_RIGHT_X, y), (_RIGHT_X - 200, y))
            _wait_band(page, down, "green")
            assert _stays_on_the_page(page)
            # ...and rightwards it scrolls that back to its left edge.
            _drag(cdp, (_LEFT_X, y), (_RIGHT_X, y))
            _wait_band(page, down, "red")
            assert _stays_on_the_page(page)
        # From there the next swipe right leaves for the chat again.
        _swipe_right(cdp, y=y)
        _wait_screen(page, "chat")
        assert _chat_on_screen(page)[1] == "Beta task"
    finally:
        context.close()


def _active_tab(page) -> str:
    return str(page.evaluate("window._testApi.getActiveTabId()"))


def _announce_run(harness, page, tab_id: str) -> None:
    """The daemon announces a task running in *tab_id*, and the page has
    heard it.  (A task needs a model to run; the announcement is the
    daemon's own ``status`` broadcast.)"""
    harness.server._printer.broadcast(
        {"type": "status", "running": True, "tabId": tab_id},
    )
    page.wait_for_function(
        "id => window._testApi.openTabs().some(t => t.id === id && t.isRunning)",
        arg=tab_id, timeout=15000,
    )


def test_stop_and_save_chat_on_the_file_screen_act_on_the_chat_swiped_from(
    browser, harness,
):
    """A file opened from one chat and looked at from another: Stop and
    "Save chat" on the file screen act on the chat the screen was swiped
    to from, the one a swipe right returns to, and not on the chat the
    file was opened from."""
    _seed_chat(harness, "owner-chat", "Owner task")
    _seed_chat(harness, "visitor-chat", "Visitor task")
    context, page, sent = _open_page_mobile(browser, harness)
    _dismiss_toasts(page)
    cdp = _touch_session(page)
    stop = page.locator("#stop-btn")
    try:
        _pick_chat(page, cdp, "owner-chat", "Owner task")
        owner = _active_tab(page)
        _open_file_from_drawer(page, "feature.txt")
        _swipe_right(cdp)
        _wait_screen(page, "chat")
        page.fill("#task-input", "a draft keeps this chat open")
        _pick_chat(page, cdp, "visitor-chat", "Visitor task")
        visitor = _active_tab(page)
        assert visitor != owner
        # The owner's file, looked at from the visitor chat while only
        # the owner's task runs: there is nothing here to stop.
        _announce_run(harness, page, owner)
        _swipe_left(cdp)
        _wait_screen(page, "file")
        assert _bar(page) == ["feature.txt"]
        assert not stop.is_visible()
        # The visitor's task starts: Stop shows, and stops that task.
        _announce_run(harness, page, visitor)
        stop.wait_for(state="visible", timeout=15000)
        assert _screen(page) == "file"
        stop.click()
        # The daemon answers that nothing runs in the tab it was asked
        # to stop, and the button goes with that tab's run.
        stop.wait_for(state="hidden", timeout=15000)
        assert [f["tabId"] for f in sent if f.get("type") == "stop"] == [visitor]
        # "Save chat" brings the visitor chat back and saves its page.
        assert _screen(page) == "file"
        page.click("#more-btn")
        page.click("#share-btn")
        _wait_screen(page, "chat")
        assert _chat_on_screen(page)[1] == "Visitor task"
        page.locator("#output").get_by_text("Chat page saved to").wait_for(timeout=30000)
        reports = Path(harness.work_dir) / "reports"
        saved = sorted(p.name for p in reports.glob("chat-*.html"))
        assert saved == ["chat-visitor-task-visitor-chat.html"]
        assert "Visitor task" in (reports / saved[0]).read_text()
    finally:
        context.close()

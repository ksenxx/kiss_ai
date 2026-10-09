# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from
#   kiss.tests.server.test_explorer_scm_commands /
#   kiss.tests.agents.vscode.test_workspace_sections are re-exported
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

The swipes are real touch sequences dispatched through the Chrome
DevTools protocol (``Input.dispatchTouchEvent``), so Chromium produces
the same touch events and native scrolling a finger would.
"""

from __future__ import annotations

import pytest

from kiss.tests.agents.vscode.test_workspace_sections import (
    _dismiss_update_toast,
    _explorer_row,
    _open_page,
    _open_page_mobile,
    _show_section,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
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


def _swipe_left(cdp, y: float = _Y) -> None:
    """The finger moves right to left: the next screen."""
    _drag(cdp, (_RIGHT_X, y), (_LEFT_X, y))


def _swipe_right(cdp, y: float = _Y) -> None:
    """The finger moves left to right: the previous screen."""
    _drag(cdp, (_LEFT_X, y), (_RIGHT_X, y))


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
    _dismiss_update_toast(page, harness)
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
        _drag(cdp, (box["x"] + 250, y), (box["x"] + 20, y), steps=12)
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
        # it is a swipe.
        _drag(cdp, (before + 300, y), (before + 60, y), steps=12)
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

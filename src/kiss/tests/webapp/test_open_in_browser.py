# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` / `browser` module fixtures are
#   imported from sibling test modules and intentionally shadowed by
#   test parameters of the same name)
"""End-to-end tests: the "Open in Browser" button of an ``.html`` content tab.

The daemon of these tests starts no browser of its own
(``KISS_HEADLESS``), as the daemon behind a remote page cannot: the
page uploads the document its preview shows (``stageHtmlView``) and
opens ``/view/<token>/<name>`` in a new browser tab, which the daemon
answers sandboxed into an opaque origin.  One test gives the daemon a
``$BROWSER`` that fails to start, slowly: the page must then ask for a
second click rather than open a tab the browser would block.

These tests drive a REAL browser (Playwright Chromium) against a REAL
:class:`RemoteAccessServer` over real ``wss://`` and ``https://`` — no
mocks.
"""

from __future__ import annotations

import base64
import contextlib
import os
import re
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_editing import _dismiss_toasts, _uncover_pane
from kiss.tests.webapp.test_content_tab_file_links import (
    _inject_file_link,
    browser,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_result_ls_listing_file_links import _open_page

_VIEW = "#content-tab-area .content-tab-view:not([style*='display: none']) "
_BAR = _VIEW + ".content-save-bar"
_BUTTON = _BAR + " .content-browser-btn"
_CLICK_AGAIN = "Open in Browser: the browser did not start, click again"
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)
# A picture beside the file, a script of its own, and more than the
# kilobyte from which the daemon compresses an answer.
_DOCUMENT = (
    "<!DOCTYPE html><html><head><title>Report</title></head><body>"
    "<h1 id='marker'>KISS-REPORT</h1><img src='dot.png'>"
    "<p>" + "filler " * 300 + "</p>"
    "<script>document.getElementById('marker').dataset.origin = window.origin;</script>"
    "</body></html>"
)
_STORAGE_JS = "() => { try { return String(localStorage.length); } catch (e) { return e.name; } }"
# The type of the daemon's answer to a sign-in as a local page makes it.
_SIGN_IN_JS = """url => new Promise(resolve => {
  const ws = new WebSocket(url);
  ws.onopen = () => ws.send(JSON.stringify({type: 'auth', password: ''}));
  ws.onmessage = ev => resolve(JSON.parse(ev.data).type);
  ws.onclose = () => resolve('closed');
})"""


@contextlib.contextmanager
def _environment(**values: str) -> Iterator[None]:
    """Set the environment variables *values* for the block; the daemon reads them as it goes."""
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def report(harness) -> Iterator[str]:
    """The path of an HTML report with a picture, on a daemon that starts no browser."""
    (harness.work_dir / "dot.png").write_bytes(_PNG)
    path = harness.work_dir / "report 1.html"
    path.write_text(_DOCUMENT, encoding="utf-8", newline="\n")
    with _environment(KISS_HEADLESS="1"):
        yield str(path)


def _commands(sent: list[dict]) -> list[str]:
    """The "Open in Browser" commands among the frames the page *sent*, in order."""
    return [
        frame["type"] for frame in sent
        if frame.get("type") in ("openInBrowser", "stageHtmlView")
    ]


def test_the_button_opens_the_document_in_a_sandboxed_browser_tab(
    browser, harness, report: str,
) -> None:
    context, page, sent = _open_page(browser, harness)
    try:
        _inject_file_link(page, report, "lnk-report")
        page.click("#lnk-report")
        page.wait_for_selector(_BUTTON, state="visible", timeout=30000)
        assert page.locator(_BAR + " button").all_inner_texts()[:3] == [
            "Edit source", "Open in Browser", "Save",
        ]
        _uncover_pane(page)
        _dismiss_toasts(page)
        with context.expect_page() as opened:
            page.click(_BUTTON)
        view = opened.value
        marker = view.locator("#marker")
        marker.wait_for(timeout=30000)
        assert re.fullmatch(
            re.escape(harness.base_url) + r"/view/[0-9a-f]{32}/report%201\.html", view.url,
        )
        assert marker.inner_text() == "KISS-REPORT"
        # The document's own script ran, in an origin of its own: the
        # storage of the chat page, which the same host serves, is out
        # of its reach.
        assert marker.get_attribute("data-origin") == "null"
        assert view.evaluate("window.origin") == "null"
        assert view.evaluate(_STORAGE_JS) == "SecurityError"
        assert page.evaluate(_STORAGE_JS).isdigit()
        # Nor can it sign in to the daemon, as the chat page does.
        assert page.evaluate(_SIGN_IN_JS, harness.ws_url) == "auth_ok"
        assert view.evaluate(_SIGN_IN_JS, harness.ws_url) == "error"
        # Its picture came inlined, and the preview's own scripts did not.
        assert view.evaluate(
            "(() => { const img = document.querySelector('img');"
            " return img.src.startsWith('data:image/png;base64,') && img.naturalWidth; })()"
        ) == 1
        assert view.evaluate("document.scripts.length") == 1
        staged = [frame for frame in sent if frame.get("type") == "stageHtmlView"]
        assert [frame["name"] for frame in staged] == ["report 1.html"]
        assert staged[0]["token"] in view.url
        assert _commands(sent) == ["stageHtmlView"]
    finally:
        context.close()


@posix_only("POSIX $BROWSER script")
def test_a_browser_that_fails_to_start_slowly_asks_for_a_second_click(
    browser, harness, report: str, tmp_path: Path,
) -> None:
    """The daemon's browser fails after the click may still open a tab.

    A browser lets a click open a tab for five seconds.  The two
    ``$BROWSER`` launchers of this daemon fail after 2.7 seconds each,
    so the tab the page falls back to would be blocked without a word:
    the bar asks for a second click, and that click opens the tab.
    """
    launcher = tmp_path / "browser.sh"
    launcher.write_text("#!/bin/sh\nsleep 2.7\nexit 1\n", encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    with _environment(KISS_HEADLESS="0", BROWSER=f"{launcher}{os.pathsep}{launcher}"):
        context, page, sent = _open_page(browser, harness)
        try:
            _inject_file_link(page, report, "lnk-slow")
            page.click("#lnk-slow")
            page.wait_for_selector(_BUTTON, state="visible", timeout=30000)
            _uncover_pane(page)
            _dismiss_toasts(page)
            page.click(_BUTTON)
            page.wait_for_function(
                "([sel, text]) => document.querySelector(sel).textContent === text",
                arg=[_BAR + " .content-save-status", _CLICK_AGAIN],
                timeout=30000,
            )
            assert len(context.pages) == 1, "no tab is opened once the click is spent"
            assert _commands(sent) == ["openInBrowser"]
            with context.expect_page() as opened:
                page.click(_BUTTON)
            opened.value.locator("#marker").wait_for(timeout=30000)
            assert _commands(sent) == ["openInBrowser", "stageHtmlView"]
        finally:
            context.close()


def test_the_bar_of_an_html_tab_fits_a_phone(browser, harness, report: str) -> None:
    context, page, _sent = _open_page(browser, harness)
    try:
        page.set_viewport_size({"width": 360, "height": 800})
        page.wait_for_function(
            "() => !document.body.classList.contains('remote-desktop')", timeout=10000,
        )
        _inject_file_link(page, report, "lnk-phone")
        page.click("#lnk-phone")
        page.wait_for_selector(_BUTTON, state="visible", timeout=30000)
        # One button more than a phone is wide: the bar takes a second
        # row, and no item leaves the bar's padding box (on one row the
        # reload button did, towards the edge the pane clips at).
        outside = page.evaluate(
            """sel => {
                 const bar = document.querySelector(sel);
                 const box = bar.getBoundingClientRect();
                 const style = getComputedStyle(bar);
                 const left = box.left + parseFloat(style.paddingLeft) - 0.5;
                 const right = box.right - parseFloat(style.paddingRight) + 0.5;
                 return [...bar.children]
                   .map(el => [el.className, el.getBoundingClientRect()])
                   .filter(([, r]) => r.left < left || r.right > right)
                   .map(([name]) => name);
               }""",
            _BAR,
        )
        assert outside == []
        for name in ("mode", "browser", "save", "refresh"):
            assert page.locator(f"{_BAR} .content-{name}-btn").is_visible()
    finally:
        context.close()

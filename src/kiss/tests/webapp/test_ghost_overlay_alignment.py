# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: the composer's ghost suggestion starts where the typed text ends.

The ghost text is drawn by ``#ghost-overlay`` (``media/chat.html``), a
box laid over the ``#task-input`` textarea that repeats the typed text
invisibly and appends the suggestion in grey (``updateGhost`` in
``media/main.js``).  The suggestion follows the typed text only while
the overlay lays its text out exactly like the textarea.

Regression: the overlay inherited the page's ``letter-spacing:
-0.08px`` (``body`` in ``media/main.css``) while browsers reset a
textarea's to ``normal``, so the overlay's copy of a long line was
narrower than the typed line and the suggestion was drawn over its
last letters.

The production RemoteAccessServer serves the page to a real headless
Chromium.  The test types into the composer, paints a suggestion
through the page's own ``ghost`` message handler and compares where
the suggestion starts with where the textarea's caret sits after the
typed text.  The caret is measured on a hidden div given every
text-layout property of the textarea (the usual way to find a
textarea's caret coordinates): on one long line at desktop width, and
on wrapped lines holding a long path at phone width.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, sync_playwright

from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.test_codex_task_panel_style import _start_live_server

_ONE_LINE = (
    "Refactor the session manager so every reconnect replays the pending "
    "events in the order they arrived"
)
_WRAPPED = (
    "Look at src/kiss/webapp/media/composer_ghost_overlay_alignment_helpers.js "
    "and fix the overlap between the typed text and the ghost suggestion"
)
_SUGGESTION = " and add a regression test"

_TYPE_AND_MEASURE_JS = """
({text, suggestion}) => {
  const overlay = document.getElementById('kiss-server-loading');
  if (overlay) overlay.style.display = 'none';
  document.getElementById('app').style.display = '';
  const inp = document.getElementById('task-input');
  inp.focus();
  inp.value = text;
  inp.setSelectionRange(text.length, text.length);
  inp.dispatchEvent(new Event('input', {bubbles: true}));
  // The daemon's ghost reply, through the page's own handler.
  const data = {type: 'ghost', suggestion, query: text};
  const tabId = window._testApi.getActiveTabId();
  if (tabId) data.tabId = tabId;
  window.dispatchEvent(new MessageEvent('message', {data}));
  const ghost = document.querySelector('#ghost-overlay .ghost-text');
  if (!ghost) return null;

  // Where the textarea puts the text after the typed text: a div with
  // the textarea's text layout, laid exactly over it.
  const MIRRORED = [
    'box-sizing', 'padding-top', 'padding-right', 'padding-bottom',
    'padding-left', 'border-top-width', 'border-right-width',
    'border-bottom-width', 'border-left-width', 'border-top-style',
    'border-right-style', 'border-bottom-style', 'border-left-style',
    'font-family', 'font-size', 'font-style', 'font-variant', 'font-weight',
    'font-stretch', 'font-kerning', 'font-feature-settings',
    'font-variation-settings', 'line-height', 'letter-spacing',
    'word-spacing', 'text-transform', 'text-indent', 'text-align',
    'text-rendering', 'tab-size', 'white-space', 'overflow-wrap',
    'word-break', 'direction',
  ];
  const cs = getComputedStyle(inp);
  const mirror = document.createElement('div');
  for (const p of MIRRORED) mirror.style.setProperty(p, cs.getPropertyValue(p));
  const box = inp.getBoundingClientRect();
  const borderX = parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth);
  Object.assign(mirror.style, {
    position: 'fixed', margin: '0', visibility: 'hidden',
    left: box.left + 'px', top: box.top - inp.scrollTop + 'px',
    // The box the textarea wraps its text in: its client width (no
    // scrollbar) plus its borders.
    boxSizing: 'border-box', width: inp.clientWidth + borderX + 'px',
  });
  mirror.textContent = text;
  const end = document.createElement('span');
  end.textContent = suggestion;
  mirror.appendChild(end);
  document.body.appendChild(mirror);
  const caret = end.getClientRects()[0];
  const first = mirror.firstChild;
  const range = document.createRange();
  range.setStart(first, 0);
  range.setEnd(first, 1);
  const firstTop = range.getClientRects()[0].top;
  mirror.remove();
  const at = ghost.getClientRects()[0];
  const ghostStyle = getComputedStyle(ghost.parentElement);
  return {
    ghost: {left: at.left, top: at.top},
    caret: {left: caret.left, top: caret.top},
    lines: Math.round((caret.top - firstTop) / parseFloat(cs.lineHeight)) + 1,
    differing: MIRRORED.filter(
      p => ghostStyle.getPropertyValue(p) !== cs.getPropertyValue(p)),
  };
}
"""


def _measure(browser: Browser, port: object, width: int, text: str) -> dict[str, Any]:
    """Type *text* on a page *width* pixels wide and measure ghost and caret."""
    page = browser.new_page(ignore_https_errors=True, viewport={"width": width, "height": 900})
    try:
        goto_retrying_network_change(page, f"https://127.0.0.1:{port}/", wait_until="load")
        page.wait_for_selector("#task-input", state="attached")
        measured: dict[str, Any] | None = page.evaluate(
            _TYPE_AND_MEASURE_JS, {"text": text, "suggestion": _SUGGESTION},
        )
    finally:
        page.close()
    assert measured is not None, "the page painted no ghost suggestion"
    return measured


@pytest.mark.timeout(180)
def test_ghost_suggestion_starts_where_the_typed_text_ends(tmp_path: Path) -> None:
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server, args=(tmp_path, ready, done, state), daemon=True,
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError("RemoteAccessServer startup failed") from startup_error
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                desktop = _measure(browser, state["port"], 1280, _ONE_LINE)
                phone = _measure(browser, state["port"], 390, _WRAPPED)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"
    assert desktop["lines"] == 1, desktop
    assert phone["lines"] >= 3, phone
    for measured in (desktop, phone):
        assert abs(measured["ghost"]["left"] - measured["caret"]["left"]) <= 0.5, measured
        assert abs(measured["ghost"]["top"] - measured["caret"]["top"]) <= 0.5, measured

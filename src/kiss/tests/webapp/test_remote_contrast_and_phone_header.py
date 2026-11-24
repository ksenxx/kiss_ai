# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Served page + real Chromium: the High contrast palettes and the phone chat header.

* The Chats strip's High contrast button paints the dark palette at
  full contrast on the dark theme (black canvas, white text, white
  primary button) and the light palette at full contrast on the light
  theme (white canvas, black text, black primary button); a second
  press restores the standard palettes (``#fafafa`` canvas with a
  ``#191919`` button, ``#0b0b0b`` canvas with an ``#efefef`` button),
  and the button keeps its name.  The page inlines the palette block
  once.
* In each of the four palettes the coloured text of a tool card (a
  diff's changed words, the path link, the argument text and its faint
  label) and the faint label of a side-panel section keep 4.5:1
  against the fill painted behind them.
* A failure keeps its red: the label of a failed ``/ask`` answer is
  the failure colour, and a failed result inside its tool call has the
  wash of a failed result that stands alone.
* At a phone width both chat-header drop-downs (machine, working
  directory) share ONE row whatever their lengths: a machine name that
  fits in its half of the row is shown whole and the directory takes
  the rest, two long names halve the row, and what does not fit is cut
  with an ellipsis: a machine name loses its end, a directory its
  beginning (the last folder stays in view).  Each name's down chevron
  follows it directly.
"""

from __future__ import annotations

import io
import threading
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import Page, sync_playwright

from kiss.server.web_server import _HIGH_CONTRAST_VARS_CSS, _build_html
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.test_codex_task_panel_style import _start_live_server
from kiss.tests.webapp.test_content_pane_drawer import (
    _CHEVRONS_JS,
    _assert_chevrons_follow_the_names,
)

_HIDE_AUTH_JS = "() => { document.getElementById('auth-modal').style.display = 'none'; }"

_PROBE_JS = """() => {
  const cs = (sel, prop) => getComputedStyle(document.querySelector(sel))[prop];
  return {
    light: document.body.classList.contains('light-theme'),
    high: document.body.classList.contains('high-contrast'),
    bodyBg: cs('body', 'backgroundColor'),
    bodyFg: cs('body', 'color'),
    send: cs('#send-btn', 'backgroundColor'),
    contrastLabel: document.getElementById('contrast-btn').getAttribute('aria-label'),
    themeLabel: document.getElementById('theme-btn').getAttribute('aria-label'),
  };
}"""

_HEADER_JS = """() => {
  const header = document.getElementById('chat-header');
  const h = header.getBoundingClientRect();
  // Whether the characters [from, to) of the label's text lie inside
  // the label's box: a cut label clips the others behind its ellipsis.
  const inside = (label, from, to) => {
    if (!label.firstChild) return false;
    const range = document.createRange();
    range.setStart(label.firstChild, from);
    range.setEnd(label.firstChild, to);
    const r = range.getBoundingClientRect();
    const l = label.getBoundingClientRect();
    return r.width > 0 && r.left >= l.left - 1 && r.right <= l.right + 1;
  };
  const box = id => {
    const select = document.getElementById(id);
    const label = select.previousElementSibling;
    const r = select.getBoundingClientRect();
    const n = label.textContent.length;
    return {left: r.left, right: r.right, top: r.top, width: r.width,
            shown: label.textContent, cut: label.scrollWidth > label.clientWidth,
            direction: getComputedStyle(label).direction,
            startShown: inside(label, 0, Math.min(n, 6)),
            endShown: inside(label, Math.max(0, n - 6), n)};
  };
  return {
    viewport: window.innerWidth,
    header: {top: h.top, height: h.height, width: h.width},
    machine: box('chat-machine'),
    workdir: box('chat-workdir'),
  };
}"""

# Names the machine and a long working directory the way the daemon
# does (a ``configData`` frame) and measures the chat header in the same
# task.  The page repaints both drop-downs when a daemon reply such as
# ``machines`` arrives, so a measurement taken by a later ``evaluate``
# can see the daemon's own names instead of these.
_LONG_DIR = "/home/someone/projects/a-very-long-directory-name/deeper"
_PHONE_HEADER_JS = (
    """machine => {
  const dir = '"""
    + _LONG_DIR
    + """';
  window.dispatchEvent(new MessageEvent('message', {data: {
    type: 'configData', machine,
    config: {work_dir: dir, recent_work_dirs: [{path: dir, ts: 1}]}}}));
  return ("""
    + _HEADER_JS
    + """)();
}"""
)


# One Bash call and one Edit call with their cards opened (the coloured
# text of a tool card, each kind on its own fill), then the failures: a
# Read call whose result failed, an error that stands alone and a
# failed /ask answer.
_TOOL_CARDS_JS = """() => {
  const add = ev => window._testApi.processEvent(ev);
  add({type: 'tool_call', name: 'Bash', command: 'ls', description: 'List the files'});
  add({type: 'tool_result', content: 'main.css', is_error: false});
  add({type: 'tool_call', name: 'Edit', path: 'media/main.css',
       old_string: 'border-radius: 6px;', new_string: 'border-radius: var(--radius-md);'});
  add({type: 'tool_call', name: 'Read', path: 'media/missing.css'});
  add({type: 'tool_result', content: 'ENOENT: no such file', is_error: true});
  add({type: 'error', text: 'boom'});
  add({type: 'ask_answer', question: 'Why?', text: 'No answer.', success: false});
  document.querySelectorAll('.ev.tc.collapsed > .tc-h').forEach(h => h.click());
}"""
# The text pinned at 4.5:1: a tool card's coloured text, the dim
# argument text and its faint label on the card's sunken fill, and the
# faint label of a side-panel section.
_PINNED_TEXT = (
    ".diff-hl-del",
    ".diff-hl-add",
    ".tc-b .tp",
    ".tc-arg",
    ".tc-arg-name",
    "#meta-panel .meta-section-toggle > span",
)

# The colours of the failures: the failure colour itself, the label of
# the failed answer, and the fill of a failed result inside its tool
# call and of one that stands alone.
_FAILURE_COLOURS_JS = """() => {
  const style = selector => getComputedStyle(document.querySelector(selector));
  const probe = document.createElement('span');
  probe.style.color = 'var(--status-fail)';
  document.body.appendChild(probe);
  const statusFail = getComputedStyle(probe).color;
  probe.remove();
  return {
    statusFail,
    answerLabel: style('.ask-answer.failed .ask-answer-label').color,
    resultInCall: style('.tc > .tr.err').backgroundColor,
    resultAlone: style('#output > .tr.err').backgroundColor,
  };
}"""

# WCAG contrast of the selector's text against the [r, g, b] fill.  The
# text colour is painted over the fill, so a translucent one blends.
_CONTRAST_JS = """([selector, fill]) => {
  const ctx = document.createElement('canvas').getContext('2d');
  const luminance = colour => {
    ctx.fillStyle = colour;
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b] = Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3), c =>
      (c / 255 <= 0.04045 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const text = getComputedStyle(document.querySelector(selector)).color;
  const [lighter, darker] = ['rgb(' + fill + ')', text].map(luminance).sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
}"""


def _low_contrast_text(page: Page) -> dict[str, float]:
    """The ``_PINNED_TEXT`` elements whose text is under 4.5:1 against the
    fill behind it (the most common pixel of the element), with the ratio.
    """
    low = {}
    for selector in _PINNED_TEXT:
        png = page.locator(selector).first.screenshot(animations="disabled")
        raw = Image.open(io.BytesIO(png)).convert("RGB").tobytes()
        fill = Counter(raw[i : i + 3] for i in range(0, len(raw), 3)).most_common(1)[0][0]
        ratio = page.evaluate(_CONTRAST_JS, [selector, list(fill)])
        if ratio < 4.5:
            low[selector] = round(ratio, 2)
    return low


def test_page_inlines_the_high_contrast_palettes_once() -> None:
    page = _build_html()
    assert page.count(_HIGH_CONTRAST_VARS_CSS) == 1
    assert "body.remote-chat.high-contrast {" in page
    assert "body.remote-chat.light-theme.high-contrast {" in page
    # A black canvas is the dark high-contrast palette's alone, a black
    # primary button the light one's.
    assert page.count("--theme-editor-background: #000000;") == 1
    assert page.count("--theme-button-background: #000000;") == 1


def _press_strip_button(page, button_id: str) -> None:
    page.evaluate(f"() => document.getElementById('{button_id}').click()")


@pytest.mark.timeout(180)
def test_live_contrast_toggle_and_phone_header(tmp_path: Path) -> None:
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server, args=(tmp_path, ready, done, state), daemon=True
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError("RemoteAccessServer startup failed") from startup_error
        port = state["port"]
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                page = browser.new_page(
                    ignore_https_errors=True, viewport={"width": 1400, "height": 900}
                )
                goto_retrying_network_change(page, f"https://127.0.0.1:{port}/", wait_until="load")
                page.wait_for_selector("#output", state="attached")
                page.evaluate(_HIDE_AUTH_JS)
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(25, 25, 25)'",
                    timeout=10000,
                )
                low_light = page.evaluate(_PROBE_JS)
                page.evaluate(_TOOL_CARDS_JS)
                failures = page.evaluate(_FAILURE_COLOURS_JS)
                low_text = {"light": _low_contrast_text(page)}
                _press_strip_button(page, "contrast-btn")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(0, 0, 0)'",
                    timeout=10000,
                )
                high_light = page.evaluate(_PROBE_JS)
                low_text["high-contrast light"] = _low_contrast_text(page)
                _press_strip_button(page, "theme-btn")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(255, 255, 255)'",
                    timeout=10000,
                )
                high_dark = page.evaluate(_PROBE_JS)
                low_text["high-contrast dark"] = _low_contrast_text(page)
                _press_strip_button(page, "contrast-btn")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(239, 239, 239)'",
                    timeout=10000,
                )
                low_dark = page.evaluate(_PROBE_JS)
                low_text["dark"] = _low_contrast_text(page)
                stored = page.evaluate(
                    "() => [localStorage.getItem('kissRemoteTheme'),"
                    " localStorage.getItem('kissRemoteContrast')]"
                )

                # The phone layout: the machine name and the directory
                # share one row, a short name as much as a long one.
                page.set_viewport_size({"width": 390, "height": 844})
                short_name = page.evaluate(_PHONE_HEADER_JS, "koushik-sorcar")
                phone_chevrons = page.evaluate(_CHEVRONS_JS)
                long_name = page.evaluate(
                    _PHONE_HEADER_JS, "a-machine-with-a-very-long-registered-name-indeed"
                )
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    assert low_light["light"] and not low_light["high"], low_light
    assert low_light["contrastLabel"] == "High contrast", low_light
    assert low_light["themeLabel"] == "Dark mode", low_light
    # The standard light palette: #fafafa canvas, #0b0b0b text, #191919 button.
    assert low_light["bodyBg"] == "rgb(250, 250, 250)", low_light
    assert low_light["bodyFg"] == "rgb(11, 11, 11)", low_light
    assert low_light["send"] == "rgb(25, 25, 25)", low_light

    # Light high contrast: white canvas, black text, black button.
    assert high_light["light"] and high_light["high"], high_light
    assert high_light["contrastLabel"] == "High contrast", high_light
    assert high_light["bodyBg"] == "rgb(255, 255, 255)", high_light
    assert high_light["bodyFg"] == "rgb(0, 0, 0)", high_light
    assert high_light["send"] == "rgb(0, 0, 0)", high_light

    # Dark high contrast: black canvas, white text, white button.
    assert not high_dark["light"] and high_dark["high"], high_dark
    assert high_dark["themeLabel"] == "Light mode", high_dark
    assert high_dark["bodyBg"] == "rgb(0, 0, 0)", high_dark
    assert high_dark["bodyFg"] == "rgb(255, 255, 255)", high_dark
    assert high_dark["send"] == "rgb(255, 255, 255)", high_dark

    # Low contrast again restores the standard dark palette: #0b0b0b
    # canvas, #f0efec text, #efefef button.
    assert not low_dark["light"] and not low_dark["high"], low_dark
    assert low_dark["bodyBg"] == "rgb(11, 11, 11)", low_dark
    assert low_dark["bodyFg"] == "rgb(240, 239, 236)", low_dark
    assert low_dark["send"] == "rgb(239, 239, 239)", low_dark
    assert stored == ["dark", "low"], stored
    assert not any(low_text.values()), f"text under 4.5:1: {low_text}"

    # The failed answer's label is the failure red, not the body colour;
    # the failed result keeps its wash inside the tool call's card.
    assert failures["answerLabel"] == failures["statusFail"] != low_light["bodyFg"], failures
    assert failures["resultInCall"] == failures["resultAlone"] != "rgba(0, 0, 0, 0)", failures

    _assert_chevrons_follow_the_names(phone_chevrons)
    # One row inside the viewport: the machine, then the directory, the
    # long directory cut with an ellipsis at its beginning: its label (the
    # path between two left-to-right marks, which keep the leading "/" in
    # front) runs right to left, so the last folder stays in view.
    for shot in (short_name, long_name):
        assert shot["viewport"] == 390, shot
        assert shot["machine"]["width"] > 40, shot
        assert shot["workdir"]["width"] > 100, shot
        assert shot["machine"]["left"] >= 0 and shot["workdir"]["right"] <= 390, shot
        assert shot["machine"]["right"] < shot["workdir"]["left"], shot
        assert shot["machine"]["top"] == shot["workdir"]["top"], shot
        workdir = shot["workdir"]
        assert workdir["shown"] == f"\u200e{_LONG_DIR}\u200e" and workdir["cut"], shot
        assert workdir["direction"] == "rtl", shot
        assert workdir["endShown"] and not workdir["startShown"], shot
        assert shot["machine"]["direction"] == "ltr", shot
    # A name that fits in its half of the row is shown whole; a long one
    # is cut too (at its end) and the two halve the row, which stays one
    # line tall.
    assert short_name["machine"]["shown"] == "koushik-sorcar", short_name
    assert not short_name["machine"]["cut"], short_name
    assert short_name["machine"]["startShown"] and short_name["machine"]["endShown"], short_name
    assert long_name["machine"]["cut"], long_name
    assert long_name["machine"]["startShown"] and not long_name["machine"]["endShown"], long_name
    assert long_name["machine"]["width"] == pytest.approx(
        long_name["workdir"]["width"], abs=1
    ), long_name
    assert long_name["header"]["height"] == short_name["header"]["height"], (
        short_name,
        long_name,
    )

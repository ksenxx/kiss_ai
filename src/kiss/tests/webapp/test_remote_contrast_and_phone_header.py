# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Served page + real Chromium: the High contrast palettes and the phone chat header.

* The ``...`` menu's High contrast item paints Cursor's Dark High
  Contrast palette on the dark theme (``#0a0a0a`` canvas, ``#f0f0f0``
  text, ``#434c5e`` button) and the light high-contrast palette on the
  light theme (white canvas, black text, ``#0f4a85`` button); Low
  contrast restores the Modern palettes.  The page inlines the
  palette block once.
* At a phone width both chat-header drop-downs (machine, working
  directory) stay on screen whatever their lengths: a long machine
  name pushes the directory onto a second line instead of squeezing it
  to its chevron.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.server.web_server import _HIGH_CONTRAST_VARS_CSS, _build_html
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.test_codex_task_panel_style import _start_live_server

_HIDE_AUTH_JS = "() => { document.getElementById('auth-modal').style.display = 'none'; }"

_PROBE_JS = """() => {
  const cs = (sel, prop) => getComputedStyle(document.querySelector(sel))[prop];
  return {
    light: document.body.classList.contains('light-theme'),
    high: document.body.classList.contains('high-contrast'),
    bodyBg: cs('body', 'backgroundColor'),
    bodyFg: cs('body', 'color'),
    send: cs('#send-btn', 'backgroundColor'),
    contrastLabel: document.querySelector('#contrast-btn .more-item-label').textContent,
    themeLabel: document.querySelector('#theme-btn .more-item-label').textContent,
  };
}"""

_HEADER_JS = """() => {
  const header = document.getElementById('chat-header');
  const h = header.getBoundingClientRect();
  const box = id => {
    const r = document.getElementById(id).getBoundingClientRect();
    return {left: r.left, right: r.right, top: r.top, width: r.width};
  };
  return {
    viewport: window.innerWidth,
    header: {top: h.top, height: h.height, width: h.width},
    machine: box('chat-machine'),
    workdir: box('chat-workdir'),
  };
}"""


def test_page_inlines_the_high_contrast_palettes_once() -> None:
    page = _build_html()
    assert page.count(_HIGH_CONTRAST_VARS_CSS) == 1
    assert "body.remote-chat.high-contrast {" in page
    assert "body.remote-chat.light-theme.high-contrast {" in page
    assert "--theme-editor-background: #0a0a0a;" in page
    assert "--theme-button-background: #0f4a85;" in page


def _click_menu_item(page, item_id: str) -> None:
    page.evaluate(f"() => document.getElementById('{item_id}').click()")


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
                    ".backgroundColor === 'rgb(0, 95, 184)'",
                    timeout=10000,
                )
                low_light = page.evaluate(_PROBE_JS)
                _click_menu_item(page, "contrast-btn")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(15, 74, 133)'",
                    timeout=10000,
                )
                high_light = page.evaluate(_PROBE_JS)
                _click_menu_item(page, "theme-btn")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(67, 76, 94)'",
                    timeout=10000,
                )
                high_dark = page.evaluate(_PROBE_JS)
                _click_menu_item(page, "contrast-btn")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(0, 120, 212)'",
                    timeout=10000,
                )
                low_dark = page.evaluate(_PROBE_JS)
                stored = page.evaluate(
                    "() => [localStorage.getItem('kissRemoteTheme'),"
                    " localStorage.getItem('kissRemoteContrast')]"
                )

                # The phone layout: a short machine name shares the row
                # with the directory; a long one sends the directory to
                # a second line.  Both stay inside the viewport.
                page.set_viewport_size({"width": 390, "height": 844})
                page.evaluate(
                    "() => {"
                    " const m = document.getElementById('chat-machine');"
                    " m.textContent = '';"
                    " const o = document.createElement('option');"
                    " o.textContent = 'koushik-sorcar'; m.appendChild(o);"
                    " const w = document.getElementById('chat-workdir');"
                    " w.textContent = '';"
                    " const d = document.createElement('option');"
                    " d.textContent = '/home/someone/projects/a-very-long-directory-name/deeper';"
                    " w.appendChild(d);"
                    " document.getElementById('chat-header-sep').textContent = ': ';"
                    "}"
                )
                short_name = page.evaluate(_HEADER_JS)
                page.evaluate(
                    "() => { document.querySelector('#chat-machine option').textContent ="
                    " 'a-machine-with-a-very-long-registered-name-indeed'; }"
                )
                long_name = page.evaluate(_HEADER_JS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    assert low_light["light"] and not low_light["high"], low_light
    assert low_light["contrastLabel"] == "High contrast", low_light
    assert low_light["themeLabel"] == "Dark mode", low_light
    assert low_light["bodyBg"] == "rgb(255, 255, 255)", low_light
    assert low_light["bodyFg"] == "rgb(59, 59, 59)", low_light

    # Light high contrast: white canvas, black text, #0f4a85 button.
    assert high_light["light"] and high_light["high"], high_light
    assert high_light["contrastLabel"] == "Low contrast", high_light
    assert high_light["bodyBg"] == "rgb(255, 255, 255)", high_light
    assert high_light["bodyFg"] == "rgb(0, 0, 0)", high_light
    assert high_light["send"] == "rgb(15, 74, 133)", high_light

    # Cursor Dark High Contrast: #0a0a0a canvas, #f0f0f0 text, #434c5e button.
    assert not high_dark["light"] and high_dark["high"], high_dark
    assert high_dark["themeLabel"] == "Light mode", high_dark
    assert high_dark["bodyBg"] == "rgb(10, 10, 10)", high_dark
    assert high_dark["bodyFg"] == "rgb(240, 240, 240)", high_dark
    assert high_dark["send"] == "rgb(67, 76, 94)", high_dark

    # Low contrast again restores Dark Modern.
    assert not low_dark["light"] and not low_dark["high"], low_dark
    assert low_dark["bodyBg"] == "rgb(31, 31, 31)", low_dark
    assert low_dark["send"] == "rgb(0, 120, 212)", low_dark
    assert stored == ["dark", "low"], stored

    for shot in (short_name, long_name):
        assert shot["viewport"] == 390, shot
        assert shot["machine"]["width"] > 40, shot
        assert shot["workdir"]["width"] > 100, shot
        assert shot["machine"]["left"] >= 0 and shot["machine"]["right"] <= 390, shot
        assert shot["workdir"]["left"] >= 0 and shot["workdir"]["right"] <= 390, shot
    assert short_name["machine"]["top"] == short_name["workdir"]["top"], short_name
    assert long_name["workdir"]["top"] > long_name["machine"]["top"], long_name
    assert long_name["header"]["height"] > short_name["header"]["height"], (
        short_name,
        long_name,
    )

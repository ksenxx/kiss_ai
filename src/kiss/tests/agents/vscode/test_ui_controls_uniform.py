# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Every surface draws its controls with one typeface, one type scale
and the page's own colour scheme.

Rendered in a real Chromium against the production remote server (the
remote page has no VS Code host stylesheet, so it is where browser
defaults would leak through) and against the trajectory visualizer:

* native inputs and buttons inherit the body typeface instead of the
  browser's Arial;
* the settings dialog's fields are body-sized and its section toggles
  (API Keys, Custom Models) share the history panel's Filters toggle;
* the history panel's collapsible chat headers are not bold;
* checkboxes take the accent, and ``color-scheme`` follows the theme so
  select popups and scrollbars are drawn dark on dark;
* the send button gains no glow on hover;
* the visualizer paints in the same neutral palette, with plain
  headings and a neutral model role.
"""

from __future__ import annotations

import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

import kiss.viz_trajectory as viz_pkg
from kiss.tests.agents.vscode.test_codex_task_panel_style import (
    _start_live_server,
)

VIZ_TEMPLATE = Path(viz_pkg.__file__).parent / "templates" / "index.html"

_PROBE_JS = """
() => {
  const cs = el => getComputedStyle(el);
  const body = document.body;
  const header = document.createElement('button');
  header.className = 'history-chat-header';
  header.textContent = 'chat';
  document.getElementById('sidebar').appendChild(header);
  const filters = document.getElementById('history-filters-toggle');
  const subpanel = document.querySelector('.config-subpanel-toggle');
  return {
    bodyFont: cs(body).fontFamily,
    searchFont: cs(document.getElementById('history-search')).fontFamily,
    closeFont: cs(document.getElementById('sidebar-close')).fontFamily,
    budgetFont: cs(document.getElementById('cfg-max-budget')).fontFamily,
    bodySize: cs(body).fontSize,
    budgetSize: cs(document.getElementById('cfg-max-budget')).fontSize,
    selectSize: cs(document.querySelector('.config-label select')).fontSize,
    headerWeight: cs(header).fontWeight,
    filters: [cs(filters).textTransform, cs(filters).fontSize,
              cs(filters).letterSpacing, cs(filters).color],
    subpanel: [cs(subpanel).textTransform, cs(subpanel).fontSize,
               cs(subpanel).letterSpacing, cs(subpanel).color],
    checkboxAccent: cs(document.getElementById('cfg-auto-commit')).accentColor,
    linkColor: cs(document.querySelector('a') || body).color,
    scheme: cs(body).colorScheme,
  };
}
"""


def test_remote_page_controls_share_typeface_scale_and_scheme(tmp_path: Path) -> None:
    """Served remote page: inherited typeface, body-sized settings
    fields, shared section toggles, non-bold chat headers, accent
    checkboxes, dark/light colour scheme, and no send-button glow."""
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
                page.goto(f"https://127.0.0.1:{port}/", wait_until="load")
                page.wait_for_selector("#output", state="attached")
                dark = page.evaluate(_PROBE_JS)

                page.evaluate("() => document.getElementById('theme-btn').click()")
                page.wait_for_function(
                    "() => document.body.classList.contains('light-theme')", timeout=10000
                )
                light_scheme = page.evaluate("() => getComputedStyle(document.body).colorScheme")
                page.evaluate("() => document.getElementById('theme-btn').click()")
                page.wait_for_function(
                    "() => !document.body.classList.contains('light-theme')", timeout=10000
                )

                # main.css's own send button (the remote sheet restyles it):
                # drop the remote class, hover, and read the shadow.
                page.evaluate("() => document.body.classList.remove('remote-chat')")
                page.hover("#send-btn")
                page.wait_for_timeout(400)
                send_shadow = page.evaluate(
                    "() => getComputedStyle(document.getElementById('send-btn')).boxShadow"
                )
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    assert dark["searchFont"] == dark["bodyFont"], dark
    assert dark["closeFont"] == dark["bodyFont"], dark
    assert dark["budgetFont"] == dark["bodyFont"], dark
    assert dark["budgetSize"] == dark["bodySize"], dark
    assert dark["selectSize"] == dark["bodySize"], dark
    assert dark["headerWeight"] == "400", dark
    assert dark["filters"][0] == "uppercase", dark
    assert dark["filters"] == dark["subpanel"], dark
    assert dark["checkboxAccent"] == "rgb(77, 170, 252)", dark
    assert dark["scheme"] == "dark", dark
    assert light_scheme == "light"
    assert send_shadow == "none"


def test_visualizer_uses_the_shared_neutral_palette() -> None:
    """The trajectory visualizer: Dark Modern greys, headings in the
    body colour, the model role in the body colour, dark scheme."""
    html = VIZ_TEMPLATE.read_text(encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.route("**/*", lambda route: route.abort())
            page.set_content(html, wait_until="domcontentloaded")
            probe = page.evaluate(
                """
                () => {
                  const box = document.getElementById('messages-container');
                  box.innerHTML = '<div class="message">' +
                    '<span class="message-role model">model</span>' +
                    '<span class="message-role user">user</span>' +
                    '<div class="message-content"><h2>Title</h2><p>Body</p></div></div>';
                  const cs = el => getComputedStyle(el);
                  return {
                    bodyBg: cs(document.body).backgroundColor,
                    bodyColor: cs(document.body).color,
                    modelRole: cs(document.querySelector('.message-role.model')).color,
                    userRole: cs(document.querySelector('.message-role.user')).color,
                    heading: cs(document.querySelector('.message-content h2')).color,
                    scheme: cs(document.documentElement).colorScheme,
                    sidebarBg: cs(document.querySelector('.jobs-sidebar')).backgroundColor,
                  };
                }
                """
            )
        finally:
            browser.close()
    assert probe["bodyBg"] == "rgb(31, 31, 31)", probe
    assert probe["sidebarBg"] == "rgb(24, 24, 24)", probe
    assert probe["bodyColor"] == "rgb(204, 204, 204)", probe
    assert probe["modelRole"] == probe["bodyColor"], probe
    assert probe["heading"] == probe["bodyColor"], probe
    assert probe["userRole"] == "rgb(77, 170, 252)", probe
    assert probe["scheme"] == "dark", probe

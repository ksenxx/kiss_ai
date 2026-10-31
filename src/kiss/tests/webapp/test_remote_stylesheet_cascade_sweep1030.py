# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end cascade tests of the remote page's stylesheets.

These tests render the REAL page returned by
:func:`kiss.server.web_server._build_html` (``media/chat.html`` with
``main.css``, ``remote-codex.css``, the inlined palettes and
``brand.css``) in headless Chromium and read what the browser computes:

* The Machine filter pill ships ``hidden`` (chat.html) and stays hidden
  until a remote machine is registered; the pill's own ``display`` must
  not undo the attribute.
* The sign-in window's box wears the same frame as the password dialog
  and the server-reset dialog.
* History row buttons take the theme foreground, a favourite the
  theme's favourite colour and a copied button the theme's ok colour.
* The settings dialog carries the theme's widget-border hairline and
  the large elevation.

The static server never answers the websocket, so main.js keeps the
page in its shipped state (no machine registered).  No mocks, patches
or fakes: a real HTTP server serves the real assets to a real browser.
"""

from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from kiss.server.web_server import MEDIA_DIR, _build_html
from kiss.tests.conftest import goto_retrying_network_change

# main.js reveals #app only after the websocket handshake, which never
# happens against a static server; an !important rule beats the veil's
# inline styles while every real CSS rule still applies.
_PREPARE_JS = """
() => {
  const style = document.createElement('style');
  style.textContent = `
    #app { display: flex !important; }
    #kiss-server-loading { display: none !important; }
    #auth-modal { display: none !important; }
  `;
  document.head.appendChild(style);
}
"""

# The frame of a dialog box, as computed.
_FRAME_JS = """
(sel) => {
  const cs = getComputedStyle(document.querySelector(sel));
  return {
    background: cs.backgroundColor,
    border: [cs.borderTopWidth, cs.borderTopStyle, cs.borderTopColor].join(' '),
    radius: cs.borderTopLeftRadius,
    shadow: cs.boxShadow,
  };
}
"""

# A history row as renderHistory builds it (a `.sidebar-item
# .running-item` with its action buttons and the ids line), plus probes
# that resolve the palette names in the same place.
_ROW_COLOURS_JS = """
() => {
  const list = document.getElementById('history-list');
  const row = document.createElement('div');
  row.className = 'sidebar-item running-item';
  row.innerHTML =
    '<div class="sidebar-item-actions">' +
    '<button class="sidebar-item-favorite"></button>' +
    '<button class="sidebar-item-favorite favorited"></button>' +
    '<button class="sidebar-item-copy"></button>' +
    '<button class="sidebar-item-copy copied"></button>' +
    '<button class="sidebar-item-collapse"></button>' +
    '</div>' +
    '<div class="running-item-info"><span class="running-item-ids">ids' +
    '<button class="ids-copy-btn"></button>' +
    '<button class="ids-copy-btn copied"></button></span></div>';
  list.appendChild(row);
  const probe = (value) => {
    const el = document.createElement('span');
    el.style.color = value;
    row.appendChild(el);
    return getComputedStyle(el).color;
  };
  const colour = (sel) => getComputedStyle(row.querySelector(sel)).color;
  const out = {
    fg: probe('var(--fg)'),
    favorite: probe('var(--favorite)'),
    ok: probe('var(--status-ok)'),
    favoriteBtn: colour('.sidebar-item-favorite:not(.favorited)'),
    favorited: colour('.sidebar-item-favorite.favorited'),
    copyBtn: colour('.sidebar-item-copy:not(.copied)'),
    copied: colour('.sidebar-item-copy.copied'),
    collapseBtn: colour('.sidebar-item-collapse'),
    idsCopyBtn: colour('.ids-copy-btn:not(.copied)'),
    idsCopied: colour('.ids-copy-btn.copied'),
  };
  row.remove();
  return out;
}
"""

# The settings dialog's border and shadow next to the palette values
# they must resolve to.
_SETTINGS_FRAME_JS = """
() => {
  const panel = document.getElementById('settings-panel');
  const probe = document.createElement('div');
  probe.style.color = 'var(--border-widget)';
  probe.style.boxShadow = 'var(--shadow-lg)';
  panel.parentElement.appendChild(probe);
  const cs = getComputedStyle(panel);
  const ps = getComputedStyle(probe);
  const out = {
    border: [cs.borderTopWidth, cs.borderTopStyle, cs.borderTopColor].join(' '),
    shadow: cs.boxShadow,
    widgetBorder: ps.color,
    shadowLg: ps.boxShadow,
  };
  probe.remove();
  return out;
}
"""


def _serve(directory: Path) -> Iterator[str]:
    """Serve ``directory`` over HTTP on an ephemeral port; yield its URL."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(directory),
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def remote_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve the real remote HTML page plus the real media assets."""
    root = tmp_path_factory.mktemp("remote-stylesheets")
    (root / "index.html").write_text(_build_html(), encoding="utf-8")
    (root / "media").symlink_to(MEDIA_DIR, target_is_directory=True)
    yield from _serve(root)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """Launch one headless Chromium for the whole module."""
    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        try:
            yield chromium
        finally:
            chromium.close()


@pytest.fixture
def page(browser: Browser, remote_url: str) -> Iterator[Page]:
    """The remote page at a desktop width, #app shown."""
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = context.new_page()
    try:
        goto_retrying_network_change(pg, remote_url, wait_until="load")
        pg.wait_for_selector("#history-filters-toggle", state="attached")
        pg.evaluate(_PREPARE_JS)
        yield pg
    finally:
        context.close()


def _machine_pill(page: Page) -> dict[str, Any]:
    """The Machine pill's ``hidden`` flag, display and box."""
    pill: dict[str, Any] = page.evaluate(
        """() => {
          const pill = document.getElementById('hf-machine').parentElement;
          const r = pill.getBoundingClientRect();
          return {hidden: pill.hidden, display: getComputedStyle(pill).display,
                  width: r.width, height: r.height};
        }"""
    )
    return pill


def test_machine_filter_stays_hidden_until_a_machine_is_registered(page: Page) -> None:
    """With the Filters drawer open and no remote machine registered,
    the Machine pill takes no room; once main.js clears ``hidden`` (a
    machine was registered) it lays out like the SEA pill beside it."""
    page.click("#history-filters-toggle")
    page.wait_for_selector("#history-filters-body", state="visible")
    sea = page.locator("#hf-sea").locator("..").bounding_box()
    assert sea and sea["width"] > 0 and sea["height"] > 0, sea

    pill = _machine_pill(page)
    assert pill["hidden"] is True, pill
    assert pill["display"] == "none", pill
    assert pill["width"] == 0 and pill["height"] == 0, pill

    page.evaluate("() => { document.getElementById('hf-machine').parentElement.hidden = false; }")
    pill = _machine_pill(page)
    assert pill["display"] != "none", pill
    assert pill["width"] > 0 and pill["height"] == pytest.approx(sea["height"]), pill


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_sign_in_box_matches_the_other_dialogs(page: Page, theme: str) -> None:
    """The sign-in window reuses the auth modal's box, so on the remote
    page it gets the same frame (background, hairline, radius, shadow)
    as the password dialog and the server-reset confirmation."""
    if theme == "dark":
        page.evaluate("() => document.body.classList.remove('light-theme')")
    signin = page.evaluate(_FRAME_JS, "#signin-modal .auth-modal-content")
    password = page.evaluate(_FRAME_JS, "#auth-modal .auth-modal-content")
    reset = page.evaluate(_FRAME_JS, ".server-reset-confirm-content")
    assert password == reset, (password, reset)
    assert signin == password, (signin, password)
    assert password["shadow"] != "none", password


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_history_row_buttons_use_the_theme_colours(page: Page, theme: str) -> None:
    """History row buttons read in the theme foreground; a favourite
    turns the favourite colour and a copied button the ok colour."""
    if theme == "dark":
        page.evaluate("() => document.body.classList.remove('light-theme')")
    got = page.evaluate(_ROW_COLOURS_JS)
    for key in ("favoriteBtn", "copyBtn", "collapseBtn", "idsCopyBtn"):
        assert got[key] == got["fg"], (key, got)
    assert got["favorited"] == got["favorite"], got
    assert got["copied"] == got["ok"], got
    assert got["idsCopied"] == got["ok"], got
    assert len({got["fg"], got["favorite"], got["ok"]}) == 3, got


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_settings_dialog_has_the_widget_hairline_and_elevation(page: Page, theme: str) -> None:
    """The settings dialog is outlined by the theme's widget border and
    lifted by the large shadow on the remote page."""
    if theme == "dark":
        page.evaluate("() => document.body.classList.remove('light-theme')")
    got = page.evaluate(_SETTINGS_FRAME_JS)
    assert got["border"] == f"1px solid {got['widgetBorder']}", got
    assert got["shadow"] == got["shadowLg"] != "none", got

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The brand skin wins the cascade on the chat page, and the "..." menu's
appearance / terminal items are displayed there.

``media/brand.css`` documents itself as the override sheet of a
white-label distribution.  It used to be linked BEFORE ``{{HEAD_STYLE}}``
(remote-codex.css and the inlined palettes), and its documented example
``:root { --accent }`` was shadowed on the chat page by the
``body.remote-chat`` re-derivation in remote-codex.css, so a brand could
re-tint the share page but not the application.  The static test pins
the stylesheet order of the built page; the live test serves the page
through the production ``RemoteAccessServer`` into a real Chromium with
the brand.css that brand.css's own header gives as the example (served
in place of the stock, empty file through a Playwright route: the
overlay is deployment data that ``install.sh`` copies over the stock
file, not code) and reads the resulting ``--accent`` in both themes.

The live test also checks that the four "..." menu items main.css once
switched on ``body.remote-chat`` (theme, contrast, auto, terminal) are
displayed: the rules went, the page always carries that body class.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.server.web_server import PRODUCT_NAME, _build_html
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.test_codex_task_panel_style import _start_live_server

MEDIA_DIR = Path(__file__).resolve().parents[2] / "webapp" / "media"

# The sheets of <head>, in document order: every <link rel="stylesheet">
# href and every inline <style>.
_SHEET_RE = re.compile(r'<link href="([^"]+)" rel="stylesheet">|(<style>)')


def _documented_brand_example() -> str:
    """Return the ``--accent`` example rule from brand.css's own header."""
    header = (MEDIA_DIR / "brand.css").read_text(encoding="utf-8")
    m = re.search(
        r"^\s*\*\s+(:root, body\.remote-chat \{ --accent: (#[0-9a-f]{6}); \})", header, re.M
    )
    assert m, "brand.css must document the :root, body.remote-chat example"
    return m.group(1)


def test_brand_css_is_the_last_stylesheet_of_the_chat_page() -> None:
    """brand.css comes after main.css, remote-codex.css and the inlined
    palette <style>, so a brand rule of equal specificity wins."""
    page = _build_html()
    head = page[: page.index("</head>")]
    sheets = [m.group(1) or m.group(2) for m in _SHEET_RE.finditer(head)]
    assert sheets[-1].startswith("/media/brand.css"), sheets
    names = [s.split("?")[0] for s in sheets]
    assert names.index("/media/main.css") < names.index("/media/remote-codex.css"), names
    assert names.index("/media/remote-codex.css") < names.index("<style>"), names
    # The two "machine running ..." tooltips carry the brand name.
    assert f"the machine running {PRODUCT_NAME} as a tab" in page
    assert page.count("machine running KISS Sorcar") == (2 if PRODUCT_NAME == "KISS Sorcar" else 0)


_PROBE_JS = """
() => {
  const cs = getComputedStyle(document.body);
  return {
    light: document.body.classList.contains('light-theme'),
    accent: cs.getPropertyValue('--accent').trim(),
  };
}
"""

_MENU_ITEMS_JS = """
() => Object.fromEntries(
  ['theme-btn', 'contrast-btn', 'theme-auto-btn', 'terminal-btn'].map(id =>
    [id, getComputedStyle(document.getElementById(id)).display]))
"""


@pytest.mark.timeout(180)
def test_live_brand_accent_and_menu_items(tmp_path: Path) -> None:
    """Served page + real Chromium: the documented brand example re-tints
    ``--accent`` in the light (default) and dark themes, and the "..."
    menu shows the theme, contrast, auto and terminal items."""
    example = _documented_brand_example()
    accent_match = re.search(r"#[0-9a-f]{6}", example)
    assert accent_match is not None, "the documented brand example names a hex --accent"
    accent = accent_match.group(0)
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
                page.route(
                    "**/media/brand.css*",
                    lambda route: route.fulfill(content_type="text/css", body=example),
                )
                goto_retrying_network_change(page, f"https://127.0.0.1:{port}/", wait_until="load")
                page.wait_for_selector("#more-btn", state="attached")
                page.wait_for_function(
                    "() => getComputedStyle(document.body).getPropertyValue('--accent')"
                    ".trim() !== ''",
                    timeout=10000,
                )
                light = page.evaluate(_PROBE_JS)
                page.click("#more-btn")
                page.wait_for_selector("#more-menu.open", state="visible")
                items = page.evaluate(_MENU_ITEMS_JS)
                page.click("#theme-btn")
                page.wait_for_function(
                    "() => !document.body.classList.contains('light-theme')", timeout=10000
                )
                dark = page.evaluate(_PROBE_JS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    assert light["light"] is True, light
    assert light["accent"] == accent, light
    assert dark["light"] is False, dark
    assert dark["accent"] == accent, dark
    assert items == {
        "theme-btn": "flex",
        "contrast-btn": "flex",
        "theme-auto-btn": "flex",
        "terminal-btn": "flex",
    }, items

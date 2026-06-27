# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: chat webview toolbar must no longer expose a
``Frequent tasks`` button, and the ``Inject instruction`` button must
render a syringe icon.

The chat webview toolbar (the row of small icon buttons under the
chat text input) used to expose two toggle buttons next to each
other: ``#frequent-tasks-btn`` (a tiny bar-chart icon, tooltip
"Frequent tasks") and ``#tricks-btn`` (a lightbulb-shaped icon,
tooltip "Inject instruction").

This change removes the Frequent-tasks toggle entirely and replaces
the lightbulb icon inside the Inject-instruction button with a
syringe icon (which matches the button's semantic meaning of
"injecting" an instruction into the chat).

The tests below boot a real headless Chromium via Playwright, load
the actual shipped ``media/chat.html`` (with template placeholders
substituted), and assert directly against the rendered DOM:

* ``#frequent-tasks-btn`` is absent — neither as a node nor as a
  visible button.  The previous tooltip text "Frequent tasks" is
  also gone from the toolbar area.
* ``#tricks-btn`` is still present, still carries its ``Inject
  instruction`` tooltip, and the SVG inside it contains the
  signature path data of a syringe (the diagonal needle/barrel
  paths from the Lucide ``syringe`` glyph) instead of the previous
  lightbulb path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

_MEDIA_DIR = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
)
_HTML = _MEDIA_DIR / "chat.html"
_CSS = _MEDIA_DIR / "main.css"

# Distinctive sub-strings of the Lucide ``syringe`` icon's path data.
# The icon has six short ``<path>`` elements; these three are
# unmistakable signatures that only appear in a syringe glyph.
_SYRINGE_PATH_SIGNATURES = (
    "M19 9 8.7 19.3",   # the long needle / barrel diagonal
    "m18 2 4 4",        # the small plunger tab at the top-right
    "m14 4 6 6",        # the cross-piece between barrel and plunger
)

# Distinctive sub-strings of the OLD lightbulb-style ``Inject
# instruction`` icon — must NOT be present after the fix.
_OLD_LIGHTBULB_PATH_SIGNATURES = (
    "M9 18h6",
    "M10 22h4",
    "M12 2a7 7 0 00-4 12.7",
)


def _render_chat_html() -> str:
    """Return ``chat.html`` with every ``{{PLACEHOLDER}}`` substituted.

    Loads the real shipped template (so any future change to the
    button markup is exercised verbatim) and replaces each ``{{KEY}}``
    placeholder with a minimal but valid value, producing a
    standalone HTML document Playwright can render without a backing
    web server.  No external scripts are loaded (the template's
    ``<script src="…">`` references resolve to local ``/media/*`` URLs
    that don't exist in this synthetic page); the tests below only
    inspect static DOM emitted directly by the template.
    """
    tpl = _HTML.read_text(encoding="utf-8")
    css = _CSS.read_text(encoding="utf-8")
    subs = {
        "VIEWPORT": "width=device-width,initial-scale=1",
        "CSP_META": "",
        "STYLE_HREF": "about:blank",
        "HLJS_CSS_HREF": "about:blank",
        "HEAD_STYLE": f"<style>{css}</style>",
        "BODY_CLASS_ATTR": "",
        "INPUT_PLACEHOLDER": "Ask me anything",
        "ENTERKEYHINT": "",
        "MODEL_NAME": "test-model",
        "VERSION_SUFFIX": "",
        "AUTH_MODAL": "",
        "NONCE_ATTR": "",
        "HLJS_SRC": "about:blank",
        "MARKED_SRC": "about:blank",
        "PANEL_COPY_SRC": "about:blank",
        "MAIN_SRC": "about:blank",
        "DEMO_SRC": "about:blank",
        "SHIM_SCRIPT": "",
        "TRICKS_JSON": "[]",
    }
    for key, value in subs.items():
        tpl = tpl.replace("{{" + key + "}}", value)
    return tpl


@pytest.fixture(scope="module")
def _browser():
    """Launch a single headless Chromium for all tests in this module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture()
def _page(_browser):
    """Open a fresh page rendering the real chat.html template."""
    context = _browser.new_context(viewport={"width": 1024, "height": 768})
    try:
        page = context.new_page()
        page.set_content(_render_chat_html(), wait_until="domcontentloaded")
        # Make #app visible — the template starts it ``display:none`` and
        # only main.js flips it on once the server is ready.  Without
        # main.js loaded in this synthetic page we toggle it ourselves
        # so layout queries (e.g. ``offsetParent``) report meaningful
        # values for the toolbar buttons.
        page.evaluate(
            "document.getElementById('app').style.display = 'block';",
        )
        yield page
    finally:
        context.close()


def test_frequent_tasks_button_is_removed_from_dom(_page) -> None:
    """``#frequent-tasks-btn`` must not exist in the rendered DOM.

    Before the fix the toolbar shipped a tiny bar-chart toggle next
    to the ``Inject instruction`` button that opened the Frequent
    tasks slide-up panel.  The button has been removed; the DOM
    query must return zero matches.
    """
    count = _page.evaluate(
        "document.querySelectorAll('#frequent-tasks-btn').length",
    )
    assert count == 0, (
        "#frequent-tasks-btn is still present in chat.html — the "
        "Frequent tasks toggle button must be removed from the "
        "chat webview toolbar."
    )


def test_no_button_advertises_frequent_tasks_tooltip(_page) -> None:
    """No toolbar button may still advertise the ``Frequent tasks``
    tooltip.

    The old button used ``data-tooltip="Frequent tasks"``.  This
    test scans every button in the input footer area (the toolbar
    that hosts the model/upload/inject buttons) and asserts none
    of them carries that tooltip string.
    """
    tooltips = _page.evaluate(
        """
        () => Array.from(
            document.querySelectorAll('#input-footer button')
        ).map(b => b.getAttribute('data-tooltip') || '')
        """,
    )
    assert isinstance(tooltips, list)
    assert "Frequent tasks" not in tooltips, (
        f"A toolbar button still advertises the 'Frequent tasks' "
        f"tooltip: {tooltips!r}.  The button must be removed."
    )


def test_inject_instruction_button_uses_syringe_icon(_page) -> None:
    """``#tricks-btn`` must render a syringe glyph, not a lightbulb.

    Asserts (a) the button is still present, (b) it still carries
    the ``Inject instruction`` tooltip, and (c) the inline SVG
    inside it contains the signature path data of the Lucide
    ``syringe`` icon and none of the previous lightbulb path data.
    """
    info = _page.evaluate(
        """
        () => {
            const btn = document.getElementById('tricks-btn');
            if (!btn) return null;
            return {
                tooltip: btn.getAttribute('data-tooltip') || '',
                svg: btn.innerHTML,
            };
        }
        """,
    )
    assert info is not None, "#tricks-btn must still exist in chat.html"
    assert info["tooltip"] == "Inject instruction", (
        f"#tricks-btn tooltip changed unexpectedly: "
        f"got {info['tooltip']!r}, want 'Inject instruction'."
    )
    svg = info["svg"]
    assert "<svg" in svg, "#tricks-btn must contain an inline <svg> icon"
    for signature in _SYRINGE_PATH_SIGNATURES:
        assert signature in svg, (
            f"#tricks-btn SVG missing syringe path signature "
            f"{signature!r}; the Inject-instruction icon must be a "
            f"syringe.  Got SVG: {svg}"
        )
    for old in _OLD_LIGHTBULB_PATH_SIGNATURES:
        assert old not in svg, (
            f"#tricks-btn SVG still contains the OLD lightbulb path "
            f"data {old!r}; the icon must be replaced by a syringe."
        )


def test_inject_instruction_button_is_visible(_page) -> None:
    """The Inject-instruction button must remain visible/clickable
    in the toolbar after the icon swap.

    Guards against accidentally hiding the button while editing the
    SVG markup (e.g. broken close tag pushing the button out of the
    flexbox row).
    """
    visible = _page.evaluate(
        """
        () => {
            const btn = document.getElementById('tricks-btn');
            if (!btn) return false;
            const r = btn.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
        }
        """,
    )
    assert visible, (
        "#tricks-btn is in the DOM but renders with zero width/"
        "height — the Inject-instruction button must stay visible."
    )



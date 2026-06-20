# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: the settings panel must be vertically scrollable.

Regression: when the settings form grew taller than the viewport (work_dir
row, API-key rows, custom-endpoint/headers, the Update / Server-reset
buttons, etc.) the bottom of the form was clipped off-screen.  The user
could not reach those controls because the panel itself did not scroll.

Root cause: ``#settings-panel`` is a fixed-position flex column with
``overflow-y: auto``, but its only child is ``.sidebar-section`` whose
shared rule (added back when the History / Frequent tabs used it) sets
``flex: 1`` + ``overflow: hidden`` + ``min-height: 0``.  That made the
child grow to exactly the parent's height and clip everything past it,
so ``#settings-panel`` never had any overflow to scroll over.

This test boots a real headless Chromium via Playwright, loads the
actual ``media/main.css`` and the actual ``<div id="settings-panel">``
markup extracted from ``media/chat.html``, sizes the viewport so the
form is guaranteed taller than the panel, and asserts that the panel:

* reports ``scrollHeight > clientHeight`` (its content overflows), and
* moves its ``scrollTop`` when the user (or JS) tries to scroll it.

Before the CSS fix both assertions fail: the inner ``.sidebar-section``
clips the form and the panel itself has nothing to scroll.
"""

from __future__ import annotations

import re
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
_CSS = _MEDIA_DIR / "main.css"
_HTML = _MEDIA_DIR / "chat.html"


def _extract_settings_panel_markup() -> str:
    """Return the literal ``<div id="settings-panel">…</div>`` block.

    Reads the actual shipped ``chat.html`` and balances the brace-
    equivalent ``<div>`` / ``</div>`` pairs from the opening tag of the
    settings panel to its matching close tag.  Doing this from the
    real file (rather than hard-coding markup in the test) guarantees
    the test exercises the same DOM the user sees in the webview.
    """
    src = _HTML.read_text(encoding="utf-8")
    m = re.search(r'<div id="settings-panel">', src)
    assert m, "settings-panel div not found in chat.html"
    start = m.start()
    i = m.end()
    depth = 1
    div_open = re.compile(r"<div\b", re.IGNORECASE)
    div_close = re.compile(r"</div>", re.IGNORECASE)
    while depth > 0:
        no = div_open.search(src, i)
        nc = div_close.search(src, i)
        assert nc, "unbalanced <div> for settings-panel"
        if no and no.start() < nc.start():
            depth += 1
            i = no.end()
        else:
            depth -= 1
            i = nc.end()
    return src[start:i]


def _build_test_page() -> str:
    """Build a self-contained HTML page that renders the settings panel.

    Inlines the real ``main.css`` (so this test exercises the same
    cascade the user gets), sets the VS Code CSS variables the stylesheet
    depends on to concrete colours, and forces the settings panel to be
    visible by adding the ``open`` class.  No JavaScript is loaded so
    the test isolates layout / scrolling from any runtime behaviour.
    """
    css = _CSS.read_text(encoding="utf-8")
    panel = _extract_settings_panel_markup()
    # Render the {{VERSION_SUFFIX}} placeholder so the page is valid.
    panel = panel.replace("{{VERSION_SUFFIX}}", "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root {{
      --vscode-font-size: 13px;
      --vscode-font-family: -apple-system, BlinkMacSystemFont,
        'Segoe UI', Roboto, sans-serif;
      --vscode-editor-background: #1e1e1e;
      --vscode-editor-foreground: #cccccc;
      --vscode-input-background: #3c3c3c;
      --vscode-input-foreground: #cccccc;
      --vscode-input-border: #3c3c3c;
      --vscode-sideBar-background: #252526;
      --vscode-panel-border: #80808059;
      --vscode-descriptionForeground: #8b8b8b;
      --vscode-textLink-foreground: #3794ff;
      --vscode-terminal-ansiRed: #f44747;
      --vscode-terminal-ansiGreen: #6a9955;
      --vscode-terminal-ansiYellow: #d7ba7d;
      --vscode-terminal-ansiBlue: #569cd6;
      --vscode-terminal-ansiMagenta: #c586c0;
      --vscode-terminal-ansiCyan: #4ec9b0;
    }}
    html, body {{ height: 100%; margin: 0; padding: 0; }}
  </style>
  <style>{css}</style>
  <title>settings panel scroll test</title>
</head>
<body>
  <div id="app">
    {panel}
  </div>
  <script>
    document.getElementById('settings-panel').classList.add('open');
  </script>
</body>
</html>
"""


@pytest.fixture(scope="module")
def _browser():
    """Launch a single headless Chromium for all tests in this module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _measure_settings_panel(_browser, viewport_height: int) -> dict:
    """Return scroll geometry for ``#settings-panel`` at *viewport_height*.

    Loads the synthesised page in a fresh context sized
    ``(420 × viewport_height)``, lets layout settle, attempts to scroll
    the panel to the bottom, and returns the relevant measurements
    (``clientHeight``, ``scrollHeight``, ``scrollTop`` after scrolling,
    and the bounding box of the last interactive control inside the
    form so the test can assert it is or isn't reachable).
    """
    context = _browser.new_context(viewport={"width": 420, "height": viewport_height})
    try:
        page = context.new_page()
        page.set_content(_build_test_page(), wait_until="load")
        # Give the browser a tick to finish layout/paint.
        page.wait_for_function(
            "document.getElementById('settings-panel') !== null",
            timeout=5000,
        )
        result = page.evaluate(
            """
            () => {
              const p = document.getElementById('settings-panel');
              p.scrollTop = 0;
              const initialScrollTop = p.scrollTop;
              p.scrollTo({top: 99999, behavior: 'instant'});
              const finalScrollTop = p.scrollTop;
              const headersInput = document.getElementById('cfg-custom-headers');
              const headersRect = headersInput
                ? headersInput.getBoundingClientRect()
                : null;
              return {
                clientHeight: p.clientHeight,
                scrollHeight: p.scrollHeight,
                initialScrollTop: initialScrollTop,
                finalScrollTop: finalScrollTop,
                viewportHeight: window.innerHeight,
                headersTop: headersRect ? headersRect.top : null,
                headersBottom: headersRect ? headersRect.bottom : null,
              };
            }
            """,
        )
        assert isinstance(result, dict)
        return result
    finally:
        context.close()


def test_settings_panel_content_overflows_short_viewport(_browser) -> None:
    """In a short viewport the form must be taller than the panel.

    This sanity check guards the rest of the test: if the form fits the
    viewport, "is it scrollable?" is meaningless.
    """
    m = _measure_settings_panel(_browser, viewport_height=300)
    assert m["scrollHeight"] > m["clientHeight"], (
        f"Settings form ({m['scrollHeight']}px) fits inside the panel "
        f"({m['clientHeight']}px) at viewport 300px — viewport is not "
        "short enough to exercise scrolling.  Adjust the test."
    )


def test_settings_panel_scrolls_when_content_overflows(_browser) -> None:
    """``#settings-panel`` must scroll when its content exceeds the viewport.

    Reproduces the original bug: with the unfixed CSS the inner
    ``.sidebar-section`` (``flex: 1; overflow: hidden``) clamps the form
    to the panel's height so ``#settings-panel`` itself has nothing to
    scroll and ``scrollTop`` is stuck at 0.
    """
    m = _measure_settings_panel(_browser, viewport_height=300)
    assert m["finalScrollTop"] > 0, (
        f"Settings panel did not scroll: scrollTop stayed at "
        f"{m['finalScrollTop']} after scrollTo(99999).  "
        f"scrollHeight={m['scrollHeight']}, clientHeight={m['clientHeight']}. "
        "The panel must be vertically scrollable when the form is "
        "taller than the viewport."
    )


def test_settings_panel_last_field_reachable_via_scroll(_browser) -> None:
    """The last interactive control must become reachable after scrolling.

    ``#cfg-custom-headers`` is the last labelled input in the settings
    form (see ``media/chat.html``).  In a short viewport it starts
    below the visible area; after scrolling the panel to the bottom it
    must end up at or above the panel's bottom edge so the user can
    actually click into it.
    """
    m = _measure_settings_panel(_browser, viewport_height=300)
    assert m["headersBottom"] is not None, "cfg-custom-headers missing"
    assert m["headersBottom"] <= m["viewportHeight"] + 1, (
        f"After scrolling, the bottom of #cfg-custom-headers is at "
        f"y={m['headersBottom']} but the viewport ends at "
        f"y={m['viewportHeight']} — the user cannot reach the last "
        "control of the settings form."
    )

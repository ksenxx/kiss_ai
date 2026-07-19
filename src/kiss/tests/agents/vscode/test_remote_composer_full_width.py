# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the remote-webapp composer spans the full chat webview.

Regression: in the remote web app the panel that holds the input
textbox and the composer buttons (``#input-container``, containing
``#input-wrap`` → ``#task-input`` and ``#input-footer`` → the
menu/model/upload/tricks/voice/send/stop buttons) rendered narrower
than the chat webview.  The chat thread (``#output``) spans the whole
webview width, but ``remote-codex.css`` capped the composer with
``body.remote-chat #input-container { max-width: 90%; margin: 0 auto }``
so the panel floated centered at only 90% of the available width.

These tests boot a real headless Chromium via Playwright, load the
REAL ``#input-area`` + ``#output`` markup extracted from the shipped
``media/chat.html`` together with the REAL ``media/main.css`` and
``media/remote-codex.css`` cascade, and measure rendered geometry:

* Remote web app (``body.remote-chat``), mobile and desktop widths:
  ``#input-container`` must fill the entire content width of
  ``#input-area`` — i.e. be as wide as the chat webview minus only the
  composer gutter padding — instead of 90% of it.
* The button row (``#input-footer``) and the textbox wrapper
  (``#input-wrap``) must span that same full panel width.
* VS Code webview parity (no ``remote-chat`` class): the composer
  already fills the width there and must stay that way.

Before the CSS fix the remote assertions fail with the composer at
~90% of the available width.
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
_MAIN_CSS = _MEDIA_DIR / "main.css"
_CODEX_CSS = _MEDIA_DIR / "remote-codex.css"
_HTML = _MEDIA_DIR / "chat.html"


def _extract_div_block(src: str, div_id: str) -> str:
    """Return the literal ``<div id="{div_id}">…</div>`` block.

    Reads the actual shipped ``chat.html`` and balances ``<div>`` /
    ``</div>`` pairs from the opening tag to its matching close tag, so
    the test exercises the exact DOM the user sees.
    """
    m = re.search(rf'<div id="{re.escape(div_id)}">', src)
    assert m, f"{div_id} div not found in chat.html"
    start = m.start()
    i = m.end()
    depth = 1
    div_open = re.compile(r"<div\b", re.IGNORECASE)
    div_close = re.compile(r"</div>", re.IGNORECASE)
    while depth > 0:
        no = div_open.search(src, i)
        nc = div_close.search(src, i)
        assert nc, f"unbalanced <div> for {div_id}"
        if no and no.start() < nc.start():
            depth += 1
            i = no.end()
        else:
            depth -= 1
            i = nc.end()
    return src[start:i]


def _build_test_page(body_class: str) -> str:
    """Build a self-contained page with the real chat + composer DOM.

    Inlines the real ``main.css`` and ``remote-codex.css`` (in the
    same order the remote page links them, so the override cascade is
    identical), renders the template placeholders, and mounts the real
    ``#output`` and ``#input-area`` blocks inside ``#app`` exactly as
    ``chat.html`` nests them.  *body_class* selects the surface:
    ``"remote-chat"`` (mobile remote), ``"remote-chat remote-desktop"``
    (desktop remote), or ``""`` (VS Code webview).
    """
    src = _HTML.read_text(encoding="utf-8")
    output = _extract_div_block(src, "output")
    input_area = (
        _extract_div_block(src, "input-area")
        .replace("{{INPUT_PLACEHOLDER}}", "Ask anything")
        .replace("{{ENTERKEYHINT}}", "")
        .replace("{{MODEL_NAME}}", "claude-fable-5")
    )
    for block_name, block in (("output", output), ("input-area", input_area)):
        assert "{{" not in block, (
            f"unexpected template placeholder in #{block_name}"
        )
    main_css = _MAIN_CSS.read_text(encoding="utf-8")
    codex_css = _CODEX_CSS.read_text(encoding="utf-8")
    cls = f' class="{body_class}"' if body_class else ""
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
  <style>{main_css}</style>
  <style>{codex_css}</style>
  <title>remote composer width test</title>
</head>
<body{cls}>
  <div id="app">
    {output}
    {input_area}
  </div>
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


def _measure(_browser, body_class: str, viewport_width: int) -> dict:
    """Return composer geometry at *viewport_width* for *body_class*.

    Loads the synthesised page in a fresh context, lets layout settle,
    and returns the rendered widths of the chat webview (``body``),
    the chat thread (``#output``), the composer gutter
    (``#input-area``, plus its horizontal padding), the composer panel
    (``#input-container``), the textbox wrapper (``#input-wrap``), and
    the button row (``#input-footer``).
    """
    context = _browser.new_context(
        viewport={"width": viewport_width, "height": 800}
    )
    try:
        page = context.new_page()
        page.set_content(_build_test_page(body_class), wait_until="load")
        page.wait_for_function(
            "document.getElementById('input-container') !== null",
            timeout=5000,
        )
        result = page.evaluate(
            """
            () => {
              const area = document.getElementById('input-area');
              const container = document.getElementById('input-container');
              const wrap = document.getElementById('input-wrap');
              const footer = document.getElementById('input-footer');
              const output = document.getElementById('output');
              const areaStyle = getComputedStyle(area);
              return {
                bodyWidth: document.body.getBoundingClientRect().width,
                outputWidth: output.getBoundingClientRect().width,
                areaWidth: area.getBoundingClientRect().width,
                areaPadLeft: parseFloat(areaStyle.paddingLeft),
                areaPadRight: parseFloat(areaStyle.paddingRight),
                containerWidth: container.getBoundingClientRect().width,
                wrapWidth: wrap.getBoundingClientRect().width,
                footerWidth: footer.getBoundingClientRect().width,
              };
            }
            """,
        )
        assert isinstance(result, dict)
        return result
    finally:
        context.close()


def _assert_full_width(m: dict) -> None:
    """Assert the composer panel spans the whole chat webview.

    The chat webview is the full body width; the composer sits in the
    ``#input-area`` gutter whose only inset is its horizontal padding.
    The panel (``#input-container``) must therefore fill the entire
    ``#input-area`` content width — the same width the chat thread
    enjoys — not 90% of it.
    """
    available = m["areaWidth"] - m["areaPadLeft"] - m["areaPadRight"]
    assert abs(m["containerWidth"] - available) <= 1, (
        f"composer panel is {m['containerWidth']}px wide but the chat "
        f"webview offers {available}px "
        f"({m['containerWidth'] / available:.0%} of the available "
        "width) — the panel with the input textbox and buttons must be "
        "as wide as the chat webview"
    )
    # The composer gutter spans the same chat column as the chat
    # thread (#output).  On desktop the docked sidebar shifts the
    # whole column right, so compare against the thread, not the body.
    assert abs(m["areaWidth"] - m["outputWidth"]) <= 1
    assert m["outputWidth"] <= m["bodyWidth"]


def test_remote_composer_fills_chat_webview_mobile(_browser) -> None:
    """Remote web app, phone width: the panel holding the input textbox
    and the buttons must be as wide as the chat webview.

    Reproduces the reported bug: with the unfixed CSS the panel is
    ``max-width: 90%`` + centered, so it measures ~90% of the available
    width and this assertion fails.
    """
    m = _measure(_browser, "remote-chat", viewport_width=420)
    _assert_full_width(m)


def test_remote_composer_fills_chat_webview_desktop(_browser) -> None:
    """Remote web app, desktop width: the composer panel must still
    span the full chat webview at large viewports."""
    m = _measure(_browser, "remote-chat remote-desktop", viewport_width=1280)
    _assert_full_width(m)


def test_remote_textbox_and_buttons_span_panel(_browser) -> None:
    """Inside the full-width panel, both the textbox row and the button
    row must stretch across the panel's whole content width so the
    controls actually use the reclaimed space."""
    m = _measure(_browser, "remote-chat", viewport_width=420)
    assert abs(m["wrapWidth"] - m["footerWidth"]) <= 1, (
        f"textbox row ({m['wrapWidth']}px) and button row "
        f"({m['footerWidth']}px) must be equally wide inside the panel"
    )
    assert m["wrapWidth"] <= m["containerWidth"]
    assert m["wrapWidth"] >= m["containerWidth"] - 40, (
        "textbox/button rows must fill the panel (minus its padding)"
    )


def test_vscode_webview_composer_unaffected(_browser) -> None:
    """VS Code webview parity: without ``body.remote-chat`` the
    composer already fills the width and must remain untouched by the
    remote-only stylesheet."""
    m = _measure(_browser, "", viewport_width=420)
    _assert_full_width(m)

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the History panel filters live in a collapsible
"Filters" panel — in the remote web view AND the extension webview.

The task-status filter chips (Running / Errored / Succeeded /
Workspace / Favorites) and the From/To date range must sit under a
collapsible panel titled "Filters".  The filter buttons and dates
MUST be visible whenever that panel is uncollapsed, and hidden while
it is collapsed.

Coverage is split across three surfaces:

* **Remote web view** — ``web_server._build_html()`` (the exact page
  served by the remote web server) must embed the Filters panel
  markup between the history search box and the history list.

* **Real browser behaviour** — a headless Chromium loads the real
  ``chat.html`` body with the real ``main.css`` + ``main.js`` (the
  same harness as ``test_history_failed_red_circle.py``) and this
  test asserts actual painted geometry: every chip and date input
  has a live layout box while the panel is uncollapsed, loses it
  after clicking the "Filters" header, and gets it back on the next
  click.  A second scenario adds ``body.remote-chat`` plus the
  ``remote-codex.css`` restyle to prove the remote skin does not
  hide the panel.

* **Extension webview wiring** — the jsdom integration test
  ``historyFiltersCollapsible.test.js`` (spawned here under node so
  it runs in CI with pytest) drives markup, ARIA state, toggling,
  ``localStorage`` persistence, and filtering through the panel.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.server import web_server

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_MEDIA_DIR = _VSCODE_DIR / "media"
_CSS = _MEDIA_DIR / "main.css"
_CODEX_CSS = _MEDIA_DIR / "remote-codex.css"
_JS = _MEDIA_DIR / "main.js"
_HTML = _MEDIA_DIR / "chat.html"
_TEST_JS = _VSCODE_DIR / "test" / "historyFiltersCollapsible.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"

#: Every filter control that must hide/show with the panel.
FILTER_CONTROL_IDS = [
    "hf-running",
    "hf-errors",
    "hf-completed",
    "hf-workspace",
    "hf-favorite",
    "hf-from",
    "hf-from-btn",
    "hf-to",
    "hf-to-btn",
]


class TestRemoteWebViewFiltersPanel(unittest.TestCase):
    """The served remote page embeds the collapsible Filters panel."""

    def test_remote_html_has_collapsible_filters_panel(self) -> None:
        """``_build_html()`` output wraps the filter bar in the
        Filters disclosure between search box and history list."""
        html = web_server._build_html()  # type: ignore[attr-defined]
        m = re.search(
            r'id="history-search".*?</div>'
            r"(?P<panel>.*?)"
            r'<div id="history-list"',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(
            m, "Filters panel must sit between search and history-list"
        )
        assert m is not None
        panel = m.group("panel")
        self.assertIn('id="history-filters-panel"', panel)
        self.assertIn('id="history-filters-toggle"', panel)
        self.assertIn('id="history-filters-body"', panel)
        # Disclosure semantics: the toggle is a button titled
        # "Filters" that controls the body and starts expanded.
        toggle = panel.split('id="history-filters-toggle"', 1)[1]
        toggle = toggle.split("</button>", 1)[0]
        self.assertIn('aria-expanded="true"', toggle)
        self.assertIn('aria-controls="history-filters-body"', toggle)
        self.assertIn(">Filters<", toggle)
        # Every filter control lives inside the collapsible body.
        body = panel.split('id="history-filters-body"', 1)[1]
        self.assertIn('class="history-filter-bar"', body)
        for cid in FILTER_CONTROL_IDS:
            self.assertIn(f'id="{cid}"', body)


def _build_test_page(remote_chat: bool = False) -> str:
    """Return a self-contained page loading the real CSS + JS.

    Mirrors the harness of ``test_history_failed_red_circle.py``: the
    production ``chat.html`` body plus inlined ``main.css`` and
    ``main.js`` with the host APIs stubbed.  With ``remote_chat=True``
    the page also inlines ``remote-codex.css`` and tags the body with
    ``remote-chat``, replicating the remote web view's restyle.
    """
    css = _CSS.read_text(encoding="utf-8")
    js = _JS.read_text(encoding="utf-8")
    html = _HTML.read_text(encoding="utf-8")
    body_start = html.find("<body")
    body_open_end = html.find(">", body_start) + 1
    body_end = html.find("</body>")
    body = html[body_open_end:body_end]
    body = "\n".join(
        line
        for line in body.splitlines()
        if "<script" not in line and "</script>" not in line
    )
    codex_css = (
        f"<style>{_CODEX_CSS.read_text(encoding='utf-8')}</style>"
        if remote_chat
        else ""
    )
    body_class = ' class="remote-chat"' if remote_chat else ""
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
  {codex_css}
  <title>collapsible history Filters panel test</title>
</head>
<body{body_class}>
{body}
  <script>
    window.__postedMessages = [];
    window.acquireVsCodeApi = function () {{
      return {{
        postMessage: function (msg) {{ window.__postedMessages.push(msg); }},
        setState: function () {{}},
        getState: function () {{ return null; }},
      }};
    }};
    window.hljs = {{
      highlightElement: function () {{}},
      highlightAll: function () {{}},
    }};
    window.marked = {{ parse: function (s) {{ return s; }} }};
    window.PanelCopy = {{ addCopyButton: function () {{}} }};
    window.__TRICKS__ = [];
    window.__iifeError = null;
    window.addEventListener('error', function (ev) {{
      if (!window.__iifeError) {{
        window.__iifeError = String(ev.error || ev.message);
      }}
    }});
  </script>
  <script>{js}</script>
</body>
</html>
"""


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium shared by the module's tests."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _open_history_page(_browser, remote_chat: bool = False):
    """Open the harness page with the sidebar revealed."""
    context = _browser.new_context(viewport={"width": 480, "height": 900})
    page = context.new_page()
    page.set_content(_build_test_page(remote_chat), wait_until="load")
    page.wait_for_function(
        "document.getElementById('history-list') !== null",
        timeout=5000,
    )
    page.evaluate(
        "() => {"
        " document.getElementById('app').style.display = '';"
        " const ov = document.getElementById('kiss-server-loading');"
        " if (ov) ov.style.display = 'none';"
        " document.getElementById('sidebar').classList.add('open');"
        " }"
    )
    iife_err = page.evaluate("() => window.__iifeError")
    if iife_err:
        pytest.fail(f"main.js IIFE setup raised: {iife_err}")
    return context, page


def _visible(page, element_id: str) -> bool:
    """True when *element_id* has a live layout box (offsetParent)."""
    return bool(
        page.evaluate(
            "(id) => {"
            "  const el = document.getElementById(id);"
            "  return !!el && el.offsetParent !== null;"
            "}",
            element_id,
        )
    )


class TestFiltersPanelRealBrowser:
    """Painted-geometry checks for the collapsible Filters panel."""

    def test_uncollapsed_panel_shows_buttons_and_dates(self, _browser):
        """Expanded by default: header + every control painted."""
        context, page = _open_history_page(_browser)
        try:
            assert _visible(page, "history-filters-toggle"), (
                "the Filters header button must be visible"
            )
            assert (
                page.get_attribute("#history-filters-toggle", "aria-expanded")
                == "true"
            ), "the Filters panel must start uncollapsed"
            for cid in FILTER_CONTROL_IDS:
                assert _visible(page, cid), (
                    f"#{cid} must be visible while the Filters panel "
                    "is uncollapsed"
                )
        finally:
            context.close()

    def test_click_collapses_then_expands(self, _browser):
        """Clicking the header hides every control; clicking again
        repaints them all."""
        context, page = _open_history_page(_browser)
        try:
            page.click("#history-filters-toggle")
            assert (
                page.get_attribute("#history-filters-toggle", "aria-expanded")
                == "false"
            ), "clicking the header must collapse the panel"
            for cid in FILTER_CONTROL_IDS:
                assert not _visible(page, cid), (
                    f"#{cid} must be hidden while the Filters panel "
                    "is collapsed"
                )
            # The header itself must stay visible so the panel can be
            # reopened.
            assert _visible(page, "history-filters-toggle")

            page.click("#history-filters-toggle")
            assert (
                page.get_attribute("#history-filters-toggle", "aria-expanded")
                == "true"
            ), "clicking the header again must uncollapse the panel"
            for cid in FILTER_CONTROL_IDS:
                assert _visible(page, cid), (
                    f"#{cid} must be visible again after uncollapsing"
                )
        finally:
            context.close()

    def test_remote_chat_restyle_keeps_panel_working(self, _browser):
        """The remote-codex restyle must not hide the Filters panel."""
        context, page = _open_history_page(_browser, remote_chat=True)
        try:
            assert _visible(page, "history-filters-toggle"), (
                "the Filters header must be visible in the remote skin"
            )
            for cid in FILTER_CONTROL_IDS:
                assert _visible(page, cid), (
                    f"#{cid} must be visible in the remote skin while "
                    "uncollapsed"
                )
            page.click("#history-filters-toggle")
            for cid in FILTER_CONTROL_IDS:
                assert not _visible(page, cid), (
                    f"#{cid} must hide in the remote skin when collapsed"
                )
        finally:
            context.close()


class TestExtensionWebviewFiltersPanel(unittest.TestCase):
    """Drive the jsdom integration test from pytest."""

    def test_history_filters_collapsible_js(self) -> None:
        """Markup, ARIA, toggling, persistence and filtering pass."""
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                "jsdom is not installed under "
                f"{_VSCODE_DIR / 'node_modules'} — run `npm install` there"
            )
        self.assertTrue(
            _TEST_JS.is_file(),
            f"missing integration test file: {_TEST_JS}",
        )
        r = subprocess.run(
            ["node", str(_TEST_JS)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(_VSCODE_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "historyFiltersCollapsible.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Capture screenshots of every KISS Sorcar UI surface in every theme.

Surfaces:

* ``vscode`` - the chat webview (``media/chat.html`` + ``main.css``) as the
  VS Code sidebar shows it, in Dark+, Light+ and Dark High Contrast.  The
  ``--vscode-*`` variables are the defaults of those built-in themes
  (microsoft/vscode ``theme-defaults`` JSON and colour registry).
* ``remote`` - the remote webapp page built by
  ``kiss.server.web_server._build_html`` (``main.css`` +
  ``remote-codex.css``), desktop and phone widths, dark and light.
* ``share`` - the self-contained shared-chat page
  (``web_server._build_share_page``), dark and light.
* ``viz`` - the trajectory visualizer (``kiss/viz_trajectory``).

Usage::

    uv run python scripts/ui_theme_screenshots.py OUT_DIR

Pages are served from the working tree, so running the script before and
after a CSS change gives a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import re
import sys
import tempfile
import threading
from pathlib import Path

import yaml
from playwright.sync_api import Page, sync_playwright
from werkzeug.serving import make_server

from kiss.server import web_server
from kiss.viz_trajectory import server as viz_server

MEDIA_DIR = Path(web_server.MEDIA_DIR)

FONT_VARS = {
    "font-family": (
        '-apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", '
        'system-ui, Ubuntu, "Droid Sans", sans-serif'
    ),
    "font-size": "13px",
    "editor-font-family": 'Menlo, Monaco, "Courier New", monospace',
    "editor-font-size": "12px",
}

# Built-in theme defaults for every --vscode-* variable the CSS reads.
# A missing key means the theme leaves that colour unset (VS Code emits
# no variable), so the stylesheet's own fallback applies.
THEMES: dict[str, tuple[str, dict[str, str]]] = {
    "dark-plus": ("vscode-dark", {
        "editor-background": "#1e1e1e", "editor-foreground": "#d4d4d4",
        "foreground": "#cccccc", "descriptionForeground": "#ccccccb3",
        "errorForeground": "#f48771", "focusBorder": "#007fd4",
        "icon-foreground": "#c5c5c5", "input-background": "#3c3c3c",
        "input-foreground": "#cccccc", "input-placeholderForeground": "#a6a6a6",
        "button-background": "#0e639c", "button-foreground": "#ffffff",
        "button-hoverBackground": "#1177bb",
        "list-hoverBackground": "#2a2d2e",
        "list-inactiveSelectionBackground": "#37373d",
        "menu-background": "#252526", "menu-foreground": "#cccccc",
        "menu-selectionBackground": "#0078d4", "panel-border": "#80808059",
        "scrollbarSlider-background": "#79797966",
        "sideBar-background": "#252526", "tab-activeForeground": "#ffffff",
        "tab-inactiveForeground": "#a6a6a6",
        "textCodeBlock-background": "#0a0a0a66",
        "textLink-foreground": "#3794ff", "textLink-activeForeground": "#3794ff",
        "toolbar-hoverBackground": "#5a5d5e50", "widget-border": "#303031",
        "widget-shadow": "#0000005c", "editorWidget-background": "#252526",
        "editorWarning-foreground": "#cca700",
        "editorGroupHeader-tabsBackground": "#252526",
        "activityBar-foreground": "#ffffff",
        "activityBar-inactiveForeground": "#ffffff66",
        "activityBar-activeBorder": "#ffffff",
        "terminal-ansiRed": "#cd3131", "terminal-ansiGreen": "#0dbc79",
        "terminal-ansiYellow": "#e5e510", "terminal-ansiMagenta": "#bc3fbc",
        "terminal-ansiCyan": "#11a8cd",
        "charts-green": "#89d185", "charts-red": "#f14c4c",
        "charts-yellow": "#cca700", "charts-purple": "#b180d7",
    }),
    "light-plus": ("vscode-light", {
        "editor-background": "#ffffff", "editor-foreground": "#000000",
        "foreground": "#616161", "descriptionForeground": "#717171",
        "errorForeground": "#a1260d", "focusBorder": "#0090f1",
        "icon-foreground": "#424242", "input-background": "#ffffff",
        "input-foreground": "#616161", "input-placeholderForeground": "#767676",
        "button-background": "#007acc", "button-foreground": "#ffffff",
        "button-hoverBackground": "#0062a3",
        "list-hoverBackground": "#e8e8e8",
        "list-inactiveSelectionBackground": "#e4e6f1",
        "menu-background": "#ffffff", "menu-foreground": "#616161",
        "menu-selectionBackground": "#0060c0", "panel-border": "#80808059",
        "scrollbarSlider-background": "#64646466",
        "sideBar-background": "#f3f3f3", "tab-activeForeground": "#333333",
        "tab-inactiveForeground": "#616161",
        "textCodeBlock-background": "#dcdcdc66",
        "textLink-foreground": "#006ab1", "textLink-activeForeground": "#006ab1",
        "toolbar-hoverBackground": "#b8b8b850", "widget-border": "#d4d4d4",
        "widget-shadow": "#00000029", "editorWidget-background": "#f3f3f3",
        "editorWarning-foreground": "#bf8803",
        "editorGroupHeader-tabsBackground": "#f3f3f3",
        "activityBar-foreground": "#ffffff",
        "activityBar-inactiveForeground": "#ffffff66",
        "activityBar-activeBorder": "#ffffff",
        "terminal-ansiRed": "#cd3131", "terminal-ansiGreen": "#107c10",
        "terminal-ansiYellow": "#949800", "terminal-ansiMagenta": "#bc05bc",
        "terminal-ansiCyan": "#0598bc",
        "charts-green": "#388a34", "charts-red": "#e51400",
        "charts-yellow": "#bf8803", "charts-purple": "#652d90",
    }),
    "high-contrast": ("vscode-high-contrast", {
        "editor-background": "#000000", "editor-foreground": "#ffffff",
        "foreground": "#ffffff", "descriptionForeground": "#ffffffb3",
        "errorForeground": "#f48771", "focusBorder": "#f38518",
        "contrastBorder": "#6fc3df", "contrastActiveBorder": "#f38518",
        "icon-foreground": "#ffffff", "input-background": "#000000",
        "input-foreground": "#ffffff", "input-border": "#6fc3df",
        "input-placeholderForeground": "#ffffffb3",
        "button-background": "#000000", "button-foreground": "#ffffff",
        "button-border": "#6fc3df", "list-hoverBackground": "#ffffff1a",
        "menu-background": "#000000", "menu-foreground": "#ffffff",
        "panel-border": "#6fc3df", "scrollbarSlider-background": "#6fc3df99",
        "sideBar-background": "#000000", "sideBar-border": "#6fc3df",
        "tab-activeForeground": "#ffffff", "tab-inactiveForeground": "#ffffff",
        "textCodeBlock-background": "#000000",
        "textLink-foreground": "#21a6ff", "textLink-activeForeground": "#21a6ff",
        "widget-border": "#6fc3df", "editorWidget-background": "#0c141f",
        "editorWarning-foreground": "#ffd370",
        "activityBar-foreground": "#ffffff",
        "activityBar-inactiveForeground": "#ffffff",
        "activityBar-activeBorder": "#6fc3df",
        "terminal-ansiRed": "#cd0000", "terminal-ansiGreen": "#00cd00",
        "terminal-ansiYellow": "#cdcd00", "terminal-ansiMagenta": "#cd00cd",
        "terminal-ansiCyan": "#00cdcd",
        "charts-green": "#89d185", "charts-red": "#f48771",
        "charts-yellow": "#ffd370", "charts-purple": "#b180d7",
    }),
}

STUB_API_JS = """
window.__posted = [];
window.acquireVsCodeApi = function () {
  return {
    postMessage: function (m) { window.__posted.push(m); },
    setState: function () {},
    getState: function () { return null; },
  };
};
window.__post = function (ev) { window.postMessage(ev, '*'); };
"""

SESSIONS = [
    {"id": "c1", "task_id": 11, "title": "Refactor main.css onto design tokens",
     "failed": False, "is_running": True, "timestamp": 1790448000},
    {"id": "c2", "task_id": 12, "title": "Fix flaky history search test",
     "failed": True, "is_running": False, "timestamp": 1790440000},
    {"id": "c3", "task_id": 13, "title": "Summarise the Q3 release notes",
     "failed": False, "is_running": False, "timestamp": 1790430000,
     "is_favorite": True},
    {"id": "c4", "task_id": 14, "title": "Add dark-mode toggle to share page",
     "failed": False, "is_running": False, "timestamp": 1790420000},
]

MODELS = ["claude-opus-5-5", "gpt-6-astra", "gemini-3-pro", "claude-haiku-5"]

TRANSCRIPT_JS = r"""
() => {
  const api = window._testApi;
  api.hideWelcome();
  const tabId = api.getActiveTabId();
  for (const ev of [
    {type: 'setTaskText', tabId, text:
      'Refactor main.css onto a design-token layer and check every theme.'},
    {type: 'clear', tabId},
  ]) {
    window.dispatchEvent(new MessageEvent('message', {data: ev}));
  }
  const E = ev => api.processEvent(ev);
  E({type: 'thinking_start'});
  E({type: 'thinking_delta', text: 'The radius values drifted: 2, 3, 4, ' +
     '5, 6, 8 and 10 px all appear. Map them onto four steps.'});
  E({type: 'thinking_end'});
  E({type: 'text_delta', text: 'I will start by listing every literal ' +
     'value.\n\n| Kind | Distinct values |\n|---|---|\n| radius | 11 |\n' +
     '| shadow | 17 |\n\nThen replace them with `var(--radius-md)` etc.'});
  E({type: 'text_end'});
  E({type: 'tool_call', name: 'Bash', command:
     "grep -oE 'border-radius:[^;]+' media/main.css | sort | uniq -c",
     description: 'Count radius values'});
  E({type: 'tool_result', content: '  41 border-radius: 4px\n  18 ' +
     'border-radius: 6px\n   9 border-radius: 3px', is_error: false});
  E({type: 'tool_call', name: 'Edit', path: 'media/main.css',
     old_string: 'border-radius: 6px;', new_string:
     'border-radius: var(--radius-md);'});
  E({type: 'tool_result', content: 'Edited media/main.css', is_error: false});
  E({type: 'tool_call', name: 'Read', file_path: 'media/remote-codex.css'});
  E({type: 'tool_result', content: 'ENOENT: no such file', is_error: true});
  E({type: 'warning', message: 'Context usage is above 60%.'});
  E({type: 'result', success: true, summary: 'All radii now use the ' +
     'four-step scale.', total_tokens: 48213, cost: '$0.42', step_count: 12});
  document.getElementById('output').scrollTop = 0;
  return document.getElementById('output').children.length;
}
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that serves ``/media/*`` and logs nothing."""

    def translate_path(self, path: str) -> str:
        """Map ``/media/<name>?v=...`` onto the media directory."""
        name = path.split("?", 1)[0].rsplit("/", 1)[-1]
        return str(MEDIA_DIR / name)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence per-request logging."""


def start_media_server() -> tuple[http.server.ThreadingHTTPServer, str]:
    """Serve ``MEDIA_DIR`` on a free localhost port.

    Returns:
        The running server and its base URL.
    """
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def theme_vars_css(values: dict[str, str]) -> str:
    """Render a ``:root`` block defining the given ``--vscode-*`` values."""
    lines = [f"  --vscode-{k}: {v};" for k, v in {**FONT_VARS, **values}.items()]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


def vscode_page_html(theme: str) -> str:
    """Fill ``chat.html`` the way ``SorcarTab.buildChatHtml`` does.

    Args:
        theme: Key of :data:`THEMES`.

    Returns:
        HTML for the webview with the theme's variables and body class.
    """
    body_class, values = THEMES[theme]
    media = "/media/"
    subs = {
        "VIEWPORT": "width=device-width, initial-scale=1.0",
        "CSP_META": "",
        "STYLE_HREF": media + "main.css",
        "BRAND_STYLE_HREF": media + "brand.css",
        "WELCOME_LOGO_SRC": media + "welcome-logo.png",
        "WELCOME_LOGO_DARK_SRC": media + "welcome-logo-dark.png",
        # The dark sheet, as SorcarTab.ts serves it; main.js swaps in
        # the light one from __HLJS_THEME_CSS__ when the body class
        # says the editor theme is light.
        "HLJS_CSS_HREF": media + "highlight-vscode-dark.css",
        "HEAD_STYLE": "<style>" + theme_vars_css(values) + "</style>",
        "BODY_CLASS_ATTR": f' class="{body_class}"',
        "PRODUCT_NAME": "KISS Sorcar",
        "TAGLINE": "Keep it simple, stupid",
        "BRAND_JSON": '{"productName": "KISS Sorcar", "shortName": "Sorcar"}',
        "INPUT_PLACEHOLDER": "Ask anything... (@ for files)",
        "ENTERKEYHINT": "",
        "MODEL_NAME": "claude-opus-5-5",
        "VERSION_SUFFIX": "",
        "AUTH_MODAL": "",
        "NONCE_ATTR": "",
        "HLJS_SRC": media + "highlight.min.js",
        "MARKED_SRC": media + "marked.min.js",
        "API_SRC": media + "api.js",
        "PANEL_COPY_SRC": media + "panelCopy.js",
        "CTX_MENU_SRC": media + "contentContextMenu.js",
        "TREE_MENU_SRC": media + "treeContextMenu.js",
        "MAIN_SRC": media + "main.js",
        "SHIM_SCRIPT": (
            "<script>window.__HLJS_THEME_CSS__ = {"
            f'"dark": "{media}highlight-vscode-dark.css", '
            f'"light": "{media}highlight-vscode-light.css"'
            "};</script><script>" + STUB_API_JS + "</script>"
        ),
        "TRICKS_JSON": "[]",
        "TIPS_JSON": '{"tips": [], "show": false}',
        "TIPS_SRC": media + "tips.js",
        "VOICE_SRC": media + "voice.js",
        "VOICE_CONFIG": '{"mode": "off"}',
    }
    tpl = (MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
    return re.sub(
        r"( ?)\{\{([A-Z_]+)\}\}",
        functools.partial(_fill_placeholder, subs),
        tpl,
    )


def _fill_placeholder(subs: dict[str, str], m: re.Match[str]) -> str:
    """Replace one ``{{KEY}}`` placeholder (see ``web_server._build_html``)."""
    space, key = m.group(1), m.group(2)
    if key not in subs:
        return m.group(0)
    if key in {"BODY_CLASS_ATTR", "ENTERKEYHINT", "NONCE_ATTR"}:
        return subs[key]
    return space + subs[key]


def remote_page_html() -> str:
    """Return the real remote page with the WebSocket shim stubbed out."""
    page = web_server._build_html()
    shim = f"<script>{web_server._WS_SHIM_JS}</script>"
    assert shim in page, "remote page no longer inlines the WebSocket shim"
    return page.replace(shim, "<script>" + STUB_API_JS + "</script>")


def open_page(page: Page, base: str, html: str, name: str) -> None:
    """Load ``html`` at a URL on the media server's origin."""
    page.route(
        f"{base}/__page__/{name}",
        functools.partial(_fulfil_html, html),
    )
    page.goto(f"{base}/__page__/{name}", wait_until="load")
    page.wait_for_function("() => !!window._testApi", timeout=10000)
    page.evaluate(
        "() => { const a = document.getElementById('app');"
        " if (a) a.style.display = '';"
        " const l = document.getElementById('kiss-server-loading');"
        " if (l) l.style.display = 'none'; }"
    )
    page.evaluate(
        "names => window.__post({type: 'models', models: names.map(n => ("
        "{name: n, vendor: 'Anthropic', inp: 5, out: 25, uses: 3})),"
        " selected: names[0]})",
        MODELS,
    )
    page.wait_for_timeout(300)


def _fulfil_html(html: str, route, request) -> None:  # noqa: ANN001
    """Answer a routed request with the page HTML."""
    route.fulfill(status=200, content_type="text/html", body=html)


def shoot(page: Page, path: Path) -> None:
    """Disable animations and save a viewport screenshot to ``path``."""
    page.add_style_tag(content=(
        "*, *::before, *::after { animation: none !important;"
        " transition: none !important; caret-color: transparent !important; }"
    ))
    page.wait_for_timeout(150)
    page.screenshot(path=str(path))
    print("saved", path)


def chat_scenes(page: Page, out: Path, prefix: str) -> str:
    """Capture welcome, transcript and overlay scenes of one chat page.

    Returns:
        The rendered transcript markup (reused for the share page).
    """
    shoot(page, out / f"{prefix}-1-welcome.png")
    page.evaluate(TRANSCRIPT_JS)
    page.wait_for_timeout(300)
    shoot(page, out / f"{prefix}-2-transcript.png")
    transcript = page.evaluate("() => document.getElementById('output').innerHTML")
    page.click("#model-btn")
    page.wait_for_timeout(200)
    shoot(page, out / f"{prefix}-3-model-menu.png")
    page.keyboard.press("Escape")
    page.mouse.click(5, 5)
    page.evaluate(
        "() => document.getElementById('sidebar').classList.add('open')"
    )
    # Answer with the generation of the page's latest getHistory request;
    # renderHistory drops a reply to an older request.
    page.evaluate(
        "s => { const req = window.__posted.filter("
        "m => m.type === 'getHistory').pop();"
        " window.__post({type: 'history', sessions: s, offset: 0,"
        " generation: req ? req.generation : 0}); }",
        [{**s, "preview": s["title"], "has_events": True} for s in SESSIONS],
    )
    page.wait_for_timeout(300)
    shoot(page, out / f"{prefix}-4-history.png")
    page.evaluate(
        "() => document.getElementById('sidebar').classList.remove('open')"
    )
    page.click("#more-btn")
    page.wait_for_timeout(200)
    shoot(page, out / f"{prefix}-5-more-menu.png")
    page.click("#settings-btn")
    page.wait_for_timeout(300)
    shoot(page, out / f"{prefix}-6-settings.png")
    return transcript


def capture(out: Path) -> None:
    """Capture every surface in every theme into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    server, base = start_media_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                transcript = capture_vscode(browser, base, out)
                capture_remote(browser, base, out)
                capture_share(browser, transcript, out)
                capture_viz(browser, out)
            finally:
                browser.close()
    finally:
        server.shutdown()


def capture_vscode(browser, base: str, out: Path) -> str:  # noqa: ANN001
    """Capture the chat webview in every VS Code theme.

    Returns:
        The rendered transcript markup (reused for the share page).
    """
    transcript = ""
    for theme in THEMES:
        ctx = browser.new_context(viewport={"width": 460, "height": 900})
        try:
            page = ctx.new_page()
            open_page(page, base, vscode_page_html(theme), f"vscode-{theme}")
            transcript = chat_scenes(page, out, f"vscode-{theme}")
        finally:
            ctx.close()
    return transcript


def capture_remote(browser, base: str, out: Path) -> None:  # noqa: ANN001
    """Capture the remote webapp at desktop and phone widths, dark and light."""
    for width, height, label in ((1280, 800, "desktop"), (390, 844, "phone")):
        for light in (False, True):
            ctx = browser.new_context(viewport={"width": width, "height": height})
            try:
                page = ctx.new_page()
                open_page(page, base, remote_page_html(), f"remote-{label}")
                if light:
                    # The page's own toggle (applyRemoteTheme in main.js)
                    # also swaps the highlight.js sheet and the menu label.
                    page.evaluate("() => document.getElementById('theme-btn').click()")
                    page.wait_for_function(
                        "() => document.body.classList.contains('light-theme')"
                    )
                    page.wait_for_timeout(300)
                theme = "light" if light else "dark"
                chat_scenes(page, out, f"remote-{label}-{theme}")
            finally:
                ctx.close()


def capture_share(browser, transcript: str, out: Path) -> None:  # noqa: ANN001
    """Capture the shared-chat page in its dark and light themes."""
    share_html = web_server._build_share_page(
        "Design tokens",
        '<div class="share-task">' + transcript + "</div>",
    )
    ctx = browser.new_context(viewport={"width": 900, "height": 900})
    try:
        page = ctx.new_page()
        page.set_content(share_html, wait_until="load")
        shoot(page, out / "share-dark.png")
        page.click("#share-theme-btn")
        page.wait_for_timeout(200)
        shoot(page, out / "share-light.png")
    finally:
        ctx.close()


def capture_viz(browser, out: Path) -> None:  # noqa: ANN001
    """Screenshot the trajectory visualizer with one sample trajectory."""
    payload = {
        "name": "SorcarAgent", "id": 1, "model": "claude-opus-5-5",
        "run_start_timestamp": 1790448000, "run_end_timestamp": 1790448090,
        "command": "Refactor main.css onto design tokens", "step_count": 3,
        "messages": [
            {"role": "user", "content": "Refactor main.css onto tokens."},
            {"role": "assistant", "content": "Counting radius values:\n"
             "```bash\ngrep -c border-radius media/main.css\n```"},
            {"role": "user", "content": "Tool result: 88"},
        ],
    }
    previous_dir = viz_server.ARTIFACT_DIR
    with tempfile.TemporaryDirectory(prefix="kiss-viz-") as artifact:
        jobs = Path(artifact) / "jobs"
        traj = jobs / "job_2026_09_26_12_00_00_1" / "trajectories"
        traj.mkdir(parents=True)
        (traj / "trajectory_1790448000.yaml").write_text(
            yaml.safe_dump(payload), encoding="utf-8",
        )
        viz_server.ARTIFACT_DIR = jobs
        srv = make_server("127.0.0.1", 0, viz_server.app)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        try:
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{srv.server_port}/", wait_until="networkidle")
            page.wait_for_timeout(300)
            for selector in ("#jobs-list .list-item", "#trajectories-list .list-item"):
                items = page.locator(selector)
                if items.count():
                    items.first.click()
                    page.wait_for_timeout(400)
            shoot(page, out / "viz-trajectory.png")
        finally:
            ctx.close()
            srv.shutdown()
            viz_server.ARTIFACT_DIR = previous_dir


def main() -> None:
    """Parse the output directory argument and capture all screenshots."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    capture(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: remote-webapp event panels match the VS Code extension.

The chat webview's EVENT PANELS (thinking, assistant text, tool calls,
tool results, bash output, nested-agent panels, system/prompt panels,
merge review, result cards, timing badges) and the FIXED TASK PANEL
(``#task-panel``) must render with the SAME style, fonts, and format on
the remote webapp (served by ``RemoteAccessServer``) as in the VS Code
extension webview.  Both pages share ``chat.html`` + ``main.css`` +
``main.js``; the remote page additionally loads ``remote-codex.css``
(scoped under ``body.remote-chat``) which must restyle only the remote
CHROME (tab bar, composer, sidebar, history rows, settings, modals) —
never the event panels or the task panel.

Reproduction/pinning:

* Static test: ``remote-codex.css`` must contain NO rule targeting an
  event-panel selector or ``#task-panel``.
* Static test: the remote page must define the same ``--vscode-*``
  typography variables this test injects into the extension-reference
  page (including ``--vscode-editor-font-family``, which the VS Code
  webview host always provides and monospace panel content relies on).
* Live test: an extension-reference page (the exact ``chat.html``
  substitutions of ``SorcarTab.ts::buildChatHtml`` + the VS Code
  variables) and the production ``RemoteAccessServer`` page render the
  same transcript in headless Chromium; the COMPUTED font family/size/
  style/weight, text-transform, letter-spacing, border radius/width/
  style and padding of every event panel and the task panel must be
  IDENTICAL, and the panels' horizontal format (gaps to the container
  edges) must match.  Screenshots of both pages are saved for visual
  validation.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page, ViewportSize, sync_playwright

MEDIA_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
)
CODEX_CSS = MEDIA_DIR / "remote-codex.css"
WEB_SERVER_PY = (
    Path(__file__).resolve().parents[3] / "server" / "web_server.py"
)

# The typography-relevant --vscode-* variables that the VS Code webview
# host provides natively and web_server.py must inject verbatim on the
# remote page.  The extension-reference page below injects the same
# block, so any computed-style difference between the two pages can
# only come from CSS rule differences (i.e. remote-codex.css).
VSCODE_VARS = {
    "--vscode-font-size": "16px",
    "--vscode-font-family": (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif"
    ),
    "--vscode-editor-font-size": "16px",
    "--vscode-editor-font-family": (
        "Menlo, Monaco, 'Courier New', monospace"
    ),
    "--vscode-editor-background": "#1e1e1e",
    "--vscode-editor-foreground": "#cccccc",
    "--vscode-descriptionForeground": "#8b8b8b",
    "--vscode-panel-border": "#80808059",
    "--vscode-sideBar-background": "#252526",
    "--vscode-textLink-foreground": "#3794ff",
    "--vscode-terminal-ansiRed": "#f44747",
    "--vscode-terminal-ansiGreen": "#6a9955",
    "--vscode-terminal-ansiYellow": "#d7ba7d",
    "--vscode-terminal-ansiBlue": "#569cd6",
    "--vscode-terminal-ansiMagenta": "#c586c0",
    "--vscode-terminal-ansiCyan": "#4ec9b0",
}

# Selector fragments that identify EVENT PANELS or the FIXED TASK
# PANEL.  remote-codex.css must not target any of them.
FORBIDDEN_SELECTOR_PATTERNS = [
    r"#task-panel",
    r"#output",
    r"\.tc\b",
    r"\.tc-",
    r"\.tr\b",
    r"\.tr\.",
    r"\.rc\b",
    r"\.rc-",
    r"\.rs\b",
    r"\.rl\b",
    r"\.think\b",
    r"\.cnt\b",
    r"\.txt\b",
    r"\.md-body",
    r"\.tp\b",
    r"\.sys\b",
    r"\.llm-panel",
    r"\.bash-panel",
    r"\.system-prompt",
    r"\.prompt\b",
    r"\.prompt-",
    r"\.merge-",
    r"\.wt-result",
    r"\.panel-time",
    r"\.collapse-preview",
    r"\.adjacent-",
    r"\.ev\b",
]


def _css_selectors(css: str) -> list[str]:
    """Return every selector in *css* (top-level and inside @media)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    # Drop @media prologues so their inner rules parse like top-level.
    css = re.sub(r"@media[^{]*\{", "", css)
    selectors: list[str] = []
    for m in re.finditer(r"([^{}]+)\{[^{}]*\}", css):
        sel = m.group(1).strip()
        if sel:
            selectors.append(sel)
    return selectors


@pytest.mark.parametrize("pattern", FORBIDDEN_SELECTOR_PATTERNS)
def test_remote_codex_does_not_restyle_event_or_task_panels(
    pattern: str,
) -> None:
    """remote-codex.css must not target event panels or #task-panel, so
    those panels inherit the extension's main.css look verbatim."""
    offenders = [
        sel
        for sel in _css_selectors(CODEX_CSS.read_text(encoding="utf-8"))
        if re.search(pattern, sel)
    ]
    assert not offenders, (
        f"remote-codex.css must not restyle event panels or the task "
        f"panel (pattern {pattern!r}); offending selectors: {offenders}"
    )


@pytest.mark.parametrize(("name", "value"), sorted(VSCODE_VARS.items()))
def test_remote_page_defines_vscode_typography_vars(
    name: str, value: str
) -> None:
    """The remote page built by web_server.py must inject the same
    --vscode-* variables that the VS Code webview host provides (and
    that this test's extension reference page uses), so fonts resolve
    identically."""
    from kiss.server.web_server import _build_html

    html = _build_html()
    assert f"{name}: {value};" in html, (
        f"the remote page must define {name}: {value}"
    )


def _vars_style_block() -> str:
    decls = "\n".join(f"      {k}: {v};" for k, v in VSCODE_VARS.items())
    return (
        "<style>\n"
        "    html, body { height: 100%; margin: 0; padding: 0; "
        "overflow: hidden; }\n"
        "    :root {\n" + decls + "\n    }\n  </style>"
    )


def _build_extension_html() -> str:
    """Build the chat page exactly as SorcarTab.ts::buildChatHtml does
    (no remote-codex.css, no ``remote-chat`` body class), plus the
    --vscode-* variables the webview host would provide and a stub
    ``acquireVsCodeApi`` in place of the real webview bridge."""
    tpl = (MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
    shim = (
        "<script>window.acquireVsCodeApi = () => ({"
        "postMessage() {}, getState() { return undefined; }, "
        "setState() {}});</script>"
    )
    subs = {
        "VIEWPORT": "width=device-width, initial-scale=1.0",
        "CSP_META": "",
        "STYLE_HREF": "main.css",
        "HLJS_CSS_HREF": "highlight-github-dark.min.css",
        "HEAD_STYLE": _vars_style_block(),
        "BODY_CLASS_ATTR": "",
        "INPUT_PLACEHOLDER": "Ask anything... (@ for files)",
        "ENTERKEYHINT": "",
        "MODEL_NAME": "test-model",
        "VERSION_SUFFIX": "",
        "AUTH_MODAL": "",
        "NONCE_ATTR": "",
        "HLJS_SRC": "highlight.min.js",
        "MARKED_SRC": "marked.min.js",
        "API_SRC": "api.js",
        "PANEL_COPY_SRC": "panelCopy.js",
        "MAIN_SRC": "main.js",
        "SHIM_SCRIPT": shim,
        "TRICKS_JSON": "[]",
        "TIPS_JSON": json.dumps({"tips": [], "show": False}),
        "TIPS_SRC": "tips.js",
        "VOICE_SRC": "voice.js",
        "VOICE_CONFIG": json.dumps({"mode": "webview"}),
    }
    return re.sub(
        r"\{\{([A-Z_]+)\}\}",
        lambda m: subs.get(m.group(1), m.group(0)),
        tpl,
    )


class _ExtensionPageHandler(http.server.SimpleHTTPRequestHandler):
    """Serve media/ files, with ``/`` mapped to the extension page."""

    extension_html = b""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(MEDIA_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        """Serve the generated extension page at ``/``."""
        if self.path in ("/", "/index.html"):
            body = type(self).extension_html
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


# Renders one instance of every event-panel type plus the pinned task
# panel, exactly like a live transcript would.  Everything main.js can
# render from an output event goes through the PRODUCTION renderer
# (window._testApi.processEvent -> handleOutputEvent): the summary
# panel, the warning/error tool-result variants, the autocommit
# ok/err worktree lines, and the result card.  The remaining panels
# (thinking, markdown text, bash tool call, nested-agent panel,
# prompts, merge review, adjacent-task placeholders) are injected as
# the exact DOM main.js produces for them.
_INJECT_PAGE_JS = r"""
(() => {
  const out = document.getElementById('output');
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  const app = document.getElementById('app');
  if (app) app.style.display = '';
  const loading = document.getElementById('kiss-server-loading');
  if (loading) loading.style.display = 'none';

  document.getElementById('task-panel-text').textContent =
    'Fix the flux capacitor so the DeLorean can time travel again.';
  document.getElementById('task-panel').classList.add('visible');

  if (!window._testApi || typeof window._testApi.processEvent !== 'function') {
    throw new Error('production output renderer is unavailable');
  }
  // Rendered FIRST so the summary panel adopts no earlier siblings.
  window._testApi.processEvent({
    type: 'tool_call',
    name: 'summary',
    description: 'Recap of the last few steps.',
  });

  out.insertAdjacentHTML('beforeend', `
    <div class="ev think">
      <div class="lbl"><span class="arrow">\u25BE</span> Thinking</div>
      <div class="cnt">Reasoning about panel typography.</div>
    </div>
    <div class="ev txt md-body">Plain assistant text with
      <code>inline code</code> and a table.
      <pre><code class="hljs language-python">print("x")</code></pre>
      <table><tr><th>h</th></tr><tr><td>cell</td></tr></table>
    </div>
    <div class="ev tc collapsed">
      <div class="tc-h collapse-header">
        <span class="collapse-chv">\u25BE</span>Read
        <span class="collapse-preview">media/main.css</span></div>
      <div class="tc-b">collapsed body</div>
    </div>
    <div class="ev tc tc-bash">
      <div class="tc-h tc-h-bash collapse-header">
        <span class="collapse-chv">\u25BE</span>Bash
        <span class="collapse-preview">ls -la</span></div>
      <div class="tc-b">
        <div class="tc-arg"><span class="tc-arg-name">path:</span>
          <span class="tp">media/main.css</span></div>
        <pre><code class="language-bash">ls</code></pre>
      </div>
      <div class="bash-panel"><div class="bash-panel-content">main.css
</div></div>
      <div class="tr"><div class="rl">Result</div>
        <div class="tr-content">ok</div></div>
      <div class="panel-time">2.1s</div>
    </div>
    <div class="ev sys">system output line</div>
    <div class="llm-panel">
      <div class="llm-panel-hdr">Thoughts</div>
      <div class="ev txt">nested agent text</div>
    </div>
    <div class="ev system-prompt">
      <div class="system-prompt-h">System prompt</div>
      <div class="system-prompt-body md-body">system prompt body</div>
    </div>
    <div class="ev prompt">
      <div class="prompt-h">Prompt</div>
      <div class="prompt-body md-body">prompt body</div>
    </div>
    <div class="ev merge-info">
      <div class="merge-info-hdr">\u2731 Reviewing 1 change(s)</div>
      <div class="merge-info-body">Red = old, Green = new.</div>
      <div class="merge-file-diff">
        <div class="merge-file-name">a.py</div>
        <pre class="merge-ctx">context line</pre>
        <pre class="merge-hunk"><span class="merge-hunk-label">Hunk
          1</span>-old\n+new</pre>
      </div>
    </div>
    <div class="adjacent-loader">Loading previous task\u2026</div>
    <div class="adjacent-task">
      <div class="adjacent-task-placeholder">No newer task</div>
    </div>`);

  // Production-rendered variants: warning / error tool results,
  // autocommit ok + err worktree lines, and the result card.
  window._testApi.processEvent({type: 'warning', message: 'low disk'});
  window._testApi.processEvent({type: 'error', text: 'boom'});
  window._testApi.processEvent({
    type: 'autocommit_done', success: true, message: 'Committed 2 files.',
  });
  window._testApi.processEvent({
    type: 'autocommit_done', success: false, message: 'Commit failed.',
  });
  window._testApi.processEvent({
    type: 'result',
    success: false,
    is_continue: true,
    summary: 'result card body text',
    total_tokens: 1200,
    cost: '$0.01',
  });

  return out.children.length;
})()
"""

# Event-panel probes: selector -> probe key.
PANEL_PROBES = {
    "think": ".ev.think",
    "thinkLbl": ".ev.think .lbl",
    "thinkCnt": ".ev.think .cnt",
    "txt": ".ev.txt",
    "txtCode": ".ev.txt code",
    "txtPre": ".ev.txt pre",
    "txtPreCode": ".ev.txt pre code",
    "txtTh": ".ev.txt th",
    "txtTd": ".ev.txt td",
    "tc": ".ev.tc",
    "tcH": ".tc-h.tc-h-bash",
    "tcB": ".tc-b",
    "tcArg": ".tc-arg",
    "tp": ".tp",
    "tcPre": ".tc-b pre",
    "bashPanel": ".tc > .bash-panel",
    "bashContent": ".bash-panel-content",
    "tr": ".tc > .tr",
    "trRl": ".tr .rl",
    "trContent": ".tr .tr-content",
    "panelTime": ".panel-time",
    "collapsePreview": ".collapse-preview",
    "sys": ".ev.sys",
    "llmPanel": ".llm-panel",
    "llmPanelHdr": ".llm-panel-hdr",
    "llmTxt": ".llm-panel .txt",
    "systemPrompt": ".ev.system-prompt",
    "systemPromptH": ".system-prompt-h",
    "systemPromptBody": ".system-prompt-body",
    "promptH": ".prompt-h",
    "promptBody": ".prompt-body",
    "mergeInfo": ".ev.merge-info",
    "mergeInfoHdr": ".merge-info-hdr",
    "mergeInfoBody": ".merge-info-body",
    "mergeCtx": ".merge-ctx",
    "mergeHunk": ".merge-hunk",
    "wtResultOk": ".ev.wt-result-ok",
    "wtResultErr": ".ev.wt-result-err",
    "trWarn": ".ev.tr.warn",
    "trWarnStrong": ".ev.tr.warn strong",
    "trErr": ".ev.tr.err",
    "summary": ".tc.tc-summary",
    "summaryDesc": ".tc-summary-desc",
    "summaryHint": ".tc-summary-hint",
    "collapsedTc": ".ev.tc.collapsed",
    "collapsedPreview": ".ev.tc.collapsed .collapse-preview",
    "adjacentLoader": ".adjacent-loader",
    "adjacentPlaceholder": ".adjacent-task-placeholder",
    "rc": ".rc",
    "rcH3": ".rc-h h3",
    "rs": ".rs",
    "rsB": ".rs b",
    "rcBody": ".rc-body",
    "rcStatus": ".rc-status",
    "taskPanel": "#task-panel",
    "taskPanelText": "#task-panel-text",
    "taskPanelCopy": "#task-panel-copy",
    "taskPanelDrawerBtn": "#task-panel-drawer-btn",
}

# Style/font/format properties that must be identical between the
# extension webview and the remote webapp (colors are theme-dependent
# and intentionally excluded).
COMPARED_PROPS = [
    "fontFamily",
    "fontSize",
    "fontStyle",
    "fontWeight",
    "textTransform",
    "letterSpacing",
    "borderTopLeftRadius",
    "borderTopWidth",
    "borderTopStyle",
    "paddingTop",
    "paddingLeft",
]

_PROBE_STYLES_JS = (
    "(() => { const probes = "
    + json.dumps(PANEL_PROBES)
    + "; const props = "
    + json.dumps(COMPARED_PROPS)
    + r""";
  const styles = {};
  for (const key of Object.keys(probes)) {
    const el = document.querySelector(probes[key]);
    if (!el) { styles[key] = 'MISSING'; continue; }
    const cs = getComputedStyle(el);
    styles[key] = props.map(p => p + '=' + cs[p]).join(' | ');
  }
  const app = document.getElementById('app');
  const out = document.getElementById('output');
  const tp = document.getElementById('task-panel');
  const tc = document.querySelector('.ev.tc');
  const appRect = app.getBoundingClientRect();
  const outRect = out.getBoundingClientRect();
  const tpRect = tp.getBoundingClientRect();
  const tcRect = tc.getBoundingClientRect();
  const tpCs = getComputedStyle(tp);
  // Collapsed drawer state: toggle, measure, restore (probes run
  // after the screenshot, so the toggle never shows up in it).
  tp.classList.add('drawer-collapsed');
  const collapsedCs = getComputedStyle(tp);
  const collapsedPadding =
    collapsedCs.paddingTop + ' ' + collapsedCs.paddingBottom;
  tp.classList.remove('drawer-collapsed');
  return {
    styles,
    taskPanelCollapsedPadding: collapsedPadding,
    taskPanelBorderColor: tpCs.borderTopColor,
    taskPanelMaxWidth: tpCs.maxWidth,
    taskPanelGapLeft: tpRect.left - appRect.left,
    taskPanelGapRight: appRect.right - tpRect.right,
    eventGapLeft: tcRect.left - outRect.left,
    eventGapRight: outRect.right - tcRect.right,
  };
})()"""
)


def _start_extension_server(
    ready: threading.Event,
    done: threading.Event,
    state: dict[str, object],
) -> None:
    """Serve the extension-reference page until *done* is set."""
    _ExtensionPageHandler.extension_html = _build_extension_html().encode(
        "utf-8"
    )
    try:
        httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _ExtensionPageHandler
        )
    except BaseException as exc:  # pragma: no cover - defensive
        state["error"] = exc
        ready.set()
        return
    state["port"] = httpd.server_address[1]
    ready.set()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    done.wait()
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=10)


def _start_remote_server(
    tmp_path: Path,
    ready: threading.Event,
    done: threading.Event,
    state: dict[str, object],
) -> None:
    """Run the production RemoteAccessServer until *done* is set."""
    from kiss.server.web_server import (
        RemoteAccessServer,
        _generate_self_signed_cert,
    )

    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)

    async def scenario() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=str(tmp_path),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp_path / "remote-url.json",
            uds_path=tmp_path / "sorcar.sock",
        )
        started = False
        try:
            await server.start_async()
            started = True
            assert server._ws_server is not None
            state["port"] = next(
                iter(server._ws_server.sockets)
            ).getsockname()[1]
            ready.set()
            while not done.is_set():
                await asyncio.sleep(0.02)
        except BaseException as exc:  # pragma: no cover - defensive
            state["error"] = exc
            ready.set()
        finally:
            if started:
                await server.stop_async()

    asyncio.run(scenario())


def _capture_page(
    page: Page, url: str, screenshot: Path
) -> dict[str, Any]:
    """Load *url*, inject the transcript, screenshot, return probes."""
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#output", state="attached")
    count = page.evaluate(_INJECT_PAGE_JS)
    assert count >= 16, "transcript injection failed"
    page.wait_for_selector(".rc", state="attached")
    page.screenshot(path=str(screenshot), full_page=False)
    probes: dict[str, Any] = page.evaluate(_PROBE_STYLES_JS)
    return probes


# Viewports the parity must hold at: the extension webview's typical
# desktop panel width AND a phone-sized viewport (remote-codex.css
# historically had <=600px event-panel overrides; parity must hold on
# both sides of that media query).
VIEWPORTS: dict[str, ViewportSize] = {
    "desktop": ViewportSize(width=1400, height=900),
    "mobile": ViewportSize(width=480, height=900),
}


def _assert_probe_parity(
    ext_probes: dict[str, Any], rem_probes: dict[str, Any], label: str
) -> None:
    """Assert the remote page's probes equal the extension's."""
    ext_styles = ext_probes["styles"]
    rem_styles = rem_probes["styles"]
    missing = sorted(
        k
        for k in PANEL_PROBES
        if ext_styles[k] == "MISSING" or rem_styles[k] == "MISSING"
    )
    assert not missing, (
        f"[{label}] probe elements missing from a page: {missing}"
    )

    mismatched = {
        k: {"extension": ext_styles[k], "remote": rem_styles[k]}
        for k in PANEL_PROBES
        if ext_styles[k] != rem_styles[k]
    }
    assert not mismatched, (
        f"[{label}] remote event/task panels must match the "
        "extension's computed style, fonts, and format; mismatches:\n"
        + json.dumps(mismatched, indent=2)
    )

    for scalar in (
        "taskPanelCollapsedPadding",
        "taskPanelBorderColor",
        "taskPanelMaxWidth",
    ):
        assert rem_probes[scalar] == ext_probes[scalar], (
            f"[{label}] {scalar}: the fixed task panel must keep the "
            f"extension's value: {ext_probes[scalar]!r} != "
            f"{rem_probes[scalar]!r}"
        )
    for key in (
        "taskPanelGapLeft",
        "taskPanelGapRight",
        "eventGapLeft",
        "eventGapRight",
    ):
        ext_gap = float(ext_probes[key])
        rem_gap = float(rem_probes[key])
        assert abs(ext_gap - rem_gap) <= 2.0, (
            f"[{label}] {key}: remote panel format must match the "
            f"extension (extension={ext_gap}px, remote={rem_gap}px)"
        )


@pytest.mark.timeout(240)
def test_live_remote_panels_match_extension(tmp_path: Path) -> None:
    """The remote webapp's event panels and fixed task panel render
    with the extension's computed style, fonts, and format, on both a
    desktop and a phone-sized viewport.

    Colors of the panel PALETTE (backgrounds, accents) are excluded
    from the probe comparison: the extension inherits them from the
    live VS Code theme's ``--vscode-*`` tokens while the remote page
    ships a fixed dark palette, so they legitimately differ per theme.
    The task panel's border color IS compared because main.css pins it
    to a literal (theme-independent) yellow.
    """
    shot_dir = Path(
        os.environ.get("KISS_PANEL_SHOT_DIR", str(tmp_path))
    )
    shot_dir.mkdir(parents=True, exist_ok=True)

    ext_ready = threading.Event()
    ext_done = threading.Event()
    ext_state: dict[str, object] = {}
    ext_thread = threading.Thread(
        target=_start_extension_server,
        args=(ext_ready, ext_done, ext_state),
        daemon=True,
    )
    ext_thread.start()

    rem_ready = threading.Event()
    rem_done = threading.Event()
    rem_state: dict[str, object] = {}
    rem_thread = threading.Thread(
        target=_start_remote_server,
        args=(tmp_path, rem_ready, rem_done, rem_state),
        daemon=True,
    )
    rem_thread.start()

    try:
        assert ext_ready.wait(30), "extension page server failed to start"
        assert rem_ready.wait(30), "RemoteAccessServer failed to start"
        for st, name in ((ext_state, "extension"), (rem_state, "remote")):
            err = st.get("error")
            if isinstance(err, BaseException):
                raise AssertionError(f"{name} server startup failed") from err

        probes: dict[str, dict[str, dict[str, Any]]] = {}
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--ignore-certificate-errors"]
            )
            try:
                for label, viewport in VIEWPORTS.items():
                    page = browser.new_page(
                        ignore_https_errors=True, viewport=viewport
                    )
                    probes[label] = {
                        "extension": _capture_page(
                            page,
                            f"http://127.0.0.1:{ext_state['port']}/",
                            shot_dir / f"extension-panels-{label}.png",
                        ),
                        "remote": _capture_page(
                            page,
                            f"https://127.0.0.1:{rem_state['port']}/",
                            shot_dir / f"remote-panels-{label}.png",
                        ),
                    }
                    page.close()
            finally:
                browser.close()
    finally:
        ext_done.set()
        rem_done.set()
        ext_thread.join(timeout=30)
        rem_thread.join(timeout=30)
    assert not ext_thread.is_alive(), "extension page server failed to stop"
    assert not rem_thread.is_alive(), "RemoteAccessServer failed to stop"
    for st, name in ((ext_state, "extension"), (rem_state, "remote")):
        err = st.get("error")
        if isinstance(err, BaseException):
            raise AssertionError(f"{name} server thread failed") from err

    for label in VIEWPORTS:
        _assert_probe_parity(
            probes[label]["extension"], probes[label]["remote"], label
        )

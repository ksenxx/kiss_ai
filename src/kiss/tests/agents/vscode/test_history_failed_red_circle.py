# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the task history panel renders a red circle next to
every failed task.

This is the **frontend** half of the "red marker for failed tasks in
the task history sidebar" feature.  The backend half — that
``getHistory`` event sets ``session.failed = True`` for every failed
task category — is covered by ``test_history_failed_flag.py``.  This
test exercises the rest of the pipeline by booting a real headless
Chromium via Playwright and driving the real ``renderHistory``
function shipped in ``media/main.js`` against the real
``media/main.css``:

* it injects a minimal stand-in for the chrome ``main.js`` touches at
  IIFE setup time (so the IIFE runs to completion and registers the
  ``message`` listener that ``renderHistory`` is dispatched from),
* it posts the exact ``{type: 'history', sessions: [...]}`` event the
  webview receives from the extension host with a mix of failed,
  running and completed sessions, and
* it asserts that the failed row gets a ``.sidebar-item-failed``
  element that is actually painted, has the right geometry (8 × 8 px,
  rounded), the right colour (the failure red ``#d32f2f`` ≡
  ``rgb(211, 47, 47)``), and the right accessibility metadata
  (``aria-label='Task failed'``), while non-failed rows do not receive
  it.

Two visibility regressions are also covered:

* unchecking the "Errored" filter checkbox hides the failed row, and
  re-checking it brings it back (guards against a regression where the
  dot exists but the row is hidden so the user never sees it), and
* a failed task whose row is also flagged ``is_running`` (the live
  task is still alive) must NOT carry a red dot — running tasks own
  the row marker.
"""

from __future__ import annotations

import json
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
_JS = _MEDIA_DIR / "main.js"
_HTML = _MEDIA_DIR / "chat.html"


def _build_test_page() -> str:
    """Return a self-contained HTML page that loads the real CSS+JS.

    The page mirrors the production ``media/chat.html`` body so the
    ``main.js`` IIFE finds every ``getElementById`` it issues during
    ``setupEventListeners`` and ``init``.  ``acquireVsCodeApi``,
    ``hljs``, ``marked``, ``PanelCopy`` and ``__TRICKS__`` are stubbed
    because they are provided by host-side scripts that are not loaded
    here.  Once the IIFE finishes, ``window.__post(ev)`` dispatches a
    ``message`` event identical to what the webview receives from the
    extension host.
    """
    css = _CSS.read_text(encoding="utf-8")
    js = _JS.read_text(encoding="utf-8")
    html = _HTML.read_text(encoding="utf-8")
    # Strip the template placeholders that the extension fills in at
    # webview load time — we don't need any of them for this test, and
    # leaving them in would inject invalid markup (e.g. ``<link
    # href="{{STYLE_HREF}}">`` resolves to a 404 link).
    body_start = html.find("<body")
    body_open_end = html.find(">", body_start) + 1
    body_end = html.find("</body>")
    body = html[body_open_end:body_end]
    # Drop the script tags — we inline ``main.js`` ourselves below and
    # the others (hljs / marked / panelCopy / demo / shim / nonce) are
    # either stubbed or irrelevant for history rendering.
    body = "\n".join(
        line for line in body.splitlines()
        if "<script" not in line and "</script>" not in line
    )
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
  <title>history failed red circle test</title>
</head>
<body>
{body}
  <script>
    // Stub the host APIs main.js depends on at IIFE setup time.
    // None of these are exercised by the history-rendering pipeline,
    // they only need to exist so the IIFE runs to completion and
    // registers the ``message`` listener.
    window.acquireVsCodeApi = function () {{
      return {{
        postMessage: function () {{}},
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
    // Dispatch a ``message`` event identical to what the webview
    // receives from the extension host.
    window.__post = function (ev) {{
      window.dispatchEvent(new MessageEvent('message', {{ data: ev }}));
    }};
    // Capture any IIFE-time error so a future regression that breaks
    // setup is reported with a precise message rather than as a
    // silent "rows never appear" timeout.
    window.__iifeError = null;
    window.addEventListener('error', function (ev) {{
      if (!window.__iifeError) window.__iifeError = String(ev.error || ev.message);
    }});
  </script>
  <script>{js}</script>
</body>
</html>
"""


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _open_history_page(_browser):
    """Open the test harness and return ``(context, page)``.

    The viewport is tall enough that all sample rows fit without
    overflow, so per-row visibility checks reflect the feature under
    test and not viewport clipping.
    """
    context = _browser.new_context(viewport={"width": 480, "height": 900})
    page = context.new_page()
    page.set_content(_build_test_page(), wait_until="load")
    page.wait_for_function(
        "document.getElementById('history-list') !== null",
        timeout=5000,
    )
    # The sidebar ships hidden behind ``transform: translateX(-100%)``
    # and is revealed by adding ``.open``; in production the user
    # clicks the history button to add that class.  Add it here so
    # the rendered DOM has real layout boxes (otherwise the row /
    # dot inherit zero geometry from the slid-off container).
    #
    # ``#app`` ships with an inline ``style="display:none;"`` and is
    # only revealed by ``setServerLoading(false)`` once the kiss-web
    # daemon socket connects.  In this harness no daemon is ever
    # connected, so we flip it manually — otherwise every descendant
    # (including the failed dot) inherits a 0×0 layout box and the
    # test cannot verify that the red circle is actually painted.
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


def _sample_sessions() -> list[dict]:
    """Return a mixed batch of one failed, one running and one
    completed session as the server would emit them."""
    return [
        {
            "id": "chat-fail",
            "task_id": 1001,
            "title": "failing task",
            "preview": "failing task",
            "timestamp": 1700000000,
            "has_events": True,
            "failed": True,
            "is_running": False,
            "tokens": 0, "cost": 0.0, "steps": 0,
            "is_favorite": False, "work_dir": "",
            "model": "", "is_worktree": False,
            "is_parallel": False, "auto_commit_mode": False,
            "startTs": 1700000000000, "endTs": 1700000001000,
        },
        {
            "id": "chat-run",
            "task_id": 1002,
            "title": "running task",
            "preview": "running task",
            "timestamp": 1700000100,
            "has_events": True,
            "failed": False,
            "is_running": True,
            "tokens": 0, "cost": 0.0, "steps": 0,
            "is_favorite": False, "work_dir": "",
            "model": "", "is_worktree": False,
            "is_parallel": False, "auto_commit_mode": False,
            "startTs": 1700000100000, "endTs": 0,
        },
        {
            "id": "chat-ok",
            "task_id": 1003,
            "title": "successful task",
            "preview": "successful task",
            "timestamp": 1700000200,
            "has_events": True,
            "failed": False,
            "is_running": False,
            "tokens": 0, "cost": 0.0, "steps": 0,
            "is_favorite": False, "work_dir": "",
            "model": "", "is_worktree": False,
            "is_parallel": False, "auto_commit_mode": False,
            "startTs": 1700000200000, "endTs": 1700000201000,
        },
    ]


def _post_history(page, sessions: list[dict]) -> None:
    """Dispatch a ``message`` event with a ``history`` payload and wait
    for ``renderHistory`` to materialise the rows in the DOM."""
    ev = {
        "type": "history",
        "sessions": sessions,
        "offset": 0,
        "generation": 0,
    }
    page.evaluate("ev => window.__post(ev)", ev)
    page.wait_for_function(
        f"document.querySelectorAll('#history-list .sidebar-item').length === {len(sessions)}",
        timeout=5000,
    )


def test_failed_session_renders_red_circle(_browser) -> None:
    """Every ``s.failed`` session must render exactly one visible red
    circle (``.sidebar-item-failed``) inside its row.

    Asserts the element exists, is painted, and matches the intended
    geometry / colour from ``main.css``.
    """
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        info = page.evaluate(
            """
            () => {
              const rows = Array.from(
                document.querySelectorAll('#history-list .sidebar-item'),
              );
              return rows.map(r => {
                const dot = r.querySelector('.sidebar-item-failed');
                const txt = r.querySelector('.sidebar-item-text');
                const out = {
                  text: txt ? txt.textContent : null,
                  category: r.dataset.category || null,
                  hasFailedDot: !!dot,
                  hasRunningDot: !!r.querySelector('.sidebar-item-running'),
                };
                if (dot) {
                  const cs = getComputedStyle(dot);
                  out.dot = {
                    width: cs.width,
                    height: cs.height,
                    borderRadius: cs.borderRadius,
                    background: cs.backgroundColor,
                    visibility: cs.visibility,
                    display: cs.display,
                    opacity: cs.opacity,
                    offsetWidth: dot.offsetWidth,
                    offsetHeight: dot.offsetHeight,
                    tooltip: dot.dataset.tooltip,
                    ariaLabel: dot.getAttribute('aria-label'),
                  };
                }
                return out;
              });
            }
            """,
        )
        by_text = {r["text"]: r for r in info}

        # Failed task — must have a red dot.
        fail = by_text["failing task"]
        assert fail["category"] == "errors", (
            f"failed row miscategorised: {fail['category']!r}"
        )
        assert fail["hasFailedDot"], (
            "failed task did not render a .sidebar-item-failed element; "
            f"row info: {json.dumps(fail, indent=2)}"
        )
        dot = fail["dot"]
        assert dot["width"] == "8px" and dot["height"] == "8px", (
            f"failed dot is not 8x8: {dot['width']} x {dot['height']}"
        )
        # ``border-radius: 50%`` resolves to half the width in pixels.
        assert dot["borderRadius"] in ("4px", "50%"), (
            f"failed dot is not rounded: border-radius={dot['borderRadius']}"
        )
        assert dot["background"] == "rgb(211, 47, 47)", (
            f"failed dot is not the failure-red colour: "
            f"background-color={dot['background']!r}; expected rgb(211, 47, 47)"
        )
        assert dot["visibility"] == "visible", (
            f"failed dot is hidden: visibility={dot['visibility']!r}"
        )
        assert dot["display"] != "none", (
            f"failed dot is display:none: display={dot['display']!r}"
        )
        assert float(dot["opacity"]) > 0.0, (
            f"failed dot is fully transparent: opacity={dot['opacity']!r}"
        )
        assert dot["offsetWidth"] > 0 and dot["offsetHeight"] > 0, (
            f"failed dot has zero box: "
            f"offset={dot['offsetWidth']}x{dot['offsetHeight']}"
        )
        assert dot["tooltip"] == "Task failed", (
            f"failed dot has wrong tooltip: {dot['tooltip']!r}"
        )
        assert dot["ariaLabel"] == "Task failed", (
            f"failed dot has wrong aria-label: {dot['ariaLabel']!r}"
        )

        # Running task — green dot, NOT red.
        run = by_text["running task"]
        assert run["category"] == "running", (
            f"running row miscategorised: {run['category']!r}"
        )
        assert not run["hasFailedDot"], (
            "running task should not render a .sidebar-item-failed element"
        )
        assert run["hasRunningDot"], (
            "running task should render a .sidebar-item-running element"
        )

        # Completed task — no dot at all.
        ok = by_text["successful task"]
        assert ok["category"] == "completed", (
            f"completed row miscategorised: {ok['category']!r}"
        )
        assert not ok["hasFailedDot"], (
            "completed task should not render a .sidebar-item-failed element"
        )
        assert not ok["hasRunningDot"], (
            "completed task should not render a .sidebar-item-running element"
        )
    finally:
        context.close()


def test_errored_filter_toggle_hides_and_shows_failed_row(_browser) -> None:
    """Unchecking the "Errored" filter must hide the failed row;
    re-checking it must restore it.  This guards against a regression
    where the failed dot exists but the row itself is hidden by the
    filter bar so the user never sees the red circle.
    """
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        # Baseline — failed row visible.
        visible0 = page.evaluate(
            "() => document.querySelector("
            "'#history-list .sidebar-item[data-category=\"errors\"]'"
            ").offsetParent !== null"
        )
        assert visible0, "failed row should be visible before any filter change"

        # Uncheck "Errored" — failed row must hide.
        page.evaluate(
            "() => { const c = document.getElementById('hf-errors');"
            " c.checked = false; c.dispatchEvent(new Event('change')); }"
        )
        hidden = page.evaluate(
            "() => document.querySelector("
            "'#history-list .sidebar-item[data-category=\"errors\"]'"
            ").offsetParent === null"
        )
        assert hidden, "failed row should hide when Errored filter is unchecked"

        # Re-check "Errored" — failed row must re-appear.
        page.evaluate(
            "() => { const c = document.getElementById('hf-errors');"
            " c.checked = true; c.dispatchEvent(new Event('change')); }"
        )
        visible1 = page.evaluate(
            "() => document.querySelector("
            "'#history-list .sidebar-item[data-category=\"errors\"]'"
            ").offsetParent !== null"
        )
        assert visible1, "failed row should reappear when Errored filter is re-checked"
    finally:
        context.close()


def test_live_running_failed_task_does_not_render_red_dot(_browser) -> None:
    """A session that is both ``is_running=True`` and ``failed=True``
    must render the **green** running dot, not the red failed dot.

    The backend already guards this by setting
    ``failed = _is_failed_result(result) and not is_running`` so this
    is the frontend belt-and-braces check: even if a future backend
    change accidentally surfaces ``failed=True`` for a running row,
    ``renderHistory`` must still pick the running marker because the
    task is alive.
    """
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, [
            {
                "id": "chat-live-fail",
                "task_id": 2001,
                "title": "still running task",
                "preview": "still running task",
                "timestamp": 1700000300,
                "has_events": True,
                "failed": True,
                "is_running": True,
                "tokens": 0, "cost": 0.0, "steps": 0,
                "is_favorite": False, "work_dir": "",
                "model": "", "is_worktree": False,
                "is_parallel": False, "auto_commit_mode": False,
                "startTs": 1700000300000, "endTs": 0,
            },
        ])
        flags = page.evaluate(
            """
            () => {
              const row = document.querySelector('#history-list .sidebar-item');
              return {
                category: row.dataset.category,
                hasFailedDot: !!row.querySelector('.sidebar-item-failed'),
                hasRunningDot: !!row.querySelector('.sidebar-item-running'),
              };
            }
            """,
        )
        assert flags["category"] == "running", (
            f"running+failed row miscategorised: {flags['category']!r}"
        )
        assert not flags["hasFailedDot"], (
            "running task must not render the .sidebar-item-failed red dot"
        )
        assert flags["hasRunningDot"], (
            "running task must render the .sidebar-item-running green dot"
        )
    finally:
        context.close()

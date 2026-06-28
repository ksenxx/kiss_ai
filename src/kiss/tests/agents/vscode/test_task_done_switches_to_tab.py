# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: when a task completes in a chat tab, the webview
must automatically switch focus to that tab so the user immediately
sees the result panel.

The Sorcar chat webview has its own internal tab bar (rendered by
``media/main.js``).  Multiple chat tabs can each run their own task in
parallel.  Before this fix, when a task finished in a background tab
(i.e. a tab the user is not currently viewing), the daemon-emitted
``task_done`` event only:

  * marked the tab's ``isRunning`` flag false,
  * recorded the tab's done label / duration in the per-tab state
    object (so a later manual click would re-render the "Done (Xm Ys)"
    status), and
  * updated the in-tab-bar status dot (red ●  / green ●).

It did NOT switch ``activeTabId`` to the just-finished tab, so a user
who started a long-running task in tab A and then opened tab B to
keep chatting would have to manually click back to tab A to see the
result.  This violates the user-facing contract: **when a task
completes in a tab, the webview must switch to that tab.**

The tests below load the real ``media/main.js`` into a headless
Chromium (Playwright) so the real event handlers, the real DOM, and
the real per-tab state run end-to-end.

The fixture-injected synthetic page mimics the same harness used by
``test_history_running_green_circle.py`` so the behaviour is exercised
against the shipped JS/CSS verbatim.
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
_CSS = _MEDIA_DIR / "main.css"
_JS = _MEDIA_DIR / "main.js"
_HTML = _MEDIA_DIR / "chat.html"


def _build_test_page() -> str:
    """Return a self-contained HTML page that loads the real CSS+JS.

    Mirrors :func:`_build_test_page` from
    ``test_history_running_green_circle.py`` so the harness is
    drop-in compatible with the rest of the VS Code webview test
    suite.
    """
    css = _CSS.read_text(encoding="utf-8")
    js = _JS.read_text(encoding="utf-8")
    html = _HTML.read_text(encoding="utf-8")
    body_start = html.find("<body")
    body_open_end = html.find(">", body_start) + 1
    body_end = html.find("</body>")
    body = html[body_open_end:body_end]
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
  <title>task_done switches to that tab test</title>
</head>
<body>
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
    window.__post = function (ev) {{
      window.dispatchEvent(new MessageEvent('message', {{ data: ev }}));
    }};
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


def _open_page(_browser, width: int = 800, height: int = 900):
    """Open the test harness and return ``(context, page)``."""
    context = _browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    page.set_content(_build_test_page(), wait_until="load")
    page.wait_for_function(
        "document.getElementById('tab-list') !== null",
        timeout=5000,
    )
    page.evaluate(
        "() => {"
        " document.getElementById('app').style.display = '';"
        " const ov = document.getElementById('kiss-server-loading');"
        " if (ov) ov.style.display = 'none';"
        " }"
    )
    iife_err = page.evaluate("() => window.__iifeError")
    if iife_err:
        pytest.fail(f"main.js IIFE setup raised: {iife_err}")
    # Wait until the initial tab is materialised in the tab bar (the
    # IIFE creates the first tab synchronously on init).
    page.wait_for_function(
        "document.querySelectorAll("
        "'#tab-list .chat-tab[data-tab-id]'"
        ").length >= 1",
        timeout=5000,
    )
    return context, page


def _make_two_tabs(page) -> tuple[str, str]:
    """Allocate two chat tabs and return ``(tab_a, tab_b)`` ids."""
    # Use the demo API exposed by main.js to create a second tab.  The
    # IIFE already minted the first one on init.
    page.evaluate("() => window._demoApi.createNewTab()")
    page.wait_for_function(
        "document.querySelectorAll("
        "'#tab-list .chat-tab[data-tab-id]'"
        ").length === 2",
        timeout=5000,
    )
    ids: list[str] = page.evaluate(
        "() => Array.from(document.querySelectorAll("
        "'#tab-list .chat-tab[data-tab-id]'"
        ")).map(el => el.dataset.tabId)"
    )
    assert len(ids) == 2
    assert ids[0] != ids[1]
    return ids[0], ids[1]


def _switch_to_tab(page, tab_id: str) -> None:
    """Click the tab in the tab bar so it becomes the active tab."""
    page.evaluate(
        """(id) => {
            const el = document.querySelector(
                `#tab-list .chat-tab[data-tab-id="${id}"]`
            );
            el.click();
        }""",
        tab_id,
    )
    page.wait_for_function(
        "id => window._demoApi.getActiveTabId() === id",
        arg=tab_id,
        timeout=5000,
    )


def _mark_tab_running(page, tab_id: str) -> None:
    """Drive ``status: running=true`` for ``tab_id`` via main.js's
    real event handler so the tab transitions into the live state
    the daemon would put it in just before a ``task_done`` arrives.
    """
    page.evaluate(
        """(id) => window.__post({
            type: 'status',
            tabId: id,
            running: true,
        })""",
        tab_id,
    )


def _post_task_done(
    page, tab_id: str, *, success: bool = True
) -> None:
    """Dispatch a ``task_done`` event for ``tab_id``."""
    page.evaluate(
        """(args) => window.__post({
            type: 'task_done',
            tabId: args.tabId,
            success: args.success,
            startTs: 0,
            endTs: 0,
        })""",
        {"tabId": tab_id, "success": success},
    )


def _post_terminal_event(
    page, ev_type: str, tab_id: str
) -> None:
    """Dispatch one of ``task_error`` / ``task_stopped`` /
    ``task_interrupted`` for ``tab_id``.
    """
    page.evaluate(
        """(args) => window.__post({
            type: args.type,
            tabId: args.tabId,
            startTs: 0,
            endTs: 0,
        })""",
        {"type": ev_type, "tabId": tab_id},
    )


def _active_tab_id(page) -> str:
    result = page.evaluate("() => window._demoApi.getActiveTabId()")
    assert isinstance(result, str)
    return result


def _active_dom_tab_id(page) -> str | None:
    """Return the ``data-tab-id`` of the ``.chat-tab.active`` DOM node."""
    result = page.evaluate(
        "() => {"
        " const el = document.querySelector("
        "'#tab-list .chat-tab.active[data-tab-id]'"
        ");"
        " return el ? el.dataset.tabId : null;"
        "}"
    )
    assert result is None or isinstance(result, str)
    return result


def test_task_done_switches_to_target_tab(_browser) -> None:
    """``task_done`` for an inactive owned tab must switch focus to it.

    Steps:
      1. Open the harness, allocate two tabs.
      2. Make tab A the active tab and tab B the background tab.
      3. Send ``status: running=true`` for tab B so it's marked
         running (mirrors the live daemon event sequence).
      4. Send ``task_done`` targeting tab B.
      5. Assert the webview auto-switched: both the in-JS
         ``activeTabId`` state and the ``.chat-tab.active`` DOM class
         move to tab B.
    """
    context, page = _open_page(_browser)
    try:
        tab_a, tab_b = _make_two_tabs(page)
        _switch_to_tab(page, tab_a)
        assert _active_tab_id(page) == tab_a
        assert _active_dom_tab_id(page) == tab_a

        _mark_tab_running(page, tab_b)
        # Tab B is now running in the background; tab A is still the
        # active tab the user is viewing.
        assert _active_tab_id(page) == tab_a

        _post_task_done(page, tab_b)
        page.wait_for_function(
            "id => window._demoApi.getActiveTabId() === id",
            arg=tab_b,
            timeout=5000,
        )
        assert _active_tab_id(page) == tab_b, (
            "Webview must switch the active tab to the tab whose task "
            "just completed (tab_b), but activeTabId stayed at the "
            "user-viewed tab."
        )
        assert _active_dom_tab_id(page) == tab_b, (
            "The .chat-tab.active DOM class must also move to the "
            "tab whose task just completed."
        )
    finally:
        context.close()


@pytest.mark.parametrize(
    "ev_type",
    ["task_error", "task_stopped", "task_interrupted"],
)
def test_terminal_event_switches_to_target_tab(_browser, ev_type) -> None:
    """Every terminal task event must also switch to the target tab.

    ``task_done`` is the success path; ``task_error``,
    ``task_stopped`` and ``task_interrupted`` are the failure /
    cancellation / shutdown paths.  All three end the task in the
    target tab so the user must be switched to that tab to see the
    final status banner, exactly the same as for ``task_done``.
    """
    context, page = _open_page(_browser)
    try:
        tab_a, tab_b = _make_two_tabs(page)
        _switch_to_tab(page, tab_a)
        _mark_tab_running(page, tab_b)
        assert _active_tab_id(page) == tab_a

        _post_terminal_event(page, ev_type, tab_b)
        page.wait_for_function(
            "id => window._demoApi.getActiveTabId() === id",
            arg=tab_b,
            timeout=5000,
        )
        assert _active_tab_id(page) == tab_b, (
            f"Webview must switch the active tab to the tab whose "
            f"task just ended via {ev_type!r}, but activeTabId stayed "
            f"at the user-viewed tab."
        )
        assert _active_dom_tab_id(page) == tab_b
    finally:
        context.close()


def test_task_done_on_active_tab_keeps_focus(_browser) -> None:
    """A ``task_done`` targeting the already-active tab is a no-op
    for the active-tab pointer — the user must not be torn away from
    the tab they are already viewing.
    """
    context, page = _open_page(_browser)
    try:
        tab_a, tab_b = _make_two_tabs(page)
        _switch_to_tab(page, tab_a)
        _mark_tab_running(page, tab_a)
        assert _active_tab_id(page) == tab_a

        _post_task_done(page, tab_a)
        # Yield a turn so any re-render runs.
        page.wait_for_function(
            "() => true", timeout=500,
        )
        assert _active_tab_id(page) == tab_a, (
            "task_done for the already-active tab must not switch "
            "the active tab away."
        )
        assert _active_dom_tab_id(page) == tab_a
    finally:
        context.close()


def test_task_done_for_unknown_tab_id_is_safe(_browser) -> None:
    """A ``task_done`` whose ``tabId`` is not in this webview's tab
    list must not switch focus and must not raise — the daemon
    broadcasts tab-stamped events to every connected client, and
    this webview must silently ignore events whose tab it doesn't
    own.
    """
    context, page = _open_page(_browser)
    try:
        tab_a, tab_b = _make_two_tabs(page)
        _switch_to_tab(page, tab_a)

        _post_task_done(page, "this-tab-does-not-exist")
        # Yield a turn so any handler completes.
        page.wait_for_function(
            "() => true", timeout=500,
        )
        assert _active_tab_id(page) == tab_a, (
            "task_done for an unknown tab must not change the "
            "active tab."
        )
        # No IIFE error should have been raised.
        iife_err = page.evaluate("() => window.__iifeError")
        assert iife_err is None, (
            f"task_done for an unknown tab must not raise; got "
            f"{iife_err!r}"
        )
    finally:
        context.close()

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
whose task in tab A was moved aside by an agent-opened tab B (a
sub-agent tab, a report tab) had to manually click back to tab A to
see the result.  The user-facing contract (``focusFinishedTab`` in
``main.js``): **when a task completes in a tab, the webview switches
to that tab, unless the user themselves moved away (clicked, typed or
scrolled) since submitting** -- their own place is theirs to keep.

The tests below load the real ``media/main.js`` into a headless
Chromium (Playwright) so the real event handlers, the real DOM, and
the real per-tab state run end-to-end.

The fixture-injected synthetic page mimics the same harness used by
``test_history_running_spinner.py`` so the behaviour is exercised
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
_API_JS = _MEDIA_DIR / "api.js"
_JS = _MEDIA_DIR / "main.js"
_HTML = _MEDIA_DIR / "chat.html"

# The tab bar has two rows: ``#main-tab-list`` holds one entry per chat
# and ``#tab-list`` (the group strip) holds the on-screen chat plus its
# sub-agents / files.  The active chat is rendered on BOTH rows, so the
# number of open tabs is the number of distinct ``data-tab-id`` values.
_OPEN_TAB_COUNT_JS = (
    "new Set(Array.from(document.querySelectorAll("
    "'.chat-tab[data-tab-id]')).map(e => e.dataset.tabId)).size"
)


def _build_test_page() -> str:
    """Return a self-contained HTML page that loads the real CSS+JS.

    Mirrors :func:`_build_test_page` from
    ``test_history_running_spinner.py`` so the harness is
    drop-in compatible with the rest of the VS Code webview test
    suite.
    """
    css = _CSS.read_text(encoding="utf-8")
    api_js = _API_JS.read_text(encoding="utf-8")
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
      --vscode-charts-red: #f44747;
      --vscode-charts-green: #6a9955;
      --vscode-charts-yellow: #d7ba7d;
      --vscode-charts-purple: #c586c0;
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
  <script>{api_js}</script>
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
    page.wait_for_function(f"{_OPEN_TAB_COUNT_JS} >= 1", timeout=5000)
    return context, page


def _open_agent_tab(page) -> str:
    """Open a second tab the way the agent does (programmatically, not
    by a user click) and return its id; it becomes the active tab.

    This is the "agent-made switch" ``focusFinishedTab`` undoes: a
    sub-agent or report tab opening on top of the user's running tab.
    """
    before = _active_tab_id(page)
    page.evaluate("() => window._testApi.createNewTab()")
    # A second chat is a new top-level tab: it is rendered on the main
    # row, not in the group strip of the first chat.
    page.wait_for_function(f"{_OPEN_TAB_COUNT_JS} === 2", timeout=5000)
    after = _active_tab_id(page)
    assert after != before
    assert _active_dom_tab_id(page) == after
    return after


def _click_tab(page, tab_id: str) -> None:
    """Click the tab in the tab bar so it becomes the active tab.

    A real click: main.js records it as the user's own interaction,
    so a later task end must leave them where they are.

    The tab is clicked on the row where it is rendered: a chat that is
    not on screen exists only on the main row (``#main-tab-list``); a
    member of the on-screen group is in the strip (``#tab-list``).
    """
    page.evaluate(
        """(id) => {
            const el =
                document.querySelector(
                    `#tab-list .chat-tab[data-tab-id="${id}"]`
                ) ||
                document.querySelector(
                    `#main-tab-list .chat-tab[data-tab-id="${id}"]`
                );
            if (!el) throw new Error('tab not rendered on either row: ' + id);
            el.click();
        }""",
        tab_id,
    )
    page.wait_for_function(
        "id => window._testApi.getActiveTabId() === id",
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
    result = page.evaluate("() => window._testApi.getActiveTabId()")
    assert isinstance(result, str)
    return result


def _active_dom_tab_id(page) -> str | None:
    """Return the ``data-tab-id`` of the ``.chat-tab.active`` DOM node.

    The on-screen tab is the ``.active`` entry of the group strip
    (``#tab-list``, rendered even while hidden for a lone chat); the
    main row's ``.active`` entry marks that tab's group, so it must
    name the same chat (the tabs here are all top-level chats).
    """
    result = page.evaluate(
        "() => {"
        " const strip = document.querySelector("
        "'#tab-list .chat-tab.active[data-tab-id]'"
        ");"
        " const main = document.querySelector("
        "'#main-tab-list .chat-tab.active[data-tab-id]'"
        ");"
        " return {strip: strip ? strip.dataset.tabId : null,"
        "         main: main ? main.dataset.tabId : null};"
        "}"
    )
    assert result["main"] == result["strip"], (
        "main row highlights a different group than the strip's "
        f"active tab: {result!r}"
    )
    assert result["strip"] is None or isinstance(result["strip"], str)
    return result["strip"]


def test_task_done_switches_to_target_tab(_browser) -> None:
    """``task_done`` for an owned tab the agent switched away from must
    switch focus back to it.

    Steps:
      1. Open the harness; tab A is the active tab.
      2. Send ``status: running=true`` for tab A so it's marked
         running (mirrors the live daemon event sequence).
      3. Open tab B the way the agent does (programmatically): tab B
         is now active, tab A is in the background.
      4. Send ``task_done`` targeting tab A.
      5. Assert the webview auto-switched: both the in-JS
         ``activeTabId`` state and the ``.chat-tab.active`` DOM class
         move back to tab A.
    """
    context, page = _open_page(_browser)
    try:
        tab_a = _active_tab_id(page)
        _mark_tab_running(page, tab_a)
        tab_b = _open_agent_tab(page)
        assert _active_tab_id(page) == tab_b

        _post_task_done(page, tab_a)
        page.wait_for_function(
            "id => window._testApi.getActiveTabId() === id",
            arg=tab_a,
            timeout=5000,
        )
        assert _active_tab_id(page) == tab_a, (
            "Webview must switch the active tab to the tab whose task "
            "just completed (tab_a), but activeTabId stayed at the "
            "agent-opened tab."
        )
        assert _active_dom_tab_id(page) == tab_a, (
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
        tab_a = _active_tab_id(page)
        _mark_tab_running(page, tab_a)
        tab_b = _open_agent_tab(page)
        assert _active_tab_id(page) == tab_b

        _post_terminal_event(page, ev_type, tab_a)
        page.wait_for_function(
            "id => window._testApi.getActiveTabId() === id",
            arg=tab_a,
            timeout=5000,
        )
        assert _active_tab_id(page) == tab_a, (
            f"Webview must switch the active tab to the tab whose "
            f"task just ended via {ev_type!r}, but activeTabId stayed "
            f"at the agent-opened tab."
        )
        assert _active_dom_tab_id(page) == tab_a
    finally:
        context.close()


def test_task_done_keeps_user_chosen_tab(_browser) -> None:
    """A tab the user clicked to since submitting is theirs to keep.

    The user ran a task in tab A, then clicked over to tab B
    themselves.  Tab A finishing must not yank them back: only an
    agent-made switch (see :func:`_open_agent_tab`) is undone.
    """
    context, page = _open_page(_browser)
    try:
        tab_a = _active_tab_id(page)
        _mark_tab_running(page, tab_a)
        tab_b = _open_agent_tab(page)
        _click_tab(page, tab_a)
        _click_tab(page, tab_b)
        assert _active_tab_id(page) == tab_b

        _post_task_done(page, tab_a)
        page.wait_for_function(
            "() => true", timeout=500,
        )
        assert _active_tab_id(page) == tab_b, (
            "task_done must not pull the user off a tab they clicked "
            "to themselves."
        )
        assert _active_dom_tab_id(page) == tab_b
    finally:
        context.close()


def test_real_submit_resets_earlier_interaction(_browser) -> None:
    """A submit through the composer forgets the clicks made before it.

    The user clicks around, then types a prompt and presses Send in
    tab A; an agent tab opens on top of it while the task runs.  The
    clicks predate the submit, so when tab A finishes the webview must
    pull the user back to it — ``sendMessage`` has to clear the
    interaction flag, otherwise those earlier clicks would count as
    the user choosing the agent tab.
    """
    context, page = _open_page(_browser)
    try:
        tab_a = _active_tab_id(page)
        page.mouse.click(400, 450)  # a real interaction before the submit
        page.fill("#task-input", "run the tests")
        page.click("#send-btn")
        page.wait_for_function(
            "() => window.__postedMessages.some(m => m.type === 'submit')",
            timeout=5000,
        )
        _mark_tab_running(page, tab_a)
        tab_b = _open_agent_tab(page)
        assert _active_tab_id(page) == tab_b

        _post_task_done(page, tab_a)
        page.wait_for_function(
            "id => window._testApi.getActiveTabId() === id",
            arg=tab_a,
            timeout=5000,
        )
        assert _active_dom_tab_id(page) == tab_a
    finally:
        context.close()


def test_task_done_on_active_tab_keeps_focus(_browser) -> None:
    """A ``task_done`` targeting the already-active tab is a no-op
    for the active-tab pointer — the user must not be torn away from
    the tab they are already viewing.
    """
    context, page = _open_page(_browser)
    try:
        tab_b = _open_agent_tab(page)
        _mark_tab_running(page, tab_b)
        assert _active_tab_id(page) == tab_b

        _post_task_done(page, tab_b)
        page.wait_for_function(
            "() => true", timeout=500,
        )
        assert _active_tab_id(page) == tab_b, (
            "task_done for the already-active tab must not switch "
            "the active tab away."
        )
        assert _active_dom_tab_id(page) == tab_b
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
        tab_b = _open_agent_tab(page)

        _post_task_done(page, "this-tab-does-not-exist")
        page.wait_for_function(
            "() => true", timeout=500,
        )
        assert _active_tab_id(page) == tab_b, (
            "task_done for an unknown tab must not change the "
            "active tab."
        )
        iife_err = page.evaluate("() => window.__iifeError")
        assert iife_err is None, (
            f"task_done for an unknown tab must not raise; got "
            f"{iife_err!r}"
        )
    finally:
        context.close()

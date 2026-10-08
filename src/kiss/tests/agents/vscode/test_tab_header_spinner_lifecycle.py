# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end check of the chat tab HEADER spinner on every surface.

Rule under test: while a task in a chat runs, the tab of that chat
shows a spinner in its header, and the spinner disappears the moment
the task finishes -- whether it finished, failed, errored or was
stopped, and whether the tab is the one on screen or a background one.

Three surfaces draw that header from the same ``media/main.js``:

* the VS Code sidebar webview's group strip ``#tab-list``: the chat on
  screen plus its sub-agent tabs (``.chat-tab-spinner.status-spinner``
  built by ``buildTabElement`` for ``renderTabBar``).  There is no row
  of chat tabs any more: a background chat is not drawn at all (its
  state lives in the ``tabs`` array, read here through
  ``window._testApi.openTabs()``), and the strip is hidden while the
  chat on screen has no sub-agent, so the tests attach a finished
  sub-agent (as the daemon replays one) to make the header visible;
* the remote webapp, which is the same page booted with
  ``<body class="remote-chat">`` and ``remote-codex.css`` restyling the
  tabs as pills -- the spinner must survive that cascade;
* the VS Code editor-tab mode, where the chat's header is the EDITOR
  tab itself: the webview posts ``panelTitle {state}`` to the host,
  which paints ``media/spinner-running.svg`` as the tab icon while
  ``state === 'running'`` (host side covered by
  ``test/editorTabsPanelStatus.test.js``).

The daemon ends a task with a terminal event (``task_done`` /
``task_error`` / ``task_stopped``, see ``task_runner.py``) FOLLOWED by
``status running:false``; some paths send only the ``status`` (a
viewer attaching to a chat that just finished, ``commands.py``). Each
of those messages must drop the spinner on its own, so the tests
assert after the terminal event alone, after the trailing status, and
after a status-only end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_history_failed_red_cross import (
    _build_test_page,
    _open_history_page,
)

_REMOTE_CSS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "remote-codex.css"
)

_START_TS = 1700000000000
_END_TS = 1700000001000
_ENDS = ["done", "failed", "error", "stopped"]


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


_HEADER_PROBE = """
(tabId) => {
  const tab = document.querySelector(
    '#tab-list .chat-tab[data-tab-id=' + JSON.stringify(tabId) + ']');
  if (!tab) return {error: 'no tab ' + tabId};
  const spinner = tab.querySelector('.chat-tab-spinner');
  const bar = document.getElementById('tab-bar');
  const out = {
    bodyClass: document.body.className,
    tabVisible: tab.offsetWidth > 0 && tab.offsetHeight > 0,
    barDisplay: getComputedStyle(bar).display,
    spinner: null,
    tick: tab.querySelector('.chat-tab-ok') !== null,
    cross: tab.querySelector('.chat-tab-fail') !== null,
  };
  if (spinner) {
    const cs = getComputedStyle(spinner);
    out.spinner = {
      classes: spinner.className,
      visible: spinner.offsetWidth > 0 && spinner.offsetHeight > 0,
      animationName: cs.animationName,
      animationIterationCount: cs.animationIterationCount,
      animationPlayState: cs.animationPlayState,
    };
  }
  return out;
}
"""


def _post(page, event: dict) -> None:
    page.evaluate("(ev) => window.__post(ev)", event)


def _header(page, tab_id: str) -> dict:
    """The strip entry of *tab_id*: it must be in the group on screen."""
    probe = dict(page.evaluate(_HEADER_PROBE, tab_id))
    assert "error" not in probe, probe
    return probe


def _running(page, tab_id: str) -> bool:
    """The webview's running flag for *tab_id* (``openTabs()``), the
    state a background chat -- drawn nowhere -- carries."""
    records = [t for t in page.evaluate("() => window._testApi.openTabs()")
               if t["id"] == tab_id]
    assert records, f"no tab {tab_id}"
    return bool(records[0]["isRunning"])


def _active_tab_id(page) -> str:
    return str(page.evaluate("() => window._testApi.getActiveTabId()"))


def _show_strip(page, tab_id: str) -> None:
    """Attach a finished sub-agent to *tab_id* so its group strip (the
    chat's header) is displayed; the chat stays the active tab."""
    _post(page, {"type": "openSubagentTab", "tab_id": tab_id + "__sub_done",
                 "parent_tab_id": tab_id, "description": "finished child",
                 "task_id": "task-finished-child", "isSubagentTab": True,
                 "isDone": True})
    assert _active_tab_id(page) == tab_id
    header = _header(page, tab_id)
    assert header["barDisplay"] != "none", header
    assert header["tabVisible"], header


def _start_task(page, tab_id: str, task_id: str) -> None:
    """Replay the daemon's task-start broadcast into *tab_id*."""
    _post(page, {"type": "setTaskText", "text": "do the thing", "tabId": tab_id})
    _post(page, {"type": "clear", "chat_id": "chat-" + tab_id, "tabId": tab_id})
    _post(page, {"type": "status", "running": True, "tabId": tab_id,
                 "startTs": _START_TS, "taskId": task_id})
    _post(page, {"type": "text_delta", "text": "working", "tabId": tab_id,
                 "taskId": task_id})
    _post(page, {"type": "text_end", "tabId": tab_id, "taskId": task_id})


def _result(page, tab_id: str, task_id: str, ok: bool) -> None:
    _post(page, {"type": "result", "text": "done" if ok else "it broke",
                 "summary": "<p>done</p>" if ok else "<p>it broke</p>",
                 "success": ok, "is_continue": False, "total_tokens": 10,
                 "cost": "$0.0001", "step_count": 1, "tabId": tab_id,
                 "taskId": task_id})


def _terminal_event(page, tab_id: str, task_id: str, how: str) -> None:
    """Replay the daemon's terminal broadcast for *how* WITHOUT the
    trailing ``status running:false``.

    ``"done"``: successful ``result`` + ``task_done``; ``"failed"``:
    ``success:false`` result + ``task_done``; ``"error"``: the failure
    ``result`` the daemon broadcasts first, then ``task_error`` with
    its ``text``; ``"stopped"``: ``task_stopped``.
    """
    if how in ("done", "failed"):
        ok = how == "done"
        _result(page, tab_id, task_id, ok)
        _post(page, {"type": "task_done", "tabId": tab_id, "success": ok,
                     "startTs": _START_TS, "endTs": _END_TS})
    elif how == "error":
        _result(page, tab_id, task_id, False)
        _post(page, {"type": "task_error", "text": "boom", "tabId": tab_id,
                     "startTs": _START_TS, "endTs": _END_TS})
    else:
        _post(page, {"type": "task_stopped", "tabId": tab_id,
                     "startTs": _START_TS, "endTs": _END_TS})


def _status_off(page, tab_id: str, task_id: str) -> None:
    _post(page, {"type": "status", "running": False, "tabId": tab_id,
                 "taskId": task_id})


def _assert_spinning(header: dict) -> None:
    sp = header["spinner"]
    assert sp is not None, f"running task must show the header spinner: {header}"
    assert "status-spinner" in sp["classes"].split(), sp
    assert sp["visible"], sp
    assert sp["animationName"] == "status-spin", sp
    assert sp["animationIterationCount"] == "infinite", sp
    assert sp["animationPlayState"] == "running", sp
    assert not header["tick"] and not header["cross"], header


def _assert_idle(header: dict) -> None:
    assert header["spinner"] is None, f"finished task must drop the spinner: {header}"


def _assert_verdict(header: dict, how: str) -> None:
    """The icon that replaces the spinner matches the outcome."""
    _assert_idle(header)
    assert header["tick"] == (how == "done"), header
    assert header["cross"] == (how != "done"), header


def _check_end(page, tab_id: str, task_id: str, how: str) -> None:
    """Spinner gone right after the terminal event, still gone after
    the trailing status, with the right verdict icon both times."""
    _terminal_event(page, tab_id, task_id, how)
    _assert_verdict(_header(page, tab_id), how)
    _status_off(page, tab_id, task_id)
    _assert_verdict(_header(page, tab_id), how)


def _check_background_end(page, worker_id: str, task_id: str, how: str) -> None:
    """A background chat's task ends: its running flag drops on the
    terminal event alone; brought on screen, its header shows the
    verdict icon and no spinner, before and after the trailing status."""
    _terminal_event(page, worker_id, task_id, how)
    assert not _running(page, worker_id), how
    _switch_to(page, worker_id)
    _assert_verdict(_header(page, worker_id), how)
    _status_off(page, worker_id, task_id)
    assert not _running(page, worker_id), how
    _assert_verdict(_header(page, worker_id), how)


def _switch_to(page, tab_id: str) -> None:
    """Show *tab_id*, as the Chats panel's pick does."""
    page.evaluate("(id) => window._testApi.switchToTab(id)", tab_id)
    assert _active_tab_id(page) == tab_id


def _open_page(_browser, body_class: str, extra_css: str = ""):
    """Boot the harness with *body_class* set BEFORE ``main.js`` runs.

    ``main.js`` reads the surface flags (``editor-tab-mode``,
    ``remote-chat``) from ``<body>`` at IIFE time, exactly as the
    extension host and ``web_server.py`` supply them, so the class
    must be in the markup, not added afterwards.
    """
    html = _build_test_page()
    html = html.replace("<body>", f'<body class="{body_class}">', 1)
    if extra_css:
        html = html.replace("</head>", f"<style>{extra_css}</style></head>", 1)
    context = _browser.new_context(viewport={"width": 480, "height": 900})
    page = context.new_page()
    page.set_content(html, wait_until="load")
    page.wait_for_function(
        "document.getElementById('history-list') !== null", timeout=5000)
    page.evaluate(
        "() => {"
        " document.getElementById('app').style.display = '';"
        " const ov = document.getElementById('kiss-server-loading');"
        " if (ov) ov.style.display = 'none';"
        " }"
    )
    err = page.evaluate("() => window.__iifeError")
    assert not err, f"main.js IIFE setup raised: {err}"
    return context, page


def _open_remote_page(_browser):
    """Open the harness as ``web_server.py`` serves the remote webapp:
    ``<body class="remote-chat">`` plus ``remote-codex.css`` (at 480px
    the stacked mobile layout, whose strip is ``#tab-list``)."""
    context, page = _open_page(
        _browser, "remote-chat", _REMOTE_CSS.read_text(encoding="utf-8"))
    body_class = page.evaluate("() => document.body.className").split()
    assert "remote-chat" in body_class, body_class
    assert "sidebar-chat-mode" not in body_class, body_class
    assert "remote-desktop" not in body_class, body_class
    return context, page


# --- VS Code sidebar webview -------------------------------------------


@pytest.mark.parametrize("how", _ENDS)
def test_sidebar_active_tab_spinner_until_task_ends(_browser, how: str) -> None:
    """VS Code sidebar strip: the active chat's header spins from the
    daemon's ``status running`` until the task ends in any way."""
    context, page = _open_history_page(_browser)
    try:
        tab_id = _active_tab_id(page)
        _show_strip(page, tab_id)
        _assert_idle(_header(page, tab_id))
        _start_task(page, tab_id, "task-" + how)
        _assert_spinning(_header(page, tab_id))
        _check_end(page, tab_id, "task-" + how, how)
    finally:
        context.close()


@pytest.mark.parametrize("how", _ENDS)
def test_sidebar_background_tab_spinner_until_task_ends(_browser, how: str) -> None:
    """A task running in a chat the user is NOT looking at keeps that
    chat's running flag (and only that one), spins its header whenever
    the chat is brought on screen, survives switching chats, and stops
    when it ends."""
    context, page = _open_history_page(_browser)
    try:
        worker_id = _active_tab_id(page)
        _show_strip(page, worker_id)
        page.evaluate("() => window._testApi.createNewTab()")
        viewer_id = _active_tab_id(page)
        assert viewer_id != worker_id

        _start_task(page, worker_id, "bg-" + how)
        assert _running(page, worker_id)
        assert not _running(page, viewer_id)
        _assert_idle(_header(page, viewer_id))

        _switch_to(page, worker_id)
        _assert_spinning(_header(page, worker_id))
        _switch_to(page, viewer_id)
        assert _running(page, worker_id)
        _assert_idle(_header(page, viewer_id))

        _check_background_end(page, worker_id, "bg-" + how, how)
        _switch_to(page, viewer_id)
        assert not _running(page, viewer_id)
        _assert_idle(_header(page, viewer_id))
    finally:
        context.close()


def test_sidebar_status_only_end_drops_spinner(_browser) -> None:
    """A bare ``status running:false`` (no terminal event, as when a
    viewer attaches to a chat whose task just ended) drops the spinner
    on the active tab and the running flag of a background tab alike."""
    context, page = _open_history_page(_browser)
    try:
        first_id = _active_tab_id(page)
        _show_strip(page, first_id)
        _start_task(page, first_id, "status-only-active")
        _assert_spinning(_header(page, first_id))
        _status_off(page, first_id, "status-only-active")
        _assert_idle(_header(page, first_id))

        page.evaluate("() => window._testApi.createNewTab()")
        second_id = _active_tab_id(page)
        _start_task(page, first_id, "status-only-bg")
        assert _running(page, first_id)
        assert not _running(page, second_id)
        _assert_idle(_header(page, second_id))
        _status_off(page, first_id, "status-only-bg")
        assert not _running(page, first_id)
        _switch_to(page, first_id)
        _assert_idle(_header(page, first_id))
    finally:
        context.close()


# --- Remote webapp -----------------------------------------------------


@pytest.mark.parametrize("how", _ENDS)
def test_remote_webapp_tab_spinner_until_task_ends(_browser, how: str) -> None:
    """Remote webapp: the pill-styled tab header spins while the task
    runs and stops when it ends; the tab bar itself stays displayed."""
    context, page = _open_remote_page(_browser)
    try:
        tab_id = _active_tab_id(page)
        _show_strip(page, tab_id)
        _assert_idle(_header(page, tab_id))
        _start_task(page, tab_id, "remote-" + how)
        running = _header(page, tab_id)
        assert running["barDisplay"] != "none", running
        assert running["tabVisible"], running
        _assert_spinning(running)
        _check_end(page, tab_id, "remote-" + how, how)
    finally:
        context.close()


@pytest.mark.parametrize("how", _ENDS)
def test_remote_webapp_background_tab_spinner(_browser, how: str) -> None:
    """Remote webapp: a background chat keeps the running flag for its
    own task only, its pill spins once it is shown, and the flag and
    spinner drop when that task ends."""
    context, page = _open_remote_page(_browser)
    try:
        worker_id = _active_tab_id(page)
        _show_strip(page, worker_id)
        page.evaluate("() => window._testApi.createNewTab()")
        viewer_id = _active_tab_id(page)
        assert viewer_id != worker_id
        _start_task(page, worker_id, "remote-bg-" + how)
        assert _running(page, worker_id)
        assert not _running(page, viewer_id)
        _assert_idle(_header(page, viewer_id))
        _switch_to(page, worker_id)
        _assert_spinning(_header(page, worker_id))
        _switch_to(page, viewer_id)
        _check_background_end(page, worker_id, "remote-bg-" + how, how)
        _switch_to(page, viewer_id)
        _assert_idle(_header(page, viewer_id))
    finally:
        context.close()


def test_remote_webapp_status_only_end_drops_spinner(_browser) -> None:
    """Remote webapp: a bare ``status running:false`` drops the spinner."""
    context, page = _open_remote_page(_browser)
    try:
        tab_id = _active_tab_id(page)
        _show_strip(page, tab_id)
        _start_task(page, tab_id, "remote-status-only")
        _assert_spinning(_header(page, tab_id))
        _status_off(page, tab_id, "remote-status-only")
        _assert_idle(_header(page, tab_id))
    finally:
        context.close()


# --- VS Code editor-tab mode -------------------------------------------


def _panel_states(page) -> list[str]:
    return list(page.evaluate(
        "() => window.__postedMessages"
        ".filter(m => m && m.type === 'panelTitle').map(m => m.state)"
    ))


@pytest.mark.parametrize("how", _ENDS)
def test_editor_tab_mode_reports_running_state_to_host(_browser, how: str) -> None:
    """VS Code editor-tabs mode: the chat's header is the editor tab, so
    the webview must tell the host ``state:'running'`` when the task
    starts and a non-running state the moment it ends (the host swaps
    the tab icon to ``spinner-running.svg`` for exactly that span)."""
    context, page = _open_page(_browser, "editor-tab-mode")
    try:
        tab_id = _active_tab_id(page)
        _start_task(page, tab_id, "editor-" + how)
        states = _panel_states(page)
        assert states and states[-1] == "running", states
        n_before_end = len(states)

        verdict = "ok" if how == "done" else "fail"
        # The terminal event alone must already report the end...
        _terminal_event(page, tab_id, "editor-" + how, how)
        states = _panel_states(page)
        assert len(states) > n_before_end, "terminal event must repaint the tab"
        assert states[-1] == verdict, states
        # ...and the trailing status must not bring the spinner back.
        _status_off(page, tab_id, "editor-" + how)
        states = _panel_states(page)
        assert states[-1] == verdict, states
        assert "running" not in states[n_before_end:], states
        assert states.count("running") == 1, states
    finally:
        context.close()


def test_editor_tab_mode_status_only_end(_browser) -> None:
    """Editor-tabs mode: a bare ``status running:false`` ends the
    running state reported to the host."""
    context, page = _open_page(_browser, "editor-tab-mode")
    try:
        tab_id = _active_tab_id(page)
        _start_task(page, tab_id, "editor-status-only")
        assert _panel_states(page)[-1] == "running"
        _status_off(page, tab_id, "editor-status-only")
        states = _panel_states(page)
        assert states[-1] != "running", states
    finally:
        context.close()

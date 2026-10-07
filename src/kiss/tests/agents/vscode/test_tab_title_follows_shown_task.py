# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end check that a chat tab's title names the task the reader
is looking at, on every surface.

Every task of the thread opens with its own task panel in the
transcript, so the tab's title is what follows the reader: the tab's
own task, a neighbouring task scrolled into from history, or a history
row's task shown read-only in a fresh tab. The title must say the same
thing, wherever the tab header is drawn:

* the VS Code sidebar webview's internal tab strip (``.chat-tab-label``
  in ``renderTabBar``);
* the remote webapp, the same page booted with ``<body
  class="remote-chat">`` and ``remote-codex.css`` restyling the tabs;
* the VS Code editor-tab mode, where the chat's header is the EDITOR
  tab itself: the webview posts ``panelTitle {title}`` to the host
  (``SorcarPanelManager._onPanelEvent`` paints it on the editor tab).

Only the ACTIVE top-level chat tab follows the reader: a background
chat keeps its own task's title, and a sub-agent tab keeps its
numbered description.
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

_OWN_TASK = "Refactor the payment service to use the new ledger API"
_PREV_TASK = "Write the migration guide for the ledger"
_PREVIEW_TASK = "Investigate the flaky nightly build on arm64"


def _clip(text: str) -> str:
    """``clipTabTitle``: 30 characters plus an ellipsis."""
    return text if len(text) <= 30 else text[:30] + "\u2026"


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _post(page, event: dict) -> None:
    page.evaluate("(ev) => window.__post(ev)", event)


def _active_tab_id(page) -> str:
    return str(page.evaluate("() => window._testApi.getActiveTabId()"))


def _own_panel_text(page) -> str:
    """The task text the transcript on screen opens with."""
    return str(page.evaluate(
        "() => document.querySelector('#output .task-panel-text').textContent"))


def _active_label(page) -> str:
    return str(page.evaluate(
        "() => document.querySelector("
        "'.chat-tab[aria-selected=\"true\"] .chat-tab-label').textContent"))


def _label(page, tab_id: str) -> str:
    """The text the tab strip shows for *tab_id* (also its aria-label)."""
    out = page.evaluate(
        """(tabId) => {
          const tab = document.querySelector(
            '.chat-tab[data-tab-id=' + JSON.stringify(tabId) + ']');
          if (!tab) return {error: 'no tab ' + tabId};
          return {label: tab.querySelector('.chat-tab-label').textContent,
                  aria: tab.getAttribute('aria-label')};
        }""",
        tab_id,
    )
    assert "error" not in out, out
    assert out["aria"] == out["label"], out
    return str(out["label"])


def _last_panel_title(page) -> str:
    """The title of the last ``panelTitle`` the webview posted to the
    host (editor-tab mode)."""
    return str(page.evaluate(
        "() => { const posts = window.__postedMessages.filter("
        "m => m.type === 'panelTitle'); return posts.length ?"
        " posts[posts.length - 1].title : null; }"))


def _long_events(task: str, n: int = 40) -> list[dict]:
    """A transcript tall enough to scroll in a 900px viewport once its
    Trajectory panel is opened (a finished task replays folded)."""
    events: list[dict] = [{"type": "task_start", "task": task}]
    for i in range(n):
        events.append({"type": "system_output",
                       "text": f"{task}: line {i} of output\n"})
    return events


def _open_trajectories(page) -> None:
    """Open every folded Trajectory panel, as a reader of a finished
    task does: its event panels then give the transcript its height."""
    page.evaluate(
        """() => {
          for (const t of document.querySelectorAll('#output .trajectory.collapsed'))
            t.querySelector(':scope > .trajectory-h').click();
        }"""
    )
    assert page.evaluate(
        "() => document.querySelectorAll('#output .trajectory.collapsed').length") == 0


def _replay_own_task(page, tab_id: str) -> None:
    _post(page, {
        "type": "task_events", "tabId": tab_id, "chat_id": "chat-" + tab_id,
        "task_id": "42", "task": _OWN_TASK, "events": _long_events(_OWN_TASK),
    })
    assert _own_panel_text(page) == _OWN_TASK
    _open_trajectories(page)


def _splice_prev_task(page, tab_id: str) -> None:
    """Splice the previous task above the own task, as the daemon's
    answer to an overscroll does."""
    _post(page, {
        "type": "adjacent_task_events", "tabId": tab_id, "direction": "prev",
        "task": _PREV_TASK, "task_id": "41", "events": _long_events(_PREV_TASK),
    })
    assert page.evaluate(
        "() => document.querySelector('.adjacent-task[data-task-id=\"41\"]')"
        " !== null")
    _open_trajectories(page)


def _scroll_output(page, where: str) -> None:
    """Scroll the transcript to its ``top`` or ``bottom`` and let the
    scroll handler re-derive the shown task."""
    page.evaluate(
        """(where) => {
          const O = document.getElementById('output');
          O.scrollTop = where === 'top' ? 0 : O.scrollHeight;
        }""",
        where,
    )
    page.wait_for_function(
        "(t) => document.querySelector("
        "'.chat-tab[aria-selected=\"true\"] .chat-tab-label').textContent === t",
        arg=_clip(_PREV_TASK if where == "top" else _OWN_TASK),
        timeout=5000,
    )


def _open_page(_browser, body_class: str, extra_css: str = ""):
    """Boot the harness with *body_class* set BEFORE ``main.js`` runs:
    the surface flags are read from ``<body>`` at IIFE time."""
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


def _check_strip_follows_reader(page) -> None:
    """Shared by the sidebar and the remote webapp: the strip's label
    for the active tab tracks the reader through a neighbour scroll."""
    tab_id = _active_tab_id(page)
    _replay_own_task(page, tab_id)
    assert _label(page, tab_id) == _clip(_OWN_TASK)

    _splice_prev_task(page, tab_id)
    # The neighbour opens with its own task panel, above the own task's.
    assert page.evaluate(
        "() => Array.from(document.querySelectorAll('#output .task-panel-text'))"
        ".map(el => el.textContent)") == [_PREV_TASK, _OWN_TASK]
    _scroll_output(page, "top")
    assert _label(page, tab_id) == _clip(_PREV_TASK)

    _scroll_output(page, "bottom")
    assert _label(page, tab_id) == _clip(_OWN_TASK)


# --- VS Code sidebar webview -------------------------------------------


def test_sidebar_label_follows_neighbour_scroll(_browser) -> None:
    """Sidebar strip: scrolling into the previous task retitles the
    active tab to that task; scrolling back restores its own."""
    context, page = _open_history_page(_browser)
    try:
        _check_strip_follows_reader(page)
    finally:
        context.close()


def test_sidebar_background_tab_keeps_own_title(_browser) -> None:
    """Only the tab on screen follows the reader: a background chat's
    label stays its own task while the active tab reads a neighbour,
    and the tab's OWN title is what comes back after a tab switch."""
    context, page = _open_history_page(_browser)
    try:
        first = _active_tab_id(page)
        _replay_own_task(page, first)
        _splice_prev_task(page, first)
        _scroll_output(page, "top")
        assert _label(page, first) == _clip(_PREV_TASK)

        page.evaluate("() => window._testApi.createNewTab()")
        second = _active_tab_id(page)
        assert second != first
        _post(page, {"type": "setTaskText", "text": _PREVIEW_TASK,
                     "tabId": second})
        assert _label(page, second) == _clip(_PREVIEW_TASK)
        # The hidden tab is labelled by its own task, not by the
        # neighbour it was parked on.
        assert _label(page, first) == _clip(_OWN_TASK)

        page.evaluate(
            "(id) => document.querySelector("
            "'.chat-tab[data-tab-id=' + JSON.stringify(id) + ']').click()",
            first,
        )
        assert _active_tab_id(page) == first
        # Back on screen, the label names the task the restored
        # transcript shows (one of the tab's own thread), never the
        # other tab's.
        assert _label(page, first) in (_clip(_OWN_TASK), _clip(_PREV_TASK))
        assert _label(page, first) == _active_label(page)
        assert _label(page, second) == _clip(_PREVIEW_TASK)
    finally:
        context.close()


def test_sidebar_history_preview_tab_is_titled(_browser) -> None:
    """A history row without a chat opens a fresh tab showing the task
    read-only (``openChatFromHistory``): the tab is titled by that
    task, not "new chat"."""
    context, page = _open_history_page(_browser)
    try:
        before = _active_tab_id(page)
        _post(page, {"type": "openChatFromHistory", "title": _PREVIEW_TASK,
                     "chatId": "", "taskId": ""})
        tab_id = _active_tab_id(page)
        assert tab_id != before
        assert _own_panel_text(page) == _PREVIEW_TASK
        assert _label(page, tab_id) == _clip(_PREVIEW_TASK)
    finally:
        context.close()


def test_sidebar_subagent_tab_keeps_numbered_title(_browser) -> None:
    """A sub-agent tab's transcript opens with its description and its
    title is the numbered description: the index is kept."""
    context, page = _open_history_page(_browser)
    try:
        parent = _active_tab_id(page)
        _replay_own_task(page, parent)
        sub_id = parent + "__sub_7"
        _post(page, {"type": "openSubagentTab", "tab_id": sub_id,
                     "parent_tab_id": parent, "task_id": "7",
                     "description": _PREVIEW_TASK, "isDone": False})
        page.evaluate(
            "(id) => document.querySelector("
            "'.chat-tab[data-tab-id=' + JSON.stringify(id) + ']').click()",
            sub_id,
        )
        assert _active_tab_id(page) == sub_id
        assert _own_panel_text(page) == _PREVIEW_TASK
        label = _label(page, sub_id)
        assert label.endswith(_PREVIEW_TASK[:40]), label
        assert label != _clip(_PREVIEW_TASK), label
    finally:
        context.close()


# --- Remote webapp -----------------------------------------------------


def test_remote_label_follows_neighbour_scroll(_browser) -> None:
    """Remote webapp pills: same rule under ``remote-codex.css``."""
    context, page = _open_page(
        _browser, "remote-chat", _REMOTE_CSS.read_text(encoding="utf-8"))
    try:
        assert "remote-chat" in page.evaluate(
            "() => document.body.className").split()
        _check_strip_follows_reader(page)
    finally:
        context.close()


# --- VS Code editor-tab mode -------------------------------------------


def test_editor_tab_title_follows_neighbour_scroll(_browser) -> None:
    """Editor-tabs mode: the ``panelTitle`` posted to the host (the
    editor tab's title) follows the reader through a neighbour scroll."""
    context, page = _open_page(_browser, "editor-tab-mode")
    try:
        tab_id = _active_tab_id(page)
        _replay_own_task(page, tab_id)
        assert _last_panel_title(page) == _clip(_OWN_TASK)

        _splice_prev_task(page, tab_id)
        _scroll_output(page, "top")
        assert _last_panel_title(page) == _clip(_PREV_TASK)

        _scroll_output(page, "bottom")
        assert _last_panel_title(page) == _clip(_OWN_TASK)
    finally:
        context.close()


def test_editor_tab_title_keeps_root_while_subagent_on_screen(_browser) -> None:
    """The editor tab is the ROOT chat's tab: while a sub-agent tab is
    on screen (its description opening the transcript) the editor tab keeps the
    root task's title."""
    context, page = _open_page(_browser, "editor-tab-mode")
    try:
        root = _active_tab_id(page)
        _replay_own_task(page, root)
        sub_id = root + "__sub_7"
        _post(page, {"type": "openSubagentTab", "tab_id": sub_id,
                     "parent_tab_id": root, "task_id": "7",
                     "description": _PREVIEW_TASK, "isDone": False})
        page.evaluate(
            "(id) => document.querySelector("
            "'.chat-tab[data-tab-id=' + JSON.stringify(id) + ']').click()",
            sub_id,
        )
        assert _active_tab_id(page) == sub_id
        assert _own_panel_text(page) == _PREVIEW_TASK
        assert _last_panel_title(page) == _clip(_OWN_TASK)
    finally:
        context.close()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the chat tab strip marks task state with a spinner,
a green tick or a red cross (never a coloured circle).

The real ``media/main.js`` + ``media/main.css`` run in headless
Chromium and receive the exact event sequence the daemon broadcasts for
a task (``status`` / ``result`` / ``task_done``) and for a sub-agent
(``openSubagentTab``).  For each state the test
asserts the tab strip's icon element and its *computed* style, so a
regression in the class wiring (``main.js``) or in the shared
``.status-spinner`` / ``.status-tick`` / ``.status-cross`` rules
(``main.css``) both fail here:

* running root tab   -> ``.chat-tab-spinner.status-spinner``: a 12x12
  ring rotated by ``status-spin`` whose leading edge is ``--green`` (the
  same ring as the composer's ``#wait-spinner``);
* succeeded root tab -> ``.chat-tab-ok.status-tick``: SVG-masked box
  filled ``--green``;
* failed root tab    -> ``.chat-tab-fail.status-cross``: SVG-masked box
  filled ``--red``;
* running sub-agent  -> ``.subagent-indicator.status-spinner`` (purple,
  the sub-agent tab's colour);
* finished sub-agent -> ``.subagent-indicator.done.status-tick`` (purple);
  a live ``subagentDone`` closes the tab, so the done state is reached
  through ``openSubagentTab {isDone: true}`` as the daemon sends it on
  replay.

The harness page is shared with ``test_history_failed_red_cross.py``.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_history_failed_red_cross import (
    _open_history_page,
)


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


# The harness :root defines --vscode-terminal-ansiGreen: #6a9955,
# --vscode-terminal-ansiRed: #f44747 and --vscode-terminal-ansiMagenta:
# #c586c0, which main.css aliases as --green / --red / --purple.
_GREEN = "rgb(106, 153, 85)"
_RED = "rgb(244, 71, 71)"
_PURPLE = "rgb(197, 134, 192)"

_ICON_PROBE = """
(tabId) => {
  const tab = document.querySelector(
    '.chat-tab[data-tab-id=' + JSON.stringify(tabId) + ']');
  if (!tab) return {error: 'no tab ' + tabId};
  const icon = tab.querySelector(
    '.chat-tab-spinner, .chat-tab-status, .subagent-indicator');
  if (!icon) return {icon: null};
  const cs = getComputedStyle(icon);
  return {
    icon: icon.className,
    text: icon.textContent,
    width: cs.width,
    height: cs.height,
    borderRadius: cs.borderRadius,
    borderTopWidth: cs.borderTopWidth,
    borderTopColor: cs.borderTopColor,
    backgroundColor: cs.backgroundColor,
    maskImage: cs.maskImage || cs.webkitMaskImage,
    animationName: cs.animationName,
    animationIterationCount: cs.animationIterationCount,
    visible: icon.offsetWidth > 0 && icon.offsetHeight > 0,
  };
}
"""


def _post(page, event: dict) -> None:
    page.evaluate("(ev) => window.__post(ev)", event)


def _icon(page, tab_id: str) -> dict:
    return dict(page.evaluate(_ICON_PROBE, tab_id))


def _root_tab_id(page) -> str:
    return str(page.evaluate("() => window._testApi.getActiveTabId()"))


def _run_task(page, tab_id: str, task_id: str, success: bool) -> None:
    """Replay one task's daemon broadcast into *tab_id* up to the
    result, leaving the task still marked running."""
    _post(page, {"type": "setTaskText", "text": "do the thing", "tabId": tab_id})
    _post(page, {"type": "clear", "chat_id": "chat-" + tab_id, "tabId": tab_id})
    _post(page, {"type": "status", "running": True, "tabId": tab_id,
                 "startTs": 1700000000000, "taskId": task_id})
    _post(page, {"type": "text_delta", "text": "working", "tabId": tab_id,
                 "taskId": task_id})
    _post(page, {"type": "text_end", "tabId": tab_id, "taskId": task_id})
    _post(page, {"type": "result", "text": "done" if success else "it broke",
                 "summary": "<p>done</p>" if success else "<p>it broke</p>",
                 "success": success, "is_continue": False, "total_tokens": 10,
                 "cost": "$0.0001", "step_count": 1, "tabId": tab_id,
                 "taskId": task_id})


def _finish_task(page, tab_id: str, task_id: str) -> None:
    _post(page, {"type": "task_done", "tabId": tab_id,
                 "startTs": 1700000000000, "endTs": 1700000001000})
    _post(page, {"type": "status", "running": False, "tabId": tab_id,
                 "taskId": task_id})


def _assert_spinner(icon: dict, colour: str, marker: str) -> None:
    assert marker in icon["icon"].split() and "status-spinner" in icon["icon"].split(), icon
    assert icon["text"] == "", f"the spinner is drawn by CSS, not a glyph: {icon}"
    assert icon["visible"], icon
    assert icon["width"] == "12px" and icon["height"] == "12px", icon
    assert icon["borderRadius"] in ("5px", "50%"), f"not a ring: {icon}"
    assert icon["borderTopWidth"] == "2px", icon
    assert icon["borderTopColor"] == colour, icon
    assert icon["animationName"] == "status-spin", icon
    assert icon["animationIterationCount"] == "infinite", icon
    assert "svg" not in (icon["maskImage"] or ""), f"a spinner has no mask: {icon}"


def _assert_mask_icon(icon: dict, shape: str, colour: str, marker: str) -> None:
    classes = icon["icon"].split()
    assert marker in classes and shape in classes, icon
    assert "status-spinner" not in classes, icon
    assert icon["text"] == "", f"the {shape} is drawn by CSS, not a glyph: {icon}"
    assert icon["visible"], icon
    assert icon["width"] == "12px" and icon["height"] == "12px", icon
    assert "svg" in (icon["maskImage"] or ""), f"{shape} must be SVG-masked: {icon}"
    assert icon["backgroundColor"] == colour, icon
    assert icon["animationName"] == "none", f"a {shape} must not animate: {icon}"


def test_root_tab_spinner_then_green_tick(_browser) -> None:
    """A running task shows the spinner; once it succeeds the tab shows
    the green tick, and the tick stays put across a re-render."""
    context, page = _open_history_page(_browser)
    try:
        tab_id = _root_tab_id(page)
        assert _icon(page, tab_id)["icon"] is None, "no icon before any task"

        _run_task(page, tab_id, "task-ok", success=True)
        _assert_spinner(_icon(page, tab_id), _GREEN, "chat-tab-spinner")

        _finish_task(page, tab_id, "task-ok")
        tick = _icon(page, tab_id)
        _assert_mask_icon(tick, "status-tick", _GREEN, "chat-tab-ok")
        assert "chat-tab-status" in tick["icon"].split(), tick
        assert "chat-tab-fail" not in tick["icon"].split(), tick

        # A later, unrelated re-render must not lose the verdict.
        _post(page, {"type": "status", "running": False, "tabId": tab_id})
        _assert_mask_icon(_icon(page, tab_id), "status-tick", _GREEN, "chat-tab-ok")
    finally:
        context.close()


def test_root_tab_failed_task_shows_red_cross(_browser) -> None:
    """A task whose result is ``success:false`` leaves a red cross on
    the tab; the next successful run replaces it with the green tick."""
    context, page = _open_history_page(_browser)
    try:
        tab_id = _root_tab_id(page)
        _run_task(page, tab_id, "task-fail", success=False)
        _assert_spinner(_icon(page, tab_id), _GREEN, "chat-tab-spinner")
        _finish_task(page, tab_id, "task-fail")
        cross = _icon(page, tab_id)
        _assert_mask_icon(cross, "status-cross", _RED, "chat-tab-fail")
        assert "chat-tab-ok" not in cross["icon"].split(), cross

        _run_task(page, tab_id, "task-retry", success=True)
        _assert_spinner(_icon(page, tab_id), _GREEN, "chat-tab-spinner")
        _finish_task(page, tab_id, "task-retry")
        _assert_mask_icon(_icon(page, tab_id), "status-tick", _GREEN, "chat-tab-ok")
    finally:
        context.close()


def test_subagent_tab_spinner_and_purple_tick(_browser) -> None:
    """A sub-agent tab spins in purple while the sub-agent runs; a
    sub-agent the daemon announces as already finished (``isDone`` on
    ``openSubagentTab``, as on replay) shows the purple tick with the
    ``done`` marker."""
    context, page = _open_history_page(_browser)
    try:
        parent_id = _root_tab_id(page)
        _post(page, {"type": "status", "running": True, "tabId": parent_id,
                     "startTs": 1700000000000, "taskId": "parent-task"})
        running_id, done_id = "sub-tab-running", "sub-tab-done"
        for sub_id, is_done in ((running_id, False), (done_id, True)):
            _post(page, {"type": "openSubagentTab", "tab_id": sub_id,
                         "parent_tab_id": parent_id,
                         "description": "child job " + sub_id,
                         "task_id": "task-" + sub_id, "isSubagentTab": True,
                         "isDone": is_done})
        _post(page, {"type": "status", "running": True, "tabId": running_id,
                     "startTs": 1700000000000, "taskId": "task-" + running_id})

        running = _icon(page, running_id)
        _assert_spinner(running, _PURPLE, "subagent-indicator")
        assert "done" not in running["icon"].split(), running

        done = _icon(page, done_id)
        _assert_mask_icon(done, "status-tick", _PURPLE, "subagent-indicator")
        assert "done" in done["icon"].split(), done
    finally:
        context.close()

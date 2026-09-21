# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The history panel's and tab strip's colour cues are clearly visible.

Three tints must read at a glance in a real Chromium:

* the collapsible chat panel's header in the task-history panel: the
  cyan of a Bash tool call's header, strong enough to stand out from
  the sidebar background;
* the task panel whose chat webview is on screen
  (``.running-item.history-active-task``): a green tint and border;
* a sub-agent's tab in the chat tab strip: purple tint, purple text,
  purple spinner while it runs and purple tick once it is done.

The harness ``:root`` maps ``--cyan`` / ``--green`` / ``--purple`` to
``#4ec9b0`` / ``#6a9955`` / ``#c586c0`` (see
``test_history_failed_red_cross._build_test_page``); the tests compare
hues against the page's own computed variables and require an alpha
well above the 8% at which the tints used to vanish.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_codex_task_panel_style import (
    _alpha_of,
    _hue_of,
)
from kiss.tests.agents.vscode.test_history_failed_red_cross import (
    _MEDIA_DIR,
    _open_history_page,
    _post_history,
    _sample_sessions,
)

_REMOTE_CSS = _MEDIA_DIR / "remote-codex.css"

# Below this alpha a tint over the sidebar background is barely visible.
_MIN_ALPHA = 0.2


@pytest.fixture(scope="module")
def _browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _var_color(page, name: str) -> str:
    """The computed ``rgb(...)`` value of the CSS variable *name*."""
    return str(
        page.evaluate(
            """(name) => {
              const probe = document.createElement('i');
              probe.style.color = 'var(' + name + ')';
              document.body.appendChild(probe);
              const c = getComputedStyle(probe).color;
              probe.remove();
              return c;
            }""",
            name,
        )
    )


def _style(page, selector: str, prop: str) -> str:
    return str(
        page.evaluate(
            "([sel, prop]) => getComputedStyle(document.querySelector(sel))[prop]",
            [selector, prop],
        )
    )


def _assert_tint(color: str, hue_of: str, what: str) -> None:
    assert _hue_of(color) == pytest.approx(_hue_of(hue_of), abs=2), (
        f"{what} has the wrong hue: {color} vs {hue_of}"
    )
    assert _alpha_of(color) >= _MIN_ALPHA, f"{what} is barely visible: {color}"


def test_chat_panel_header_is_a_visible_cyan(_browser) -> None:
    """Every chat panel's header carries a cyan tint strong enough to
    see, and darkens further on hover."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        cyan = _var_color(page, "--cyan")
        idle = _style(page, ".history-chat-header", "backgroundColor")
        _assert_tint(idle, cyan, "the chat panel header")
        page.hover(".history-chat-header")
        hovered = _style(page, ".history-chat-header:hover", "backgroundColor")
        _assert_tint(hovered, cyan, "the hovered chat panel header")
        assert _alpha_of(hovered) > _alpha_of(idle), (idle, hovered)
    finally:
        context.close()


_ACTIVE_ROW = ".running-item.history-active-task"


def _settle(page, selector: str) -> None:
    """Wait for *selector*'s 0.15s colour transition (``.sidebar-item``
    transitions everything) to reach its final value rather than a
    mid-transition ``oklab(...)`` blend."""
    page.wait_for_function(
        "(sel) => { const cs = getComputedStyle(document.querySelector(sel));"
        " return !cs.backgroundColor.startsWith('oklab')"
        " && !cs.borderTopColor.startsWith('oklab'); }",
        arg=selector,
        timeout=5000,
    )


def _assert_active_row_green(page, what: str) -> None:
    green = _var_color(page, "--green")
    _settle(page, _ACTIVE_ROW)
    _assert_tint(_style(page, _ACTIVE_ROW, "backgroundColor"), green, what)
    border = _style(page, _ACTIVE_ROW, "borderTopColor")
    _assert_tint(border, green, what + "'s border")
    assert _alpha_of(border) >= 0.5, border


def _show_active_task(page) -> None:
    """Paint the sample history and make ``chat-run``'s task the one the
    visible tab shows."""
    _post_history(page, _sample_sessions())
    page.evaluate(
        "() => window.__post({type: 'task_settings',"
        " tabId: window._testApi.getActiveTabId(), taskId: '1002',"
        " settings: {model: 'm', chat_id: 'chat-run', task_id: 1002,"
        " start_ts: 1700000100000}})"
    )
    page.wait_for_selector(_ACTIVE_ROW, state="attached")


def _use_remote_surface(page) -> None:
    """Restyle the page as the remote webapp: ``remote-codex.css`` on
    top of ``main.css`` under ``body.remote-chat``."""
    page.evaluate(
        "(css) => { const st = document.createElement('style');"
        " st.textContent = css; document.head.appendChild(st);"
        " document.body.classList.add('remote-chat'); }",
        _REMOTE_CSS.read_text(encoding="utf-8"),
    )


def test_active_task_panel_is_a_visible_green(_browser) -> None:
    """The task panel of the task the visible tab shows is painted in a
    clearly visible green tint with a green border, hovered or not; the
    other panels are not tinted."""
    context, page = _open_history_page(_browser)
    try:
        _show_active_task(page)
        _assert_active_row_green(page, "the active task panel")
        others = page.evaluate(
            "() => [...document.querySelectorAll('.running-item:not(.history-active-task)')]"
            ".map(el => getComputedStyle(el).backgroundColor)"
        )
        assert len(others) == 2, others
        for bg in others:
            assert _alpha_of(bg) < _MIN_ALPHA, f"an inactive panel is tinted: {bg}"
        # The generic .sidebar-item:hover rule must not take the cue away.
        page.hover(_ACTIVE_ROW)
        _assert_active_row_green(page, "the hovered active task panel")
    finally:
        context.close()


def test_active_task_panel_is_a_visible_green_on_the_remote_page(_browser) -> None:
    """The remote webapp's own row and hover rules (``remote-codex.css``)
    must not take the green cue away either."""
    context, page = _open_history_page(_browser)
    try:
        _show_active_task(page)
        _use_remote_surface(page)
        _assert_active_row_green(page, "the remote active task panel")
        page.hover(_ACTIVE_ROW)
        _assert_active_row_green(page, "the hovered remote active task panel")
    finally:
        context.close()


def _open_subagent_tabs(page) -> None:
    """Open a running (``sub-run``) and a finished (``sub-done``)
    sub-agent tab under the active tab, then uncover the tab strip."""
    parent_id = page.evaluate("() => window._testApi.getActiveTabId()")
    page.evaluate(
        "(id) => window.__post({type: 'status', running: true, tabId: id,"
        " startTs: 1700000000000, taskId: 'parent-task'})",
        parent_id,
    )
    for sub_id, is_done in (("sub-run", False), ("sub-done", True)):
        page.evaluate(
            "([id, parent, done]) => window.__post({type: 'openSubagentTab',"
            " tab_id: id, parent_tab_id: parent, description: 'child ' + id,"
            " task_id: 'task-' + id, isSubagentTab: true, isDone: done})",
            [sub_id, parent_id, is_done],
        )
    page.evaluate(
        "() => window.__post({type: 'status', running: true, tabId: 'sub-run',"
        " startTs: 1700000000000, taskId: 'task-sub-run'})"
    )
    # The harness opens the history sidebar over the tab strip.
    page.evaluate("() => document.getElementById('sidebar').classList.remove('open')")


_RUNNING_TAB = '.chat-tab.subagent-tab[data-tab-id="sub-run"]'
_DONE_TAB = '.chat-tab.subagent-tab[data-tab-id="sub-done"]'


def _assert_purple_tabs(page) -> None:
    """Both sub-agent tabs are tinted purple, the active one more
    strongly, with purple text; the running one's spinner and the done
    one's tick are purple too."""
    purple = _var_color(page, "--purple")
    page.click(_DONE_TAB)
    page.wait_for_selector(_DONE_TAB + ".active", state="attached")
    for sel in (_RUNNING_TAB, _DONE_TAB):
        _assert_tint(_style(page, sel, "backgroundColor"), purple, f"tab {sel}")
    assert _alpha_of(_style(page, _DONE_TAB, "backgroundColor")) > _alpha_of(
        _style(page, _RUNNING_TAB, "backgroundColor")
    ), "the active sub-agent tab is tinted more strongly"
    assert _style(page, _DONE_TAB, "color") == purple
    assert _hue_of(_style(page, _RUNNING_TAB, "color")) == pytest.approx(
        _hue_of(purple), abs=2
    )
    spinner = _RUNNING_TAB + " .subagent-indicator.status-spinner"
    assert _style(page, spinner, "borderTopColor") == purple
    assert _style(page, spinner, "animationName") == "status-spin"
    tick = _DONE_TAB + " .subagent-indicator.done.status-tick"
    assert _style(page, tick, "backgroundColor") == purple


def test_subagent_tab_is_purple(_browser) -> None:
    """In the VS Code webviews a sub-agent's tab reads purple whether
    idle or active, and so do its spinner (running) and tick (done)."""
    context, page = _open_history_page(_browser)
    try:
        _open_subagent_tabs(page)
        _assert_purple_tabs(page)
    finally:
        context.close()


def test_subagent_tab_is_purple_on_the_remote_page(_browser) -> None:
    """The remote webapp restyles the tab strip as neutral pills
    (``remote-codex.css`` under ``body.remote-chat``); a sub-agent's tab
    must still come out purple there."""
    context, page = _open_history_page(_browser)
    try:
        _open_subagent_tabs(page)
        _use_remote_surface(page)
        # An ordinary tab is a neutral pill on the remote page ...
        plain = page.evaluate(
            "() => getComputedStyle(document.querySelector("
            "'.chat-tab:not(.subagent-tab)')).backgroundColor"
        )
        purple_hue = _hue_of(_var_color(page, "--purple"))
        assert _alpha_of(plain) == 0 or abs(_hue_of(plain) - purple_hue) > 10, plain
        # ... and a sub-agent's tab is still purple.
        _assert_purple_tabs(page)
    finally:
        context.close()

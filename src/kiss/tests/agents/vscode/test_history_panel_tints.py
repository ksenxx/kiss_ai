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
    _build_test_page,
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


def test_chat_panel_header_is_a_light_cyan(_browser) -> None:
    """Every chat panel's header carries a light cyan tint (15-20%, so
    it stays visible without dominating the list), and darkens further
    on hover."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        cyan = _var_color(page, "--cyan")
        idle = _style(page, ".history-chat-header", "backgroundColor")
        assert _hue_of(idle) == pytest.approx(_hue_of(cyan), abs=2), (idle, cyan)
        assert 0.15 <= _alpha_of(idle) <= 0.2, f"the header tint is not light: {idle}"
        page.hover(".history-chat-header")
        hovered = _style(page, ".history-chat-header:hover", "backgroundColor")
        _assert_tint(hovered, cyan, "the hovered chat panel header")
        assert _alpha_of(hovered) > _alpha_of(idle), (idle, hovered)
    finally:
        context.close()


_HEADER_INSET_JS = r"""(expand) => {
  const g = document.querySelector('#history-list .history-chat-group');
  if (g.classList.contains('collapsed') === expand) {
    g.querySelector('.history-chat-header').click();
  }
  const h = g.querySelector('.history-chat-header');
  const gr = g.getBoundingClientRect();
  const hr = h.getBoundingClientRect();
  const hcs = getComputedStyle(h);
  const row = g.querySelector('.history-chat-body > .sidebar-item');
  const rr = row ? row.getBoundingClientRect() : null;
  return {
    collapsed: g.classList.contains('collapsed'),
    top: hr.top - (gr.top + g.clientTop),
    left: hr.left - (gr.left + g.clientLeft),
    right: gr.left + g.clientLeft + g.clientWidth - hr.right,
    bottom: gr.top + g.clientTop + g.clientHeight - hr.bottom,
    radii: [hcs.borderTopLeftRadius, hcs.borderTopRightRadius,
            hcs.borderBottomRightRadius, hcs.borderBottomLeftRadius],
    rowLeft: rr ? rr.left - (gr.left + g.clientLeft) : null,
    rowGapBelowHeader: rr ? rr.top - hr.bottom : null,
  };
}"""


@pytest.mark.parametrize("expand", [False, True])
def test_chat_panel_header_meets_the_panel_border(_browser, expand: bool) -> None:
    """The header's tint touches the panel's border on every side it
    borders (no strip of bare sidebar between them), its corners follow
    the square panel (no rounding), and the task rows keep a small
    inset."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        probe = page.evaluate(_HEADER_INSET_JS, expand)
        assert probe["collapsed"] is not expand, probe
        assert probe["top"] == pytest.approx(0, abs=0.5), probe
        assert probe["left"] == pytest.approx(0, abs=0.5), probe
        assert probe["right"] == pytest.approx(0, abs=0.5), probe
        assert probe["radii"] == ["0px"] * 4, probe
        if expand:
            # The inset is one --space-1 step (4px) of the spacing scale.
            assert probe["rowLeft"] == pytest.approx(4, abs=0.5), probe
            assert probe["rowGapBelowHeader"] == pytest.approx(4, abs=0.5), probe
        else:
            assert probe["bottom"] == pytest.approx(0, abs=0.5), probe
    finally:
        context.close()


# The chat panels' edges against the history panel's boundaries: the
# left boundary is the activity bar's right edge where the bar shows
# (remote webapp), else the panel's inner left edge.
_PANEL_EDGES_JS = r"""() => {
  const sb = document.getElementById('sidebar');
  const sr = sb.getBoundingClientRect();
  const bar = document.getElementById('activity-bar');
  const barShown = getComputedStyle(bar).display !== 'none';
  const inner = sr.left + sb.clientLeft;
  const search = document.querySelector('.history-search-row')
    .getBoundingClientRect();
  const sep = document.querySelector('#history-list > .history-day-sep')
    .getBoundingClientRect();
  const groups = [...document.querySelectorAll(
    '#history-list > .history-chat-group')];
  const rects = groups.map(g => g.getBoundingClientRect());
  const leftBoundary = barShown ? bar.getBoundingClientRect().right : inner;
  const rightBoundary = inner + sb.clientWidth;
  // Rects alone miss clipping by an ancestor: probe what is painted
  // 1px inside each boundary, halfway down every header (looking
  // through the transparent drag handle on the docked panel's edge).
  const painted = groups.map(g => {
    const hr = g.querySelector('.history-chat-header').getBoundingClientRect();
    const y = (hr.top + hr.bottom) / 2;
    return [leftBoundary + 1, rightBoundary - 1].map(x => {
      const el = document.elementsFromPoint(x, y)
        .find(e => e.id !== 'sidebar-resizer');
      return !!el && el.closest('.history-chat-group') === g;
    });
  });
  return {
    painted,
    leftBoundary,
    rightBoundary,
    lefts: rects.map(r => r.left),
    rights: rects.map(r => r.right),
    gaps: rects.slice(1).map((r, i) => r.top - rects[i].bottom),
    radii: groups.map(g => getComputedStyle(g).borderRadius),
    borderTops: groups.map(g => getComputedStyle(g).borderTopWidth),
    borderSides: groups.map(g => {
      const cs = getComputedStyle(g);
      return [cs.borderLeftWidth, cs.borderRightWidth];
    }),
    insetL: search.left,
    insetR: search.right,
    sepL: sep.left,
    sepR: sep.right,
  };
}"""


def _assert_seamless_panels(page, what: str) -> None:
    _post_history(page, _sample_sessions())
    page.wait_for_function(
        "document.querySelectorAll('#history-list > .history-chat-group')"
        ".length === 3",
        timeout=5000,
    )
    e = page.evaluate(_PANEL_EDGES_JS)
    assert e["leftBoundary"] < e["insetL"], (what, e)
    for left, right in zip(e["lefts"], e["rights"], strict=True):
        assert left == pytest.approx(e["leftBoundary"], abs=0.5), (what, e)
        assert right == pytest.approx(e["rightBoundary"], abs=0.5), (what, e)
    assert e["painted"] == [[True, True]] * 3, (what, e)
    assert e["gaps"] == [pytest.approx(0, abs=0.5)] * 2, (what, e)
    assert e["radii"] == ["0px"] * 3, (what, e)
    # One shared 1px line between neighbours, none on the sides.
    assert e["borderTops"] == ["1px", "0px", "0px"], (what, e)
    assert e["borderSides"] == [["0px", "0px"]] * 3, (what, e)
    # The day separator keeps the panel's usual inset.
    assert e["sepL"] == pytest.approx(e["insetL"], abs=0.5), (what, e)
    assert e["sepR"] == pytest.approx(e["insetR"], abs=0.5), (what, e)


def test_chat_panels_are_seamless_in_the_vscode_sidebar(_browser) -> None:
    """VS Code chat webview's history drawer (16px panel padding): the
    chat panels span the panel edge to edge with no gap between them;
    the legacy flat list keeps its rows inset."""
    context, page = _open_history_page(_browser)
    try:
        _assert_seamless_panels(page, "the VS Code history drawer")
        page.click("#history-view-toggle")
        page.wait_for_selector("#history-list.legacy-view > .running-item")
        e = page.evaluate(
            "() => { const r = document.querySelector("
            "'#history-list > .running-item').getBoundingClientRect();"
            " const s = document.querySelector('.history-search-row')"
            ".getBoundingClientRect();"
            " return [r.left - s.left, s.right - r.right]; }"
        )
        assert e == [pytest.approx(0, abs=0.5)] * 2, e
    finally:
        context.close()


def test_chat_panels_are_seamless_in_history_panel_mode(_browser) -> None:
    """The primary-sidebar history view (10px panel padding)."""
    context, page = _open_history_page(_browser)
    try:
        page.evaluate("() => document.body.classList.add('history-panel-mode')")
        _assert_seamless_panels(page, "the history-panel-mode view")
    finally:
        context.close()


@pytest.mark.parametrize("desktop", [False, True])
def test_chat_panels_are_seamless_on_the_remote_page(_browser, desktop: bool) -> None:
    """The remote webapp: the phone drawer and the docked desktop panel
    (collapsing padding), both beside the activity bar."""
    context, page = _open_history_page(_browser, width=1200 if desktop else 480)
    try:
        _use_remote_surface(page)
        if desktop:
            page.evaluate("() => document.body.classList.add('remote-desktop')")
        _assert_seamless_panels(page, f"the remote page (desktop={desktop})")
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


def _open_remote_desktop_page(browser):
    """The remote desktop page with ``body.remote-chat`` set BEFORE
    main.js runs, so its remote-only wiring (the panel resizer) is live,
    and a history list long enough to scroll."""
    context = browser.new_context(viewport={"width": 1200, "height": 500})
    page = context.new_page()
    html = _build_test_page().replace("<body>", '<body class="remote-chat">', 1)
    remote_css = "<style>" + _REMOTE_CSS.read_text(encoding="utf-8") + "</style>"
    page.set_content(html.replace("</head>", remote_css + "</head>", 1))
    page.evaluate(
        "() => { document.getElementById('app').style.display = '';"
        " const ov = document.getElementById('kiss-server-loading');"
        " if (ov) ov.style.display = 'none'; }"
    )
    page.wait_for_selector("body.remote-desktop #sidebar.open", state="attached")
    assert page.evaluate("() => window.__iifeError") is None
    template = _sample_sessions()[0]
    sessions = [
        dict(template, id=f"chat-{i}", task_id=5000 + i,
             timestamp=1700000000 - i, preview=f"task {i}")
        for i in range(40)
    ]
    generation = page.evaluate(
        "() => window.__postedMessages.filter(m => m.type === 'getHistory')"
        ".at(-1)?.generation || 0"
    )
    _post_history(page, sessions, generation=generation)
    return context, page


def test_remote_scrollbar_and_resize_handle_do_not_overlap(_browser) -> None:
    """On the docked remote panel the flush history list's scrollbar
    runs along the panel's right edge; the resize handle sits just
    outside it.  Dragging the scrollbar thumb scrolls the list without
    resizing, and dragging the handle still resizes the panel."""
    # Real (non-overlay) scrollbars, as on a desktop browser.
    browser = _browser.browser_type.launch(ignore_default_args=["--hide-scrollbars"])
    try:
        context, page = _open_remote_desktop_page(browser)
        geo = page.evaluate(
            """() => {
              const l = document.getElementById('history-list');
              const lr = l.getBoundingClientRect();
              const hr = document.getElementById('sidebar-resizer')
                .getBoundingClientRect();
              return {listRight: lr.right, top: lr.top,
                      bar: l.offsetWidth - l.clientWidth,
                      overflow: l.scrollHeight > l.clientHeight,
                      handleLeft: hr.left, handleW: hr.width,
                      width: document.getElementById('sidebar')
                        .getBoundingClientRect().width};
            }"""
        )
        assert geo["overflow"] and geo["bar"] > 0, geo
        assert geo["handleLeft"] >= geo["listRight"] - 1, geo
        # The thumb, at the top of the scrollbar.
        x, y = geo["listRight"] - geo["bar"] / 2, geo["top"] + 10
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y + 100, steps=8)
        page.mouse.up()
        assert page.evaluate(
            "() => document.getElementById('history-list').scrollTop"
        ) > 0
        assert page.evaluate(
            "() => document.getElementById('sidebar')"
            ".getBoundingClientRect().width"
        ) == geo["width"]
        # The handle outside the edge is painted and clickable.
        hx = geo["handleLeft"] + geo["handleW"] / 2
        assert page.evaluate(
            "([x, y]) => document.elementFromPoint(x, y).id", [hx, 250]
        ) == "sidebar-resizer"
        page.mouse.move(hx, 250)
        page.mouse.down()
        page.mouse.move(hx + 60, 250, steps=8)
        page.mouse.up()
        assert page.evaluate(
            "() => document.getElementById('sidebar')"
            ".getBoundingClientRect().width"
        ) == pytest.approx(geo["width"] + 60, abs=2)
        # Dragged down to its 10px minimum, the panel's content (the
        # 40px activity bar, the history rows) stays clipped at the
        # panel's edge: nothing of it shows or takes clicks beside it.
        page.mouse.move(hx + 60, 250)
        page.mouse.down()
        page.mouse.move(0, 250, steps=8)
        page.mouse.up()
        leaks = page.evaluate(
            """() => {
              const sb = document.getElementById('sidebar');
              const right = sb.getBoundingClientRect().right;
              const out = [];
              for (const y of [60, 250]) {
                for (let x = right + 0.5; x < right + 12; x += 1) {
                  const el = document.elementsFromPoint(x, y)
                    .find(e => e.id !== 'sidebar-resizer');
                  if (el && sb.contains(el)) out.push([x, y, el.id || el.tagName]);
                }
              }
              return {right, out};
            }"""
        )
        assert leaks["right"] < 20 and leaks["out"] == [], leaks
        context.close()
    finally:
        browser.close()

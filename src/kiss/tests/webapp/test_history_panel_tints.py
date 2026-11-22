# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The history panel's and tab strip's state cues are clearly visible.

Four cues must read at a glance in a real Chromium:

* the collapsible chat panel's header in the task-history panel: a
  plain row with no fill of its own and the quiet NEUTRAL hover fill
  (no hue) under the pointer, so a list of many chats does not become
  a stack of bands;
* the header of the chat whose page is on screen
  (``.history-chat-group.history-active-chat``): the neutral selection
  fill under the strongest text, hovered or not, folded or not, with
  no accent wash and no accent bar;
* the task panel whose chat page is on screen
  (``.running-item.history-active-task``): the same selection fill and
  text, with no accent tint and no outline;
* a sub-agent's tab in the chat tab strip: purple tint, purple text
  and a purple spinner (a sub-agent has a tab only while it runs).

The harness ``:root`` maps ``--purple`` to ``#c586c0`` (see
``test_history_failed_red_cross._build_test_page``) and leaves the
neutral fills at main.css's dark fallbacks; the tests compare the
colours against the page's own computed variables.  A selection must
clearly outweigh the hover fill, and a purple tint must keep an alpha
well above the 8% at which the tints used to vanish.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.webapp.test_codex_task_panel_style import (
    _alpha_of,
    _chroma_of,
    _hue_of,
)
from kiss.tests.webapp.test_history_failed_red_cross import (
    _MEDIA_DIR,
    _build_test_page,
    _open_history_page,
    _post_history,
    _sample_sessions,
)
from kiss.tests.webapp.test_history_filter_menu import pick_filter_option

_REMOTE_CSS = _MEDIA_DIR / "remote-codex.css"

# On the harness's dark sidebar a white fill below this alpha reads as a
# row under the pointer, not as the selected row.
_MIN_SELECTED_ALPHA = 0.1

# The selected row must stay the strongest fill of the list while the
# pointer rests on another row.
_MIN_SELECTED_OVER_HOVER = 1.5

# A sub-agent tab is a quiet cue: its purple text carries most of the
# signal, the fill only has to be visible.
_MIN_TAB_ALPHA = 0.15


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


def _assert_tint(color: str, hue_of: str, what: str, min_alpha: float) -> None:
    assert _hue_of(color) == pytest.approx(_hue_of(hue_of), abs=2), (
        f"{what} has the wrong hue: {color} vs {hue_of}"
    )
    assert _alpha_of(color) >= min_alpha, f"{what} is barely visible: {color}"


def test_chat_panel_header_is_a_quiet_neutral(_browser) -> None:
    """The header of a chat that is not on screen is a plain row (no
    fill of its own) and takes the faint neutral hover fill (--hover,
    no hue) under the pointer: weaker than the selection fill that
    marks the chat on screen (checked by
    test_active_chat_header_carries_the_selection_fill)."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        idle = _style(page, ".history-chat-header", "backgroundColor")
        assert _alpha_of(idle) == 0, f"the idle header has a fill: {idle}"
        page.hover(".history-chat-header")
        _settle(page, ".history-chat-header:hover")
        hovered = _style(page, ".history-chat-header:hover", "backgroundColor")
        assert hovered == _var_color(page, "--hover"), hovered
        assert _chroma_of(hovered) <= 8, f"the hovered header has a hue: {hovered}"
        assert 0.03 <= _alpha_of(hovered) <= 0.1, f"the hover tint is off: {hovered}"
    finally:
        context.close()


_HEADER_LINE_JS = r"""(expand) => {
  const list = document.getElementById('history-list');
  const g = list.querySelector('.history-chat-group');
  const h = g.querySelector('.history-chat-header');
  const menu = h.querySelector(':scope > .sidebar-item-actions');
  const chevron = h.querySelector(':scope > .history-chat-collapse');
  if (g.classList.contains('collapsed') === expand) chevron.click();
  const lr = list.getBoundingClientRect();
  const gr = g.getBoundingClientRect();
  const hr = h.getBoundingClientRect();
  const hcs = getComputedStyle(h);
  const tr = h.querySelector('.history-chat-title').getBoundingClientRect();
  const more = h.querySelector(':scope > .sidebar-item-more');
  const mr = more.getBoundingClientRect();
  const cr = chevron.getBoundingClientRect();
  const menuBox = menu.getBoundingClientRect();
  const row = g.querySelector('.history-chat-body > .sidebar-item');
  const rr = row ? row.getBoundingClientRect() : null;
  return {
    collapsed: g.classList.contains('collapsed'),
    top: hr.top - gr.top,
    bottom: gr.bottom - hr.bottom,
    left: hr.left - lr.left,
    right: lr.right - hr.right,
    paddingAroundTitle: hr.height - tr.height,
    radii: [hcs.borderTopLeftRadius, hcs.borderTopRightRadius,
            hcs.borderBottomRightRadius, hcs.borderBottomLeftRadius],
    borders: [hcs.borderTopWidth, hcs.borderRightWidth,
              hcs.borderBottomWidth, hcs.borderLeftWidth],
    moreOnTitleLine: mr.top < tr.bottom && mr.bottom > tr.top,
    moreAfterTitle: mr.left >= tr.right,
    moreToFillEdge: hr.right - mr.right,
    chevronOnTitleLine: cr.top < tr.bottom && cr.bottom > tr.top,
    chevronAfterTitle: cr.left >= tr.right,
    chevronToMore: mr.left - cr.right,
    chevronBox: [cr.width, cr.height],
    moreBox: [mr.width, mr.height],
    moreOpensMenu: more.popoverTargetElement === menu,
    menuOpen: menu.matches(':popover-open'),
    menuBox: [menuBox.width, menuBox.height],
    chevronTooltip: chevron.dataset.tooltip,
    launchedText: menu.querySelector(':scope > .history-chat-launched').textContent,
    rowLeft: rr ? rr.left - gr.left : null,
    rowRightToFillEdge: rr ? hr.right - rr.right : null,
    rowGapBelowHeader: rr ? rr.top - hr.bottom : null,
  };
}"""


@pytest.mark.parametrize("expand", [False, True])
def test_chat_panel_header_is_one_inset_line(_browser, expand: bool) -> None:
    """The header row is one line: the title and, at its right end, the
    "Show details" chevron and the "…" button, two boxes of one size
    that do not make the row taller.  The row's fill (pointer, chat on
    screen) keeps 8px from the list's left and right edges - the room
    the list keeps for its scrollbar counted in - and 1px from the rows
    above and below, under an 8px corner.  The row's actions and the
    chat's "... ago" label are in the "…" button's menu, closed until
    the button opens it; the task rows the chevron unfolds are indented
    behind the nesting guide and their fills end where the header's
    does."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        probe = page.evaluate(_HEADER_LINE_JS, expand)
        assert probe["collapsed"] is not expand, probe
        assert probe["chevronTooltip"] == ("Hide details" if expand else "Show details"), probe
        assert probe["top"] == pytest.approx(1, abs=0.5), probe
        assert probe["left"] == pytest.approx(8, abs=0.5), probe
        assert probe["right"] == pytest.approx(8, abs=0.5), probe
        assert probe["radii"] == ["8px"] * 4, probe
        assert probe["borders"] == ["0px"] * 4, probe
        # The row is as tall as its title's line and the 6px above and
        # below it: the 24px button hangs over that padding.
        assert probe["paddingAroundTitle"] == pytest.approx(12, abs=0.5), probe
        assert probe["moreOnTitleLine"] is True, probe
        assert probe["moreAfterTitle"] is True, probe
        assert probe["moreToFillEdge"] == pytest.approx(4, abs=0.5), probe
        assert probe["chevronOnTitleLine"] is True, probe
        assert probe["chevronAfterTitle"] is True, probe
        assert probe["chevronToMore"] == pytest.approx(0, abs=0.5), probe
        assert probe["chevronBox"] == probe["moreBox"], probe
        assert probe["moreOpensMenu"] is True, probe
        assert probe["menuOpen"] is False and probe["menuBox"] == [0, 0], probe
        assert probe["launchedText"].endswith(" ago"), probe
        assert "last launched" not in probe["launchedText"], probe
        if expand:
            # .history-chat-body: a 20px margin, the 1px guide line and
            # a 4px padding (--space-4 + --space-1, 1px, --space-1).
            assert probe["rowLeft"] == pytest.approx(25, abs=0.5), probe
            assert probe["rowRightToFillEdge"] == pytest.approx(0, abs=0.5), probe
            assert probe["rowGapBelowHeader"] == pytest.approx(1, abs=0.5), probe
        else:
            assert probe["bottom"] == pytest.approx(1, abs=0.5), probe
    finally:
        context.close()


# Per chat header: where the title starts and the centre of the status
# icon (a turning spinner keeps its centre) from the header's left
# edge, and how far the "…" button ends before its right edge.
_TITLE_EDGES_JS = r"""() => [...document.querySelectorAll(
    '#history-list .history-chat-header')].map(h => {
  const left = el => el.getBoundingClientRect().left - h.getBoundingClientRect().left;
  const icon = h.querySelector(':scope > .history-chat-status');
  return {
    title: left(h.querySelector('.history-chat-title')),
    more: h.getBoundingClientRect().right
      - h.querySelector(':scope > .sidebar-item-more').getBoundingClientRect().right,
    iconCentre: left(icon) + icon.getBoundingClientRect().width / 2,
  };
})"""


def test_chat_titles_share_one_edge_behind_any_status_icon(_browser) -> None:
    """The header's first column is the status icon's slot: the titles
    of a failed, a running, an asking and a succeeded chat start on one
    edge (and their "…" buttons end on one), so a title stays put when
    its icon changes; each icon, the narrow "?" included, is centred in
    the 12px slot that ends 6px before the title."""
    context, page = _open_history_page(_browser)
    try:
        sessions = _sample_sessions()
        asking = {**sessions[1], "id": "chat-ask", "task_id": 1004, "awaiting_answer": True}
        _post_history(page, [*sessions, asking])
        rows = page.evaluate(_TITLE_EDGES_JS)
        assert len(rows) == 4, rows
        for row in rows:
            assert row["title"] == pytest.approx(rows[0]["title"], abs=0.5), rows
            assert row["more"] == pytest.approx(rows[0]["more"], abs=0.5), rows
            assert row["title"] - row["iconCentre"] == pytest.approx(12, abs=0.5), rows
    finally:
        context.close()


# The chat panels' edges against the history panel's boundaries: its
# inner left and right edges.
_PANEL_EDGES_JS = r"""() => {
  const sb = document.getElementById('sidebar');
  const sr = sb.getBoundingClientRect();
  const inner = sr.left + sb.clientLeft;
  const list = document.getElementById('history-list');
  const search = document.querySelector('.history-search-row')
    .getBoundingClientRect();
  const sep = document.querySelector('#history-list > .history-day-sep')
    .getBoundingClientRect();
  const groups = [...document.querySelectorAll(
    '#history-list > .history-chat-group')];
  const rects = groups.map(g => g.getBoundingClientRect());
  const headers = groups.map(g => g.querySelector('.history-chat-header'));
  const leftBoundary = inner;
  const rightBoundary = inner + sb.clientWidth;
  // Rects alone miss clipping by an ancestor: probe what is painted
  // 1px inside each end of every header's fill, halfway down the
  // header.
  const painted = headers.map(h => {
    const hr = h.getBoundingClientRect();
    const y = (hr.top + hr.bottom) / 2;
    return [hr.left + 1, hr.right - 1].map(x => {
      const el = document.elementFromPoint(x, y);
      return !!el && el.closest('.history-chat-header') === h;
    });
  });
  return {
    painted,
    leftBoundary,
    rightBoundary,
    // The room the list keeps for its scrollbar.
    gutter: list.offsetWidth - list.clientWidth,
    lefts: rects.map(r => r.left),
    rights: rects.map(r => r.right),
    fills: headers.map(h => {
      const hr = h.getBoundingClientRect();
      return [hr.left - leftBoundary, rightBoundary - hr.right];
    }),
    gaps: rects.slice(1).map((r, i) => r.top - rects[i].bottom),
    sectionHeight: document.querySelector(
      '#history-list > .history-day-sep[data-day]:not([data-day="running"])'
    ).getBoundingClientRect().height,
    sections: [...document.querySelectorAll('#history-list > .history-day-sep')]
      .map(el => el.textContent),
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
    # A desktop-wide page asks for the history at load (the docked
    # sidebar opens on its own), which bumps the generation a reply
    # must carry; a narrower page has asked for nothing yet.
    generation = page.evaluate(
        "() => window.__postedMessages.filter(m => m.type === 'getHistory')"
        ".at(-1)?.generation || 0"
    )
    _post_history(page, _sample_sessions(), generation=generation)
    page.wait_for_function(
        "document.querySelectorAll('#history-list > .history-chat-group')"
        ".length === 3",
        timeout=5000,
    )
    e = page.evaluate(_PANEL_EDGES_JS)
    assert e["leftBoundary"] < e["insetL"], (what, e)
    # The panels span the list: from the panel's left edge to the room
    # the list keeps for its scrollbar on a page driven by a mouse.
    assert e["gutter"] == 6, (what, e)
    for left, right in zip(e["lefts"], e["rights"], strict=True):
        assert left == pytest.approx(e["leftBoundary"], abs=0.5), (what, e)
        assert right == pytest.approx(e["rightBoundary"] - e["gutter"], abs=0.5), (what, e)
    # A header's fill keeps the same 8px from both of the panel's
    # edges, that room counted in, and none of it is clipped away.
    assert e["fills"] == [[pytest.approx(8, abs=0.5)] * 2] * 3, (what, e)
    assert e["painted"] == [[True, True]] * 3, (what, e)
    assert e["sections"][0] == "Running", (what, e)
    assert len(e["sections"]) == 2 and e["sectionHeight"] > 0, (what, e)
    assert e["gaps"] == [
        pytest.approx(e["sectionHeight"], abs=0.5), pytest.approx(0, abs=0.5),
    ], (what, e)
    assert e["radii"] == ["0px"] * 3, (what, e)
    # A flat list: no line between chat panels, within a section or after
    # a section's separator.
    assert e["borderTops"] == ["0px"] * 3, (what, e)
    assert e["borderSides"] == [["0px", "0px"]] * 3, (what, e)
    # The day separator keeps the panel's usual inset.
    assert e["sepL"] == pytest.approx(e["insetL"], abs=0.5), (what, e)
    assert e["sepR"] == pytest.approx(e["insetR"], abs=0.5), (what, e)


def test_legacy_list_rows_stay_inset(_browser) -> None:
    """The chat panels span the history drawer with no gap between
    them; the legacy flat list's rows keep the chat headers' inset,
    their fills 8px from both of the panel's edges."""
    context, page = _open_history_page(_browser)
    try:
        _use_remote_surface(page)
        _assert_seamless_panels(page, "the history drawer")
        pick_filter_option(page, "group", "None")
        page.wait_for_selector("#history-list.legacy-view > .running-item")
        e = page.evaluate(
            "() => { const r = document.querySelector("
            "'#history-list > .running-item').getBoundingClientRect();"
            " const sb = document.getElementById('sidebar');"
            " const inner = sb.getBoundingClientRect().left + sb.clientLeft;"
            " return [r.left - inner, inner + sb.clientWidth - r.right]; }"
        )
        assert e == [pytest.approx(8, abs=0.5)] * 2, e
    finally:
        context.close()


@pytest.mark.parametrize("desktop", [False, True])
def test_chat_panels_are_seamless_on_the_remote_page(_browser, desktop: bool) -> None:
    """The phone drawer and the docked desktop panel (collapsing
    padding)."""
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
    """Wait for *selector*'s colour transition (a history row and a chat
    header fade their fill in) to end, so that the computed colours are
    the final ones rather than a blend on the way there."""
    page.wait_for_function(
        "(sel) => { const el = document.querySelector(sel);"
        " return !!el && !el.getAnimations().some(a => a instanceof CSSTransition); }",
        arg=selector,
        timeout=5000,
    )


def _assert_selected(page, selector: str, what: str) -> None:
    """*selector* carries the selection cue: the page's neutral
    ``--selected`` fill, clearly stronger than the hover fill, under the
    strongest text, with no accent tint, no outline and no accent bar."""
    _settle(page, selector)
    fill = _style(page, selector, "backgroundColor")
    assert fill == _var_color(page, "--selected"), f"{what} is not the selection fill: {fill}"
    assert _chroma_of(fill) <= 8, f"{what} has a hue: {fill}"
    assert _alpha_of(fill) >= _MIN_SELECTED_ALPHA, f"{what} is barely visible: {fill}"
    hover = _var_color(page, "--hover")
    assert _alpha_of(fill) >= _MIN_SELECTED_OVER_HOVER * _alpha_of(hover), (
        f"{what} does not outweigh a row under the pointer: {fill} vs {hover}"
    )
    assert _style(page, selector, "color") == _var_color(page, "--fg-strong"), (
        f"{what} must take the strongest text"
    )
    border = _style(page, selector, "borderTopColor")
    width = _style(page, selector, "borderTopWidth")
    assert width == "0px" or _alpha_of(border) == 0, f"{what} is outlined: {width} {border}"
    shadow = _style(page, selector, "boxShadow")
    assert shadow == "none", f"{what} carries a bar: {shadow}"


_ACTIVE_HEADER = ".history-chat-group.history-active-chat > .history-chat-header"
_OTHER_HEADER = ".history-chat-group:not(.history-active-chat) > .history-chat-header"


def _show_active_task(page) -> None:
    """Paint the sample history (``_post_history`` opens every chat
    panel through its chevron, as a user would) and make ``chat-run``'s
    task the one the visible tab shows."""
    _post_history(page, _sample_sessions())
    page.evaluate(
        "() => window.__post({type: 'task_settings',"
        " tabId: window._testApi.getActiveTabId(), taskId: '1002',"
        " settings: {model: 'm', chat_id: 'chat-run', task_id: 1002,"
        " start_ts: 1700000100000}})"
    )
    page.wait_for_selector(_ACTIVE_ROW, state="visible")
    page.wait_for_selector(_ACTIVE_HEADER, state="visible")


def _fold_active_chat(page) -> None:
    """The user folds the chat on screen through the chevron left of
    its "…" button: the task row hides, the header stays."""
    page.click(_ACTIVE_HEADER + " > .history-chat-collapse")
    page.wait_for_selector(_ACTIVE_ROW, state="hidden")


def _use_remote_surface(page) -> None:
    """Restyle the page as the remote webapp: ``remote-codex.css`` on
    top of ``main.css`` under ``body.remote-chat``."""
    page.evaluate(
        "(css) => { const st = document.createElement('style');"
        " st.textContent = css; document.head.appendChild(st);"
        " document.body.classList.add('remote-chat'); }",
        _REMOTE_CSS.read_text(encoding="utf-8"),
    )


def test_active_task_panel_carries_the_selection_fill(_browser) -> None:
    """The task panel of the task the visible tab shows is painted in
    the clearly visible neutral selection fill under the strongest
    text, hovered or not; the other panels have no fill."""
    context, page = _open_history_page(_browser)
    try:
        _show_active_task(page)
        _use_remote_surface(page)
        _assert_selected(page, _ACTIVE_ROW, "the active task panel")
        others = page.evaluate(
            "() => [...document.querySelectorAll('.running-item:not(.history-active-task)')]"
            ".map(el => getComputedStyle(el).backgroundColor)"
        )
        assert len(others) == 2, others
        for bg in others:
            assert _alpha_of(bg) == 0, f"an inactive panel is filled: {bg}"
        # The generic .sidebar-item:hover rule must not take the cue away.
        page.hover(_ACTIVE_ROW)
        _assert_selected(page, _ACTIVE_ROW + ":hover", "the hovered active task panel")
    finally:
        context.close()


def test_active_chat_header_carries_the_selection_fill(_browser) -> None:
    """The header of the chat the visible tab shows is painted in the
    neutral selection fill under the strongest text, unfolded or folded
    by the user, hovered or not; the other headers keep no fill."""
    context, page = _open_history_page(_browser)
    try:
        _show_active_task(page)
        _use_remote_surface(page)
        _assert_selected(page, _ACTIVE_HEADER, "the active chat header")
        others = page.evaluate(
            "(sel) => [...document.querySelectorAll(sel)]"
            ".map(el => getComputedStyle(el).backgroundColor)",
            _OTHER_HEADER,
        )
        assert len(others) == 2, others
        for bg in others:
            assert _alpha_of(bg) == 0, f"another chat's header is filled: {bg}"
        page.hover(_ACTIVE_HEADER)
        _assert_selected(page, _ACTIVE_HEADER + ":hover", "the hovered active chat header")
        _fold_active_chat(page)
        _assert_selected(page, _ACTIVE_HEADER, "the folded active chat header")
    finally:
        context.close()


def _open_subagent_tabs(page) -> None:
    """Open two running sub-agent tabs (``sub-idle``, ``sub-shown``)
    under the active tab, then uncover the tab strip."""
    parent_id = page.evaluate("() => window._testApi.getActiveTabId()")
    page.evaluate(
        "(id) => window.__post({type: 'status', running: true, tabId: id,"
        " startTs: 1700000000000, taskId: 'parent-task'})",
        parent_id,
    )
    for sub_id in ("sub-idle", "sub-shown"):
        page.evaluate(
            "([id, parent]) => { window.__post({type: 'openSubagentTab',"
            " tab_id: id, parent_tab_id: parent, description: 'child ' + id,"
            " task_id: 'task-' + id, isSubagentTab: true});"
            " window.__post({type: 'status', running: true, tabId: id,"
            " startTs: 1700000000000, taskId: 'task-' + id}); }",
            [sub_id, parent_id],
        )
    # The harness opens the history sidebar over the tab strip.
    page.evaluate("() => document.getElementById('sidebar').classList.remove('open')")


_IDLE_TAB = '.chat-tab.subagent-tab[data-tab-id="sub-idle"]'
_SHOWN_TAB = '.chat-tab.subagent-tab[data-tab-id="sub-shown"]'


def _assert_purple_tabs(page) -> None:
    """Both sub-agent tabs are tinted purple, the active one more
    strongly, with purple text and a purple spinner."""
    purple = _var_color(page, "--purple")
    page.click(_SHOWN_TAB)
    page.wait_for_selector(_SHOWN_TAB + ".active", state="attached")
    # A tab keeps its node when it becomes active, so on the remote page
    # its chip fades to the active colours (``transition`` in
    # remote-codex.css): the colours are read once that has run.
    page.wait_for_function(
        "() => !document.getAnimations().some(a => a instanceof CSSTransition)"
    )
    for sel in (_IDLE_TAB, _SHOWN_TAB):
        _assert_tint(_style(page, sel, "backgroundColor"), purple, f"tab {sel}", _MIN_TAB_ALPHA)
        spinner = sel + " .subagent-indicator.status-spinner"
        assert _style(page, spinner, "borderTopColor") == purple
        assert _style(page, spinner, "animationName") == "status-spin"
    assert _alpha_of(_style(page, _SHOWN_TAB, "backgroundColor")) > _alpha_of(
        _style(page, _IDLE_TAB, "backgroundColor")
    ), "the active sub-agent tab is tinted more strongly"
    assert _style(page, _SHOWN_TAB, "color") == purple
    assert _hue_of(_style(page, _IDLE_TAB, "color")) == pytest.approx(
        _hue_of(purple), abs=2
    )


def test_subagent_tab_is_purple(_browser) -> None:
    """The tab strip is quiet neutral chips (``remote-codex.css`` under
    ``body.remote-chat``); a sub-agent's tab reads purple whether idle
    or active, and so does its spinner."""
    context, page = _open_history_page(_browser)
    try:
        _open_subagent_tabs(page)
        _use_remote_surface(page)
        # An ordinary tab is a neutral chip on the remote page ...
        plain = page.evaluate(
            "() => getComputedStyle(document.querySelector("
            "'.chat-tab:not(.subagent-tab)')).backgroundColor"
        )
        assert _alpha_of(plain) == 0 or _chroma_of(plain) <= 8, plain
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
        # search row, the history rows) stays clipped at the
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

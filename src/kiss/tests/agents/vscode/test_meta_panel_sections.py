# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end geometry tests for the task-info panel's sections.

The right-hand task-info panel stacks the Task Info list and the Task
update report as SECTIONS (``.meta-section`` in ``chat.html``, the
``metasections`` block of ``main.js``).  Each one is collapsible from
its header chevron, scrolls inside its share of the panel, and the
separator between them (``.meta-section-resizer``) drags the boundary.
These checks need a layout engine, so they run the real remote webapp
page (``_build_html()`` plus the real media assets) in headless
Chromium and measure:

* both sections start expanded and their bodies share the panel
  equally, whatever their content; the Task update body reaches the
  bottom of the panel and scrolls;
* on every surface (remote desktop and mobile drawer, VS Code Task Info
  view and sidebar-chat drawer) all five expanded bodies are equally
  tall, and a drag leaves the bodies above the moved boundary alone;
* the Spend section's heatmap fills the panel's width with 11px week
  columns, pinned to the latest week, scrolls back to the oldest day
  with its pager, shades the cells by cost and shows a tooltip above
  the hovered cell;
* dragging the separator down grows the list and shrinks the report by
  the same amount, the height persists across a reload, and a
  double-click restores the equal share; the arrow keys resize too;
* collapsing a section hides its body (the header stays) and the
  remaining expanded section fills the panel; collapsing both leaves
  two headers and no separator handle;
* on a short window both headers stay visible and the bodies scroll;
* the mobile drawer shows the same sections, its close button clear of
  the first header.

No mocks, patches or fakes: a real HTTP server serves the real assets
to a real browser.  The page's outgoing ``postMessage`` calls are
recorded (the same observation the jsdom suites make) so the test can
answer the ``getTaskUpdate`` poll with the token the page chose.
"""

from __future__ import annotations

import datetime
import functools
import http.server
import json
import math
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from playwright.sync_api import Browser, Locator, Page, sync_playwright

from kiss.server.web_server import MEDIA_DIR, _build_html
from kiss.tests.conftest import (
    goto_retrying_network_change,
    reload_retrying_network_change,
)

# main.js only reveals #app once the websocket handshake succeeds,
# which never happens against a static server; an !important rule
# beats the inline re-hiding (see test_remote_desktop_layout.py).
_PREPARE_JS = """
() => {
  const style = document.createElement('style');
  style.textContent = `
    #app { display: flex !important; }
    #kiss-server-loading { display: none !important; }
    #auth-modal { display: none !important; }
  `;
  document.head.appendChild(style);
}
"""

# Runs before any page script: records every message main.js posts
# through the remote shim's acquireVsCodeApi() in window.__posted.
_RECORD_POSTS_JS = """
(() => {
  window.__posted = [];
  let real;
  Object.defineProperty(window, 'acquireVsCodeApi', {
    configurable: true,
    get() {
      if (!real) return undefined;
      return function () {
        const api = real();
        const post = api.postMessage;
        api.postMessage = msg => { window.__posted.push(msg); return post.call(api, msg); };
        return api;
      };
    },
    set(fn) { real = fn; },
  });
})();
"""

# A report long enough to overflow any panel used below.
_LONG_REPORT = "".join(
    f"<p>Step {i}: read the data, ran the baseline and recorded the metrics.</p>" for i in range(80)
)

# A third panel, added the way chat.html documents it: one more
# section (header with a toggle, a body) plus its resizer.
_EXTRA_SECTION = (
    '<section class="meta-section" id="meta-section-extra">'
    '<div class="sidebar-hdr meta-section-hdr">'
    '<button type="button" class="meta-section-toggle" aria-expanded="true" '
    'aria-controls="meta-extra-body"><span>Extra</span></button></div>'
    '<div id="meta-extra-body" class="meta-section-body">'
    + "".join(f"<p>extra line {i}</p>" for i in range(40))
    + "</div></section>"
    '<div class="meta-section-resizer" role="separator" aria-orientation="horizontal" '
    'aria-label="Resize the Extra section" tabindex="0"></div>'
)

_GEOMETRY_JS = """
() => {
  const rect = id => document.getElementById(id).getBoundingClientRect();
  const shown = el => el.getClientRects().length > 0;
  const panel = document.getElementById('meta-panel');
  const pr = panel.getBoundingClientRect();
  const pad = parseFloat(getComputedStyle(panel).paddingBottom);
  const list = document.getElementById('meta-list');
  const content = document.getElementById('meta-info-content');
  // The per-task sections only: the global Schedule, Apps and Spend
  // sections (hidden by _open_page unless asked for) are measured
  // separately.
  const isGlobal = el =>
    el && ['meta-schedule', 'meta-apps', 'meta-spend'].includes(el.id);
  const resizers = Array.from(
    document.querySelectorAll('#meta-panel > .meta-section-resizer'))
    .filter(r => !isGlobal(r.previousElementSibling));
  const hdrs = Array.from(document.querySelectorAll('#meta-panel .meta-section-hdr'))
    .filter(h => !isGlobal(h.parentElement));
  return {
    panelInnerBottom: pr.bottom - pad,
    panelTop: pr.top,
    listShown: shown(list),
    listTop: rect('meta-list').top,
    listHeight: rect('meta-list').height,
    listScrolls: list.scrollHeight > list.clientHeight + 1,
    contentShown: shown(content),
    contentHeight: rect('meta-info-content').height,
    contentBottom: rect('meta-info-content').bottom,
    contentScrolls: content.scrollHeight > content.clientHeight + 1,
    headers: hdrs.map(h => {
      const r = h.getBoundingClientRect();
      const toggle = h.querySelector('.meta-section-toggle');
      return {text: toggle.textContent,
              top: r.top, bottom: r.bottom,
              toggleRight: toggle.getBoundingClientRect().right,
              expanded: toggle.getAttribute('aria-expanded')};
    }),
    resizers: resizers.map(r => ({
      shown: shown(r), handle: !r.hidden && !r.classList.contains('static'),
      cursor: shown(r) ? getComputedStyle(r).cursor : null,
      top: r.getBoundingClientRect().top,
      height: r.getBoundingClientRect().height,
    })),
    storedH: localStorage.getItem('kiss-meta-section-h:meta-section-info'),
    storedCollapsed: {
      info: localStorage.getItem('kiss-meta-section-collapsed:meta-section-info'),
      update: localStorage.getItem('kiss-meta-section-collapsed:meta-info'),
    },
  };
}
"""


def _serve(directory: Path) -> Iterator[str]:
    """Serve ``directory`` over HTTP on an ephemeral port; yield its URL."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(directory),
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def remote_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve the real remote HTML page plus the real media assets.

    ``three.html`` beside it is the same page with a third section
    appended after Task update (see ``three_sections_url``)."""
    root = tmp_path_factory.mktemp("remote-webapp")
    html = _build_html()
    (root / "index.html").write_text(html, encoding="utf-8")
    marker = '<div id="meta-resizer"'
    assert marker in html
    (root / "three.html").write_text(
        html.replace(marker, _EXTRA_SECTION + marker), encoding="utf-8"
    )
    (root / "media").symlink_to(MEDIA_DIR, target_is_directory=True)
    yield from _serve(root)


@pytest.fixture(scope="module")
def three_sections_url(remote_url: str) -> str:
    """The remote page with a third panel added per the chat.html recipe."""
    return remote_url.replace("index.html", "three.html")


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """Launch one headless Chromium for the whole module."""
    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        try:
            yield chromium
        finally:
            chromium.close()


# Takes the global Schedule, Apps and Spend sections out of the stack
# (the `hidden` attribute; a Task Info toggle round trip re-applies the
# layout), leaving the per-task sections these geometry tests measure.
_HIDE_GLOBAL_SECTIONS_JS = """
() => {
  document.getElementById('meta-schedule').hidden = true;
  document.getElementById('meta-apps').hidden = true;
  document.getElementById('meta-spend').hidden = true;
  const toggle = document.querySelector('#meta-section-info .meta-section-toggle');
  toggle.click();
  toggle.click();
}
"""


def _open_page(
    browser: Browser,
    url: str,
    width: int,
    height: int = 900,
    global_sections: bool = False,
    storage: dict[str, str] | None = None,
) -> Page:
    """Open the remote page at the given viewport with post recording.

    Unless ``global_sections``, the Schedule, Apps and Spend sections
    are hidden so only the per-task sections share the panel.
    ``storage`` entries land in localStorage before the page's scripts
    run, the way a previous visit would have left them."""
    page = browser.new_page(viewport={"width": width, "height": height})
    page.add_init_script(_RECORD_POSTS_JS)
    if storage:
        page.add_init_script(
            f"for (const [k, v] of Object.entries({json.dumps(storage)})) "
            "localStorage.setItem(k, v);"
        )
    goto_retrying_network_change(page, url)
    page.wait_for_selector("body.remote-chat", state="attached")
    page.evaluate(_PREPARE_JS)
    if not global_sections:
        page.evaluate(_HIDE_GLOBAL_SECTIONS_JS)
    return page


def _deliver(page: Page, data: dict[str, object]) -> None:
    """Deliver a daemon message to main.js the way the shim does."""
    page.evaluate(
        "data => window.dispatchEvent(new MessageEvent('message', {data}))",
        data,
    )


def _start_running_task(page: Page) -> dict[str, Any]:
    """Mark the task running and return its first ``getTaskUpdate`` poll.

    The page has no WebSocket server behind it, so the shim keeps
    reporting ``daemonStatus connected:false`` on every failed attempt
    and main.js holds its pollers back while the daemon is down.  The
    connected report and the ``status`` are dispatched in ONE evaluate
    so no shim report can land between them."""
    poll = page.evaluate(
        """() => {
          const deliver = data =>
            window.dispatchEvent(new MessageEvent('message', {data}));
          deliver({type: 'daemonStatus', connected: true});
          deliver({type: 'status', running: true});
          return window.__posted.filter(m => m.type === 'getTaskUpdate').pop() || null;
        }"""
    )
    assert poll is not None, "a running task must start the getTaskUpdate poll"
    return cast(dict[str, Any], poll)


def _show_task_update(page: Page, html: str) -> None:
    """Run a task and answer its getTaskUpdate poll with ``html``."""
    _deliver(page, {"type": "configData", "config": {}, "apiKeys": {}})
    poll = _start_running_task(page)
    _deliver(
        page,
        {
            "type": "taskUpdate",
            "tabId": poll["tabId"],
            "token": poll["token"],
            "taskId": "task-1",
            "exists": True,
            "sig": "sig-1",
            "content": html,
            "error": "",
            "running": False,
            "cost": 0,
            "updatedAt": 0,
        },
    )
    page.wait_for_selector("#meta-info.visible", state="attached")


def _geometry(page: Page) -> dict[str, Any]:
    """Measure the panel's sections; see ``_GEOMETRY_JS``."""
    geo: dict[str, Any] = page.evaluate(_GEOMETRY_JS)
    return geo


def _section_resizer(page: Page) -> Locator:
    """The separator after the Task Info section."""
    return page.locator("#meta-section-info + .meta-section-resizer")


def _drag_separator_by(page: Page, dy: float) -> None:
    box = _section_resizer(page).bounding_box()
    assert box is not None
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y + dy, steps=8)
    page.mouse.up()


def _toggle(page: Page, section_id: str) -> None:
    page.locator(f"#{section_id} .meta-section-toggle").click()


def test_both_sections_expanded_share_the_panel_equally(
    browser: Browser, remote_url: str
) -> None:
    page = _open_page(browser, remote_url, 1200)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        geo = _geometry(page)
        # No task yet: Task Info alone, expanded, no separator drawn.
        assert [h["expanded"] for h in geo["headers"]] == ["true", "true"]
        assert geo["headers"][0]["text"] == "Task Info"
        assert geo["contentShown"] is False
        assert [r["shown"] for r in geo["resizers"]] == [False, False]

        _show_task_update(page, _LONG_REPORT)
        geo = _geometry(page)
        assert geo["headers"][1]["text"] == "Task update"
        assert geo["headers"][1]["expanded"] == "true"
        # The two bodies are equally tall, whatever their content.
        assert geo["listShown"] and geo["contentShown"], geo
        assert geo["listHeight"] == pytest.approx(geo["contentHeight"], abs=1), geo
        # The report reaches the panel's padding and scrolls.
        assert geo["contentBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2), geo
        assert geo["contentScrolls"], geo
        # One separator between them, a real handle; none after the last.
        assert [r["shown"] for r in geo["resizers"]] == [True, False]
        assert geo["resizers"][0]["handle"] is True
        assert geo["resizers"][0]["cursor"] == "row-resize"
        assert geo["listTop"] < geo["resizers"][0]["top"] < geo["headers"][1]["top"]
    finally:
        page.close()


def test_drag_moves_the_boundary_persists_and_dblclick_restores(
    browser: Browser, remote_url: str
) -> None:
    page = _open_page(browser, remote_url, 1200)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _show_task_update(page, _LONG_REPORT)
        before = _geometry(page)
        _drag_separator_by(page, 150)
        after = _geometry(page)
        assert after["listHeight"] == pytest.approx(before["listHeight"] + 150, abs=2), (
            before,
            after,
        )
        assert after["contentHeight"] == pytest.approx(before["contentHeight"] - 150, abs=2), (
            before,
            after,
        )
        assert after["contentBottom"] == pytest.approx(after["panelInnerBottom"], abs=2)
        assert after["storedH"] == str(round(after["listHeight"]))
        assert _section_resizer(page).get_attribute("aria-valuenow") == after["storedH"]

        # Dragging up past the list's content makes the list scroll.
        _drag_separator_by(page, -400)
        shrunk = _geometry(page)
        assert shrunk["listHeight"] < before["listHeight"], (before, shrunk)
        assert shrunk["listScrolls"] is True, shrunk
        assert shrunk["contentBottom"] == pytest.approx(shrunk["panelInnerBottom"], abs=2)

        # The height survives a reload.
        reload_retrying_network_change(page)
        page.wait_for_selector("body.remote-desktop", state="attached")
        page.evaluate(_PREPARE_JS)
        page.evaluate(_HIDE_GLOBAL_SECTIONS_JS)
        _show_task_update(page, _LONG_REPORT)
        reloaded = _geometry(page)
        assert reloaded["listHeight"] == pytest.approx(shrunk["listHeight"], abs=2), (
            shrunk,
            reloaded,
        )

        # Double-click restores the equal share.
        _section_resizer(page).dblclick()
        restored = _geometry(page)
        assert restored["listHeight"] == pytest.approx(before["listHeight"], abs=2)
        assert restored["storedH"] is None

        # Arrow keys move the boundary by 16px steps.
        _section_resizer(page).focus()
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowDown")
        keyed = _geometry(page)
        assert keyed["listHeight"] == pytest.approx(before["listHeight"] + 32, abs=2)
        page.keyboard.press("ArrowUp")
        keyed = _geometry(page)
        assert keyed["listHeight"] == pytest.approx(before["listHeight"] + 16, abs=2)
    finally:
        page.close()


def test_collapsing_hides_the_body_and_the_other_section_fills(
    browser: Browser, remote_url: str
) -> None:
    page = _open_page(browser, remote_url, 1200)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _show_task_update(page, _LONG_REPORT)

        _toggle(page, "meta-info")
        geo = _geometry(page)
        assert geo["headers"][1]["expanded"] == "false"
        assert geo["contentShown"] is False
        assert geo["storedCollapsed"]["update"] == "1"
        # The header is still there, and the separator stays as a
        # divider but is no longer a handle.
        assert geo["headers"][1]["bottom"] > geo["headers"][1]["top"]
        assert geo["resizers"][0]["shown"] is True
        assert geo["resizers"][0]["handle"] is False
        assert geo["resizers"][0]["cursor"] != "row-resize"
        # Task Info now fills the panel: the separator and the Task
        # update header sit at the bottom.
        assert geo["headers"][1]["bottom"] == pytest.approx(geo["panelInnerBottom"], abs=12)

        _toggle(page, "meta-section-info")
        geo = _geometry(page)
        assert geo["listShown"] is False
        assert geo["contentShown"] is False
        assert [h["expanded"] for h in geo["headers"]] == ["false", "false"]
        # Two headers stacked at the top, nothing else.
        assert geo["headers"][0]["top"] < geo["headers"][1]["top"] < geo["panelTop"] + 120

        _toggle(page, "meta-section-info")
        _toggle(page, "meta-info")
        geo = _geometry(page)
        assert [h["expanded"] for h in geo["headers"]] == ["true", "true"]
        assert geo["storedCollapsed"] == {"info": None, "update": None}
        assert geo["listShown"] and geo["contentShown"]
        assert geo["resizers"][0]["handle"] is True
        assert geo["contentBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2)
    finally:
        page.close()


def test_short_window_keeps_both_headers_and_scrolls_the_bodies(
    browser: Browser, remote_url: str
) -> None:
    page = _open_page(browser, remote_url, 1200, height=320)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        page.evaluate(
            """
            () => {
              const long = 'ksen-vm-32.c.r2eg-441800.internal (x86_64)';
              for (const id of ['meta-tokens', 'meta-cost', 'meta-steps',
                                'meta-time', 'meta-machine']) {
                document.getElementById(id).textContent = long + ' ' + long;
              }
            }
            """
        )
        _show_task_update(page, _LONG_REPORT)
        geo = _geometry(page)
        # The list is taller than the window on its own; it scrolls,
        # and the Task update header still fits inside the viewport.
        assert geo["listScrolls"] is True, geo
        for hdr in geo["headers"]:
            assert 0 <= hdr["top"] < hdr["bottom"] <= 320, geo
        assert geo["resizers"][0]["shown"] is True
        # Give the report some room: dragging the separator up shrinks
        # the list and the report scrolls in the space it gains.
        _drag_separator_by(page, -100)
        geo = _geometry(page)
        assert geo["contentHeight"] > 50, geo
        assert geo["contentScrolls"] is True, geo
        assert geo["contentBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2), geo
    finally:
        page.close()


def test_mobile_drawer_has_the_same_sections_clear_of_the_close_button(
    browser: Browser, remote_url: str
) -> None:
    page = _open_page(browser, remote_url, 500)
    try:
        assert page.locator("body.remote-desktop").count() == 0
        page.locator("#meta-drawer-btn").click()
        page.wait_for_selector("#meta-panel.open", state="attached")
        # Let the drawer's slide-in transition finish before measuring.
        page.wait_for_function(
            "() => Math.abs(document.getElementById('meta-panel')"
            ".getBoundingClientRect().right - innerWidth) < 1"
        )
        _show_task_update(page, _LONG_REPORT)
        geo = _geometry(page)
        assert [h["expanded"] for h in geo["headers"]] == ["true", "true"]
        close = page.locator("#meta-close").bounding_box()
        assert close is not None
        # The first header's toggle ends before the close button starts.
        assert geo["headers"][0]["toggleRight"] <= close["x"] + 1, (geo["headers"][0], close)
        assert geo["contentScrolls"] is True
        assert geo["resizers"][0]["handle"] is True

        _toggle(page, "meta-section-info")
        geo = _geometry(page)
        assert geo["listShown"] is False
        assert geo["headers"][0]["expanded"] == "false"
        assert geo["contentBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2)
    finally:
        page.close()


_EXTRA_GEOMETRY_JS = """
() => {
  const rect = id => document.getElementById(id).getBoundingClientRect();
  const extra = document.getElementById('meta-extra-body');
  const section = document.getElementById('meta-section-extra');
  const hdr = section.querySelector('.meta-section-hdr');
  return {
    sectionShown: hdr.getClientRects().length > 0 || extra.getClientRects().length > 0,
    extraHeight: rect('meta-extra-body').height,
    extraBottom: rect('meta-extra-body').bottom,
    listHeight: rect('meta-list').height,
    contentHeight: rect('meta-info-content').height,
    valueNow: document.querySelector('#meta-section-info + .meta-section-resizer')
      .getAttribute('aria-valuenow'),
    valueMax: document.querySelector('#meta-section-info + .meta-section-resizer')
      .getAttribute('aria-valuemax'),
  };
}
"""


def test_a_third_section_stacks_resizes_and_hides(
    browser: Browser, three_sections_url: str
) -> None:
    """A panel added per the chat.html recipe takes part in the stack:
    it takes an equal share, gets its own separator, yields its share
    before a dragged neighbour does when the panel is short, and leaves
    the stack when its owner sets ``hidden``."""
    page = _open_page(browser, three_sections_url, 1200, height=700)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        geo = _geometry(page)
        assert [h["text"] for h in geo["headers"]] == ["Task Info", "Task update", "Extra"]
        assert [h["expanded"] for h in geo["headers"]] == ["true", "true", "true"]
        # No task: Task Info and Extra are shown, equally tall; Extra,
        # last, reaches the bottom.
        assert [r["shown"] for r in geo["resizers"]] == [True, False, False]
        assert geo["resizers"][0]["handle"] is True
        extra = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert extra["extraBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2), extra
        assert extra["extraHeight"] == pytest.approx(extra["listHeight"], abs=1), extra

        # With a long Task update in the middle the three bodies share
        # the panel equally.
        _show_task_update(page, _LONG_REPORT)
        geo = _geometry(page)
        assert [r["shown"] for r in geo["resizers"]] == [True, True, False]
        assert [r["handle"] for r in geo["resizers"]] == [True, True, False]
        before = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert before["contentHeight"] == pytest.approx(before["listHeight"], abs=1), before
        assert before["extraHeight"] == pytest.approx(before["listHeight"], abs=1), before

        # Dragging the Task Info separator down grows the list by the
        # full 100px, taken equally from the two bodies below; the
        # requested height is the rendered height, and aria-valuenow
        # reports it.
        _drag_separator_by(page, 100)
        after = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert after["listHeight"] == pytest.approx(before["listHeight"] + 100, abs=2), (
            before,
            after,
        )
        assert after["contentHeight"] == pytest.approx(before["contentHeight"] - 50, abs=2)
        assert after["extraHeight"] == pytest.approx(before["extraHeight"] - 50, abs=2)
        assert after["valueNow"] == str(round(after["listHeight"]))

        # A drag further than the space below can give stops where the
        # space ends: every body below keeps the minimum height
        # .meta-section-body has in main.css (the same for both, a
        # couple of rows), so no expanded section is left as a header
        # over an empty body; the stored value and aria-valuemax match
        # what shows.
        _drag_separator_by(page, 2000)
        capped = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert capped["valueNow"] == str(round(capped["listHeight"])), capped
        assert capped["valueMax"] == capped["valueNow"], capped
        assert capped["contentHeight"] > 40, capped
        assert capped["extraHeight"] == pytest.approx(capped["contentHeight"], abs=1), capped
        for hdr in _geometry(page)["headers"]:
            assert 0 <= hdr["top"] < hdr["bottom"] <= 700, hdr

        # The owner hides its section with `hidden` and re-applies the
        # layout (a toggle round trip does that here): the section is
        # gone, Task update is last again and fills, its separator hides.
        _section_resizer(page).dblclick()
        page.evaluate("() => { document.getElementById('meta-section-extra').hidden = true; }")
        _toggle(page, "meta-section-info")
        _toggle(page, "meta-section-info")
        geo = _geometry(page)
        extra = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert extra["sectionShown"] is False, extra
        assert [r["shown"] for r in geo["resizers"]] == [True, False, False]
        assert geo["contentBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2), geo
    finally:
        page.close()


def test_long_error_status_scrolls_instead_of_pushing_the_report_out(
    browser: Browser, remote_url: str
) -> None:
    """A long agent error fills the Task update status line; on a short
    panel the status is capped and scrolls, so nothing is pushed out of
    the panel and collapsing Task Info gives the report body its room."""
    page = _open_page(browser, remote_url, 1200, height=250)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, {"type": "configData", "config": {}, "apiKeys": {}})
        poll = _start_running_task(page)
        _deliver(
            page,
            {
                "type": "taskUpdate",
                "tabId": poll["tabId"],
                "token": poll["token"],
                "taskId": "task-1",
                "exists": True,
                "sig": "sig-1",
                "content": _LONG_REPORT,
                "error": "task-update agent failed: " + "x" * 800,
                "running": False,
                "cost": 0,
                "updatedAt": 0,
            },
        )
        page.wait_for_selector("#meta-info.visible", state="attached")
        status_js = """
            () => {
              const status = document.getElementById('meta-info-status');
              const content = document.getElementById('meta-info-content');
              const panel = document.getElementById('meta-panel').getBoundingClientRect();
              return {
                statusScrolls: status.scrollHeight > status.clientHeight + 1,
                statusBottom: status.getBoundingClientRect().bottom,
                contentHeight: content.getBoundingClientRect().height,
                contentBottom: content.getBoundingClientRect().bottom,
                panelBottom: panel.bottom,
              };
            }
        """
        state = page.evaluate(status_js)
        assert state["statusScrolls"] is True, state
        assert state["statusBottom"] < state["panelBottom"], state
        assert state["contentBottom"] <= state["panelBottom"], state
        # 250px is too short for the list AND the report; collapsing
        # Task Info hands the report the room the list held.
        _toggle(page, "meta-section-info")
        state = page.evaluate(status_js)
        assert state["contentHeight"] > 40, state
        assert state["contentBottom"] <= state["panelBottom"], state
    finally:
        page.close()


# Forty apps (one connected, one whose check failed): more rows than
# the panel can show, so the Apps body must scroll.
_APPS = [
    {"name": f"app{i:02d}", "label": f"App {i:02d}", "authenticated": i == 7, "error": ""}
    for i in range(38)
] + [
    {"name": "slack", "label": "Slack", "authenticated": False, "error": ""},
    {"name": "matrix", "label": "Matrix", "authenticated": None, "error": "TimeoutError: slow"},
]


def _day_key(days_ago: int) -> str:
    """The local calendar day *days_ago* days back, as the reply keys it."""
    return (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()


def _spend_report() -> dict[str, Any]:
    """A ``spendReport`` reply: a cheap day 300 days back and today's dear one."""
    old, today = _day_key(300), _day_key(0)
    return {
        "type": "spendReport",
        "total": {"cost": 12.1, "tokens": 300000, "tasks": 3},
        "days": [
            {"date": old, "cost": 0.1, "tokens": 1000, "tasks": 1},
            {"date": today, "cost": 12.0, "tokens": 299000, "tasks": 2},
        ],
        "totalByModel": [
            {"model": "claude-fable-5-1", "cost": 9.0, "tokens": 200000, "tasks": 2},
            {"model": "gpt-6-astra", "cost": 3.1, "tokens": 100000, "tasks": 1},
        ],
        "daysByModel": {
            old: [{"model": "gpt-6-astra", "cost": 0.1, "tokens": 1000, "tasks": 1}],
            today: [
                {"model": "claude-fable-5-1", "cost": 9.0, "tokens": 200000, "tasks": 2},
                {"model": "gpt-6-astra", "cost": 3.0, "tokens": 99000, "tasks": 1},
            ],
        },
    }


_SPEND_GEOMETRY_JS = """
() => {
  const graph = document.getElementById('meta-spend-graph');
  const viewport = graph.querySelector('.spend-viewport');
  const cols = Array.from(graph.querySelectorAll('.spend-col'));
  const cell = graph.querySelector('.spend-cell[data-spend-date]');
  const cellRect = cell.getBoundingClientRect();
  const vp = viewport.getBoundingClientRect();
  const nav = dir => {
    const b = graph.querySelector('.spend-nav.' + dir);
    return {hidden: b.classList.contains('nav-hidden'),
            opacity: getComputedStyle(b).opacity};
  };
  const bars = Array.from(graph.querySelectorAll('.spend-model-row')).map(r => ({
    model: r.dataset.spendModel,
    bar: r.querySelector('.spend-model-bar').getBoundingClientRect().width,
    track: r.querySelector('.spend-model-track').getBoundingClientRect().width,
    level: r.querySelector('.spend-model-bar').className,
  }));
  const dated = graph.querySelectorAll('.spend-cell[data-spend-date]');
  return {
    graphWidth: graph.clientWidth,
    cellWidth: cellRect.width,
    cellHeight: cellRect.height,
    columns: cols.length,
    columnWidth: cols[0].getBoundingClientRect().width,
    gap: cols[1].getBoundingClientRect().left - cols[0].getBoundingClientRect().right,
    days: dated.length,
    firstDay: dated[0].dataset.spendDate,
    lastDay: dated[dated.length - 1].dataset.spendDate,
    scrollLeft: viewport.scrollLeft,
    scrollMax: viewport.scrollWidth - viewport.clientWidth,
    lastCellRight: dated[dated.length - 1].getBoundingClientRect().right,
    firstCellRight: dated[0].getBoundingClientRect().right,
    viewportLeft: vp.left,
    viewportRight: vp.right,
    total: graph.querySelector('.spend-total').textContent,
    left: nav('left'),
    right: nav('right'),
    bars,
  };
}
"""


def test_spend_heatmap_fills_the_panel_pinned_to_the_latest_week(
    browser: Browser, remote_url: str
) -> None:
    """The Spend heatmap lays its week columns out to the panel's width,
    reaches back to the oldest day, scrolls with its pagers and tells
    the hovered day's cost."""
    page = _open_page(browser, remote_url, 1200, height=900, global_sections=True)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, {"type": "cronJobs", "jobs": []})
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": 1})
        report = _spend_report()
        _deliver(page, report)
        geo = page.evaluate(_SPEND_GEOMETRY_JS)
        assert geo["total"] == "All time \u00b7 $12.10 \u00b7 300K tok \u00b7 3 tasks"
        assert (geo["cellWidth"], geo["cellHeight"]) == (11, 11), geo
        assert geo["gap"] == pytest.approx(3, abs=0.5), geo
        # 300 days back to the oldest day: more week columns than fit
        # the panel, each column a Monday-first week.
        assert geo["days"] == 301, geo
        assert geo["firstDay"] == _day_key(300) and geo["lastDay"] == _day_key(0), geo
        first = datetime.date.fromisoformat(geo["firstDay"])
        assert geo["columns"] == math.ceil((first.weekday() + 301) / 7), geo
        fits = (geo["graphWidth"] + 3) // 14
        assert geo["columns"] > fits > 10, geo
        # Pinned to the right: the latest week hugs the viewport's right
        # edge and the oldest one is scrolled out of view to the left.
        assert geo["scrollLeft"] == pytest.approx(geo["scrollMax"], abs=1), geo
        assert geo["lastCellRight"] == pytest.approx(geo["viewportRight"], abs=1), geo
        assert geo["firstCellRight"] < geo["viewportLeft"], geo
        assert not geo["left"]["hidden"] and geo["right"]["hidden"], geo
        # The pagers are invisible until the graph is hovered.
        assert geo["left"]["opacity"] == "0", geo
        page.hover("#meta-spend-graph .spend-total")
        page.wait_for_function(
            "() => getComputedStyle(document.querySelector('.spend-nav.left')).opacity > 0.5"
        )
        # Shading: today's $12 is the dearest day (l4), the 10-cent day l1.
        levels = page.evaluate(
            "keys => keys.map(k => document.querySelector("
            "`.spend-cell[data-spend-date=\"${k}\"]`).className)",
            [_day_key(0), _day_key(300), _day_key(1)],
        )
        assert levels == ["spend-cell l4", "spend-cell l1", "spend-cell"], levels
        # Model bars: the dearest model's bar is its share of the track.
        assert [b["model"] for b in geo["bars"]] == ["claude-fable-5-1", "gpt-6-astra"]
        assert geo["bars"][0]["bar"] == pytest.approx(
            geo["bars"][0]["track"] * 9.0 / 12.1, abs=1
        ), geo
        assert geo["bars"][0]["level"] == "spend-model-bar l4", geo
        assert geo["bars"][1]["level"] == "spend-model-bar l2", geo

        # Hovering today's cell shows the tooltip right above it.
        today_cell = page.locator(f'.spend-cell[data-spend-date="{_day_key(0)}"]')
        today_cell.hover()
        tip = page.locator("#meta-spend-graph .spend-tip")
        assert tip.is_visible()
        text = tip.inner_text()
        assert text.splitlines() == [
            f"{datetime.date.today():%b} {datetime.date.today().day} · $12.00 · 299K tok · 2 tasks",
            "claude-fable-5-1 · $9.00 · 200K tok · 2 tasks · 75%",
            "gpt-6-astra · $3.00 · 99.0K tok · 1 task · 25%",
        ], text
        tip_box = tip.bounding_box()
        cell_box = today_cell.bounding_box()
        graph_box = page.locator("#meta-spend-graph").bounding_box()
        assert tip_box is not None and cell_box is not None and graph_box is not None
        # Above the cell, or below it when the cell is in the top rows;
        # never over it, and never wider than the graph.
        assert (
            tip_box["y"] + tip_box["height"] <= cell_box["y"]
            or tip_box["y"] >= cell_box["y"] + cell_box["height"]
        ), (tip_box, cell_box)
        assert tip_box["x"] >= graph_box["x"] - 1, (tip_box, graph_box)
        assert tip_box["x"] + tip_box["width"] <= graph_box["x"] + graph_box["width"] + 1, (
            tip_box,
            graph_box,
        )
        page.hover("#meta-spend-graph .spend-total")
        assert not tip.is_visible()
        # Hovering cells at the right edge, then the left edge, then the
        # bottom row: the tooltip keeps one width per text (it is
        # measured at the left edge, not where the last hover left it)
        # and stays inside the graph both ways, whichever row the cell
        # is in.
        widths: dict[str, set[float]] = {}
        for days_ago in (0, 1, 2, 3, 6, 5, 4, 200, 0):
            cell = page.locator(f'.spend-cell[data-spend-date="{_day_key(days_ago)}"]')
            # The oldest cell is scrolled out of the viewport: bring it in.
            cell.scroll_into_view_if_needed()
            cell.hover()
            box = tip.bounding_box()
            cell_box = cell.bounding_box()
            assert box is not None and cell_box is not None
            widths.setdefault(tip.inner_text(), set()).add(round(box["width"], 1))
            assert box["x"] >= graph_box["x"] - 1, (days_ago, box, graph_box)
            assert box["x"] + box["width"] <= graph_box["x"] + graph_box["width"] + 1, (
                days_ago, box, graph_box,
            )
            assert box["y"] >= graph_box["y"] - 1, (days_ago, box, graph_box)
            assert box["y"] + box["height"] <= graph_box["y"] + graph_box["height"] + 1, (
                days_ago, box, graph_box,
            )
            assert (
                box["y"] + box["height"] <= cell_box["y"]
                or box["y"] >= cell_box["y"] + cell_box["height"]
            ), (days_ago, box, cell_box)
        # One width per distinct text, not per position: today's cell is
        # hovered first after a right-edge tooltip and again after a
        # left-edge one, and must measure the same both times.  The seven
        # empty days each read "<date> · no usage", and in a proportional
        # font every date has a width of its own (Sep 21 is narrower than
        # Sep 26), so distinct texts are not expected to share widths.
        assert len(widths) == 8, widths
        assert all(len(seen) == 1 for seen in widths.values()), widths
        assert page.evaluate(
            "() => { const b = document.getElementById('meta-spend-body');"
            " return b.scrollWidth <= b.clientWidth; }"
        ), "no horizontal overflow"
        page.hover("#meta-spend-graph .spend-total")
        # The pagers sit over the heatmap itself, not over the model bars.
        heatmap_box = page.locator("#meta-spend-graph .spend-heatmap").bounding_box()
        nav_box = page.locator(".spend-nav.left").bounding_box()
        assert heatmap_box is not None and nav_box is not None
        assert nav_box["y"] >= heatmap_box["y"], (nav_box, heatmap_box)
        assert nav_box["y"] + nav_box["height"] <= heatmap_box["y"] + heatmap_box["height"], (
            nav_box,
            heatmap_box,
        )

        # The left pager scrolls back towards the oldest week; enough
        # clicks reach it, hiding the pager.
        page.evaluate(
            "() => { const v = document.querySelector('.spend-viewport');"
            " v.scrollLeft = v.scrollWidth; }"
        )
        before = page.evaluate("() => document.querySelector('.spend-viewport').scrollLeft")
        page.locator(".spend-nav.left").click()
        after = page.evaluate("() => document.querySelector('.spend-viewport').scrollLeft")
        assert after < before, (before, after)
        for _ in range(30):
            if page.evaluate("() => document.querySelector('.spend-viewport').scrollLeft") == 0:
                break
            page.locator(".spend-nav.left").click()
        assert page.evaluate("() => document.querySelector('.spend-viewport').scrollLeft") == 0
        page.hover("#meta-spend-graph .spend-total")
        assert page.evaluate(
            "() => document.querySelector('.spend-nav.left').classList.contains('nav-hidden')"
        )
        assert not page.evaluate(
            "() => document.querySelector('.spend-nav.right').classList.contains('nav-hidden')"
        )
        # A fresh reply redraws where the user left off (the oldest week).
        _deliver(page, report)
        assert page.evaluate("() => document.querySelector('.spend-viewport').scrollLeft") == 0
    finally:
        page.close()


_GLOBAL_GEOMETRY_JS = """
() => {
  const rect = id => document.getElementById(id).getBoundingClientRect();
  const panel = document.getElementById('meta-panel');
  const pad = parseFloat(getComputedStyle(panel).paddingBottom);
  const apps = document.getElementById('meta-apps-list');
  return {
    panelInnerBottom: panel.getBoundingClientRect().bottom - pad,
    scheduleTop: rect('meta-schedule-list').top,
    scheduleHeight: rect('meta-schedule-list').height,
    appsTop: rect('meta-apps-list').top,
    appsBottom: rect('meta-apps-list').bottom,
    appsHeight: rect('meta-apps-list').height,
    spendBottom: rect('meta-spend-body').bottom,
    spendHeight: rect('meta-spend-body').height,
    listHeight: rect('meta-list').height,
    appsScrolls: apps.scrollHeight > apps.clientHeight + 1,
    headers: Array.from(document.querySelectorAll('#meta-panel .meta-section-hdr'))
      .filter(h => h.getClientRects().length > 0)
      .map(h => h.querySelector('.meta-section-toggle').textContent),
    appsStatus: document.getElementById('meta-apps-status').textContent,
    firstApp: apps.querySelector('.app-row .sidebar-panel-name').textContent,
    buttons: apps.querySelectorAll('button.app-row-main').length,
  };
}
"""


def test_schedule_and_apps_sections_fill_scroll_and_launch_a_connect_task(
    browser: Browser, remote_url: str
) -> None:
    page = _open_page(browser, remote_url, 1200, height=700, global_sections=True)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        posted = page.evaluate("() => window.__posted.map(m => m.type)")
        assert "getCronJobs" in posted and "getAppsStatus" in posted
        assert "getSpendReport" in posted
        _deliver(page, {
            "type": "cronJobs",
            "jobs": [{
                "id": "j1", "name": "Morning email digest", "schedule": "0 9 * * *",
                "kind": "prompt", "what": "summarize my email", "enabled": True,
                "running": False, "nextRunAt": "2099-01-01T09:00:00", "lastRunAt": "",
                "lastStatus": "", "workDir": "",
            }],
        })
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": "2026-09-26T11:00:00"})
        _deliver(page, _spend_report())
        geo = page.evaluate(_GLOBAL_GEOMETRY_JS)
        # No running task: Task Info, Schedule, Apps and Spend are on screen.
        assert geo["headers"] == ["Task Info", "Schedule", "Apps", "Spend"], geo
        # The four bodies are equally tall; Spend reaches the bottom
        # and Apps scrolls.
        assert geo["scheduleTop"] < geo["appsTop"], geo
        assert geo["scheduleHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert geo["listHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert geo["spendHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert geo["spendBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2), geo
        assert geo["appsScrolls"], geo
        assert geo["appsStatus"] == "1 of 40 connected"
        assert geo["firstApp"] == "App 07", "connected apps are listed first"
        assert geo["buttons"] == 39, "every app that is not connected is a button"

        # The static server never completes the websocket handshake, so
        # the page believes the daemon is down (sends are held back).
        # Connecting re-requests the three subpanels.
        page.evaluate("() => { window.__posted.length = 0; }")
        _deliver(page, {"type": "daemonStatus", "connected": True})
        posted = page.evaluate("() => window.__posted.map(m => m.type)")
        assert "getCronJobs" in posted and "getAppsStatus" in posted
        assert "getSpendReport" in posted

        # Clicking an app that is not connected submits a connect task
        # in a NEW chat, which takes the screen; the idle chat left
        # behind (no task, no draft) is retired, so the open-tab records
        # end up with the new root chat in place of the old one.
        tabs_before = page.evaluate("() => window._testApi.openTabs()")
        chat_before = page.evaluate("() => window._testApi.getActiveTabId()")
        assert [t["id"] for t in tabs_before] == [chat_before], tabs_before
        # The shim keeps retrying the websocket and reports the daemon
        # down again after every failed attempt (sendMessage then holds
        # the prompt back), so the "connected" report and the click run
        # in one JS turn where no retry can interleave.
        submit = page.evaluate(
            """() => {
              window.dispatchEvent(new MessageEvent('message', {
                data: {type: 'daemonStatus', connected: true},
              }));
              document.querySelector('.app-row[data-app="slack"] button').click();
              return window.__posted.filter(m => m.type === 'submit').pop() || null;
            }"""
        )
        assert submit is not None
        assert submit["prompt"].startswith('Connect my Slack app: authenticate the "slack"')
        assert 'run_agent with agent "slack"' in submit["prompt"]
        chat_after = page.evaluate("() => window._testApi.getActiveTabId()")
        assert chat_after != chat_before, "the connect task runs in a new chat"
        tabs_after = page.evaluate("() => window._testApi.openTabs()")
        assert [t["id"] for t in tabs_after] == [chat_after], (
            f"the new root chat replaces the retired idle one: {tabs_after}"
        )
        assert not tabs_after[0]["isSubagentTab"] and not tabs_after[0]["isContentTab"]
    finally:
        page.close()


def test_a_long_task_update_leaves_the_apps_list_a_usable_share(
    browser: Browser, remote_url: str
) -> None:
    """A long Task update above the Apps list takes only its equal
    share and scrolls instead of squeezing the Apps list."""
    page = _open_page(browser, remote_url, 1200, height=700, global_sections=True)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, {"type": "cronJobs", "jobs": []})
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": 1})
        _show_task_update(page, _LONG_REPORT)
        geo = page.evaluate(
            """() => {
              const r = id => document.getElementById(id).getBoundingClientRect();
              const c = document.getElementById('meta-info-content');
              return {apps: r('meta-apps-list').height,
                      update: r('meta-info-content').height,
                      panel: r('meta-panel').height,
                      updateScrolls: c.scrollHeight > c.clientHeight + 1};
            }"""
        )
        assert geo["apps"] >= 80, geo
        assert geo["update"] == pytest.approx(geo["apps"], abs=1), geo
        assert geo["updateScrolls"], geo
    finally:
        page.close()


# Each surface that shows the task-info panel, as (viewport width, JS
# run after boot).  The remote page boots as the desktop dock (wide) or
# the mobile drawer (narrow, opened from its footer button); the VS
# Code surfaces are the same markup under the extension's body classes
# (remote-codex.css only styles body.remote-chat, so dropping it leaves
# main.css's own rules for that mode).
_SURFACES = {
    "remote desktop": (1200, ""),
    "remote mobile drawer": (500, "document.getElementById('meta-drawer-btn').click();"),
    "VS Code Task Info view": (1200, "document.body.className = 'meta-panel-mode';"),
    "VS Code sidebar-chat drawer": (
        1200,
        "document.body.className = 'sidebar-chat-mode';"
        "document.getElementById('meta-panel').classList.add('open');",
    ),
}

_BODY_HEIGHTS_JS = """
() => ['meta-list', 'meta-info-content', 'meta-schedule-list', 'meta-apps-list',
       'meta-spend-body']
  .map(id => document.getElementById(id).getBoundingClientRect().height)
"""


@pytest.mark.parametrize("surface", list(_SURFACES))
def test_every_surface_gives_the_expanded_sections_equal_heights(
    browser: Browser, remote_url: str, surface: str
) -> None:
    """All five sections expanded: their bodies are equally tall by
    default on every surface, whatever their content.  Dragging the
    Schedule / Apps boundary moves only that boundary: the bodies above
    it keep their heights, the two below absorb the loss between them
    (Spend, last, keeps at least its minimum share)."""
    width, setup = _SURFACES[surface]
    page = _open_page(browser, remote_url, width, height=900, global_sections=True)
    try:
        page.evaluate(f"() => {{ {setup} }}")
        # Let a drawer's slide-in transition finish before measuring.
        page.wait_for_function(
            "() => { const r = document.getElementById('meta-panel').getBoundingClientRect();"
            " return r.width > 0 && Math.abs(r.right - innerWidth) < 1; }"
        )
        _deliver(page, {"type": "cronJobs", "jobs": []})
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": 1})
        _deliver(page, _spend_report())
        # The VS Code surfaces get their Task update relayed from the
        # editor chat, so show the section directly (as setMetaInfoHTML
        # does) and re-apply the layout with a toggle round trip.
        page.evaluate(
            """html => {
              document.getElementById('meta-info-content').innerHTML = html;
              document.getElementById('meta-info').classList.add('visible');
              const toggle = document.querySelector('#meta-section-info .meta-section-toggle');
              toggle.click();
              toggle.click();
            }""",
            _LONG_REPORT,
        )
        heights = page.evaluate(_BODY_HEIGHTS_JS)
        assert min(heights) > 40, heights
        assert max(heights) - min(heights) <= 1, heights

        resizer = page.locator("#meta-schedule + .meta-section-resizer")
        box = resizer.bounding_box()
        assert box is not None
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y + 30, steps=6)
        page.mouse.up()
        dragged = page.evaluate(_BODY_HEIGHTS_JS)
        assert dragged[0] == pytest.approx(heights[0], abs=1), (heights, dragged)
        assert dragged[1] == pytest.approx(heights[1], abs=1), (heights, dragged)
        assert dragged[2] == pytest.approx(heights[2] + 30, abs=2), (heights, dragged)
        assert dragged[3] < heights[3] and dragged[4] < heights[4], (heights, dragged)
        assert dragged[3] + dragged[4] == pytest.approx(
            heights[3] + heights[4] - 30, abs=2
        ), (heights, dragged)
    finally:
        page.close()


_MINIMUM_GEOMETRY_JS = """
() => {
  const panel = document.getElementById('meta-panel');
  const rect = id => document.getElementById(id).getBoundingClientRect();
  const list = document.getElementById('meta-schedule-list');
  const row = list.querySelector('li');
  const share = parseFloat(getComputedStyle(panel).getPropertyValue('--meta-body-share'));
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
  return {
    share,
    minimum: Math.min(4 * rem, share / 2),
    listHeight: rect('meta-list').height,
    scheduleHeight: rect('meta-schedule-list').height,
    appsHeight: rect('meta-apps-list').height,
    spendHeight: rect('meta-spend-body').height,
    scheduleScrolls: list.scrollHeight > list.clientHeight + 1,
    firstRowInside: row !== null
      && row.getBoundingClientRect().top >= list.getBoundingClientRect().top - 0.5
      && row.getBoundingClientRect().bottom <= list.getBoundingClientRect().bottom + 0.5,
    panelOverflows: panel.scrollHeight > panel.clientHeight + 1,
    headerBottoms: Array.from(document.querySelectorAll('#meta-panel .meta-section-hdr'))
      .map(h => h.getBoundingClientRect().bottom),
  };
}
"""


def _cron_jobs(count: int) -> dict[str, object]:
    return {
        "type": "cronJobs",
        "jobs": [
            {
                "id": f"j{i}", "name": f"Job {i}", "schedule": "0 9 * * *",
                "kind": "prompt", "what": f"task {i}", "enabled": True,
                "running": False, "nextRunAt": "2099-01-01T09:00:00", "lastRunAt": "",
                "lastStatus": "", "workDir": "",
            }
            for i in range(count)
        ],
    }


def test_a_persisted_drag_cannot_hide_the_schedule_section(
    browser: Browser, remote_url: str
) -> None:
    """The report behind this test: the Schedule header stood over an
    empty body on every page load and "showed no scheduled tasks".  The
    Task Info separator had once been dragged far down; that height
    persists, and a middle body used to have no minimum, so the dragged
    body took its whole share.  Every expanded body now keeps half of an
    equal share (at most 4rem): the jobs stay on screen and scroll, the
    dragged body yields instead, and because the minimum follows the
    panel's height a short window never makes the panel itself
    overflow.  A double-click on the separator still restores the equal
    shares."""
    page = _open_page(
        browser,
        remote_url,
        1200,
        height=800,
        global_sections=True,
        storage={"kiss-meta-section-h:meta-section-info": "2000"},
    )
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, _cron_jobs(12))
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": "2026-09-26T11:00:00"})
        _deliver(page, _spend_report())
        geo = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert geo["share"] > 100, geo
        assert geo["minimum"] > 40, geo
        # Schedule, Apps and Spend each keep the minimum; Task Info has
        # the rest, far less than the 2000px it asked for.
        for key in ("scheduleHeight", "appsHeight", "spendHeight"):
            assert geo[key] == pytest.approx(geo["minimum"], abs=1), (key, geo)
        assert geo["listHeight"] < 4 * geo["share"] - 3 * geo["minimum"] + 2, geo
        assert geo["firstRowInside"] is True and geo["scheduleScrolls"] is True, geo
        assert geo["panelOverflows"] is False, geo

        # Measuring the share touches nothing: a scrolled body keeps its
        # position through the sidebar's periodic refresh.
        page.evaluate(
            "() => { const l = document.getElementById('meta-apps-list');"
            " l.scrollTop = l.scrollHeight; }"
        )
        scrolled = page.evaluate("() => document.getElementById('meta-apps-list').scrollTop")
        assert scrolled > 0
        _deliver(page, _cron_jobs(12))
        assert page.evaluate(
            "() => document.getElementById('meta-apps-list').scrollTop"
        ) == scrolled

        # A much shorter window: the share and the minimums shrink with
        # it (ResizeObserver), every header stays on screen, the bodies
        # still show something, and the panel still does not overflow.
        page.set_viewport_size({"width": 1200, "height": 300})
        page.wait_for_function(
            "() => parseFloat(getComputedStyle(document.getElementById('meta-panel'))"
            ".getPropertyValue('--meta-body-share')) < 60"
        )
        short = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert 0 < short["minimum"] < geo["minimum"], short
        assert short["scheduleHeight"] == pytest.approx(short["minimum"], abs=1), short
        assert short["panelOverflows"] is False, short
        assert all(bottom <= 300 for bottom in short["headerBottoms"]), short

        # A Task update whose status line then grows (a long error) adds
        # a fifth section and a taller header row on this short panel:
        # the share is measured after the status is set, so the
        # minimums shrink to fit and the panel still does not overflow.
        _show_task_update(page, _LONG_REPORT)
        poll = page.evaluate(
            "() => window.__posted.filter(m => m.type === 'getTaskUpdate').pop()"
        )
        _deliver(
            page,
            {
                "type": "taskUpdate",
                "tabId": poll["tabId"],
                "token": poll["token"],
                "taskId": "task-1",
                "exists": True,
                "sig": "sig-2",
                "content": _LONG_REPORT,
                "error": "task-update agent failed: " + "x" * 800,
                "running": False,
                "cost": 0,
                "updatedAt": 0,
            },
        )
        crowded = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert len(crowded["headerBottoms"]) == 5, crowded
        assert 0 < crowded["minimum"] < short["minimum"], (short, crowded)
        assert crowded["panelOverflows"] is False, crowded
        assert all(bottom <= 300 for bottom in crowded["headerBottoms"]), crowded

        # Double-clicking the separator forgets the dragged height:
        # the four bodies share the panel equally again.
        page.set_viewport_size({"width": 1200, "height": 800})
        page.wait_for_function(
            "() => parseFloat(getComputedStyle(document.getElementById('meta-panel'))"
            ".getPropertyValue('--meta-body-share')) > 100"
        )
        _section_resizer(page).dblclick()
        reset = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert reset["scheduleHeight"] == pytest.approx(reset["share"], abs=1), reset
        assert reset["listHeight"] == pytest.approx(reset["share"], abs=1), reset
        assert page.evaluate(
            "() => localStorage.getItem('kiss-meta-section-h:meta-section-info')"
        ) is None

        # The dragged body has the same minimum: Arrow Up past it leaves
        # the body at the minimum, and the stored height and
        # aria-valuenow say what shows; Arrow Down then moves it again.
        resizer = _section_resizer(page)
        resizer.focus()
        for _ in range(30):
            page.keyboard.press("ArrowUp")
        floor = page.evaluate(_MINIMUM_GEOMETRY_JS)
        stored = page.evaluate(
            "() => localStorage.getItem('kiss-meta-section-h:meta-section-info')"
        )
        assert floor["listHeight"] == pytest.approx(floor["minimum"], abs=1), floor
        assert stored == str(round(floor["listHeight"])), (stored, floor)
        assert resizer.get_attribute("aria-valuenow") == stored
        page.keyboard.press("ArrowDown")
        assert page.evaluate(_MINIMUM_GEOMETRY_JS)["listHeight"] == pytest.approx(
            floor["listHeight"] + 16, abs=1
        )

        # A pointer drag to the bottom of the window (past the cap) stops
        # where the bodies below reach their minimum; moving back to 50px
        # below the start in the same drag leaves the body 50px taller
        # than it began (both moves stay inside the viewport so the
        # pointer events reach the page).
        start = page.evaluate(_MINIMUM_GEOMETRY_JS)["listHeight"]
        box = resizer.bounding_box()
        assert box is not None
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, 790, steps=8)
        capped = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert capped["scheduleHeight"] == pytest.approx(capped["minimum"], abs=1), capped
        assert start + 50 < capped["listHeight"] < 790 - y - 50, (start, y, capped)
        assert resizer.get_attribute("aria-valuemax") == str(round(capped["listHeight"]))
        page.mouse.move(x, y + 50, steps=4)
        page.mouse.up()
        back = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert back["listHeight"] == pytest.approx(start + 50, abs=2), (start, capped, back)
    finally:
        page.close()

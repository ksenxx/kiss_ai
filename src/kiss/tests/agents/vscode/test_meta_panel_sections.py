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

* both sections start expanded; the list keeps its natural height
  and the Task update body fills the rest of the panel and scrolls;
* dragging the separator down grows the list and shrinks the report by
  the same amount, the height persists across a reload, and a
  double-click restores the natural height; the arrow keys resize too;
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

import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Locator, Page, sync_playwright

from kiss.server.web_server import MEDIA_DIR, _build_html

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
  // The per-task sections only: the global Schedule and Apps sections
  // (hidden by _open_page unless asked for) are measured separately.
  const isGlobal = el => el && (el.id === 'meta-schedule' || el.id === 'meta-apps');
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


# Takes the global Schedule and Apps sections out of the stack (the
# `hidden` attribute; a Task Info toggle round trip re-applies the
# layout), leaving the per-task sections these geometry tests measure.
_HIDE_GLOBAL_SECTIONS_JS = """
() => {
  document.getElementById('meta-schedule').hidden = true;
  document.getElementById('meta-apps').hidden = true;
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
) -> Page:
    """Open the remote page at the given viewport with post recording.

    Unless ``global_sections``, the Schedule and Apps sections are
    hidden so only the per-task sections share the panel."""
    page = browser.new_page(viewport={"width": width, "height": height})
    page.add_init_script(_RECORD_POSTS_JS)
    page.goto(url)
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


def _show_task_update(page: Page, html: str) -> None:
    """Run a task and answer its getTaskUpdate poll with ``html``."""
    _deliver(page, {"type": "configData", "config": {}, "apiKeys": {}})
    _deliver(page, {"type": "status", "running": True})
    poll = page.evaluate(
        "() => window.__posted.filter(m => m.type === 'getTaskUpdate').pop() || null"
    )
    assert poll is not None, "a running task must start the getTaskUpdate poll"
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


def test_both_sections_expanded_list_natural_report_fills_and_scrolls(
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
        # The list keeps its natural height: nothing to scroll.
        assert geo["listShown"] and not geo["listScrolls"], geo
        # The report fills the panel down to its padding and scrolls.
        assert geo["contentShown"], geo
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
        page.reload()
        page.wait_for_selector("body.remote-desktop", state="attached")
        page.evaluate(_PREPARE_JS)
        page.evaluate(_HIDE_GLOBAL_SECTIONS_JS)
        _show_task_update(page, _LONG_REPORT)
        reloaded = _geometry(page)
        assert reloaded["listHeight"] == pytest.approx(shrunk["listHeight"], abs=2), (
            shrunk,
            reloaded,
        )

        # Double-click restores the natural height.
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
  };
}
"""


def test_a_third_section_stacks_resizes_and_hides(
    browser: Browser, three_sections_url: str
) -> None:
    """A panel added per the chat.html recipe takes part in the stack:
    it fills while last, gets its own separator, yields its natural
    height before a dragged neighbour does when the panel is short, and
    leaves the stack when its owner sets ``hidden``."""
    page = _open_page(browser, three_sections_url, 1200, height=560)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        geo = _geometry(page)
        assert [h["text"] for h in geo["headers"]] == ["Task Info", "Task update", "Extra"]
        assert [h["expanded"] for h in geo["headers"]] == ["true", "true", "true"]
        # No task: Task Info and Extra are shown; Extra, last, fills.
        assert [r["shown"] for r in geo["resizers"]] == [True, False, False]
        assert geo["resizers"][0]["handle"] is True
        extra = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert extra["extraBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2), extra

        # With a long Task update in the middle the panel is too short
        # for everything: the update keeps a natural body that shrinks,
        # Extra (last) fills what is left.
        _show_task_update(page, _LONG_REPORT)
        geo = _geometry(page)
        assert [r["shown"] for r in geo["resizers"]] == [True, True, False]
        assert [r["handle"] for r in geo["resizers"]] == [True, True, False]
        before = page.evaluate(_EXTRA_GEOMETRY_JS)

        # Dragging the Task Info separator down grows the list by the
        # full 100px: the natural Task update body yields first, the
        # requested height is the rendered height, and aria-valuenow
        # reports it.
        _drag_separator_by(page, 100)
        after = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert after["listHeight"] == pytest.approx(before["listHeight"] + 100, abs=2), (
            before,
            after,
        )
        assert after["contentHeight"] == pytest.approx(before["contentHeight"] - 100, abs=2)
        assert after["valueNow"] == str(round(after["listHeight"]))

        # A drag further than the space below can give stops where the
        # space ends, and the stored value matches what shows.
        _drag_separator_by(page, 2000)
        capped = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert capped["valueNow"] == str(round(capped["listHeight"])), capped
        assert capped["contentHeight"] == pytest.approx(0, abs=1), capped
        # The filling (last expanded) body keeps its minimum share
        # (.meta-section-fill in main.css) even against a long drag.
        assert capped["extraHeight"] > 50, capped
        for hdr in _geometry(page)["headers"]:
            assert 0 <= hdr["top"] < hdr["bottom"] <= 560, hdr

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
        _deliver(page, {"type": "status", "running": True})
        poll = page.evaluate(
            "() => window.__posted.filter(m => m.type === 'getTaskUpdate').pop() || null"
        )
        assert poll is not None
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
        geo = page.evaluate(_GLOBAL_GEOMETRY_JS)
        # No running task: Task Info, Schedule and Apps are on screen.
        assert geo["headers"] == ["Task Info", "Schedule", "Apps"], geo
        # Schedule keeps its natural height; Apps fills the rest and scrolls.
        assert geo["scheduleTop"] < geo["appsTop"], geo
        assert geo["appsBottom"] == pytest.approx(geo["panelInnerBottom"], abs=2), geo
        assert geo["appsScrolls"], geo
        assert geo["appsStatus"] == "1 of 40 connected"
        assert geo["firstApp"] == "App 07", "connected apps are listed first"
        assert geo["buttons"] == 39, "every app that is not connected is a button"

        # The static server never completes the websocket handshake, so
        # the page believes the daemon is down (sends are held back).
        # Connecting re-requests both subpanels.
        page.evaluate("() => { window.__posted.length = 0; }")
        _deliver(page, {"type": "daemonStatus", "connected": True})
        posted = page.evaluate("() => window.__posted.map(m => m.type)")
        assert "getCronJobs" in posted and "getAppsStatus" in posted

        # Clicking an app that is not connected submits a connect task
        # in a NEW tab.
        tabs_before = page.locator("#tab-list .chat-tab").count()
        page.locator('.app-row[data-app="slack"] button').click()
        submit = page.evaluate(
            "() => window.__posted.filter(m => m.type === 'submit').pop() || null"
        )
        assert submit is not None
        assert submit["prompt"].startswith('Connect my Slack app: authenticate the "slack"')
        assert 'run_agent with agent "slack"' in submit["prompt"]
        assert page.locator("#tab-list .chat-tab").count() == tabs_before + 1
    finally:
        page.close()


def test_a_long_task_update_leaves_the_apps_list_a_usable_share(
    browser: Browser, remote_url: str
) -> None:
    """The filling (last expanded) body keeps a minimum share: a long
    Task update above it shrinks and scrolls instead of squeezing the
    Apps list to zero height."""
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
                      panel: r('meta-panel').height,
                      updateScrolls: c.scrollHeight > c.clientHeight + 1};
            }"""
        )
        assert geo["apps"] >= 80, geo
        assert geo["updateScrolls"], geo
    finally:
        page.close()

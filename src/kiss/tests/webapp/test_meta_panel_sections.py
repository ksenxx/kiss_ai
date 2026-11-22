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
* on every surface (remote desktop and mobile drawer) all expanded
  bodies are equally tall, and a drag leaves the bodies above the moved
  boundary alone;
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
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from kiss.agents.sorcar.task_metadata import ALL_TAGS
from kiss.server.web_server import MEDIA_DIR, _build_html
from kiss.tests.conftest import (
    goto_retrying_network_change,
    mark_version_seen,
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
# through the remote shim's acquireTransport() in window.__posted.
_RECORD_POSTS_JS = """
(() => {
  window.__posted = [];
  let real;
  Object.defineProperty(window, 'acquireTransport', {
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
  const list = document.getElementById('meta-list');
  const content = document.getElementById('meta-info-content');
  // The per-task sections only: the workspace (Explorer, Source
  // Control) and global (Schedule, Machines, Skills, Connectors,
  // Spend, Recommendations) sections (hidden by _open_page unless
  // asked for) are measured separately.
  const isGlobal = el =>
    el && ['meta-explorer', 'meta-scm', 'meta-schedule', 'meta-machines',
           'meta-skills', 'meta-apps', 'meta-spend',
           'meta-recommendations'].includes(el.id);
  const resizers = Array.from(
    document.querySelectorAll('#meta-panel > .meta-section-resizer'))
    .filter(r => !isGlobal(r.previousElementSibling));
  const hdrs = Array.from(document.querySelectorAll('#meta-panel .meta-section-hdr'))
    .filter(h => !isGlobal(h.parentElement));
  return {
    // The sections end where the strip along the panel's bottom begins.
    sectionsBottom: rect('meta-footer').top,
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
    marker = '<div id="meta-footer"'
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


# Takes the workspace (Explorer, Source Control) and global (Schedule,
# Machines, Skills, Connectors, Spend) sections out of the stack (the
# `hidden` attribute; the panel watches it and re-applies the layout),
# leaving the per-task sections these geometry tests measure: the page
# itself keeps Recommendations hidden while it has no task.
_HIDE_WORKSPACE_SECTIONS_JS = """
() => {
  document.getElementById('meta-explorer').hidden = true;
  document.getElementById('meta-scm').hidden = true;
}
"""

_HIDE_GLOBAL_SECTIONS_JS = """
() => {
  for (const id of ['meta-explorer', 'meta-scm', 'meta-schedule',
                    'meta-machines', 'meta-skills', 'meta-apps',
                    'meta-spend'])
    document.getElementById(id).hidden = true;
}
"""


def _open_page(
    browser: Browser,
    url: str,
    width: int,
    height: int = 900,
    global_sections: bool = False,
    storage: dict[str, str] | None = None,
    fresh_profile: bool = False,
    touch: bool = False,
) -> Page:
    """Open the remote page at the given viewport with post recording.

    The workspace sections (Explorer, Source Control) are always hidden:
    they browse a daemon this page has none of.  Unless
    ``global_sections``, the Schedule, Machines, Skills, Connectors and
    Spend sections are hidden too (Recommendations hides itself while
    it has no task), so only the per-task sections share the panel.
    ``storage`` entries land in localStorage before the page's scripts
    run, the way a previous visit would have left them.  The profile
    has already seen the running version unless ``fresh_profile``: the
    first load of a version folds every section that folds, which
    leaves Recommendations alone open.  ``touch`` makes the page a
    phone's: a finger and no pointer to hover with."""
    page = browser.new_page(
        viewport={"width": width, "height": height}, has_touch=touch, is_mobile=touch,
    )
    page.add_init_script(_RECORD_POSTS_JS)
    if not fresh_profile:
        mark_version_seen(page)
    if storage:
        page.add_init_script(
            f"for (const [k, v] of Object.entries({json.dumps(storage)})) "
            "localStorage.setItem(k, v);"
        )
    goto_retrying_network_change(page, url)
    page.wait_for_selector("body.remote-chat", state="attached")
    page.evaluate(_PREPARE_JS)
    page.evaluate(
        _HIDE_GLOBAL_SECTIONS_JS if not global_sections else _HIDE_WORKSPACE_SECTIONS_JS
    )
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


def _search_skills(page: Page) -> None:
    """Search the registry from the Skills section and answer the search.

    The page shows a ``skillSearch`` only when it echoes the
    ``generation`` of the search it awaits, so the search is made the
    way a user makes it and the reply carries the stamp of the posted
    ``searchSkills``.  The connected report, the click and the reply
    are dispatched in ONE evaluate, for the reason
    ``_start_running_task`` gives: a search is neither sent nor awaited
    while the daemon is down."""
    asked = page.evaluate(
        """reply => {
          const deliver = data =>
            window.dispatchEvent(new MessageEvent('message', {data}));
          deliver({type: 'daemonStatus', connected: true});
          document.getElementById('meta-skills-query').value = reply.query;
          document.getElementById('meta-skills-search-btn').click();
          const asked = window.__posted.filter(m => m.type === 'searchSkills').pop() || null;
          if (asked) deliver({...reply, generation: asked.generation});
          return asked;
        }""",
        _SKILL_SEARCH,
    )
    assert asked is not None and asked["query"] == _SKILL_SEARCH["query"], asked


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


# The ids of the sections that are open, what the pinned Recommendations
# section shows, where its body lies between the separator over it
# (the one after the section with the given id) and the bottom strip,
# the free height over that separator, and how far that separator's
# end is under its section's header (next to the same of the folded
# Skills section, whose separator holds no free height).
_OPEN_SECTIONS_JS = """
above => {
  const rect = sel => document.querySelector(sel).getBoundingClientRect();
  const panel = document.getElementById('meta-panel');
  const body = rect('#meta-recommendations-list');
  const separator = rect('#' + above + ' + .meta-section-resizer');
  const under = id => rect('#' + id + ' + .meta-section-resizer').bottom
    - rect('#' + id + ' .meta-section-hdr').bottom;
  return {
    open: Array.from(document.querySelectorAll('#meta-panel > .meta-section:not(.collapsed)'),
                     s => s.id),
    toggles: document.querySelectorAll('#meta-recommendations .meta-section-toggle').length,
    text: document.getElementById('meta-recommendations').innerText.trim(),
    stored: localStorage.getItem('kiss-meta-section-collapsed:meta-recommendations'),
    bodyHeight: body.height,
    cardHeight: rect('.recommendation-row:not([inert])').height,
    gapAbove: body.top - separator.bottom,
    gapBelow: rect('#meta-footer').top - body.bottom,
    freeAbove: separator.top - rect('#' + above + ' .meta-section-hdr').bottom,
    separatorUnder: [under(above), under('meta-skills')],
    infoHeight: rect('#meta-list').height,
    share: parseFloat(panel.style.getPropertyValue('--meta-body-share')),
    panelOverflows: panel.scrollHeight > panel.clientHeight,
  };
}
"""

_TWO_TASKS: dict[str, object] = {
    "type": "recommendations", "generatedAt": 1, "tasks": ["Fix the parser", "Run the tests"],
}


def test_a_fresh_profile_shows_only_the_pinned_recommendations(
    browser: Browser, remote_url: str
) -> None:
    """The first load after an install or an update folds every section
    that folds, whatever the profile stored: Recommendations, pinned open
    (its card alone: no header, no toggle) and hidden until it has a
    task, is alone with the folded headers, as tall as its card and
    pinned to the bottom strip, the free height over it and over the
    separator of the section shown above it.  It stays there when
    another section opens, and the next load of the same version keeps
    the user's choice."""
    page = _open_page(
        browser, remote_url, 1200, global_sections=True, fresh_profile=True,
        storage={"kiss-meta-section-collapsed:meta-recommendations": "1"},
    )
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        # Without a task the section is not on screen.
        assert page.locator("#meta-recommendations").is_hidden()
        _deliver(page, _TWO_TASKS)
        fresh = page.evaluate(_OPEN_SECTIONS_JS, "meta-spend")
        assert fresh["open"] == ["meta-recommendations"], fresh
        # The task on show is all the section's text: no title, no
        # position counter, no status line.
        assert [fresh["toggles"], fresh["text"], fresh["stored"]] == [
            0, "Fix the parser", None,
        ], fresh
        assert page.locator("#meta-recommendations button").count() == 0
        assert page.evaluate("localStorage.getItem('kiss-meta-sections-version')")
        # The body is its card, right under the separator (whose 2px
        # margin and the card's own 4px part them) and 8px over the
        # strip; the panel is empty from the folded headers down to
        # the separator, and that counts as free height.
        assert fresh["bodyHeight"] == pytest.approx(fresh["cardHeight"], abs=0.5), fresh
        assert 20 < fresh["cardHeight"] < 60, fresh
        assert fresh["gapAbove"] == pytest.approx(6, abs=0.5), fresh
        assert fresh["gapBelow"] == pytest.approx(8, abs=0.5), fresh
        assert fresh["freeAbove"] > 300, fresh
        assert fresh["share"] == pytest.approx(fresh["bodyHeight"] + fresh["freeAbove"], abs=0.5)
        # With Spend out of the panel the separator over the card is
        # the one of Connectors, and the free height is over that one.
        page.evaluate("document.getElementById('meta-spend').hidden = true")
        _deliver(page, _TWO_TASKS)
        moved = page.evaluate(_OPEN_SECTIONS_JS, "meta-apps")
        assert [moved["gapAbove"], moved["gapBelow"]] == pytest.approx([6, 8], abs=0.5), moved
        assert moved["freeAbove"] > 300, moved
        page.evaluate("document.getElementById('meta-spend').hidden = false")
        _deliver(page, _TWO_TASKS)
        # The user opens Task Info: it fills the panel down to the
        # card, which stays, 8px over the strip, its separator as far
        # under the Spend header as any other is under its own ...
        _toggle(page, "meta-section-info")
        opened = page.evaluate(_OPEN_SECTIONS_JS, "meta-spend")
        assert opened["open"] == ["meta-section-info", "meta-recommendations"], opened
        assert opened["bodyHeight"] == pytest.approx(fresh["bodyHeight"], abs=0.5), opened
        assert [opened["gapAbove"], opened["gapBelow"]] == pytest.approx([6, 8], abs=0.5), opened
        assert opened["freeAbove"] == pytest.approx(0, abs=0.5), opened
        assert opened["separatorUnder"] == pytest.approx([11, 11], abs=0.5), opened
        assert opened["infoHeight"] > 300 and not opened["panelOverflows"], opened
        reload_retrying_network_change(page)
        page.wait_for_selector("body.remote-chat", state="attached")
        # ... and a reload of the same version keeps that.
        assert page.evaluate(
            "() => Array.from(document.querySelectorAll("
            "'#meta-panel > .meta-section:not(.collapsed)'), s => s.id)"
        ) == ["meta-section-info", "meta-recommendations"]
    finally:
        page.close()


# The Recommendations body: the tasks whose cards show (a card sliding
# out is visible only while it slides, so it does not count), the
# animation each card runs, the body's middle and height, the cards'
# heights and how many different tops they have (one: they fill a grid
# cell), and what the message box and the keyboard focus hold.
_RECOMMENDATIONS_JS = """
() => {
  const panel = document.getElementById('meta-panel');
  const list = document.getElementById('meta-recommendations-list');
  const rows = Array.from(list.children);
  const box = list.getBoundingClientRect();
  return {
    shown: rows.filter(r => getComputedStyle(r).visibility === 'visible'
                            && !r.classList.contains('recommendation-out'))
      .map(r => r.dataset.task),
    slides: rows.map(r => getComputedStyle(r).animationName),
    middle: [box.left + box.width / 2, box.top + box.height / 2],
    bodyHeight: box.height,
    cardHeights: rows.map(r => r.getBoundingClientRect().height),
    tops: new Set(rows.map(r => Math.round(r.getBoundingClientRect().top))).size,
    scrolls: list.scrollHeight > list.clientHeight + 1,
    share: parseFloat(panel.style.getPropertyValue('--meta-body-share')),
    infoHeight: document.getElementById('meta-list').getBoundingClientRect().height,
    panelOverflows: panel.scrollHeight > panel.clientHeight,
    focused: document.activeElement.dataset.task || document.activeElement.id,
    composer: document.getElementById('task-input').value,
  };
}
"""

# The slide of the card with the given task, held a tenth of the way
# in: whether the card is in sight and how far it is off its place
# (px; to the right is positive).
_HOLD_SLIDE_JS = """
task => {
  const row = Array.from(document.querySelectorAll('.recommendation-row'))
    .find(r => r.dataset.task === task);
  const [slide] = row.getAnimations();
  slide.pause();
  slide.currentTime = slide.effect.getComputedTiming().duration / 10;
  const style = getComputedStyle(row);
  return {visible: style.visibility === 'visible',
          shift: new DOMMatrix(style.transform).m41};
}
"""

# End the slides of the Recommendations cards: the tasks of the cards
# in sight afterwards, and how many cards are off their place.
_END_SLIDES_JS = """
() => {
  const list = document.getElementById('meta-recommendations-list');
  for (const slide of list.getAnimations({subtree: true})) slide.finish();
  const rows = Array.from(list.children);
  return {
    visible: rows.filter(r => getComputedStyle(r).visibility === 'visible')
      .map(r => r.dataset.task),
    moved: rows.filter(r => getComputedStyle(r).transform !== 'none').length,
  };
}
"""


def test_recommendations_show_one_card_at_a_time_and_the_next_every_30_seconds(
    browser: Browser, remote_url: str
) -> None:
    """Every task's card fills the one cell of the Recommendations body,
    as tall as the longest task needs, and one of them shows: the next
    slides in every 30 seconds as the one before slides out, unless
    the pointer is over the card or the keyboard focus in it.  The
    body clips the sliding cards at its sides, and under
    prefers-reduced-motion the slide takes no time.  A card off show
    takes neither the focus nor a click.  A task taller than a third
    of the window, or than an equal share of the panel when that is
    less, is cut there and scrolls."""
    short, long = "Fix the parser", ("Review the open pull requests and " * 6).strip()
    words, more = ("word " * 900).strip(), ("more " * 900).strip()
    page = _open_page(browser, remote_url, 1400, global_sections=True, fresh_profile=True)

    def look() -> dict[str, Any]:
        return cast(dict[str, Any], page.evaluate(_RECOMMENDATIONS_JS))

    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        # From here on only run_for moves the page's time.
        page.clock.install(time=0)
        page.clock.pause_at(60)
        _deliver(page, {"type": "recommendations", "generatedAt": 1, "tasks": [short, long]})
        first = look()
        assert [first["shown"], first["tops"]] == [[short], 1], first
        assert first["slides"] == ["none", "none"], first
        # Six lines of text and the card's padding.
        assert first["cardHeights"] == [first["bodyHeight"]] * 2 and first["bodyHeight"] > 100
        assert page.evaluate(
            "getComputedStyle(document.getElementById('meta-recommendations-list')).overflowX"
        ) == "hidden"
        # A slide takes 0.45 s of real time, which the page's clock does
        # not hold; a minute here, so the test can look at one under way.
        duration = "getComputedStyle(document.documentElement).getPropertyValue('--dur-slide')"
        assert page.evaluate(duration).strip() == "0.45s"
        page.evaluate("document.documentElement.style.setProperty('--dur-slide', '60s')")
        page.clock.run_for(29_999)
        assert look()["shown"] == [short]
        page.clock.run_for(1)
        second = look()
        assert second["shown"] == [long], second
        assert second["slides"] == [
            "recommendation-slide-out", "recommendation-slide-in",
        ], second
        assert second["bodyHeight"] == first["bodyHeight"], (first, second)
        # Under way, both cards are in sight: the old one on its way out
        # to the left, the new one coming from the right.  At the end
        # the new one is alone, in its place.
        leaving = page.evaluate(_HOLD_SLIDE_JS, short)
        coming = page.evaluate(_HOLD_SLIDE_JS, long)
        assert leaving["visible"] and leaving["shift"] < -10, leaving
        assert coming["visible"] and coming["shift"] > 10, coming
        assert page.evaluate(_END_SLIDES_JS) == {"visible": [long], "moved": 0}
        # The pointer over the card keeps it; so does the keyboard focus,
        # which Tab takes out of the list, not to the card off show.
        page.mouse.move(*first["middle"])
        page.clock.run_for(30_000)
        assert look()["shown"] == [long]
        page.mouse.move(10, 10)
        page.locator(".recommendation-row:not([inert])").focus()
        page.clock.run_for(30_000)
        held = look()
        assert [held["shown"], held["focused"]] == [[long], long], held
        page.keyboard.press("Tab")
        assert look()["focused"] not in (short, long)
        page.clock.run_for(30_000)
        assert look()["shown"] == [short]
        # Once the slide is over the long card lies over the short one,
        # hidden: a click there opens the one on show.
        assert page.evaluate(_END_SLIDES_JS) == {"visible": [short], "moved": 0}
        page.mouse.click(*first["middle"])
        assert look()["composer"] == short
        # A task taller than a third of the window (33vh of 900px) is
        # cut there and scrolls, though the folded panel has more room.
        # In a 600px window an equal share is less than its third once
        # Task Info, open above the card, has the other share: the
        # task is cut at the share.  The next card shows from its top.
        _deliver(page, {"type": "recommendations", "generatedAt": 1, "tasks": [words, more]})
        capped = look()
        assert capped["scrolls"] and not capped["panelOverflows"], capped
        assert capped["share"] > 297, capped
        assert capped["bodyHeight"] == pytest.approx(297, abs=0.5), capped
        page.set_viewport_size({"width": 1400, "height": 600})
        _toggle(page, "meta-section-info")
        cut = look()
        assert cut["scrolls"] and not cut["panelOverflows"], cut
        assert cut["share"] < 198, cut
        assert cut["bodyHeight"] == pytest.approx(cut["share"], abs=0.5), cut
        assert cut["infoHeight"] == pytest.approx(cut["share"], abs=0.5), cut
        scroll_top = "document.getElementById('meta-recommendations-list').scrollTop"
        assert page.evaluate(f"{scroll_top} = 300, {scroll_top}") == 300
        page.clock.run_for(30_000)
        assert page.evaluate(scroll_top) == 0 and look()["shown"] == [more]
        page.evaluate("document.documentElement.style.removeProperty('--dur-slide')")
        page.emulate_media(reduced_motion="reduce")
        assert page.evaluate(duration).strip() == "0s"
    finally:
        page.close()


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
        assert geo["headers"][1]["text"] == "Task Progress"
        assert geo["headers"][1]["expanded"] == "true"
        # The two bodies are equally tall, whatever their content.
        assert geo["listShown"] and geo["contentShown"], geo
        assert geo["listHeight"] == pytest.approx(geo["contentHeight"], abs=1), geo
        # The report reaches the strip at the panel's bottom and scrolls.
        assert geo["contentBottom"] == pytest.approx(geo["sectionsBottom"], abs=2), geo
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
        assert after["contentBottom"] == pytest.approx(after["sectionsBottom"], abs=2)
        assert after["storedH"] == str(round(after["listHeight"]))
        assert _section_resizer(page).get_attribute("aria-valuenow") == after["storedH"]

        # Dragging up past the list's content makes the list scroll.
        _drag_separator_by(page, -400)
        shrunk = _geometry(page)
        assert shrunk["listHeight"] < before["listHeight"], (before, shrunk)
        assert shrunk["listScrolls"] is True, shrunk
        assert shrunk["contentBottom"] == pytest.approx(shrunk["sectionsBottom"], abs=2)

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
        assert geo["headers"][1]["bottom"] == pytest.approx(geo["sectionsBottom"], abs=12)

        _toggle(page, "meta-section-info")
        geo = _geometry(page)
        assert geo["listShown"] is False
        assert geo["contentShown"] is False
        assert [h["expanded"] for h in geo["headers"]] == ["false", "false"]
        # Two headers stacked at the top, nothing else.
        assert geo["headers"][0]["top"] < geo["headers"][1]["top"] < geo["panelTop"] + 120

        # A click on a collapsed header opens that section alone: the
        # other folds (one section at a time), so the opened body fills
        # the panel and the separator is a divider again.
        _toggle(page, "meta-section-info")
        geo = _geometry(page)
        assert [h["expanded"] for h in geo["headers"]] == ["true", "false"]
        assert geo["storedCollapsed"] == {"info": None, "update": "1"}
        assert geo["listShown"] and not geo["contentShown"]
        _toggle(page, "meta-info")
        geo = _geometry(page)
        assert [h["expanded"] for h in geo["headers"]] == ["false", "true"]
        assert geo["storedCollapsed"] == {"info": "1", "update": None}
        assert geo["contentShown"] and not geo["listShown"]
        assert geo["resizers"][0]["handle"] is False
        assert geo["contentBottom"] == pytest.approx(geo["sectionsBottom"], abs=2)
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
                                'meta-time', 'meta-workdir']) {
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
        assert geo["contentBottom"] == pytest.approx(geo["sectionsBottom"], abs=2), geo
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
        assert geo["contentBottom"] == pytest.approx(geo["sectionsBottom"], abs=2)
        # With Task Info collapsed the Task update header moves up to
        # right under the first header row.  The close button ends above
        # it, so the refresh button's whole face takes the tap.
        hit = page.evaluate(_REFRESH_UNDER_CLOSE_JS)
        assert hit["closeBottom"] <= hit["refreshTop"], hit
        assert hit["hitsRefresh"] is True, hit
    finally:
        page.close()


# The Task update refresh button against the drawer's close button:
# whether a tap on the refresh button's top edge lands on it.
_REFRESH_UNDER_CLOSE_JS = """
() => {
  const refresh = document.getElementById('meta-info-refresh');
  const r = refresh.getBoundingClientRect();
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + 1);
  return {
    closeBottom: document.getElementById('meta-close').getBoundingClientRect().bottom,
    refreshTop: r.top,
    hitsRefresh: refresh.contains(hit),
    hitId: hit && (hit.id || hit.tagName),
  };
}
"""


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
        assert [h["text"] for h in geo["headers"]] == ["Task Info", "Task Progress", "Extra"]
        assert [h["expanded"] for h in geo["headers"]] == ["true", "true", "true"]
        # No task: Task Info and Extra are shown, equally tall; Extra,
        # last, reaches the bottom.
        assert [r["shown"] for r in geo["resizers"]] == [True, False, False]
        assert geo["resizers"][0]["handle"] is True
        extra = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert extra["extraBottom"] == pytest.approx(geo["sectionsBottom"], abs=2), extra
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

        # The owner hides its section with `hidden` (the panel watches
        # the attribute and re-applies the layout): the section is
        # gone, Task update is last again and fills, its separator hides.
        _section_resizer(page).dblclick()
        page.evaluate("() => { document.getElementById('meta-section-extra').hidden = true; }")
        geo = _geometry(page)
        extra = page.evaluate(_EXTRA_GEOMETRY_JS)
        assert extra["sectionShown"] is False, extra
        assert [r["shown"] for r in geo["resizers"]] == [True, False, False]
        assert geo["contentBottom"] == pytest.approx(geo["sectionsBottom"], abs=2), geo
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


# The local machine and one reachable remote, as the daemon reports them.
_MACHINES = [
    {
        "name": "laptop", "address": "https://10.0.0.5:8787", "ip": "10.0.0.5", "port": 8787,
        "local": True, "status": "reachable", "error": "",
    },
    {
        "name": "gpu-box", "address": "ssh me@gpu", "ip": "10.0.0.9", "port": 41234,
        "local": False, "status": "reachable", "error": "",
    },
]

# The Skills section's three replies: the skills the agent can use
# (`skills`), a search of the registry (`skillSearch`: one skill to add,
# one already there) and a check that asks for a person's review
# (`skillInstall`).
_SKILLS: dict[str, object] = {
    "type": "skills", "error": "",
    "skills": [
        {"name": "pdf", "description": "Read, fill and write PDF files.", "source": "bundled"},
        {
            "name": "web-design", "source": "user",
            "description": "Guidelines for the layout, type and colour of a web page. " * 4,
        },
    ],
}
_SKILL_SEARCH: dict[str, object] = {
    "type": "skillSearch", "query": "design", "error": "",
    "results": [
        {
            "id": "acme/skills/frontend-design", "name": "frontend-design",
            "source": "acme/skills", "installs": 233480, "installed": False,
            "url": "https://skills.example/acme/skills/frontend-design",
        },
        {
            "id": "acme/skills/pdf", "name": "pdf", "source": "acme/skills",
            "installs": 1200, "installed": True,
            "url": "https://skills.example/acme/skills/pdf",
        },
    ],
}
_SKILL_REVIEW: dict[str, object] = {
    "type": "skillInstall", "id": "acme/skills/frontend-design", "status": "review",
    "name": "frontend-design", "message": "The check found something to look at first.",
    "findings": [{
        "severity": "medium", "rule": "network", "file": "scripts/setup.sh", "line": 12,
        "message": "Downloads and runs a script.",
    }],
}


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
    viewportWidth: viewport.clientWidth,
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
            # The oldest cell is scrolled out of the viewport: bring it
            # in.  With six sections sharing the panel the Spend body
            # scrolls vertically too, which moves the graph: measure it
            # again where the scroll left it.
            cell.scroll_into_view_if_needed()
            cell.hover()
            box = tip.bounding_box()
            cell_box = cell.bounding_box()
            graph_box = page.locator("#meta-spend-graph").bounding_box()
            assert box is not None and cell_box is not None and graph_box is not None
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


def test_spend_heatmap_draws_only_the_columns_the_viewport_fits(
    browser: Browser, remote_url: str
) -> None:
    """A history no longer than the panel is wide draws exactly the week
    columns the heatmap's viewport fits: the graph's own padding is no
    room for a column (review 2026-11-08, W2), so nothing scrolls and no
    pager shows.  The panel is one the user widened on an earlier visit
    so that the fitting columns outnumber the 98-day minimum, and its
    viewport is 4 px short of one more column: measured on the padded
    graph instead, that column is drawn and sticks out."""
    page = _open_page(
        browser,
        remote_url,
        1200,
        height=900,
        global_sections=True,
        storage={"kiss-meta-w": "254"},
    )
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, {"type": "cronJobs", "jobs": []})
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": 1})
        report = _spend_report()
        report["days"] = [d for d in report["days"] if d["date"] == _day_key(0)]
        report["daysByModel"] = {_day_key(0): report["daysByModel"][_day_key(0)]}
        _deliver(page, report)
        geo = page.evaluate(_SPEND_GEOMETRY_JS)
        fits = (geo["viewportWidth"] + 3) // 14
        assert fits >= 15, ("setup: the widened panel fits more than the minimum", geo)
        assert geo["columns"] == fits, geo
        assert geo["scrollMax"] == 0, ("the drawn weeks stick out of the viewport", geo)
        assert geo["left"]["hidden"] and geo["right"]["hidden"], geo
        assert geo["lastDay"] == _day_key(0), geo
    finally:
        page.close()


# The element that has the keyboard focus: its class, whether it is in
# the Spend section, and its computed opacity.
_ACTIVE_ELEMENT_JS = """
() => {
  const a = document.activeElement;
  return {
    cls: a.className,
    inSpend: document.getElementById('meta-spend').contains(a),
    opacity: getComputedStyle(a).opacity,
  };
}
"""


def _spend_tab_order(page: Page) -> list[dict[str, Any]]:
    """Tab from the Spend header's refresh button until the focus leaves
    the section; the elements focused on the way.  A pager's opacity is
    read once its reveal transition has had time to settle."""
    page.focus("#meta-spend-refresh")
    order: list[dict[str, Any]] = []
    for _ in range(8):
        page.keyboard.press("Tab")
        active = page.evaluate(_ACTIVE_ELEMENT_JS)
        if not active["inSpend"]:
            break
        if "spend-nav" in active["cls"]:
            try:
                page.wait_for_function(
                    "() => getComputedStyle(document.activeElement).opacity === '1'",
                    timeout=2000,
                )
            except PlaywrightTimeoutError:
                pass
            active = page.evaluate(_ACTIVE_ELEMENT_JS)
        order.append(active)
    return order


def _spend_navs(order: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The pagers in a Tab order, each with its opacity when focused."""
    return [(o["cls"], o["opacity"]) for o in order if "spend-nav" in o["cls"]]


def test_spend_pagers_are_keyboard_reachable_only_when_available(
    browser: Browser, remote_url: str
) -> None:
    """With the pointer away from the graph, Tab reaches the pager that
    can still scroll and shows it (a focused control the user cannot
    see is no control), and skips the pager at the end the heatmap has
    reached, as the mouse does.  Both ends, and a heatmap that fits
    without scrolling (the panel is resized to 250px so the columns
    drawn to fit the graph's width also fit its 4px-narrower viewport)."""
    page = _open_page(
        browser,
        remote_url,
        1200,
        height=900,
        global_sections=True,
        storage={"kiss-meta-w": "250"},
    )
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, {"type": "cronJobs", "jobs": []})
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": 1})
        report = _spend_report()
        _deliver(page, report)
        page.mouse.move(1, 1)
        # Pinned to the latest week: Earlier is available, Later is not.
        order = _spend_tab_order(page)
        assert _spend_navs(order) == [("spend-nav left", "1")], order
        # Scrolled back to the oldest week: the other way round.
        page.evaluate("() => { document.querySelector('.spend-viewport').scrollLeft = 0; }")
        page.wait_for_function(
            "() => document.querySelector('.spend-nav.left').classList.contains('nav-hidden')"
        )
        order = _spend_tab_order(page)
        assert _spend_navs(order) == [("spend-nav right", "1")], order
        # Today alone fits the panel: nothing to scroll, no pager to tab to.
        today = _day_key(0)
        report["days"] = [d for d in report["days"] if d["date"] == today]
        report["daysByModel"] = {today: report["daysByModel"][today]}
        _deliver(page, report)
        assert page.evaluate("() => document.querySelectorAll('.spend-nav.nav-hidden').length") == 2
        order = _spend_tab_order(page)
        assert _spend_navs(order) == [], order
    finally:
        page.close()


_GLOBAL_GEOMETRY_JS = """
() => {
  const rect = id => document.getElementById(id).getBoundingClientRect();
  const apps = document.getElementById('meta-apps-list');
  return {
    sectionsBottom: rect('meta-footer').top,
    scheduleTop: rect('meta-schedule-body').top,
    scheduleHeight: rect('meta-schedule-body').height,
    machinesTop: rect('meta-machines-body').top,
    machinesHeight: rect('meta-machines-body').height,
    machines: Array.from(document.querySelectorAll(
      '#meta-machines-list .sidebar-panel-name')).map(e => e.textContent),
    skillsTop: rect('meta-skills-body').top,
    skillsHeight: rect('meta-skills-body').height,
    skills: Array.from(document.querySelectorAll(
      '#meta-skills-list .sidebar-panel-name')).map(e => e.textContent),
    skillsStatus: document.getElementById('meta-skills-status').textContent,
    appsTop: rect('meta-apps-list').top,
    appsBottom: rect('meta-apps-list').bottom,
    appsHeight: rect('meta-apps-list').height,
    spendBottom: rect('meta-spend-body').bottom,
    spendHeight: rect('meta-spend-body').height,
    recommendationsBottom: rect('meta-recommendations-list').bottom,
    recommendationsHeight: rect('meta-recommendations-list').height,
    recommendations: Array.from(document.querySelectorAll(
      '#meta-recommendations-list .sidebar-panel-name')).map(e => e.textContent),
    share: parseFloat(document.getElementById('meta-panel').style.getPropertyValue(
      '--meta-body-share')),
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
        # The subpanels are asked for once the daemon is up (the shim's
        # `daemonStatus connected`), not by the boot itself.
        posted = page.evaluate("() => window.__posted.map(m => m.type)")
        assert "getCronJobs" not in posted and "getAppsStatus" not in posted
        _deliver(page, {"type": "daemonStatus", "connected": True})
        posted = page.evaluate("() => window.__posted.map(m => m.type)")
        assert posted.count("getCronJobs") == 1 and posted.count("getAppsStatus") == 1
        assert posted.count("getSpendReport") == 1 and posted.count("getMachines") == 1
        assert posted.count("getSkills") == 1
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
        _deliver(page, {"type": "machines", "machines": _MACHINES, "checkedAt": 1})
        _deliver(page, _SKILLS)
        _deliver(page, _spend_report())
        _deliver(page, _TWO_TASKS)
        geo = page.evaluate(_GLOBAL_GEOMETRY_JS)
        # No running task: Task Info, Schedule, Machines, Skills,
        # Connectors (the Apps section's header) and Spend are on
        # screen under their headers; Recommendations, last, has none.
        assert geo["headers"] == [
            "Task Info", "Schedule", "Machines", "Skills", "Connectors", "Spend",
        ], geo
        # The six bodies that share are equally tall and Apps scrolls;
        # the pinned Recommendations body, last, is its card (cut at an
        # equal share on this crowded panel), 8px over the strip.
        assert geo["scheduleTop"] < geo["machinesTop"] < geo["skillsTop"] < geo["appsTop"], geo
        assert geo["scheduleHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert geo["machinesHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert geo["skillsHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert geo["skills"] == ["pdf", "web-design"], geo
        assert geo["skillsStatus"] == "2 skills available"
        assert geo["listHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert geo["spendHeight"] == pytest.approx(geo["appsHeight"], abs=1), geo
        assert 0 < geo["recommendationsHeight"] <= geo["share"] + 0.5 <= geo["appsHeight"] + 1, geo
        assert geo["recommendations"] == ["Fix the parser", "Run the tests"], geo
        assert geo["machines"] == ["laptop", "gpu-box"], geo
        assert geo["spendBottom"] < geo["recommendationsBottom"], geo
        assert geo["recommendationsBottom"] == pytest.approx(
            geo["sectionsBottom"] - 8, abs=1,
        ), geo
        assert geo["appsScrolls"], geo
        assert geo["appsStatus"] == "1 of 40 connected"
        assert geo["firstApp"] == "App 07", "connected apps are listed first"
        assert geo["buttons"] == 39, "every app that is not connected is a button"

        # The static server never completes the websocket handshake, so
        # the page believes the daemon is down (sends are held back).
        # A reconnect after an outage re-requests the six subpanels; a
        # repeated `connected` with no outage in between (the shim sends
        # one on `auth_required` and another on `auth_ok`) asks nothing.
        page.evaluate("() => { window.__posted.length = 0; }")
        _deliver(page, {"type": "daemonStatus", "connected": True})
        posted = page.evaluate("() => window.__posted.map(m => m.type)")
        assert "getCronJobs" not in posted and "getAppsStatus" not in posted
        _deliver(page, {"type": "daemonStatus", "connected": False, "reconnecting": True})
        _deliver(page, {"type": "daemonStatus", "connected": True})
        posted = page.evaluate("() => window.__posted.map(m => m.type)")
        assert posted.count("getCronJobs") == 1 and posted.count("getAppsStatus") == 1
        assert posted.count("getSpendReport") == 1 and posted.count("getMachines") == 1
        assert posted.count("getRecommendations") == 1 and posted.count("getSkills") == 1

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
    share and scrolls instead of squeezing the Apps list, whose first
    app stays readable."""
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
              const name = document.querySelector('#meta-apps-list .sidebar-panel-name');
              return {apps: r('meta-apps-list').height,
                      update: r('meta-info-content').height,
                      panel: r('meta-panel').height,
                      nameBottom: name.getBoundingClientRect().bottom - r('meta-apps-list').top,
                      updateScrolls: c.scrollHeight > c.clientHeight + 1};
            }"""
        )
        # Seven equal shares of a 700px panel after the headers, status
        # lines, the Skills search row, the separators and the tools
        # footer leave the Apps list about 37px at the 13px root size
        # (Recommendations, with no task, is hidden): its first app's
        # 30px row is in view whole.
        assert geo["apps"] >= 24, geo
        assert geo["nameBottom"] <= geo["apps"], geo
        assert geo["update"] == pytest.approx(geo["apps"], abs=1), geo
        assert geo["updateScrolls"], geo
    finally:
        page.close()


_SKILLS_CHROME_JS = """
() => {
  const shown = id => document.getElementById(id).getClientRects().length > 0;
  const rect = id => document.getElementById(id).getBoundingClientRect();
  return {
    header: shown('meta-skills-refresh'),
    search: shown('meta-skills-search'),
    body: shown('meta-skills-body'),
    searchBottom: rect('meta-skills-search').bottom,
    bodyTop: rect('meta-skills-body').top,
    resultsTop: rect('meta-skills-results').top,
    statusTop: rect('meta-skills-status').top,
  };
}
"""


def test_a_folded_skills_section_hides_its_search_row(browser: Browser, remote_url: str) -> None:
    """The Skills search field sits between the header and the scrolling
    body, with the found skills above the available ones.  Folding the
    section takes the field away with the body, so a folded section is
    its header alone; unfolding brings it back, ready for typing."""
    page = _open_page(browser, remote_url, 1200, global_sections=True)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, _SKILLS)
        _search_skills(page)
        opened = page.evaluate(_SKILLS_CHROME_JS)
        assert opened["header"] and opened["search"] and opened["body"], opened
        assert opened["searchBottom"] <= opened["bodyTop"] + 0.5, opened
        assert opened["bodyTop"] <= opened["resultsTop"] < opened["statusTop"], opened
        _toggle(page, "meta-skills")
        folded = page.evaluate(_SKILLS_CHROME_JS)
        assert folded["header"] is True, folded
        assert folded["search"] is False and folded["body"] is False, folded
        _toggle(page, "meta-skills")
        page.fill("#meta-skills-query", "pdf")
        assert page.evaluate(_SKILLS_CHROME_JS)["search"] is True
    finally:
        page.close()


# The verdict of the skill under review: its buttons as [text, left,
# top, right, bottom], the verdict's own [left, right], whether a row's
# + has the keyboard, and how many installs were asked for.
_SKILL_VERDICT_JS = """
() => {
  const verdict = document.querySelector('#meta-skills-results .skill-verdict');
  const box = el => el.getBoundingClientRect();
  return {
    buttons: verdict ? Array.from(verdict.querySelectorAll('button'), b =>
      [b.textContent, box(b).left, box(b).top, box(b).right, box(b).bottom]) : null,
    room: verdict ? [box(verdict).left, box(verdict).right] : null,
    addFocused: document.activeElement.classList.contains('skill-add'),
    installs: window.__posted.filter(m => m.type === 'installSkill').length,
  };
}
"""


def test_a_skill_to_review_offers_add_anyway_and_cancel_side_by_side(
    browser: Browser, remote_url: str
) -> None:
    """Under a skill whose check asks for a review, "Add anyway" and
    "Cancel" stand on one line inside the verdict's box.  Cancel takes
    the verdict away and hands the keyboard to the row's +; nothing is
    sent."""
    page = _open_page(browser, remote_url, 1200, global_sections=True)
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, _SKILLS)
        _search_skills(page)
        _deliver(page, _SKILL_REVIEW)
        shown = page.evaluate(_SKILL_VERDICT_JS)
        (add, a_left, a_top, a_right, a_bottom), (cancel, c_left, c_top, c_right, c_bottom) = (
            shown["buttons"]
        )
        assert [add, cancel] == ["Add anyway", "Cancel"], shown
        assert [a_top, a_bottom] == [c_top, c_bottom] and a_right < c_left, shown
        assert shown["room"][0] < a_left and c_right < shown["room"][1], shown
        page.click(".skill-cancel")
        gone = page.evaluate(_SKILL_VERDICT_JS)
        assert gone == {
            "buttons": None, "room": None, "addFocused": True, "installs": 0,
        }, gone
    finally:
        page.close()


# The example line of the section with the given id: whether it shows,
# its text, how many lines its command takes, whether it is cut, the
# edges met going down the section (the header's bottom, the status
# line's top, the body's top, the line's bottom, the list's top), how
# far under the body's top the line and the first row start, the
# command's outline with the room left over it under the body's top
# edge, and what the message box and the keyboard hold.
_EXAMPLE_JS = """
id => {
  const section = document.getElementById(id);
  const body = section.querySelector('.meta-section-body');
  const line = body.firstElementChild;
  const row = body.querySelector('li');
  const top = sel => section.querySelector(sel).getBoundingClientRect().top;
  const box = line.getBoundingClientRect();
  const cmd = line.querySelector('.meta-example-cmd');
  const ring = getComputedStyle(cmd);
  return {
    shown: line.getClientRects().length > 0,
    text: line.textContent,
    lines: cmd.getClientRects().length,
    cut: line.scrollWidth > line.clientWidth,
    order: [section.querySelector('.meta-section-hdr').getBoundingClientRect().bottom,
            top('.meta-info-status'), top('.meta-section-body'), box.bottom,
            top('.sidebar-panel-list')],
    offset: box.top - top('.meta-section-body'),
    rowOffset: row && row.getBoundingClientRect().top - top('.meta-section-body'),
    ring: [ring.outlineStyle, cmd.getBoundingClientRect().top - top('.meta-section-body')
      - parseFloat(ring.outlineWidth) - parseFloat(ring.outlineOffset)],
    composer: document.getElementById('task-input').value,
    focused: document.activeElement.id,
  };
}
"""


def test_the_example_lines_wrap_fold_with_their_sections_and_insert_their_commands(
    browser: Browser, remote_url: str
) -> None:
    """Schedule and Machines show an example of their slash command as
    the first line of their body, under the status line and over the
    list: it scrolls away with the rows, and a body with no more than
    a row's room shows its rows instead.  A command too long for the
    panel wraps instead of being cut.  A click on a command, or Enter
    on it, puts it in the message box; folding the section takes the
    line away."""
    cron, machine = (
        "/cron every weekday at 9am summarize my unread email", "/machine add user@host",
    )
    # The panel of a 1400px window has 247px for a line: room for the
    # Machines line in every platform's font (198px to 199px on Windows
    # and Linux, 209px in the wider San Francisco of macOS, which the
    # 207px of a 1200px window would wrap) and not for the Schedule
    # line (380px to 400px).
    width = 1400
    page = _open_page(browser, remote_url, width, global_sections=True)

    def example(section_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], page.evaluate(_EXAMPLE_JS, section_id))

    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        # Before the daemon answers, each status line reads that it is
        # loading.
        for section_id, command, lines in (
            ("meta-machines", machine, 1), ("meta-schedule", cron, 2),
        ):
            shown = example(section_id)
            assert [shown["shown"], shown["text"]] == [True, "Example: " + command], shown
            assert [shown["lines"], shown["cut"]] == [lines, False], shown
            assert shown["order"] == sorted(shown["order"]) and shown["order"][0] > 0, shown
            assert shown["offset"] == pytest.approx(0, abs=0.5), shown
        # More jobs than the Schedule body shows: the line scrolls out
        # of it with the rows, and back.
        _deliver(page, _cron_jobs(12))
        height = example("meta-schedule")["order"][3] - example("meta-schedule")["order"][2]
        scroll_top = "document.getElementById('meta-schedule-body').scrollTop"
        scrolled = page.evaluate(f"{scroll_top} = 1e6, {scroll_top}")
        assert scrolled > height > 0, (scrolled, height)
        assert example("meta-schedule")["offset"] == pytest.approx(-scrolled, abs=0.5)
        page.evaluate(f"{scroll_top} = 0")
        # A window so short that a body has no more than a row's room:
        # the line gives that room to the rows, and a tall window brings
        # it back.
        page.set_viewport_size({"width": width, "height": 500})
        page.locator("#meta-schedule .meta-section-example").wait_for(state="hidden")
        assert example("meta-schedule")["rowOffset"] == pytest.approx(0, abs=0.5)
        assert example("meta-machines")["shown"] is False
        page.set_viewport_size({"width": width, "height": 900})
        page.locator("#meta-schedule .meta-section-example").wait_for(state="visible")
        assert example("meta-schedule")["rowOffset"] > height
        page.click("#meta-schedule .meta-example-cmd")
        clicked = example("meta-schedule")
        assert [clicked["composer"], clicked["focused"]] == [cron, "task-input"], clicked
        page.fill("#task-input", "")
        page.focus("#meta-machines .meta-example-cmd")
        # The keyboard's ring is drawn outside the command's box: the
        # line keeps room for it under the body's top edge, which clips.
        ring = example("meta-machines")["ring"]
        assert ring[0] == "solid" and ring[1] >= 0, ring
        page.keyboard.press("Enter")
        assert example("meta-machines")["composer"] == machine
        _toggle(page, "meta-schedule")
        assert example("meta-schedule")["shown"] is False
        assert example("meta-machines")["shown"] is True
    finally:
        page.close()


# Each surface that shows the task-info panel, as (viewport width, JS
# run after boot).  The remote page boots as the desktop dock (wide) or
# the mobile drawer (narrow, opened from its footer button).
_SURFACES = {
    "remote desktop": (1200, ""),
    "remote mobile drawer": (500, "document.getElementById('meta-drawer-btn').click();"),
}

# The heights of the seven bodies that share the panel, then that of
# the pinned Recommendations body.
_BODY_HEIGHTS_JS = """
() => ['meta-list', 'meta-info-content', 'meta-schedule-body', 'meta-machines-body',
       'meta-skills-body', 'meta-apps-list', 'meta-spend-body',
       'meta-recommendations-list']
  .map(id => document.getElementById(id).getBoundingClientRect().height)
"""


@pytest.mark.parametrize("surface", list(_SURFACES))
def test_every_surface_gives_the_expanded_sections_equal_heights(
    browser: Browser, remote_url: str, surface: str
) -> None:
    """All eight sections expanded: the bodies of the seven that fold
    are equally tall by default on every surface, whatever their
    content, and the pinned Recommendations body is its card.  Dragging
    the Schedule / Machines boundary moves only that boundary: the
    bodies above it keep their heights, the four below that share
    absorb the loss between them, and the card keeps its height."""
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
        _deliver(page, {"type": "machines", "machines": _MACHINES, "checkedAt": 1})
        _deliver(page, _SKILLS)
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": 1})
        _deliver(page, _spend_report())
        _deliver(page, _TWO_TASKS)
        # Show the Task update section directly (as setMetaInfoHTML
        # does) and re-apply the layout the way a `hidden` change does.
        page.evaluate(
            """html => {
              document.getElementById('meta-info-content').innerHTML = html;
              const update = document.getElementById('meta-info');
              update.classList.add('visible');
              update.hidden = true;
              update.hidden = false;
            }""",
            _LONG_REPORT,
        )
        *heights, card = page.evaluate(_BODY_HEIGHTS_JS)
        assert min(heights) > 40, heights
        assert max(heights) - min(heights) <= 1, heights
        assert 20 < card < min(heights), (card, heights)

        resizer = page.locator("#meta-schedule + .meta-section-resizer")
        box = resizer.bounding_box()
        assert box is not None
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y + 30, steps=6)
        page.mouse.up()
        *dragged, dragged_card = page.evaluate(_BODY_HEIGHTS_JS)
        assert dragged_card == card, (card, dragged_card)
        assert dragged[0] == pytest.approx(heights[0], abs=1), (heights, dragged)
        assert dragged[1] == pytest.approx(heights[1], abs=1), (heights, dragged)
        assert dragged[2] == pytest.approx(heights[2] + 30, abs=2), (heights, dragged)
        assert all(d < h for d, h in zip(dragged[3:], heights[3:], strict=True)), (
            heights, dragged,
        )
        assert sum(dragged[3:]) == pytest.approx(sum(heights[3:]) - 30, abs=2), (
            heights, dragged,
        )
    finally:
        page.close()


# Lists one chat in the Chats sidebar: the answer to the page's own
# getHistory request.
_LIST_ONE_CHAT_JS = """
() => {
  const asked = window.__posted.filter(m => m.type === 'getHistory').pop();
  window.dispatchEvent(new MessageEvent('message', {data: {
    type: 'history', offset: 0, generation: asked ? asked.generation : 0,
    sessions: [{id: 'c1', task_id: 1, title: 'Fix the flaky search test',
                preview: 'Fix the flaky search test', has_events: true,
                failed: false, is_running: false, timestamp: 1790440000}]}}));
}
"""

# The two kinds of text of the Chats sidebar: a chat's summary and the
# day label over a day's chats.
_CHAT_SUMMARY = "#history-list .history-chat-title"
_DAY_LABEL = "#history-list .history-day-sep"

# Reads the font of the elements at two selectors (a chat's summary and
# a day label), of each kind of row and of each kind of detail in the
# task-info panel, and of the Chats search field and the fields of an
# inline form.
_FONTS_JS = """
([summary, label]) => {
  const font = selector => {
    const cs = getComputedStyle(document.querySelector(selector));
    return [cs.fontFamily, cs.fontSize, cs.fontWeight, cs.fontStyle, cs.lineHeight];
  };
  const fonts = selectors => Object.fromEntries(selectors.map(s => [s, font(s)]));
  return {
    summary: font(summary),
    label: font(label),
    rows: fonts([
      '#meta-list .meta-item', '#meta-list .meta-label', '#meta-list .meta-value',
      '#meta-info-status', '#meta-info-content', '#meta-info-content p',
      '#meta-info-content li', '#meta-info-content td',
      '#meta-schedule .meta-section-example', '#meta-machines .meta-example-cmd',
      '#meta-schedule-list .sidebar-panel-name', '#meta-machines-list .sidebar-panel-name',
      '#meta-skills-status', '#meta-skills-list .sidebar-panel-name',
      '#meta-skills-results a.sidebar-panel-name',
      '#meta-apps-status', '#meta-apps-list .sidebar-panel-name',
      '.spend-total', '.spend-model-name', '.spend-model-value',
      '#meta-recommendations-list .sidebar-panel-name',
    ]),
    details: fonts([
      '.sidebar-panel-sub', '.sidebar-panel-badge', '.app-state', '.spend-models-title',
      '.sidebar-inline-form-label',
      '#meta-skills-found-title', '.skills-clear', '.skill-state', '.skill-verdict-message',
      '.skill-severity', '.skill-finding-text', '.skill-finding-where', '.skill-add-anyway',
      '.skill-cancel',
      '#meta-info-content h1', '#meta-info-content h2', '#meta-info-content h4',
    ]),
    fields: fonts([
      '#history-search', '.sidebar-inline-form input', '.sidebar-inline-form select',
      '.sidebar-inline-form textarea', '#meta-skills-query',
    ]),
  };
}
"""

# Reads the font ("family | size | weight | style") of the elements at
# two selectors and of every element the task-info panel shows text in,
# each named by its class, id or tag; the code a report quotes is kept
# apart.  The drawer's close button is a glyph, not text.
_SHOWN_FONTS_JS = """
([row, label]) => {
  const font = el => {
    const cs = getComputedStyle(el);
    return [cs.fontFamily, cs.fontSize, cs.fontWeight, cs.fontStyle].join(' | ');
  };
  const shown = {};
  const code = {};
  const walker = document.createTreeWalker(
    document.getElementById('meta-panel'), NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const el = node.parentElement;
    if (!node.nodeValue.trim() || el.id === 'meta-close') continue;
    if (el.getClientRects().length === 0) continue;
    const fonts = el.closest('code') ? code : shown;
    const key = font(el);
    (fonts[key] = fonts[key] || []).push(el.getAttribute('class') || el.id || el.tagName);
  }
  return {
    row: font(document.querySelector(row)),
    label: font(document.querySelector(label)),
    shown,
    code,
  };
}
"""


def shown_panel_fonts(page: Page, row_selector: str, label_selector: str) -> str:
    """Check the font of every text the task-info panel shows.

    All of it is in the family and style of the element at
    ``row_selector``, at that element's size or at the size of the
    element at ``label_selector``, in the regular or the medium weight.
    Code is the one other face: regular, at the label's size.

    Returns:
        The classes (or id, or tag) of the elements showing text, joined
        by spaces, for the caller to check that the text it is about was
        on screen.
    """
    fonts = page.evaluate(_SHOWN_FONTS_JS, [row_selector, label_selector])
    family, row_size, _, style = fonts["row"].split(" | ")
    label_size = fonts["label"].split(" | ")[1]
    assert row_size.endswith("px") and label_size != row_size, fonts
    names: list[str] = []
    for key, elements in fonts["shown"].items():
        shown_family, size, weight, shown_style = key.split(" | ")
        assert [shown_family, shown_style] == [family, style], (key, elements)
        assert size in (row_size, label_size), (key, elements)
        assert weight in ("400", "500"), (key, elements)
        names += elements
    for key, elements in fonts["code"].items():
        code_family, size, weight, code_style = key.split(" | ")
        assert code_family != family, (key, elements)
        assert [size, weight, code_style] == [label_size, "400", style], (key, elements)
        names += elements
    return " ".join(names)


def _deliver_global_sections(page: Page) -> None:
    """Fill Schedule, Machines, Skills (with a search and a skill whose
    check asks for a review), Apps, Spend and Recommendations."""
    _deliver(page, _cron_jobs(2))
    _deliver(page, {"type": "machines", "machines": _MACHINES, "checkedAt": 1})
    _deliver(page, _SKILLS)
    _search_skills(page)
    _deliver(page, _SKILL_REVIEW)
    _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": "2026-09-26T11:00:00"})
    _deliver(page, _spend_report())
    _deliver(page, {
        "type": "recommendations", "generatedAt": 1,
        "tasks": ["Fix the parser", "Run the tests"],
    })


# A task update with what a report is written in: headings, bold text,
# a table, and code inline and in a block.
_REPORT = (
    "<h1>Progress</h1><p>Read the <strong>data</strong>, <b>ran</b> <code>make test</code>.</p>"
    "<h2>Results</h2><table><thead><tr><th>Model</th></tr></thead>"
    "<tbody><tr><td>baseline</td></tr></tbody></table>"
    "<h4>Next</h4><ul><li>Train the model.</li></ul><pre><code>uv run pytest</code></pre>"
)


@pytest.mark.parametrize("surface", list(_SURFACES))
def test_the_panel_text_is_set_like_the_chats_sidebar(
    browser: Browser, remote_url: str, surface: str
) -> None:
    """The task-info panel has the Chats sidebar's two fonts and no
    other.  A row's text (a Task Info row, the Task update status line
    and report, the name of a job, a machine, an app or a model, a
    recommendation, a skill, a section's status line or its example
    line) has the font of a chat's summary: the same family, size,
    weight, style and line height.
    What is secondary to a row (a job's schedule line, a badge, an
    app's state, a group label, a form's label, what a skill's check
    found) has the family and size of a day label, and a form's field
    and the Skills search field are set as the Chats search field.
    Every other piece of text in the panel is in that family at one of
    those two sizes, at the regular or the medium weight.  The report
    keeps to it whatever it is written in: a heading is a label over its
    part, bold text and a table's header are the medium weight, and code
    is the code face at the label's size."""
    width, setup = _SURFACES[surface]
    page = _open_page(browser, remote_url, width, global_sections=True)
    try:
        page.evaluate(f"() => {{ {setup} }}")
        page.evaluate(_LIST_ONE_CHAT_JS)
        _deliver_global_sections(page)
        page.click("#meta-schedule-add")
        # Every section is expanded until a task starts; the task then
        # folds all but Task Info and Task update, which it fills.
        idle = shown_panel_fonts(page, _CHAT_SUMMARY, _DAY_LABEL)
        for name in (
            "sidebar-panel-name", "sidebar-panel-sub", "app-state", "spend-model-name",
            "sidebar-inline-form-label", "skill-state review", "skill-state added",
            "skill-severity medium", "skill-finding-where", "skill-add-anyway", "skill-cancel",
            "meta-section-example", "meta-example-cmd",
        ):
            assert name in idle, (name, idle)
        # The recommendation on show was among the texts checked.
        assert page.locator(".recommendation-row:not([inert])").is_visible()
        _show_task_update(page, _REPORT)
        running = shown_panel_fonts(page, _CHAT_SUMMARY, _DAY_LABEL)
        assert "meta-info-status" in running and "meta-value" in running, running
        assert {"H1", "H2", "H4", "STRONG", "B", "TH", "TD", "CODE"} <= set(running.split())
        fonts = page.evaluate(_FONTS_JS, [_CHAT_SUMMARY, _DAY_LABEL])
        summary, label = fonts["summary"], fonts["label"]
        for selector, font in fonts["rows"].items():
            assert font == summary, (selector, fonts)
        for selector, font in fonts["details"].items():
            assert [font[0], font[1], font[3]] == [label[0], label[1], label[3]], (
                selector, fonts,
            )
        for selector, font in fonts["fields"].items():
            assert font[:4] == summary[:4], (selector, fonts)
    finally:
        page.close()


# The Tags row of the Task Info list: its chips' names and how many
# lines they take, each remove button's box, whether the list overflows
# sideways, and the add select's options and state.
_TAGS_ROW_JS = """
() => {
  const row = document.getElementById('meta-tags');
  const list = document.getElementById('meta-list');
  const chips = [...row.querySelectorAll('.meta-tag')];
  const add = row.querySelector('.meta-tag-add');
  return {
    chips: chips.map(c => c.firstChild.data),
    lines: new Set(chips.map(c => Math.round(c.getBoundingClientRect().top))).size,
    removes: [...row.querySelectorAll('.meta-tag-remove')].map(b => {
      const r = b.getBoundingClientRect();
      return Math.min(r.width, r.height);
    }),
    overflows: list.scrollWidth > list.clientWidth,
    options: add ? [...add.options].map(o => o.textContent) : null,
    locked: add ? add.disabled : null,
  };
}
"""

_FOCUS_RING_JS = """
() => {
  const cs = getComputedStyle(document.activeElement);
  return [document.activeElement.className, cs.outlineStyle, cs.outlineWidth];
}
"""


@pytest.mark.parametrize("touch", [False, True], ids=["desktop dock", "phone drawer"])
def test_the_tags_row_shows_and_edits_the_tags_of_the_task(
    browser: Browser, remote_url: str, touch: bool
) -> None:
    """The Tags row in a real browser: three tags, an edit by pointer and
    by keyboard, one tag, and every tag at once.  The chips and their
    buttons are set in the panel's fonts; the chips wrap inside the
    panel; a remove button takes a finger on a phone (24px at least)
    and shows the keyboard focus; the lone tag of a task cannot be
    removed."""
    page = _open_page(browser, remote_url, *((390, 844) if touch else (1400, 900)), touch=touch)
    task_id = "a" * 32

    def tags(names: list[str]) -> dict[str, Any]:
        _deliver(page, {
            "type": "taskTags", "taskId": task_id, "tags": names,
            "allTags": list(ALL_TAGS), "editable": True,
        })
        return cast(dict[str, Any], page.evaluate(_TAGS_ROW_JS))

    def edits() -> list[list[str]]:
        posted = page.evaluate("window.__posted.filter(m => m.type === 'setTaskTags')")
        assert all(m["taskId"] == task_id for m in posted), posted
        return [m["tags"] for m in posted]

    try:
        if touch:
            page.locator("#meta-drawer-btn").tap()
            page.wait_for_function(
                "() => Math.abs(document.getElementById('meta-panel')"
                ".getBoundingClientRect().right - innerWidth) < 1"
            )
        page.evaluate(_LIST_ONE_CHAT_JS)
        _deliver(page, {"type": "daemonStatus", "connected": True})
        _deliver(page, {"type": "task_settings", "settings": {"task_id": task_id}})
        asked = page.evaluate("window.__posted.filter(m => m.type === 'getTaskTags')")
        assert asked == [{"type": "getTaskTags", "taskId": task_id}]

        row = tags(["work", "coding", "testing"])
        assert row["chips"] == ["work", "coding", "testing"] and len(row["removes"]) == 3
        rest = [name for name in ALL_TAGS if name not in row["chips"]]
        assert row["options"] == ["+ Add tag", *rest] and row["overflows"] is False
        shown = shown_panel_fonts(page, _CHAT_SUMMARY, _DAY_LABEL).split()
        assert {"meta-tag", "meta-tag-remove"} <= set(shown), shown
        if touch:
            assert min(row["removes"]) >= 24, row
            page.locator(".meta-tag-remove[data-tag='coding']").tap()
        else:
            # Tab from the Task id reaches the first chip's button, ringed.
            page.locator("#meta-task-id").focus()
            page.keyboard.press("Tab")
            assert page.evaluate(_FOCUS_RING_JS) == ["meta-tag-remove", "solid", "1px"]
            page.keyboard.press("Tab")
            page.keyboard.press("Enter")
        assert edits() == [["work", "testing"]]
        assert page.evaluate(_TAGS_ROW_JS)["locked"] is True

        assert tags(["work", "testing"])["locked"] is False
        # The keyboard edit gets the focus back, on the select; a tap leaves none there.
        assert page.evaluate("document.activeElement.tagName") == ("BODY" if touch else "SELECT")
        page.select_option("#meta-tags .meta-tag-add", "review")
        assert edits()[1] == ["work", "testing", "review"]
        assert tags(edits()[1])["chips"] == ["work", "testing", "review"]

        row = tags(["work"])
        assert (row["chips"], row["removes"], len(row["options"])) == (["work"], [], len(ALL_TAGS))
        row = tags(list(ALL_TAGS))
        assert row["lines"] > 1 and row["overflows"] is False and row["options"] is None, row
    finally:
        page.close()


# The room left and right of the first element at each selector inside
# the padding box of the panel with the given id.
_INSETS_JS = """
([panelId, selectors]) => {
  const panel = document.getElementById(panelId);
  const left = panel.getBoundingClientRect().left + panel.clientLeft;
  return Object.fromEntries(selectors.map(selector => {
    const rect = document.querySelector(selector).getBoundingClientRect();
    return [selector, [rect.left - left, left + panel.clientWidth - rect.right]];
  }));
}
"""


def panel_insets(page: Page, panel_id: str, selectors: list[str]) -> dict[str, list[float]]:
    """Measure how far inside a side panel its elements start and end.

    Returns:
        For the first element at each of ``selectors``, the room left
        and right of it inside the padding box of the panel ``panel_id``.
    """
    insets: dict[str, list[float]] = page.evaluate(_INSETS_JS, [panel_id, selectors])
    return insets


# The corner and the vertical padding of the first element at each
# selector.
_SHAPES_JS = """
selectors => selectors.map(selector => {
  const cs = getComputedStyle(document.querySelector(selector));
  return [cs.borderTopLeftRadius, cs.paddingTop, cs.paddingBottom];
})
"""

_CHAT_ROW = "#history-list .history-chat-header"
_BOXED_ROWS = [
    ".sched-row", ".machine-row", ".skill-row", ".skill-found-row", "button.app-row-main",
]


@pytest.mark.parametrize("surface", list(_SURFACES))
def test_a_panel_row_is_boxed_like_a_chat_row(
    browser: Browser, remote_url: str, surface: str
) -> None:
    """A Schedule, Machines, Skills or Apps row is a chat row's box: its
    corner, its vertical padding, and a fill that ends as far inside the
    panel as a chat row's fill does inside the Chats sidebar (a model row
    of Spend, a denser list, has the fill's reach).  The fill reaches out of
    the panel's column while the row's content stays in it: a job's name
    starts where the section header does and an app's state ends where
    the header does.  A body spans the panel, so a list that overflows
    scrolls at the panel's edge, and its rows are as wide as the rows of
    a list that does not overflow."""
    width, setup = _SURFACES[surface]
    page = _open_page(browser, remote_url, width, global_sections=True)
    try:
        page.evaluate(f"() => {{ {setup} }}")
        page.evaluate(_LIST_ONE_CHAT_JS)
        _deliver_global_sections(page)
        chat = panel_insets(page, "sidebar", [_CHAT_ROW])[_CHAT_ROW]
        assert min(chat) > 0, chat
        name, state = ".sched-row .sidebar-panel-name", "button.app-row-main .app-state"
        header, apps = "#meta-schedule .meta-section-hdr", "#meta-apps-list"
        insets = panel_insets(
            page, "meta-panel", [*_BOXED_ROWS, ".spend-model-row", header, name, state, apps],
        )
        for selector in (*_BOXED_ROWS, ".spend-model-row"):
            assert insets[selector] == pytest.approx(chat, abs=0.5), (selector, insets)
        shapes = page.evaluate(_SHAPES_JS, [_CHAT_ROW, *_BOXED_ROWS])
        assert shapes.count(shapes[0]) == len(shapes), shapes
        assert insets[header][0] > chat[0], insets
        assert insets[name][0] == pytest.approx(insets[header][0], abs=0.5), insets
        assert insets[state][1] == pytest.approx(insets[header][1], abs=0.5), insets
        # The 40 apps overflow their body, which scrolls at the panel's edges.
        assert page.evaluate(
            "() => { const list = document.getElementById('meta-apps-list');"
            " return list.scrollHeight > list.clientHeight; }"
        )
        assert insets[apps] == pytest.approx([0, 0], abs=0.5), insets
    finally:
        page.close()


_MINIMUM_GEOMETRY_JS = """
() => {
  const panel = document.getElementById('meta-panel');
  const rect = id => document.getElementById(id).getBoundingClientRect();
  const list = document.getElementById('meta-schedule-body');
  const row = list.querySelector('li');
  const share = parseFloat(getComputedStyle(panel).getPropertyValue('--meta-body-share'));
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
  return {
    share,
    minimum: Math.min(5 * rem, share / 2),
    listHeight: rect('meta-list').height,
    scheduleHeight: rect('meta-schedule-body').height,
    machinesHeight: rect('meta-machines-body').height,
    skillsHeight: rect('meta-skills-body').height,
    appsHeight: rect('meta-apps-list').height,
    spendHeight: rect('meta-spend-body').height,
    recommendationsHeight: rect('meta-recommendations-list').height,
    scheduleScrolls: list.scrollHeight > list.clientHeight + 1,
    firstRowInside: row !== null
      && row.getBoundingClientRect().top >= list.getBoundingClientRect().top - 0.5
      && row.getBoundingClientRect().bottom <= list.getBoundingClientRect().bottom + 0.5,
    panelOverflows: panel.scrollHeight > panel.clientHeight + 1,
    // By how much the panel's content is taller than the panel, and
    // how far above the panel's bottom edge the bottom strip ends.
    overflow: panel.scrollHeight - panel.clientHeight,
    stripGap: panel.getBoundingClientRect().bottom - rect('meta-footer').bottom,
    // The shown sections' headers (the hidden workspace sections have
    // none on screen).
    headerBottoms: Array.from(document.querySelectorAll(
        '#meta-panel > .meta-section:not([hidden]) > .meta-section-hdr'))
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
    equal share (at most 5rem): the jobs stay on screen and scroll, the
    dragged body yields instead, and because the minimum follows the
    panel's height a short window never makes the panel itself
    overflow.  A double-click on the separator still restores the equal
    shares.

    The short window is also where the example lines of Schedule and
    Machines once cost the panel its bottom strip: as rows of their own
    (61px together) they left the bodies nothing, the panel overflowed
    and the strip was clipped.  They lie in the bodies now, and a body
    down to its minimum shows its rows instead."""
    page = _open_page(
        browser,
        remote_url,
        1200,
        height=1300,
        global_sections=True,
        storage={"kiss-meta-section-h:meta-section-info": "2000"},
    )
    try:
        page.wait_for_selector("body.remote-desktop", state="attached")
        _deliver(page, _cron_jobs(12))
        _deliver(page, {"type": "machines", "machines": _MACHINES, "checkedAt": 1})
        _deliver(page, _SKILLS)
        _deliver(page, {"type": "appsStatus", "apps": _APPS, "checkedAt": "2026-09-26T11:00:00"})
        _deliver(page, _spend_report())
        geo = page.evaluate(_MINIMUM_GEOMETRY_JS)
        # Six equal shares of a 1300px panel (the six bodies that fold;
        # Recommendations, with no task, is hidden, and there is no
        # Task update yet): over 130px each, so the minimum, half a
        # share, is the 5rem cap, taller than a Schedule row whose
        # detail line wraps once (as it does on this 240px panel).
        assert geo["share"] > 130, geo
        assert geo["minimum"] == pytest.approx(65, abs=0.5), geo
        # Schedule, Machines, Skills, Apps and Spend each keep the
        # minimum and Recommendations, with nothing to show, takes no
        # room; Task Info has the rest, far less than the 2000px it
        # asked for.
        for key in (
            "scheduleHeight", "machinesHeight", "skillsHeight", "appsHeight", "spendHeight",
        ):
            assert geo[key] == pytest.approx(geo["minimum"], abs=1), (key, geo)
        assert geo["recommendationsHeight"] == 0, geo
        assert geo["listHeight"] == pytest.approx(
            6 * geo["share"] - 5 * geo["minimum"], abs=2,
        ), geo
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
        # 500px: the parts that do not shrink (seven header rows, the
        # status lines, the Skills search row, the separators and the
        # tools footer, with the long error below about 400px in all)
        # leave the bodies a few pixels each, never none.
        page.set_viewport_size({"width": 1200, "height": 500})
        page.wait_for_function(
            "() => parseFloat(getComputedStyle(document.getElementById('meta-panel'))"
            ".getPropertyValue('--meta-body-share')) < 60"
        )
        short = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert 0 < short["minimum"] < geo["minimum"], short
        assert short["scheduleHeight"] == pytest.approx(short["minimum"], abs=1), short
        assert short["panelOverflows"] is False, short
        assert short["overflow"] == 0 and abs(short["stripGap"]) <= 0.5, short
        assert all(bottom <= 500 for bottom in short["headerBottoms"]), short

        # A Task update whose status line then grows (a long error) adds
        # a seventh section and a taller header row on this short panel:
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
        assert len(crowded["headerBottoms"]) == 7, crowded
        assert 0 < crowded["minimum"] < short["minimum"], (short, crowded)
        assert crowded["panelOverflows"] is False, crowded
        assert crowded["overflow"] == 0 and abs(crowded["stripGap"]) <= 0.5, crowded
        assert all(bottom <= 500 for bottom in crowded["headerBottoms"]), crowded
        # Two recommendations on top of that (their card comes on
        # show, an eighth section), the fullest this panel gets: the
        # bodies still have a share and the strip still ends, whole, at
        # the panel's bottom edge.
        _deliver(page, _TWO_TASKS)
        full = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert 0 < full["recommendationsHeight"] <= full["share"] + 0.5, full
        assert 0 < full["minimum"] < crowded["minimum"], (crowded, full)
        assert full["overflow"] == 0 and abs(full["stripGap"]) <= 0.5, full
        _deliver(page, {"type": "recommendations", "generatedAt": 1, "tasks": []})

        # Double-clicking the separator forgets the dragged height:
        # the bodies have equal parts of the panel again, a share each
        # (Recommendations, without a task again, is hidden).
        # (The share with seven sections and the long status line in
        # 1000px is about 85px: the tall window is back once it is
        # above the short window's 60px.)
        page.set_viewport_size({"width": 1200, "height": 1000})
        page.wait_for_function(
            "() => parseFloat(getComputedStyle(document.getElementById('meta-panel'))"
            ".getPropertyValue('--meta-body-share')) > 60"
        )
        _section_resizer(page).dblclick()
        reset = page.evaluate(_MINIMUM_GEOMETRY_JS)
        assert reset["scheduleHeight"] == pytest.approx(reset["listHeight"], abs=1), reset
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

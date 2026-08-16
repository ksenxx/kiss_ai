# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the task history panel renders a pulsing green
circle next to every task that is currently running.

This is the running-indicator counterpart to
``test_history_failed_red_circle.py``.  It exercises the full
pipeline used by the History sidebar to surface the
``.sidebar-item-running`` green pulsing dot:

* The **backend** half — :meth:`VSCodeServer._get_running_task_ids`
  must report the row id of a task whose worker thread is alive.
  ``_get_history`` must then stamp ``is_running=True`` (and
  ``failed=False``) on every such row in the ``history`` broadcast,
  EVEN when the persisted ``result`` is the "Agent Failed Abruptly"
  sentinel left behind by a prior crashed task on the same row.

* The **frontend** half — ``renderHistory`` in ``media/main.js``
  must render a visible ``.sidebar-item-running`` dot as the FIRST
  child of every row whose ``is_running`` is ``True``, using the
  green colour ``#2e7d32`` (``rgb(46, 125, 50)``), the
  ``running-pulse`` keyframe animation, and 8x8 geometry.
  Rows whose ``is_running`` is ``False`` must NOT render the dot.

* A **live update** half — a ``status: running=true`` event must
  trigger ``refreshHistory()`` and a subsequent ``history`` reply
  with ``is_running=true`` must make the dot appear on the
  matching row WITHOUT a full page reload.  Likewise for the
  ``status: running=false`` transition that drops the dot.

The Playwright harness mirrors ``test_history_failed_red_circle.py``
so the same review checklist applies (visible offsetParent, real
viewport, real CSS).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.server.test_history_running_green_circle import _history_event_from_real_backend

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


def _build_test_page() -> str:
    """Return a self-contained HTML page that loads the real CSS+JS."""
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
      --vscode-terminal-ansiRed: #f44747;
      --vscode-terminal-ansiGreen: #6a9955;
      --vscode-terminal-ansiYellow: #d7ba7d;
      --vscode-terminal-ansiMagenta: #c586c0;
      --vscode-terminal-ansiCyan: #4ec9b0;
    }}
    html, body {{ height: 100%; margin: 0; padding: 0; }}
  </style>
  <style>{css}</style>
  <title>history running green circle test</title>
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


def _open_history_page(_browser, width: int = 480, height: int = 900):
    """Open the test harness and return ``(context, page)``."""
    context = _browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    page.set_content(_build_test_page(), wait_until="load")
    page.wait_for_function(
        "document.getElementById('history-list') !== null",
        timeout=5000,
    )
    page.evaluate(
        "() => {"
        " document.getElementById('app').style.display = '';"
        " const ov = document.getElementById('kiss-server-loading');"
        " if (ov) ov.style.display = 'none';"
        " document.getElementById('sidebar').classList.add('open');"
        " }"
    )
    iife_err = page.evaluate("() => window.__iifeError")
    if iife_err:
        pytest.fail(f"main.js IIFE setup raised: {iife_err}")
    return context, page


def _sample_sessions() -> list[dict]:
    """Return one running, one failed and one completed session."""
    return [
        {
            "id": "chat-run",
            "task_id": 2001,
            "title": "running task",
            "preview": "running task",
            "timestamp": 1700000100,
            "has_events": True,
            "failed": False,
            "is_running": True,
            "tokens": 0, "cost": 0.0, "steps": 0,
            "is_favorite": False, "work_dir": "",
            "model": "", "is_worktree": False,
            "is_parallel": False, "auto_commit_mode": False,
            "startTs": 1700000100000, "endTs": 0,
        },
        {
            "id": "chat-fail",
            "task_id": 2002,
            "title": "failing task",
            "preview": "failing task",
            "timestamp": 1700000000,
            "has_events": True,
            "failed": True,
            "is_running": False,
            "tokens": 0, "cost": 0.0, "steps": 0,
            "is_favorite": False, "work_dir": "",
            "model": "", "is_worktree": False,
            "is_parallel": False, "auto_commit_mode": False,
            "startTs": 1700000000000, "endTs": 1700000001000,
        },
        {
            "id": "chat-ok",
            "task_id": 2003,
            "title": "successful task",
            "preview": "successful task",
            "timestamp": 1700000200,
            "has_events": True,
            "failed": False,
            "is_running": False,
            "tokens": 0, "cost": 0.0, "steps": 0,
            "is_favorite": False, "work_dir": "",
            "model": "", "is_worktree": False,
            "is_parallel": False, "auto_commit_mode": False,
            "startTs": 1700000200000, "endTs": 1700000201000,
        },
    ]


def _post_history(
    page,
    sessions: list[dict],
    offset: int = 0,
    generation: int = 0,
    expected_total: int | None = None,
) -> None:
    """Dispatch a ``message`` event with a ``history`` payload."""
    ev = {
        "type": "history",
        "sessions": sessions,
        "offset": offset,
        "generation": generation,
    }
    if expected_total is None:
        expected_total = len(sessions) if offset == 0 else None
    if expected_total is None:
        before = page.evaluate(
            "() => document.querySelectorAll("
            "'#history-list .sidebar-item'"
            ").length"
        )
        expected_total = int(before) + len(sessions)
    page.evaluate("event => window.__post(event)", ev)
    page.wait_for_function(
        "expected => document.querySelectorAll("
        "'#history-list .sidebar-item'"
        ").length === expected",
        arg=expected_total,
        timeout=5000,
    )


def test_running_session_renders_green_circle(_browser) -> None:
    """Every ``s.is_running`` session must render exactly one visible
    green pulsing circle inside its row.

    Asserts the element exists, is painted, and matches the intended
    geometry / colour / animation from ``main.css``.
    """
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        info = page.evaluate(
            """
            () => {
              const rows = Array.from(
                document.querySelectorAll('#history-list .sidebar-item'),
              );
              return rows.map(r => {
                const dot = r.querySelector('.sidebar-item-running');
                const txt = r.querySelector('.sidebar-item-text');
                const out = {
                  text: txt ? txt.textContent : null,
                  category: r.dataset.category || null,
                  hasRunningDot: !!dot,
                  hasFailedDot: !!r.querySelector('.sidebar-item-failed'),
                  dotIsFirst: !!dot && r.firstElementChild === dot,
                };
                if (dot) {
                  const cs = getComputedStyle(dot);
                  out.dot = {
                    width: cs.width,
                    height: cs.height,
                    borderRadius: cs.borderRadius,
                    background: cs.backgroundColor,
                    animationName: cs.animationName,
                    animationDuration: cs.animationDuration,
                    animationIterationCount: cs.animationIterationCount,
                    visibility: cs.visibility,
                    display: cs.display,
                    opacity: cs.opacity,
                    offsetWidth: dot.offsetWidth,
                    offsetHeight: dot.offsetHeight,
                    tooltip: dot.dataset.tooltip,
                    ariaLabel: dot.getAttribute('aria-label'),
                  };
                }
                return out;
              });
            }
            """,
        )
        by_text = {r["text"]: r for r in info}

        run = by_text["running task"]
        assert run["category"] == "running", (
            f"running row miscategorised: {run['category']!r}"
        )
        assert run["hasRunningDot"], (
            "running task did not render a .sidebar-item-running element; "
            f"row info: {run}"
        )
        assert run["dotIsFirst"], (
            "running dot must be the FIRST child of the row so it sits "
            "to the left of the title"
        )
        dot = run["dot"]
        assert dot["width"] == "8px" and dot["height"] == "8px", (
            f"running dot is not 8x8: {dot['width']} x {dot['height']}"
        )
        assert dot["borderRadius"] in ("4px", "50%"), (
            f"running dot is not rounded: border-radius={dot['borderRadius']}"
        )
        assert dot["background"] == "rgb(46, 125, 50)", (
            "running dot is not the success-green colour: "
            f"background-color={dot['background']!r}; expected rgb(46, 125, 50)"
        )
        assert dot["animationName"] == "running-pulse", (
            "running dot must animate via 'running-pulse'; "
            f"animation-name={dot['animationName']!r}"
        )
        assert dot["animationDuration"] == "1.5s", (
            f"running dot animation duration must be 1.5s; "
            f"got: {dot['animationDuration']!r}"
        )
        assert dot["animationIterationCount"] == "infinite", (
            "running dot must pulse indefinitely; "
            f"iteration-count={dot['animationIterationCount']!r}"
        )
        assert dot["visibility"] == "visible", (
            f"running dot is hidden: visibility={dot['visibility']!r}"
        )
        assert dot["display"] != "none", (
            f"running dot is display:none: display={dot['display']!r}"
        )
        assert float(dot["opacity"]) > 0.0, (
            f"running dot is fully transparent: opacity={dot['opacity']!r}"
        )
        assert dot["offsetWidth"] > 0 and dot["offsetHeight"] > 0, (
            f"running dot has zero box: "
            f"offset={dot['offsetWidth']}x{dot['offsetHeight']}"
        )
        assert dot["tooltip"] == "Task running", (
            f"running dot has wrong tooltip: {dot['tooltip']!r}"
        )
        assert dot["ariaLabel"] == "Task running", (
            f"running dot has wrong aria-label: {dot['ariaLabel']!r}"
        )

        fail = by_text["failing task"]
        assert not fail["hasRunningDot"], (
            "failed task should not render a .sidebar-item-running element"
        )
        assert fail["hasFailedDot"], (
            "failed task should render a .sidebar-item-failed element"
        )

        ok = by_text["successful task"]
        assert not ok["hasRunningDot"], (
            "completed task should not render a .sidebar-item-running element"
        )
        assert not ok["hasFailedDot"], (
            "completed task should not render a .sidebar-item-failed element"
        )
    finally:
        context.close()


def test_running_filter_is_checked_by_default(_browser) -> None:
    """The History sidebar's #hf-running checkbox must default to
    checked so running rows are visible on first open.

    A default-unchecked running filter would make the feature appear
    broken even though the dot DOM exists.
    """
    context, page = _open_history_page(_browser)
    try:
        defaults = page.evaluate(
            """
            () => ({
              running: document.getElementById('hf-running').checked,
              errors: document.getElementById('hf-errors').checked,
              completed: document.getElementById('hf-completed').checked,
            })
            """
        )
        assert defaults == {"running": True, "errors": True, "completed": True}

        _post_history(page, _sample_sessions())
        visible_running_dot = page.evaluate(
            """
            () => {
              const row = document.querySelector(
                '#history-list .sidebar-item[data-category="running"]',
              );
              const dot = row && row.querySelector('.sidebar-item-running');
              return !!row && row.offsetParent !== null &&
                !!dot && dot.offsetParent !== null;
            }
            """
        )
        assert visible_running_dot, (
            "running task green circle should be visible with default filters"
        )
    finally:
        context.close()


def test_running_filter_toggle_hides_and_shows_running_row(_browser) -> None:
    """Unchecking the "Running" filter must hide the running row;
    re-checking it must restore it."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, _sample_sessions())
        visible0 = page.evaluate(
            "() => document.querySelector("
            "'#history-list .sidebar-item[data-category=\"running\"]'"
            ").offsetParent !== null"
        )
        assert visible0, "running row should be visible before any filter change"

        page.evaluate(
            "() => { const c = document.getElementById('hf-running');"
            " c.checked = false; c.dispatchEvent(new Event('change')); }"
        )
        hidden = page.evaluate(
            "() => document.querySelector("
            "'#history-list .sidebar-item[data-category=\"running\"]'"
            ").offsetParent === null"
        )
        assert hidden, "running row should hide when Running filter is unchecked"

        page.evaluate(
            "() => { const c = document.getElementById('hf-running');"
            " c.checked = true; c.dispatchEvent(new Event('change')); }"
        )
        visible1 = page.evaluate(
            "() => document.querySelector("
            "'#history-list .sidebar-item[data-category=\"running\"]'"
            ").offsetParent !== null"
        )
        assert visible1, (
            "running row should reappear when Running filter is re-checked"
        )
    finally:
        context.close()


def test_status_running_true_event_triggers_history_refresh(_browser) -> None:
    """A backend ``status: running=true`` event must trigger the
    frontend to refetch history.  The follow-up ``history`` reply
    with ``is_running=true`` must then make the pulsing green dot
    appear on the matching row WITHOUT a full page reload.
    """
    context, page = _open_history_page(_browser)
    try:
        finished_sample = dict(_sample_sessions()[2])
        finished_sample["task_id"] = 4001
        finished_sample["id"] = "chat-live"
        finished_sample["title"] = "live task"
        finished_sample["preview"] = "live task"
        _post_history(page, [finished_sample], expected_total=1)
        assert page.evaluate(
            "() => !document.querySelector("
            "'#history-list .sidebar-item .sidebar-item-running'"
            ")"
        ), "no green dot expected before the task starts running"

        page.evaluate(
            "() => { window.__postedMessages.length = 0; }"
        )
        page.evaluate(
            "() => window.__post({"
            " type: 'status', running: true,"
            " startTs: Date.now() - 5000"
            " })"
        )
        page.wait_for_function(
            "() => window.__postedMessages.some("
            "m => m && m.type === 'getHistory'"
            ")",
            timeout=5000,
        )
        generation = page.evaluate(
            "() => {"
            " const m = window.__postedMessages.slice().reverse().find("
            " x => x && x.type === 'getHistory'"
            " );"
            " return m ? m.generation : 0;"
            " }"
        )

        live = dict(finished_sample)
        live["is_running"] = True
        live["endTs"] = 0
        _post_history(
            page, [live], generation=int(generation), expected_total=1
        )
        assert page.evaluate(
            "() => !!document.querySelector("
            "'#history-list .sidebar-item .sidebar-item-running'"
            ")"
        ), "green dot must appear after status running=true refresh"

        page.evaluate("() => { window.__postedMessages.length = 0; }")
        page.evaluate(
            "() => window.__post({ type: 'status', running: false })"
        )
        page.wait_for_function(
            "() => window.__postedMessages.some("
            "m => m && m.type === 'getHistory'"
            ")",
            timeout=5000,
        )
        generation2 = page.evaluate(
            "() => {"
            " const m = window.__postedMessages.slice().reverse().find("
            " x => x && x.type === 'getHistory'"
            " );"
            " return m ? m.generation : 0;"
            " }"
        )
        finished = dict(finished_sample)
        finished["is_running"] = False
        finished["endTs"] = finished_sample["startTs"] + 1000
        _post_history(
            page, [finished], generation=int(generation2), expected_total=1
        )
        assert page.evaluate(
            "() => !document.querySelector("
            "'#history-list .sidebar-item .sidebar-item-running'"
            ")"
        ), "green dot must disappear after status running=false refresh"
    finally:
        context.close()


def test_running_dot_is_centered_middle_left_in_history_task_panel(
    _browser,
) -> None:
    """At a narrow viewport the running marker must remain at the
    middle-left of the task panel's first text line.

    The History sidebar renders each task as a multi-line panel with
    title, metrics, and workspace metadata.  The green running marker
    should stay on the left edge, vertically centered on the first line
    of the task title.  The action buttons occupy a line of their own
    below the title, so centering on the whole panel would drop the
    marker away from the text it belongs to.
    """
    context, page = _open_history_page(_browser, width=180, height=900)
    try:
        session = dict(_sample_sessions()[0])
        session["title"] = "a deliberately long running history task title"
        session["preview"] = session["title"]
        _post_history(page, [session])
        geometry = page.evaluate(
            """
            () => {
              const row = document.querySelector('#history-list .sidebar-item');
              const dot = row.querySelector('.sidebar-item-running');
              const text = row.querySelector('.sidebar-item-text');
              const rowRect = row.getBoundingClientRect();
              const dotRect = dot.getBoundingClientRect();
              const textRect = text.getBoundingClientRect();
              return {
                rowWidth: rowRect.width,
                firstLineMiddle: (() => {
                  const probe = document.createElement('span');
                  probe.textContent = 'x';
                  text.appendChild(probe);
                  const lineHeight = probe.getBoundingClientRect().height;
                  probe.remove();
                  return textRect.top + lineHeight / 2;
                })(),
                dotMiddle: dotRect.top + dotRect.height / 2,
                textLeft: textRect.left,
                dotLeft: dotRect.left,
              };
            }
            """
        )
        assert geometry["rowWidth"] < 170, geometry
        assert abs(geometry["dotMiddle"] - geometry["firstLineMiddle"]) <= 2, (
            geometry
        )
        assert geometry["dotLeft"] < geometry["textLeft"], geometry
    finally:
        context.close()


def test_search_results_can_render_running_green_circle(_browser) -> None:
    """Search-triggered history results must use the same running-dot
    rendering path as normal history loads."""
    context, page = _open_history_page(_browser)
    try:
        page.fill("#history-search", "running")
        page.wait_for_function(
            "() => window.__postedMessages.some("
            "m => m && m.type === 'getHistory' && m.query === 'running'"
            ")",
            timeout=5000,
        )
        search_generation = page.evaluate(
            "() => {"
            " const m = window.__postedMessages.slice().reverse().find("
            " x => x && x.type === 'getHistory' && x.query === 'running'"
            " );"
            " return m ? m.generation : 0;"
            " }"
        )
        _post_history(
            page, [_sample_sessions()[0]],
            generation=int(search_generation),
        )
        dot_visible = page.evaluate(
            """
            () => {
              const row = document.querySelector('#history-list .sidebar-item');
              const dot = row && row.querySelector('.sidebar-item-running');
              return !!row && row.offsetParent !== null &&
                row.querySelector('.sidebar-item-text').textContent ===
                  'running task' &&
                !!dot && dot.offsetParent !== null &&
                getComputedStyle(dot).backgroundColor === 'rgb(46, 125, 50)' &&
                getComputedStyle(dot).animationName ===
                  'running-pulse';
            }
            """
        )
        assert dot_visible, (
            "running search result did not show a visible green pulsing dot"
        )
    finally:
        context.close()


def test_paginated_history_batch_can_append_running_green_circle(
    _browser,
) -> None:
    """A running task arriving in an ``offset > 0`` pagination batch
    must still get a visible green circle when appended to existing
    rows."""
    context, page = _open_history_page(_browser)
    try:
        completed = _sample_sessions()[2]
        running = _sample_sessions()[0]
        _post_history(page, [completed], offset=0, expected_total=1)
        _post_history(page, [running], offset=1, expected_total=2)
        state = page.evaluate(
            """
            () => Array.from(
              document.querySelectorAll('#history-list .sidebar-item'),
            ).map(row => ({
              text: row.querySelector('.sidebar-item-text').textContent,
              category: row.dataset.category,
              hasRunningDot: !!row.querySelector('.sidebar-item-running'),
              visible: row.offsetParent !== null,
            }))
            """
        )
        assert state == [
            {
                "text": "successful task",
                "category": "completed",
                "hasRunningDot": False,
                "visible": True,
            },
            {
                "text": "running task",
                "category": "running",
                "hasRunningDot": True,
                "visible": True,
            },
        ]
    finally:
        context.close()


def test_backend_history_event_renders_green_circle_end_to_end(
    _browser,
) -> None:
    """Drive a real ``getHistory`` broadcast (with a synthetic alive
    worker thread) through the real frontend renderer and assert the
    green pulsing dot is visible on the row backed by the real DB."""
    context, page = _open_history_page(_browser)
    try:
        event = _history_event_from_real_backend(fake_running_task_id="-1")
        row = next(
            s for s in event["sessions"]
            if s["preview"] == "running task from real backend"
        )
        assert row["is_running"] is True, (
            "synthetic alive-thread fixture should produce is_running=True"
        )
        page.evaluate("event => window.__post(event)", event)
        page.wait_for_function(
            "() => document.querySelectorAll("
            "'#history-list .sidebar-item'"
            ").length >= 1",
            timeout=5000,
        )
        dot_visible = page.evaluate(
            """
            () => {
              const row = document.querySelector(
                '#history-list .sidebar-item[data-category="running"]',
              );
              const dot = row && row.querySelector('.sidebar-item-running');
              if (!row || !dot) return false;
              const cs = getComputedStyle(dot);
              return row.offsetParent !== null &&
                dot.offsetParent !== null &&
                cs.backgroundColor === 'rgb(46, 125, 50)' &&
                cs.animationName === 'running-pulse';
            }
            """
        )
        assert dot_visible, (
            "real backend → frontend pipeline did not render a visible "
            "green pulsing dot on the running row"
        )
    finally:
        context.close()

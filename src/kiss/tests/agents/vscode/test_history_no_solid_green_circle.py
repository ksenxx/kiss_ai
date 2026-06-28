# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test pinning the History sidebar's status-dot invariant:

    The History panel MUST NOT show a SOLID green circle on a row
    that was simply loaded as "completed" from the backend.  The
    solid green circle is reserved exclusively for the live
    running→completed transition the user just witnessed in the
    current page session.  Specifically:

      * A *running* row renders the PULSING green dot
        (``.sidebar-item-running`` with the ``sidebar-running-pulse``
        keyframe animation).
      * When a row that the current session previously rendered as
        running transitions to ``is_running:false`` and
        ``failed:false`` on a follow-up ``history`` event, its dot
        becomes SOLID green (``.sidebar-item-completed``, no
        animation) and STAYS solid green across subsequent
        ``refreshHistory()`` reloads.
      * Every other completed row — including all rows on the very
        first ``history`` event, search results, and pagination
        batches — renders NO dot at all.

The Playwright harness mirrors
``test_history_running_green_circle.py`` so the same review
checklist applies (visible offsetParent, real viewport, real CSS).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

_MEDIA_DIR = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
)
_CSS = _MEDIA_DIR / "main.css"
_JS = _MEDIA_DIR / "main.js"
_HTML = _MEDIA_DIR / "chat.html"


def _build_test_page() -> str:
    """Return a self-contained HTML page that loads the real CSS+JS."""
    css = _CSS.read_text(encoding="utf-8")
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
      --vscode-terminal-ansiBlue: #569cd6;
      --vscode-terminal-ansiMagenta: #c586c0;
      --vscode-terminal-ansiCyan: #4ec9b0;
    }}
    html, body {{ height: 100%; margin: 0; padding: 0; }}
  </style>
  <style>{css}</style>
  <title>history no solid green circle test</title>
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


def _make_session(
    *,
    task_id: int,
    title: str,
    is_running: bool = False,
    failed: bool = False,
    timestamp: int = 1700000000,
) -> dict:
    return {
        "id": f"chat-{task_id}",
        "task_id": task_id,
        "title": title,
        "preview": title,
        "timestamp": timestamp,
        "has_events": True,
        "failed": failed,
        "is_running": is_running,
        "tokens": 0,
        "cost": 0.0,
        "steps": 0,
        "is_favorite": False,
        "work_dir": "",
        "model": "",
        "is_worktree": False,
        "is_parallel": False,
        "auto_commit_mode": False,
        "startTs": timestamp * 1000,
        "endTs": 0 if is_running else timestamp * 1000 + 1000,
    }


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


def _row_states(page) -> list[dict]:
    """Return a per-row snapshot of which status dots are rendered."""
    result: list[dict] = page.evaluate(
        """
        () => Array.from(
          document.querySelectorAll('#history-list .sidebar-item'),
        ).map(r => ({
          text: (r.querySelector('.sidebar-item-text') || {}).textContent || '',
          category: r.dataset.category || '',
          hasRunning: !!r.querySelector('.sidebar-item-running'),
          hasCompleted: !!r.querySelector('.sidebar-item-completed'),
          hasFailed: !!r.querySelector('.sidebar-item-failed'),
          firstChildClass: r.firstElementChild
            ? r.firstElementChild.className : '',
        }))
        """
    )
    return result


# --- The invariant on a fresh load --------------------------------


def test_fresh_history_load_with_completed_row_renders_no_solid_green(
    _browser,
) -> None:
    """A fresh ``history`` event delivering a single completed row
    must render NO ``.sidebar-item-completed`` dot — and no
    ``.sidebar-item-running`` dot — on it.  This is the regression
    the user reported: simply opening History showed a solid green
    circle on every old, persisted, completed task.
    """
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, [
            _make_session(task_id=501, title="old completed task")
        ])
        rows = _row_states(page)
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["text"] == "old completed task"
        assert row["category"] == "completed"
        assert row["hasCompleted"] is False, (
            "fresh history load of a completed task MUST NOT render a "
            f"solid green circle; row state: {row}"
        )
        assert row["hasRunning"] is False, (
            "fresh history load of a completed task MUST NOT render a "
            f"pulsing green circle either; row state: {row}"
        )
        assert row["hasFailed"] is False, (
            "fresh history load of a completed task MUST NOT render a "
            f"red failed circle; row state: {row}"
        )
    finally:
        context.close()


def test_fresh_history_load_with_many_completed_rows_renders_no_dots(
    _browser,
) -> None:
    """Every row in a fresh batch of finished tasks must render with
    NO status dot — the same invariant, scaled up."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, [
            _make_session(task_id=600 + i, title=f"task {i}",
                          timestamp=1700000000 + i)
            for i in range(5)
        ])
        rows = _row_states(page)
        assert len(rows) == 5
        for row in rows:
            assert row["hasCompleted"] is False, (
                "no row in a fresh completed-history batch may render a "
                f"solid green circle; offending row: {row}"
            )
            assert row["hasRunning"] is False, row
            assert row["hasFailed"] is False, row
            # And the first child must NOT be one of the status-dot
            # classes — the title span is the first child.
            assert row["firstChildClass"] not in (
                "sidebar-item-running",
                "sidebar-item-completed",
                "sidebar-item-failed",
            ), row
    finally:
        context.close()


def test_failed_row_still_renders_red_circle(_browser) -> None:
    """The new invariant must NOT silence the red ``failed`` dot.
    Failed rows always render their red circle."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, [
            _make_session(task_id=701, title="failed task", failed=True)
        ])
        rows = _row_states(page)
        assert rows[0]["hasFailed"] is True, rows
        assert rows[0]["hasCompleted"] is False, rows
        assert rows[0]["hasRunning"] is False, rows
    finally:
        context.close()


# --- The live transition --------------------------------------------


def test_live_running_to_completed_transition_shows_solid_green(
    _browser,
) -> None:
    """A row first rendered as ``is_running:true`` and then
    re-rendered as ``is_running:false, failed:false`` (the live
    completion path the user just witnessed) MUST display the SOLID
    green ``.sidebar-item-completed`` dot.  The dot MUST stay solid
    green across a subsequent ``refreshHistory()`` reload.
    """
    context, page = _open_history_page(_browser)
    try:
        # Initial render: row is running → pulsing green dot.
        _post_history(page, [
            _make_session(task_id=801, title="live task", is_running=True)
        ])
        rows = _row_states(page)
        assert rows[0]["hasRunning"] is True, rows
        assert rows[0]["hasCompleted"] is False, rows

        # The follow-up event delivers the same task_id as finished.
        # The dot MUST swap from pulsing green to SOLID green.
        _post_history(page, [
            _make_session(task_id=801, title="live task", is_running=False)
        ])
        rows = _row_states(page)
        assert rows[0]["hasRunning"] is False, rows
        assert rows[0]["hasCompleted"] is True, (
            "after a live running→completed transition the row MUST "
            f"render the solid green circle; rows: {rows}"
        )
        # Verify the dot is genuinely solid (no pulse animation).
        anim = page.evaluate(
            "() => getComputedStyle("
            "document.querySelector('#history-list .sidebar-item-completed')"
            ").animationName"
        )
        assert "sidebar-running-pulse" not in (anim or ""), (
            "completed dot must NOT inherit the pulse animation; "
            f"got animation-name={anim!r}"
        )

        # A subsequent ``refreshHistory()`` reload — same task still
        # completed — MUST keep the solid green dot.
        _post_history(page, [
            _make_session(task_id=801, title="live task", is_running=False)
        ])
        rows = _row_states(page)
        assert rows[0]["hasCompleted"] is True, (
            "solid green circle MUST persist across subsequent history "
            f"refreshes once the task has completed; rows: {rows}"
        )
    finally:
        context.close()


def test_unrelated_completed_row_after_a_transition_still_has_no_dot(
    _browser,
) -> None:
    """Once one row in the session has transitioned to solid green,
    other unrelated completed rows arriving in the same session must
    still render with NO dot — the solid green stick only applies to
    task_ids the user actually saw running."""
    context, page = _open_history_page(_browser)
    try:
        # Witness a running→completed transition for task 901.
        _post_history(page, [
            _make_session(task_id=901, title="witnessed task", is_running=True)
        ])
        _post_history(page, [
            _make_session(task_id=901, title="witnessed task",
                          is_running=False)
        ])
        rows = _row_states(page)
        assert rows[0]["hasCompleted"] is True, rows

        # Now reload history with an additional, unrelated, completed
        # row that was never running in this session.
        _post_history(page, [
            _make_session(task_id=901, title="witnessed task",
                          is_running=False, timestamp=1700000100),
            _make_session(task_id=902, title="unrelated task",
                          is_running=False, timestamp=1700000000),
        ], expected_total=2)
        rows = _row_states(page)
        by_text = {r["text"]: r for r in rows}
        assert by_text["witnessed task"]["hasCompleted"] is True, by_text
        assert by_text["unrelated task"]["hasCompleted"] is False, (
            "an unrelated completed row arriving in the same session "
            "must NOT inherit the solid green circle from another "
            f"transitioned row; rows: {by_text}"
        )
        assert by_text["unrelated task"]["hasRunning"] is False
        assert by_text["unrelated task"]["hasFailed"] is False
    finally:
        context.close()


def test_no_solid_green_for_pagination_batch_of_completed_rows(
    _browser,
) -> None:
    """A pagination batch (``offset > 0``) of completed rows must
    not render any solid green dots — these are also "fresh" rows
    the user never saw running."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, [
            _make_session(task_id=1001, title="page1 row",
                          is_running=False, timestamp=1700000200)
        ], offset=0, expected_total=1)
        _post_history(page, [
            _make_session(task_id=1002, title="page2 row",
                          is_running=False, timestamp=1700000100)
        ], offset=1, expected_total=2)
        rows = _row_states(page)
        for row in rows:
            assert row["hasCompleted"] is False, (
                "pagination batch of completed rows must not introduce "
                f"a solid green circle; offending row: {row}"
            )
            assert row["hasRunning"] is False, row
    finally:
        context.close()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test of the Chats sidebar's green tick:

    Every task that finished successfully shows the green tick
    (``.sidebar-item-completed.status-tick``) on its row, as a failed
    task shows the red cross, whether or not the page saw the task
    run: on the first ``history`` page, on a later page, and after a
    live running→completed transition.  A chat's header shows the mark
    of the chat's last task.  A row carries exactly one mark: the
    spinner while the task runs, the cross once it failed, the tick
    otherwise, and the rows the tick marks are the rows the
    "Succeeded" filter counts.

The Playwright harness mirrors
``test_history_running_spinner.py`` so the same review
checklist applies (visible offsetParent, real viewport, real CSS).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

_MEDIA_DIR = (
    Path(__file__).resolve().parents[3]
    / "kiss"
    / "webapp"
    / "media"
)
_CSS = _MEDIA_DIR / "main.css"
_API_JS = _MEDIA_DIR / "api.js"
_PANEL_COPY_JS = _MEDIA_DIR / "panelCopy.js"
_JS = _MEDIA_DIR / "main.js"
_HTML = _MEDIA_DIR / "chat.html"


def _build_test_page() -> str:
    """Return a self-contained HTML page that loads the real CSS+JS."""
    css = _CSS.read_text(encoding="utf-8")
    api_js = _API_JS.read_text(encoding="utf-8")
    panel_copy_js = _PANEL_COPY_JS.read_text(encoding="utf-8")
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
      --theme-font-size: 13px;
      --theme-font-family: -apple-system, BlinkMacSystemFont,
        'Segoe UI', Roboto, sans-serif;
      --theme-editor-background: #1e1e1e;
      --theme-editor-foreground: #cccccc;
      --theme-input-background: #3c3c3c;
      --theme-input-foreground: #cccccc;
      --theme-input-border: #3c3c3c;
      --theme-sideBar-background: #252526;
      --theme-panel-border: #80808059;
      --theme-descriptionForeground: #8b8b8b;
      --theme-textLink-foreground: #3794ff;
      --theme-charts-red: #f44747;
      --theme-charts-green: #6a9955;
      --theme-charts-yellow: #d7ba7d;
      --theme-charts-purple: #c586c0;
      --theme-terminal-ansiCyan: #4ec9b0;
    }}
    html, body {{ height: 100%; margin: 0; padding: 0; }}
  </style>
  <style>{css}</style>
  <title>history green tick test</title>
</head>
<body>
{body}
  <script>
    window.__postedMessages = [];
    window.acquireTransport = function () {{
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
  <script>{panel_copy_js}</script>
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



_CHAT_MARKS_JS = """
() => Array.from(
  document.querySelectorAll('#history-list .history-chat-group'),
).map(group => {
  const row = group.querySelector('.sidebar-item');
  const header = group.querySelector(':scope > .history-chat-header');
  const mark = header.querySelector(':scope > .history-chat-status');
  const cs = getComputedStyle(mark);
  return {
    text: row.querySelector('.sidebar-item-text').textContent,
    rowMark: row.firstElementChild.className,
    headerMark: mark.className,
    shown: header.offsetParent !== null,
    box: [mark.offsetWidth, mark.offsetHeight],
    color: cs.backgroundColor,
    mask: cs.maskImage || cs.webkitMaskImage,
    animation: cs.animationName,
    label: mark.getAttribute('aria-label'),
    tooltip: mark.dataset.tooltip,
  };
})
"""


def _chat_marks(page) -> dict[str, dict]:
    """Return what each chat shows, by the title of its one task: the
    status mark of the task's row, and that of the chat's header with
    how it is drawn and whether the header is on screen."""
    return {chat["text"]: chat for chat in page.evaluate(_CHAT_MARKS_JS)}


def test_every_finished_task_shows_the_green_tick(_browser) -> None:
    """A task that finished before the page loaded shows the green
    tick on its row and on its chat's header, on the first page and on
    a later one; the tick is the cross in another glyph and colour,
    and it marks the rows the "Succeeded" filter counts."""
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, [
            _make_session(task_id=501, title="done 1", timestamp=1700000500),
            _make_session(task_id=502, title="broken", failed=True,
                          timestamp=1700000400),
            _make_session(task_id=503, title="busy", is_running=True,
                          timestamp=1700000300),
        ])
        _post_history(page, [
            _make_session(task_id=504, title="done 2", timestamp=1700000200),
        ], offset=3)
        chats = _chat_marks(page)
        tick = "sidebar-item-completed status-tick"
        assert {title: chat["rowMark"] for title, chat in chats.items()} == {
            "done 1": tick,
            "broken": "sidebar-item-failed status-cross",
            "busy": "sidebar-item-running status-spinner",
            "done 2": tick,
        }
        for title, chat in chats.items():
            assert chat["headerMark"] == chat["rowMark"] + " history-chat-status", title

        done, broken = chats["done 1"], chats["broken"]
        assert done["shown"], "the header that carries the tick is on screen"
        assert done["box"] == broken["box"] == [12, 12], (done["box"], broken["box"])
        assert "svg" in done["mask"] and done["mask"] != broken["mask"], done["mask"]
        status_ok = page.evaluate(
            "() => { const probe = document.createElement('i');"
            " probe.style.color = 'var(--status-ok)';"
            " document.body.appendChild(probe);"
            " const color = getComputedStyle(probe).color;"
            " probe.remove(); return color; }"
        )
        assert done["color"] == status_ok, (done["color"], status_ok)
        assert done["color"] != broken["color"]
        assert done["animation"] == "none", done["animation"]
        assert done["label"] == done["tooltip"] == "Task completed"

        page.evaluate(
            "() => { const c = document.getElementById('hf-completed');"
            " c.checked = false; c.dispatchEvent(new Event('change')); }"
        )
        hidden = {title for title, chat in _chat_marks(page).items() if not chat["shown"]}
        assert hidden == {"done 1", "done 2"}, (
            f"the Succeeded filter must hide exactly the ticked chats: {hidden}"
        )
    finally:
        context.close()


def test_failed_row_still_renders_red_cross(_browser) -> None:
    """A failed row renders its red cross and no tick."""
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



def test_live_running_to_completed_transition_shows_green_tick(
    _browser,
) -> None:
    """A row first rendered as ``is_running:true`` and then
    re-rendered as ``is_running:false, failed:false`` (a task that
    finishes while the page is open) swaps its spinner for the green
    ``.sidebar-item-completed`` tick, which does not spin and stays
    across a subsequent ``refreshHistory()`` reload.
    """
    context, page = _open_history_page(_browser)
    try:
        _post_history(page, [
            _make_session(task_id=801, title="live task", is_running=True)
        ])
        rows = _row_states(page)
        assert rows[0]["hasRunning"] is True, rows
        assert rows[0]["hasCompleted"] is False, rows

        _post_history(page, [
            _make_session(task_id=801, title="live task", is_running=False)
        ])
        rows = _row_states(page)
        assert rows[0]["hasRunning"] is False, rows
        assert rows[0]["hasCompleted"] is True, (
            "after a live running→completed transition the row MUST "
            f"render the green tick; rows: {rows}"
        )
        anim = page.evaluate(
            "() => getComputedStyle("
            "document.querySelector('#history-list .sidebar-item-completed')"
            ").animationName"
        )
        assert anim == "none", (
            "the completed tick must NOT inherit the spinner animation; "
            f"got animation-name={anim!r}"
        )

        _post_history(page, [
            _make_session(task_id=801, title="live task", is_running=False)
        ])
        rows = _row_states(page)
        assert rows[0]["hasCompleted"] is True, (
            "green tick MUST persist across subsequent history "
            f"refreshes once the task has completed; rows: {rows}"
        )
    finally:
        context.close()


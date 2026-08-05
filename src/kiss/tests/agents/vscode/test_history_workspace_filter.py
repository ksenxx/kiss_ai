# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the History panel's "Workspace" filter.

The chip is unchecked by default; when enabled it keeps only the tasks
that ran in the window's own workspace.  Two cases were wrong:

* A task that ran in a git worktree of the workspace
  (``<workspace>/.kiss-worktrees/kiss_wt-...``) — which is where the
  agent does most of its work — was treated as belonging to a different
  workspace and hidden from the project it belongs to.
* After ``./sorcar-cloud`` the remote deployment lives at another path
  (``/home/ubuntu/kiss`` instead of ``/Users/me/work/kiss``), so *every*
  imported task mismatched and the panel showed "No tasks match the
  filter" even though the whole database had arrived.  The paths are
  rewritten while shipping (``scripts/ship-task-db.sh``); this test
  covers the frontend contract that rewrite relies on.

The real ``media/main.js`` runs in headless Chromium against the real
``media/chat.html`` body, driven by the same ``configData`` and
``history`` events the webview receives from its host.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

_MEDIA_DIR = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media"
)


def _build_test_page() -> str:
    """Return a page that runs the real media CSS + JS, host APIs stubbed."""
    html = _HTML_TEMPLATE.format(
        css=(_MEDIA_DIR / "main.css").read_text(encoding="utf-8"),
        body=_chat_body(),
        api_js=(_MEDIA_DIR / "api.js").read_text(encoding="utf-8"),
        js=(_MEDIA_DIR / "main.js").read_text(encoding="utf-8"),
    )
    return html


def _chat_body() -> str:
    """Return ``chat.html``'s body with its host-provided scripts removed."""
    html = (_MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
    start = html.find(">", html.find("<body")) + 1
    body = html[start:html.find("</body>")]
    return "\n".join(
        line for line in body.splitlines()
        if "<script" not in line and "</script>" not in line
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<style>html, body {{ height: 100%; margin: 0; }}</style>
<style>{css}</style><title>workspace filter</title></head>
<body>
{body}
<script>
  window.acquireVsCodeApi = function () {{
    return {{ postMessage: function () {{}},
              setState: function () {{}},
              getState: function () {{ return null; }} }};
  }};
  window.hljs = {{ highlightElement: function () {{}},
                   highlightAll: function () {{}} }};
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
</body></html>
"""


def _session(task_id: int, title: str, work_dir: str) -> dict[str, object]:
    """Return one completed history entry as the server emits it."""
    return {
        "id": f"chat-{task_id}", "task_id": task_id, "title": title,
        "preview": title, "timestamp": 1700000000 + task_id,
        "has_events": True, "failed": False, "is_running": False,
        "tokens": 0, "cost": 0.0, "steps": 0, "is_favorite": False,
        "work_dir": work_dir, "model": "", "is_worktree": False,
        "is_parallel": False, "auto_commit_mode": False,
        "startTs": 0, "endTs": 0,
    }


@pytest.fixture(scope="module")
def browser():
    """Launch one headless Chromium for every test in the module."""
    with sync_playwright() as p:
        chromium = p.chromium.launch(headless=True)
        try:
            yield chromium
        finally:
            chromium.close()


def _visible_titles(browser, work_dir: str, sessions: list) -> list[str]:
    """Render *sessions* with the window's workspace set to *work_dir*.

    Returns the titles the History panel actually shows.
    """
    context = browser.new_context(viewport={"width": 520, "height": 1200})
    try:
        page = context.new_page()
        page.set_content(_build_test_page(), wait_until="load")
        page.wait_for_function(
            "document.getElementById('history-list') !== null", timeout=5000)
        error = page.evaluate("() => window.__iifeError")
        assert not error, f"main.js setup raised: {error}"
        page.evaluate(
            "() => { document.getElementById('app').style.display = '';"
            " const ov = document.getElementById('kiss-server-loading');"
            " if (ov) ov.style.display = 'none';"
            " document.getElementById('sidebar').classList.add('open'); }"
        )
        page.evaluate(
            "wd => window.__post({type: 'configData', config: {work_dir: wd}})",
            work_dir,
        )
        page.evaluate(
            "s => window.__post({type: 'history', sessions: s,"
            " offset: 0, generation: 0})",
            sessions,
        )
        page.wait_for_function(
            "n => document.querySelectorAll("
            "'#history-list .sidebar-item').length === n",
            arg=len(sessions), timeout=5000,
        )
        assert not page.is_checked("#hf-workspace"), \
            "the Workspace chip must be OFF by default"
        page.evaluate(
            "() => { const ws = document.getElementById('hf-workspace');"
            " ws.checked = true;"
            " ws.dispatchEvent(new Event('change', {bubbles: true})); }"
        )
        painted = page.evaluate(
            "() => Array.from(document.querySelectorAll("
            "'#history-list .sidebar-item'))"
            ".filter(r => r.offsetParent !== null)"
            ".map(r => r.querySelector('.sidebar-item-text').textContent)"
        )
        return [str(title) for title in painted]
    finally:
        context.close()


def test_worktree_tasks_belong_to_their_workspace(browser) -> None:
    """Tasks run in ``<workspace>/.kiss-worktrees/...`` stay visible."""
    titles = _visible_titles(browser, "/Users/me/work/kiss", [
        _session(1, "in the checkout", "/Users/me/work/kiss"),
        _session(2, "in a worktree",
                 "/Users/me/work/kiss/.kiss-worktrees/kiss_wt-17858-6062"),
        _session(3, "another project", "/Users/me/3rdparty/skydiscover"),
    ])
    assert "in the checkout" in titles
    assert "in a worktree" in titles
    assert "another project" not in titles


def test_a_sibling_directory_is_not_the_workspace(browser) -> None:
    """A shared name prefix is not a shared directory."""
    titles = _visible_titles(browser, "/Users/me/work/kiss", [
        _session(1, "the workspace", "/Users/me/work/kiss"),
        _session(2, "a lookalike", "/Users/me/work/kiss_ai"),
    ])
    assert titles == ["the workspace"]


def test_a_relocated_history_is_visible_on_the_remote(browser) -> None:
    """What ``scripts/ship-task-db.sh`` rewrites must show up remotely.

    The deployment runs from ``/home/ubuntu/kiss``; tasks shipped from
    the laptop arrive already re-pointed there, worktrees included.
    """
    titles = _visible_titles(browser, "/home/ubuntu/kiss", [
        _session(1, "shipped task", "/home/ubuntu/kiss"),
        _session(2, "shipped worktree task",
                 "/home/ubuntu/kiss/.kiss-worktrees/kiss_wt-17858-6062"),
    ])
    assert sorted(titles) == ["shipped task", "shipped worktree task"]


def test_unrewritten_paths_are_what_emptied_the_panel(browser) -> None:
    """The defect: laptop paths on a remote deployment match nothing."""
    titles = _visible_titles(browser, "/home/ubuntu/kiss", [
        _session(1, "laptop task", "/Users/me/work/kiss"),
        _session(2, "another laptop task", "/Users/me/work/kiss"),
    ])
    assert titles == []

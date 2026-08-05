# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM "hidden run_parallel panel" test.

The real assertions live in
``src/kiss/agents/vscode/test/runParallelNestedPanelCollapse.test.js``
(run under node, like ``runParallelPanelTabsSync.test.js``).  This
wrapper spawns ``node`` on that file so the test is also picked up by
``uv run pytest``.

Invariant under test (chat webview, ``media/main.js`` — shared verbatim
by the VS Code extension webview and the remote webapp)
--------------------------------------------------------------------
While a ``run_parallel`` tool event panel is COLLAPSED, every tab of the
sub-agents that tool created MUST be closed.

``runParallelPanelTabsSync.test.js`` covers the panel's own chevron.
This suite covers the panels that collapse a ``run_parallel`` panel by
swallowing it: the ``summary`` tool panel adopts the event panels that
precede it into a ``.summary-sub`` child and collapses itself, and
``.tc.collapsed > :not(.tc-h, .panel-copy-btn) {display: none}`` in
``media/main.css`` then hides the adopted fan-out panel.  Before the fix
those sub-agent tabs stayed open forever behind a panel the user could
no longer see — let alone reach the chevron of.

Behaviours verified by the underlying JS test:

* the ``summary`` tool adopting a live ``run_parallel`` panel collapses
  that panel and closes every sub-agent tab (``closeTab`` per tab);
* expanding the summary alone does NOT reopen the fan-out; expanding the
  nested ``run_parallel`` panel does, resuming each sub-agent task
  (``resumeSession``), and re-collapsing the summary closes them again;
* a sub-agent the daemon announces after the summary hid its fan-out
  panel opens no tab, and is opened when that panel is expanded;
* the task-end collapse pass and the background-tab collapse pass
  (``collapseAllExceptResult``) close the tabs of a summary-nested
  fan-out instead of skipping it;
* replaying a background chat's transcript (``task_events``) collapses
  the replacement ``run_parallel`` panel AND closes the sub-agent tabs
  of the panel it replaced -- the replacement panel is handed to the tab
  before the replay so the collapse pass can attribute it;
* closing a sub-agent tab takes the tabs of the fan-out that sub-agent
  ran itself with it, and those grandchildren are forgotten, so a later
  announcement cannot resurrect one behind a collapsed panel;
* throwing a chat's transcript away -- a new task's ``clear`` or a
  ``showWelcome`` reset, for the active tab and for a background tab --
  closes the sub-agent tabs of the fan-out panels it discards;
* a replay that collapses the fan-out whose sub-agent tab is ON SCREEN
  defers the closes until the transcript is complete, so the parent chat
  the user is moved to shows the whole replayed transcript;
* a replay that fails half way keeps the chat's previous transcript and
  leaves no close queued, so later collapses still close their tabs;
* a neighbouring task's replayed transcript (``adjacent_task_events``),
  whose own summary hides its own fan-out panel, leaves this
  conversation's live fan-out panel and sub-agent tabs untouched.

Every scenario runs twice: once against the VS Code extension host
(``acquireVsCodeApi`` stub) and once against the remote webapp — the
real ``_WS_SHIM_JS`` from ``src/kiss/server/web_server.py`` driven over a
stub WebSocket — because the invariant must hold on both surfaces.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "runParallelNestedPanelCollapse.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunParallelNestedPanelCollapse(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_parallel_nested_panel_collapse(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                "jsdom is not installed under "
                f"{_VSCODE_DIR/'node_modules'} — run `npm install` there"
            )
        self.assertTrue(
            _TEST_JS.is_file(),
            f"missing integration test file: {_TEST_JS}",
        )
        r = subprocess.run(
            ["node", str(_TEST_JS)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_VSCODE_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "runParallelNestedPanelCollapse.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

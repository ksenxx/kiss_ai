# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM run_parallel panel ⇔ sub-agent tabs test.

The real assertions live in
``src/kiss/agents/vscode/test/runParallelPanelTabsSync.test.js`` (run
under node, mirrors ``tab_timer_per_tab.test.js``).  This wrapper
spawns ``node`` on that file so the integration test is also picked up
by ``uv run pytest`` and shows up in CI alongside the rest of the
VS Code-extension Python tests.

Invariant under test (chat webview, ``media/main.js``)
------------------------------------------------------
* While a ``run_parallel`` tool-call panel is UNCOLLAPSED, the tabs of
  its sub-agents MUST be open.
* While a ``run_parallel`` tool-call panel is COLLAPSED, the tabs of
  its sub-agents MUST be closed.

Behaviours verified by the underlying JS test (see file for details):

* collapsing the run_parallel panel closes every sub-agent tab (and
  notifies the backend with ``closeTab`` per tab);
* re-expanding the panel reopens each sub-agent tab and resumes its
  backend task via ``resumeSession``;
* manually closing a sub-agent tab restores consistency (the owning
  panel collapses, closing the sibling sub-agent tabs);
* the automatic collapse passes (``collapseOlderPanels`` while
  streaming, ``collapseAllExceptResult`` at task end) never leave the
  panel collapsed while its sub-agent tabs are open;
* a delayed ``openSubagentTab`` cannot recreate a tab after collapsing
  the owning panel;
* replayed/persisted sub-agent tabs opened by ``openSubagentTab`` alone
  are associated with the owning run_parallel panel;
* sub-agents spawned while the panel is collapsed do not open tabs —
  their tabs open when the panel is expanded.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

# __file__ lives at src/kiss/tests/agents/vscode/<this file>.py — walk
# back to ``src/kiss`` (parents[3]) and step into ``agents/vscode``.
_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "runParallelPanelTabsSync.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunParallelPanelTabsSync(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_parallel_panel_tabs_sync(self) -> None:
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
                "runParallelPanelTabsSync.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

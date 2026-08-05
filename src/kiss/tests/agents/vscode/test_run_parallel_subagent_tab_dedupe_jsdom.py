# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM "one sub-agent, one tab" test.

The real assertions live in
``src/kiss/agents/vscode/test/runParallelSubagentTabDedupe.test.js``
(run under node, like ``runParallelPanelTabsSync.test.js``).  This
wrapper spawns ``node`` on that file so the test is also picked up by
``uv run pytest``.

Invariant under test (chat webview, ``media/main.js``)
------------------------------------------------------
A sub-agent occupies AT MOST ONE tab on a client, no matter how many
tab ids the daemon addresses it by.  A sub-agent is identified by its
task id; over one fan-out's life the daemon names the same sub-agent's
tab by

* the live fan-out id minted by
  ``ChatSorcarAgent._run_tasks_parallel`` (``task-<parentTaskId>__sub_<idx>``),
* the deterministic replay id minted by
  :meth:`kiss.server.server.VSCodeServer._open_persisted_subagent_tabs`
  (``<parentTabId>__sub_<subTaskId>``), and
* whatever tab id the webview itself asked to resume the sub-agent on
  (``media/main.js`` mints a fresh one when a collapsed ``run_parallel``
  panel is expanded again).

Behaviours verified by the underlying JS test:

* a persisted-replay announcement for a sub-agent that already has a
  tab renames that tab instead of opening a second one;
* the same holds when the replay burst lands after a collapse/expand
  cycle reopened the tabs under client-minted ids;
* a re-delivered ``new_tab`` for a sub-agent that already has a tab
  opens nothing;
* a panel that learned about one sub-agent twice (live tab + spawn
  recorded while collapsed) still opens exactly one tab per sub-agent
  when expanded;
* a collapsed ``run_parallel`` panel keeps every sub-agent tab closed
  whichever id form the daemon uses, and expanding it opens one tab per
  sub-agent;
* a sub-agent tab the user closed by hand is never resurrected by a
  later announcement under a different tab id (its siblings stay open
  and the panel stays uncollapsed);
* collapsing still sends ``closeTab`` for every sub-agent tab and
  expanding still sends ``resumeSession`` for every sub-agent task.

Every behaviour is asserted twice: once on the VS Code webview host and
once on the remote-webapp host — the latter runs the real ``_WS_SHIM_JS``
lifted out of :mod:`kiss.server.web_server` over a fake WebSocket
(including its ``auth``/``auth_ok`` handshake), because the extension
and the webapp must behave identically.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "runParallelSubagentTabDedupe.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunParallelSubagentTabDedupe(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_parallel_subagent_tab_dedupe(self) -> None:
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
            timeout=180,
            cwd=str(_VSCODE_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "runParallelSubagentTabDedupe.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM "one sub-agent, one tab" test.

The real assertions live in
``src/kiss/webapp/test/runParallelSubagentTabDedupe.test.js``
(run under node).  This wrapper spawns ``node`` on that file so the
test is also picked up by ``uv run pytest``.

Invariant under test (chat page, ``media/main.js``)
------------------------------------------------------
A sub-agent occupies AT MOST ONE tab on a client, no matter how many
tab ids the daemon addresses it by.  A sub-agent is identified by its
task id; over one fan-out's life the daemon may name the same
sub-agent's tab by

* a live id that differs from the tab's current one
  (``task-<parentTaskId>__sub_<idx>``),
* the deterministic replay id minted by
  :meth:`kiss.server.server.AgentServer._open_persisted_subagent_tabs`
  (``<parentTabId>__sub_<subTaskId>``), and
* the tab id the page itself resumed the sub-agent on (the same
  deterministic id, minted when the spawn is announced).

Behaviours verified by the underlying JS test:

* a replay's announcement of a sub-agent that already has a tab lands
  on that tab, and one under another id renames it instead of opening
  a second one (the retired id is released; the tabs and files below
  follow the rename);
* a re-delivered ``new_tab`` for a sub-agent that already has a tab
  opens nothing, also when it is mixed with the daemon's own
  announcement of the same sub-agent;
* a finished sub-agent has no tab, whichever id form the daemon uses;
* the sub-agent of a parent this page does not show opens no tab.

The page runs on the real webapp transport (``media/transport.js``
over a fake WebSocket, including its ``auth``/``auth_ok`` handshake).
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[2]
_WEBAPP_DIR = _KISS_ROOT / "webapp"
_TEST_JS = _WEBAPP_DIR / "test" / "runParallelSubagentTabDedupe.test.js"
_JSDOM_PKG = _WEBAPP_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunParallelSubagentTabDedupe(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_parallel_subagent_tab_dedupe(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                "jsdom is not installed under "
                f"{_WEBAPP_DIR/'node_modules'} — run `npm install` there"
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
            cwd=str(_WEBAPP_DIR),
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

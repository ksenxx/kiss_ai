# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM run_agent / run_parallel tab-parity test.

The assertions live in
``src/kiss/webapp/test/runAgentSubagentTabParity.test.js`` (run
under node); this wrapper spawns ``node`` on that file so the
integration test is also picked up by ``uv run pytest``.

Invariant under test (chat page, ``media/main.js``)
------------------------------------------------------
A ``run_agent`` dispatch's sub-agent tab lives and dies exactly like a
``run_parallel`` sub-task's: the tab opens when the child starts and
``subagentDone`` closes it, a client that connects while the parent
runs shows the running child's tab, and a replay of the finished
parent opens no tab for a finished child.  The page ties no tab to a
tool-call panel; the share export puts each child under the panel of
the call that was running when the child started.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[2]
_WEBAPP_DIR = _KISS_ROOT / "webapp"
_TEST_JS = _WEBAPP_DIR / "test" / "runAgentSubagentTabParity.test.js"
_JSDOM_PKG = _WEBAPP_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunAgentSubagentTabParity(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_agent_subagent_tab_parity(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                "jsdom is not installed under "
                f"{_WEBAPP_DIR / 'node_modules'} — run `npm install` there"
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
            cwd=str(_WEBAPP_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "runAgentSubagentTabParity.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM run_agent / run_parallel tab-parity test.

The assertions live in
``src/kiss/agents/vscode/test/runAgentSubagentTabParity.test.js`` (run
under node); this wrapper spawns ``node`` on that file so the
integration test is also picked up by ``uv run pytest``.

Invariant under test (chat webview, ``media/main.js``)
------------------------------------------------------
A ``run_agent`` dispatch's sub-agent tab behaves exactly like a
``run_parallel`` sub-task's: ``subagentDone`` closes it, the tool-call
panel collapses once the fan-out is complete (for ``run_agent`` when
its ``tool_result`` arrives, since a dispatch may spawn none, one, or
several children in sequence), the daemon's re-announce of the finished
child on every parent replay (``openSubagentTab {isDone: true,
startTs}``, sent on each webview ``ready``) is attributed to the call
that was running when the child started and opens no tab, and
expanding the panel is what brings the child's transcript back.  Before
the fix the ``run_agent`` panel was not a fan-out panel, so every
reconnect re-opened the finished child's tab — the tab looked like it
never closed.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "runAgentSubagentTabParity.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunAgentSubagentTabParity(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_agent_subagent_tab_parity(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                "jsdom is not installed under "
                f"{_VSCODE_DIR / 'node_modules'} — run `npm install` there"
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
                "runAgentSubagentTabParity.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

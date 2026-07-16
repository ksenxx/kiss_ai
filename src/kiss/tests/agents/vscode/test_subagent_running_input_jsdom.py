# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM sub-agent running-input test.

The real assertions live in
``src/kiss/agents/vscode/test/subagentRunningInput.test.js`` (run
under node, mirrors ``subagentTabAutoCloseOnDone.test.js``).  This
wrapper spawns ``node`` on that file so the integration test is also
picked up by ``uv run pytest`` and shows up in CI alongside the rest
of the VS Code-extension Python tests.

Invariant under test (chat webview, ``media/main.js``)
------------------------------------------------------
A sub-agent chat tab must show the input textbox and the buttons
below it (``#input-container``) WHILE its sub-agent task is RUNNING —
so the user can inject follow-up prompts into the running sub-agent
(``appendUserMessage`` posted with the sub-agent's tab id) and stop
ONLY that sub-agent's task (``stop`` posted with the sub-agent's tab
id) — and must remove the input as soon as the sub-agent task
completes (``subagentDone`` / ``openSubagentTab`` with ``isDone``).
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
_TEST_JS = _VSCODE_DIR / "test" / "subagentRunningInput.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestSubagentRunningInput(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_subagent_running_input(self) -> None:
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
                "subagentRunningInput.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

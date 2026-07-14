# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM-based history ids copy-button test.

The real assertions live in
``src/kiss/agents/vscode/test/historyIdsCopyButtons.test.js`` (run
under node, mirrors ``historyTaskIds.test.js``).  This wrapper spawns
``node`` on that file so the integration test is also picked up by
``uv run pytest`` and shows up in CI alongside the rest of the
VS Code-extension Python tests.

The test exercises the copy buttons rendered next to the chat id and
task id inside the per-task ``.running-item-ids`` line of the History
sidebar: clicking the chat button copies exactly the raw chat id to
the clipboard, clicking the task button copies exactly the raw task
id, neither click opens the row's chat tab, the buttons flash a
check-mark confirmation, and a ``document.execCommand('copy')``
fallback is used when the async clipboard API is unavailable.
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
_TEST_JS = _VSCODE_DIR / "test" / "historyIdsCopyButtons.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestHistoryIdsCopyButtonsIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_history_ids_copy_buttons(self) -> None:
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
            timeout=60,
            cwd=str(_VSCODE_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "historyIdsCopyButtons.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

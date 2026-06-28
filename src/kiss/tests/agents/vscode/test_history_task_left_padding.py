# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM-based History sidebar
"reduced left whitespace" integration test.

The real assertions live in
``src/kiss/agents/vscode/test/historyTaskLeftPadding.test.js``
(run under node, mirrors ``historyTaskIds.test.js``).  This
wrapper spawns ``node`` on that file so the integration test is
also picked up by ``uv run pytest`` and shows up in CI alongside
the rest of the VS Code-extension Python tests.

The test exercises the CSS reduction of the left-whitespace
column reserved for each History row's status indicator:

* ``.running-item`` ``padding-left`` must be halved from 26px to
  13px.
* The compound ``.running-item > .sidebar-item-{failed,running,
  completed}`` rule must halve ``left`` from 10px to 5px so the
  8px status dot still sits inside the now-narrower padding
  column.
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
_TEST_JS = _VSCODE_DIR / "test" / "historyTaskLeftPadding.test.js"


class TestHistoryTaskLeftPaddingIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_history_task_left_padding_halved(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
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
                "historyTaskLeftPadding.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

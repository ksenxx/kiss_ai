# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM-based history filter date-group test.

The real assertions live in
``src/kiss/agents/vscode/test/historyFilterDateGroup.test.js`` (run
under node, mirrors ``historyTaskMeta.test.js``).  This wrapper
spawns ``node`` on that file so the integration test is also picked
up by ``uv run pytest`` and shows up in CI alongside the rest of the
VS Code-extension Python tests.

The test enforces the invariant that the "From" label, the From
date textbox (``#hf-from``) and the From calendar-picker button
(``#hf-from-btn``) NEVER split across multiple visual lines (and
likewise for the "To" trio).  Both trios live inside a single
``.history-filter-date-group`` wrapper whose CSS guarantees the
three pieces stay glued on one line.
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
_TEST_JS = _VSCODE_DIR / "test" / "historyFilterDateGroup.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestHistoryFilterDateGroupIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_history_filter_date_group(self) -> None:
        """From/To label+input+button must stay on a single line."""
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
                "historyFilterDateGroup.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

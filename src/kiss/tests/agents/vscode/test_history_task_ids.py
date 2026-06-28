# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM-based history task-ids integration
test.

The real assertions live in
``src/kiss/agents/vscode/test/historyTaskIds.test.js`` (run under
node, mirrors ``historyTaskMeta.test.js``).  This wrapper spawns
``node`` on that file so the integration test is also picked up by
``uv run pytest`` and shows up in CI alongside the rest of the
VS Code-extension Python tests.

The test exercises the per-task chat/task/parent-ids line in the
History sidebar: every row whose persisted record carries one or
more of ``id`` (chat id), ``task_id``, or ``parent_task_id`` must
render a ``.running-item-ids`` span as a third dot-separated line
right under the workspace+meta line::

    chat <chat_id> • task <task_id> • parent <parent_task_id>

The three lines (metrics, workspace+meta, ids) must sit inside a
single ``.running-item-info`` container that eliminates the
vertical gap between them — they render flush, with no row-gap
between them.

Rows with none of the three ids render no ``.running-item-ids``
span at all.
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
_TEST_JS = _VSCODE_DIR / "test" / "historyTaskIds.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestHistoryTaskIdsIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_history_task_ids_line(self) -> None:
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
                "historyTaskIds.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM-based history task-meta integration
test.

The real assertions live in
``src/kiss/webapp/test/historyTaskMeta.test.js`` (run under
node).  This wrapper spawns ``node`` on that file so the
integration test is also picked up by ``uv run pytest`` and shows
up in CI alongside the rest of the webapp Python tests.

The test exercises the per-task workspace+metadata line in the
History sidebar: every row with a non-empty ``work_dir`` renders a
``.running-item-workspace`` span right after the
``.running-item-metrics`` span, and every row whose persisted
``extra`` carries a ``model`` appends the model name and the three
run flags to that span, separated by ``•``, so everything renders
on the SAME visual line as the workspace path::

    <work_dir> • <model> • <wt|no-wt>
        • <parallel|sequential> • <auto-commit|manual-commit>

Rows whose persisted ``model`` is missing render the workspace
alone (legacy behaviour); rows with neither workspace nor model
render no ``.running-item-workspace`` span at all.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[2]
_WEBAPP_DIR = _KISS_ROOT / "webapp"
_TEST_JS = _WEBAPP_DIR / "test" / "historyTaskMeta.test.js"
_JSDOM_PKG = _WEBAPP_DIR / "node_modules" / "jsdom" / "package.json"


class TestHistoryTaskMetaIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_history_task_meta_line(self) -> None:
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
            timeout=60,
            cwd=str(_WEBAPP_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "historyTaskMeta.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM multi-call + nested run_parallel test.

The real assertions live in
``src/kiss/webapp/test/runParallelMultiCallNested.test.js`` (run
under node).  This wrapper spawns ``node`` on that file so the
integration test is also picked up by ``uv run pytest`` and shows up in
CI alongside the rest of the webapp Python tests.

Invariant under test (chat page, ``media/main.js``)
-----------------------------------------------------------------------
* EVERY ``run_parallel`` call made by an agent OR a sub-agent opens one
  tab per spawned sub-agent — irrespective of how many run_parallel
  calls that agent/sub-agent already made (verified with three
  sequential calls by the root agent, three by a sub-agent, and a
  3-level-deep nesting of run_parallel calls), also under a parent tab
  that has no chat DOM yet and under one that has already closed (the
  nearest open ancestor then takes the tab).
* A tab closes when its own sub-agent finishes (``subagentDone``) and
  only then: a finished ancestor leaves the tabs of the sub-agents
  still running below it, and folding or unfolding a fan-out panel,
  live or in adjacent history, opens and closes nothing.
* A sub-agent announced as finished gets no tab.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[2]
_WEBAPP_DIR = _KISS_ROOT / "webapp"
_TEST_JS = _WEBAPP_DIR / "test" / "runParallelMultiCallNested.test.js"
_JSDOM_PKG = _WEBAPP_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunParallelMultiCallNested(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_parallel_multi_call_nested(self) -> None:
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
            timeout=120,
            cwd=str(_WEBAPP_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "runParallelMultiCallNested.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

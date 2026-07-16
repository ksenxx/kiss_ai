# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM multi-call + nested run_parallel test.

The real assertions live in
``src/kiss/agents/vscode/test/runParallelMultiCallNested.test.js`` (run
under node, mirrors ``runParallelPanelTabsSync.test.js``).  This
wrapper spawns ``node`` on that file so the integration test is also
picked up by ``uv run pytest`` and shows up in CI alongside the rest of
the VS Code-extension Python tests.

Invariant under test (chat webview, ``media/main.js`` — shared by the
VS Code extension and the remote web app)
-----------------------------------------------------------------------
* EVERY ``run_parallel`` call made by an agent OR a sub-agent opens one
  tab per spawned sub-agent — irrespective of how many run_parallel
  calls that agent/sub-agent already made (verified with three
  sequential calls by the root agent, three by a sub-agent, and a
  3-level-deep nesting of run_parallel calls).
* When a sub-agent finishes (``subagentDone``), ONLY the corresponding
  tab closes — at every nesting level.
* Collapsing a run_parallel panel (by the user clicking its header or
  by the agent's automatic collapse passes) closes the tabs of ALL
  sub-agents spawned by THAT call — and never the tabs spawned by a
  DIFFERENT run_parallel call of the same agent.
* Uncollapsing a run_parallel panel reopens the tabs of ALL sub-agents
  spawned by THAT call (via ``resumeSession``) — and only that call.
* A run_parallel panel replayed inside an adjacent-task history block
  (whose parent tab belongs to a long-gone session) is inert.
* A sub-agent spawned under a parent tab that has no chat DOM yet
  still gets its own tab; a late-rendered run_parallel panel adopts
  such unregistered tabs so its collapse closes them.
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
_TEST_JS = _VSCODE_DIR / "test" / "runParallelMultiCallNested.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestRunParallelMultiCallNested(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_run_parallel_multi_call_nested(self) -> None:
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
                "runParallelMultiCallNested.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

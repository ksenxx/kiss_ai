# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM sub-agent live-stream test.

The real assertions live in
``src/kiss/agents/vscode/test/subagentLiveEventsStream.test.js``
(run under node, like ``runParallelSubagentTabDedupe.test.js``).  This
wrapper spawns ``node`` on that file so the test is also picked up by
``uv run pytest``.

Invariant under test (chat webview, ``media/main.js``)
------------------------------------------------------
A freshly spawned sub-agent's tab must show the sub-agent's events:

* live events fanned out to the tab (stamped with its tab id and the
  sub-agent's task id) render into its transcript whether the tab is
  in the background or on screen;
* the sub-agent's output never leaks into the parent's transcript;
* a ``task_events`` replay (the transcript head) and later live events
  (the tail) are both visible.

Every behaviour is asserted twice: once on the VS Code webview host and
once on the remote-webapp host — the latter runs the real ``_WS_SHIM_JS``
lifted out of :mod:`kiss.server.web_server` over a fake WebSocket,
because the extension and the webapp must behave identically.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "subagentLiveEventsStream.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestSubagentLiveEventsStream(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_subagent_live_events_stream(self) -> None:
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
            timeout=180,
            cwd=str(_VSCODE_DIR),
        )
        if r.returncode != 0:
            self.fail(
                "subagentLiveEventsStream.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

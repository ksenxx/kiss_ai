# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM eager-Thoughts-panel integration test.

The real assertions live in
``src/kiss/agents/vscode/test/thoughtPanelEagerToolResult.test.js``
(run under node, mirrors ``panelTimeSpent.test.js``).  This wrapper
spawns ``node`` on that file so the integration test is also picked up
by ``uv run pytest`` and shows up in CI alongside the rest of the
VS Code-extension Python tests.

Feature under test (chat webview, ``media/main.js``)
----------------------------------------------------
As soon as the agent has finished a tool call and got its result
(``tool_result``) — before the tool response / queued user message is
sent back to the model — the chat webview must immediately append a
new "Thoughts" ``.llm-panel`` with a live ticking ``.panel-time``
footer (like the other chat panels).  Thought tokens streamed later by
the model land inside that SAME panel.

Behaviours verified by the underlying JS test (see file for details):

* the Thoughts panel appears eagerly right after ``tool_result``, with
  a ``.panel-time`` footer that keeps ticking while waiting for the
  model, and later thinking deltas stream into that same panel with the
  step count advancing exactly once;
* the eager panel is provisional: a following ``tool_call`` with no
  thinking/text in between (parallel tool calls) removes the empty
  panel again without touching the step count;
* ``text_delta``-only model turns also fill the eager panel;
* no eager panel is created after the ``finish`` tool's result;
* the same eager/provisional behavior applies to background tabs, and
  the eager panel survives switching tabs away and back.
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
_TEST_JS = _VSCODE_DIR / "test" / "thoughtPanelEagerToolResult.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestThoughtPanelEagerToolResult(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_thought_panel_eager_tool_result(self) -> None:
        """Node JSDOM test for the eager Thoughts panel must pass."""
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                "jsdom is not installed under "
                f"{_VSCODE_DIR / 'node_modules'} — run `npm install` there"
            )
        self.assertTrue(
            _TEST_JS.is_file(),
            f"missing JS test file: {_TEST_JS}",
        )
        proc = subprocess.run(
            ["node", str(_TEST_JS)],
            cwd=str(_VSCODE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            proc.returncode,
            0,
            "thoughtPanelEagerToolResult.test.js failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        self.assertIn("All tests passed", proc.stdout)


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM "Suggested next" click-to-copy test.

The real assertions live in
``src/kiss/agents/vscode/test/followupSuggestionClickCopies.test.js``
(run under node against the real ``media/main.js`` in a JSDOM webview).
This wrapper spawns ``node`` on that file so the integration test is
picked up by ``uv run pytest`` alongside the rest of the VS Code
extension Python tests.

Bug under test (chat webview, ``media/main.js``)
------------------------------------------------
Clicking a "Suggested next" bar must copy the suggested prompt into the
chat input box.  After a chat webview reload only the newest task's
transcript is replayed through ``replayTaskEvents`` (whose bars were
wired); earlier tasks of the chat are spliced in on overscroll through
``adjacent_task_events`` -> ``replayDetachedTranscript``, which passed
no ``onFollowupClick`` — those bars rendered but clicking them silently
did nothing.

The JS test verifies the click-to-copy invariant on every surface the
bar (or the welcome "Suggested prompt" chip) is rendered on:

* the live ``followup_suggestion`` event of the running task;
* the active tab's ``task_events`` replay after a reload;
* a background tab's replay, after switching to that tab;
* an earlier task spliced in via ``adjacent_task_events`` (the bug);
* the welcome screen's suggestion chips.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "followupSuggestionClickCopies.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestFollowupClickCopies(unittest.TestCase):
    """Drive the JSDOM click-to-copy integration test from pytest."""

    def test_followup_suggestion_click_copies_to_input(self) -> None:
        """Node JSDOM test for the Suggested-next click must pass."""
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
            "followupSuggestionClickCopies.test.js failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        self.assertIn("All tests passed", proc.stdout)


if __name__ == "__main__":
    unittest.main()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM-based history-click integration test.

The real assertions live in
``src/kiss/agents/vscode/test/historyClickNoInputCopy.test.js`` (run
under node, mirrors ``historyClickSwitchExistingChat.test.js``).  This
wrapper spawns ``node`` on that file so the integration test is also
picked up by ``uv run pytest`` and shows up in CI alongside the rest of
the VS Code-extension Python tests.

The test exercises what a click on a Task History row must NOT do:
copy the historical task text into the chat input textarea
(``#task-input``).  That textarea holds the user's draft for the next
prompt, so overwriting it both destroys typed text and invites the
user to accidentally re-send an old task.  Echoing the task into the
read-only task panel (``#task-panel-text``, via ``setTaskText``) is the
correct behaviour and is asserted to keep working.

All three branches of the row click handler are covered — resume
(``has_events``), plain new tab (no events) and switch-to-already-open
tab — plus draft preservation and the ``setTaskText`` backend message.
Because ``src/kiss/server/web_server.py`` serves the very same
``media/chat.html`` + ``media/main.js`` to the remote web app, every
scenario runs twice: once as the extension webview and once as the
remote webview (``<body class="remote-chat">``).
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "historyClickNoInputCopy.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestHistoryClickNoInputCopyIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_history_click_does_not_fill_input(self) -> None:
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
                "historyClickNoInputCopy.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

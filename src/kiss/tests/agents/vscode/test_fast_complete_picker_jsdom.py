# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM-based fast-complete picker test.

The real assertions live in
``src/kiss/agents/vscode/test/fast_complete_picker.test.js`` (run
under node, mirrors ``historyTaskMeta.test.js``).  This wrapper
spawns ``node`` on that file so the integration test is also picked
up by ``uv run pytest`` and shows up in CI alongside the rest of
the VS Code-extension Python tests.

The test exercises the dropdown picker that the chat input
textbox uses to surface fast-complete suggestions while the user
types — the same DOM that the ``@``-mention file picker reuses,
but driven by the ``completions`` message from the backend's
:class:`kiss.agents.vscode.autocomplete.Autocomplete` worker.

Behaviours verified by the underlying JS test (see file for
details):

* the ``completions`` message renders one row per suggestion with
  the correct icon (lightning for ``frequent``, spark for
  ``identifier``, ``</>`` for ``snippet``);
* the picker is suppressed during ``@``-mentions, when the cursor
  is not at end-of-input, and when the input is empty;
* stale replies (``ev.query !== inp.value``) are dropped;
* Tab/Enter/Click accepts the highlighted suggestion in-place,
  preserving any trailing whitespace already typed;
* ArrowDown moves the selection without losing focus;
* Escape dismisses the picker;
* the legacy ``ghost`` handler is unaffected.
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
_TEST_JS = _VSCODE_DIR / "test" / "fast_complete_picker.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestFastCompletePickerIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_fast_complete_picker(self) -> None:
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
                "fast_complete_picker.test.js failed "
                f"(rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n"
                f"--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

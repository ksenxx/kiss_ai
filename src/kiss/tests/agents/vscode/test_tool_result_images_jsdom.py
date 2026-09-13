# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM tool-result-images integration test.

The real assertions live in
``src/kiss/agents/vscode/test/toolResultImages.test.js`` (run under
node, mirrors ``thoughtPanelEagerToolResult.test.js``).  This wrapper
spawns ``node`` on that file so the integration test is also picked up
by ``uv run pytest`` alongside the rest of the VS Code-extension
Python tests.

Feature under test (chat webview, ``media/main.js``)
----------------------------------------------------
When a tool call generates an image — a browser ``screenshot``, a plot
written by a Bash command, an SVG written by ``Write`` — the server
embeds the image bytes as base64 ``images`` payloads on the
``tool_result`` event (``JsonPrinter._collect_result_images``), and
the webview must render them inline in the corresponding event panel.

Behaviours verified by the underlying JS test (see file for details):

* the image renders as a ``data:`` ``<img>`` inside the tool-call
  panel, after the textual output panel, with its path as caption and
  tooltip, and click-to-zoom toggles the full-size class both ways;
* images render on the streamed-bash path too (where the textual
  result panel is suppressed because the output already streamed);
* image-less results render no ``.tr-images`` wrapper, entries missing
  their base64 payload are skipped, and a path-less image falls back
  to a generic alt text with no caption.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[3]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TEST_JS = _VSCODE_DIR / "test" / "toolResultImages.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"


class TestToolResultImagesJsdom(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_tool_result_images(self) -> None:
        """Node JSDOM test for inline tool-result images must pass."""
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
            "toolResultImages.test.js failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        self.assertIn("All tests passed", proc.stdout)


if __name__ == "__main__":
    unittest.main()

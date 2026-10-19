# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pytest wrapper for the JSDOM test of the CHATS panel's Cron chip.

The assertions live in ``src/kiss/webapp/test/historyCronChip.test.js``:
the chip is off by default, the rows the daemon marks ``is_cron`` are
hidden until it is switched on, and the other chips keep applying to
them.  This wrapper runs that file under node so ``uv run pytest``
picks it up with the rest of the webapp tests.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

_KISS_ROOT = Path(__file__).resolve().parents[2]
_WEBAPP_DIR = _KISS_ROOT / "webapp"
_TEST_JS = _WEBAPP_DIR / "test" / "historyCronChip.test.js"
_JSDOM_PKG = _WEBAPP_DIR / "node_modules" / "jsdom" / "package.json"


class TestHistoryCronChipIntegration(unittest.TestCase):
    """Drive the JSDOM integration test from pytest."""

    def test_history_cron_chip(self) -> None:
        """Cron rows hide by default and show only while the chip is on."""
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                "jsdom is not installed under "
                f"{_WEBAPP_DIR / 'node_modules'} — run `npm install` there"
            )
        self.assertTrue(_TEST_JS.is_file(), f"missing test file: {_TEST_JS}")
        r = subprocess.run(
            ["node", str(_TEST_JS)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_WEBAPP_DIR),
        )
        if r.returncode != 0:
            self.fail(
                f"historyCronChip.test.js failed (rc={r.returncode})\n"
                f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
            )


if __name__ == "__main__":
    unittest.main()

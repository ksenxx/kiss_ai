"""Test that createNewTab stops the timer when the new tab is not running.

Bug: When a user opens a new chat while a task is running on the current
tab, the timer interval from the old tab kept firing.  Since the new tab
has t0 = null, ``Date.now() - null`` evaluates to ``Date.now()`` (epoch
milliseconds), causing the status to show "Running 29000000m …".

Fix: ``createNewTab`` now includes the same ``if (!tab.isRunning) { t0 =
null; stopTimer(); removeSpinner(); }`` guard that ``switchToTab`` and
``closeTab`` already had.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


class TestCreateNewTabStopsTimer(unittest.TestCase):
    """Structural assertion: createNewTab calls stopTimer for non-running tabs."""

    def test_create_new_tab_stops_timer_when_not_running(self) -> None:
        source = MAIN_JS.read_text()
        # Extract the createNewTab function body
        m = re.search(r"function createNewTab\(\)\s*\{", source)
        assert m is not None, "createNewTab not found in main.js"
        start = m.start()
        # Find matching closing brace
        depth = 0
        body_start = source.index("{", start)
        i = body_start
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = source[body_start : i + 1]
        # The body must contain the stopTimer guard after setRunningState
        self.assertIn("stopTimer()", body, "createNewTab must call stopTimer()")
        self.assertIn("t0 = null", body, "createNewTab must reset t0 to null")
        self.assertIn(
            "removeSpinner()", body, "createNewTab must call removeSpinner()"
        )

    def test_pattern_matches_switch_to_tab_and_close_tab(self) -> None:
        """All three tab-change functions must have the stopTimer guard."""
        source = MAIN_JS.read_text()
        for fn_name in ("createNewTab", "switchToTab", "closeTab"):
            m = re.search(rf"function {fn_name}\(", source)
            assert m is not None, f"{fn_name} not found"
            start = m.start()
            depth = 0
            body_start = source.index("{", start)
            i = body_start
            while i < len(source):
                if source[i] == "{":
                    depth += 1
                elif source[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            body = source[body_start : i + 1]
            self.assertIn(
                "stopTimer()",
                body,
                f"{fn_name} must call stopTimer()",
            )


if __name__ == "__main__":
    unittest.main()

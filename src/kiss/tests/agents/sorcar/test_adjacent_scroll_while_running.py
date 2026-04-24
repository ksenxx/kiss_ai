"""Tests that adjacent task scrolling works while a task is running.

Bug: the wheel-event handler in main.js gates the entire adjacent-scroll
logic on ``!isRunning``, which means the user cannot scroll to
previous/next tasks when the active tab has a running task.

The fix removes the ``!isRunning`` guard from the adjacent-scroll
overscroll-detection block so that scrolling to adjacent tasks is
independent of the tab's running state.
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


class TestAdjacentScrollWhileRunning(unittest.TestCase):
    """Structural assertions on the wheel-event adjacent-scroll guard."""

    def setUp(self) -> None:
        self.src = MAIN_JS.read_text()

    # ------------------------------------------------------------------
    # Structural: the overscroll guard must NOT block on isRunning
    # ------------------------------------------------------------------

    def test_overscroll_guard_does_not_check_is_running(self) -> None:
        """The ``if (…)`` that gates adjacent-task loading must not
        include ``!isRunning`` or ``isRunning`` as a conjunct."""
        # Find the line that contains the adjacent-loading guard.
        # It's the condition that also checks adjacentLoading,
        # activeTabId, and currentTaskName.
        pattern = re.compile(
            r"if\s*\([^)]*adjacentLoading[^)]*currentTaskName[^)]*\)"
        )
        m = pattern.search(self.src)
        self.assertIsNotNone(m, "Could not locate the adjacent-scroll guard")
        assert m is not None
        guard = m.group(0)
        self.assertNotIn(
            "isRunning",
            guard,
            f"Adjacent-scroll guard still blocks on isRunning: {guard!r}",
        )

    def test_scroll_lock_still_set_on_upward_scroll_while_running(self) -> None:
        """The ``_scrollLock = true`` line for upward scroll while running
        must still exist — it prevents auto-scroll-to-bottom from fighting
        the user's manual scroll."""
        self.assertIn(
            "if (isRunning && e.deltaY < 0) _scrollLock = true;",
            self.src,
            "_scrollLock line for running + upward scroll is missing",
        )


if __name__ == "__main__":
    unittest.main()

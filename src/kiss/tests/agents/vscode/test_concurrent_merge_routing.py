"""Tests for concurrent merge routing in SorcarSidebarView.

When two tasks in separate tabs finish simultaneously, each tab's merge
review must be shown independently — not both in the same tab.

The root cause was a FIFO queue (``_mergeOwnerTabIdQueue``) that
blindly serialized merge opens, causing the second ``_doOpenMerge``
to overwrite the first tab's state.  The fix replaces the FIFO with
``_activeMergeTabId`` (tracks the current merge owner) and
``_pendingMergeData`` (defers the second merge until the first finishes).

These tests verify the fix by inspecting the TypeScript source.
"""

from __future__ import annotations

import unittest
from pathlib import Path

_SIDEBAR_TS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "SorcarSidebarView.ts"
).read_text()


class TestConcurrentMergeRouting(unittest.TestCase):
    """Verify the FIFO queue was replaced with per-tab merge tracking."""

    def test_no_fifo_queue_remains(self) -> None:
        """The old _mergeOwnerTabIdQueue must not exist."""
        assert "_mergeOwnerTabIdQueue" not in _SIDEBAR_TS

    def test_active_merge_tab_id_field_exists(self) -> None:
        """_activeMergeTabId field must be declared."""
        assert "_activeMergeTabId" in _SIDEBAR_TS

    def test_pending_merge_data_field_exists(self) -> None:
        """_pendingMergeData field must be declared as a Map."""
        assert "_pendingMergeData" in _SIDEBAR_TS
        assert "Map<string, MergeData>" in _SIDEBAR_TS

    def test_merge_data_handler_checks_active_merge(self) -> None:
        """merge_data handler must check if another tab's merge is active."""
        idx = _SIDEBAR_TS.index("msg.type === 'merge_data'")
        block = _SIDEBAR_TS[idx : idx + 600]
        # Must check _activeMergeTabId to decide whether to defer
        assert "_activeMergeTabId" in block
        # Must save to _pendingMergeData when deferring
        assert "_pendingMergeData" in block

    def test_merge_data_defers_when_another_merge_active(self) -> None:
        """When another tab's merge is active, merge_data must defer."""
        idx = _SIDEBAR_TS.index("msg.type === 'merge_data'")
        block = _SIDEBAR_TS[idx : idx + 600]
        # Must set pending data when another merge is active
        assert "this._pendingMergeData.set(" in block

    def test_all_done_uses_active_merge_tab_id(self) -> None:
        """allDone handler must use _activeMergeTabId, not a FIFO shift."""
        idx = _SIDEBAR_TS.index("'allDone'")
        block = _SIDEBAR_TS[idx : idx + 400]
        assert "this._activeMergeTabId" in block
        # Must NOT use .shift() which was the old FIFO approach
        assert ".shift()" not in block

    def test_all_done_clears_active_merge(self) -> None:
        """allDone must set _activeMergeTabId to undefined."""
        idx = _SIDEBAR_TS.index("'allDone'")
        block = _SIDEBAR_TS[idx : idx + 400]
        assert "_activeMergeTabId = undefined" in block

    def test_all_done_starts_next_pending_merge(self) -> None:
        """allDone must call _startNextPendingMerge."""
        idx = _SIDEBAR_TS.index("'allDone'")
        block = _SIDEBAR_TS[idx : idx + 500]
        assert "_startNextPendingMerge" in block

    def test_start_next_pending_merge_method_exists(self) -> None:
        """_startNextPendingMerge must be defined."""
        assert "_startNextPendingMerge" in _SIDEBAR_TS
        assert "private _startNextPendingMerge" in _SIDEBAR_TS

    def test_start_next_pending_merge_sets_active_tab(self) -> None:
        """_startNextPendingMerge must set _activeMergeTabId."""
        idx = _SIDEBAR_TS.index("private _startNextPendingMerge")
        block = _SIDEBAR_TS[idx : idx + 800]
        assert "_activeMergeTabId = " in block

    def test_start_next_pending_merge_opens_merge(self) -> None:
        """_startNextPendingMerge must call openMerge."""
        idx = _SIDEBAR_TS.index("private _startNextPendingMerge")
        block = _SIDEBAR_TS[idx : idx + 800]
        assert "openMerge(" in block

    def test_start_next_pending_merge_notifies_webview(self) -> None:
        """_startNextPendingMerge must send merge_started to the webview."""
        idx = _SIDEBAR_TS.index("private _startNextPendingMerge")
        block = _SIDEBAR_TS[idx : idx + 800]
        assert "merge_started" in block

    def test_all_done_action_uses_active_merge(self) -> None:
        """The 'all-done' mergeAction must use _activeMergeTabId."""
        idx = _SIDEBAR_TS.index("'all-done'")
        block = _SIDEBAR_TS[idx : idx + 100]
        assert "_activeMergeTabId" in block


if __name__ == "__main__":
    unittest.main()

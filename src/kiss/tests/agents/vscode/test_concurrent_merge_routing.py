"""Tests for concurrent merge routing in SorcarSidebarView.

When two tasks in separate tabs finish simultaneously, each tab's merge
review must start immediately — no deferral.

The implementation uses per-tab ``MergeManager`` instances stored in
``_mergeManagers: Map<string, MergeManager>``.  Each tab gets its own
MergeManager created on demand via ``_getOrCreateMergeManager``, so
concurrent merges never interfere.

These tests verify the implementation by inspecting the TypeScript source.
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

_EXTENSION_TS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "extension.ts"
).read_text()


class TestConcurrentMergeRouting(unittest.TestCase):
    """Verify per-tab MergeManager instances for concurrent merge support."""

    def test_no_fifo_queue_remains(self) -> None:
        """The old _mergeOwnerTabIdQueue must not exist."""
        assert "_mergeOwnerTabIdQueue" not in _SIDEBAR_TS

    def test_no_single_merge_manager_field(self) -> None:
        """No single _mergeManager field — only the per-tab Map."""
        assert "_mergeManagers" in _SIDEBAR_TS
        assert "private _mergeManager:" not in _SIDEBAR_TS

    def test_no_active_merge_tab_id(self) -> None:
        """_activeMergeTabId deferral field must not exist."""
        assert "_activeMergeTabId" not in _SIDEBAR_TS

    def test_no_pending_merge_data(self) -> None:
        """_pendingMergeData deferral map must not exist."""
        assert "_pendingMergeData" not in _SIDEBAR_TS

    def test_no_start_next_pending_merge(self) -> None:
        """_startNextPendingMerge deferral method must not exist."""
        assert "_startNextPendingMerge" not in _SIDEBAR_TS

    def test_merge_managers_map_field_exists(self) -> None:
        """_mergeManagers must be declared as a Map<string, MergeManager>."""
        assert "Map<string, MergeManager>" in _SIDEBAR_TS
        assert "_mergeManagers" in _SIDEBAR_TS

    def test_get_or_create_merge_manager_method(self) -> None:
        """_getOrCreateMergeManager must be defined."""
        assert "_getOrCreateMergeManager" in _SIDEBAR_TS
        assert "private _getOrCreateMergeManager" in _SIDEBAR_TS

    def test_get_or_create_creates_new_manager(self) -> None:
        """_getOrCreateMergeManager must create a new MergeManager."""
        idx = _SIDEBAR_TS.index("private _getOrCreateMergeManager")
        block = _SIDEBAR_TS[idx : idx + 600]
        assert "new MergeManager()" in block

    def test_get_or_create_stores_in_map(self) -> None:
        """_getOrCreateMergeManager must store the manager in the map."""
        idx = _SIDEBAR_TS.index("private _getOrCreateMergeManager")
        block = _SIDEBAR_TS[idx : idx + 600]
        assert "_mergeManagers.set(" in block

    def test_get_or_create_listens_all_done(self) -> None:
        """_getOrCreateMergeManager must listen for allDone per-manager."""
        idx = _SIDEBAR_TS.index("private _getOrCreateMergeManager")
        block = _SIDEBAR_TS[idx : idx + 800]
        assert "'allDone'" in block

    def test_all_done_disposes_manager(self) -> None:
        """allDone handler must dispose the tab's MergeManager."""
        idx = _SIDEBAR_TS.index("private _getOrCreateMergeManager")
        block = _SIDEBAR_TS[idx : idx + 800]
        assert "mgr.dispose()" in block

    def test_all_done_removes_from_map(self) -> None:
        """allDone handler must remove the tab's manager from the map."""
        idx = _SIDEBAR_TS.index("private _getOrCreateMergeManager")
        block = _SIDEBAR_TS[idx : idx + 800]
        assert "_mergeManagers.delete(" in block

    def test_merge_data_handler_creates_per_tab_manager(self) -> None:
        """merge_data handler must call _getOrCreateMergeManager."""
        idx = _SIDEBAR_TS.index("msg.type === 'merge_data'")
        block = _SIDEBAR_TS[idx : idx + 500]
        assert "_getOrCreateMergeManager" in block

    def test_merge_data_handler_no_deferral(self) -> None:
        """merge_data handler must NOT defer — no _pendingMergeData."""
        idx = _SIDEBAR_TS.index("msg.type === 'merge_data'")
        block = _SIDEBAR_TS[idx : idx + 500]
        assert "_pendingMergeData" not in block

    def test_merge_action_routes_to_tab_manager(self) -> None:
        """mergeAction handler must look up manager from _mergeManagers."""
        idx = _SIDEBAR_TS.index("case 'mergeAction'")
        block = _SIDEBAR_TS[idx : idx + 600]
        assert "_mergeManagers.get(" in block

    def test_handle_merge_command_public_method(self) -> None:
        """handleMergeCommand must exist for extension.ts keyboard shortcuts."""
        assert "public handleMergeCommand" in _SIDEBAR_TS

    def test_handle_merge_command_routes_to_active_tab(self) -> None:
        """handleMergeCommand must use _activeTabId to find the manager."""
        idx = _SIDEBAR_TS.index("public handleMergeCommand")
        block = _SIDEBAR_TS[idx : idx + 400]
        assert "_mergeManagers.get(" in block
        assert "_activeTabId" in block

    def test_dispose_cleans_up_all_managers(self) -> None:
        """dispose must dispose all per-tab MergeManagers."""
        idx = _SIDEBAR_TS.index("public dispose(): void")
        block = _SIDEBAR_TS[idx : idx + 400]
        assert "_mergeManagers" in block

    def test_extension_no_global_merge_manager(self) -> None:
        """extension.ts must not create a global MergeManager."""
        assert "new MergeManager()" not in _EXTENSION_TS

    def test_extension_merge_commands_route_through_sidebar(self) -> None:
        """extension.ts merge keyboard commands must use handleMergeCommand."""
        assert "handleMergeCommand" in _EXTENSION_TS

    def test_constructor_takes_no_merge_manager(self) -> None:
        """SorcarSidebarView constructor must not take a MergeManager param."""
        idx = _SIDEBAR_TS.index("constructor(extensionUri")
        sig = _SIDEBAR_TS[idx : idx + 100]
        assert "mergeManager" not in sig


if __name__ == "__main__":
    unittest.main()

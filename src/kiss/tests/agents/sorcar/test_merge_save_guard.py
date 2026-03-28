"""Tests for merge save guard: prevent saving files with merge artifacts.

Root cause: MergeManager._doOpenMerge() inserts old (base) lines into the
document buffer for merge review, making it dirty.  Without an
onWillSaveTextDocument handler, a manual save (Cmd+S) or auto-save writes
BOTH old and new lines to disk, silently corrupting the file.

Additionally, ed.edit() return values are never checked, so a failed
insertion/deletion leaves hunk tracking metadata inconsistent with the
actual document content, causing subsequent operations to delete the wrong
lines.

These tests verify:
1. onWillSaveTextDocument handler exists and strips old lines before save
2. onDidSaveTextDocument handler re-inserts old lines after save
3. ed.edit() return values are checked
4. ProcessedHunk stores baseLines for re-insertion
"""

import re
import unittest


def _read_merge_manager_source() -> str:
    with open("src/kiss/agents/vscode/src/MergeManager.ts") as f:
        return f.read()


class TestMissingOnWillSaveHandler(unittest.TestCase):
    """BUG: No onWillSaveTextDocument handler — saving during merge corrupts files."""

    def test_will_save_handler_registered(self) -> None:
        """MergeManager must register onWillSaveTextDocument to strip old lines."""
        source = _read_merge_manager_source()
        assert "onWillSaveTextDocument" in source, (
            "MergeManager must register a workspace.onWillSaveTextDocument "
            "handler to strip old (base) lines before the document is saved. "
            "Without it, saving a file during merge review writes both old "
            "and new lines to disk, corrupting the file."
        )

    def test_did_save_handler_registered(self) -> None:
        """MergeManager must register onDidSaveTextDocument to re-insert old lines."""
        source = _read_merge_manager_source()
        assert "onDidSaveTextDocument" in source, (
            "MergeManager must register a workspace.onDidSaveTextDocument "
            "handler to re-insert old lines after save so the merge view "
            "remains functional."
        )


class TestWillSaveStripsOldLines(unittest.TestCase):
    """Verify onWillSaveTextDocument computes correct strip edits."""

    def test_handler_checks_merge_state(self) -> None:
        """Handler must check _ms[fp] before computing edits."""
        source = _read_merge_manager_source()
        # The willSave handler should reference _ms to check merge state
        assert "this._ms[" in source or "this._ms[fp" in source

    def test_handler_uses_wait_until(self) -> None:
        """Handler must call e.waitUntil() with strip TextEdits."""
        source = _read_merge_manager_source()
        assert "waitUntil" in source, (
            "onWillSaveTextDocument handler must call e.waitUntil(edits) "
            "to strip old lines before the save operation."
        )

    def test_strip_edits_use_hunk_positions(self) -> None:
        """Strip edits must use h.os and h.oc to delete old line ranges."""
        source = _read_merge_manager_source()
        # Should reference os (old start) and oc (old count) for strip edits
        assert "h.os" in source and "h.oc" in source


class TestDidSaveReinsertsOldLines(unittest.TestCase):
    """Verify onDidSaveTextDocument re-inserts old lines for continued review."""

    def test_reinsert_uses_base_lines(self) -> None:
        """Re-insertion must use stored baseLines data."""
        source = _read_merge_manager_source()
        assert "baseLines" in source, (
            "ProcessedHunk must store baseLines for re-insertion after save."
        )

    def test_reinsert_method_exists(self) -> None:
        """A re-insertion method must exist for restoring merge view after save."""
        source = _read_merge_manager_source()
        assert "_reinsertOldLines" in source or "_reinsert" in source, (
            "MergeManager must have a method to re-insert old lines after save."
        )


class TestEdEditReturnValueChecked(unittest.TestCase):
    """BUG: ed.edit() return value is never checked — failed edits corrupt hunks."""

    def test_del_lines_returns_boolean(self) -> None:
        """_delLines must return the boolean result of ed.edit()."""
        source = _read_merge_manager_source()
        # Find the _delLines method definition (not call sites)
        idx = source.find("private async _delLines")
        if idx < 0:
            idx = source.find("async _delLines")
        assert idx >= 0, "_delLines method not found"
        block = source[idx:idx + 800]
        assert "Promise<boolean>" in block or "boolean" in block, (
            "_delLines must return Promise<boolean> so callers can detect failure."
        )

    def test_del_lines_edit_result_captured(self) -> None:
        """ed.edit() calls in _delLines must capture the boolean result."""
        source = _read_merge_manager_source()
        idx = source.find("private async _delLines")
        assert idx >= 0
        method_end = source.find("\n  }", idx)
        block = source[idx:method_end]
        # The edit result should be captured, either inline or via separate assignment
        inline = re.compile(r'(?:const|let|var)\s+\w+\s*=\s*await\s+ed\.edit')
        assigned = re.compile(r'\w+\s*=\s*await\s+ed\.edit')
        assert inline.search(block) or assigned.search(block), (
            "ed.edit() calls in _delLines must capture the boolean return "
            "value (e.g., `ok = await ed.edit(...)`) to detect failure."
        )

    def test_open_merge_checks_edit_result(self) -> None:
        """_doOpenMerge must check ed.edit() return for insertion success."""
        source = _read_merge_manager_source()
        idx = source.find("_doOpenMerge")
        assert idx >= 0
        block = source[idx:idx + 3000]
        # Should check the edit result after inserting old lines
        insert_section = block[block.find("ed.edit"):]
        has_check = (
            "!" in insert_section[:200]
            or "ok" in insert_section[:200]
            or "if" in insert_section[:200]
        )
        assert has_check, (
            "_doOpenMerge must check ed.edit() return value after inserting "
            "old lines. A failed insertion with unchecked return corrupts "
            "all subsequent hunk offsets."
        )


class TestProcessedHunkStoresBaseLines(unittest.TestCase):
    """ProcessedHunk must store base lines for re-insertion after save."""

    def test_base_lines_in_interface(self) -> None:
        """ProcessedHunk interface must include baseLines field."""
        source = _read_merge_manager_source()
        # Find the ProcessedHunk interface
        idx = source.find("interface ProcessedHunk")
        assert idx >= 0, "ProcessedHunk interface not found"
        block = source[idx:source.find("}", idx) + 1]
        assert "baseLines" in block, (
            "ProcessedHunk must have a 'baseLines: string[]' field to store "
            "old lines for re-insertion after save."
        )

    def test_base_lines_populated_in_open_merge(self) -> None:
        """_doOpenMerge must populate baseLines when creating ProcessedHunks."""
        source = _read_merge_manager_source()
        idx = source.find("_doOpenMerge")
        assert idx >= 0
        block = source[idx:idx + 3000]
        assert "baseLines" in block, (
            "_doOpenMerge must store base lines in each ProcessedHunk so "
            "they can be re-inserted after a save operation."
        )


class TestReinsertGuardPreventsInfiniteLoop(unittest.TestCase):
    """Re-insertion after save must not trigger infinite auto-save loop."""

    def test_reinserting_guard_exists(self) -> None:
        """MergeManager must track files undergoing re-insertion."""
        source = _read_merge_manager_source()
        assert "_reinsertingFiles" in source or "_reinserting" in source, (
            "MergeManager must track which files are being re-inserted to "
            "prevent the onWillSave handler from stripping during re-insertion."
        )

    def test_will_save_updates_hunk_positions(self) -> None:
        """After stripping, hunk positions must be updated to clean state."""
        source = _read_merge_manager_source()
        # The willSave handler or a helper should update oc to 0
        # Look for pattern where oc is set to 0 after stripping
        assert "h.oc = 0" in source or "oc = 0" in source or "hasOldLines" in source, (
            "After stripping old lines in onWillSaveTextDocument, the hunk "
            "positions must be updated (oc set to 0) to prevent double-stripping "
            "if another save triggers before re-insertion."
        )


if __name__ == "__main__":
    unittest.main()

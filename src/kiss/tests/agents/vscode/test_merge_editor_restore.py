"""Test that the merge flow snapshots open editors before merge and
restores them (closing extra tabs) after all merges are resolved.

Feature: When a task finishes and the diff/merge interface opens, the
extension remembers which files were open in the editor window.  After
all diffs/merges have been resolved, only those original files remain
open — any files opened during the merge review are closed.

Implementation lives in SorcarSidebarView.ts:
  - ``_preMergeOpenFiles`` field stores the snapshot
  - ``_getOpenEditorFiles()`` captures open editor tab file paths
  - ``_restorePreMergeEditors()`` closes tabs not in the snapshot
  - ``_setupProcessListeners`` calls snapshot before ``openMerge``
  - ``allDone`` handler calls ``_restorePreMergeEditors``
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

SIDEBAR_TS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "SorcarSidebarView.ts"
)


def _extract_source() -> str:
    """Read the SorcarSidebarView.ts source."""
    return SIDEBAR_TS.read_text()


def _extract_method_body(source: str, method_name: str) -> str:
    """Extract the body of a class method definition by name.

    Looks for patterns like ``private _methodName(`` or
    ``private async _methodName(`` to avoid matching call sites.
    Returns the brace-delimited body including the outermost braces.
    """
    pattern = rf"(?:private|public|protected)\s+(?:async\s+)?{re.escape(method_name)}\s*\("
    m = re.search(pattern, source)
    assert m is not None, f"Method definition not found: {method_name}"
    start = m.start()
    brace_start = source.index("{", source.index(")", start))
    depth = 0
    i = brace_start
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start : i + 1]
        i += 1
    raise AssertionError(f"Unbalanced braces for method: {method_name}")  # noqa: F821


class TestPreMergeOpenFilesField(unittest.TestCase):
    """The class must have a ``_preMergeOpenFiles`` field."""

    def test_field_declared(self) -> None:
        src = _extract_source()
        assert re.search(
            r"private\s+_preMergeOpenFiles\b", src
        ), "_preMergeOpenFiles field must be declared as private"


class TestGetOpenEditorFilesMethod(unittest.TestCase):
    """A private method ``_getOpenEditorFiles`` must exist and use
    ``vscode.window.tabGroups`` to collect open editor file paths.
    """

    def test_method_exists(self) -> None:
        src = _extract_source()
        assert re.search(
            r"private\s+_getOpenEditorFiles\s*\(", src
        ), "_getOpenEditorFiles method must exist"

    def test_uses_tab_groups(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_getOpenEditorFiles")
        assert "tabGroups" in body, (
            "_getOpenEditorFiles must use vscode.window.tabGroups"
        )

    def test_returns_set(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_getOpenEditorFiles")
        assert "new Set" in body or "Set<string>" in body, (
            "_getOpenEditorFiles must return a Set of file paths"
        )


class TestRestorePreMergeEditorsMethod(unittest.TestCase):
    """A private method ``_restorePreMergeEditors`` must exist, close
    tabs not in the snapshot, and clear the snapshot.
    """

    def test_method_exists(self) -> None:
        src = _extract_source()
        assert re.search(
            r"private\s+async\s+_restorePreMergeEditors\s*\(", src
        ), "_restorePreMergeEditors method must exist as async"

    def test_closes_extra_tabs(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_restorePreMergeEditors")
        assert "tabGroups.close" in body, (
            "_restorePreMergeEditors must close tabs via tabGroups.close"
        )

    def test_clears_snapshot(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_restorePreMergeEditors")
        assert "_preMergeOpenFiles" in body and "null" in body, (
            "_restorePreMergeEditors must clear _preMergeOpenFiles to null"
        )

    def test_checks_snapshot_against_current_tabs(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_restorePreMergeEditors")
        # Must iterate over current tabs and check against the snapshot
        assert "tabGroups" in body, (
            "_restorePreMergeEditors must iterate tabGroups"
        )
        assert ".has(" in body or "has(" in body, (
            "_restorePreMergeEditors must check snapshot.has() for each tab"
        )


class TestMergeDataSnapshotsEditors(unittest.TestCase):
    """When a ``merge_data`` message arrives in ``_setupProcessListeners``,
    the handler must snapshot open editors before calling ``openMerge``.
    """

    def test_snapshot_before_open_merge(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_setupProcessListeners")
        # Find the merge_data handling block
        merge_idx = body.find("merge_data")
        assert merge_idx != -1, "merge_data handling must exist"

        # Extract the merge_data block
        merge_block = body[merge_idx:]

        # Find snapshot call and openMerge call positions
        snapshot_pos = merge_block.find("_getOpenEditorFiles")
        open_merge_pos = merge_block.find("openMerge")
        assert snapshot_pos != -1, (
            "_getOpenEditorFiles must be called in merge_data handler"
        )
        assert open_merge_pos != -1, (
            "openMerge must be called in merge_data handler"
        )
        assert snapshot_pos < open_merge_pos, (
            "_getOpenEditorFiles must be called BEFORE openMerge"
        )

    def test_snapshot_stored_in_field(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_setupProcessListeners")
        merge_idx = body.find("merge_data")
        merge_block = body[merge_idx:]
        assert "_preMergeOpenFiles" in merge_block, (
            "merge_data handler must store snapshot in _preMergeOpenFiles"
        )


class TestAllDoneRestoresEditors(unittest.TestCase):
    """The ``allDone`` event handler must call ``_restorePreMergeEditors``
    to close extra tabs after merge is complete.
    """

    def test_all_done_handler_calls_restore(self) -> None:
        src = _extract_source()
        # Find the allDone listener setup
        all_done_match = re.search(
            r"on\(\s*['\"]allDone['\"]", src
        )
        assert all_done_match is not None, (
            "allDone event listener must be registered"
        )
        # Extract the callback body
        start = all_done_match.start()
        # Find the arrow function or callback body after allDone
        arrow_or_fn = src[start:]
        # Find the first { after the allDone registration
        brace_idx = arrow_or_fn.find("{")
        assert brace_idx != -1
        depth = 0
        i = brace_idx
        while i < len(arrow_or_fn):
            if arrow_or_fn[i] == "{":
                depth += 1
            elif arrow_or_fn[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        callback_body = arrow_or_fn[brace_idx : i + 1]
        assert "_restorePreMergeEditors" in callback_body, (
            "allDone handler must call _restorePreMergeEditors"
        )


class TestSnapshotOnlyOnce(unittest.TestCase):
    """The snapshot should only be taken if there isn't already one,
    to avoid overwriting when multiple merge_data messages queue up.
    """

    def test_guarded_by_null_check(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_setupProcessListeners")
        merge_idx = body.find("merge_data")
        merge_block = body[merge_idx:]

        # The snapshot call should be guarded by a null check
        # e.g., if (!this._preMergeOpenFiles) { ... }
        snapshot_idx = merge_block.find("_getOpenEditorFiles")
        preceding = merge_block[:snapshot_idx]
        assert (
            "_preMergeOpenFiles" in preceding
        ), (
            "Snapshot must be guarded by checking _preMergeOpenFiles is null/falsy"
        )


if __name__ == "__main__":
    unittest.main()

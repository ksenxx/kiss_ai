"""Test that the merge flow snapshots open editors before merge and
restores them (closing extra tabs) after all merges are resolved.

Feature: When a task finishes and the diff/merge interface opens, the
extension remembers which files were open in the editor window.  After
all diffs/merges have been resolved, only those original files remain
open — any files opened during the merge review are closed.

The snapshot is maintained **per tab** so that concurrent merge flows
from different tabs do not interfere with each other.

Implementation lives in SorcarSidebarView.ts:
  - ``_preMergeOpenFiles`` field: ``Map<string, Set<string>>`` keyed by tabId
  - ``_getOpenEditorFiles()`` captures open editor tab file paths
  - ``_restorePreMergeEditors(tabId)`` closes tabs not in that tab's snapshot
  - ``_setupProcessListeners`` calls snapshot before ``openMerge``
  - ``allDone`` handler calls ``_restorePreMergeEditors`` with the tab's id
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


# ---------------------------------------------------------------------------
# Field declaration
# ---------------------------------------------------------------------------


class TestPreMergeOpenFilesField(unittest.TestCase):
    """The class must have a per-tab ``_preMergeOpenFiles`` Map field."""

    def test_field_declared(self) -> None:
        src = _extract_source()
        assert re.search(
            r"private\s+_preMergeOpenFiles\b", src
        ), "_preMergeOpenFiles field must be declared as private"

    def test_field_is_map(self) -> None:
        """The field must be a Map<string, Set<string>>, not a single Set."""
        src = _extract_source()
        # Match the field declaration line
        m = re.search(
            r"private\s+_preMergeOpenFiles\s*[:=].*", src
        )
        assert m is not None, "_preMergeOpenFiles declaration not found"
        decl = m.group(0)
        assert "Map" in decl, (
            f"_preMergeOpenFiles must be a Map (per-tab), got: {decl}"
        )

    def test_field_initialized_as_new_map(self) -> None:
        """The field must be initialized as ``new Map()``."""
        src = _extract_source()
        m = re.search(
            r"private\s+_preMergeOpenFiles\b[^;]*new\s+Map\s*\(\s*\)", src
        )
        assert m is not None, (
            "_preMergeOpenFiles must be initialized with new Map()"
        )


# ---------------------------------------------------------------------------
# _getOpenEditorFiles
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _restorePreMergeEditors — now accepts tabId
# ---------------------------------------------------------------------------


class TestRestorePreMergeEditorsMethod(unittest.TestCase):
    """``_restorePreMergeEditors`` must accept a tabId, close tabs not in
    that tab's snapshot, and delete the entry from the Map.
    """

    def test_method_exists(self) -> None:
        src = _extract_source()
        assert re.search(
            r"private\s+async\s+_restorePreMergeEditors\s*\(", src
        ), "_restorePreMergeEditors method must exist as async"

    def test_accepts_tab_id_parameter(self) -> None:
        """The method must accept a tabId parameter."""
        src = _extract_source()
        m = re.search(
            r"private\s+async\s+_restorePreMergeEditors\s*\(\s*(\w+)", src
        )
        assert m is not None, (
            "_restorePreMergeEditors must have a parameter"
        )
        param_name = m.group(1)
        assert "tab" in param_name.lower() or "id" in param_name.lower(), (
            f"Parameter should be a tab id, got: {param_name}"
        )

    def test_closes_extra_tabs(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_restorePreMergeEditors")
        assert "tabGroups.close" in body, (
            "_restorePreMergeEditors must close tabs via tabGroups.close"
        )

    def test_deletes_entry_from_map(self) -> None:
        """After restoring, the per-tab entry must be deleted from the Map."""
        src = _extract_source()
        body = _extract_method_body(src, "_restorePreMergeEditors")
        assert "_preMergeOpenFiles" in body, (
            "_restorePreMergeEditors must reference _preMergeOpenFiles"
        )
        assert ".delete(" in body or ".delete (" in body, (
            "_restorePreMergeEditors must .delete() the tab entry from the Map"
        )

    def test_gets_snapshot_from_map(self) -> None:
        """Must use .get() to retrieve the per-tab snapshot."""
        src = _extract_source()
        body = _extract_method_body(src, "_restorePreMergeEditors")
        assert ".get(" in body, (
            "_restorePreMergeEditors must .get() the snapshot from the Map"
        )

    def test_checks_snapshot_against_current_tabs(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_restorePreMergeEditors")
        assert "tabGroups" in body, (
            "_restorePreMergeEditors must iterate tabGroups"
        )
        assert ".has(" in body, (
            "_restorePreMergeEditors must check snapshot.has() for each tab"
        )


# ---------------------------------------------------------------------------
# merge_data handler — per-tab snapshot
# ---------------------------------------------------------------------------


class TestMergeDataSnapshotsEditors(unittest.TestCase):
    """When a ``merge_data`` message arrives in ``_setupProcessListeners``,
    the handler must snapshot open editors per-tab before calling ``openMerge``.
    """

    def test_snapshot_before_open_merge(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_setupProcessListeners")
        merge_idx = body.find("merge_data")
        assert merge_idx != -1, "merge_data handling must exist"
        merge_block = body[merge_idx:]

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

    def test_snapshot_stored_in_map_with_set(self) -> None:
        """Snapshot must be stored via .set() on the per-tab Map."""
        src = _extract_source()
        body = _extract_method_body(src, "_setupProcessListeners")
        merge_idx = body.find("merge_data")
        merge_block = body[merge_idx:]
        assert "_preMergeOpenFiles" in merge_block, (
            "merge_data handler must reference _preMergeOpenFiles"
        )
        # Must use .set() to store per-tab snapshot
        assert re.search(
            r"_preMergeOpenFiles\.set\s*\(", merge_block
        ), (
            "merge_data handler must use _preMergeOpenFiles.set() to store "
            "the per-tab snapshot"
        )


# ---------------------------------------------------------------------------
# allDone handler — passes tabId to restore
# ---------------------------------------------------------------------------


class TestAllDoneRestoresEditors(unittest.TestCase):
    """The ``allDone`` event handler must call ``_restorePreMergeEditors``
    with the tabId to close extra tabs after merge is complete.
    """

    def test_all_done_handler_calls_restore(self) -> None:
        src = _extract_source()
        all_done_match = re.search(r"on\(\s*['\"]allDone['\"]", src)
        assert all_done_match is not None, (
            "allDone event listener must be registered"
        )
        start = all_done_match.start()
        arrow_or_fn = src[start:]
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

    def test_all_done_passes_tab_id_to_restore(self) -> None:
        """_restorePreMergeEditors must be called with the tabId."""
        src = _extract_source()
        all_done_match = re.search(r"on\(\s*['\"]allDone['\"]", src)
        assert all_done_match is not None
        start = all_done_match.start()
        arrow_or_fn = src[start:]
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
        # Must pass tabId (not call with empty args)
        m = re.search(
            r"_restorePreMergeEditors\s*\(\s*(\w+)\s*\)", callback_body
        )
        assert m is not None, (
            "_restorePreMergeEditors must be called with a tabId argument"
        )


# ---------------------------------------------------------------------------
# Snapshot guard — per-tab .has() check
# ---------------------------------------------------------------------------


class TestSnapshotOnlyOncePerTab(unittest.TestCase):
    """The snapshot should only be taken for a tab if that tab doesn't
    already have one, using .has() on the Map — not a global falsy check.
    """

    def test_guarded_by_map_has_check(self) -> None:
        src = _extract_source()
        body = _extract_method_body(src, "_setupProcessListeners")
        merge_idx = body.find("merge_data")
        merge_block = body[merge_idx:]

        snapshot_idx = merge_block.find("_getOpenEditorFiles")
        preceding = merge_block[:snapshot_idx]
        # Must use .has() to check if this tab already has a snapshot
        assert re.search(
            r"_preMergeOpenFiles\.has\s*\(", preceding
        ), (
            "Snapshot guard must use _preMergeOpenFiles.has() to check "
            "per-tab existence (not a global falsy check)"
        )


if __name__ == "__main__":
    unittest.main()

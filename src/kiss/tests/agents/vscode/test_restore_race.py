"""Tests for the race condition in _restorePreMergeEditors.

Race: The allDone handler calls ``_restorePreMergeEditors`` with ``void``
(fire-and-forget).  Since the method is async and does ``await
tabGroups.close(...)``, it returns a pending promise.  If the backend
sends a ``merge_data`` message before that promise settles:

  1. The snapshot for the new merge captures stale editor state (the
     previous restore hasn't closed its extra tabs yet).
  2. ``openMerge`` opens the new merge's files.
  3. The pending restore resumes, reads current tabs (which now include
     the new merge's files), and closes them — they weren't in the
     old snapshot.

Fix: Introduce a ``_restoreChain`` promise field.  Both the ``allDone``
restore and the ``merge_data`` snapshot+openMerge are chained through it,
ensuring full serialization of editor-state-mutating operations.
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


def _src() -> str:
    return SIDEBAR_TS.read_text()


def _method_body(source: str, method_name: str) -> str:
    pattern = (
        rf"(?:private|public|protected)\s+(?:async\s+)?{re.escape(method_name)}\s*\("
    )
    m = re.search(pattern, source)
    assert m is not None, f"Method definition not found: {method_name}"
    brace_start = source.index("{", source.index(")", m.start()))
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


def _all_done_body(source: str) -> str:
    m = re.search(r"on\(\s*['\"]allDone['\"]", source)
    assert m is not None, "allDone listener not found"
    start = m.start()
    text = source[start:]
    brace_idx = text.find("{")
    assert brace_idx != -1
    depth = 0
    i = brace_idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx : i + 1]
        i += 1
    raise AssertionError("Unbalanced braces in allDone handler")  # noqa: F821


def _merge_data_block(source: str) -> str:
    """Extract the merge_data if-block from _setupProcessListeners."""
    body = _method_body(source, "_setupProcessListeners")
    idx = body.find("merge_data")
    assert idx != -1, "merge_data handling not found"
    # Find the if-block opening brace
    brace_start = body.index("{", idx)
    depth = 0
    i = brace_start
    while i < len(body):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                return body[brace_start : i + 1]
        i += 1
    raise AssertionError("Unbalanced braces in merge_data block")  # noqa: F821


# ---------------------------------------------------------------------------
# _restoreChain field
# ---------------------------------------------------------------------------


class TestRestoreChainField(unittest.TestCase):
    """A ``_restoreChain`` promise field must exist to serialize restores."""

    def test_field_exists(self) -> None:
        src = _src()
        assert re.search(
            r"private\s+_restoreChain\b", src
        ), "_restoreChain field must be declared"

    def test_field_initialized_as_resolved_promise(self) -> None:
        src = _src()
        assert re.search(
            r"_restoreChain\b[^;]*Promise\.resolve\s*\(\s*\)", src
        ), "_restoreChain must be initialized with Promise.resolve()"


# ---------------------------------------------------------------------------
# allDone handler — must chain, not fire-and-forget
# ---------------------------------------------------------------------------


class TestAllDoneChainsRestore(unittest.TestCase):
    """The allDone handler must chain _restorePreMergeEditors through
    _restoreChain, NOT call it with ``void`` (fire-and-forget).
    """

    def test_no_void_fire_and_forget(self) -> None:
        """Must NOT have ``void this._restorePreMergeEditors``."""
        body = _all_done_body(_src())
        assert not re.search(
            r"void\s+this\._restorePreMergeEditors", body
        ), (
            "allDone handler must NOT use void _restorePreMergeEditors "
            "(fire-and-forget causes race)"
        )

    def test_chains_through_restore_chain(self) -> None:
        """Must assign to _restoreChain with .then()."""
        body = _all_done_body(_src())
        assert "_restoreChain" in body, (
            "allDone handler must use _restoreChain to serialize restores"
        )
        assert ".then(" in body, (
            "allDone handler must chain via .then()"
        )

    def test_restore_inside_then_callback(self) -> None:
        """_restorePreMergeEditors must be called inside the .then() callback."""
        body = _all_done_body(_src())
        # Find the .then( ... _restorePreMergeEditors ... ) pattern
        then_idx = body.find(".then(")
        assert then_idx != -1
        rest = body[then_idx:]
        assert "_restorePreMergeEditors" in rest, (
            "_restorePreMergeEditors must be inside the .then() callback"
        )


# ---------------------------------------------------------------------------
# merge_data handler — must wait for restore chain
# ---------------------------------------------------------------------------


class TestMergeDataAwaitsRestoreChain(unittest.TestCase):
    """The merge_data handler must chain snapshot+openMerge through
    _restoreChain so they run after any pending restore completes.
    """

    def test_merge_data_references_restore_chain(self) -> None:
        block = _merge_data_block(_src())
        assert "_restoreChain" in block, (
            "merge_data handler must reference _restoreChain to "
            "wait for pending restores"
        )

    def test_snapshot_inside_chain(self) -> None:
        """_getOpenEditorFiles must be called inside the chained callback,
        not synchronously before the chain.
        """
        block = _merge_data_block(_src())
        chain_idx = block.find("_restoreChain")
        assert chain_idx != -1
        # Everything after _restoreChain should contain the snapshot
        after_chain = block[chain_idx:]
        assert "_getOpenEditorFiles" in after_chain, (
            "_getOpenEditorFiles must be inside the _restoreChain callback"
        )

    def test_open_merge_inside_chain(self) -> None:
        """openMerge must be called inside the chained callback."""
        block = _merge_data_block(_src())
        chain_idx = block.find("_restoreChain")
        assert chain_idx != -1
        after_chain = block[chain_idx:]
        assert "openMerge" in after_chain, (
            "openMerge must be inside the _restoreChain callback"
        )

    def test_chain_is_extended_not_just_awaited(self) -> None:
        """_restoreChain must be reassigned (extended), not just .then()'d
        without capturing, so subsequent operations also wait.
        """
        block = _merge_data_block(_src())
        assert re.search(
            r"_restoreChain\s*=\s*(?:this\.)?_restoreChain", block
        ), (
            "_restoreChain must be reassigned (this._restoreChain = "
            "this._restoreChain.then(...)) to extend the chain"
        )


if __name__ == "__main__":
    unittest.main()

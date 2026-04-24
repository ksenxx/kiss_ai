"""Integration tests for bug fixes, redundancy acknowledgement, and
consistency improvements in ``kiss.agents.vscode``.

These tests assert the FIXED behavior — each test confirms the bug is
resolved or the inconsistency is eliminated.

Bugs fixed
----------
B5: ``_close_tab`` now calls ``_cleanup_merge_data()`` to remove
    on-disk merge artifacts when a tab is closed.
B6: ``model_vendor`` now correctly classifies ``openai/``-prefixed
    models (e.g. ``openai/gpt-4o``) as ``"OpenAI"``.
B7: ``_finish_merge`` now guards against both ``None`` and empty string
    via ``if not tab_id:``, preventing ``_merge_data_dir("")`` from
    returning the parent directory and nuking all tabs' merge data.
B8: ``_run_task`` now broadcasts ``status: running: False`` INSIDE
    the ``_state_lock`` critical section (A2 fix).

Bugs acknowledged (not fixed — intentional)
--------------------------------------------
B4: ``_complete_from_active_file`` returns the LONGEST matching suffix.
    This is intentional behavior per user feedback.

Redundancies acknowledged
-------------------------
R2: ``clip_autocomplete_suggestion`` applied to local completions is
    a no-op for clean identifier suffixes. Kept for safety against
    unexpected LLM output.

Inconsistencies fixed
---------------------
I2: ``tab_id`` parameter types are now consistently ``str = ""``.
I3: ``_broadcast_worktree_done`` now always includes ``tabId``.
"""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import threading
import typing
import unittest

from kiss.agents.vscode.diff_merge import (
    _cleanup_merge_data,
    _merge_data_dir,
)
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
    model_vendor,
)
from kiss.agents.vscode.merge_flow import _MergeFlowMixin
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.task_runner import _TaskRunnerMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        # Still call original to exercise _record_event, but suppress stdout
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


# ===================================================================
# B4 — _complete_from_active_file returns LONGEST suffix (intentional)
# ===================================================================


class TestCompleteFromActiveFileLongestMatch(unittest.TestCase):
    """B4: ``_complete_from_active_file`` prefers the longest matching
    suffix.  This is INTENTIONAL behavior — test confirms it still works.
    """

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.content = (
            "server = start_server()\n"
            "server_config = load_config()\n"
            "server_manager = create_manager()\n"
        )

    def test_returns_longest_suffix(self) -> None:
        """The function returns 'er_manager' (10 chars) — intentional."""
        result = self.server._complete_from_active_file(
            "use serv", snapshot_content=self.content,
        )
        assert result == "er_manager", (
            f"B4 intentional: expected longest suffix 'er_manager', got {result!r}"
        )

    def test_source_confirms_longest_preference(self) -> None:
        """Structural: the comparison uses ``len(suffix) > len(best)``."""
        from kiss.agents.vscode.autocomplete import _AutocompleteMixin

        src = inspect.getsource(
            _AutocompleteMixin._complete_from_active_file,
        )
        assert "len(suffix) > len(best)" in src, (
            "B4 intentional: source confirms longest-wins comparison"
        )


# ===================================================================
# B5 — _close_tab now cleans up merge data on disk (FIXED)
# ===================================================================


class TestCloseTabMergeDataCleanup(unittest.TestCase):
    """B5 fix: ``_close_tab`` now calls ``_cleanup_merge_data`` to
    remove on-disk merge artifacts when a tab is closed.
    """

    def test_source_has_merge_cleanup_call(self) -> None:
        """Structural: ``_close_tab`` now references cleanup functions."""
        src = inspect.getsource(VSCodeServer._close_tab)
        assert "_cleanup_merge_data" in src, (
            "B5 fix: _close_tab should call _cleanup_merge_data"
        )
        assert "_merge_data_dir" in src, (
            "B5 fix: _close_tab should reference _merge_data_dir"
        )

    def test_merge_data_removed_after_tab_close(self) -> None:
        """Behavioral: merge data directory is removed after tab close."""
        server, _ = _make_server()
        tab_id = "leak-test-tab"
        server._get_tab(tab_id)

        merge_dir = _merge_data_dir(tab_id)
        merge_dir.mkdir(parents=True, exist_ok=True)
        sentinel = merge_dir / "pending-merge.json"
        sentinel.write_text('{"files": []}')

        try:
            server._close_tab(tab_id)

            # B5 fix: the merge directory is cleaned up after close
            assert not sentinel.exists(), (
                "B5 fix: pending-merge.json should be removed after _close_tab"
            )
            assert not merge_dir.exists(), (
                "B5 fix: merge_dir should be removed after _close_tab"
            )
        finally:
            if merge_dir.exists():
                shutil.rmtree(merge_dir)


# ===================================================================
# B6 — model_vendor correctly classifies openai/ models (FIXED)
# ===================================================================


class TestModelVendorOpenAIClassification(unittest.TestCase):
    """B6 fix: ``model_vendor("openai/gpt-4o")`` now correctly
    returns ``("OpenAI", 1)``.
    """

    def test_openai_gpt4o_classified_as_openai(self) -> None:
        vendor, order = model_vendor("openai/gpt-4o")
        assert vendor == "OpenAI" and order == 1, (
            f"B6 fix: openai/gpt-4o should be OpenAI, got ({vendor}, {order})"
        )

    def test_openai_o1_classified_as_openai(self) -> None:
        vendor, order = model_vendor("openai/o1-preview")
        assert vendor == "OpenAI" and order == 1, (
            f"B6 fix: openai/o1-preview should be OpenAI, got ({vendor}, {order})"
        )

    def test_bare_gpt4o_still_classified_correctly(self) -> None:
        """The bare name without ``openai/`` prefix still works."""
        vendor, order = model_vendor("gpt-4o")
        assert vendor == "OpenAI" and order == 1, (
            f"Bare gpt-4o should be OpenAI, got ({vendor}, {order})"
        )

    def test_openrouter_not_misclassified(self) -> None:
        """openrouter/ models still classified as OpenRouter."""
        vendor, order = model_vendor("openrouter/anthropic/claude-haiku")
        assert vendor == "OpenRouter" and order == 4, (
            f"openrouter/ should be OpenRouter, got ({vendor}, {order})"
        )

    def test_source_confirms_openai_prefix_branch(self) -> None:
        """Structural: the OpenAI branch now includes 'openai/' prefix."""
        src = inspect.getsource(model_vendor)
        assert 'name.startswith("openai/")' in src, (
            "B6 fix: source should check for openai/ prefix"
        )


# ===================================================================
# B7 — _finish_merge guards against empty string (FIXED)
# ===================================================================


class TestFinishMergeEmptyTabIdGuard(unittest.TestCase):
    """B7 fix: ``_finish_merge("")`` is now a no-op instead of nuking
    the parent merge directory.
    """

    def test_merge_data_dir_empty_still_returns_parent(self) -> None:
        """``_merge_data_dir("")`` still returns the parent — the guard is
        at the _finish_merge level."""
        parent = _merge_data_dir("")
        child = _merge_data_dir("some-tab")
        assert child.parent == parent, (
            f"_merge_data_dir('') is parent of per-tab dirs; "
            f"parent={parent}, child.parent={child.parent}"
        )

    def test_finish_merge_guards_against_empty_string(self) -> None:
        """Structural: ``_finish_merge`` uses ``if not tab_id:``."""
        src = inspect.getsource(_MergeFlowMixin._finish_merge)
        assert "not tab_id" in src, (
            "B7 fix: _finish_merge should guard against falsy tab_id"
        )

    def test_finish_merge_empty_is_noop(self) -> None:
        """Behavioral: calling _finish_merge('') does not destroy data."""
        server, _ = _make_server()

        # Create merge data for a real tab
        real_tab_id = "real-tab"
        merge_dir = _merge_data_dir(real_tab_id)
        merge_dir.mkdir(parents=True, exist_ok=True)
        sentinel = merge_dir / "pending-merge.json"
        sentinel.write_text('{"files": []}')

        try:
            # Call _finish_merge with empty tab_id — should be a no-op
            server._finish_merge("")

            # B7 fix: the real tab's data should survive
            assert merge_dir.exists(), (
                "B7 fix: real tab's merge_dir should survive _finish_merge('')"
            )
            assert sentinel.exists(), (
                "B7 fix: real tab's data should survive _finish_merge('')"
            )
        finally:
            if merge_dir.exists():
                shutil.rmtree(merge_dir)

    def test_cleanup_merge_data_would_rmtree_parent(self) -> None:
        """Behavioral: ``_cleanup_merge_data`` removes whatever path is given.
        This confirms why the guard is necessary."""
        td = tempfile.mkdtemp()
        child1 = os.path.join(td, "tab1")
        child2 = os.path.join(td, "tab2")
        os.makedirs(child1)
        os.makedirs(child2)
        open(os.path.join(child1, "data.json"), "w").close()
        open(os.path.join(child2, "data.json"), "w").close()

        _cleanup_merge_data(td)

        assert not os.path.exists(td), (
            "_cleanup_merge_data removes the entire tree"
        )


# ===================================================================
# B8 — _run_task broadcasts status OUTSIDE _state_lock (FIXED)
# ===================================================================


class TestRunTaskStatusBroadcastInsideLock(unittest.TestCase):
    """B8 / A2 fix: ``_run_task``'s finally block broadcasts
    ``status: running: False`` INSIDE the ``_state_lock`` block to
    prevent a race where a new ``_cmd_run`` broadcasts ``status: True``
    before the stale ``status: False``.
    """

    def test_source_confirms_broadcast_inside_state_lock(self) -> None:
        """Structural: broadcast call is deeper than the with line."""
        src = inspect.getsource(_TaskRunnerMixin._run_task)
        lines = src.splitlines()

        finally_idx = None
        lock_idx = None
        broadcast_idx = None
        for i, line in enumerate(lines):
            if "finally:" in line:
                finally_idx = i
            if finally_idx is not None and "_state_lock" in line and "with" in line:
                if lock_idx is None:
                    lock_idx = i
            if (
                lock_idx is not None
                and "broadcast" in line
                and "running" in line
                and "False" in line
            ):
                broadcast_idx = i
                break

        assert lock_idx is not None, "Found _state_lock in finally block"
        assert broadcast_idx is not None, "Found status broadcast"
        assert broadcast_idx > lock_idx, (
            "broadcast is after the lock line"
        )

        # A2 fix: the broadcast should be DEEPER than the `with`
        # statement, meaning it's inside the critical section
        indent_lock = len(lines[lock_idx]) - len(lines[lock_idx].lstrip())
        indent_bc = len(lines[broadcast_idx]) - len(lines[broadcast_idx].lstrip())
        assert indent_bc > indent_lock, (
            f"A2 fix: broadcast indent ({indent_bc}) > lock indent "
            f"({indent_lock}), confirming it's inside the critical section"
        )


# ===================================================================
# R2 — clip_autocomplete_suggestion is a no-op for identifier suffixes
# ===================================================================


class TestClipAutocompleteSuggestionRedundant(unittest.TestCase):
    """R2 redundancy: ``clip_autocomplete_suggestion`` is applied to
    the output of ``_complete_from_active_file`` but all its
    transformations are no-ops for clean identifier suffixes.
    Kept for safety — these tests document the behavior.
    """

    def test_no_op_for_plain_suffix(self) -> None:
        result = clip_autocomplete_suggestion("serv", "er_manager")
        assert result == "er_manager", f"Expected identity, got {result!r}"

    def test_no_op_for_dotted_suffix(self) -> None:
        result = clip_autocomplete_suggestion("se", "lf.setup")
        assert result == "lf.setup", f"Expected identity, got {result!r}"

    def test_no_op_for_underscore_suffix(self) -> None:
        result = clip_autocomplete_suggestion("server", "_config")
        assert result == "_config", f"Expected identity, got {result!r}"

    def test_source_confirms_clip_applied_to_local_completions(self) -> None:
        from kiss.agents.vscode.autocomplete import _AutocompleteMixin

        src = inspect.getsource(_AutocompleteMixin._complete)
        assert "_complete_from_active_file" in src
        assert "clip_autocomplete_suggestion" in src


# ===================================================================
# I2 — tab_id parameter types now consistently use str = "" (FIXED)
# ===================================================================


class TestTabIdTypeConsistency(unittest.TestCase):
    """I2 fix: ``tab_id`` parameter types and defaults are now
    consistently ``str = ""`` across all methods.
    """

    def test_all_use_str_default_empty(self) -> None:
        """All tab_id params with defaults use ``str`` type and ``""`` default."""
        methods_with_defaults: dict[str, typing.Any] = {}
        for name, method in [
            ("_finish_merge", _MergeFlowMixin._finish_merge),
            ("_handle_worktree_action", _MergeFlowMixin._handle_worktree_action),
            ("_handle_autocommit_action", _MergeFlowMixin._handle_autocommit_action),
            ("_stop_task", _TaskRunnerMixin._stop_task),
        ]:
            sig = inspect.signature(method)  # type: ignore[arg-type]
            for pname, param in sig.parameters.items():
                if "tab" in pname.lower() and param.default is not inspect.Parameter.empty:
                    methods_with_defaults[name] = param.default

        # I2 fix: all defaults should be "" (no None mixed in)
        defaults = set(methods_with_defaults.values())
        assert defaults == {""}, (
            f"I2 fix: all tab_id defaults should be '', got: {methods_with_defaults}"
        )

    def test_consistent_type_annotations(self) -> None:
        """All tab_id params with defaults annotate as ``str``."""
        annotations: dict[str, str] = {}
        for name, method in [
            ("_finish_merge", _MergeFlowMixin._finish_merge),
            ("_stop_task", _TaskRunnerMixin._stop_task),
        ]:
            sig = inspect.signature(method)  # type: ignore[arg-type]
            for pname, param in sig.parameters.items():
                if "tab" in pname.lower() and param.default is not inspect.Parameter.empty:
                    annotations[name] = str(param.annotation)

        # I2 fix: all should be 'str', no 'str | None'
        for name, ann in annotations.items():
            assert "None" not in ann, (
                f"I2 fix: {name} tab_id annotation should be str, got: {ann}"
            )


# ===================================================================
# I3 — _broadcast_worktree_done always includes tabId (FIXED)
# ===================================================================


class TestBroadcastWorktreeDoneAlwaysHasTabId(unittest.TestCase):
    """I3 fix: ``worktree_done`` broadcast (now inlined in
    ``_present_pending_worktree``) always includes ``tabId``.
    """

    def test_source_confirms_unconditional_tab_id(self) -> None:
        """Structural: ``tabId`` is set directly in the worktree_done event."""
        src = inspect.getsource(_MergeFlowMixin._present_pending_worktree)
        assert '"worktree_done"' in src, (
            "I3: worktree_done event should be inlined in _present_pending_worktree"
        )
        assert '"tabId"' in src, (
            "I3 fix: tabId should be in the event dict"
        )


if __name__ == "__main__":
    unittest.main()

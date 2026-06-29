# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


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



class TestCloseTabMergeDataCleanup(unittest.TestCase):
    """B5 fix: ``_close_tab`` now calls ``_cleanup_merge_data`` to
    remove on-disk merge artifacts when a tab is closed.
    """


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

            assert not sentinel.exists(), (
                "B5 fix: pending-merge.json should be removed after _close_tab"
            )
            assert not merge_dir.exists(), (
                "B5 fix: merge_dir should be removed after _close_tab"
            )
        finally:
            if merge_dir.exists():
                shutil.rmtree(merge_dir)


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


    def test_finish_merge_empty_is_noop(self) -> None:
        """Behavioral: calling _finish_merge('') does not destroy data."""
        server, _ = _make_server()

        real_tab_id = "real-tab"
        merge_dir = _merge_data_dir(real_tab_id)
        merge_dir.mkdir(parents=True, exist_ok=True)
        sentinel = merge_dir / "pending-merge.json"
        sentinel.write_text('{"files": []}')

        try:
            server._finish_merge("")

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

        for name, ann in annotations.items():
            assert "None" not in ann, (
                f"I2 fix: {name} tab_id annotation should be str, got: {ann}"
            )




if __name__ == "__main__":
    unittest.main()

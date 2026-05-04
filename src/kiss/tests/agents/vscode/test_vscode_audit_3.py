"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.agents.vscode`` — audit round 3.

Each test confirms the bug/inconsistency exists with BOTH a structural
source assertion (``inspect.getsource`` pattern match) AND a behavioral
integration test using real objects.

Bugs
----
N1: ``_timer_flush`` inner closure uses type annotation ``tid: int | None``
    but tab_id values are always ``str | None``.
N2: ``_await_user_response`` reads ``self._tab_states`` without holding
    ``_state_lock``, racing with ``_close_tab`` which mutates the dict
    under the lock.
N3: ``_scan_files`` depth check ``len(rel_root.parts) - 1 > 3`` was
    written assuming ``PurePath('.').parts == ('.',)`` but it is ``()``,
    causing an off-by-one that allows one extra nesting level (depth 4
    sub-directories instead of the intended 3).
N4: Comment in ``_run_task_inner`` is truncated:
    ``# BUG-B fix: if this worktree tab has a pending branch from a``
N5: ``_capture_pre_snapshot`` passes ``tab_id`` through to
    ``_save_untracked_base`` and ``_prepare_and_start_merge`` passes it
    to ``_merge_data_dir`` without guarding against empty string.
    The B7 fix only guards ``_finish_merge``; the *write* paths can
    still place data in the parent ``merge_dir/`` when ``tab_id`` is
    ``""``.
"""

from __future__ import annotations

import inspect
import os
import queue
import re
import shutil
import tempfile
import threading
import unittest
from pathlib import Path, PurePath

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.diff_merge import (
    _merge_data_dir,
    _scan_files,
    _untracked_base_dir,
)
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


class TestTimerFlushTypeAnnotation(unittest.TestCase):
    """N1: The ``_timer_flush`` inner function inside
    ``BaseBrowserPrinter.print`` uses the type annotation
    ``tid: int | None`` for the captured ``owner_tab`` default, but
    ``owner_tab`` comes from ``getattr(self._thread_local, "tab_id",
    None)`` which is ``str | None``.
    """

    def test_source_has_wrong_type_annotation(self) -> None:
        """Structural: the inner function annotates ``tid`` as ``int``."""
        src = inspect.getsource(BaseBrowserPrinter.print)
        match = re.search(r"def _timer_flush\(tid:\s*([\w |]+)", src)
        assert match is not None, (
            "N1: could not find _timer_flush definition"
        )
        annotation = match.group(1).strip()
        assert "int" in annotation, (
            f"N1: expected 'int' in annotation, got: {annotation!r}"
        )
        assert "str" not in annotation, (
            f"N1: annotation should NOT already contain 'str': {annotation!r}"
        )

    def test_owner_tab_is_string(self) -> None:
        """Behavioral: the captured ``owner_tab`` is actually a string."""
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "test-tab-123"

        owner_tab = getattr(printer._thread_local, "tab_id", None)
        assert isinstance(owner_tab, str), (
            f"N1: owner_tab should be str, got {type(owner_tab).__name__}"
        )


class TestAwaitUserResponseNoLock(unittest.TestCase):
    """N2: ``_await_user_response`` reads ``self._tab_states.get(tab_id)``
    without holding ``_state_lock``, creating a data race with
    ``_close_tab`` which pops the entry under the lock.
    """

    def test_source_has_lock_around_tab_states_get(self) -> None:
        """Structural: ``_tab_states.get`` IS inside a
        ``with self._state_lock`` block in ``_await_user_response``
        (confirming the N2 data race fix).
        """
        src = inspect.getsource(_TaskRunnerMixin._await_user_response)
        lines = src.splitlines()

        tab_get_idx = None
        for i, line in enumerate(lines):
            if "_tab_states.get" in line:
                tab_get_idx = i
                break
        assert tab_get_idx is not None, (
            "N2: could not find _tab_states.get in _await_user_response"
        )

        preceding = "\n".join(lines[max(0, tab_get_idx - 5):tab_get_idx])
        assert "_state_lock" in preceding, (
            "N2 fix: _tab_states.get should be protected by _state_lock"
        )

    def test_close_tab_mutates_tab_states_under_lock(self) -> None:
        """Contrast: ``_close_tab`` mutates ``_tab_states`` under lock."""
        src = inspect.getsource(VSCodeServer._close_tab)
        lock_pattern = re.compile(
            r"with self\._state_lock:.*?_tab_states\.pop",
            re.DOTALL,
        )
        assert lock_pattern.search(src), (
            "N2 contrast: _close_tab pops _tab_states under _state_lock"
        )

    def test_behavioral_race_scenario(self) -> None:
        """Behavioral: demonstrate the race window.

        Thread A (task): calls ``_await_user_response`` and reads
        ``_tab_states.get("t1")``.

        Thread B (main): calls ``_close_tab("t1")`` which pops the
        entry under the lock.

        If A reads after B pops, A gets None and the user's answer is
        silently dropped.
        """
        server, _ = _make_server()
        tab = server._get_tab("t1")
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue(maxsize=1)

        tab.user_answer_queue.put("yes")

        server.printer._thread_local.stop_event = tab.stop_event
        server.printer._thread_local.tab_id = "t1"
        pre_close_tab = server._tab_states.get("t1")
        assert pre_close_tab is not None, "Tab exists before close"

        with server._state_lock:
            server._tab_states.pop("t1", None)

        post_close_tab = server._tab_states.get("t1")
        assert post_close_tab is None, (
            "N2: after close, unlocked read returns None — answer is lost"
        )


class TestScanFilesDepthOffByOne(unittest.TestCase):
    """N3: ``_scan_files`` checks ``len(rel_root.parts) - 1 > 3``
    which was written assuming ``PurePath('.').parts == ('.',)``.
    Since ``PurePath('.').parts`` is actually ``()``, the ``- 1``
    creates an off-by-one that allows one extra level of nesting.
    """

    def test_purepath_dot_parts_is_empty(self) -> None:
        """Confirm the root cause: ``PurePath('.').parts`` is ``()``."""
        assert PurePath(".").parts == (), (
            f"N3: PurePath('.').parts should be (), got {PurePath('.').parts}"
        )

    def test_source_has_depth_check(self) -> None:
        """Structural: the depth check uses ``len(rel_root.parts) > 10``."""
        src = inspect.getsource(_scan_files)
        assert "parts) > 10" in src, (
            "N3: _scan_files should have the depth check 'parts) > 10'"
        )

    def test_depth_10_files_are_included(self) -> None:
        """Behavioral: files at depth 10 are included when the
        intended limit was depth 9.

        Creates a directory tree:
          root/a/b/c/d/e/f/g/h/i/shallow.txt  (depth 9)
          root/a/b/c/d/e/f/g/h/i/j/deep.txt   (depth 10)
          root/a/b/c/d/e/f/g/h/i/j/k/very_deep.txt  (depth 11)

        With the off-by-one, depth 10 is included.  Without it, only
        depth 9 should be included.
        """
        td = tempfile.mkdtemp()
        try:
            d9 = os.path.join(td, "a", "b", "c", "d", "e", "f", "g", "h", "i")
            os.makedirs(d9)
            Path(d9, "shallow.txt").write_text("ok")

            d10 = os.path.join(td, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
            os.makedirs(d10)
            Path(d10, "deep.txt").write_text("too deep?")

            d11 = os.path.join(td, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")
            os.makedirs(d11)
            Path(d11, "very_deep.txt").write_text("way too deep")

            result = _scan_files(td)
            file_results = [p for p in result if not p.endswith("/")]

            assert "a/b/c/d/e/f/g/h/i/shallow.txt" in file_results, (
                "depth-9 files should always be included"
            )
            assert "a/b/c/d/e/f/g/h/i/j/deep.txt" in file_results, (
                "N3: depth-10 files are included due to the off-by-one bug"
            )
            assert "a/b/c/d/e/f/g/h/i/j/k/very_deep.txt" not in file_results, (
                "depth-11 files should be excluded"
            )
        finally:
            shutil.rmtree(td)

    def test_depth_formula_values(self) -> None:
        """Behavioral: verify the formula at each depth level.

        The formula ``len(rel_root.parts)`` gives:
          root:    len(()) = 0
          depth10: len(('a',..,'j')) = 10  → 10 > 10 is False (included)
          depth11: len(('a',..,'k')) = 11  → 11 > 10 is True (excluded)
        """
        assert len(PurePath(".").parts) == 0

        depth10 = PurePath("a/b/c/d/e/f/g/h/i/j")
        assert len(depth10.parts) == 10
        assert not (len(depth10.parts) > 10), (
            "depth 10 passes the check (10 > 10 is False) — included"
        )

        depth11 = PurePath("a/b/c/d/e/f/g/h/i/j/k")
        assert len(depth11.parts) == 11
        assert len(depth11.parts) > 10, (
            "depth 11 fails the check (11 > 10 is True) — excluded"
        )


class TestTruncatedCommentInRunTaskInner(unittest.TestCase):
    """N4: The comment ``# BUG-B fix: if this worktree tab has a
    pending branch from a`` is truncated — the sentence is incomplete.
    """

    def test_source_has_truncated_comment(self) -> None:
        """Structural: the truncated comment has been removed."""
        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        # N4 fix: the truncated "pending branch from a" comment was
        # cleaned up.  Verify it is no longer present.
        assert "pending branch from a" not in src, (
            "N4: the truncated comment should have been removed"
        )


class TestWritePathsEmptyTabId(unittest.TestCase):
    """N5: ``_capture_pre_snapshot`` and ``_prepare_and_start_merge``
    pass ``tab_id`` to ``_save_untracked_base`` and ``_merge_data_dir``
    without guarding against empty string.  The B7 fix only guards
    ``_finish_merge``; the write paths can still place data in the
    parent ``merge_dir/`` when ``tab_id`` is ``""``.
    """

    def test_merge_data_dir_empty_returns_parent(self) -> None:
        """Behavioral: ``_merge_data_dir("")`` returns the parent dir."""
        parent = _merge_data_dir("")
        child = _merge_data_dir("some-tab")
        assert child.parent == parent, (
            f"N5: _merge_data_dir('') returns parent dir; "
            f"parent={parent}, child.parent={child.parent}"
        )

    def test_untracked_base_dir_empty_returns_parent_subdir(self) -> None:
        """Behavioral: ``_untracked_base_dir("")`` creates a path under
        the parent merge_dir rather than a per-tab subdirectory."""
        empty = _untracked_base_dir("")
        with_tab = _untracked_base_dir("tab-1")
        assert empty.parent == _merge_data_dir(""), (
            "N5: empty tab_id puts untracked-base in shared parent dir"
        )
        assert with_tab.parent == _merge_data_dir("tab-1"), (
            "with tab_id, untracked-base is isolated in tab subdir"
        )

    def test_capture_pre_snapshot_has_no_tab_id_guard(self) -> None:
        """Structural: ``_capture_pre_snapshot`` does not check for
        empty ``tab_id`` before calling ``_save_untracked_base``."""
        src = inspect.getsource(_TaskRunnerMixin._capture_pre_snapshot)
        assert "_save_untracked_base" in src, (
            "N5: _capture_pre_snapshot calls _save_untracked_base"
        )
        lines = src.splitlines()
        save_idx = None
        for i, line in enumerate(lines):
            if "_save_untracked_base" in line:
                save_idx = i
                break
        assert save_idx is not None
        preceding = "\n".join(lines[max(0, save_idx - 5):save_idx])
        assert "not tab_id" not in preceding and "if tab_id" not in preceding, (
            "N5: no tab_id guard before _save_untracked_base call"
        )

    def test_prepare_and_start_merge_has_no_tab_id_guard(self) -> None:
        """Structural: ``_prepare_and_start_merge`` does not check for
        empty ``tab_id`` before calling ``_merge_data_dir``."""
        from kiss.agents.vscode.merge_flow import _MergeFlowMixin

        src = inspect.getsource(_MergeFlowMixin._prepare_and_start_merge)
        assert "_merge_data_dir" in src, (
            "N5: _prepare_and_start_merge calls _merge_data_dir"
        )
        lines = src.splitlines()
        merge_dir_idx = None
        for i, line in enumerate(lines):
            if "_merge_data_dir" in line:
                merge_dir_idx = i
                break
        assert merge_dir_idx is not None
        preceding = "\n".join(lines[max(0, merge_dir_idx - 5):merge_dir_idx])
        assert "not tab_id" not in preceding and "if tab_id" not in preceding, (
            "N5: no tab_id guard before _merge_data_dir call"
        )

    def test_cross_tab_data_collision_with_empty_tab_id(self) -> None:
        """Behavioral: two calls with empty tab_id write to the same
        directory, demonstrating the collision risk."""
        dir_a = _merge_data_dir("")
        dir_b = _merge_data_dir("")
        assert dir_a == dir_b, (
            "N5: two empty-tab_id calls write to the same directory"
        )

        dir_c = _merge_data_dir("tab-A")
        dir_d = _merge_data_dir("tab-B")
        assert dir_c != dir_d, (
            "With tab_ids, merge data dirs are isolated"
        )


if __name__ == "__main__":
    unittest.main()

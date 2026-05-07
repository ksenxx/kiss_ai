"""Integration tests for redundancy fixes in kiss/agents/vscode/.

Redundancy 1 — ``_cmd_run`` in ``commands.py`` duplicated the
get-or-create tab pattern that ``_get_tab`` already provides.  After
the fix, ``_cmd_run`` delegates to ``_get_tab`` and no inline
``_TabState(...)`` construction remains.

Redundancy 2 — The autocommit-prompt broadcast pattern
(``_main_dirty_files`` → ``autocommit_prompt`` broadcast) was
copy-pasted in ``task_runner.py::_run_task_inner`` and
``merge_flow.py::_finish_merge``.  After the fix, both call a single
``_broadcast_autocommit_prompt`` method defined in ``merge_flow.py``.
"""

from __future__ import annotations

import inspect
import threading

# ── Redundancy 1: _cmd_run must not inline _TabState construction ──


class TestCmdRunUsesGetTab:
    """Verify that _cmd_run delegates to _get_tab instead of duplicating it."""

    def test_cmd_run_does_not_construct_tab_state_inline(self) -> None:
        """The body of _cmd_run must not contain '_TabState(' — it
        should call self._get_tab instead."""
        from kiss.agents.vscode.commands import _CommandsMixin

        src = inspect.getsource(_CommandsMixin._cmd_run)
        assert "_TabState(" not in src, (
            "_cmd_run still constructs _TabState inline; "
            "should use self._get_tab(tab_id)"
        )

    def test_cmd_run_calls_get_tab(self) -> None:
        """The body of _cmd_run must contain a call to self._get_tab."""
        from kiss.agents.vscode.commands import _CommandsMixin

        src = inspect.getsource(_CommandsMixin._cmd_run)
        assert "_get_tab(" in src, (
            "_cmd_run does not call _get_tab; "
            "it should delegate tab creation to _get_tab"
        )

    def test_get_tab_creates_tab_for_new_id(self) -> None:
        """_get_tab must create a new _TabState when tab_id is unknown."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer.__new__(VSCodeServer)
        server._tab_states = {}
        server._default_model = "test-model"
        server._state_lock = threading.Lock()
        tab = server._get_tab("new-tab")
        assert tab is not None
        assert tab.selected_model == "test-model"
        assert "new-tab" in server._tab_states

    def test_get_tab_returns_existing_tab(self) -> None:
        """_get_tab must return the same object for repeated calls."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer.__new__(VSCodeServer)
        server._tab_states = {}
        server._default_model = "test-model"
        server._state_lock = threading.Lock()
        tab1 = server._get_tab("t1")
        tab2 = server._get_tab("t1")
        assert tab1 is tab2


# ── Redundancy 2: autocommit prompt broadcast extracted ──


class TestAutocommitPromptExtracted:
    """Verify the autocommit-prompt broadcast is a single shared method."""

    def test_broadcast_autocommit_prompt_exists_on_merge_flow_mixin(self) -> None:
        """_MergeFlowMixin must define _broadcast_autocommit_prompt."""
        from kiss.agents.vscode.merge_flow import _MergeFlowMixin

        assert hasattr(_MergeFlowMixin, "_broadcast_autocommit_prompt"), (
            "_MergeFlowMixin is missing _broadcast_autocommit_prompt; "
            "the duplicated pattern should be extracted here"
        )

    def test_finish_merge_calls_broadcast_autocommit_prompt(self) -> None:
        """_finish_merge must call _broadcast_autocommit_prompt."""
        from kiss.agents.vscode.merge_flow import _MergeFlowMixin

        src = inspect.getsource(_MergeFlowMixin._finish_merge)
        assert "_broadcast_autocommit_prompt(" in src, (
            "_finish_merge should call _broadcast_autocommit_prompt "
            "instead of inlining the pattern"
        )

    def test_finish_merge_does_not_inline_main_dirty_files(self) -> None:
        """_finish_merge must NOT call _main_dirty_files directly.

        The dirty-files check + broadcast should be delegated entirely
        to _broadcast_autocommit_prompt.
        """
        from kiss.agents.vscode.merge_flow import _MergeFlowMixin

        src = inspect.getsource(_MergeFlowMixin._finish_merge)
        assert "_main_dirty_files" not in src, (
            "_finish_merge still calls _main_dirty_files inline; "
            "should use _broadcast_autocommit_prompt"
        )

    def test_run_task_inner_calls_broadcast_autocommit_prompt(self) -> None:
        """_run_task_inner must call _broadcast_autocommit_prompt."""
        from kiss.agents.vscode.task_runner import _TaskRunnerMixin

        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        assert "_broadcast_autocommit_prompt(" in src, (
            "_run_task_inner should call _broadcast_autocommit_prompt "
            "instead of inlining the pattern"
        )

    def test_run_task_inner_does_not_inline_main_dirty_files(self) -> None:
        """_run_task_inner must NOT call _main_dirty_files directly."""
        from kiss.agents.vscode.task_runner import _TaskRunnerMixin

        src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        assert "_main_dirty_files" not in src, (
            "_run_task_inner still calls _main_dirty_files inline; "
            "should use _broadcast_autocommit_prompt"
        )

    def test_only_one_definition_of_autocommit_prompt_pattern(self) -> None:
        """The autocommit_prompt broadcast literal should appear only in
        _broadcast_autocommit_prompt, not in _finish_merge or
        _run_task_inner."""
        from kiss.agents.vscode.merge_flow import _MergeFlowMixin
        from kiss.agents.vscode.task_runner import _TaskRunnerMixin

        fm_src = inspect.getsource(_MergeFlowMixin._finish_merge)
        tr_src = inspect.getsource(_TaskRunnerMixin._run_task_inner)
        literal = '"autocommit_prompt"'
        assert literal not in fm_src, (
            f"_finish_merge still has {literal} inline"
        )
        assert literal not in tr_src, (
            f"_run_task_inner still has {literal} inline"
        )

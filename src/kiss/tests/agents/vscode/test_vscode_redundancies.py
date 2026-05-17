"""Integration tests for redundancy fixes in kiss/agents/vscode/.

Redundancy 1 — ``_cmd_run`` and ``_get_tab`` both implemented the
get-or-create tab pattern.  Naively delegating ``_cmd_run`` to
``_get_tab`` introduced a TOCTOU bug (see audit round 4, A6): the
lock would be released between get-or-create and the alive-check /
thread-start, allowing a concurrent ``_close_tab`` to drop the tab
from ``_running_agent_states``.  The correct shape inlines the get-or-create
inside the same ``_state_lock`` block that performs the alive check
and starts the thread, so no other helper that would re-acquire the
lock is called.  ``_RunningAgentState`` import is shared at module top, not
duplicated locally.

Redundancy 2 — The autocommit-prompt broadcast pattern
(``_main_dirty_files`` → ``autocommit_prompt`` broadcast) was
copy-pasted in ``task_runner.py::_run_task_inner`` and
``merge_flow.py::_finish_merge``.  After the fix, both call a single
``_broadcast_autocommit_prompt`` method defined in ``merge_flow.py``.
"""

from __future__ import annotations

import inspect
import re
import threading

# ── Redundancy 1: _cmd_run must keep get-or-create + alive-check + thread
#    start in a single _state_lock block (no nested re-acquisition) ──


class TestCmdRunUsesGetTab:
    """Verify that _cmd_run uses a single _state_lock block."""

    def test_cmd_run_does_not_call_get_tab(self) -> None:
        """_cmd_run must not call self._get_tab.

        Calling ``_get_tab`` would acquire and release ``_state_lock``
        once, then ``_cmd_run`` re-acquires it for the alive check and
        thread start — opening a TOCTOU window where ``_close_tab`` can
        drop the tab from ``_running_agent_states`` between the two lock blocks.
        The audit-round-4 A6 fix mandates a single lock block.
        """
        from kiss.agents.vscode.commands import _CommandsMixin

        src = inspect.getsource(_CommandsMixin._cmd_run)
        assert "_get_tab(" not in src, (
            "_cmd_run must not call _get_tab; the get-or-create logic "
            "must be inlined inside the same _state_lock block as the "
            "alive check and thread start (audit-round-4 A6 fix)."
        )

    def test_cmd_run_uses_single_state_lock_block(self) -> None:
        """_cmd_run must contain exactly one ``with self._state_lock:`` block."""
        from kiss.agents.vscode.commands import _CommandsMixin

        src = inspect.getsource(_CommandsMixin._cmd_run)
        count = len(re.findall(r"with self\._state_lock:", src))
        assert count == 1, (
            f"_cmd_run must use exactly one _state_lock block, found {count}"
        )

    def test_cmd_run_uses_module_level_tab_state_import(self) -> None:
        """_cmd_run must use the module-level _RunningAgentState import, not a
        local re-import inside the function body."""
        from kiss.agents.vscode.commands import _CommandsMixin

        src = inspect.getsource(_CommandsMixin._cmd_run)
        assert "from kiss.agents.sorcar.running_agent_state import _RunningAgentState" not in src, (
            "_cmd_run must not redundantly re-import _RunningAgentState; the "
            "module-level import in commands.py is sufficient."
        )

    def test_get_tab_creates_tab_for_new_id(self) -> None:
        """_get_tab must create a new _RunningAgentState when tab_id is unknown."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer.__new__(VSCodeServer)
        server._running_agent_states.clear()
        server._default_model = "test-model"
        server._state_lock = threading.Lock()
        tab = server._get_tab("new-tab")
        assert tab is not None
        assert tab.selected_model == "test-model"
        assert "new-tab" in server._running_agent_states

    def test_get_tab_returns_existing_tab(self) -> None:
        """_get_tab must return the same object for repeated calls."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer.__new__(VSCodeServer)
        server._running_agent_states.clear()
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

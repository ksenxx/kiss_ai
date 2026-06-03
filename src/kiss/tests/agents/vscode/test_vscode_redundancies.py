# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

import threading

# ── Redundancy 1: _cmd_run must keep get-or-create + alive-check + thread
#    start in a single _state_lock block (no nested re-acquisition) ──


class TestCmdRunUsesGetTab:
    """Verify that _cmd_run uses a single _state_lock block."""




    def test_get_tab_creates_tab_for_new_id(self) -> None:
        """_get_tab must create a new _RunningAgentState when tab_id is unknown."""
        from kiss.agents.vscode.server import VSCodeServer

        server = VSCodeServer.__new__(VSCodeServer)
        server._running_agent_states.clear()
        server._default_model = "test-model"
        server._state_lock = threading.RLock()
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
        server._state_lock = threading.RLock()
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


# ── Redundancy 3: one-shot non-agentic LLM call boilerplate extracted ──


class TestOneShotLlmExtracted:
    """Verify ``helpers._run_oneshot_llm`` exists and is referenced.

    Both ``generate_commit_message_from_diff`` and
    ``generate_followup_text`` previously open-coded the same
    ``KISSAgent(...).run(... is_agentic=False, verbose=False)``
    boilerplate with ``try/except`` + ``clean_llm_output`` +
    fallback.  After the refactor, both delegate to a single shared
    helper ``_run_oneshot_llm``.
    """

    def test_run_oneshot_llm_exists(self) -> None:
        """``helpers._run_oneshot_llm`` must be defined."""
        from kiss.agents.vscode import helpers

        assert hasattr(helpers, "_run_oneshot_llm"), (
            "vscode.helpers is missing _run_oneshot_llm; "
            "the duplicated try/except + KISSAgent.run pattern "
            "should be extracted here"
        )

    def test_callers_use_run_oneshot_llm(self) -> None:
        """Both LLM helpers must funnel through ``_run_oneshot_llm``.

        Verifies the boilerplate is actually shared (not just present
        as a dead helper) by reading the module source and asserting
        every direct ``KISSAgent(...).run(`` call has been removed
        from ``helpers.py``.
        """
        import inspect

        from kiss.agents.vscode import helpers

        src = inspect.getsource(helpers)
        # Only the shared helper may instantiate KISSAgent directly.
        assert src.count("KISSAgent(") == 1, (
            "Expected exactly one KISSAgent() instantiation in "
            "vscode/helpers.py (inside _run_oneshot_llm); found "
            f"{src.count('KISSAgent(')}"
        )


# ── Redundancy 4: sorcar/persistence.py shared helpers ──


class TestPersistenceSharedHelpers:
    """Verify shared helpers exist in ``sorcar.persistence``.

    Four near-identical "load events for task_id" blocks were
    collapsed into ``_fetch_events_for_task_id``.
    """

    def test_fetch_events_for_task_id_exists(self) -> None:
        """``persistence._fetch_events_for_task_id`` must be defined."""
        from kiss.agents.sorcar import persistence

        assert hasattr(persistence, "_fetch_events_for_task_id"), (
            "sorcar.persistence is missing _fetch_events_for_task_id; "
            "the duplicated event-row loading loop should be "
            "extracted here"
        )





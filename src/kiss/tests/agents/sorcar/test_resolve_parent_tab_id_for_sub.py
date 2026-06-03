# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for
:meth:`VSCodeServer._resolve_parent_tab_id_for_sub`.

The helper is responsible for populating ``parent_tab_id`` on the
``openSubagentTab`` broadcast.  A blank return value breaks the
cascade-close walk in ``media/main.js`` (closing the parent fails
to close its sub-agent tabs).  These tests pin the lookup tiers
(task-id match → chat-id match → synthetic-tab-id parse →
warning + ``""``) so future regressions surface immediately.
"""

from __future__ import annotations

import logging

import pytest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer


def _clear_states() -> None:
    _RunningAgentState.running_agent_states.clear()


def _register(
    tab_id: str,
    *,
    is_subagent: bool = False,
    chat_id: str = "",
    task_history_id: int | None = None,
) -> _RunningAgentState:
    st = _RunningAgentState(tab_id, "test-model")
    st.is_subagent = is_subagent
    st.chat_id = chat_id
    st.task_history_id = task_history_id
    _RunningAgentState.running_agent_states[tab_id] = st
    return st


class TestResolveParentTabIdForSub:
    """Each tier of the lookup must work in isolation."""

    def setup_method(self) -> None:
        _clear_states()

    def teardown_method(self) -> None:
        _clear_states()

    def test_task_id_match_via_task_history_id(self) -> None:
        """Tier 1: parent's ``task_history_id`` equals lookup id."""
        server = VSCodeServer()
        _register("parent-tab", task_history_id=42, chat_id="c1")
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=42, chat_id="c1", sub_tab_id="task-42__sub_0",
        )
        assert out == "parent-tab"

    def test_task_id_match_skips_subagent_states(self) -> None:
        """Sub-agent states must never be returned as the parent."""
        server = VSCodeServer()
        _register(
            "sub-tab", is_subagent=True, task_history_id=42, chat_id="c1",
        )
        _register("parent-tab", task_history_id=42, chat_id="c1")
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=42, chat_id="c1", sub_tab_id="task-42__sub_0",
        )
        assert out == "parent-tab"

    def test_chat_id_fallback_when_task_id_not_found(self) -> None:
        """Tier 2: when no state matches the task id, fall back to
        the unique non-subagent state sharing the sub-agent's
        ``chat_id``."""
        server = VSCodeServer()
        _register("parent-tab", task_history_id=None, chat_id="c2")
        _register(
            "other-sub", is_subagent=True, task_history_id=None,
            chat_id="c2",
        )
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=999, chat_id="c2", sub_tab_id="x__sub_0",
        )
        assert out == "parent-tab"

    def test_chat_id_fallback_ambiguous_two_parents(self) -> None:
        """Two non-subagent states with the same ``chat_id`` is
        ambiguous; the helper must NOT guess.  It falls through to
        the next tier (synthetic-tab-id parse, then warning)."""
        server = VSCodeServer()
        _register("parent-a", task_history_id=None, chat_id="c3")
        _register("parent-b", task_history_id=None, chat_id="c3")
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=None, chat_id="c3", sub_tab_id="x__sub_0",
        )
        assert out == ""

    def test_chat_id_fallback_with_no_chat_id(self) -> None:
        """Empty ``chat_id`` must NOT trigger the chat-id tier (which
        would match every state with default ``chat_id=""``)."""
        server = VSCodeServer()
        _register("parent-tab", task_history_id=None, chat_id="")
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=None, chat_id="", sub_tab_id="no-suffix",
        )
        assert out == ""

    def test_synthetic_tab_id_parse(self) -> None:
        """Tier 3: when the sub-tab id is
        ``f"{parent_tab_id}__sub_{n}"`` and the prefix matches a
        known non-subagent ``tab_id``, return that prefix."""
        server = VSCodeServer()
        _register("real-parent-id", task_history_id=None, chat_id="")
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=None, chat_id="",
            sub_tab_id="real-parent-id__sub_3",
        )
        assert out == "real-parent-id"

    def test_synthetic_tab_id_parse_unknown_prefix(self) -> None:
        """Synthetic prefix that doesn't match any registered tab
        must fall through to the warning tier."""
        server = VSCodeServer()
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=None, chat_id="",
            sub_tab_id="ghost-parent__sub_0",
        )
        assert out == ""

    def test_all_tiers_fail_emits_warning_and_blank(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When every tier fails, the helper logs a WARNING (so the
        silent cascade-close-bug surfaces in logs) and returns ``""``.

        ``""`` is still the only safe return — the frontend treats
        it as "no parent" — but the warning makes the failure
        observable."""
        server = VSCodeServer()
        with caplog.at_level(
            logging.WARNING, logger="kiss.agents.vscode.server",
        ):
            out = server._resolve_parent_tab_id_for_sub(
                parent_task_id=12345, chat_id="missing-chat",
                sub_tab_id="no-suffix",
            )
        assert out == ""
        msgs = [r.getMessage() for r in caplog.records]
        assert any("resolve parent tab id" in m for m in msgs), msgs

    def test_task_id_match_preferred_over_chat_id(self) -> None:
        """Tier 1 wins even when Tier 2 would match a different
        state."""
        server = VSCodeServer()
        _register(
            "correct-parent", task_history_id=7, chat_id="other-chat",
        )
        _register(
            "wrong-parent", task_history_id=None, chat_id="lookup-chat",
        )
        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=7, chat_id="lookup-chat",
            sub_tab_id="x__sub_0",
        )
        assert out == "correct-parent"

    def test_task_id_match_via_last_task_id_on_agent(self) -> None:
        """``agent._last_task_id`` (set at run-start, before
        ``task_history_id`` is populated) must also be honored."""

        class _StubAgent:
            _last_task_id: int | None = None

        server = VSCodeServer()
        st = _register(
            "running-parent", task_history_id=None, chat_id="c4",
        )
        stub = _StubAgent()
        stub._last_task_id = 88
        st.agent = stub  # type: ignore[assignment]

        out = server._resolve_parent_tab_id_for_sub(
            parent_task_id=88, chat_id="c4",
            sub_tab_id="task-88__sub_0",
        )
        assert out == "running-parent"

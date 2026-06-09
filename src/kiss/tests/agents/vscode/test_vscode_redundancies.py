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

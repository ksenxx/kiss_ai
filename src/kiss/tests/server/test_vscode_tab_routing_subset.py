# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for a subset of tab-routing fixes in VSCode server.

Targets only the five bug sites fixed in this change:
  1. ``_handle_command`` unknown-command error — carries ``tabId`` from cmd.
  2. ``run()`` generic-exception error — carries ``tabId`` from parsed cmd.
  3. ``_handle_worktree_action`` — ``worktree_progress`` carries tab.
  4. ``_get_adjacent_task`` — ``adjacent_task_events`` carries tab.

No mocks: uses a real ``VSCodeServer`` with its ``printer.broadcast``
replaced by a capture-list helper.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Build a VSCodeServer whose broadcasts are captured in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestUnknownCommandErrorRouted(unittest.TestCase):
    def test_unknown_command_error_carries_tab_id(self) -> None:
        server, events = _make_server()
        server._handle_command({"type": "bogusCmd", "tabId": "t-7"})
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert err[0].get("tabId") == "t-7"
        assert "Unknown command" in err[0]["text"]

    def test_unknown_command_without_tab_id_omits_field(self) -> None:
        server, events = _make_server()
        server._handle_command({"type": "bogusCmd"})
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert "tabId" not in err[0]


class TestWorktreeProgressRouted(unittest.TestCase):
    def test_worktree_progress_carries_tab_id(self) -> None:
        import tempfile
        from pathlib import Path as _Path

        from kiss.agents.sorcar.git_worktree import GitWorktree

        server, events = _make_server()
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        state = agent_state.AgentState(
            "routing-t13", agent=agent, tab_id="t-13", server_owned=True,
        )
        state.use_worktree = True
        agent_state.register(state)
        try:
            with tempfile.TemporaryDirectory() as td:
                agent._wt = GitWorktree(
                    repo_root=_Path(td),
                    branch="kiss/wt-x",
                    original_branch="main",
                    wt_dir=_Path(td) / ".kiss-worktrees" / "kiss_wt-x",
                )

                def fake_merge() -> str:
                    return "Successfully merged"

                agent.merge = fake_merge  # type: ignore[assignment]

                result = server._handle_worktree_action(
                    "merge", tab_id="t-13",
                )
                assert result["success"] is True
        finally:
            agent_state.agent_states.clear()

        wp = [e for e in events if e.get("type") == "worktree_progress"]
        assert len(wp) == 1
        assert wp[0].get("tabId") == "t-13"


class TestAdjacentTaskRouted(unittest.TestCase):
    def test_adjacent_task_events_carries_tab_id(self) -> None:
        server, events = _make_server()
        server._get_adjacent_task(
            chat_id="does-not-exist",
            task_id=None,
            direction="prev",
            tab_id="t-17",
        )
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert ate[0].get("tabId") == "t-17"

    def test_cmd_handler_propagates_tab_id(self) -> None:
        """`_cmd_get_adjacent_task` forwards cmd.tabId into the event."""
        server, events = _make_server()
        server._cmd_get_adjacent_task({
            "type": "getAdjacentTask",
            "tabId": "t-19",
            "taskId": None,
            "direction": "prev",
        })
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert ate[0].get("tabId") == "t-19"

    def test_no_tab_id_still_tags_event(self) -> None:
        """Empty tab_id still carries a (empty) tabId field (B4 fix).

        Previously, an empty tab_id caused the event to be emitted
        untagged, which reached every tab's frontend handler and
        overwrote whichever tab was active.  With the fix the event
        always carries a tabId so no tab mis-interprets it.
        """
        server, events = _make_server()
        server._get_adjacent_task(
            chat_id="does-not-exist",
            task_id=None,
            direction="prev",
            tab_id="",
        )
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert "tabId" in ate[0]
        assert ate[0]["tabId"] == ""


if __name__ == "__main__":
    unittest.main()

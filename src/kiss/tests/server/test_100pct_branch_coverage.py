# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Server-only branch-coverage tests extracted from
``kiss.tests.agents.third_party_agents.test_100pct_branch_coverage``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server); the
``_channel_cli`` coverage stayed behind in
tests/agents/third_party_agents.

Targets remaining uncovered branches in:
  json_printer.py (_format_tool_call, peek_recording)
  server.py (changed-files/result-summary paths)

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.conftest import seed_worktree
from kiss.tests.server.worktree_helpers import worktree_changed_files


def _register_worktree_state(tab_id: str) -> AgentState:
    """Register a server-owned worktree state for *tab_id* and return it."""
    state = AgentState(
        f"task-{tab_id}",
        agent=ChatSorcarAgent("Sorcar"),
        tab_id=tab_id,
        server_owned=True,
    )
    agent_state.register(state)
    return state


class TestFormatToolCallBranches:
    """Cover _format_tool_call branches in json_printer.py."""

    def test_format_tool_call_with_all_fields(self) -> None:
        """All optional fields present in tool_input."""
        p = JsonPrinter()
        p._thread_local.task_id = "t1"
        p.start_recording()
        p._format_tool_call("Edit", {
            "file_path": "/path/to/file.py",
            "description": "edit desc",
            "command": "some cmd",
            "content": "file content",
            "old_string": "old",
            "new_string": "new",
            "extra_param": "extra_val",
        })
        events = p.stop_recording()
        ev = events[0]
        assert ev["type"] == "tool_call"
        assert ev["name"] == "Edit"
        assert ev["path"] == "/path/to/file.py"
        assert ev["description"] == "edit desc"
        assert ev["command"] == "some cmd"
        assert ev["content"] == "file content"
        assert ev["old_string"] == "old"
        assert ev["new_string"] == "new"
        assert "extras" in ev

class TestAgentServerUncoveredBranches:
    """Cover remaining uncovered branches in AgentServer."""

    def test_get_worktree_changed_files_no_branches(self) -> None:
        """The changed-files listing is [] when no branches."""
        server = AgentServer()
        state = _register_worktree_state("0")
        try:
            assert state.agent is not None
            seed_worktree(state.agent, None)
            assert worktree_changed_files(server, "0") == []
        finally:
            agent_state.unregister(state.task_id, state)

class TestBrowserPrinterPeekRecording:
    """Cover peek_recording for empty/non-existent recording."""

    def test_peek_active_recording(self) -> None:
        """peek_recording returns current events without stopping."""
        p = JsonPrinter()
        p._thread_local.task_id = "t1"
        p.start_recording()
        p.broadcast({"type": "text_delta", "text": "hello"})
        events = p.peek_recording_for_task(p.thread_binding())
        assert len(events) == 1
        p.broadcast({"type": "text_delta", "text": " world"})
        events2 = p.peek_recording_for_task(p.thread_binding())
        assert len(events2) == 1
        assert events2[0]["text"] == "hello world"
        p.stop_recording()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a run the printer bridge registers itself frees its per-task state.

Regression (audit V7): the daemon's printer keeps per-task state for
every run it prints (usage offsets, the ``Bash`` stream buffer, the
agent's model pick, the tool-call start, the image budget, the event
recording).  Only the task runner freed it, with ``cleanup_task``.  The
daemon's side-channel runs (the ``/update`` agent of
:mod:`kiss.server.task_update`, the merge-conflict resolver) share the
printer without the runner, so each of them left its state behind for
the daemon's lifetime, and a straggler event of the finished run was
still accepted.

The runs below drive a real ``JsonPrinter`` through the bridge calls
``ChatSorcarAgent.run`` makes, in its order (``agent_task_allocated``,
``start_recording``, the run's events, then ``agent_task_finished`` and
``stop_recording`` in its ``finally``), for a real ``ChatSorcarAgent``,
with a real streaming ``Bash`` call that writes a fresh image.
"""

from __future__ import annotations

import functools
import shlex
from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.shell_scan import bash_path
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter

_PER_TASK_STATE = (
    "_bash_states",
    "_budget_offsets",
    "_tokens_offsets",
    "_steps_offsets",
    "_task_model_override",
    "_tool_call_started",
    "_image_b64_used",
    "_recordings",
    "_retired_recordings",
)
"""The printer's per-task dicts a run fills, all keyed by its task id."""


def _run_events(printer: JsonPrinter, key: str, image: Path) -> None:
    """Print one run's events on the calling thread, bound to task *key*.

    Usage offsets, the agent's model pick, a ``Bash`` tool call whose
    streamed output names a PNG file the command wrote (so the result
    embeds it and charges the task's image budget), and a last tool
    call left unanswered, as by a run stopped mid-tool.  The command
    quotes *image* (the shell's spelling of it for the write) and
    prints it in double quotes, so a path with spaces is read whole.
    """
    printer.start_recording()
    printer.set_usage_offsets(key, 0.5, 100, 3)
    printer.broadcast_agent_model_pick("sweep1030-model", "", key)
    written = shlex.quote(bash_path(str(image)))
    shown = shlex.quote(f'"{image}"')
    command = f"printf '\\211PNG sweep1030' > {written}; echo {shown}"
    tool_input = {"command": command, "description": "write an image"}
    printer.print("Bash", type="tool_call", tool_input=tool_input)
    tools = UsefulTools(stream_callback=functools.partial(printer.print, type="bash_stream"))
    output = tools.Bash(command, description="write an image", timeout_seconds=30)
    printer.print(output, type="tool_result", tool_name="Bash", tool_input=tool_input)
    printer.print("Bash", type="tool_call", tool_input={"command": "sleep 60"})


def _held_state(printer: JsonPrinter, key: str) -> list[str]:
    """Return the names of the per-task dicts that still hold *key*."""
    return [name for name in _PER_TASK_STATE if key in getattr(printer, name)]


def test_side_channel_run_frees_its_printer_state(tmp_path: Path) -> None:
    """An ``/update``-style run leaves nothing keyed by its task id, and refuses stragglers."""
    printer = JsonPrinter()
    agent = ChatSorcarAgent("Task update")
    agent._tab_id = "task-7__update-1"
    agent._subagent_info = {
        "parent_task_id": "7", "parent_tab_id": "tab-7", "reviewer": False,
        "side_channel": True,
    }
    key = "sweep1030-side-channel"
    printer._thread_local.task_id = key
    printer.agent_task_allocated(agent, key, "chat-sweep1030")
    try:
        state = agent_state.get(key)
        assert state is not None and state.agent is agent and not state.server_owned
        _run_events(printer, key, tmp_path / "a shot.png")
        # The run filled every per-task dict, so the checks below can fail.
        assert _held_state(printer, key) == [
            name for name in _PER_TASK_STATE if name != "_retired_recordings"
        ]
        assert printer._image_b64_used[key] > 0

        printer.agent_task_finished(agent, key)
        printer.stop_recording()
    finally:
        printer._thread_local.task_id = ""
        agent_state.unregister(key)

    assert agent_state.get(key) is None
    assert _held_state(printer, key) == []
    assert printer.peek_recording_for_task(key) == []
    # A straggler offset for the finished run (a late spend fold) is refused.
    printer.set_usage_offsets(key, 9.0, 900, 9)
    assert key not in printer._budget_offsets


def test_server_owned_run_keeps_its_state_for_the_runner(tmp_path: Path) -> None:
    """A run the task runner registered keeps its state until the runner frees it."""
    printer = JsonPrinter()
    agent = ChatSorcarAgent("Sorcar")
    key = "sweep1030-server-owned"
    agent_state.register(agent_state.AgentState(
        "sweep1030-provisional", agent=agent, tab_id="tab-sweep1030",
        server_owned=True, is_task_active=True,
    ))
    printer._thread_local.task_id = key
    try:
        printer.agent_task_allocated(agent, key, "chat-sweep1030")
        state = agent_state.get(key)
        assert state is not None and state.server_owned
        _run_events(printer, key, tmp_path / "a shot.png")

        printer.agent_task_finished(agent, key)
        printer.stop_recording()

        # The runner still persists the run: its offsets and its
        # recording (retired, for a viewer attaching now) are kept.
        assert agent_state.get(key) is state and state.is_task_active
        assert printer._budget_offsets[key] == 0.5
        assert key in printer._bash_states
        assert printer.peek_recording_for_task(key)
        printer.cleanup_task(key)
        assert _held_state(printer, key) == []
    finally:
        printer._thread_local.task_id = ""
        agent_state.unregister(key)

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

The last two tests interrupt the finish hook's wait for ``STATE_LOCK``:
a real Ctrl-C on the main thread, then the Stop of the real
merge-conflict resolver's tab on a pooled thread, with the server's
own guarded injector and watchdog.
"""

from __future__ import annotations

import functools
import shlex
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.merge_conflict_resolver import run_merge_sea
from kiss.agents.sorcar.shell_scan import bash_path
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.locked import ACQUIRE_SLICE_SECONDS
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.task_runner import _state_owns_thread, _TaskRunnerMixin, inject_if_owned
from kiss.tests.conftest import posix_only
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)
from kiss.tests.server.pause import Pause

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


def _hold_state_lock_across_ctrl_c(held: threading.Event, waiter: int) -> None:
    """Hold ``STATE_LOCK`` while a Ctrl-C reaches thread *waiter*, which waits for it."""
    with agent_state.STATE_LOCK:
        held.set()
        time.sleep(0.3)
        signal.pthread_kill(waiter, signal.SIGINT)
        time.sleep(0.3)


@posix_only("the test injects SIGINT with pthread_kill")
def test_ctrl_c_in_the_finish_hooks_lock_wait_still_drops_the_state() -> None:
    """Ctrl-C ends the main thread's wait for ``STATE_LOCK`` itself; the state is dropped anyway."""
    assert threading.current_thread() is threading.main_thread()
    printer = JsonPrinter()
    agent = ChatSorcarAgent("bridge-cleanup-ctrl-c")
    key = "bridge-run-ctrl-c"
    printer.agent_task_allocated(agent, key, "chat-ctrl-c")
    held = threading.Event()
    holder = threading.Thread(
        target=_hold_state_lock_across_ctrl_c, args=(held, threading.get_ident()),
    )
    # A process started with SIGINT ignored (a shell's background job) has no Ctrl-C.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt) as ctrl_c:
            holder.start()
            assert held.wait(10)
            printer.agent_task_finished(agent, key)
            time.sleep(10)  # only reached by a hook that did not wait for the lock
        assert any(entry.name == "agent_task_finished" for entry in ctrl_c.traceback)
        assert agent_state.get(key) is None
        assert not cast(Any, agent_state.STATE_LOCK)._is_owned()
    finally:
        signal.signal(signal.SIGINT, previous)
        holder.join(10)
        agent_state.unregister(key)


def _other_work(release: threading.Event) -> str:
    """Unrelated work on a pooled thread: wait for *release*; say whether a Stop cut it short."""
    try:
        while not release.wait(0.05):
            pass
    except KeyboardInterrupt:
        return "interrupted"
    return "finished"


def test_stops_in_the_finish_hooks_lock_wait_still_drop_the_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two Stops land while the merge resolver's finish hook waits for ``STATE_LOCK``.

    The resolver runs on a pooled thread, and a Stop's watchdog injects
    for as long as the bridge's state names that thread.  An injection
    raised in the hook's wait for the lock skipped the unregister; a
    second one in the hook's one repeat left the state registered for
    good, and the watchdog then hit whatever the thread ran next.
    """
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    home = IsolatedKissHome("kiss-bridge-retire-")
    home.write_config()
    standin = StandInModelServer(lambda _request: finish_response("conflict resolved"))
    parent = ChatSorcarAgent("merge-parent")
    parent.work_dir = str(home.repo)
    parent.model_name = STANDIN_MODEL
    parent.model_config = standin.model_config
    parent.printer = printer = JsonPrinter()
    finish_hook = Pause(JsonPrinter.agent_task_finished.__code__, lambda _locals: True)
    state: agent_state.AgentState | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool, finish_hook:
            resolver = pool.submit(run_merge_sea, parent, "resolve f.txt", home.repo)
            assert finish_hook.paused.wait(60), "the resolver never reached its finish hook"
            [state] = [s for s in agent_state.snapshot() if s.tab_id.endswith("__merge")]
            worker = state.task_thread
            assert worker is not None and not state.server_owned
            # The guard a Stop on the resolver's tab arms its watchdog with.
            still_owns = functools.partial(_state_owns_thread, state, worker)
            with agent_state.STATE_LOCK:
                finish_hook.release()
                for _ in range(2):
                    # The hook waits for the lock held here, as for a watchdog
                    # that is injecting; a wait in slices raises within one.
                    time.sleep(ACQUIRE_SLICE_SECONDS + 0.2)
                    assert inject_if_owned(worker, still_owns) == 1
                time.sleep(ACQUIRE_SLICE_SECONDS + 0.2)
            # The resolver's work was done; its Stop is honoured all the same.
            with pytest.raises(KeyboardInterrupt):
                resolver.result(timeout=60)
            # The thread goes on to other work with the Stop's watchdog still
            # armed: one second of grace, then its guard.
            assert pool.submit(threading.current_thread).result(timeout=10) is worker
            release = threading.Event()
            other = pool.submit(_other_work, release)
            watchdog = threading.Thread(
                target=_TaskRunnerMixin._force_stop_thread, args=(worker, still_owns), daemon=True,
            )
            watchdog.start()
            watchdog.join(3)
            release.set()
            assert other.result(timeout=10) == "finished" and not watchdog.is_alive()
            assert agent_state.get(state.task_id) is None
            assert _held_state(printer, state.task_id) == []
    finally:
        standin.stop()
        if state is not None:
            # A state left registered must not hand a later injection the idle thread.
            with agent_state.STATE_LOCK:
                state.task_thread = None
                agent_state.unregister(state.task_id, state)
        home.cleanup()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""One parallel fan-out engine, with no drift between callers.

The parallel fan-out engine used to exist twice — module-level
``run_tasks_parallel`` in ``sorcar_agent.py`` and
``ChatSorcarAgent._run_tasks_parallel`` — and four correctness fixes
had landed only in the subclass copy:

* **1a** a per-sub-agent stop event, so stopping ONE child does not
  kill the parent and its siblings;
* **1b** a ``KeyboardInterrupt`` handler, so a stopped child does not
  discard its already-finished siblings' results;
* **1c** a real ``parent_task_id``, so a sub-task is not persisted as
  a top-level history row;
* **1d** ``chat_id`` / ``_tab_id`` propagation, so the child shares the
  parent's chat context and the ``subagentDone`` broadcast names a tab
  that was actually opened.

These tests drive the BASE engine (``SorcarAgent._run_tasks_parallel``,
inherited by every third-party channel agent) against real threads, a
real temp git repo, a real temp SQLite history DB under an isolated
``KISS_HOME``, a real ``JsonPrinter``, and a local stand-in model
server.  Nothing is mocked or patched.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import stop_signal
from kiss.server import agent_state
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    history_rows,
    request_text,
    tool_call_response,
    wait_for,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f1-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _configure_parent(
    parent: SorcarAgent,
    printer: CapturePrinter,
    server: StandInModelServer,
    work_dir: Path,
) -> None:
    """Give *parent* the run-state the fan-out reads off it."""
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.work_dir = str(work_dir)
    parent.model_config = server.model_config


class _FanoutThread(threading.Thread):
    """Runs a fan-out on its own thread with a real parent stop event."""

    def __init__(
        self,
        parent: SorcarAgent,
        tasks: list[str],
        printer: CapturePrinter,
        parent_task_id: str,
    ) -> None:
        """Prepare (but do not start) the fan-out thread."""
        super().__init__(daemon=True)
        self.parent = parent
        self.tasks = tasks
        self.printer = printer
        self.parent_task_id = parent_task_id
        self.stop_event = threading.Event()
        self.results: list[str] | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        """Bind the stop event to this thread, then fan out."""
        stop_signal.set_thread_stop_event(self.stop_event)
        self.printer._thread_local.task_id = self.parent_task_id
        try:
            self.results = SorcarAgent._run_tasks_parallel(
                self.parent, self.tasks, max_workers=len(self.tasks),
            )
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            self.error = exc
        finally:
            stop_signal.set_thread_stop_event(None)


def _summaries(results: list[str]) -> list[str]:
    """Parse each YAML result string and return its ``summary``."""
    out: list[str] = []
    for raw in results:
        parsed = yaml.safe_load(raw)
        out.append(str(parsed.get("summary", "")) if isinstance(parsed, dict) else "")
    return out


def test_stopping_one_subagent_spares_its_siblings(
    env: IsolatedKissHome,
) -> None:
    """F1a + F1b: a per-child stop, with sibling results preserved.

    Three real sub-agents run concurrently.  The victim is parked in a
    real ``Bash`` sleep; the siblings each write a sentinel file and
    finish.  Once every child has reached a known point, the test
    resolves the victim's registered ``AgentState`` — exactly what
    ``VSCodeServer._stop_task`` does — and stops only that child.

    Before the fix the base engine handed every child the PARENT's own
    stop event, so this stop killed the whole task, and the victim's
    ``KeyboardInterrupt`` escaped the worker and discarded both
    siblings' finished results.
    """
    alpha_done = env.repo / "alpha.done"
    gamma_done = env.repo / "gamma.done"
    victim_started = env.repo / "victim.started"

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        if "VICTIM" in text:
            if str(victim_started) in text:
                return finish_response("victim should never get here")
            return tool_call_response(
                "Bash",
                {
                    "command": f"touch {victim_started}; sleep 30",
                    "description": "park the victim sub-agent",
                },
            )
        sentinel = alpha_done if "ALPHA" in text else gamma_done
        if str(sentinel) in text:
            return finish_response(
                "alpha ok" if "ALPHA" in text else "gamma ok"
            )
        return tool_call_response(
            "Write", {"file_path": str(sentinel), "content": "reached\n"},
        )

    server = StandInModelServer(responder)
    printer = CapturePrinter()
    parent = SorcarAgent("f1-stop-parent")
    _configure_parent(parent, printer, server, env.repo)
    tasks = [
        "ALPHA sibling task",
        "VICTIM middle task",
        "GAMMA sibling task",
    ]
    fanout = _FanoutThread(parent, tasks, printer, "f1parenttask")
    fanout.start()
    try:
        assert wait_for(
            lambda: alpha_done.exists()
            and gamma_done.exists()
            and victim_started.exists()
        ), "sub-agents never reached their synchronisation points"

        with agent_state.STATE_LOCK:
            victim_states = [
                state
                for state in agent_state.agent_states.values()
                if "VICTIM" in str(getattr(state.agent, "name", ""))
            ]
        assert len(victim_states) == 1, (
            "expected exactly one live sub-agent state for the victim, "
            f"got {len(victim_states)}"
        )
        victim_stop = victim_states[0].stop_event
        assert victim_stop is not None
        assert victim_stop is not fanout.stop_event, (
            "the victim sub-agent was registered with the PARENT's stop "
            "event, so stopping it stops the parent and both siblings"
        )
        victim_stop.set()

        fanout.join(timeout=90)
        assert not fanout.is_alive(), "fan-out did not unwind after the stop"
        assert fanout.error is None, (
            f"stopping one sub-agent aborted the whole fan-out: "
            f"{fanout.error!r}"
        )
        assert fanout.results is not None
        summaries = _summaries(fanout.results)
        assert summaries[0] == "<p>alpha ok</p>", summaries
        assert summaries[2] == "<p>gamma ok</p>", summaries
        assert summaries[1] == "Sub-agent task stopped by user.", summaries
        assert not fanout.stop_event.is_set(), (
            "stopping one sub-agent must not set the parent's stop event"
        )
    finally:
        fanout.stop_event.set()
        fanout.join(timeout=60)
        server.stop()


def test_base_fanout_nests_history_and_shares_chat(
    env: IsolatedKissHome,
) -> None:
    """F1c + F1d: children are persisted under the parent's task/chat.

    The base engine used to write ``parent_task_id = ""`` for every
    child, which ``persistence``'s ``_HISTORY_NOT_SUBAGENT`` predicate
    reads as "top-level task", polluting the history list with one
    bogus root row per sub-agent; it also gave each child a brand-new
    chat id and never set the child's ``_tab_id``, so the
    ``subagentDone`` broadcast named a tab nothing had opened.
    """

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        return finish_response("child done")

    server = StandInModelServer(responder)
    printer = CapturePrinter()
    parent = ChatSorcarAgent("f1-history-parent")
    parent._tab_id = "tab-parent-42"
    parent.run(
        prompt_template="PARENT TASK",
        model_name=STANDIN_MODEL,
        model_config=server.model_config,
        work_dir=str(env.repo),
        printer=printer,
    )
    parent_task_id = parent._last_task_id
    parent_chat_id = parent._chat_id
    assert isinstance(parent_task_id, str) and parent_task_id

    fanout = _FanoutThread(
        parent, ["child one", "child two"], printer, parent_task_id,
    )
    fanout.start()
    fanout.join(timeout=120)
    server.stop()
    assert fanout.error is None, fanout.error
    assert fanout.results is not None and len(fanout.results) == 2

    rows = history_rows()
    children = [r for r in rows if r["id"] != parent_task_id]
    assert len(children) == 2, rows
    for child in children:
        assert child["parent_task_id"] == parent_task_id, (
            "base fan-out persisted a sub-agent as a TOP-LEVEL history "
            f"row (parent_task_id={child['parent_task_id']!r})"
        )
        assert child["chat_id"] == parent_chat_id, (
            "sub-agent did not inherit the parent's chat session"
        )

    opened_tabs = {
        e.get("task_id") for e in printer.events_of_type("new_tab")
    }
    assert opened_tabs == {c["id"] for c in children}, (
        "every sub-agent must announce its own tab"
    )
    for event in printer.events_of_type("new_tab"):
        assert event.get("parent_tab_id") == "tab-parent-42", (
            "sub-agent new_tab must carry the parent's tab id so the "
            "owning webview can claim it"
        )
    done_tabs = {
        e.get("tab_id") for e in printer.events_of_type("subagentDone")
    }
    expected = {f"task-{parent_task_id}__sub_{i}" for i in range(2)}
    assert expected <= done_tabs, (
        f"subagentDone fired for tabs {done_tabs}, expected {expected}"
    )

def test_channel_agent_fanout_children_stay_out_of_the_history_list(
    env: IsolatedKissHome,
) -> None:
    """A parent with no history row of its own still nests its children.

    Third-party channel agents (Slack, email, voice) are plain
    ``SorcarAgent`` subclasses: they persist no ``task_history`` row,
    so there is no parent id to stamp on the children they fan out.
    Their children are real ``ChatSorcarAgent`` runs and DO create
    rows, and a blank parent id makes every one of them a top-level
    entry in the VS Code history sidebar.
    """
    from kiss.agents.sorcar import persistence

    server = StandInModelServer(lambda request: finish_response("child done"))
    printer = CapturePrinter()
    parent = SorcarAgent("channel-agent-parent")
    _configure_parent(parent, printer, server, env.repo)

    # No persisted parent row, and no task id on the printer either:
    # exactly what a channel agent's fan-out looks like.
    fanout = _FanoutThread(parent, ["child one", "child two"], printer, "")
    fanout.start()
    fanout.join(timeout=120)
    server.stop()
    assert fanout.error is None, fanout.error
    assert fanout.results is not None and len(fanout.results) == 2

    rows = history_rows()
    assert len(rows) == 2, rows
    for row in rows:
        assert row["parent_task_id"], (
            "a channel agent's sub-agent was persisted as a TOP-LEVEL "
            "history row, so every fan-out pollutes the history sidebar"
        )
    assert {row["parent_task_id"] for row in rows} == {
        rows[0]["parent_task_id"]
    }, "siblings of one fan-out must share a parent id"
    assert persistence._load_history() == [], (
        "the history list must show no root entry for a fan-out whose "
        "parent has no row of its own"
    )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 S8 F2/F3: ``ChatSorcarAgent.run`` under a Stop in its setup or its cleanup.

* The run's ``task_history`` row was committed, then its id published
  under ``_task_id_lock`` BEFORE the cleanup ``try`` began.  A Stop
  landing in that interruptible lock wait left the row on its abrupt-
  failure sentinel with ``end_ts = 0`` forever: no cleanup ran, and the
  task runner, which finds the row through ``last_task_id``, never
  learnt the id.  The protected region now starts right after the id
  is assigned, and a run whose id was never published persists its own
  final state even when the runner would normally do so.
* The cleanup was one linear ``finally``: a second Stop in the
  printer's finish hook (its ``STATE_LOCK`` wait) skipped
  ``stop_recording``, the restoration of the calling thread's task
  binding and the final persistence, and left the bridge-created
  agent state registered as active.  The steps are nested
  ``try``/``finally`` regions now, and the idempotent retirement is
  retried once.

The Stops are the server's real ``inject_keyboard_interrupt``; the
locks are the real ones, held by the test thread so the worker waits
on them.  Real history database and printer; nothing is mocked.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core.locked import locked_call
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
)
from kiss.tests.server.pause import Pause

CALLER_BINDING = "caller-parent"


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[IsolatedKissHome]:
    """An isolated KISS home; the task classifier (a model call) is off."""
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    home = IsolatedKissHome("audit1101-s8-stop-")
    try:
        yield home
    finally:
        home.cleanup()


class _Worker:
    """``agent.run`` on its own thread, recording how it ended and the printer binding it left."""

    def __init__(self, agent: ChatSorcarAgent, printer: JsonPrinter, repo: Path, **run_kwargs: Any):
        self.agent = agent
        self.printer = printer
        self.repo = repo
        self.run_kwargs = run_kwargs
        self.errors: list[BaseException] = []
        self.binding_after: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.printer._thread_local.task_id = CALLER_BINDING
        try:
            self.agent.run(
                prompt_template="check the arithmetic",
                model_name=STANDIN_MODEL,
                work_dir=str(self.repo),
                web_tools=False,
                use_memory=False,
                max_budget=1,
                printer=self.printer,
                tool_profile="review",
                **self.run_kwargs,
            )
        except BaseException as exc:  # noqa: BLE001 — the Stop is the expected outcome
            self.errors.append(exc)
        finally:
            self.binding_after.append(self.printer._thread_local.task_id)

    def join(self) -> None:
        """Join the worker, stopping it again if it is still running."""
        self.thread.join(10)
        if self.thread.is_alive():
            assert self.thread.ident is not None
            inject_keyboard_interrupt(self.thread.ident)
            self.thread.join(10)
        assert not self.thread.is_alive(), "the worker never ended"


def _task_rows(home: IsolatedKissHome) -> list[tuple[str, str, int]]:
    with closing(sqlite3.connect(home.kiss_home / "history.db")) as db:
        return db.execute("SELECT id, result, end_ts FROM task_history").fetchall()


def _stop_in_the_publication(
    home: IsolatedKissHome, printer: JsonPrinter | None = None, **run_kwargs: Any,
) -> _Worker:
    """Run the agent and Stop it while ``run`` waits to publish the allocated task id."""
    agent = ChatSorcarAgent("audit1101-stop-at-publication")
    worker = _Worker(agent, printer or CapturePrinter(), home.repo, **run_kwargs)
    publication = Pause(
        locked_call.__code__,
        lambda frame_locals: frame_locals.get("lock") is agent._task_id_lock
        and frame_locals.get("fn") is setattr,
    )
    with publication:
        worker.thread.start()
        assert publication.paused.wait(30), worker.errors
        with agent._task_id_lock:
            publication.release()
            assert worker.thread.ident is not None
            # The worker is now waiting for the lock this thread holds.
            time.sleep(0.2)
            inject_keyboard_interrupt(worker.thread.ident)
            worker.join()
    assert [type(exc) for exc in worker.errors] == [KeyboardInterrupt]
    return worker


def test_stop_while_publishing_the_task_id_finalises_the_row(home: IsolatedKissHome) -> None:
    # Audit 2026-11-08 S1/F4: the calling task is live and recording under
    # the binding the stopped run never replaced; the run's cleanup must
    # stop its own recording only, not the caller's.
    printer = CapturePrinter()
    caller = ChatSorcarAgent("audit1101-caller")
    printer._thread_local.task_id = CALLER_BINDING
    printer.agent_task_allocated(caller, CALLER_BINDING, "caller-chat")
    printer.start_recording()
    printer.broadcast({"type": "text_delta", "text": "caller-before", "task_id": CALLER_BINDING})
    try:
        worker = _stop_in_the_publication(home, printer)

        rows = _task_rows(home)
        assert len(rows) == 1, rows
        _task_id, result, end_ts = rows[0]
        assert result == "Task interrupted"
        assert end_ts > 0
        assert worker.binding_after == [CALLER_BINDING]
        assert agent_state.agent_states[CALLER_BINDING].is_task_active
        # The caller's recording is still live: its own stop returns it.
        texts = [event.get("text") for event in printer.stop_recording()]
        assert texts == ["caller-before"], texts
    finally:
        printer.agent_task_finished(caller, CALLER_BINDING)


def test_stop_while_publishing_persists_a_runner_owned_row_too(home: IsolatedKissHome) -> None:
    """The runner finds the row through ``last_task_id``, which the Stop kept unpublished."""
    worker = _stop_in_the_publication(home, _skip_persistence=True)

    assert worker.agent.last_task_id == ""
    rows = _task_rows(home)
    assert len(rows) == 1, rows
    _task_id, result, end_ts = rows[0]
    assert result == "Task interrupted"
    assert end_ts > 0


def _stop_in_own_thread(task_id: str, chat_id: str) -> None:
    """``_on_task_id_allocated`` hook: the first Stop, delivered to the run's own thread."""
    inject_keyboard_interrupt(threading.get_ident())


def test_second_stop_in_the_finish_hook_completes_the_cleanup(home: IsolatedKissHome) -> None:
    agent = ChatSorcarAgent("audit1101-second-stop")
    printer = CapturePrinter()
    worker = _Worker(agent, printer, home.repo, _on_task_id_allocated=_stop_in_own_thread)
    finish_hook = Pause(JsonPrinter.agent_task_finished.__code__, lambda frame_locals: True)
    with finish_hook:
        worker.thread.start()
        assert finish_hook.paused.wait(30), worker.errors
        with agent_state.STATE_LOCK:
            finish_hook.release()
            assert worker.thread.ident is not None
            # The hook is now waiting for STATE_LOCK, which this thread holds.
            time.sleep(0.2)
            inject_keyboard_interrupt(worker.thread.ident)
        worker.join()
    assert [type(exc) for exc in worker.errors] == [KeyboardInterrupt]

    task_id = agent.last_task_id
    assert task_id
    with agent_state.STATE_LOCK:
        active = [
            state for state in agent_state.agent_states.values()
            if state.task_id == task_id and state.is_task_active
        ]
    assert active == []
    assert worker.binding_after == [CALLER_BINDING]
    rows = {row[0]: row[1:] for row in _task_rows(home)}
    assert rows[task_id][0] == "Task interrupted"
    assert rows[task_id][1] > 0


class _RecordingStaysLive(CapturePrinter):
    """A printer whose recording cannot be retired: the failure is logged, never hidden."""

    def stop_recording(self) -> list[dict[str, Any]]:
        raise RuntimeError("the recording could not be retired")


def test_a_failing_stop_recording_is_logged_and_the_cleanup_completes(
    home: IsolatedKissHome, caplog: pytest.LogCaptureFixture
) -> None:
    """``stop_recording`` raising leaves the recording live for good: silent before, logged now."""
    agent = ChatSorcarAgent("audit1101-stop-recording-fails")
    worker = _Worker(agent, _RecordingStaysLive(), home.repo,
                     _on_task_id_allocated=_stop_in_own_thread)
    with caplog.at_level(logging.WARNING, logger="kiss.agents.sorcar.chat_sorcar_agent"):
        worker.thread.start()
        worker.join()
    assert [type(exc) for exc in worker.errors] == [KeyboardInterrupt]
    task_id = agent.last_task_id
    logged = [r for r in caplog.records if r.getMessage() == f"stop_recording({task_id!r}) raised"]
    assert len(logged) == 1 and logged[0].exc_info is not None
    assert logged[0].exc_info[0] is RuntimeError
    # The remaining cleanup steps still ran.
    assert worker.binding_after == [CALLER_BINDING]
    rows = {row[0]: row[1:] for row in _task_rows(home)}
    assert rows[task_id][0] == "Task interrupted"
    assert rows[task_id][1] > 0

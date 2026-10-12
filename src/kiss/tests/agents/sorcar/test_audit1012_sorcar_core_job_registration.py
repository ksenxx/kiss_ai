# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (sorcar-core): a Stop inside ``start_agent_job`` cannot orphan the job.

A task's Stop is a ``KeyboardInterrupt`` injected into the agent
thread between two bytecodes.  ``start_agent_job`` used to start the
job thread and only then insert the job into the registry, so a Stop
landing in between left a running sub-task that ``kill_jobs_of`` could
not find: it outlived its caller and kept its tab open.

Registering first closed that window but not the one inside
``Thread.start()`` itself: the native thread exists before the parent
observes ``is_alive()``, so a Stop landing there leaves a registered
job whose thread looks unstarted yet runs later.  ``kill_jobs_of``
used to cancel only the observably alive threads, so that job was
forgotten uncancelled and dispatched a sub-task nobody could kill.

Both tests reproduce their race on a REAL dispatch to a local daemon
stand-in: a trace function raises ``KeyboardInterrupt`` at the exact
line boundary where the injected Stop lands.
"""

from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import DEFAULT_AGENT_PATH
from kiss.tests.agents.sorcar.test_dispatch_timeout import _StopConfirmingDaemon

_START_LINES, _START_FIRST_LINE = inspect.getsourcelines(threading.Thread.start)
_START_WAIT_LINE = _START_FIRST_LINE + next(
    offset for offset, line in enumerate(_START_LINES) if "self._started.wait()" in line
)
"""The line of ``Thread.start()`` at which the native thread exists but is not yet alive."""


def _stop_once_the_thread_runs(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace function: raise KeyboardInterrupt at the first line where ``job.thread`` is alive.

    A line boundary inside a ``with`` block's exit sequence is skipped
    (the registry lock is still held there): an injected interrupt is
    delivered after a call completes, so it never lands between the
    instructions that call ``__exit__``.
    """
    if frame.f_code.co_name != "start_agent_job":
        return None
    if event == "line":
        thread = getattr(frame.f_locals.get("job"), "thread", None)
        if (
            thread is not None
            and thread.is_alive()
            and not agent_dispatch._AGENT_JOBS_LOCK.locked()
        ):
            sys.settrace(None)
            raise KeyboardInterrupt
    return _stop_once_the_thread_runs


def _stop_inside_thread_start(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace function: raise KeyboardInterrupt inside ``Thread.start()`` of the job thread.

    Fires at the ``self._started.wait()`` line: the native thread has
    been created but has not yet reported itself started, so the
    caller sees ``is_alive()`` false although the thread will run.
    """
    if frame.f_code is not threading.Thread.start.__code__:
        return None
    if (
        event == "line"
        and frame.f_lineno == _START_WAIT_LINE
        and frame.f_locals["self"].name.startswith("agent-job-")
    ):
        sys.settrace(None)
        raise KeyboardInterrupt
    return _stop_inside_thread_start


def _start_job_interrupted_by(
    trace: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_StopConfirmingDaemon, object]:
    """Call ``start_agent_job`` on a worker thread traced by *trace* and return (daemon, owner).

    The worker must have been interrupted (not returned) for the race
    to count as exercised.
    """
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    owner = object()
    kwargs = {
        "name": "sorcar",
        "prompt": "never finishes",
        "sea_path": DEFAULT_AGENT_PATH,
        "work_dir": str(tmp_path),
        "model_name": "",
        "budget": None,
        "timeout": 30.0,
        "alias": "",
    }
    outcome: list[str] = []

    def call() -> None:
        sys.settrace(trace)
        try:
            agent_dispatch.start_agent_job("sorcar", kwargs, owner)
            outcome.append("returned")
        except KeyboardInterrupt:
            outcome.append("interrupted")
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=call, name="stopped-caller")
    worker.start()
    worker.join(30)
    assert not worker.is_alive()
    assert outcome == ["interrupted"], outcome
    return daemon, owner


def _job_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("agent-job-")]


def test_stop_landing_right_after_thread_start_cannot_orphan_the_sub_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, owner = _start_job_interrupted_by(_stop_once_the_thread_runs, tmp_path, monkeypatch)
    try:
        job_threads = _job_threads()
        assert job_threads, "the job thread never started: the race was not exercised"
        registered = agent_dispatch.agent_jobs_of(owner)
        for thread in job_threads:
            job_id = thread.name.removeprefix("agent-job-")
            assert job_id in registered, (
                f"{thread.name} runs a sub-task its owner cannot kill (unregistered)"
            )
        # The owner's run ending stops it, as for every registered job.
        killed = agent_dispatch.kill_jobs_of(owner)
        assert killed == list(registered)
        for thread in job_threads:
            thread.join(30)
            assert not thread.is_alive()
        assert agent_dispatch.agent_jobs_of(owner) == {}
    finally:
        for job in agent_dispatch.agent_jobs_of(owner).values():
            job.cancel.set()
        daemon.close()


def test_stop_landing_inside_thread_start_cannot_orphan_the_sub_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, owner = _start_job_interrupted_by(_stop_inside_thread_start, tmp_path, monkeypatch)
    try:
        registered = agent_dispatch.agent_jobs_of(owner)
        assert len(registered) == 1
        (job,) = registered.values()
        # The owner's run ends right away: the job is cancelled although
        # its thread is not (yet) observably alive ...
        killed = agent_dispatch.kill_jobs_of(owner)
        assert killed == [job.job_id]
        assert job.cancel.is_set()
        assert agent_dispatch.agent_jobs_of(owner) == {}
        # ... and when the thread does run it dispatches nothing: the
        # daemon never receives a run and the job records the stop.
        assert job.running.wait(30), "the job thread never ran"
        if job.thread.is_alive():
            job.thread.join(30)
        assert not job.thread.is_alive()
        assert job.finished
        if "stopped before it started" in job.result:
            assert getattr(daemon, "run_cmd", None) is None, daemon.run_cmd
            assert daemon.commands == []
        else:
            # Under load the thread can reach its dispatch before
            # kill_jobs_of runs; the stop then lands mid-flight, which is
            # the same guarantee (cancelled, never orphaned).
            assert "stopped before it finished" in job.result, job.result
    finally:
        for job in agent_dispatch.agent_jobs_of(owner).values():
            job.cancel.set()
        daemon.close()

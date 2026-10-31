# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing anywhere in a job's start or kill leaves no job uncancelled and unreachable.

Audit 2026-10-30 (S6), two RACE findings:

* ``kill_jobs_of`` dropped each job from the registry before setting
  its cancel, so a Stop landing between the two left a job no later
  ``kill_jobs_of``, ``agent_job`` or finish gate could reach, running
  to completion.
* ``run_agent`` / ``run_parallel`` learned of their job only when
  ``start_agent_job`` returned, so a Stop landing after the job thread
  started but before that left their cleanup with nothing to cancel;
  for a job without an owner (a cron ``run_now`` dispatch) nothing
  ever stopped it.

A task's Stop is a ``KeyboardInterrupt`` injected between bytecodes;
trace functions raise one at exact line boundaries of REAL dispatches
to a local daemon stand-in that never finishes a task and confirms
every stop.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    make_run_agent_tool,
    make_run_parallel_tool,
)
from kiss.tests.agents.sorcar.test_dispatch_timeout import _StopConfirmingDaemon


@pytest.fixture()
def daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_StopConfirmingDaemon]:
    """A daemon stand-in that runs every sub-task until it is stopped."""
    stand_in = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(stand_in.endpoint_file))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    try:
        yield stand_in
    finally:
        stand_in.close()


def _job_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "name": "sorcar",
        "prompt": "never finishes",
        "sea_path": DEFAULT_AGENT_PATH,
        "work_dir": str(tmp_path),
        "model_name": "",
        "budget": None,
        "timeout": 30.0,
        "alias": "",
    }


def _interrupted(call: Callable[[], object], trace: Any) -> bool:
    """Run *call* on a worker thread traced by *trace*; return whether a Stop interrupted it."""
    outcome: list[str] = []

    def body() -> None:
        sys.settrace(trace)
        try:
            call()
            outcome.append("returned")
        except KeyboardInterrupt:
            outcome.append("interrupted")
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=body, name="stopped-caller")
    worker.start()
    worker.join(60)
    assert not worker.is_alive()
    return outcome == ["interrupted"]


class _StopAtLine:
    """Trace function: raise KeyboardInterrupt at the *n*-th line event of *code*'s own frame."""

    def __init__(self, code: CodeType, n: int) -> None:
        self.code = code
        self.n = n
        self.seen = 0

    def __call__(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is not self.code:
            return None
        if event == "line":
            self.seen += 1
            if self.seen == self.n:
                sys.settrace(None)
                raise KeyboardInterrupt
        return self


def test_a_stop_anywhere_in_kill_jobs_of_leaves_every_job_cancelled_or_registered(
    tmp_path: Path, daemon: _StopConfirmingDaemon,
) -> None:
    line = 0
    interrupted = True
    while interrupted:  # one Stop per line event, until kill_jobs_of runs to its end
        line += 1
        owner = object()
        jobs = [
            agent_dispatch.start_agent_job("sorcar", _job_kwargs(tmp_path), owner)
            for _ in range(2)
        ]
        try:
            for job in jobs:
                assert job.running.wait(30), job.job_id
            stop = _StopAtLine(agent_dispatch.kill_jobs_of.__code__, line)
            interrupted = _interrupted(partial(agent_dispatch.kill_jobs_of, owner), stop)
            registered = agent_dispatch.agent_jobs_of(owner)
            for job in jobs:
                assert job.cancel.is_set() or job.job_id in registered, (
                    f"a Stop at line event {line} of kill_jobs_of left {job.job_id} "
                    "out of the registry and never cancelled"
                )
        finally:
            agent_dispatch.kill_jobs_of(owner)
            for job in jobs:
                job.cancel.set()
                assert agent_dispatch.join_agent_job(job, 30.0), job.job_id
    assert line > 4, "kill_jobs_of ran too few lines for the race to be exercised"


def _job_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name.startswith("agent-job-")}


class _StopOnceANewJobRuns:
    """Trace function: raise KeyboardInterrupt once a new job thread runs.

    Fires at the first line of an ``agent_dispatch`` frame that runs
    with the registry lock free (a Stop inside the lock is out of
    scope: ``locked_call`` releases it either way) after a job thread
    that did not exist at construction is alive, and records those new
    threads in ``started``: a job the Stop's cleanup cancels at once
    may be gone from ``threading.enumerate()`` by the time the test
    looks.
    """

    def __init__(self) -> None:
        self.existing = _job_threads()
        self.started: set[threading.Thread] = set()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename != agent_dispatch.__file__:
            return None
        if (
            event == "line"
            and not agent_dispatch._AGENT_JOBS_LOCK._is_owned()  # type: ignore[attr-defined]
            and (started := _job_threads() - self.existing)
        ):
            self.started = started
            sys.settrace(None)
            raise KeyboardInterrupt
        return self


def _assert_the_ownerless_job_was_abandoned(stop: _StopOnceANewJobRuns) -> None:
    threads = stop.started
    assert threads, "no job thread started: the race was not exercised"
    registered = agent_dispatch.agent_jobs_of(None)
    try:
        for thread in threads:
            assert thread.name.removeprefix("agent-job-") not in registered, (
                f"{thread.name} has no owner and was left registered and running"
            )
            thread.join(30)
            assert not thread.is_alive(), f"{thread.name} was never cancelled"
    finally:
        for job in registered.values():
            if job.thread in threads:
                job.cancel.set()
                agent_dispatch.forget_agent_job(job)


def test_a_stop_inside_an_ownerless_run_agent_start_cancels_its_job(
    tmp_path: Path, daemon: _StopConfirmingDaemon,
) -> None:
    run_agent = make_run_agent_tool(str(tmp_path))
    stop = _StopOnceANewJobRuns()
    call = partial(run_agent, "never finishes", DEFAULT_AGENT_PATH, timeout="30")
    assert _interrupted(call, stop)
    _assert_the_ownerless_job_was_abandoned(stop)


def test_a_stop_inside_an_ownerless_run_parallel_start_cancels_its_job(
    tmp_path: Path, daemon: _StopConfirmingDaemon,
) -> None:
    run_parallel = make_run_parallel_tool(str(tmp_path))
    stop = _StopOnceANewJobRuns()
    tasks = json.dumps(["never finishes", "never finishes either"])
    assert _interrupted(partial(run_parallel, tasks, DEFAULT_AGENT_PATH, timeout="30"), stop)
    _assert_the_ownerless_job_was_abandoned(stop)

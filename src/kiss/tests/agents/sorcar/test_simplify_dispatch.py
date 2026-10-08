# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Sub-agent dispatch fixes (PART=dispatch of the sorcar simplification).

D1: ``kill_agent_job`` must honour a Stop of the calling tool call while
it waits for the daemon's stop confirmation, as ``join_agent_job`` does,
instead of one un-interruptible 30-s ``Thread.join``.

D2: a ``run_agent`` call without an owning agent (cron's
``make_run_agent_tool(work_dir)``) that is interrupted must drop its job
from the process-wide registry itself: no run end ever collects an
ownerless job.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool
from kiss.core import tool_interrupt
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.agents.sorcar.test_dispatch_timeout import _HELPER_SEA, _StopConfirmingDaemon


def _linger_after_cancel(cancel: threading.Event, release: threading.Event) -> None:
    """Thread body of a stand-in job: once cancelled, stay alive until *release*."""
    cancel.wait(10)
    release.wait(5)


def _interrupt_later(thread_ident: int, delay: float) -> None:
    """Press the tool call's Stop button on thread *thread_ident* after *delay* seconds."""
    time.sleep(delay)
    assert tool_interrupt.interrupt_tool_call(thread_ident)


def test_kill_agent_job_honours_the_tool_call_stop_while_waiting() -> None:
    """A Stop of the ``agent_job(id, "kill")`` call lands during the kill's wait.

    The job thread stands in for a dispatch whose daemon is slow to
    confirm the stop: it stays alive for seconds after the cancel.  The
    kill must raise :class:`ToolCallInterrupted` within one wake slice
    of the Stop, with the cancel already sent, rather than return only
    when the thread exits.
    """
    release = threading.Event()
    job = agent_dispatch.AgentJob(
        "agent-simplify1", "lingering", None, threading.Event(), threading.Event(),
    )
    job.thread = threading.Thread(
        target=_linger_after_cancel, args=(job.cancel, release), daemon=True,
    )
    job.thread.start()
    token = tool_interrupt.begin_tool_call("agent_job")
    stopper = threading.Thread(
        target=_interrupt_later, args=(threading.get_ident(), 0.3), daemon=True,
    )
    started = time.monotonic()
    try:
        stopper.start()
        with pytest.raises(tool_interrupt.ToolCallInterrupted):
            agent_dispatch.kill_agent_job(job)
        elapsed = time.monotonic() - started
        assert elapsed < 2.5, f"the kill ignored the Stop for {elapsed:.1f}s"
        assert job.cancel.is_set()
        assert job.thread.is_alive()
    finally:
        tool_interrupt.unregister_tool_call(token)
        release.set()
        stopper.join(5)
        job.thread.join(5)
    assert not job.thread.is_alive()


def test_kill_agent_job_returns_the_result_once_the_thread_finishes() -> None:
    """Without a Stop the kill returns the job's result as soon as its thread exits."""
    release = threading.Event()
    job = agent_dispatch.AgentJob(
        "agent-simplify2", "quick", None, threading.Event(), threading.Event(),
    )
    job.thread = threading.Thread(
        target=_linger_after_cancel, args=(job.cancel, release), daemon=True,
    )
    job.thread.start()
    release.set()
    job.result = "stopped"
    assert agent_dispatch.kill_agent_job(job) == "stopped"
    assert not job.thread.is_alive()


def _run_tool_recording(tool: Any, args: tuple[str, ...], outcome: dict[str, Any]) -> None:
    """Call *tool* with *args*, recording its return or exception in *outcome*."""
    try:
        outcome["result"] = tool(*args)
    except BaseException as exc:  # noqa: BLE001 — capture for assert
        outcome["exc"] = exc


def _wait_for_ownerless_job(deadline_seconds: float) -> agent_dispatch.AgentJob:
    """Return the first standalone job once its sub-task's tab exists."""
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        for job in agent_dispatch.agent_jobs_of(None).values():
            if job.running.is_set():
                return job
        time.sleep(0.02)
    raise AssertionError("the ownerless run_agent call never registered a running job")


def test_interrupted_ownerless_run_agent_forgets_its_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted standalone ``run_agent`` leaves nothing in the job registry.

    End-to-end through the real tool built without a parent: the daemon
    stand-in never finishes the task, the calling thread is stopped by
    an injected ``KeyboardInterrupt`` (a daemon shutdown reaching a cron
    worker), and the call cancels its sub-task (``stop`` goes out on
    the job thread) and drops the job, which no owner's run end would
    ever collect.
    """
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(_HELPER_SEA)
    outcome: dict[str, Any] = {}
    worker = threading.Thread(
        target=_run_tool_recording,
        args=(make_run_agent_tool(str(tmp_path)), ("never finishes", str(script)), outcome),
        daemon=True,
    )
    worker.start()
    try:
        job = _wait_for_ownerless_job(10)
        assert worker.is_alive()
        tid = worker.ident
        assert tid is not None
        assert inject_keyboard_interrupt(tid) == 1
        worker.join(10)
        assert not worker.is_alive(), "the blocking run_agent call never let the Stop land"
        assert isinstance(outcome.get("exc"), KeyboardInterrupt), outcome
        assert job.cancel.is_set()
        assert job.job_id not in agent_dispatch.agent_jobs_of(None), (
            "the interrupted ownerless call left its job in the registry"
        )
        assert daemon.wait_for_command("stop")
        job.thread.join(10)
        assert not job.thread.is_alive()
        assert "was stopped before it finished" in job.result, job.result
    finally:
        agent_dispatch.kill_jobs_of(None)
        daemon.close()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Waiting on jobs, not on files: ``agent_job(ids, "wait_any")`` and the early end of waits.

The 2026-11-08 sweep (task 36b8e451) dispatched its reviewers with
``run_agent(wait="false")`` and then polled a folder of their reports
with ``Bash(until=)``; three polls ran to their 50-60 minute timeouts
after every reviewer had finished, one report having gone elsewhere.
Two harness changes cover that: ``wait_any`` returns whichever job
finishes first (so a follow-up starts as soon as its job returns, not
after the slowest of a wave), and a ``Bash(until=)`` poll or a
``bash_job`` wait ends early once every job of the task has finished
with a result the task never read (:func:`unread_results_note`).
"""

from __future__ import annotations

import asyncio
import functools
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    make_agent_job_tool,
    make_run_agent_tool,
    make_run_parallel_tool,
    notice_job_id,
    unread_results_note,
    wait_any_results,
)
from kiss.agents.sorcar.coding_gate import dispatch_successes
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_agent import is_routine_turn
from kiss.core.locked import locked_call
from kiss.core.prompt_cache_keepalive import is_long_running_call
from kiss.core.tool_interrupt import (
    inject_async_exception,
    raise_if_interrupted,
    wait_interruptible,
)
from kiss.tests.agents.sorcar.test_dispatch_timeout import _LocalDaemon, _send_event
from kiss.tests.server.pause import Pause, Pauses


class _StaggeredFinishDaemon(_LocalDaemon):
    """A daemon stand-in whose n-th connection's task finishes after ``delays[n]`` seconds.

    Each connection reads the ``run`` command, sends ``status
    running=true``, sleeps its delay and then sends the terminal
    ``result`` + ``status running=false`` pair; a connection past the
    list, or one whose delay is ``None``, never finishes on its own (the
    shape of a reviewer still at work) and confirms a ``stop`` with the
    terminal status, as a live daemon does.
    """

    def __init__(self, delays: list[float | None]) -> None:
        self.delays = list(delays)
        self.connections = 0
        super().__init__("kiss_wait_any_")

    async def _handle(self, ws: ServerConnection) -> None:
        tab_id = await self._read_run(ws)
        if tab_id is None:
            return
        index, self.connections = self.connections, self.connections + 1
        delay = self.delays[index] if index < len(self.delays) else None
        await _send_event(ws, {"type": "status", "running": True, "tabId": tab_id})
        if delay is None:
            async for message in ws:
                if json.loads(message).get("type") == "stop":
                    await _send_event(ws, {"type": "status", "running": False, "tabId": tab_id})
            return
        await asyncio.sleep(delay)
        await _send_event(ws, {
            "type": "result", "tabId": tab_id, "taskId": f"task-{index}", "success": True,
            "text": f"child {index} done", "cost": "$0.0100", "total_tokens": 5, "step_count": 1,
        })
        await _send_event(ws, {"type": "status", "running": False, "tabId": tab_id})
        try:
            await ws.recv()
        except Exception:  # noqa: BLE001 — the client closed the tab
            pass


@pytest.fixture(autouse=True)
def _standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each test's jobs belong to the standalone owner and die with the test."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    yield
    agent_dispatch.kill_jobs_of(None)


def _start(work_dir: Path, task: str) -> agent_dispatch.AgentJob:
    """Dispatch *task* as a detached job of the standalone owner and return it."""
    out = make_run_agent_tool(str(work_dir))(task, wait="false")
    job = agent_dispatch.agent_jobs_of(None).get(notice_job_id(out))
    assert job is not None, out
    assert "agent_job('', 'wait_any')" in out, out
    return job


def _finished(job: agent_dispatch.AgentJob, seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while not job.finished and time.monotonic() < deadline:
        time.sleep(0.05)
    return job.finished


def test_wait_any_returns_the_first_job_to_finish_and_never_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _StaggeredFinishDaemon([0.3, None, 0.3, 0.3])
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    tool = make_agent_job_tool()
    try:
        fast, slow = _start(tmp_path, "fast child"), _start(tmp_path, "slow child")
        started = time.monotonic()
        out = tool("", "wait_any", "20")
        assert time.monotonic() - started < 15, out
        # The lead line names both sides; the slow job is not "still
        # running" there, so the turn is not filed as routine.
        assert out.startswith(f"Finished: {fast.job_id}; running: {slow.job_id}.\n"), out
        results = wait_any_results(out)
        assert list(results) == [fast.job_id]
        assert dispatch_successes(results[fast.job_id]) == [True]
        assert "child 0 done" in results[fast.job_id]
        assert fast.reported and not slow.reported
        assert not is_routine_turn("agent_job", {"job_id": "", "action": "wait_any"}, out, False)
        assert is_long_running_call("agent_job", {"job_id": "", "action": "wait_any"})
        # A returned result is not returned again: the explicit id says
        # so, and the next wait on both is a wait on the slow one alone.
        assert tool(fast.job_id, "wait_any", "0") == (
            f"Every listed job has finished and its result was already returned: {fast.job_id}."
        )
        timed_out = tool(f"{fast.job_id}, {slow.job_id}", "wait_any", "0")
        assert timed_out.startswith(
            f"None of {slow.job_id} finished within 0s; they are still running",
        )
        assert is_routine_turn("agent_job", {"job_id": "", "action": "wait_any"}, timed_out, False)
        assert wait_any_results(timed_out) == {}
        # A JSON number waits like its text: Bash and bash_job take
        # timeout_seconds as a number, so models pass one here too.
        assert tool(f"{fast.job_id}, {slow.job_id}", "wait_any", 0) == timed_out
        assert tool(slow.job_id, "wait", 0) == (
            f"Job {slow.job_id} ({slow.name} agent task) is still running."
        )
        # The other actions mark a result read too: after a kill the
        # job's error result is the one wait_any would have returned.
        assert tool(slow.job_id, "nap") == (
            "Error: action must be tail, wait, wait_any or kill, got 'nap'."
        )
        killed = tool(slow.job_id, "kill")
        assert "was stopped before it finished" in killed, killed
        assert slow.reported
        # A blocking run_agent or run_parallel returns its results itself
        # and takes the jobs off the registry: neither wait_any nor the
        # early-end hook can return them a second time.
        blocking = make_run_agent_tool(str(tmp_path))("blocking child", timeout="20")
        assert "child 2 done" in blocking and notice_job_id(blocking) == "", blocking
        wave = make_run_parallel_tool(str(tmp_path))('["wave child"]', timeout="20")
        assert "child 3 done" in wave and notice_job_id(wave) == "", wave
        assert set(agent_dispatch.agent_jobs_of(None)) == {fast.job_id, slow.job_id}
        assert unread_results_note(None) == ""
        assert tool("", "wait_any") == (
            "Every listed job has finished and its result was already returned: "
            f"{fast.job_id}, {slow.job_id}."
        )
    finally:
        daemon.close()


def _on_thread(prefix: str, frame_locals: dict[str, Any]) -> bool:
    return threading.current_thread().name.startswith(prefix)


def _wait_any(tool: Callable[..., str], ids: str, replies: list[str]) -> None:
    replies.append(tool(ids, "wait_any", "120"))


def _launch(work_dir: Path) -> agent_dispatch.AgentJob:
    """Launch a real job as ``run_agent`` does; no daemon is there, so it ends in an error."""
    job = agent_dispatch.new_agent_job("sorcar", {
        "name": "sorcar", "prompt": "no daemon runs this", "work_dir": str(work_dir),
        "sea_path": agent_dispatch.DEFAULT_AGENT_PATH, "model_name": "", "budget": None,
        "timeout": 5.0, "alias": "",
    }, None)
    agent_dispatch.launch_agent_job(job)
    return job


def test_wait_any_returns_a_result_once_to_concurrent_waiters_and_for_repeated_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-10, S1-F3: tools run on helper threads, so two waits can overlap.

    Each waiter chose its unread jobs once, before it slept: two waiters
    of one job both returned its result, a waiter returned a result a
    single-job read had returned meanwhile, and an id named twice got
    two sections.  All three waiters are held at their first poll,
    after that choice, while the two jobs, held as their threads start,
    finish one after the other.
    """
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    tool = make_agent_job_tool()
    twice: list[str] = []
    single: list[str] = []
    starts = [Pause(agent_dispatch._finish_agent_job.__code__,
                    functools.partial(_on_thread, "agent-job-")) for _ in range(2)]
    holds = [Pause(raise_if_interrupted.__code__, functools.partial(_on_thread, "waiter"))
             for _ in range(3)]
    with Pauses(*starts, *holds):
        unread = _launch(tmp_path)
        assert starts[0].paused.wait(20)
        read = _launch(tmp_path)
        assert starts[1].paused.wait(20)
        waits = [(f"{unread.job_id} {unread.job_id}", twice)] * 2 + [(read.job_id, single)]
        waiters = [threading.Thread(target=_wait_any, args=(tool, *wait), name="waiter")
                   for wait in waits]
        for waiter, hold in zip(waiters, holds):
            waiter.start()
            assert hold.paused.wait(20)
        starts[0].release()
        unread.thread.join(20)  # finishes; nobody has read it
        starts[1].release()
        assert tool(read.job_id, "wait", "20") == read.result  # a single-job read
    for waiter in waiters:
        waiter.join(20)
    returned = "Every listed job has finished and its result was already returned: "
    assert sorted(twice) == sorted([
        f"{returned}{unread.job_id}.",
        f"Finished: {unread.job_id}; running: none.\n\n"
        f"Job {unread.job_id} ({unread.name} agent task) finished:\n{unread.result}",
    ])
    assert single == [f"{returned}{read.job_id}."]


class _ClaimReturned:
    """Profile hook parking its thread as ``wait_any`` gets its claim back, the lock released."""

    def __init__(self) -> None:
        self.parked = threading.Event()
        self.resume = threading.Event()

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "return" and frame.f_code is locked_call.__code__ and (
                frame.f_locals["fn"] is agent_dispatch._claim_finished):
            sys.setprofile(None)
            self.parked.set()
            self.resume.wait(30)

    def wait_any(self, tool: Callable[..., str], ends: list[object]) -> None:
        """Run ``wait_any`` over every job on a thread this hook profiles; record how it ends."""
        sys.setprofile(self)
        try:
            ends.append(tool("", "wait_any", "0"))
        except BaseException as exc:  # noqa: BLE001 — the test records whatever escaped
            ends.append(exc)


def test_an_interrupted_wait_any_leaves_its_results_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review r1 of B12, R1: a Stop raised after the claim, before ``wait_any`` returned it.

    Both results stayed marked as returned and the next ``wait_any``
    returned neither.  The one a ``tail`` returns in between stays that
    read's: it is not returned again.
    """
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    tool = make_agent_job_tool()
    lost, read = _launch(tmp_path), _launch(tmp_path)
    lost.thread.join(20)
    read.thread.join(20)
    hook = _ClaimReturned()
    ends: list[object] = []
    waiter = threading.Thread(target=hook.wait_any, args=(tool, ends))
    waiter.start()
    assert hook.parked.wait(20) and lost.reported and read.reported and waiter.ident is not None
    assert tool(read.job_id, "tail") == read.result
    assert inject_async_exception(waiter.ident, KeyboardInterrupt) == 1
    hook.resume.set()
    waiter.join(20)
    assert [type(end) for end in ends] == [KeyboardInterrupt], ends
    assert tool("", "wait_any", "0") == (
        f"Finished: {lost.job_id}; running: none.\n\n"
        f"Job {lost.job_id} ({lost.name} agent task) finished:\n{lost.result}"
    )


def test_wait_any_refuses_unknown_ids_bad_timeouts_and_an_empty_task() -> None:
    tool = make_agent_job_tool()
    assert tool("", "wait_any") == (
        "Error: this task has no agent jobs; run_agent(..., wait='false') starts one."
    )
    assert tool("agent-nope", "wait_any") == (
        "Error: unknown agent job(s) agent-nope; this task's jobs: none."
    )
    assert tool("", "wait_any", "soon").startswith("Error: timeout_seconds must be a number")
    assert tool("", "wait_any", True).startswith("Error: timeout_seconds must be a number")
    # A JSON number or null is no error of its own: the call gets as far as its jobs.
    assert tool("", "wait_any", 5) == tool("", "wait_any", None) == tool("", "wait_any")


def test_waits_end_early_once_every_job_finished_with_a_result_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A folder poll and a job wait return as soon as the children are done and unread, once."""
    daemon = _StaggeredFinishDaemon([0.3, None, 0.3, None, 0.3])
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    tool = make_agent_job_tool()
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.idle_children = functools.partial(unread_results_note, None)
    try:
        # No job yet: the hook is quiet and a poll runs to its deadline.
        assert unread_results_note(None) == ""
        out = tools.Bash("ls | wc -l", "count", until="^99$", poll_seconds=1, timeout_seconds=2)
        assert "did not match" in out and "the call's deadline" in out, out
        fast, slow = _start(tmp_path, "fast child"), _start(tmp_path, "slow child")
        assert _finished(fast)
        # One child still live: the finished one's unread result is not
        # yet a reason to end anything (the poll may be for the live one).
        assert unread_results_note(None) == ""
        tool(slow.job_id, "kill")
        assert slow.finished and slow.reported
        started = time.monotonic()
        out = tools.Bash("ls | wc -l", "count", until="^99$", poll_seconds=5, timeout_seconds=60)
        assert time.monotonic() - started < 10, out
        note = (
            f"every sub-agent job of this task has finished and the result of {fast.job_id} "
            "has not been read yet: agent_job('', 'wait_any') returns it"
        )
        assert "did not match after 1 run(s)" in out and f"the poll ended early: {note}" in out, out
        # Told once: a wait started without reading the result (a readiness
        # poll of the task's own) runs its course.
        assert unread_results_note(None) == "" and fast.nudged and not fast.reported
        out = tools.Bash("ls | wc -l", "count", until="^99$", poll_seconds=1, timeout_seconds=2)
        assert "the call's deadline" in out and "ended early" not in out, out
        # The next job to finish unread ends the next wait, naming every
        # unread result; a sleep that became a wait on a running job says so too.
        third = _start(tmp_path, "third child")
        assert _finished(third)
        job_id = tools.Bash("sleep 30", "hold", background=True).split()[3]
        started = time.monotonic()
        report = tools.bash_job(job_id, "wait", timeout_seconds=60)
        assert time.monotonic() - started < 10, report
        assert report.lstrip().startswith(f"Job {job_id}: running (pid"), report
        both = f"the result of {fast.job_id}, {third.job_id} has not been read yet"
        assert report.endswith("returns it]") and "[wait ended early: every sub-agent job" in report
        assert both in report, report
        assert is_routine_turn("bash_job", {"job_id": job_id, "action": "wait"}, report, False)
        # A job killed through agent_job is read (the kill returns its
        # result): it ends no wait.  A fourth finishing unread does.
        killed = _start(tmp_path, "killed child")  # past the delays: never finishes on its own
        tool(killed.job_id, "kill")
        assert killed.reported and unread_results_note(None) == ""
        fourth = _start(tmp_path, "fourth child")
        assert _finished(fourth)
        started = time.monotonic()
        out = tools.Bash("sleep 31; echo AFTER_WAIT", "nap", timeout_seconds=40)
        assert time.monotonic() - started < 10, out
        assert f"Waited on job {job_id} for up to 31 s instead of sleeping (running (pid" in out
        assert "the wait ended early: every sub-agent job" in out and "AFTER_WAIT" in out, out
        assert f"the result of {fast.job_id}, {third.job_id}, {fourth.job_id} has not" in out, out
        # Reading the results (any action) leaves nothing to tell: the
        # same poll runs to its deadline and the wait ends with the job.
        assert "child 0 done" in tool(fast.job_id, "tail")
        assert "child 2 done" in tool(third.job_id, "tail")
        assert "child 4 done" in tool(fourth.job_id, "tail")
        assert unread_results_note(None) == ""
        out = tools.Bash("ls | wc -l", "count", until="^99$", poll_seconds=1, timeout_seconds=2)
        assert "the call's deadline" in out and "ended early" not in out, out
        tools.bash_job(job_id, "kill")
        report = tools.bash_job(job_id, "wait", timeout_seconds=10)
        assert "ended early" not in report and "running (pid" not in report.split("\n", 1)[0]
    finally:
        daemon.close()


def _poll(tools: UsefulTools, outs: list[str]) -> None:
    outs.append(
        tools.Bash("ls | wc -l", "count", until="^99$", poll_seconds=60, timeout_seconds=60))


def _look(notes: list[str]) -> str:
    """Answer the idle-children hook as the agent's does, recording each answer."""
    notes.append(unread_results_note(None))
    return notes[-1]


def test_a_child_finishing_during_a_polls_pause_ends_the_poll_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-10, S13-F2: a poll looked at its children once per run, before its pause.

    A child finishing during the pause ended the poll after another run,
    or at the deadline.  The child is held as its thread starts and the
    poll as it starts to wait, after one run and one look that found the
    child live; the child then finishes before the poll goes on.
    """
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    notes: list[str] = []
    outs: list[str] = []
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.idle_children = functools.partial(_look, notes)
    start = Pause(agent_dispatch._finish_agent_job.__code__,
                  functools.partial(_on_thread, "agent-job-"))
    pause = Pause(wait_interruptible.__code__, functools.partial(_on_thread, "poller"))
    poller = threading.Thread(target=_poll, args=(tools, outs), name="poller", daemon=True)
    with Pauses(start, pause):
        late = _launch(tmp_path)
        assert start.paused.wait(20)
        started = time.monotonic()
        poller.start()
        assert pause.paused.wait(20) and notes == [""] and not late.finished
        start.release()
        late.thread.join(20)  # finishes during the pause; nobody has read it
        pause.release()
        poller.join(30)
    assert outs and time.monotonic() - started < 30, "the poll went on past the child's finish"
    note = (
        f"every sub-agent job of this task has finished and the result of {late.job_id} "
        "has not been read yet: agent_job('', 'wait_any') returns it"
    )
    assert notes == ["", note]
    assert "did not match after 1 run(s)" in outs[0], outs
    assert f"the poll ended early: {note}" in outs[0], outs


def test_a_plain_tool_set_has_no_hook_and_waits_as_before(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    marker = tmp_path / "done"
    job_id = tools.Bash(
        f"sleep 0.5; echo ready > {json.dumps(str(marker))}", "mark", background=True,
    ).split()[3]
    report = tools.bash_job(job_id, "wait", timeout_seconds=20)
    assert "exited with code 0" in report.split("\n", 1)[0] and "ended early" not in report, report
    out = tools.Bash(f"cat {json.dumps(str(marker))}", "read", until="ready", poll_seconds=1)
    assert "matched after 1 run(s)" in out, out

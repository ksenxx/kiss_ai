# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""U1 of ``reports/sea-run-agent-u2-u5-implementation-u1-design-2026-10-05.html``.

``run_agent``'s ``timeout`` bounds the CALL, not the sub-task.  Every
dispatch runs as an agent job on its own thread; a blocking call joins
it for the bound and, when the bound expires, returns the job's
still-running notice while the sub-task keeps running.  What must be
enforced rather than advised: the first ``finish`` with a live job is
rejected once, and the calling run's end kills every job it left.

Against a real private daemon (``_Daemon``) with a stand-in model:

1. ``run_agent(timeout="1")`` on a 3-s sub-task returns the notice, the
   sub-task stays live on the daemon (its tab stays open),
   ``agent_job(id, "wait")`` returns the YAML whose ``ran`` line says
   ``timeout=1s``, and the sub-task's history row is a success.
2. ``agent_job(id, "kill")`` after the notice stops the sub-task: the
   daemon no longer holds it (its tab closes) and its row is a stop.
3. A parent that calls ``finish`` with a live job gets the gate text
   once; its second ``finish`` passes, after which the job is dead and
   gone from the registry.
4. A ``kind: "channel"`` sub-task detached at its bound keeps holding
   its workspace inside the daemon; a dispatch for another workspace
   waits and fails with the existing error until the first is killed.
5. ``wait="false"`` keeps its notice; ``daemon_client.run`` is called
   with no deadline and the bound as ``record_timeout`` only.

Plus the interrupt path: a Stop of the ``run_agent`` tool call kills
the job it was waiting on.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import agent_dispatch, channel_workspace, daemon_client, persistence
from kiss.agents.sorcar.agent_dispatch import make_agent_job_tool, make_run_agent_tool
from kiss.core import tool_interrupt
from kiss.server import task_runner
from kiss.server.server import _subagent_is_done
from kiss.tests.agents.sorcar.test_dispatch_timeout import _StopConfirmingDaemon
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)
from kiss.tests.server.test_sea_simplification_proposals import _Daemon

_JOB_ID = re.compile(r"job (agent-[0-9a-f]{8})")


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    isolated = IsolatedKissHome(prefix="kiss-sea-u1-")
    isolated.write_config(is_worktree=False, auto_commit_mode=False, classify_tasks=False)
    try:
        yield isolated
    finally:
        isolated.cleanup()


@pytest.fixture(autouse=True)
def _no_leftover_jobs() -> Iterator[None]:
    yield
    agent_dispatch.kill_jobs_of(None)


def _task_row(marker: str) -> dict[str, Any]:
    """Return the newest persisted task whose text contains *marker*."""
    with persistence._rw_lock.read_lock():
        row = (
            persistence._get_db()
            .execute(
                persistence._HISTORY_SELECT + "WHERE task LIKE ? ORDER BY timestamp DESC LIMIT 1",
                (f"%{marker}%",),
            )
            .fetchone()
        )
    assert row is not None, f"no persisted task mentions {marker!r}"
    return persistence._history_row_to_dict(row)


def _is_child(text: str, marker: str) -> bool:
    """Whether the stand-in request *text* is a sub-task's, by the marker after its task header."""
    return re.search(rf"# Task[^\n]*\s+{marker}", text) is not None


def _busy_step() -> dict[str, Any]:
    """A short Bash step: the sub-task is never wedged in one model call, so a stop is prompt."""
    return tool_call_response("Bash", {"command": "sleep 0.3"})


def _run_parent(
    home: IsolatedKissHome, daemon: _Daemon, model: StandInModelServer, task: str,
) -> daemon_client.TaskResult:
    from kiss.server import sorcar

    return sorcar.run(
        task,
        work_dir=str(home.repo),
        model=STANDIN_MODEL,
        model_config=model.model_config,
        use_worktree=False,
        auto_commit=False,
        endpoint_file=daemon.endpoint_file,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# 1. timeout detaches; wait collects; nothing is lost
# ---------------------------------------------------------------------------


def test_timeout_hands_back_a_running_job_whose_result_wait_collects(
    home: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _Daemon(home)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", daemon.endpoint_file)
    parent_steps: list[str] = []
    observed: dict[str, Any] = {}

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        if _is_child(text, "CHILD-SLOW"):
            started = observed.setdefault("child_started", time.monotonic())
            if time.monotonic() - started < 3.0:
                return _busy_step()
            return finish_response("slow child done")
        parent_steps.append(text)
        step = len(parent_steps)
        if step == 1:
            return tool_call_response(
                "run_agent", {"task": "CHILD-SLOW keep busy for three seconds", "timeout": "1"},
            )
        if step == 2:
            # The notice arrived; the sub-task is still live on the daemon.
            child = _task_row("CHILD-SLOW")
            observed["child_task_id"] = child["id"]
            observed["child_live_at_notice"] = not _subagent_is_done(child["id"])
            job_id = _JOB_ID.search(text).group(1)  # type: ignore[union-attr]
            return tool_call_response(
                "agent_job", {"job_id": job_id, "action": "wait", "timeout_seconds": "60"},
            )
        return finish_response("parent done")

    model = StandInModelServer(responder)
    try:
        result = _run_parent(home, daemon, model, "PARENT run a slow child with a 1s bound")
        assert result.success is True, result
        assert "parent done" in result.text
        notice = parent_steps[1]
        assert re.search(
            r"The sorcar agent task is still running after 1s as job agent-[0-9a-f]{8}; "
            r"its tab stays open\. agent_job\(", notice,
        ), notice
        assert "did not finish" not in notice and "was stopped" not in notice
        assert "a job still running when this task ends is killed" in notice
        assert observed["child_live_at_notice"] is True
        collected = parent_steps[2]
        assert "success: true" in collected and "slow child done" in collected, collected
        assert "timeout=1s" in collected
        row = _task_row("CHILD-SLOW")
        assert row["id"] == observed["child_task_id"]
        assert _subagent_is_done(row["id"])
        assert "slow child done" in str(row["result"]), row["result"]
        assert not persistence._is_failed_result(str(row["result"]))
    finally:
        model.stop()
        daemon.stop()


# ---------------------------------------------------------------------------
# 2. kill after the notice stops the sub-task and closes its tab
# ---------------------------------------------------------------------------


def test_kill_after_the_notice_stops_the_sub_task(
    home: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _Daemon(home)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", daemon.endpoint_file)
    release = threading.Event()
    parent_steps: list[str] = []
    observed: dict[str, Any] = {}

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        if _is_child(text, "CHILD-FOREVER"):
            return finish_response("released") if release.is_set() else _busy_step()
        parent_steps.append(text)
        step = len(parent_steps)
        if step == 1:
            return tool_call_response(
                "run_agent", {"task": "CHILD-FOREVER never finish", "timeout": "1"},
            )
        if step == 2:
            child = _task_row("CHILD-FOREVER")
            observed["child_task_id"] = child["id"]
            observed["live_before_kill"] = not _subagent_is_done(child["id"])
            return tool_call_response(
                "agent_job", {"job_id": _JOB_ID.search(text).group(1), "action": "kill"},  # type: ignore[union-attr]
            )
        observed["live_after_kill"] = not _subagent_is_done(observed["child_task_id"])
        return finish_response("parent done")

    model = StandInModelServer(responder)
    try:
        result = _run_parent(home, daemon, model, "PARENT start a child and kill it")
        assert result.success is True, result
        assert "is still running after 1s as job agent-" in parent_steps[1]
        assert observed["live_before_kill"] is True
        assert "was stopped before it finished" in parent_steps[2], parent_steps[2]
        assert observed["live_after_kill"] is False
        row = _task_row("CHILD-FOREVER")
        assert persistence._is_failed_result(str(row["result"])), row["result"]
        assert "released" not in str(row["result"])
    finally:
        release.set()
        model.stop()
        daemon.stop()


# ---------------------------------------------------------------------------
# 3. the finish gate fires once; the parent's end kills what is left
# ---------------------------------------------------------------------------


def test_finish_is_rejected_once_with_live_jobs_and_the_parents_end_kills_them(
    home: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _Daemon(home)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", daemon.endpoint_file)
    release = threading.Event()
    parent_steps: list[str] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        if _is_child(text, "CHILD-FOREVER"):
            return finish_response("released") if release.is_set() else _busy_step()
        parent_steps.append(text)
        if len(parent_steps) == 1:
            return tool_call_response(
                "run_agent", {"task": "CHILD-FOREVER never finish", "wait": "false"},
            )
        return finish_response(f"finish attempt {len(parent_steps) - 1}")

    model = StandInModelServer(responder)
    try:
        result = _run_parent(home, daemon, model, "PARENT start a job and finish at once")
        assert result.success is True, result
        # The first finish (step 2) was rejected with the gate text; the
        # second (step 3) ended the task.
        assert "finish attempt 2" in result.text and "attempt 1" not in result.text, result.text
        assert len(parent_steps) == 3
        job_id = _JOB_ID.search(parent_steps[1]).group(1)  # type: ignore[union-attr]
        gate = parent_steps[2]
        assert "Error: finish rejected — run_agent jobs are still running: " in gate, gate
        assert re.search(rf"{job_id} \(sorcar, running \d+s\)", gate), gate
        assert "finishing again kills every job still running" in gate
        row = _task_row("CHILD-FOREVER")
        assert _subagent_is_done(row["id"]), "the parent's end left its job running"
        assert persistence._is_failed_result(str(row["result"])), row["result"]
        assert job_id not in agent_dispatch._AGENT_JOBS
    finally:
        release.set()
        model.stop()
        daemon.stop()


# ---------------------------------------------------------------------------
# 4. a detached channel sub-task keeps its workspace until killed
# ---------------------------------------------------------------------------


def test_detached_channel_sub_task_holds_its_workspace_until_killed(
    home: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _Daemon(home)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", daemon.endpoint_file)
    monkeypatch.setattr(task_runner, "WORKSPACE_WAIT_TIMEOUT_SECONDS", 1.0)
    release = threading.Event()
    sea = home.repo / "chan_sea.py"
    sea.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return 'a channel'

    def settings(self, settings):
        return settings | {'kind': 'channel'}
""",
        encoding="utf-8",
    )

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        if "CHAN-B" in text:
            return finish_response("b done")
        return finish_response("released") if release.is_set() else _busy_step()

    model = StandInModelServer(responder)
    run_agent = make_run_agent_tool(str(home.repo))

    def dispatch(task: str, workspace: str, timeout: str) -> str:
        return run_agent(
            task, str(sea), model=STANDIN_MODEL, timeout=timeout,
            options=json.dumps({"workspace": workspace, "model_config": model.model_config}),
        )

    try:
        notice = dispatch("CHAN-A keep the workspace busy", "a", "1")
        assert re.search(
            r"The chan agent task is still running after 1s as job agent-[0-9a-f]{8} "
            r'\(it holds channel workspace "a"\); its tab stays open\.', notice,
        ), notice
        assert dict(channel_workspace._ACTIVE_WORKSPACES) == {"a": 1}
        # Another workspace waits for the detached run, then fails.
        refused = yaml.safe_load(dispatch("CHAN-B say hi", "b", "30"))
        assert refused["success"] is False, refused
        assert "workspace 'b' could not be activated within 1s" in refused["summary"], refused
        assert dict(channel_workspace._ACTIVE_WORKSPACES) == {"a": 1}
        killed = make_agent_job_tool()(_JOB_ID.search(notice).group(1), "kill")  # type: ignore[union-attr]
        assert "was stopped before it finished" in killed, killed
        assert dict(channel_workspace._ACTIVE_WORKSPACES) == {}
        # With the workspace released the other dispatch runs.
        done = yaml.safe_load(dispatch("CHAN-B say hi", "b", "30"))
        assert done["success"] is True and "b done" in done["summary"], done
        assert done["ran"].startswith("chan (channel) ")
    finally:
        release.set()
        agent_dispatch.kill_jobs_of(None)
        model.stop()
        daemon.stop()


# ---------------------------------------------------------------------------
# 5. wait="false" is unchanged; the daemon wait has no deadline
# ---------------------------------------------------------------------------


def test_wait_false_keeps_its_notice_and_the_daemon_wait_has_no_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        # A daemon whose task ends before any ``status running=true``:
        # the stand-in leaves ``running`` to the job thread, which sets
        # it after recording the result.
        calls.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(text="done", success=True, cost=0.0, tokens=1, steps=1)

    monkeypatch.setattr(daemon_client, "run", fake_run)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(tmp_path / "no-daemon.json"))
    script = tmp_path / "helper.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'timeout': 45}
""")
    out = make_run_agent_tool(str(tmp_path))("say hi", str(script), wait="false")
    # The stand-in finishes the task without a tab, so the woken call
    # finds the result recorded and answers with it (no job is left for
    # ``agent_job``); a task still running gets the "Started ..." notice
    # instead (``test_audit1005_agent_job_finished``).
    collected = yaml.safe_load(out)
    assert collected["success"] is True and collected["summary"] == "done"
    assert "timeout=45s" in collected["ran"]
    assert agent_dispatch.notice_job_id(out) == ""
    assert agent_dispatch.agent_jobs_of(None) == {}
    (call,) = calls
    assert call["timeout"] is None
    assert call["record_timeout"] == 45.0
    assert "stop_on_timeout" not in call
    assert isinstance(call["cancel"], threading.Event)
    # A blocking call that finishes in time leaves nothing registered either.
    out = make_run_agent_tool(str(tmp_path))("say hi again", str(script))
    assert yaml.safe_load(out)["summary"] == "done"
    assert agent_dispatch.agent_jobs_of(None) == {}
    assert agent_dispatch.notice_job_id("Error: no such agent") == ""


# ---------------------------------------------------------------------------
# One deadline: the bound covers the sub-task's startup as well
# ---------------------------------------------------------------------------


def test_timeout_bounds_the_whole_call_including_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sub-task whose tab never opens still hands back at the bound, not 30 s later."""
    from kiss.agents.sorcar import cron_agent

    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    daemon = _StopConfirmingDaemon(initial_running=False)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n    def settings(self, settings):\n"
        "        return settings | {'model': 'm'}\n"
    )
    try:
        started = time.monotonic()
        out = make_run_agent_tool(str(tmp_path))("never starts", str(script), timeout="0.3")
        assert time.monotonic() - started < 3.0
        assert "is still running after 0.3s as job agent-" in out, out
        (job,) = agent_dispatch.agent_jobs_of(None).values()
        assert not job.running.is_set()
        agent_dispatch.kill_agent_job(job)
        assert daemon.wait_for_command("stop")
    finally:
        daemon.close()


def test_interrupting_wait_false_during_startup_cancels_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop while ``wait="false"`` waits for the tab cancels the sub-task too."""
    from kiss.agents.sorcar import cron_agent

    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    daemon = _StopConfirmingDaemon(initial_running=False)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n    def settings(self, settings):\n"
        "        return settings | {'model': 'm'}\n"
    )
    token = tool_interrupt.begin_tool_call("run_agent")
    caller = threading.get_ident()
    timer = threading.Timer(0.5, tool_interrupt.interrupt_tool_call, args=(caller, "run_agent"))
    timer.start()
    try:
        started = time.monotonic()
        with pytest.raises(tool_interrupt.ToolCallInterrupted):
            make_run_agent_tool(str(tmp_path))("never starts", str(script), wait="false")
        assert time.monotonic() - started < 5.0
        (job,) = agent_dispatch.agent_jobs_of(None).values()
        assert job.cancel.is_set()
        assert daemon.wait_for_command("stop"), "the interrupted call left its sub-task running"
    finally:
        timer.cancel()
        tool_interrupt.end_tool_call(token)
        daemon.close()


# ---------------------------------------------------------------------------
# A Stop landing in the parent-end join is honoured after the cleanup
# ---------------------------------------------------------------------------


def _finish_body(request: dict[str, Any]) -> dict[str, Any]:
    """Stand-in model: the task body finishes on its first step."""
    del request
    return finish_response("body done")


def test_a_stop_during_the_parent_end_join_still_ends_the_run_as_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run``'s end joins its cancelled jobs briefly; a Stop injected there is kept.

    The body returns normally while a job is live; ``kill_jobs_of`` holds
    the end of the run while the child's stop confirms (2 s late here).
    A ``KeyboardInterrupt`` injected into that join must neither skip
    the rest of the cleanup nor be swallowed into a success: the
    cleanup runs and the interrupt then leaves ``run``.
    """
    from kiss.agents.sorcar import cron_agent
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent
    from kiss.server.task_runner import inject_keyboard_interrupt

    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    daemon = _StopConfirmingDaemon(confirm_delay=2.0)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n    def settings(self, settings):\n"
        "        return settings | {'model': 'm'}\n"
    )
    agent = SorcarAgent("u1-parent")
    job = agent_dispatch.start_agent_job("helper", {
        "name": "helper", "prompt": "never finishes", "agent_path": str(script),
        "work_dir": str(tmp_path), "model_name": "", "budget": None, "timeout": 60.0,
        "parent_agent": agent,
    }, agent)
    agent_dispatch.wait_until_started(job)
    assert job.running.is_set() and job.thread.is_alive()
    model = StandInModelServer(_finish_body)
    outcome: dict[str, Any] = {}

    def run_parent() -> None:
        try:
            outcome["result"] = agent.run(
                model_name=STANDIN_MODEL, prompt_template="finish at once", web_tools=False,
                max_steps=3, max_budget=1.0, verbose=False, model_config=model.model_config,
                work_dir=str(tmp_path),
            )
        except BaseException as exc:  # noqa: BLE001 — captured for the asserts
            outcome["exc"] = exc

    worker = threading.Thread(target=run_parent, daemon=True)
    worker.start()
    try:
        # The end of the run cancels the job, then joins it; inject there.
        deadline = time.monotonic() + 20
        while not job.cancel.is_set():
            assert time.monotonic() < deadline, "the run's end never cancelled its job"
            time.sleep(0.02)
        assert agent_dispatch.agent_jobs_of(agent) == {}, "forgotten before the join"
        time.sleep(0.3)
        assert worker.ident is not None and inject_keyboard_interrupt(worker.ident) == 1
        worker.join(timeout=20)
        assert not worker.is_alive()
        assert isinstance(outcome.get("exc"), KeyboardInterrupt), outcome
        # The cleanup after the join still ran.
        assert agent.tool_call_guard is None and agent.pre_step_hook is None
        assert daemon.wait_for_command("stop")
    finally:
        model.stop()
        daemon.close()


# ---------------------------------------------------------------------------
# A Stop of the run_agent tool call kills the job it was waiting on
# ---------------------------------------------------------------------------


def test_interrupting_the_run_agent_call_kills_its_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kiss.agents.sorcar import cron_agent

    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    daemon = _StopConfirmingDaemon()
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n    def settings(self, settings):\n"
        "        return settings | {'model': 'm'}\n"
    )
    token = tool_interrupt.begin_tool_call("run_agent")
    caller = threading.get_ident()
    timer = threading.Timer(0.5, tool_interrupt.interrupt_tool_call, args=(caller, "run_agent"))
    timer.start()
    try:
        started = time.monotonic()
        with pytest.raises(tool_interrupt.ToolCallInterrupted):
            make_run_agent_tool(str(tmp_path))("never finishes", str(script), timeout="30")
        # The call unwound at once; the job's thread sends the stop.
        assert time.monotonic() - started < 5.0
        (job,) = agent_dispatch.agent_jobs_of(None).values()
        assert job.cancel.is_set()
        assert daemon.wait_for_command("stop"), "the interrupted call left its sub-task running"
        assert daemon.wait_for_command("closeTab")
        job.thread.join(10)
        assert not job.thread.is_alive()
        assert "was stopped before it finished" in job.result, job.result
        # The run's end drops what the interrupt left behind.
        assert agent_dispatch.kill_jobs_of(None) == []
        assert agent_dispatch.agent_jobs_of(None) == {}
    finally:
        timer.cancel()
        tool_interrupt.end_tool_call(token)
        daemon.close()

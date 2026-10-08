# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the sorcar agent fixes of the simplification pass.

* A1 — ``context_reset_hook`` is a bound method of the run's
  ``UsefulTools``; it is installed for the run and cleared afterwards,
  so a reused agent never keeps the previous run's tool object alive.
* A2 — a session's spend is banked BEFORE the executor is dropped from
  ``_current_executor``, so a reader summing "banked + live" never
  sees the whole session vanish.
* A3 — the run's model is resolved once: a model-picker change landing
  while the run is still setting up cannot make the run execute on a
  model other than the one its history row and settings event record.
* A4 — the trajectory summarizer's shell is killed by the stop event
  bound to the task THREAD, not only by one set on the printer.
* R3 — the auto-commit fallback message is the one
  :func:`fallback_commit_message` builds.

Every test drives a real agent against a real local stand-in model
endpoint; no code under test is mocked or patched.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.commit_message import FALLBACK_SUBJECT, fallback_commit_message
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _live_agent_usage,
    auto_commit_changes,
)
from kiss.core.base import Base
from kiss.core.processes import pid_alive
from kiss.core.stop_signal import set_thread_stop_event
from kiss.core.tool_verdict import ALLOW, Verdict
from kiss.tests.conftest import posix_only
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    run_git,
    tool_call_response,
    wait_for,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo, classifier off."""
    isolated = IsolatedKissHome("kiss-simplify-agents-")
    isolated.write_config(classify_tasks=False)
    try:
        yield isolated
    finally:
        isolated.cleanup()


@pytest.fixture
def server() -> Iterator[StandInModelServer]:
    """A stand-in model whose every answer is ``finish``."""
    stand_in = StandInModelServer(lambda _request: finish_response("done"))
    try:
        yield stand_in
    finally:
        stand_in.stop()


def _run(agent: SorcarAgent, env: IsolatedKissHome, server: StandInModelServer,
         **kwargs: Any) -> str:
    """Run *agent* on the stand-in model with the browser off."""
    return agent.run(
        prompt_template="say done",
        model_name=STANDIN_MODEL,
        model_config=server.model_config,
        work_dir=str(env.repo),
        max_steps=5,
        max_sub_sessions=1,
        web_tools=False,
        verbose=False,
        **kwargs,
    )


# A1 ---------------------------------------------------------------------


def test_context_reset_hook_is_installed_for_the_run_and_cleared_after(
    env: IsolatedKissHome, server: StandInModelServer,
) -> None:
    agent = SorcarAgent("a1")
    assert agent.context_reset_hook is None
    hook_seen_during_run: list[bool] = []

    def observe(name: str, _args: dict[str, Any]) -> Verdict:
        hook_seen_during_run.append(
            name == "finish" and agent.context_reset_hook is not None,
        )
        return ALLOW

    _run(agent, env, server, tool_call_hook=observe)
    assert hook_seen_during_run == [True]
    assert agent.context_reset_hook is None, (
        "the run's UsefulTools.forget_reads survived the run"
    )


# A2 ---------------------------------------------------------------------


class _BankObservingAgent(SorcarAgent):
    """Records what a concurrent reader would see when a session is banked."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.banked: list[tuple[bool, tuple[float, int, int]]] = []

    def _accumulate_usage(self, agent: Base) -> None:
        if agent is self._current_executor:
            self.banked.append((True, _live_agent_usage(self)))
        elif not self.banked:
            self.banked.append((False, _live_agent_usage(self)))
        super()._accumulate_usage(agent)


def test_session_is_still_live_while_its_spend_is_banked(
    env: IsolatedKissHome, server: StandInModelServer,
) -> None:
    agent = _BankObservingAgent("a2")
    _run(agent, env, server)
    assert agent.banked, "no session was banked"
    executor_live, (budget, tokens, steps) = agent.banked[0]
    assert executor_live, "the executor was dropped before its spend was banked"
    # The stand-in reports 15 tokens per call; a reader at bank time
    # must see the session's spend through the live executor.
    assert tokens >= 15 and steps >= 1
    assert agent.total_tokens_used >= 15
    assert agent._current_executor is None


# A3 ---------------------------------------------------------------------


def _row_model(task_id: str) -> str:
    db = persistence._get_db()
    with persistence._rw_lock.read_lock():
        row = db.execute(
            "SELECT model FROM task_history WHERE id = ?", (task_id,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_a_picker_change_during_setup_cannot_change_the_runs_model(
    env: IsolatedKissHome, server: StandInModelServer,
) -> None:
    """``model_name=None`` is resolved once, before the history row exists.

    The picker's ``last_model`` is changed from the ``_on_task_id_allocated``
    callback — a real server hook that runs after the early resolution
    and before ``_reset`` — so a second resolution would pick the new
    (unreachable) model and the run would fail, and the row would
    disagree with the run.
    """
    env.write_config(last_model=STANDIN_MODEL)
    agent = ChatSorcarAgent("a3")
    printer = CapturePrinter()
    allocated: list[str] = []

    def on_allocated(task_id: str, _chat_id: str) -> None:
        allocated.append(task_id)
        env.write_config(last_model="picked-later-model")

    result = agent.run(
        prompt_template="say done",
        model_name=None,
        model_config=server.model_config,
        work_dir=str(env.repo),
        printer=printer,
        max_steps=5,
        max_sub_sessions=1,
        web_tools=False,
        _on_task_id_allocated=on_allocated,
    )
    assert "success: true" in result
    assert agent._launch_model_name == STANDIN_MODEL
    assert agent.model_name == STANDIN_MODEL
    assert agent.task_settings["model"] == STANDIN_MODEL
    (task_id,) = allocated
    assert _row_model(task_id) == STANDIN_MODEL


# A4 ---------------------------------------------------------------------


def _pid_recorded(pid_file: Path) -> bool:
    return pid_file.exists() and bool(pid_file.read_text().strip())


@posix_only("bash's $$ is an MSYS pid, not a Windows pid")
def test_summarizer_bash_is_killed_by_a_stop_event_bound_only_to_the_thread(
    env: IsolatedKissHome,
) -> None:
    """The thread binding alone must reach the summarizer's ``UsefulTools``.

    The executor is starved of steps so it raises the recoverable
    "exceeded N steps" error that spawns the summarizer; the summarizer's
    only action is a long shell command.  The stop event is published
    through :func:`set_thread_stop_event` only, and the run has no
    printer at all (``verbose=False``): ``JsonPrinter``'s thread-local
    is a view over the same per-thread storage, so only a printer-less
    (or non-JSON-printer) run exposes a summarizer that reads the stop
    event off the printer instead of the thread.
    """
    pid_file = env.repo / "summarizer.pid"

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        if "The executor's trajectory is saved at" in request_text(request):
            return tool_call_response(
                "Bash",
                {
                    "command": f"echo $$ > {pid_file}; sleep 120",
                    "description": "long summarizer analysis",
                },
            )
        return tool_call_response(
            "Bash", {"command": "echo working", "description": "executor step"},
        )

    stand_in = StandInModelServer(responder)
    agent = SorcarAgent("a4-summarizer")
    stop_event = threading.Event()
    outcome: dict[str, Any] = {}

    def run_agent() -> None:
        set_thread_stop_event(stop_event)
        try:
            outcome["result"] = agent.run(
                prompt_template="summarizer stop path",
                model_name=STANDIN_MODEL,
                model_config=stand_in.model_config,
                work_dir=str(env.repo),
                max_steps=3,
                max_sub_sessions=1,
                web_tools=False,
                verbose=False,
            )
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            outcome["error"] = exc
        finally:
            set_thread_stop_event(None)

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: _pid_recorded(pid_file)), (
            "the summarizer never started its shell command"
        )
        pid = int(pid_file.read_text().strip())
        assert pid_alive(pid), "the shell exited before the stop"
        assert agent._stop_event is stop_event
        stop_event.set()
        assert wait_for(lambda: not pid_alive(pid), timeout=10.0), (
            "the summarizer's shell survived a stop bound only to the thread"
        )
    finally:
        stop_event.set()
        thread.join(timeout=30)
        stand_in.stop()
    assert not thread.is_alive()


# R3 ---------------------------------------------------------------------


def test_auto_commit_uses_the_shared_fallback_when_the_message_fn_fails(
    env: IsolatedKissHome,
) -> None:
    (env.repo / "new.txt").write_text("hello\n", encoding="utf-8")

    def broken_message_fn(_dir: Path, _prompt: str | None, _result: str | None) -> str:
        raise RuntimeError("diff unreadable")

    assert auto_commit_changes(
        env.repo, "add a greeting", broken_message_fn, task_result="<p>Added it.</p>",
    )
    message = run_git(env.repo, "log", "-1", "--format=%B").stdout.strip()
    assert message == fallback_commit_message("add a greeting", "<p>Added it.</p>")
    assert message.startswith(FALLBACK_SUBJECT)
    assert "User prompt:\nadd a greeting" in message
    assert "Result:\nAdded it." in message
    assert not GitWorktreeOps.has_staged_changes(env.repo)


def test_fallback_commit_message_without_trailers() -> None:
    assert fallback_commit_message() == FALLBACK_SUBJECT
    assert fallback_commit_message("", "") == FALLBACK_SUBJECT

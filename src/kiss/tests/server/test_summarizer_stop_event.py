# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop during the hand-off summarizer ends the run at once.

When an executor session dies with a recoverable error,
:meth:`RelentlessAgent.perform_task` falls back to a summarizer agent
that turns a scripted digest of the session into the continuation
summary.  The summarizer has no shell (its only tool is ``finish``),
so nothing but its model call can outlive a Stop: the stop event bound
to the task thread (the printer's thread-local, where the web server
publishes it) must end the run as soon as that call returns, without
a second summarizer step.

This test drives the real recovery path: a real agent, a real
executor session that really exceeds its step budget, and a real
summarizer whose model call is held open by the stand-in server until
the test has pressed Stop.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    request_text,
    tool_call_response,
    wait_for,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f3-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _text_response(text: str) -> dict[str, Any]:
    """Build a chat-completions body holding plain text and no tool call."""
    body = tool_call_response("finish", {})
    body["choices"][0]["message"] = {"role": "assistant", "content": text}
    body["choices"][0]["finish_reason"] = "stop"
    return body


def test_stop_during_the_summarizer_call_ends_the_run(env: IsolatedKissHome) -> None:
    """Stop set while the summarizer's call is pending ends the run after that call.

    The executor session is starved of steps so it raises the
    recoverable "exceeded N steps" error, which is exactly the branch
    that spawns the summarizer.  The summarizer's one model call is
    held open until the test sets the stop event; its reply is plain
    text (no ``finish``), which without the stop would earn it a
    second step.
    """
    summarizer_requests: list[dict[str, Any]] = []
    summarizer_started = threading.Event()
    release = threading.Event()

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        if "# Summarizer" in request_text(request):
            summarizer_requests.append(request)
            summarizer_started.set()
            release.wait(timeout=30)
            return _text_response("still thinking about the digest")
        return tool_call_response(
            "Bash", {"command": "echo working", "description": "executor step"},
        )

    server = StandInModelServer(responder)
    printer = CapturePrinter()
    agent = SorcarAgent("f3-summarizer")
    stop_event = threading.Event()
    outcome: dict[str, Any] = {}

    def run_agent() -> None:
        printer._thread_local.stop_event = stop_event
        try:
            outcome["result"] = agent.run(
                prompt_template="F3 recovery path",
                model_name=STANDIN_MODEL,
                model_config=server.model_config,
                work_dir=str(env.repo),
                printer=printer,
                max_steps=3,
                max_sub_sessions=1,
                web_tools=False,
            )
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            outcome["error"] = exc

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()
    try:
        assert wait_for(summarizer_started.is_set), "the summarizer never called the model"
        stop_event.set()
        release.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the run survived a Stop set during the summarizer"
    finally:
        stop_event.set()
        release.set()
        thread.join(timeout=30)
        server.stop()

    assert len(summarizer_requests) == 1, "the summarizer took a step after the Stop"
    tools = [t["function"]["name"] for t in summarizer_requests[0]["tools"]]
    assert tools == ["finish"], tools
    assert isinstance(outcome.get("error"), KeyboardInterrupt), outcome

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Concurrency sweep of 2026-10-26, ``core-agent`` slice: three ``KISSAgent`` fixes.

1. ``_execute_tool`` broadcast the ``tool_call`` event (carrying the
   call's ``call_id``) BEFORE ``register_tool_call``; a per-panel Stop
   that arrived in that window found no running call and was rejected
   (``tool_interrupt_ack accepted=False``) while the tool kept running.
   The test's printer presses Stop from inside the ``tool_call`` event
   itself (the earliest moment a UI can know the id) and requires the
   interrupt to be accepted and the tool to be interrupted.

2. ``_billed_model_call``'s model-error path billed the partial usage and
   emitted the ``llm_call`` event unguarded, so an accounting exception
   (``calculate_cost`` on a model with no catalog price) replaced the
   provider's own error.  A local Chat Completions server returns a
   truncated completion (``finish_reason="length"``, billed) for a model
   name the catalog does not know; the run must end with the truncation
   error, not with the pricing one.  The retryable-error variant of the
   finding (a provider error that the loop would have retried) cannot be
   reproduced end to end: every Chat Completions path that leaves billed
   partial usage behind raises a terminal ``KISSError``, and an HTTP
   error leaves none.

3. ``_turn_effort`` edited the caller's ``model_config`` dict in place
   (``Model.__init__`` stores it without copying), so two agents built
   from one shared dict saw each other's routine-effort override.
   ``_reset`` now copies the dict once; the test enters a routine turn
   and requires the override to stay out of the caller's dict.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.printer import Printer
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    interrupt_tool_call,
    raise_if_interrupted,
)
from kiss.tests.core.test_budget_enforcement_e2e import _read_body, _start_server


class _StopFromToolCallEventPrinter(Printer):
    """Presses the panel's Stop from inside the ``tool_call`` event it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.accepted: bool | None = None
        self.token_callback = None  # type: ignore[method-assign,assignment]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:  # noqa: A002
        self.events.append({"type": type, "content": content, **kwargs})
        if type == "tool_call":
            self.accepted = interrupt_tool_call(threading.get_ident(), content, kwargs["call_id"])

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        pass

    def reset(self) -> None:
        pass


def cooperative_tool() -> str:
    """Tool that checks for a pending interrupt at its first safe point."""
    raise_if_interrupted()
    return "ran to completion"


#: A catalog entry with a ``thinking`` level, the only kind ``_turn_effort`` overrides.
_THINKING_MODEL = "claude-sonnet-4-6"


def finish(result: str) -> str:
    """The task's result tool, as the agent registers it."""
    return result


def _agent(
    printer: Printer | None, *tools: Any, model_config: dict[str, Any] | None = None
) -> KISSAgent:
    agent = KISSAgent("sweep1026-core-agent")
    agent._reset(
        model_name=_THINKING_MODEL,
        is_agentic=True,
        max_steps=5,
        max_budget=0.01,
        model_config=model_config,
        printer=printer,
        verbose=False,
    )
    agent._add_functions(list(tools))
    return agent


def test_stop_pressed_on_the_tool_call_event_interrupts_the_call() -> None:
    printer = _StopFromToolCallEventPrinter()
    agent = _agent(printer, cooperative_tool)
    name, result = agent._execute_tool({"name": "cooperative_tool", "arguments": {}})
    assert printer.accepted is True, "the Stop found no registered call"
    assert (name, result) == ("cooperative_tool", USER_INTERRUPTED_MESSAGE)
    types = [e["type"] for e in printer.events]
    assert types == ["tool_call", "tool_result"]
    assert printer.events[1]["interrupted"] is True
    assert printer.events[1]["is_error"] is False


def test_finish_is_never_interruptible_and_a_blocked_call_is_announced_then_refused() -> None:
    printer = _StopFromToolCallEventPrinter()
    agent = _agent(printer, cooperative_tool, finish)
    name, result = agent._execute_tool({"name": "finish", "arguments": {"result": "<p>ok</p>"}})
    assert printer.accepted is False, "finish must not be registered as interruptible"
    assert (name, result) == ("finish", "<p>ok</p>")
    printer.events.clear()
    name, result = agent._execute_tool(
        {"name": "cooperative_tool", "arguments": {}}, blocked="not now"
    )
    assert printer.accepted is False, "a refused call never runs, so it is not registered"
    assert [e["type"] for e in printer.events] == ["tool_call", "tool_result"]
    assert printer.events[1]["is_error"] is True and "interrupted" not in printer.events[1]
    assert "not now" in result and "did NOT run" in result


class _TruncatedHandler(BaseHTTPRequestHandler):
    """Always answers with a billed completion cut off at the token limit."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        body = json.dumps(
            {
                "id": "chatcmpl-sweep1026",
                "object": "chat.completion",
                "model": "sweep1026-unpriced",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "The answer is"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def test_model_error_survives_a_failing_bill_of_its_partial_usage() -> None:
    srv, url = _start_server(_TruncatedHandler)
    try:
        agent = KISSAgent("sweep1026-unpriced")
        with pytest.raises(KISSError) as info:
            agent.run(
                model_name="sweep1026-unpriced",
                prompt_template="Say something.",
                tools=[],
                max_steps=2,
                max_budget=1.0,
                verbose=False,
                model_config={"base_url": url, "api_key": "test-key", "stream": False},
            )
    finally:
        srv.shutdown()
    message = str(info.value)
    assert "truncated" in message, message
    assert "Cannot calculate budget" not in message, message
    # The failed bill left the run's usage untouched, exactly as a Stop would.
    assert agent.budget_used == 0.0 and agent.total_tokens_used == 0


def test_routine_effort_override_never_reaches_the_callers_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "routine_turn_effort", "low")
    shared: dict[str, Any] = {"max_tokens": 4096}
    agent = _agent(None, model_config=shared)
    built = agent.model
    agent._routine_next_turn = True
    with agent._turn_effort():
        assert agent.model.model_config.get("reasoning_effort") == "low"
        assert "reasoning_effort" not in shared, "the override leaked into the caller's dict"
    assert "reasoning_effort" not in agent.model.model_config
    shared["temperature"] = 0.5
    assert "temperature" not in agent.model.model_config, "the agent shares the caller's dict"
    # Re-running with an equal config still reuses the built model (compared by value).
    agent._reset(
        model_name=_THINKING_MODEL,
        is_agentic=True,
        max_steps=5,
        max_budget=0.01,
        model_config={"max_tokens": 4096},
        verbose=False,
    )
    assert agent.model is built

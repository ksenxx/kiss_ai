# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C3/F2): a Stop during tool-result submission rolls the state back.

``OpenAICompatibleModel2.add_function_results_to_conversation_and_return``
snapshots the conversation and the pending-call queue so that a rejected
result leaves both exactly as they were, but the rollback caught only
``Exception``.  Sorcar's Stop reaches the task thread as a
``KeyboardInterrupt`` (a ``BaseException``), and the inner loop pops the
pending call before it appends the call's output: a Stop landing between
the two left the queue short of a call the conversation still had no
output for, so resubmitting that result after the Stop was rejected
(``No pending function_call named ...``) instead of completing the turn.

The tool turn comes from a real local ``/v1/responses`` endpoint through
the real SDK; nothing is seeded.  The Stop is raised into the thread at
the line after the pop, the way the runner's asynchronous
``KeyboardInterrupt`` lands at an arbitrary line, with a trace function
that fires once; no production code is patched.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_MODEL = "gpt-audit1101-stop"
_RESULTS = [("first", {"result": "first result"}), ("second", {"result": "second result"})]


def first() -> str:
    """Test-only tool stub."""
    return "first result"


def second() -> str:
    """Test-only tool stub."""
    return "second result"


def _function_call(name: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{name}",
        "call_id": f"call_{name}",
        "name": name,
        "arguments": "{}",
    }


def _respond(request: Request) -> Reply:
    """Two function calls on the first turn, a plain answer afterwards."""
    answered = any(
        item.get("type") == "function_call_output" for item in request.body.get("input", [])
    )
    output: list[dict[str, Any]] = (
        [
            {
                "type": "message",
                "id": "msg_done",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ]
        if answered
        else [_function_call("first"), _function_call("second")]
    )
    return Reply(
        json_body={
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": _MODEL,
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    )


def _stop_after_pending_pop(model: OpenAICompatibleModel2) -> None:
    """Submit both results with a Stop landing right after the first call is popped."""
    inner = OpenAICompatibleModel2._add_function_results_inner
    lines, start = inspect.getsourcelines(inner)
    stop_line = start + next(
        i for i, line in enumerate(lines) if 'result_dict.get("tool_use_id")' in line
    )
    fired = False

    def trace(frame: Any, event: str, arg: Any) -> Any:
        nonlocal fired
        if (
            not fired
            and event == "line"
            and frame.f_code is inner.__code__
            and frame.f_lineno == stop_line
        ):
            fired = True
            raise KeyboardInterrupt("Stop pressed while the tool results were being added")
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            model.add_function_results_to_conversation_and_return(_RESULTS)
    finally:
        sys.settrace(previous)
    assert fired


def test_stop_between_pop_and_append_leaves_the_turn_resubmittable() -> None:
    """After the Stop both results are accepted again and the next turn carries both."""
    with ScriptedOpenAIServer(_respond) as server:
        model = OpenAICompatibleModel2(_MODEL, base_url=server.base_url, api_key="k")
        model.initialize("Run first and second.")
        calls, _, _ = model.generate_and_process_with_tools({"first": first, "second": second})
        assert [call["name"] for call in calls] == ["first", "second"]
        conversation_before = list(model.conversation)

        _stop_after_pending_pop(model)

        # The advertised atomic rollback: nothing of the half-done
        # submission survives the Stop.
        assert model.conversation == conversation_before
        # The resumed run submits the same results again; the queue must
        # still know both calls.
        model.add_function_results_to_conversation_and_return(_RESULTS)
        text, _ = model.generate()
        assert text == "done"
        outputs = [
            (item["call_id"], item["output"])
            for item in server.requests[-1].body["input"]
            if item.get("type") == "function_call_output"
        ]
        assert outputs == [("call_first", "first result"), ("call_second", "second result")]

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: mid-step prompt budget enforcement in ``KISSAgent``.

Covers the production bug where agents kept working after exceeding
``max_budget`` (the settings-panel budget): the budget was only checked at
the top of each step, so an over-budget response still executed a full
round of tools, and spend attributed to the parent task by parallel
sub-agents was never enforced mid-session at all.

The sorcar-side half of the fix (RelentlessAgent stopping promptly after
attributed spend, and fair sub-agent budget distribution) is covered by
``kiss.tests.agents.sorcar.test_budget_enforcement_e2e``, which reuses
this file's fake OpenAI-compatible HTTP harness.

All tests drive real agents over real HTTP against a local
OpenAI-chat-completions-compatible server.  No mocks, patches, fakes, or
test doubles.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError


def _tool_calls_response(
    calls: list[tuple[str, str]],
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Build a chat-completions response containing one or more tool calls."""
    return {
        "id": "chatcmpl-budget-e2e",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for index, (name, arguments) in enumerate(calls, start=1)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }



def _tool_call_response(
    name: str,
    arguments: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Build a chat-completions response containing one tool call."""
    return _tool_calls_response(
        [(name, arguments)], prompt_tokens, completion_tokens
    )



def _send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    """Write *payload* as a JSON HTTP 200 response."""
    body = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)



def _read_body(handler: BaseHTTPRequestHandler) -> str:
    """Read and return the request body as text."""
    cl = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(cl).decode() if cl else ""



def _start_server(
    handler_cls: type[BaseHTTPRequestHandler],
) -> tuple[ThreadingHTTPServer, str]:
    """Start a local HTTP server; return (server, base_url)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"



_EXPENSIVE = (500_000, 500_000)

_CHEAP = (10, 5)



class _ExpensiveNoopHandler(BaseHTTPRequestHandler):
    """Always returns a non-finish ``noop`` tool call costing $0.375."""

    requests = 0

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        type(self).requests += 1
        _send_json(self, _tool_call_response("noop", "{}", *_EXPENSIVE))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass



class _ExpensiveFinishHandler(BaseHTTPRequestHandler):
    """Always returns a ``finish`` tool call costing $0.375."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_call_response("finish", '{"result": "done"}', *_EXPENSIVE),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass



class _ExpensiveMixedFinishHandler(BaseHTTPRequestHandler):
    """Returns an expensive non-finish tool followed by ``finish``."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_calls_response(
                [("noop", "{}"), ("finish", '{"result": "done"}')],
                *_EXPENSIVE,
            ),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass



class _CheapTwoToolsHandler(BaseHTTPRequestHandler):
    """Returns ``sub_spend`` followed by a tool that must be blocked."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_calls_response(
                [("sub_spend", "{}"), ("must_not_run", "{}")], *_CHEAP
            ),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass



class TestMidStepBudgetEnforcement:
    """The agent must not execute further tools once over budget."""

    def test_exhausted_budget_blocks_model_request(self) -> None:
        """A budget is a hard cap: once exactly exhausted, the agent must
        not start another paid model request.  In particular, a zero-dollar
        settings-panel budget must make zero provider requests."""
        _ExpensiveNoopHandler.requests = 0
        srv, url = _start_server(_ExpensiveNoopHandler)

        def noop() -> str:
            """A tool that must never run."""
            return "ok"

        try:
            agent = KISSAgent("zero-budget")
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call noop.",
                    tools=[noop],
                    max_steps=10,
                    max_budget=0.0,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "test-key"},
                )
            assert _ExpensiveNoopHandler.requests == 0
            assert agent.budget_used == 0.0
        finally:
            srv.shutdown()

    def test_tools_not_executed_once_over_budget(self) -> None:
        """The very response that exceeds the budget must abort the step
        BEFORE its (non-finish) tool calls are executed — previously the
        agent executed one more full round of tools and only stopped at
        the top of the next step."""
        srv, url = _start_server(_ExpensiveNoopHandler)
        calls: list[str] = []

        def noop() -> str:
            """A no-op tool that records that it was called."""
            calls.append("noop")
            return "ok"

        try:
            agent = KISSAgent("mid-step-budget")
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call noop.",
                    tools=[noop],
                    is_agentic=True,
                    max_steps=10,
                    max_budget=0.01,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "test-key"},
                )
            assert agent.budget_used > 0.01
            assert calls == [], (
                f"Tools were executed {len(calls)} time(s) AFTER the budget "
                f"was already exceeded — the agent must stop immediately."
            )
        finally:
            srv.shutdown()

    def test_finish_result_returned_even_when_over_budget(self) -> None:
        """When the over-budget response contains ONLY a ``finish`` call the
        agent IS stopping — the result must be returned, not discarded."""
        srv, url = _start_server(_ExpensiveFinishHandler)
        try:
            agent = KISSAgent("finish-over-budget")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Finish immediately.",
                is_agentic=True,
                max_steps=10,
                max_budget=0.01,
                verbose=False,
                model_config={"base_url": url, "api_key": "test-key"},
            )
            assert result == "done"
            assert agent.budget_used > 0.01
        finally:
            srv.shutdown()

    def test_finish_does_not_bypass_nonfinish_tool_budget_check(self) -> None:
        """A response containing both a normal tool and ``finish`` must
        not use the finish call as a loophole to execute the normal tool
        after the model response has already exceeded the budget."""
        srv, url = _start_server(_ExpensiveMixedFinishHandler)
        calls: list[str] = []

        def noop() -> str:
            """Record an invocation that must never happen."""
            calls.append("noop")
            return "ok"

        try:
            agent = KISSAgent("mixed-finish-over-budget")
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call noop and then finish.",
                    tools=[noop],
                    max_steps=10,
                    max_budget=0.01,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "test-key"},
                )
            assert calls == []
        finally:
            srv.shutdown()

    def test_attributed_spend_blocks_later_tool_in_same_response(self) -> None:
        """After one tool attributes sub-agent spend over the parent
        limit, no later tool from that same model response may execute."""
        srv, url = _start_server(_CheapTwoToolsHandler)
        parent_spend = 0.0
        forbidden_calls: list[str] = []

        def sub_spend() -> str:
            """Simulate spend attributed by ``run_parallel``."""
            nonlocal parent_spend
            parent_spend = 5.0
            return "spent"

        def must_not_run() -> str:
            """Record a post-budget tool invocation that is forbidden."""
            forbidden_calls.append("ran")
            return "bad"

        agent = KISSAgent("same-response-parent-budget")

        def check_parent_budget() -> None:
            if parent_spend > 1.0:
                raise KISSError("Parent budget exceeded.")

        agent.budget_check_hook = check_parent_budget
        try:
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Spend, then run the second tool.",
                    tools=[sub_spend, must_not_run],
                    max_steps=10,
                    max_budget=10.0,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "test-key"},
                )
            assert forbidden_calls == []
        finally:
            srv.shutdown()
